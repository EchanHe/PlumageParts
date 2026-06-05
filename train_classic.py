# SPDX-License-Identifier: MIT

import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import numpy as np
from datetime import datetime
import json
import logging
import gc
from collections import defaultdict

from sklearn.model_selection import train_test_split

from dataset.dataset import SegmentationDataset, cls_weights, make_pad_if_needed
from models.unet import UNet
from models.deeplab import DeepLabWrapper
from models.unet_resnet50 import UNetResNet50

from models.util import (
    apply_colormap,
    compute_dice,
    overlay_mask_on_image,
    DiceCELoss,
    build_augmentation_from_config,
    accumulate_per_class_stats,
    compute_dice_from_stats,
    compute_custom_class_weight,
    compute_metrics_from_stats,
    denormalize_tensor
)
import numpy as np
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


from datetime import datetime
import json

from collections import defaultdict

from torch.amp import autocast, GradScaler


# optional, for LR warmup
try:
    from models.warmup import LRWarmupScheduler
except Exception:
    LRWarmupScheduler = None

def evaluate_model_on_loader(model, loader, criterion, args, writer, global_epoch_label, denorm_stats, device):
    model.eval()
    eval_loss = 0.0
    eval_stats = defaultdict(lambda: {'intersection': 0, 'pred': 0, 'target': 0})
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            eval_loss += loss.item()
            preds = outputs.argmax(dim=1)
            accumulate_per_class_stats(preds, masks, args.num_classes, eval_stats)
    eval_loss /= len(loader)
    dice_per_class, _, iou_scores, mean_iou = compute_metrics_from_stats(eval_stats)
    dice_mean = np.mean(dice_per_class)
    iou_no_bg = np.mean(iou_scores[1:])
    dice_no_bg = np.mean(dice_per_class[1:])

    # write to tensorboard
    if writer is not None:
        writer.add_scalar(f"Loss/{global_epoch_label}", eval_loss, args.epochs)
        writer.add_scalar(f"IOU/{global_epoch_label}_mean", mean_iou, args.epochs)
        writer.add_scalar(f"Dice/{global_epoch_label}_mean", dice_mean, args.epochs)
        writer.add_scalar(f"IOU/{global_epoch_label}_mean_no_bg", iou_no_bg, args.epochs)
        writer.add_scalar(f"Dice/{global_epoch_label}_mean_no_bg", dice_no_bg, args.epochs)
    logging.info(f"{global_epoch_label}: Loss: {eval_loss:.4f} | meanIoU: {mean_iou:.4f}| Dice (no bg): {dice_no_bg:.4f} | mIoU(no bg): {iou_no_bg:.4f}")

    return eval_loss, dice_no_bg, iou_no_bg


def main_general(args):
    """
    Main training function for segmentation models.
    
    Supports UNet and DeepLab architectures with various optimizations:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Custom augmentations
    - Separate tracking for loss and dice optimization
    """
    device = args.device

    log_dir = os.path.join(args.log_dir, f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logging.info("Arguments:")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)

    # Load augmentation configuration with exception handling
    aug_config = None
    if args.aug_config:
        try:
            with open(args.aug_config, 'r') as f:
                aug_config = json.load(f)
                train_transform = build_augmentation_from_config(
                    aug_list=aug_config['train'],
                    resize_height=args.resize,
                    resize_width=args.resize
                )

                val_transform = build_augmentation_from_config(
                    aug_list=aug_config['val'],
                    resize_height=args.resize,
                    resize_width=args.resize
                )
        except Exception as e:
            print(f"Error loading augmentation config from {args.aug_config}: {e}")
            print("Falling back to default augmentations")
            args.aug_config = None  # Fall back to default

    if not args.aug_config:
        train_transform = A.Compose([
            A.LongestMaxSize(max_size=args.resize, interpolation=1),
            make_pad_if_needed(min_height=args.resize, min_width=args.resize, border_mode=0, image_fill=0, mask_fill=0),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
        val_transform = A.Compose([
            A.LongestMaxSize(max_size=args.resize, interpolation=1),
            make_pad_if_needed(min_height=args.resize, min_width=args.resize, border_mode=0, image_fill=0, mask_fill=0),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    # Dataset
    all_filenames = sorted(os.listdir(args.img_dir))
    all_filenames = [f for f in all_filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    # with explicit validation directories
    if args.val_img_dir and args.val_mask_dir:
        # Use provided validation directories
        # train_fns = [f for f in all_filenames if f not in os.listdir(args.val_img_dir)]
        # only files with image extensions
        train_fns = all_filenames
        val_fns = [f for f in os.listdir(args.val_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    # without explicit validation directories, split the dataset
    else:
        train_fns, val_fns = train_test_split(all_filenames, test_size=args.val_split, random_state=42)

    train_dataset = SegmentationDataset(img_dir=args.img_dir, mask_dir=args.mask_dir, file_list=train_fns, 
                                        album_aug=train_transform,
                                        mask_suffix=args.mask_suffix)
    
    if args.val_img_dir and args.val_mask_dir:
        val_dataset = SegmentationDataset(args.val_img_dir, args.val_mask_dir, file_list= val_fns,
                                        album_aug=val_transform, mask_suffix=args.mask_suffix)
    else:
        val_dataset = SegmentationDataset(args.img_dir, args.mask_dir, file_list = val_fns,
                                        album_aug=val_transform, mask_suffix=args.mask_suffix)

    logging.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory,
                              drop_last=True,
                              persistent_workers=args.persistent_workers if args.num_workers > 0 else False)

    val_loader = DataLoader(val_dataset, 
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=args.pin_memory,
                             persistent_workers=args.persistent_workers if args.num_workers > 0 else False)

    logging.info(f"Number of batches - Train: {len(train_loader)}, Val: {len(val_loader)}")


    # Model
    if args.model == "unet":
        model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    elif args.model == "deeplab":
        model = DeepLabWrapper(num_classes=args.num_classes, pretrained_backbone=False).to(device)
    elif args.model == "deeplab_resnet50":
        model = DeepLabWrapper(num_classes=args.num_classes, pretrained_backbone=True).to(device)
    elif args.model == "unet_resnet50":
        model = UNetResNet50(num_classes=args.num_classes, in_channels=3,
                            pretrained=True, freeze_encoder=False).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Loss function configuration with optional class weighting
    if args.weighted_loss:
        # class_weights = compute_class_weight(
        #     class_weight='balanced',
        #     classes=np.arange(args.num_classes),
        #     y=cls_weights(train_dataset)
        # )

        # Calculate custom class weights to handle class imbalance
        class_weights = compute_custom_class_weight(y_true=cls_weights(train_dataset),
                                                    all_possible_classes=np.arange(args.num_classes),
                                                    handle_missing=True, missing_weight=1e-6)

        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        logging.info(f"Class weights: {class_weights}")

        if args.loss_fn == "DiceCE":
            criterion = DiceCELoss(weight=class_weights, dice_weight=1.0, ce_weight=1.0)
        elif args.loss_fn == "cross_entropy":
            criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        if args.loss_fn == "DiceCE":
            criterion = DiceCELoss(dice_weight=1.0, ce_weight=1.0)
        elif args.loss_fn == "cross_entropy":
            criterion = nn.CrossEntropyLoss()

    # Optimizer configuration
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Scheduler

    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    elif args.scheduler == "OneCycleLR":
        steps_per_epoch_after_accumulation = max(1, len(train_loader) // args.grad_accum)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                        steps_per_epoch=steps_per_epoch_after_accumulation,
                                                        epochs=args.epochs)
    else:
        scheduler = None

    # LR Warmup
    warmup_scheduler = None
    global_step = 0
    if args.lr_warmup and LRWarmupScheduler is not None:
        warmup_scheduler = LRWarmupScheduler(optimizer=optimizer, warmup_steps=args.lr_warmup_steps, base_lr=args.lr)
        logging.info(f"Using LR warmup for {args.lr_warmup_steps} steps.")

    # tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Save arguments configuration
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
            

    # Save augmentation config if used
    if aug_config is not None:
        try:
            with open(os.path.join(log_dir, "aug_used.json"), "w") as f:
                json.dump(aug_config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save aug_used.json: {e}")



    # Training and validation loop
    best_val_loss = float('inf')
    best_val_dice_no_bg = 0.
    best_loss_epoch = 0
    best_dice_epoch = 0
    best_iou = 0.
    best_iou_epoch = 0

    amp_enabled = bool(args.amp and str(device).startswith("cuda"))
    scaler = GradScaler(device="cuda", enabled=amp_enabled)
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        # For gradient accumulation
        accum = 0
        num_batches = len(train_loader)

        for step, (images, masks, _) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device)
            optimizer.zero_grad()
            # Forward pass with mixed precision support
            with autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, masks)
                # Scale loss for gradient accumulation (divide by accumulation steps)
                loss = loss / args.grad_accum

            # Backward pass with mixed precision support
            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum += 1
            
            # Check if we should update parameters (every grad_accum steps or last batch)
            is_flush_step = (accum % args.grad_accum == 0) or (step == num_batches - 1)
            
            # Parameter update when accumulated gradients are ready
            if is_flush_step:
                if warmup_scheduler is not None and global_step < args.lr_warmup_steps:
                    warmup_scheduler.step(global_step)
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            # Accumulate loss for logging (scale back up to get true loss value)
            train_loss += loss.item() * args.grad_accum

        
        train_loss /= len(train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        # log learning rate
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        # eval/val
        model.eval()
        val_loss = 0.0
        # Per-class dice statistics accumulator
        val_stats = defaultdict(lambda: {'intersection': 0, 'pred': 0, 'target': 0})
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.squeeze(1).long().to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                
                # Accumulate per-class statistics for dice computation
                accumulate_per_class_stats(preds, masks, args.num_classes, val_stats)

            denorm = torch.stack([
                denormalize_tensor(im, norm_mean, norm_std) for im in images
            ])
            writer.add_images("Valid/Images", denorm, epoch)
            colored_gt = apply_colormap(masks, num_classes=args.num_classes)
            colored_pred = apply_colormap(outputs.argmax(dim=1), num_classes=args.num_classes)
            writer.add_images("Valid/ColoredGT", colored_gt, epoch)
            writer.add_images("Valid/ColoredPred", colored_pred, epoch)


        val_loss /= len(val_loader)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        # dice/iou metrics
        dice_per_class, _, iou_scores, mean_iou = compute_metrics_from_stats(val_stats)
        val_dice = np.mean(dice_per_class)
        val_dice_no_bg = np.mean(dice_per_class[1:])
        val_mean_iou_no_bg = np.mean(iou_scores[1:])
        writer.add_scalar("Dice/Val_mean", val_dice, epoch)
        writer.add_scalar("Dice/Val_mean_no_bg", val_dice_no_bg, epoch)
        writer.add_scalar("IOU/Val_mean", mean_iou, epoch)
        writer.add_scalar("IOU/Val_mean_no_bg", val_mean_iou_no_bg, epoch)
  
        for cls_id, cls_dice in enumerate(dice_per_class):
            writer.add_scalar(f"Dice/Class_{cls_id}", cls_dice, epoch)
        for cls_id, cls_iou in enumerate(iou_scores):
            writer.add_scalar(f"IOU/Class_{cls_id}", cls_iou, epoch)
        
        # best weights of loss/dice/iou
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            torch.save(model.state_dict(), os.path.join(log_dir, "best_loss_model.pth"))
            logging.info(f"Saved best loss model at epoch {epoch}")
        if val_dice_no_bg > best_val_dice_no_bg:
            best_val_dice_no_bg = val_dice_no_bg
            best_dice_epoch = epoch
            torch.save(model.state_dict(), os.path.join(log_dir, "best_dice_model.pth"))
            logging.info(f"Saved best dice model at epoch {epoch}")
        if val_mean_iou_no_bg > best_iou:
            best_iou = val_mean_iou_no_bg
            best_iou_epoch = epoch
            torch.save(model.state_dict(), os.path.join(log_dir, "best_iou_model.pth"))
            logging.info(f"Saved best iou model at epoch {epoch}")

        # save to TensorBoard
        writer.add_scalar("Best/Val_Dice_no_bg", best_val_dice_no_bg, epoch)
        writer.add_scalar("Best/Val_Dice_no_bg_epoch", best_dice_epoch, epoch)
        writer.add_scalar("Best/Val_IOU", best_iou, epoch)
        writer.add_scalar("Best/Val_IOU_epoch", best_iou_epoch, epoch)
        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice (no bg): {val_dice_no_bg:.4f}")
        logging.info(f"\tIOU: {mean_iou:.4f}, IOU (no bg): {val_mean_iou_no_bg:.4f}")
        logging.info(f"Best Val Dice (No BG): {best_val_dice_no_bg:.4f} at epoch {best_dice_epoch}")
        logging.info(f"Best Val IOU: {best_iou:.4f} at epoch {best_iou_epoch}")
        logging.info(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_loss_epoch}")

        if scheduler is not None and args.scheduler != "OneCycleLR":
            if args.scheduler == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        

    # save the last model after training
    torch.save(model.state_dict(), os.path.join(log_dir, "last_model.pth"))

    # TEST SET*******************************************************
    if args.test_img_dir and args.test_mask_dir:
        test_fns = [f for f in os.listdir(args.test_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        test_dataset = SegmentationDataset(
            img_dir=args.test_img_dir,
            mask_dir=args.test_mask_dir,
            file_list=test_fns,
            album_aug=val_transform,
            mask_suffix=args.mask_suffix,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory,
                                 persistent_workers=args.persistent_workers if args.num_workers > 0 else False)
        logging.info(f"Test dataset size: {len(test_dataset)}")

        # evaluate last model
        evaluate_model_on_loader(model, test_loader, criterion, args, writer, "Test_last", norm_mean, device)

        # evaluate best dice model
        best_dice_model_path = os.path.join(log_dir, "best_dice_model.pth")
        if os.path.exists(best_dice_model_path):
            model.load_state_dict(torch.load(best_dice_model_path, map_location=device))
            evaluate_model_on_loader(model, test_loader, criterion, args, writer, "Test_best_dice", norm_mean, device)

        # evaluate best iou model
        best_iou_model_path = os.path.join(log_dir, "best_iou_model.pth")
        if os.path.exists(best_iou_model_path):
            model.load_state_dict(torch.load(best_iou_model_path, map_location=device))
            evaluate_model_on_loader(model, test_loader, criterion, args, writer, "Test_best_iou", norm_mean, device)

        # free resources
        del test_loader
        del test_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--val_img_dir", type=str, default=None, help="Validation image directory (if different from training)")
    parser.add_argument("--val_mask_dir", type=str, default=None, help="Validation mask directory (if different from training)")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_img_dir", type=str, default=None, help="Test image directory")
    parser.add_argument("--test_mask_dir", type=str, default=None, help="Test mask directory")

    parser.add_argument("--optimizer", type=str, default="adamw", 
                        choices=["adam", "adamw"], help="Optimizer to use")
    parser.add_argument("--model", type=str, choices=["unet", "deeplab","deeplab_resnet50", "unet_resnet50"], default="deeplab")
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR",
                        choices=["CosineAnnealingLR", "StepLR", "ReduceLROnPlateau", "OneCycleLR"])
    parser.add_argument("--weighted_loss", action="store_true", help="Use weighted loss based on class frequencies")
    parser.add_argument("--loss_fn", type=str, default="cross_entropy", choices=["cross_entropy", "DiceCE"], help="Loss function to use")
    
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    parser.add_argument("--log_dir", type=str, default="runs/train_general")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mask_suffix", type=str, default="", help="Suffix appended to image stem for mask files, e.g. '' for 'image.png' or '_mask' for 'image_mask.png'")

    parser.add_argument("--aug_config", type=str, default=None, help="Path to JSON defining augmentation pipeline")
    
    # Dataloader related
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr-warmup", action="store_true", help="Enable learning rate warmup")
    parser.add_argument("--lr-warmup-steps", type=int, default=500, help="Warmup steps")
 
    
    args = parser.parse_args()

    #############################################################################
    # DEBUG PARAMETERS SECTION - FOR DEVELOPMENT USE ONLY
    # WARNING: Uncomment the lines below ONLY for debugging in IDE/VS Code
    # ALWAYS comment out before running in production or submitting code
    #############################################################################
    
    # args = parser.parse_args([])  
    # args.img_dir = "./data/cefe_multi/data"
    # args.mask_dir = "./data/cefe_multi/mask"
    
    # args.val_img_dir = None
    # args.val_mask_dir = None

    # args.model = "deeplab"  # or "unet"
    # args.resize = 256
    # args.num_classes = 11
    # args.batch_size = 2
    # args.epochs = 5
    # args.optimizer = "adamw"
    # args.lr = 1e-4
    # args.val_split = 0.2
    # args.log_dir = f"./runs"
    # args.device = "cuda"  # or "cuda" if available
    # args.mask_suffix = "_mask"  # e.g., "_mask" for "image_mask.png"
    # args.scheduler = "OneCycleLR"  
    # args.weighted_loss = True  # Use weighted loss
    # args.loss_fn = "cross_entropy"  # or "dice+ce"
    # args.aug_config = "./models/augs/default.json"  # Path to augmentation config JSON
    
    # args.num_workers = 4
    # args.pin_memory = True
    # args.persistent_workers = True
    
    # args.amp = True  # Enable mixed precision
    # args.grad_accum = 1
    
    #############################################################################
    # END DEBUG PARAMETERS SECTION
    #############################################################################

    main_general(args)
