# SPDX-License-Identifier: MIT
# train_sam.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import sys
from pathlib import Path
import gc
# root_logger = logging.getLogger()
# for h in list(root_logger.handlers):
#     root_logger.removeHandler(h)

# root_logger.setLevel(logging.INFO)



from dataset.dataset import SegmentationDataset, cls_weights, compute_class_counts_loader, make_pad_if_needed
from models.sam_net import SAM_MSU
from models.util import (apply_colormap, compute_dice, overlay_mask_on_image, DiceCELoss,
                         build_augmentation_from_config, accumulate_per_class_stats, compute_dice_from_stats,
                         denormalize_tensor, compute_custom_class_weight, balanced_class_weight_from_counts,
                         compute_metrics_from_stats)
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import json
from collections import defaultdict

from torch.amp import autocast, GradScaler

# Optional: LR warmup support
try:
    from models.warmup import LRWarmupScheduler
except Exception:
    LRWarmupScheduler = None


# Image file extensions filter
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def filter_image_files(file_list):
    """Filter file list to include only image files with supported extensions."""
    return [f for f in file_list if f.lower().endswith(IMAGE_EXTENSIONS)]


def evaluate_model_on_loader(model, loader, criterion, args, writer, global_epoch_label, denorm_stats, device):
    """
    Evaluate model on a given data loader (validation or test set).
    
    Args:
        model: The model to evaluate
        loader: DataLoader to evaluate on
        criterion: Loss function
        args: Arguments namespace
        writer: TensorBoard SummaryWriter (can be None)
        global_epoch_label: Label for TensorBoard logging (e.g., "Test_last", "Test_best_dice")
        denorm_stats: Denormalization statistics (unused, kept for compatibility)
        device: Device to run evaluation on
    
    Returns:
        tuple: (eval_loss, dice_no_bg, iou_no_bg)
    """
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
    if args.eval_partimagenet:
        # For PartImageNet, use last class as background
        iou_no_bg = np.mean(iou_scores[:-1])
        dice_no_bg = np.mean(dice_per_class[:-1])
    else:
        iou_no_bg = np.mean(iou_scores[1:])
        dice_no_bg = np.mean(dice_per_class[1:])

    # Write to TensorBoard
    if writer is not None:
        writer.add_scalar(f"Loss/{global_epoch_label}", eval_loss, args.epochs)
        writer.add_scalar(f"IOU/{global_epoch_label}_mean", mean_iou, args.epochs)
        writer.add_scalar(f"Dice/{global_epoch_label}_mean", dice_mean, args.epochs)
        writer.add_scalar(f"IOU/{global_epoch_label}_mean_no_bg", iou_no_bg, args.epochs)
        writer.add_scalar(f"Dice/{global_epoch_label}_mean_no_bg", dice_no_bg, args.epochs)
        
        # Log per-class metrics
        for cls_id, cls_dice in enumerate(dice_per_class):
            writer.add_scalar(f"Dice/{global_epoch_label}_Class_{cls_id}", cls_dice, args.epochs)
        for cls_id, cls_iou in enumerate(iou_scores):
            writer.add_scalar(f"IOU/{global_epoch_label}_Class_{cls_id}", cls_iou, args.epochs)
    
    logging.info(f"{global_epoch_label}: Loss: {eval_loss:.4f} | meanIoU: {mean_iou:.4f} | Dice (no bg): {dice_no_bg:.4f} | mIoU(no bg): {iou_no_bg:.4f}")

    return eval_loss, dice_no_bg, iou_no_bg

def main_sam(args):
    """
    Main training function for SAM-based segmentation models.
    
    Supports SAM encoder + Multi stage upsampling decoder with various optimizations:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Custom augmentations
    - Separate tracking for loss, dice, and IoU optimization
    - Test set evaluation at training end
    """

    device = args.device

    assert args.output_size is not None, "Please specify --output_size"       
    out_hw = (args.output_size[0], args.output_size[1])
    
    # Configure logging to output to both console and file
    run_id = f"{args.backend}_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    log_dir = os.path.join(args.log_dir, run_id)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging configuration
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log all arguments
    logging.info("Arguments:")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

    # Configure mixed precision training
    amp_enabled = bool(args.amp and device == "cuda" or (isinstance(device, str) and device.startswith("cuda")))
    scaler = GradScaler(device="cuda", enabled=amp_enabled)
    
    # Normalization statistics
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    
    # Load augmentation configuration
    aug_config = None
    if args.aug_config:
        try:
            with open(args.aug_config, 'r') as f:
                aug_config = json.load(f)
                train_transform = build_augmentation_from_config(
                    aug_list=aug_config['train'],
                    resize_height=out_hw[0],
                    resize_width=out_hw[1]
                )

                val_transform = build_augmentation_from_config(
                    aug_list=aug_config['val'],
                    resize_height=out_hw[0],
                    resize_width=out_hw[1]
                )
        except Exception as e:
            logging.error(f"Error loading augmentation config from {args.aug_config}: {e}")
            logging.info("Falling back to default augmentations")
            args.aug_config = None

    if not args.aug_config:
        train_transform = A.Compose([
            A.LongestMaxSize(max_size=max(out_hw[0],out_hw[1] ), interpolation=1),
            make_pad_if_needed(min_height=out_hw[0], min_width=out_hw[1], border_mode=0, image_fill=0, mask_fill=0),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ],
            additional_targets={'mask': 'mask'}
        )

        val_transform = A.Compose([
            A.LongestMaxSize(max_size=max(out_hw[0],out_hw[1] ), interpolation=1),
            make_pad_if_needed(min_height=out_hw[0], min_width=out_hw[1], border_mode=0, image_fill=0, mask_fill=0),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})


    all_filenames = sorted(os.listdir(args.img_dir))
    all_filenames = filter_image_files(all_filenames)
    
    # With explicit validation directories
    if args.val_img_dir and args.val_mask_dir:
        train_fns = all_filenames 
        val_fns = filter_image_files(os.listdir(args.val_img_dir))
    # Without explicit validation directories, split the dataset
    else:
        train_fns, val_fns = train_test_split(all_filenames, test_size=args.val_split, random_state=42)


    train_dataset = SegmentationDataset(img_dir=args.img_dir, mask_dir=args.mask_dir, file_list=train_fns, 
                                        album_aug=train_transform,
                                        mask_suffix=args.mask_suffix)
    
    
    if args.val_img_dir and args.val_mask_dir:
        val_dataset = SegmentationDataset(img_dir=args.val_img_dir, mask_dir=args.val_mask_dir, file_list=val_fns, 
                                          album_aug=val_transform,
                                          mask_suffix=args.mask_suffix)
    else:
        val_dataset = SegmentationDataset(img_dir=args.img_dir, mask_dir=args.mask_dir, file_list=val_fns, 
                                        album_aug=val_transform,
                                        mask_suffix=args.mask_suffix)

    
    logging.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
    

    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              drop_last=True,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory,
                              persistent_workers=args.persistent_workers if args.num_workers > 0 else False)
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            persistent_workers=args.persistent_workers if args.num_workers > 0 else False)
    
        
    logging.info(f"Number of batches - Train: {len(train_loader)}, Val: {len(val_loader)}")

    # SAM encoder + selectable decoder
    logging.info(f"Loading SAM model: backend={args.backend}, model_type={args.model_type}")
    model = SAM_MSU(
        sam_checkpoint=args.sam_checkpoint,
        model_type=args.model_type,
        backend=args.backend,
        num_classes=args.num_classes,
        full_train=args.full_train,
        enhanced_decoder=args.enhanced_decoder,
        sam2_config=args.sam2_config,
        output_size=out_hw
    ).to(device)
    
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, out_hw[0], out_hw[1], device=device))
    logging.info("Model initialized successfully")

    # Loss function configuration with optional class weighting
    if args.weighted_loss:
        counts = compute_class_counts_loader(train_loader, args.num_classes)
        class_weights = balanced_class_weight_from_counts(counts)
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
    logging.info(f"Using loss function: {args.loss_fn}")
    
    # Optimizer configuration
    if args.full_train:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

    # TensorBoard writer (log_dir already created during logging setup)
    writer = SummaryWriter(log_dir=log_dir)

    # Save arguments configuration
    args_dict = vars(args)
    try:
        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump(args_dict, f, indent=2)
    except Exception as e:
        logging.warning(f"Could not save args.json: {e}")

    # Save augmentation config if used
    if aug_config is not None:
        try:
            with open(os.path.join(log_dir, "aug_used.json"), "w") as f:
                json.dump(aug_config, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save aug_used.json: {e}")

    # Learning rate scheduler configuration
    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    else:
        scheduler = None

    # Initialize learning rate warmup scheduler if enabled
    warmup_scheduler = None
    global_step = 0
    if args.lr_warmup and LRWarmupScheduler is not None:
        warmup_scheduler = LRWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=args.lr_warmup_steps,
            base_lr=args.lr
        )
        logging.info(f"Learning rate warmup enabled: {args.lr_warmup_steps} steps")

    # Initialize best metrics tracking
    best_val_loss = float('inf')
    best_val_dice_no_bg = 0.0
    best_val_iou = 0.0
    best_loss_epoch = 0
    best_dice_epoch = 0
    best_iou_epoch = 0
    
    logging.info("Starting training...")
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

            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision support
            with autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, masks)
                # Scale loss for gradient accumulation
                loss = loss / args.grad_accum
            
            # Backward pass with mixed precision support
            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accum += 1
            
            # Check if we should update parameters (every grad_accum steps or last batch)
            is_flush_step = (accum % args.grad_accum == 0) or (step == num_batches - 1)
            
            # Parameter update when accumulated gradients are ready
            if is_flush_step:
                # Apply learning rate warmup if enabled and still in warmup phase
                if warmup_scheduler is not None and global_step < args.lr_warmup_steps:
                    warmup_scheduler.step(global_step)
                
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Increment global step counter
                global_step += 1
            
            # Accumulate loss for logging (scale back to get true loss value)
            train_loss += loss.item() * args.grad_accum

        train_loss /= len(train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        # Log learning rate
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_stats = defaultdict(lambda: {'intersection': 0, 'pred': 0, 'target': 0})
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.squeeze(1).long().to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                accumulate_per_class_stats(preds, masks, args.num_classes, val_stats)

        # TensorBoard logging for validation images (every 5 epochs to improve performance)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            denorm = torch.stack([denormalize_tensor(im, norm_mean, norm_std) for im in images])
            writer.add_images("Valid/Images", denorm, epoch)
            colored_gt = apply_colormap(masks, num_classes=args.num_classes)
            colored_pred = apply_colormap(outputs.argmax(dim=1), num_classes=args.num_classes)

            writer.add_images("Valid/ColoredGT", colored_gt, epoch)
            writer.add_images("Valid/ColoredPred", colored_pred, epoch)
        
        val_loss /= len(val_loader)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        
        # Compute dice and IoU metrics from accumulated statistics
        dice_per_class, _, iou_scores, mean_iou = compute_metrics_from_stats(val_stats)
        val_dice = np.mean(dice_per_class)
        writer.add_scalar("Dice/Val_mean", val_dice, epoch)
        writer.add_scalar("IOU/Val_mean", mean_iou, epoch)
        
        # Dice and IoU scores excluding background (class 0)
        if args.eval_partimagenet:
            # For PartImageNet, use last class as background
            dice_per_class_no_bg = dice_per_class[:-1]
            iou_per_class_no_bg = iou_scores[:-1]
        else:
            dice_per_class_no_bg = dice_per_class[1:]
            iou_per_class_no_bg = iou_scores[1:]
        val_dice_no_bg = np.mean(dice_per_class_no_bg)
        val_iou_no_bg = np.mean(iou_per_class_no_bg)
        writer.add_scalar("Dice/Val_mean_no_bg", val_dice_no_bg, epoch)
        writer.add_scalar("IOU/Val_mean_no_bg", val_iou_no_bg, epoch)

        # Log per-class metrics
        for cls_id, cls_dice in enumerate(dice_per_class):
            writer.add_scalar(f"Dice/Class_{cls_id}", cls_dice, epoch)
        for cls_id, cls_iou in enumerate(iou_scores):
            writer.add_scalar(f"IOU/Class_{cls_id}", cls_iou, epoch)

        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice (no bg): {val_dice_no_bg:.4f}")
        logging.info(f"\tIOU: {mean_iou:.4f}, IOU (no bg): {val_iou_no_bg:.4f}")
        
        # Save best loss model with exception handling
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            try:
                loss_model_path = os.path.join(log_dir, 'best_loss_model.pth')
                torch.save(model.state_dict(), loss_model_path)
                logging.info(f"Saved best loss model at epoch {epoch}")
            except Exception as e:
                logging.error(f"Error saving best loss model at epoch {epoch}: {e}")
        
        # Save best dice model with exception handling
        if val_dice_no_bg > best_val_dice_no_bg:
            best_val_dice_no_bg = val_dice_no_bg
            best_dice_epoch = epoch
            try:
                dice_model_path = os.path.join(log_dir, 'best_dice_model.pth')
                torch.save(model.state_dict(), dice_model_path)
                logging.info(f"Saved best dice model at epoch {epoch}")
            except Exception as e:
                logging.error(f"Error saving best dice model at epoch {epoch}: {e}")
        
        # Save best IoU model with exception handling
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            best_iou_epoch = epoch
            try:
                iou_model_path = os.path.join(log_dir, 'best_iou_model.pth')
                torch.save(model.state_dict(), iou_model_path)
                logging.info(f"Saved best iou model at epoch {epoch}")
            except Exception as e:
                logging.error(f"Error saving best iou model at epoch {epoch}: {e}")
        
        # Step the learning rate scheduler (only after warmup is complete)
        if scheduler is not None:
            if warmup_scheduler is None or global_step >= args.lr_warmup_steps:
                if args.scheduler == "ReduceLROnPlateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        
        # Log best metrics
        logging.info(f"Best Val Dice (No BG): {best_val_dice_no_bg:.4f} at epoch {best_dice_epoch}")
        logging.info(f"Best Val IOU: {best_val_iou:.4f} at epoch {best_iou_epoch}")
        logging.info(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_loss_epoch}")
        
        # Log best metrics to TensorBoard
        writer.add_scalar("Best/Val_Dice_no_bg", best_val_dice_no_bg, epoch)
        writer.add_scalar("Best/Val_Dice_no_bg_epoch", best_dice_epoch, epoch)
        writer.add_scalar("Best/Val_IOU", best_val_iou, epoch)
        writer.add_scalar("Best/Val_IOU_epoch", best_iou_epoch, epoch)

    # Save the last model after training
    try:
        last_model_path = os.path.join(log_dir, 'last_model.pth')
        torch.save(model.state_dict(), last_model_path)
        logging.info("Saved last model at end of training")
    except Exception as e:
        logging.error(f"Error saving last model at end of training: {e}")

    # Test set evaluation (if test directories are provided)
    if args.test_img_dir and args.test_mask_dir:
        logging.info("Starting test set evaluation...")
        
        # Free training/validation loaders and reclaim memory
        try:
            del train_loader
            del val_loader
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build test dataset/loader (re-use val_transform for deterministic preprocessing)
        test_fns = filter_image_files(os.listdir(args.test_img_dir))
        test_dataset = SegmentationDataset(
            img_dir=args.test_img_dir,
            mask_dir=args.test_mask_dir,
            file_list=test_fns,
            album_aug=val_transform,
            mask_suffix=args.mask_suffix,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        )
        logging.info(f"Test dataset size: {len(test_dataset)}")
        logging.info(f"Number of batches - Test: {len(test_loader)}")

        # Evaluate last model (already loaded)
        evaluate_model_on_loader(model, test_loader, criterion, args, writer, "Test_last", denorm_stats=None, device=device)

        # Evaluate best dice model if exists
        best_dice_model_path = os.path.join(log_dir, 'best_dice_model.pth')
        if os.path.exists(best_dice_model_path):
            logging.info("Loading best validation dice model for test set evaluation...")
            model.load_state_dict(torch.load(best_dice_model_path, map_location=device))
            evaluate_model_on_loader(model, test_loader, criterion, args, writer, "Test_best_dice", denorm_stats=None, device=device)
        else:
            logging.warning("best_dice_model.pth not found, skipping best dice model evaluation on test set.")

        # Evaluate best iou model if exists
        best_iou_model_path = os.path.join(log_dir, 'best_iou_model.pth')
        if os.path.exists(best_iou_model_path):
            logging.info("Loading best validation iou model for test set evaluation...")
            model.load_state_dict(torch.load(best_iou_model_path, map_location=device))
            evaluate_model_on_loader(model, test_loader, criterion, args, writer, "Test_best_iou", denorm_stats=None, device=device)
        else:
            logging.warning("best_iou_model.pth not found, skipping best iou model evaluation on test set.")

        # Cleanup test loader/dataset
        del test_loader
        del test_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()
    logging.info("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM-based segmentation model")
    
    # Data directories
    parser.add_argument("--img_dir", type=str, required=False, help="Training image directory")
    parser.add_argument("--mask_dir", type=str, required=False, help="Training mask directory")
    parser.add_argument("--val_img_dir", type=str, default=None, help="Validation image directory (if different from training)")
    parser.add_argument("--val_mask_dir", type=str, default=None, help="Validation mask directory (if different from training)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation, won't affect if val dirs are provided")
    parser.add_argument("--test_img_dir", type=str, default=None,
                        help="Test image directory (optional). If provided together with --test_mask_dir, will be evaluated at the end.")
    parser.add_argument("--test_mask_dir", type=str, default=None,
                        help="Test mask directory (optional). Must be provided together with --test_img_dir.")

    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        default=[1024, 1024],   # default output size
        metavar=('H', 'W'),
        help="Final output segmentation resolution as (H W)."
    )


    # Model configuration
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM checkpoint file")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="SAM model type")
    parser.add_argument("--backend", type=str, default="sam", choices=["sam", "sam2", "sam3"], help="SAM backend to use (sam, sam2, or sam3)")
    parser.add_argument("--sam2_config", type=str, default=None, 
                        help="Path to SAM2 config yaml file (e.g., sam2_hiera_l.yaml). If not provided, will be inferred from model_type.")
    parser.add_argument("--full_train", action="store_true", help="Full model training (SAM + decoder)")
    parser.add_argument("--enhanced_decoder", action="store_true", help="Use enhanced decoder (GN+SiLU, deeper upsampling)")
    
    # Training configuration
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR", 
                        choices=["CosineAnnealingLR", "StepLR", "ReduceLROnPlateau"], help="Learning rate scheduler")
    parser.add_argument("--weighted_loss", action="store_true", help="Use weighted loss based on class frequencies")
    parser.add_argument("--loss_fn", type=str, default="cross_entropy", choices=["cross_entropy", "DiceCE"], help="Loss function to use")
    
    parser.add_argument("--num_classes", type=int, default=10, help="Number of segmentation classes, including background")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # Logging and output
    parser.add_argument("--log_dir", type=str, default="", help="Directory for logs and checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--mask_suffix", type=str, default="", help="Suffix appended to image stem for mask files, e.g. '' for 'image.png' or '_mask' for 'image_mask.png'")

    # Augmentation
    parser.add_argument("--aug_config", type=str, default="./models/augs/default.json", help="Path to JSON defining augmentation pipeline")
    
    # DataLoader configuration
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for faster GPU transfer")
    parser.add_argument("--persistent_workers", action="store_true", help="Keep data loading workers alive between epochs")
    
    # Mixed precision and gradient accumulation
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (fp16)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    
    # Learning rate warmup
    parser.add_argument("--lr-warmup", action="store_true", help="Enable learning rate warmup")
    parser.add_argument("--lr-warmup-steps", type=int, default=500, 
                        help="Number of warmup steps (iterations) for learning rate warmup (default: 500)")
    
    parser.add_argument("--eval_partimagenet", action="store_true", help="if true use the last class (40) as bg class for partimagenet")

    args = parser.parse_args()

    #############################################################################
    # DEBUG PARAMETERS SECTION - FOR DEVELOPMENT USE ONLY
    # WARNING: Uncomment the lines below ONLY for debugging in IDE/VS Code
    # ALWAYS comment out before running in production or submitting code
    #############################################################################
    
    # args = parser.parse_args([])  # for debug inside script
    # args.img_dir = "./data/cefe_multi/data"
    # args.mask_dir = "./data/cefe_multi/mask"
    # args.val_img_dir = None
    # args.val_mask_dir = None
    # args.val_split = 0.2
    # args.test_img_dir = None
    # args.test_mask_dir = None

    # args.sam_checkpoint = "path/to/sam_checkpoint.pth"
    # args.model_type = "vit_b"
    # args.scheduler = "CosineAnnealingLR"  # or "StepLR" or "ReduceLROnPlateau"
    # args.weighted_loss = True  # Use weighted loss
    # args.loss_fn = "cross_entropy"  # or "DiceCE"
    
    # args.num_classes = 11
    # args.batch_size = 2
    # args.epochs = 30
    # args.lr = 1e-4
    
    # args.mask_suffix = "_mask" 
    # args.log_dir = f"./runs/sam_train_run"
    # args.device = "cuda"  # or "cpu"
    
    # args.aug_config = "./models/augs/default.json"
    
    # args.num_workers = 4
    # args.pin_memory = True
    # args.persistent_workers = True
    
    # args.amp = False
    # args.grad_accum = 1
    # args.lr_warmup = False
    # args.lr_warmup_steps = 500
    # args.eval_partimagenet = False
    
    #############################################################################
    # END DEBUG PARAMETERS SECTION
    #############################################################################

    main_sam(args)
