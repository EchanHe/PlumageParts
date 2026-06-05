# SPDX-License-Identifier: MIT
# train_dinov3_net.py

import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import logging

from dataset.dataset import SegmentationDataset, cls_weights, compute_class_counts_loader, make_pad_if_needed


# Import new Multi-Stage Upsampling (MSU) and Multi-Layer Fusion (MLF) models
from models.dinov3_multistage_upsampling import DINOv3_MSU,build_dinov3_backbone
from models.dinov3_multilayer_fusion import DINOv3_MLF, build_dinov3_backbone_mlf
from models.dinov2_multistage_upsampling import DINOv2_MSU, build_dinov2_backbone
from models.dinov2_multilayer_fusion import DINOv2_MLF, build_dinov2_backbone_mlf

from models.util import (apply_colormap, compute_dice, overlay_mask_on_image, DiceCELoss,
                         build_augmentation_from_config, accumulate_per_class_stats, compute_dice_from_stats,
                         denormalize_tensor,compute_custom_class_weight,balanced_class_weight_from_counts,
                         compute_metrics_from_stats)
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import json
from collections import defaultdict
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
import gc
from models.warmup import LRWarmupScheduler

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


def main_dinov3(args):
    device = args.device
    
    assert args.output_size is not None, "Please specify --output_size"       
    out_hw = (args.output_size[0], args.output_size[1])
    
    # Configure logging to output to both console and file
    if "dinov3" in args.model:
        log_dir = os.path.join(args.log_dir, f"{args.model}_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    else:
        log_dir = os.path.join(args.log_dir, f"{args.model}_{args.dinov2_variant}_{datetime.now().strftime('%Y%m%d_%H%M')}")
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
    
    # print the args per line
    logging.info("Arguments:")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

    # Configure mixed precision training
    amp_enabled = bool(args.amp and str(device).startswith("cuda"))
    scaler = GradScaler(device = "cuda", enabled=amp_enabled)
    
    run_id = f"dinov3_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    aug_config = None
    if args.aug_config:
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

    else:
        train_transform = A.Compose([
            A.LongestMaxSize(max_size=max(out_hw[0],out_hw[1] ), interpolation=1) ,
            make_pad_if_needed(min_height=out_hw[0], min_width=out_hw[1], border_mode=0, image_fill=0, mask_fill=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ],
            additional_targets={'mask': 'mask'}  #  mask
        )


        val_transform = A.Compose([
            A.LongestMaxSize(max_size=max(out_hw[0],out_hw[1] ), interpolation=1) ,
            make_pad_if_needed(min_height=out_hw[0], min_width=out_hw[1], border_mode=0, image_fill=0, mask_fill=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})


    all_filenames = sorted(os.listdir(args.img_dir))
    all_filenames = [f for f in all_filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    # with explicit validation directories
    if args.val_img_dir and args.val_mask_dir:
        # Use provided validation directories
        # train_fns = [f for f in all_filenames if f not in os.listdir(args.val_img_dir)]
        
        train_fns = all_filenames 
        val_fns = [f for f in os.listdir(args.val_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    # without explicit validation directories, split the dataset
    else:
        train_fns, val_fns = train_test_split(all_filenames, test_size=args.val_split, random_state=42)


    train_dataset = SegmentationDataset(img_dir=args.img_dir, mask_dir=args.mask_dir, file_list=train_fns, 
                                        album_aug=train_transform,
                                        mask_suffix=args.mask_suffix)
    
    
    if args.val_img_dir and args.val_mask_dir:
        # Use provided validation directories
        val_dataset = SegmentationDataset(img_dir=args.val_img_dir, mask_dir=args.val_mask_dir, file_list=val_fns, 
                                          album_aug=val_transform,
                                          mask_suffix=args.mask_suffix)
    else:
        # Use the same directories as training for validation
        # This is common in many segmentation tasks where the validation set is a subset of the training set
        val_dataset = SegmentationDataset(img_dir=args.img_dir, mask_dir=args.mask_dir, file_list=val_fns, 
                                        album_aug=val_transform,
                                        mask_suffix=args.mask_suffix)

     # if there is a subset ratio, use a subset of the training dataset
    if args.subset_ratio < 1.0:
        subset_ratio = args.subset_ratio
        num_samples = int(len(train_dataset) * subset_ratio)
        np.random.seed(42)  
        subset_indices = np.random.choice(len(train_dataset), num_samples, replace=False)
        train_dataset = Subset(train_dataset, subset_indices)
 
    
    logging.info(f"Train dataset size: {len(train_dataset)} with subset ratio {args.subset_ratio}, Val dataset size: {len(val_dataset)}")
    


    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory,
                              persistent_workers=args.persistent_workers if args.num_workers > 0 else False)

    val_loader = DataLoader(val_dataset, 
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=args.pin_memory,
                             persistent_workers=args.persistent_workers if args.num_workers > 0 else False)
    
        
    logging.info(f"Number of batches - Train: {len(train_loader)}, Val: {len(val_loader)}")

    # Build backbone and model based on --model argument
    logging.info(f"Building model: {args.model}...")
    
   
    if args.model == "dinov3_msu":
        # DINOv3 Multi-Stage Upsampling (MSU)
        logging.info(f"Loading DINOv3 backbone: {args.variant}")
        dino_backbone = build_dinov3_backbone(
            variant=args.variant,
            weights=args.dinov3_weights,
        )

        model = DINOv3_MSU(
            dino_backbone=dino_backbone,
            num_classes=args.num_classes,
            freeze_encoder=not args.full_train,
            enhanced_decoder=args.enhanced_decoder,
            take_n=args.take_n,
            output_size=out_hw,
            use_batch_norm=args.batch_norm
        ).to(device)
    elif args.model == "dinov3_mlf":
        # DINOv3 Multi-Layer Fusion (MLF)
        logging.info(f"Loading DINOv3 backbone: {args.variant}")
        dino_backbone = build_dinov3_backbone_mlf(
            variant=args.variant,
            weights=args.dinov3_weights,
        )

        model = DINOv3_MLF(
            dino_backbone=dino_backbone,
            num_classes=args.num_classes,
            freeze_encoder=not args.full_train,
            take_n=args.take_n,
            output_size=out_hw
        ).to(device)
    elif args.model == "dinov2_msu":
        # DINOv2 Multi-Stage Upsampling (MSU)
        logging.info(f"Loading DINOv2 backbone: {args.dinov2_variant}")
        dino_backbone = build_dinov2_backbone(
            variant=args.dinov2_variant,
            weights=args.dinov2_weights,
        )

        model = DINOv2_MSU(
            dino_backbone=dino_backbone,
            num_classes=args.num_classes,
            freeze_encoder=not args.full_train,
            enhanced_decoder=args.enhanced_decoder,
            take_n=args.take_n,
            output_size=out_hw
        ).to(device)
    elif args.model == "dinov2_mlf":
        # DINOv2 Multi-Layer Fusion (MLF)
        logging.info(f"Loading DINOv2 backbone: {args.dinov2_variant}")
        dino_backbone = build_dinov2_backbone_mlf(
            variant=args.dinov2_variant,
            weights=args.dinov2_weights,
        )

        model = DINOv2_MLF(
            dino_backbone=dino_backbone,
            num_classes=args.num_classes,
            freeze_encoder=not args.full_train,
            take_n=args.take_n,
            output_size=out_hw
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")


    # Initialize lazy projection with a dummy forward at the target input size (out_hw[0], out_hw[1])
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, out_hw[0], out_hw[1], device=device))
    logging.info("finished model initialization")
    if args.weighted_loss:
        # class_weights = compute_class_weight(
        #     class_weight='balanced',
        #     classes=np.arange(args.num_classes),
        #     y=cls_weights(train_dataset)
        # )
        
        # class_weights = compute_custom_class_weight(y_true = cls_weights(train_dataset),
        #                             all_possible_classes=np.arange(args.num_classes),
        #                             handle_missing=True, missing_weight=1e-6)
        
        
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
    
    if args.optim == "adamw":
        if args.full_train:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            # Train only projection + decoder when encoder is frozen
            # MSU models (unet-style) use single proj layer
            if args.model in ("dinov3_msu", "dinov2_msu"):
                optimizer = torch.optim.AdamW(list(model.proj.parameters()) + list(model.decoder.parameters()), lr=args.lr)
            # MLF models (fpn-style) use multiple proj_layers
            elif args.model in ( "dinov3_mlf", "dinov2_mlf"):
                optimizer = torch.optim.AdamW(list(model.proj_layers.parameters()) + list(model.decoder.parameters()), lr=args.lr)
    elif args.optim == "adam":  # default to adam    
        if args.full_train:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            # Train only projection + decoder when encoder is frozen
            # MSU models (unet-style) use single proj layer
            if args.model in ("dinov3_msu", "dinov2_msu"):
                optimizer = torch.optim.Adam(list(model.proj.parameters()) + list(model.decoder.parameters()), lr=args.lr)
            # MLF models (fpn-style) use multiple proj_layers
            elif args.model in ("dinov3_mlf", "dinov2_mlf"):
                optimizer = torch.optim.Adam(list(model.proj_layers.parameters()) + list(model.decoder.parameters()), lr=args.lr)

    # log_dir was already created during logging setup
    # log_dir = os.path.join(args.log_dir, run_id)
    # os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    args_dict = vars(args)  # args to dict
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    if aug_config is not None:
        with open(os.path.join(log_dir, "aug_used.json"), "w") as f:
            json.dump(aug_config, f, indent=2)

    
    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize learning rate warmup scheduler if enabled
    warmup_scheduler = None
    if args.lr_warmup:
        warmup_scheduler = LRWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=args.lr_warmup_steps,
            base_lr=args.lr
        )
        logging.info(f"Learning rate warmup enabled: {args.lr_warmup_steps} steps")
    
    best_val_loss = float('inf')
    best_iou = 0
    best_loss_epoch = 0
    best_iou_epoch = 0
    
    best_dice = 0
    best_dice_epoch = 0
    
    # Global step counter for learning rate warmup (tracks iterations across all epochs)
    global_step = 0
    
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
            masks = masks.long().to(device)

            optimizer.zero_grad()

            # Forward pass with mixed precision support
            with autocast(device_type = "cuda", enabled=amp_enabled):
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
                # Apply learning rate warmup if enabled and still in warmup phase
                if warmup_scheduler is not None and global_step < args.lr_warmup_steps:
                    warmup_scheduler.step(global_step)
                
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Increment global step counter after each optimizer step
                global_step += 1
            
            # Accumulate loss for logging (scale back up to get true loss value)
            train_loss += loss.item() * args.grad_accum
            

        train_loss /= len(train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        # log learning rate
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        
        
        model.eval()
        val_loss = 0.0
        # val_dice = 0.0
        val_stats = defaultdict(lambda: {'intersection': 0, 'pred': 0, 'target': 0})
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.long().to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                
                accumulate_per_class_stats(preds, masks, args.num_classes, val_stats)
                # val_dice += compute_dice(preds, masks, args.num_classes)

        # TensorBoard logging
        denorm = torch.stack([denormalize_tensor(im, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) for im in images])
        writer.add_images("Valid/Images", denorm, epoch)
        colored_gt = apply_colormap(masks, num_classes=args.num_classes)
        colored_pred = apply_colormap(outputs.argmax(dim=1), num_classes=args.num_classes)

        writer.add_images("Valid/ColoredGT", colored_gt, epoch)
        writer.add_images("Valid/ColoredPred", colored_pred, epoch)
        
        val_loss /= len(val_loader)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        
        # Compute dice scores from accumulated statistics
        # dice_per_class = compute_dice_from_stats(val_stats)
        dice_per_class, _, iou_scores, mean_iou = compute_metrics_from_stats(val_stats)
        val_dice = np.mean(dice_per_class)
        writer.add_scalar("Dice/Val_mean", val_dice, epoch) 
        writer.add_scalar("IOU/Val_mean", mean_iou, epoch) 
        # Dice score excluding background (class 0)
        if args.eval_partimagenet:
            # for partimagenet, use last class as bg
            dice_per_class_no_bg = dice_per_class[:-1]  # assumes last class is background
            iou_par_per_class_no_bg = iou_scores[:-1]
        else:
            dice_per_class_no_bg = dice_per_class[1:]  # assumes class 0 is background
            iou_par_per_class_no_bg = iou_scores[1:]
    
        val_dice_no_bg = np.mean(dice_per_class_no_bg)
        writer.add_scalar("Dice/Val_mean_no_bg", val_dice_no_bg, epoch)
        val_mean_iou_no_bg = np.mean(iou_par_per_class_no_bg)
        writer.add_scalar("IOU/Val_mean_no_bg", val_mean_iou_no_bg, epoch)

        

        # Update best iou tracking and save model if improved
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_iou_epoch = epoch
            # Save best metrics model
            try:
                iou_model_path = os.path.join(log_dir, 'best_iou_model.pth')
                torch.save(model.state_dict(), iou_model_path)
                logging.info(f"Saved best iou model at epoch {epoch}")
            except Exception as e:
                logging.info(f"Error saving best iou model at epoch {epoch}: {e}")
        
        if val_dice_no_bg > best_dice:
            best_dice = val_dice_no_bg
            best_dice_epoch = epoch
            # Save best dice model
            try:
                dice_model_path = os.path.join(log_dir, 'best_dice_model.pth')
                torch.save(model.state_dict(), dice_model_path)
                logging.info(f"Saved best dice model at epoch {epoch}")
            except Exception as e:
                logging.info(f"Error saving best dice model at epoch {epoch}: {e}")
        
        for cls_id, cls_dice in enumerate(dice_per_class):
            writer.add_scalar(f"Dice/Class_{cls_id}", cls_dice, epoch)
        
        for cls_id, cls_iou in enumerate(iou_scores):
            writer.add_scalar(f"IOU/Class_{cls_id}", cls_iou, epoch)

        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice (no bg): {val_dice_no_bg:.4f}")
        logging.info(f"\tIOU: {mean_iou:.4f}, IOU (no bg): {val_mean_iou_no_bg:.4f}")
        
        # Save model checkpoints with exception handling
        # Save best loss model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            try:
                loss_model_path = os.path.join(log_dir, 'best_loss_model.pth')
                torch.save(model.state_dict(), loss_model_path)
                logging.info(f"Saved best loss model at epoch {epoch}")
            except Exception as e:
                logging.info(f"Error saving best loss model at epoch {epoch}: {e}")

        # Step the learning rate scheduler (only after warmup is complete)
        # During warmup, the warmup_scheduler handles LR adjustments
        if args.scheduler != "OneCycleLR":
            # Only step the regular scheduler if warmup is disabled or warmup is complete
            if warmup_scheduler is None or global_step >= args.lr_warmup_steps:
                if args.scheduler == "ReduceLROnPlateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        
        # Print best metrics for both loss and dice
        logging.info(f"Best Val Dice (No BG): {best_dice:.4f} at epoch {best_dice_epoch}")
        logging.info(f"Best Val IOU: {best_iou:.4f} at epoch {best_iou_epoch}")
        logging.info(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_loss_epoch}")
        
        # Log the best dice and epoch to tensorboard
        
        writer.add_scalar("Best/Val_Dice_no_bg", best_dice, epoch)
        writer.add_scalar("Best/Val_Dice_no_bg_epoch", best_dice_epoch, epoch)
        writer.add_scalar("Best/Val_IOU", best_iou, epoch)
        writer.add_scalar("Best/Val_IOU_epoch", best_iou_epoch, epoch)


    # save the last model
    try:
        last_model_path = os.path.join(log_dir, 'last_model.pth')
        torch.save(model.state_dict(), last_model_path)
        logging.info(f"Saved last model at end of training")
    except Exception as e:
        logging.info(f"Error saving last model at end of training: {e}")

    # Final test evaluation (if test dirs provided)
    if args.test_img_dir and args.test_mask_dir:
        # 1) Free training/validation loaders and reclaim memory
        try:
            del train_loader
            del val_loader
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2) Build test dataset/loader now (re-use val_transform for deterministic preprocessing)
        test_fns = [
            f for f in os.listdir(args.test_img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
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

        # 3) Evaluate (same logic as validation)
        def evaluate_model_on_test(model, test_loader, criterion, args, writer, global_epoch_label):
            model.eval()
            test_loss = 0.0
            test_stats = defaultdict(lambda: {'intersection': 0, 'pred': 0, 'target': 0})
            with torch.no_grad():
                for images, masks, _ in test_loader:
                    images = images.to(args.device)
                    masks = masks.long().to(args.device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    test_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    accumulate_per_class_stats(preds, masks, args.num_classes, test_stats)
            test_loss /= len(test_loader)
            writer.add_scalar(f"Loss/Test_{global_epoch_label}", test_loss, args.epochs)
            # dice_per_class = compute_dice_from_stats(test_stats)
            dice_per_class, _, iou_scores, mean_iou = compute_metrics_from_stats(test_stats)
            test_dice = np.mean(dice_per_class)
            writer.add_scalar(f"IOU/Test_mean_{global_epoch_label}", mean_iou, args.epochs)
            writer.add_scalar(f"Dice/Test_mean_{global_epoch_label}", test_dice, args.epochs)
            
            if args.eval_partimagenet:
                # for partimagenet, use last class as bg
                dice_per_class_no_bg = dice_per_class[:-1]  # exclude background class
                iou_per_class_no_bg = iou_scores[:-1]
            else:
                dice_per_class_no_bg = dice_per_class[1:]  # exclude background class 0
                iou_per_class_no_bg = iou_scores[1:]
                
            test_dice_no_bg = np.mean(dice_per_class_no_bg)
            writer.add_scalar(f"Dice/Test_mean_no_bg_{global_epoch_label}", test_dice_no_bg, args.epochs)
            for cls_id, cls_dice in enumerate(dice_per_class):
                writer.add_scalar(f"Dice/Test_Class_{cls_id}_{global_epoch_label}", cls_dice, args.epochs)
            logging.info(f"Test ({global_epoch_label}) - Loss: {test_loss:.4f}, Dice (no bg): {test_dice_no_bg:.4f}")
            logging.info(f"\tIOU: {mean_iou:.4f}, IOU (no bg): {np.mean(iou_per_class_no_bg):.4f}")
            
            return test_loss, test_dice_no_bg

        # # Evaluate last model (already loaded)
        # evaluate_model_on_test(model, test_loader, criterion, args, writer, "last")
        
        # # Evaluate best dice model if exists
        # best_dice_model_path = os.path.join(log_dir, 'best_dice_model.pth')
        # if os.path.exists(best_dice_model_path):
        #     logging.info("Loading best validation dice model for test set evaluation...")
        #     model.load_state_dict(torch.load(best_dice_model_path, map_location=args.device))
        #     evaluate_model_on_test(model, test_loader, criterion, args, writer, "best_dice")
        # else:
        #     logging.warning("best_dice_model.pth not found, skipping best dice model evaluation on test set.")


        # # Evaluate best iou model if exists
        # best_iou_model_path = os.path.join(log_dir, 'best_iou_model.pth')
        # if os.path.exists(best_iou_model_path):
        #     logging.info("Loading best validation iou model for test set evaluation...")
        #     model.load_state_dict(torch.load(best_iou_model_path, map_location=args.device))
        #     evaluate_model_on_test(model, test_loader, criterion, args, writer, "best_iou")

        # evaluate last model
        evaluate_model_on_loader(model, test_loader, criterion, args, writer, "Test_last", denorm_stats =None, device=device)

        # evaluate best dice model
        best_dice_model_path = os.path.join(log_dir, "best_dice_model.pth")
        if os.path.exists(best_dice_model_path):
            model.load_state_dict(torch.load(best_dice_model_path, map_location=device))
            evaluate_model_on_loader(model, test_loader, criterion, args, writer, "Test_best_dice", denorm_stats= None, device=device)

        # evaluate best iou model
        best_iou_model_path = os.path.join(log_dir, "best_iou_model.pth")
        if os.path.exists(best_iou_model_path):
            model.load_state_dict(torch.load(best_iou_model_path, map_location=device))
            evaluate_model_on_loader(model, test_loader, criterion, args, writer, "Test_best_iou", denorm_stats= None, device=device)


        
        # 4) Cleanup test loader/dataset too
        del test_loader
        del test_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=False)
    parser.add_argument("--mask_dir", type=str, required=False)
    parser.add_argument("--val_img_dir", type=str, default=None, help="Validation image directory (if different from training)")
    parser.add_argument("--val_mask_dir", type=str, default=None, help="Validation mask directory (if different from training)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation, won't affect if val dirs are provided")
    parser.add_argument("--test_img_dir", type=str, default=None,
                    help="Test image directory (optional). If provided together with --test_mask_dir, will be evaluated at the end.")
    parser.add_argument("--test_mask_dir", type=str, default=None,
                    help="Test mask directory (optional). Must be provided together with --test_img_dir.")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Use a subset of training data for quick experiments (0-1].")

    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        default=[1024, 1024],   # default output size
        metavar=('H', 'W'),
        help="Final output segmentation resolution as (H W)."
    )

    # DINOv3-specific arguments
    parser.add_argument("--model", type=str, default="dinov3_msu", 
                       choices=["dinov3_msu", "dinov3_mlf", "dinov2_msu", "dinov2_mlf"], 
                       help="Model architecture: dinov3_msu, dinov3_mlf, dinov2_msu, dinov2_mlf, or new variants (msu=multi-stage upsampling, mlf=multi-layer fusion)")

    parser.add_argument("--dinov3_weights", type=str, default=None, help="Path to DINOv3 checkpoint weights (required for dinov3_msu and dinov3_mlf models)")
    parser.add_argument("--variant", type=str, default="vitl16", 
                       choices=["vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16"],
                       help="DINOv3 ViT variant")
    parser.add_argument("--take_n", type=int, default=1, help="Number of intermediate layers to take from backbone, MSU uses the last one. MLF uses all.")

    # DINOv2-specific arguments
    parser.add_argument("--dinov2_weights", type=str, default=None, help="Path to DINOv2 checkpoint weights. If not provided, pretrained weights from torch.hub are used.")
    parser.add_argument("--dinov2_variant", type=str, default="vitl14", 
                       choices=["vits14", "vitb14", "vitl14", "vitg14", "vits14_reg", "vitb14_reg", "vitl14_reg", "vitg14_reg"],
                       help="DINOv2 ViT variant")

    parser.add_argument("--full_train", action="store_true", help="Full model training (backbone + decoder)")
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR", choices=["CosineAnnealingLR", "StepLR"], help="Learning rate scheduler")
    parser.add_argument("--weighted_loss", action="store_true", help="Use weighted loss based on class frequencies")
    parser.add_argument("--loss_fn", type=str, default="cross_entropy", choices=["cross_entropy", "DiceCE"], help="Loss function to use")
    parser.add_argument("--enhanced_decoder", action="store_true", help="Use enhanced decoder (GN+SiLU, deeper upsampling), applies to dinov3_msu and dinov2_msu models")
    parser.add_argument("--optim", type=str, choices=["adam", "adamw"], default="adam", help="Optimizer to use: adam or adamw")
    
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mask_suffix", type=str, default="", help="Suffix appended to image stem for mask files, e.g. '' for 'image.png' or '_mask' for 'image_mask.png'")

    parser.add_argument("--aug_config", type=str, default="./models/augs/default.json", help="Path to JSON defining augmentation pipeline")
    parser.add_argument("--batch_norm", action="store_true", help="Use BatchNorm instead of GroupNorm in decoder")
    
    # Dataloader related
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (fp16)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    
    # Learning rate warmup arguments
    parser.add_argument("--lr-warmup", action="store_true", help="Enable learning rate warmup")
    parser.add_argument("--lr-warmup-steps", type=int, default=500, 
                       help="Number of warmup steps (iterations) for learning rate warmup (default: 500)")
    
    parser.add_argument("--eval_partimagenet", action="store_true", help="if true use the last class (40) as bg class for partimagenet")

    
    args = parser.parse_args()

    #### For debugging without command line args
    # args = parser.parse_args([])  # for debug inside script
    # args.img_dir = "./data/cefe_multi/data"
    # args.mask_dir = "./data/cefe_multi/mask"
    # args.val_img_dir = None
    # args.val_mask_dir = None
    # args.val_split = 0.2

    # args.dinov3_weights = "/path/to/dinov3_checkpoint.pth"
    # args.variant = "vitl16"
    # args.scheduler = "CosineAnnealingLR"  # or "StepLR"
    # args.weighted_loss = True  # Use weighted loss
    # args.loss_fn = "cross_entropy"  # or "DiceCE"
    
    # args.num_classes = 10
    # args.batch_size = 2
    # args.epochs = 30
    # args.lr = 1e-4
    
    # args.mask_suffix = "_mask" 
    # args.log_dir = f"./runs/dinov3_train_run"
    # args.device = "cuda"  # or "cuda" if available
    
    # args.aug_config = "./models/augs/default.json"



    # Validate required weights based on model type
    if args.model in ("dinov3_msu", "dinov3_mlf"):
        if args.dinov3_weights is None:
            parser.error(f"--dinov3_weights is required for model '{args.model}'")
    elif args.model in ("dinov2_msu", "dinov2_mlf"):
        # dinov2_weights can be None (will use torch.hub weights), so no error needed
        pass

    main_dinov3(args)
