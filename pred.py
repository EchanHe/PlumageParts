# SPDX-License-Identifier: MIT
import os
import argparse
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.dataset import (
    PredictEvalDataset,
    restore_pred_to_orig_replay_lmspad,
    restore_probs_to_orig_replay_lmspad,
    save_probs_tiff_int,
    compute_entropy_map,
)
from models.util import (
    compute_dice,
    visualize_segmentation,
)

from models.dinov3_multistage_upsampling import DINOv3_MSU,build_dinov3_backbone
from models.dinov3_multilayer_fusion import DINOv3_MLF, build_dinov3_backbone_mlf

from PIL import Image
import json


def collate_keep_none(batch):
    """
    Custom collate function that keeps None masks.
    batch: list of tuples (image_t, mask_t, image_np, mask_np, meta)
    """
    images_t = torch.stack([b[0] for b in batch], dim=0)  # [B,3,H,W]

    valid_masks_t = [b[1] for b in batch if b[1] is not None]
    if valid_masks_t:
        masks_t = torch.stack(valid_masks_t, dim=0)
    else:
        masks_t = None

    images_np = [b[2] for b in batch]   # list of np arrays
    masks_np = [b[3] for b in batch]    # list, can contain None
    metas = [b[4] for b in batch]       # list of dict
    return images_t, masks_t, images_np, masks_np, metas


def build_dinov3_model(args, device):
    """
    Build DINOv3 segmentation model for inference.
    
    Supports both  (dinov3_msu, dinov3_mlf) that better describe the architecture:
    - dinov3_msu: Single-scale encoder + Multi-Stage Upsampling decoder
    - dinov3_mlf: Multi-layer feature fusion decoder
    
    Legacy names are maintained for backward compatibility with existing checkpoints.
    """
    assert args.output_size is not None and len(args.output_size) == 2, \
        "Please provide --output_size H W"

    out_h, out_w = args.output_size

    # Define model configurations: (new_name, legacy_name, model_class, backbone_builder, is_msu_type)
    model_configs = {
        'dinov3_msu': ('DINOv3_MSU', DINOv3_MSU, build_dinov3_backbone, True),
        'dinov3_mlf': ('DINOv3_MLF', DINOv3_MLF, build_dinov3_backbone_mlf, False),
    }
    
    if args.model not in model_configs:
        raise ValueError(f"Unsupported model type for this script: {args.model}")
    
    model_name, model_class, backbone_builder, is_msu_type = model_configs[args.model]
    
    print(f"Building {model_name} with backbone variant: {args.variant}")
    dino_backbone = backbone_builder(
        variant=args.variant,
        weights=args.dinov3_weights,
    )
    
    # Build model with appropriate parameters based on type
    if is_msu_type:
        model = model_class(
            dino_backbone=dino_backbone,
            num_classes=args.num_classes,
            freeze_encoder=True,  # encoder is frozen for inference
            enhanced_decoder=args.enhanced_decoder,
            take_n=args.take_n,
            output_size=(out_h, out_w),
            use_batch_norm=args.batch_norm,
        ).to(device)
    else:
        model = model_class(
            dino_backbone=dino_backbone,
            num_classes=args.num_classes,
            freeze_encoder=True,
            take_n=args.take_n,
            output_size=(out_h, out_w),
        ).to(device)

    # Initialize lazy projection with a dummy forward (same idea as in train_dinov3_net.py)
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, out_h, out_w, device=device))

    # Load checkpoint
    print(f"Loading model weights from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    # In training we saved with torch.save(model.state_dict()), so direct load is expected.
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_and_save(args):
    device = args.device

    print("Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # Decide resize / input size for PredictEvalDataset
    # For DINOv3 models we use args.output_size

    assert args.output_size is not None and len(args.output_size) == 2, \
        "For DINOv3 models, please provide --output_size H W"
    resize = max(args.output_size)



    file_list = sorted(os.listdir(args.img_dir))
    file_list = [
        f for f in file_list
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]

    dataset = PredictEvalDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        file_list=file_list,
        resize=resize,
        pad_if_needed=True,
        normalize=True,
        mask_suffix=args.mask_suffix,
    )
    print(f"Prediction dataset size: {len(dataset)}")

    if args.shuffle:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_keep_none,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_keep_none,
        )

    # Create output sub-directories
    os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)
    if args.save_probs:
        os.makedirs(os.path.join(args.output_dir, "probs"), exist_ok=True)
    if args.save_overlay:
        os.makedirs(os.path.join(args.output_dir, "overlay"), exist_ok=True)
    if args.save_entropy:
        os.makedirs(os.path.join(args.output_dir, "entropy"), exist_ok=True)

    # Build model and load weights
    model = build_dinov3_model(args, device)
    model.eval()

    # Prepare optional label map for visualization
    if args.label_map is not None:
        if os.path.isfile(args.label_map):
            with open(args.label_map, "r") as f:
                label_map = json.load(f)
            # Convert keys to int
            label_map = {int(k): v for k, v in label_map.items()}
        else:
            raise FileNotFoundError(f"Label map file not found: {args.label_map}")
    else:
        label_map = None

    dice_records = []

    for image_t, mask_t, image_np, mask_np, meta in tqdm(loader):
        image_t = image_t.to(device)

        with torch.no_grad():
            output = model(image_t)  # [B, C, H, W] logits (resized)

        # Restore per-class probabilities to original image size
        probs_np = restore_probs_to_orig_replay_lmspad(output, meta[0])  # [B, C, Horig, Worig]
        probs_np_1 = probs_np[0]  # [C, Horig, Worig]

        # Optionally save probability volume as 16-bit TIFF
        if args.save_probs:
            tif_path = os.path.join(
                args.output_dir,
                "probs",
                f"{meta[0]['fname']}_probs.tif",
            )
            save_probs_tiff_int(probs_np_1, tif_path)

        # Optionally save entropy map
        if args.save_entropy:
            ent = compute_entropy_map(probs_np_1)  # [H,W] in [0,1]
            ent_png = (ent * 255).astype(np.uint8)
            entropy_path = os.path.join(
                args.output_dir,
                "entropy",
                f"{meta[0]['fname']}_entropy.png",
            )
            Image.fromarray(ent_png, mode="L").save(entropy_path)

        # Restore predicted labels (argmax over channels) to original size
        preds_np = restore_pred_to_orig_replay_lmspad(output, meta[0], is_logits=True)
        preds_np_1 = preds_np[0]  # [Horig, Worig], int labels

        # Save raw prediction mask
        mask_save_path = os.path.join(
            args.output_dir,
            "masks",
            f"{meta[0]['fname']}.png",
        )
        cv2.imwrite(mask_save_path, preds_np_1.astype(np.uint8))

        # Save overlay visualization
        if args.save_overlay:
            overlay_path = os.path.join(
                args.output_dir,
                "overlay",
                f"{meta[0]['fname']}_overlay.png",
            )
            visualize_segmentation(
                image_np[0],  # original image (H,W,3)
                preds_np_1,
                label_map=label_map,
                alpha=args.alpha,
                save_path=overlay_path,
            )

            # If ground-truth masks are provided, also visualize GT overlay
            if args.mask_dir and mask_t is not None:
                gt_mask_np = restore_pred_to_orig_replay_lmspad(
                    mask_t, meta[0], is_logits=False
                )[0]
                gt_overlay_path = os.path.join(
                    args.output_dir,
                    "overlay",
                    f"{meta[0]['fname']}_gt_overlay.png",
                )
                visualize_segmentation(
                    image_np[0],
                    gt_mask_np,
                    label_map=label_map,
                    alpha=args.alpha,
                    save_path=gt_overlay_path,
                )

        # Compute Dice scores if ground-truth masks are available
        if args.mask_dir and mask_np[0] is not None:
            dice_scores = compute_dice(
                preds_np_1,
                mask_np[0],
                args.num_classes,
                per_class=True,
            )
            record = {"filename": meta[0]["fname"]}
            record.update(
                {f"dice_class_{i}": dice_scores[i] for i in range(args.num_classes)}
            )
            dice_records.append(record)

    # Save per-image dice and summary
    if args.mask_dir and len(dice_records) > 0:
        df = pd.DataFrame(dice_records)

        if not 0 <= args.background_class < args.num_classes:
            raise ValueError(
                f"--background_class must be in [0, {args.num_classes - 1}], "
                f"got {args.background_class}"
            )

        # Per-class mean, excluding the configured background class.
        class_cols = [
            f"dice_class_{i}"
            for i in range(args.num_classes)
            if i != args.background_class
        ]
        mean_per_class = df[class_cols].mean().to_dict()
        overall_mean = np.mean(list(mean_per_class.values()))

        print(f"\n=== Dice Scores Summary (excluding background class {args.background_class}) ===")
        for cls, val in mean_per_class.items():
            print(f"{cls}: {val:.4f}")
        print(f"Overall mean (excluding background class {args.background_class}): {overall_mean:.4f}")

        # Per-image dice scores
        out_csv = os.path.join(args.output_dir, "dice_scores.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved per-image dice scores to {out_csv}")

        # Simple summary CSV (mean per class + overall)
        summary_path = os.path.join(args.output_dir, "dice_summary.csv")
        summary_df = pd.DataFrame(
            {
                "metric": list(mean_per_class.keys()) + [f"overall_mean_no_bg_class_{args.background_class}"],
                "value": list(mean_per_class.values()) + [overall_mean],
            }
        )
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved dice summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="DINOv3 inference script")

    parser.add_argument("--img_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pth")
    parser.add_argument("--output_dir", type=str, default="predictions_dinov3", help="Output directory")
    parser.add_argument(
        "--model",
        type=str,
        choices=["dinov3_msu", "dinov3_mlf"],
        required=True,
        help=(
            "Model architecture to use for inference. "
            "'dinov3_msu' (Multi-Stage Upsampling), "
            "'dinov3_mlf' (Multi-Layer Fusion). "
        ),
    )
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mask_dir", type=str, default=None, help="Optional GT mask directory")
    parser.add_argument("--mask_suffix", type=str, default="", help="Suffix appended to image stem for mask files, e.g. '' for 'image.png' or '_mask' for 'image_mask.png'")
    parser.add_argument("--background_class", type=int, default=0, help="Class ID to exclude from no-background Dice summaries")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay transparency")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle input images during inference")
    parser.add_argument("--save_overlay", action="store_true", help="Save overlay visualizations")
    parser.add_argument("--save_probs", action="store_true", help="Save per-class probability volume as 16-bit TIFF")
    parser.add_argument("--save_entropy", action="store_true", help="Save pixel-wise entropy map")
    parser.add_argument(
        "--label_map",
        type=str,
        default=None,
        help='Path to JSON mapping class indices to names, e.g. {"0": "background", "1": "bird"}',
    )


    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        metavar=("H", "W"),
        help="Final output segmentation resolution as (H W)",
    )
    parser.add_argument(
        "--dinov3_weights",
        type=str,
        required=True,
        help="Path to DINOv3 backbone weights",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="vitl16",
        choices=["vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16"],
        help="DINOv3 ViT variant",
    )
    parser.add_argument(
        "--take_n",
        type=int,
        default=1,
        help="Number of intermediate layers to take from backbone",
    )
    parser.add_argument(
        "--enhanced_decoder",
        action="store_true",
        help="Use enhanced decoder (only applies to dinov3_msu)",
    )
    parser.add_argument(
        "--batch_norm",
        action="store_true",
        help="Use BatchNorm in decoder (only applies to dinov3_msu)",
    )

    args = parser.parse_args()

    # Basic validation
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not os.path.isfile(args.dinov3_weights):
        raise FileNotFoundError(f"DINOv3 backbone weights not found: {args.dinov3_weights}")
    if not os.path.isdir(args.img_dir):
        raise NotADirectoryError(f"Image directory not found: {args.img_dir}")
    if args.mask_dir is not None and not os.path.isdir(args.mask_dir):
        raise NotADirectoryError(f"Mask directory not found: {args.mask_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    predict_and_save(args)


if __name__ == "__main__":
    main()
