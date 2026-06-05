# SPDX-License-Identifier: MIT
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np

from models.util import visualize_segmentation


class DINOv3TokenEncoder(nn.Module):
    """
    Wrap an official DINOv3 ViT backbone (from facebookresearch/dinov3) to output a spatial
    feature map (B, C, H', W') using the model's get_intermediate_layers API.

    This avoids having to manually handle PE/RoPE or token reshaping.
    """

    def __init__(self, dino_vit: nn.Module, take_n: int = 1):
        super().__init__()
        self.vit = dino_vit
        self.take_n = take_n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get_intermediate_layers(..., reshape=True) returns (B, C, H', W') per selected layer
        feats_tuple = self.vit.get_intermediate_layers(
            x, n=self.take_n, reshape=True, return_class_token=False, return_extra_tokens=False, norm=True
        )
        # If n>1, take the last
        feat = feats_tuple[-1]
        return feat  # (B, C, H', W')


class DINOv3_MSU(nn.Module):
    """
    Multi-Stage Upsampling (MSU) segmentation head on top of a DINOv3 ViT backbone.
    
    This architecture uses a sequence of upsampling stages with convolutions to progressively
    increase spatial resolution from the ViT feature map to the final segmentation mask.
    Unlike traditional U-Net, this does not use encoder-decoder skip connections.

    Args:
        dino_backbone: DINOv3 ViT model loaded via torch.hub from facebookresearch/dinov3.
        num_classes: number of output classes.
        freeze_encoder: if True, encoder params are not updated.
        enhanced_decoder: if True, use a deeper GN+SiLU decoder; otherwise a lightweight decoder is used.
        output_size: if set, final masks are interpolated to (H, W).
    """

    def __init__(
        self,
        dino_backbone: nn.Module,
        num_classes: int = 10,
        freeze_encoder: bool = True,
        enhanced_decoder: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
        take_n: int = 1,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = DINOv3TokenEncoder(dino_backbone, take_n=take_n)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Lazy 1x1 projection to 256 channels
        self.proj: Optional[nn.Conv2d] = None

        if enhanced_decoder:

            # Helper to create normalization layer based on use_batch_norm flag
            def norm_layer(num_channels):
                if use_batch_norm:
                    return nn.BatchNorm2d(num_channels)
                else:
                    return nn. GroupNorm(32, num_channels)

            self.decoder = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), norm_layer(256), nn.SiLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False), norm_layer(128), nn.SiLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False), norm_layer(64), nn.SiLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), norm_layer(64), nn.SiLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, num_classes, kernel_size=1)
            )            
        else:
            # Lightweight decoder
            self.decoder = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, num_classes, 3, padding=1),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            )

        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode (handle frozen encoder under no_grad)
        if any(p.requires_grad for p in self.encoder.parameters()):
            feat = self.encoder(x)
        else:
            with torch.no_grad():
                feat = self.encoder(x)

        # Initialize projection lazily
        if self.proj is None:
            in_ch = feat.shape[1]
            self.proj = nn.Conv2d(in_ch, 256, kernel_size=1)
            self.proj.to(feat.device)

        feat = self.proj(feat)
        masks = self.decoder(feat)

        if self.output_size is not None:
            masks = F.interpolate(masks, size=self.output_size, mode='bilinear', align_corners=False)
        return masks


# ------------------------
# Backbone builders / utils
# ------------------------

def build_dinov3_backbone(variant: str = "vitl16", weights: str = None):
    """
    Load a DINOv3 ViT backbone via torch.hub and apply locally saved weights.

    Args:
        variant: one of {vits16, vits16plus, vitb16, vitl16, vith16plus, vit7b16}
        weights: path to a local checkpoint (.pth/.pt) containing the backbone state_dict.
    """
    valid = {"vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16"}
    if variant not in valid:
        raise ValueError(f"Unknown DINOv3 variant '{variant}'. Expected one of {sorted(valid)}.")
    if weights is None or not os.path.isfile(weights):
        raise FileNotFoundError(f"Local weights file not found: {weights}")

    entry = f"dinov3_{variant}"
    try:
        # Instantiate the backbone architecture without downloading pretrained weights
        vit = torch.hub.load("facebookresearch/dinov3", entry, pretrained=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to construct DINOv3 backbone via torch.hub (entry={entry}). "
            f"Ensure torch.hub can access 'facebookresearch/dinov3' or it's cached locally.\n{e}"
        )

    # Load local checkpoint
    try:
        sd = torch.load(weights, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load weights from '{weights}': {e}")

    # Unwrap common checkpoint formats
    if isinstance(sd, dict):
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        elif "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
    if isinstance(sd, dict):
        # Strip common prefixes
        sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
        if sd and all(k.startswith("backbone.") for k in sd.keys()):
            sd = { k[len("backbone."):]: v for k, v in sd.items() }

    missing, unexpected = vit.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[build_dinov3_backbone] Loaded local weights with mismatches: missing={len(missing)}, unexpected={len(unexpected)}")
    return vit


# ------------------------
# Quick-test CLI (mirrors vit_unet.py)
# ------------------------

def main():
    parser = argparse.ArgumentParser(description="Quick-test a DINOv3 ViT backbone + Multi-Stage Upsampling decoder: run a forward pass and optional visualization.")
    parser.add_argument("--variant", type=str, default="vitl16", choices=["vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16"], help="DINOv3 ViT variant")
    parser.add_argument("--weights", type=str, default=None, help="Path to a local DINOv3 checkpoint (backbone weights)")
    parser.add_argument("--image", type=str, default=None, help="Optional path to an RGB image for inference test")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for mask prediction")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--enhanced_decoder", action="store_true", help="Use enhanced GN+SiLU decoder")
    parser.add_argument("--output_mask", type=str, default=None, help="Optional path to save predicted mask PNG")
    parser.add_argument("--output_overlay", type=str, default=None, help="Optional path to save an overlay PNG (mask on image)")
    parser.add_argument("--overlay_alpha", type=float, default=0.5, help="Alpha transparency for mask overlay [0-1]")
    args = parser.parse_args()

    if args.weights is None:
        raise ValueError("Please provide --weights with the path to your local DINOv3 checkpoint.")

    print(f"Loading DINOv3 backbone: {args.variant}")
    vit = build_dinov3_backbone(
        variant=args.variant,
        weights=args.weights,
    )

    model = DINOv3_MSU(
        dino_backbone=vit,
        num_classes=args.num_classes,
        freeze_encoder=True,
        enhanced_decoder=args.enhanced_decoder,
    ).eval().to(args.device)

    # Input: image or dummy
    if args.image is not None:
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")

        img = Image.open(args.image).convert("RGB")
        orig_w, orig_h = img.size
        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        x = tfm(img)

        # Pad to multiples of the patch size to avoid ragged tokens
        patch_size = getattr(vit, "patch_size", 16)
        pad_w = (patch_size - (orig_w % patch_size)) % patch_size
        pad_h = (patch_size - (orig_h % patch_size)) % patch_size
        if pad_w != 0 or pad_h != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        x = x.unsqueeze(0).to(args.device)
        model.output_size = (orig_h, orig_w)
    else:
        x = torch.randn(1, 3, 224, 224, device=args.device)
        model.output_size = (224, 224)

    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)

    print(f"Input: {tuple(x.shape)} | Logits: {tuple(logits.shape)} | Preds: {tuple(preds.shape)}")
    uniq = torch.unique(preds).tolist()
    print(f"Unique predicted classes: {uniq}")

    if args.output_mask is not None:
        os.makedirs(os.path.dirname(args.output_mask) or ".", exist_ok=True)
        mask_np = preds[0].detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(mask_np).save(args.output_mask)
        print(f"Saved mask to: {args.output_mask}")

    if args.output_overlay is not None:
        if args.image is None:
            print("Warning: --output_overlay specified but no --image provided; skipping.")
        else:
            os.makedirs(os.path.dirname(args.output_overlay) or ".", exist_ok=True)
            img_np = np.array(img)
            mask_np_int = preds[0].detach().cpu().numpy().astype(np.int32)
            visualize_segmentation(img_np, mask_np_int, alpha=args.overlay_alpha, cmap='tab20', save_path=args.output_overlay)
            print(f"Saved overlay to: {args.output_overlay}")


if __name__ == "__main__":
    main()


__all__ = ["DINOv3TokenEncoder", "DINOv3_MSU", "build_dinov3_backbone"]
