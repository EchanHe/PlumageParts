# SPDX-License-Identifier: MIT
import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_multistage_upsampling import build_dinov3_backbone


class DINOv3MultiLayerEncoder(nn.Module):
    """
    Wrap DINOv3 ViT backbone and return multiple intermediate layers as spatial maps.
    We use the official get_intermediate_layers API with n=take_n and reshape=True.

    Output:
        list of feature maps [f_1, ..., f_n], each shape (B, C, H', W').
        These are the last `take_n` transformer blocks (all same spatial size for ViT).
    """
    def __init__(self, dino_vit: nn.Module, take_n: int = 4):
        super().__init__()
        self.vit = dino_vit
        self.take_n = take_n

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats_tuple = self.vit.get_intermediate_layers(
            x,
            n=self.take_n,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        # feats_tuple is a tuple of tensors (B, C, H', W') from the last n layers
        return list(feats_tuple)


class MultiLayerFusionDecoder(nn.Module):
    """
    A multi-layer feature fusion decoder that combines multiple DINOv3 feature maps.
    
    This decoder performs top-down fusion of features from different transformer layers.
    Unlike traditional FPN with multi-resolution pyramids, this fuses multiple layers
    at the same spatial resolution for semantic multi-layer feature fusion.

    Assumes input is a list of feature maps [f_1, ..., f_n] with the same spatial size,
    each already projected to the same channel dimension (e.g. 256).
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 10, num_levels: int = 4):
        super().__init__()

        # We will do a top-down fusion: start from the last feature and go backwards.
        # Since ViT features are same spatial size, this is more like "multi-level semantic fusion"
        # than classical multi-resolution FPN, but empirically it works well.

        self.conv_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, in_channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(num_levels-1)   
        ])

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        feats: list of Tensors [f_1, ..., f_n], each (B, C, H, W)
        We fuse from deepest to shallowest:
            x = f_n
            for i = n-1 ... 0:
                x = x + f_i
                x = conv_fuse[i](x)
        """
        n = len(feats)
        assert n - 1 == len(self.conv_fuse), "num_levels mismatch"
        
        if n == 0:
            raise ValueError("MultiLayerFusionDecoder expects a non-empty list of features.")

        # Start from deepest feature
        x = feats[-1]
        # Fuse back to shallower features
        for i in reversed(range(n - 1)):
            # all layers have same spatial size, so no need to upsample
            # but for generality, we keep the interpolate step
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = x + feats[i]
            x = self.conv_fuse[i](x)

        out = self.head(x)
        return out


class DINOv3_MLF(nn.Module):
    """
    Multi-Layer Fusion (MLF) segmentation model with DINOv3 backbone.
    
    This architecture extracts features from multiple transformer layers and fuses them
    in a top-down manner to create rich semantic representations for segmentation.
    The fusion is performed at a single spatial resolution (unlike traditional FPN
    which operates on multi-resolution pyramids).

    Args:
        dino_backbone: DINOv3 ViT model instance.
        num_classes: number of segmentation classes.
        freeze_encoder: whether to freeze backbone weights.
        take_n: number of last DINOv3 layers to use for fusion (e.g. 4).
        output_size: if set, masks are interpolated to (H, W).
    """

    def __init__(
        self,
        dino_backbone: nn.Module,
        num_classes: int = 10,
        freeze_encoder: bool = True,
        take_n: int = 4,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()

        self.encoder = DINOv3MultiLayerEncoder(dino_backbone, take_n=take_n)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.take_n = take_n
        self.output_size = output_size

        # We lazily build projection layers once we see the first batch,
        # because we need to know in_channels from DINOv3 output.
        self.proj_layers: Optional[nn.ModuleList] = None

        # Multi-layer fusion decoder (works on projected features with fixed channel dim)
        self.decoder = MultiLayerFusionDecoder(in_channels=256, num_classes=num_classes, num_levels=take_n)

    def _init_proj_layers(self, feats: List[torch.Tensor]):
        """
        Initialize 1x1 projection layers for each encoder feature map.
        This is called on the first forward pass.
        """
        in_ch = feats[0].shape[1]
        proj_layers = []
        for _ in range(len(feats)):
            proj_layers.append(nn.Conv2d(in_ch, 256, kernel_size=1))
        self.proj_layers = nn.ModuleList(proj_layers)
        self.proj_layers.to(feats[0].device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # no grad to wrap encoder if frozen
        if any(p.requires_grad for p in self.encoder.parameters()):
            feats = self.encoder(x)
        else:
            with torch.no_grad():
                feats = self.encoder(x)

        if self.proj_layers is None:
            self._init_proj_layers(feats)

        proj_feats = [proj_layer(f) for proj_layer, f in zip(self.proj_layers, feats)]

        masks = self.decoder(proj_feats)

        if self.output_size is not None:
            masks = F.interpolate(masks, size=self.output_size, mode="bilinear", align_corners=False)

        return masks


def build_dinov3_backbone_mlf(variant: str, weights: str) -> nn.Module:
    """
    Wrapper around build_dinov3_backbone from dinov3_unet.py,
    for use with the Multi-Layer Fusion model.
    """
    return build_dinov3_backbone(variant=variant, weights=weights)


__all__ = ["DINOv3_MLF", "build_dinov3_backbone_mlf", "DINOv3MultiLayerEncoder"]
