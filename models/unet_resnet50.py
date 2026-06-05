# SPDX-License-Identifier: MIT
# UNet with a pretrained ResNet-50 encoder (ImageNet-based) using torchvision.
# - Encoder: torchvision.models.resnet50(pretrained=True) feature stages
# - Decoder: classic UNet-style upsampling with skip connections
# - All comments and logs are in English.
#
# Notes:
# - Input normalization: recommended to use ImageNet mean/std for the encoder:
#   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
# - Feature stages used (spatial downscale relative to input):
#   enc1: after conv1+bn1+relu   (down x2, channels=64)
#   enc2: layer1 (ResNet stage 2) (down x4, channels=256)
#   enc3: layer2 (ResNet stage 3) (down x8, channels=512)
#   enc4: layer3 (ResNet stage 4) (down x16, channels=1024)
#   bottleneck: layer4 (ResNet stage 5) (down x32, channels=2048)
#
# Decoder channel plan:
#   up4: 2048 -> 1024, concat with enc4 (1024) => 2048 -> dec4 -> 1024
#   up3: 1024 -> 512,  concat with enc3 (512)  => 1024 -> dec3 -> 512
#   up2: 512  -> 256,  concat with enc2 (256)  => 512  -> dec2 -> 256
#   up1: 256  -> 64,   concat with enc1 (64)   => 128  -> dec1 -> 64
#   final: 64 -> num_classes
#
# Because the encoder downsamples to /32 and the decoder upsamples 4 times
# back to /2, we upsample the final logits back to the input spatial size
# to match the target mask resolution for loss computation.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Identity(nn.Module):
    def forward(self, x):
        return x


class ConvBNReLU(nn.Module):
    """Conv -> BN -> ReLU"""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """Two ConvBNReLU in sequence"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_c, out_c, k=3, s=1, p=1),
            ConvBNReLU(out_c, out_c, k=3, s=1, p=1),
        )

    def forward(self, x):
        return self.block(x)


class ResNet50Encoder(nn.Module):
    """
    ResNet-50 encoder returning multi-scale features for UNet-style decoder.
    Outputs:
      enc1: after conv1+bn1+relu (stride 2), channels=64
      enc2: after layer1, channels=256
      enc3: after layer2, channels=512
      enc4: after layer3, channels=1024
      bottleneck: after layer4, channels=2048
    """
    def __init__(self, pretrained=True, in_channels=3):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        if in_channels != 3:
            old_conv1 = resnet.conv1
            resnet.conv1 = nn.Conv2d(in_channels, old_conv1.out_channels,
                                     kernel_size=old_conv1.kernel_size,
                                     stride=old_conv1.stride,
                                     padding=old_conv1.padding,
                                     bias=old_conv1.bias)
            nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256 ch
        self.layer2 = resnet.layer2  # 512 ch
        self.layer3 = resnet.layer3  # 1024 ch
        self.layer4 = resnet.layer4  # 2048 ch

    def forward(self, x):
        # Stage 1
        x = self.conv1(x)     # [B,64,H/2,W/2]
        x = self.bn1(x)
        enc1 = self.relu(x)   # 64 channels, /2

        x = self.maxpool(enc1)    # [B,64,H/4,W/4]
        # Stage 2
        enc2 = self.layer1(x)     # [B,256,H/4,W/4]
        # Stage 3
        enc3 = self.layer2(enc2)  # [B,512,H/8,W/8]
        # Stage 4
        enc4 = self.layer3(enc3)  # [B,1024,H/16,W/16]
        # Bottleneck
        bottleneck = self.layer4(enc4)  # [B,2048,H/32,W/32]
        return enc1, enc2, enc3, enc4, bottleneck


class UNetResNet50(nn.Module):
    """
    UNet with ResNet-50 encoder (ImageNet pretrained) and classic decoder.
    """
    def __init__(self, num_classes=8, in_channels=3, pretrained=True, freeze_encoder=False):
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained, in_channels=in_channels)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Decoder
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024 + 1024, 1024)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512 + 512, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256 + 256, 256)

        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        enc1, enc2, enc3, enc4, bottleneck = self.encoder(x)

        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([d4, enc4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, enc3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, enc2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, enc1], dim=1))

        logits = self.final(d1)
        # Upsample logits back to the input spatial size to match target mask size
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits
