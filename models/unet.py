# SPDX-License-Identifier: MIT
# models/unet.py
# UNet with optional BatchNorm2d after each Conv2d.
# All comments are in English.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    """A no-op layer to simplify conditional module composition."""
    def forward(self, x):
        return x


class ConvBlock(nn.Module):
    """
    Two 3x3 Conv layers with optional BatchNorm and ReLU activations.
    Structure: Conv -> (BN) -> ReLU -> Conv -> (BN) -> ReLU
    """
    def __init__(self, in_c: int, out_c: int, use_batchnorm: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(out_c) if use_batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(out_c) if use_batchnorm else Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    A classic UNet encoder-decoder with skip connections.
    BatchNorm can be toggled via use_batchnorm (default: True).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 8, use_batchnorm: bool = True):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64, use_batchnorm=use_batchnorm)
        self.enc2 = ConvBlock(64, 128, use_batchnorm=use_batchnorm)
        self.enc3 = ConvBlock(128, 256, use_batchnorm=use_batchnorm)
        self.enc4 = ConvBlock(256, 512, use_batchnorm=use_batchnorm)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(512, 1024, use_batchnorm=use_batchnorm)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512, use_batchnorm=use_batchnorm)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256, use_batchnorm=use_batchnorm)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128, use_batchnorm=use_batchnorm)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64, use_batchnorm=use_batchnorm)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))
        dec3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return self.final(dec1)
