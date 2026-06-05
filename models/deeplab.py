# SPDX-License-Identifier: MIT
# models/deeplab.py

import torch.nn as nn
import torchvision.models.segmentation as models_seg
import torchvision.models as models

class DeepLabWrapper(nn.Module):
    def __init__(self, num_classes=8, pretrained_backbone=False):
        super().__init__()
        if pretrained_backbone:
            self.model = models_seg.deeplabv3_resnet50(weights_backbone=models.ResNet50_Weights.IMAGENET1K_V2, num_classes=num_classes)
        else:
            self.model = models_seg.deeplabv3_resnet50(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)["out"]
