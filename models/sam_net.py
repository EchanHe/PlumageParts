# SPDX-License-Identifier: MIT
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import SAM (original Segment Anything Model)
SAM_AVAILABLE = False
sam_model_registry = None
try:
    from segment_anything import sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    warnings.warn(
        "segment_anything is not installed. The 'sam' backend will not be available. "
        "Install it with: pip install segment-anything"
    )

# Try to import SAM2
SAM2_AVAILABLE = False
try:
    from sam2.build_sam import build_sam2
    SAM2_AVAILABLE = True
except ImportError:
    warnings.warn(
        "sam2 is not installed. The 'sam2' backend will not be available. "
        "Install it from the official SAM2 repository."
    )



# SAM2 config mapping: map SAM1-style model_type to SAM2.1 config names
SAM2_CONFIG_MAP = {
    'vit_b': 'sam2.1_hiera_b+.yaml',
    'vit_l': 'sam2.1_hiera_l.yaml',
    'vit_h': 'sam2.1_hiera_l.yaml',  # SAM2 doesn't have vit_h, use large
    'hiera_t': 'sam2.1_hiera_t.yaml',
    'hiera_s': 'sam2.1_hiera_s.yaml',
    'hiera_b+': 'sam2.1_hiera_b+.yaml',
    'hiera_l': 'sam2.1_hiera_l.yaml',
}


class SAM_UNet(nn.Module):
    """
    A segmentation model using SAM-family encoders with multi-stage upsampling decoder.
    
    Architecture: This model uses a SAM encoder (single-scale feature extraction) followed by
    a multi-stage upsampling decoder. This is more accurately described as Multi-Stage Upsampling (MSU)
    rather than a traditional U-Net (which would have skip connections from encoder to decoder).
    
    For the clearer MSU terminology, use SAM_MSU class instead (they are architecturally identical).
    
    Supports three interchangeable encoder backends:
    - 'sam': Original Segment Anything Model (default)
    - 'sam2': SAM version 2
    - 'sam3': SAM version 3
    
    All backends output 256 channels for compatibility with the decoder.
    
    Args:
        sam_checkpoint (str): Path to the checkpoint file for the encoder.
        model_type (str): Model variant. For SAM, use 'vit_b', 'vit_l', or 'vit_h'.
            For SAM2, use 'hiera_t', 'hiera_s', 'hiera_b+', 'hiera_l' or SAM1-style names.
            For SAM3, this may be a config file path or variant name.
        backend (str): Which SAM-family encoder to use ('sam', 'sam2', 'sam3').
        num_classes (int): Number of output segmentation classes.
        full_train (bool): If True, train the encoder. If False, freeze encoder weights.
        enhanced_decoder (bool): If True, use an enhanced decoder architecture.
    """
    
    SUPPORTED_BACKENDS = ('sam', 'sam2', 'sam3')
    
    def __init__(
        self,
        sam_checkpoint,
        model_type="vit_b",
        backend="sam",
        num_classes=8,
        full_train=False,
        enhanced_decoder: bool = False,
        sam2_config: str = None, 
        output_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.full_train = full_train
        self.num_classes = num_classes
        self.backend = backend
        self.model_type = model_type
        self.sam2_config = sam2_config
        self.output_size = output_size
        
        # Validate backend argument
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Supported backends are: {self.SUPPORTED_BACKENDS}"
            )
        
        # Initialize the image encoder based on selected backend
        self.image_encoder = self._load_encoder(sam_checkpoint, model_type, backend, sam2_config)
        
        # Freeze encoder if not doing full training
        if not full_train:
            for param in self.image_encoder.parameters():
                param.requires_grad = False  # freeze encoder
        
        if enhanced_decoder:
            # Subtle enhanced decoder: GN+SiLU and split final 4x upsample into two 2x steps
            self.decoder = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), nn.GroupNorm(32, 256), nn.SiLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64→128
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False), nn.GroupNorm(32, 128), nn.SiLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128→256
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False), nn.GroupNorm(32, 64), nn.SiLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 256→512
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.GroupNorm(32, 64), nn.SiLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 512→1024
                nn.Conv2d(64, num_classes, kernel_size=1)
            )
        else:
            # Original lightweight decoder
            self.decoder = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64→128
                nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128→256
                nn.Conv2d(64, num_classes, 3, padding=1),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)   # 256→1024
            )

    def _load_encoder(self, checkpoint_path, model_type, backend, sam2_config=None):
        """
        Load the image encoder from the specified backend.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            model_type (str): Model variant (e.g., 'vit_b', 'vit_l', 'vit_h').
            backend (str): Which backend to use ('sam', 'sam2', 'sam3').
            sam2_config (str, optional): Path to SAM2 config yaml file. If provided, overrides model_type mapping.
            
        Returns:
            nn.Module: The image encoder module.
            
        Raises:
            RuntimeError: If the requested backend is not available or checkpoint loading fails.
        """
        if backend == "sam":
            if not SAM_AVAILABLE:
                raise RuntimeError(
                    "The 'sam' backend was requested but segment_anything is not installed. "
                    "Install it with: pip install segment-anything"
                )
            try:
                sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
                return sam_model.image_encoder
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Failed to load SAM checkpoint from '{checkpoint_path}': File not found. "
                    f"Please ensure the checkpoint file exists at the specified path."
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SAM checkpoint from '{checkpoint_path}': {e}"
                ) from e
        
        elif backend == "sam2":
            if not SAM2_AVAILABLE:
                raise RuntimeError(
                    "The 'sam2' backend was requested but sam2 is not installed. "
                    "Install it from the official SAM2 repository."
                )
            try:
                # SAM2 uses build_sam2 function with config file names.
                # Use provided sam2_config if available, otherwise map from model_type
                import os
                if sam2_config is not None:
                    # Extract just the filename if a full path was provided
                    config_to_use = os.path.basename(sam2_config)
                elif model_type.endswith('.yaml'):
                    config_to_use = os.path.basename(model_type)
                else:
                    config_to_use = SAM2_CONFIG_MAP.get(model_type, model_type)
                sam2_model = build_sam2(config_to_use, checkpoint_path)
                return sam2_model.image_encoder
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Failed to load SAM2 checkpoint from '{checkpoint_path}': File not found. "
                    f"Please ensure the checkpoint file exists at the specified path."
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SAM2 checkpoint from '{checkpoint_path}': {e}"
                ) from e
        
        elif backend == "sam3":
            # Try to import SAM3
            SAM3_AVAILABLE = False
            try:
                from sam3 import build_sam3_image_model
                SAM3_AVAILABLE = True
            except ImportError:
                warnings.warn(
                    "sam3 is not installed. The 'sam3' backend will not be available. "
                    "Install it from the official SAM3 repository."
                )

            if not SAM3_AVAILABLE:
                raise RuntimeError(
                    "The 'sam3' backend was requested but sam3 is not installed. "
                    "Install it from the official SAM3 repository."
                )
            try:
                # SAM3 uses build_sam3 function similar to SAM2.
                # model_type can be a config file path or model variant depending on SAM3 version.
                # The exact API may vary depending on the SAM3 implementation.
                # sam3_model = build_sam3(model_type, checkpoint_path)
                sam3_model = build_sam3_image_model()
                return sam3_model.backbone.vision_backbone
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Failed to load SAM3 checkpoint from '{checkpoint_path}': File not found. "
                    f"Please ensure the checkpoint file exists at the specified path."
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SAM3 checkpoint from '{checkpoint_path}': {e}"
                ) from e
        
        # This should never be reached due to validation in __init__
        raise ValueError(f"Unknown backend: {backend}")

    def forward(self, x):
        """
        Forward pass through the encoder and decoder.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            
        Returns:
            torch.Tensor: Segmentation masks of shape (B, num_classes, H, W).
        """
        if not self.full_train:
            with torch.no_grad():
                features = self.image_encoder(x)
        else:
            features = self.image_encoder(x)

        if isinstance(features, (list, tuple)):
            # this output is from SAM3 which return a tuple/list
            # each element is a feature map from different stages
            features_list = features[0]  # choose the first element
            if not isinstance(features_list, list):
                raise TypeError(f"Expected list at enc_out[0], got {type(features_list)}")
    
            features = features_list[0]

        # sam1 or sam2, stays the same


        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Expected Tensor as encoder feature, got {type(features)}")

        masks = self.decoder(features)
        if self.output_size is not None:
            masks = F.interpolate(masks, size=self.output_size, mode='bilinear', align_corners=False)
        return masks


class SAM_MSU(SAM_UNet):
    """
    Multi-Stage Upsampling (MSU) segmentation model using SAM-family encoders.
    
    This class provides a clearer architectural name for the SAM-based segmentation model.
    It uses a SAM encoder (single-scale feature extraction at 64x64 resolution) followed by
    a multi-stage upsampling decoder that progressively increases spatial resolution.
    
    Unlike traditional U-Net architectures with skip connections, this is a single-scale
    encoder with multi-stage upsampling, hence the MSU (Multi-Stage Upsampling) designation.
    
    This class is architecturally identical to SAM_UNet and is provided for clearer naming.
    All parameters and functionality are inherited from SAM_UNet.
    
    Supports three interchangeable encoder backends:
    - 'sam': Original Segment Anything Model (default)
    - 'sam2': SAM version 2
    - 'sam3': SAM version 3
    
    Args:
        sam_checkpoint (str): Path to the checkpoint file for the encoder.
        model_type (str): Model variant. For SAM, use 'vit_b', 'vit_l', or 'vit_h'.
            For SAM2, use 'hiera_t', 'hiera_s', 'hiera_b+', 'hiera_l' or SAM1-style names.
            For SAM3, this may be a config file path or variant name.
        backend (str): Which SAM-family encoder to use ('sam', 'sam2', 'sam3').
        num_classes (int): Number of output segmentation classes.
        full_train (bool): If True, train the encoder. If False, freeze encoder weights.
        enhanced_decoder (bool): If True, use an enhanced decoder architecture.
    """
    # This is an alias class - all implementation is inherited from SAM_UNet
    pass


__all__ = ["SAM_UNet", "SAM_MSU"]
