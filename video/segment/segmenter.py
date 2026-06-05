# SPDX-License-Identifier: MIT
"""
Unified bird part segmentation interface.

This module provides a common API around several segmentation backbones,
including classic models (UNet / DeepLab), SAM-based models, and DINOv3
models with MSU / MLF decoders, consistent with pred.py.
"""

import logging
from typing import Dict, Optional, Tuple, Callable
import numpy as np
import torch
import cv2
from pathlib import Path
import sys
import json
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup logging
logger = logging.getLogger(__name__)


def segment_parts(
    image: np.ndarray,
    bbox: list,
    prior_mask: Optional[np.ndarray] = None,
    model=None,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Unified interface for bird part segmentation.

    This function segments a bird into its constituent parts (head, body, wing, etc.)
    given an image and bounding box. 

    Args:
        image: Input image as numpy array (H, W, 3) in BGR format.
        bbox: Bounding box as [x1, y1, x2, y2] defining the bird region.
        prior_mask: Optional prior mask (H, W) as guidance for segmentation (unused).
        model: Optional pre-loaded model / segmenter (if None, a default is created).
        device: Device to run inference ('cuda' or 'cpu').

    Returns:
        Dictionary mapping part names to binary masks:
        {
            'head': np.ndarray (H, W) with values 0 or 255,
            'body': np.ndarray (H, W) with values 0 or 255,
            'wing': np.ndarray (H, W) with values 0 or 255,
            'tail': np.ndarray (H, W) with values 0 or 255,
            ...
        }
    """
    if model is None:
        # Create a new segmenter (less efficient, but works as a fallback)
        segmenter = BirdPartSegmenter(device=device)
        model = segmenter

    return model.segment(image, bbox, prior_mask)


class BirdPartSegmenter:
    """
    Bird part segmentation wrapper.

    This class loads a trained segmentation model and provides a unified
    interface for segmenting birds into their constituent parts.

    Supported built-in model types:
        - 'unet'
        - 'deeplab'
        - 'sam_unet'
        - 'dinov3_msu'
        - 'dinov3_mlf'

    Additional model types can be registered via MODEL_REGISTRY.
    """

    # Model registry for extensible model selection
    MODEL_REGISTRY: Dict[str, Callable] = {}

    def __init__(
        self,
        model_path: str = None,
        model_type: str = "deeplab",
        num_classes: int = 11,
        device: str = "cuda",
        resize: int = 512,
        label_map: str = None,
        # DINOv3-specific parameters
        variant: str = "vitl16",
        dinov3_weights: str = None,
        enhanced_decoder: bool = False,
        take_n: int = 1,
        # Optional: BatchNorm flag for MSU decoder (mirrors pred.py, default False)
        use_batch_norm: bool = False,
    ):
        """
        Initialize bird part segmenter.

        Args:
            model_path: Path to trained model weights (.pth file).
            model_type: Type of model:
                'unet', 'deeplab', 'sam_unet',
                'dinov3_msu', 'dinov3_mlf'.
            num_classes: Number of segmentation classes (including background).
            device: Device to run inference ('cuda' or 'cpu').
            resize: Image resize dimension for non-DINOv3 models (square).
            label_map: Optional path to JSON file mapping class IDs to part names.
            variant: DINOv3 variant (for DINOv3 models): vits16, vitb16, vitl16, etc.
            dinov3_weights: Path to DINOv3 backbone weights (for DINOv3 models).
            enhanced_decoder: Use enhanced decoder (DINOv3 MSU/MLF).
            take_n: Number of intermediate layers for DINOv3.
            use_batch_norm: Use BatchNorm in decoder (only applies to dinov3_msu).
        """
        self.model_path = model_path
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device
        self.resize = resize
        self.label_map_path = label_map

        # DINOv3-specific parameters
        self.variant = variant
        self.dinov3_weights = dinov3_weights
        self.enhanced_decoder = enhanced_decoder
        self.take_n = take_n
        self.use_batch_norm = use_batch_norm

        # Override resize for specific model types
        if model_type == "sam_unet":
            self.resize = 1024
            logger.info("SAM model detected, overriding resize to 1024")
        elif model_type in ("dinov3_msu", "dinov3_mlf"):
            # DINOv3 models are trained/inferred at 1024×1024 by default (see pred.py)
            self.resize = 1024
            logger.info("DINOv3 model detected, overriding resize to 1024")
        else:
            self.resize = resize

        # Default class ID to part name mapping (can be overridden by label_map)
        self.id_to_part = {
            0: "background",
            1: "head",   # Head region (head + throat + eye)
            2: "body",   # Body (back + breast + belly + vent)
            3: "wing",   # Wing (coverts + remiges)
            4: "tail",   # Tail
            5: "beak",   # Beak
            6: "feet",   # Feet
            7: "label",  # Label region
        }

        # Load label map if provided
        if label_map and os.path.isfile(label_map):
            self._load_label_map()

        # Load model if path is provided
        self.model = None
        if model_path:
            self._load_model()
        else:
            logger.warning(
                "No model path provided. Segmentation will return dummy masks. "
                "Please provide a trained model path for real segmentation."
            )

    def _load_label_map(self):
        """Load label map from JSON file to override id_to_part."""
        try:
            with open(self.label_map_path, "r") as f:
                label_map = json.load(f)

            # Convert string keys to integers
            self.id_to_part = {int(k): v for k, v in label_map.items()}
            logger.info(
                f"Loaded label map from {self.label_map_path}: {self.id_to_part}"
            )
        except Exception as e:
            logger.error(f"Failed to load label map from {self.label_map_path}: {e}")
            logger.info("Using default label mapping")

    @classmethod
    def register_model(cls, model_name: str, loader_func: Callable):
        """
        Register a model loader function for extensible model support.

        Args:
            model_name: Name of the model type (e.g., 'unet', 'dinov3_msu').
            loader_func: Function that takes (segmenter_instance) and returns a model.
        """
        cls.MODEL_REGISTRY[model_name] = loader_func
        logger.debug(f"Registered model type: {model_name}")

    def _load_model(self):
        """Load the segmentation model using the registry pattern."""
        try:
            logger.info(f"Loading {self.model_type} model from {self.model_path}")

            # Use registry if model type is registered
            if self.model_type in self.MODEL_REGISTRY:
                self.model = self.MODEL_REGISTRY[self.model_type](self)
            else:
                # Fallback to built-in models
                self.model = self._load_builtin_model()

            if self.model is not None:
                self.model.eval()
                logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _load_builtin_model(self):
        """Load built-in model types (unet, deeplab, sam_unet, dinov3_msu, dinov3_mlf)."""
        from models.unet import UNet
        from models.deeplab import DeepLabWrapper

        # Classic models
        if self.model_type == "unet":
            model = UNet(in_channels=3, num_classes=self.num_classes).to(self.device)
        elif self.model_type == "deeplab":
            model = DeepLabWrapper(num_classes=self.num_classes).to(self.device)
        elif self.model_type == "sam_unet":
            from models.sam_net import SAM_UNet

            model = SAM_UNet(
                sam_checkpoint=self.model_path, num_classes=self.num_classes
            ).to(self.device)
        # DINOv3 models (MSU / MLF), consistent with pred.py
        elif self.model_type in ("dinov3_msu", "dinov3_mlf"):
            model = self._load_dinov3_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Load weights (except for SAM which typically loads internally from sam_checkpoint)
        if self.model_type != "sam_unet" and self.model_type not in ("dinov3_msu", "dinov3_mlf"):
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        # For DINOv3_MSU / DINOv3_MLF we assume self.model_path contains model.state_dict()
        if self.model_type in ("dinov3_msu", "dinov3_mlf"):
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)

        return model

    def _load_dinov3_model(self):
        """
        Load a DINOv3-based segmentation model.

        Uses:
            - DINOv3_MSU (multi-stage upsampling)
            - DINOv3_MLF (multi-layer fusion)

        The implementation mirrors the logic in pred.py for building the model.
        """
        from models.dinov3_multistage_upsampling import (
            DINOv3_MSU,
            build_dinov3_backbone,
        )
        from models.dinov3_multilayer_fusion import (
            DINOv3_MLF,
            build_dinov3_backbone_mlf,
        )

        if self.dinov3_weights is None:
            raise ValueError(
                "dinov3_weights parameter is required for DINOv3-based models "
                f"(model_type={self.model_type})"
            )

        if not os.path.isfile(self.dinov3_weights):
            raise FileNotFoundError(
                f"DINOv3 weights file not found: {self.dinov3_weights}"
            )

        out_h = out_w = self.resize

        if self.model_type == "dinov3_msu":
            logger.info(f"Loading DINOv3 MSU backbone: {self.variant}")
            dino_backbone = build_dinov3_backbone(
                variant=self.variant, weights=self.dinov3_weights
            )
            model = DINOv3_MSU(
                dino_backbone=dino_backbone,
                num_classes=self.num_classes,
                freeze_encoder=True,  # for inference
                enhanced_decoder=self.enhanced_decoder,
                take_n=self.take_n,
                output_size=(out_h, out_w),
                use_batch_norm=self.use_batch_norm,
            ).to(self.device)
        elif self.model_type == "dinov3_mlf":
            logger.info(f"Loading DINOv3 MLF backbone: {self.variant}")
            dino_backbone = build_dinov3_backbone_mlf(
                variant=self.variant, weights=self.dinov3_weights
            )
            model = DINOv3_MLF(
                dino_backbone=dino_backbone,
                num_classes=self.num_classes,
                freeze_encoder=True,
                take_n=self.take_n,
                output_size=(out_h, out_w),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported DINOv3 model_type: {self.model_type}")

        # Initialize lazy projection with a dummy forward
        with torch.no_grad():
            _ = model(torch.zeros(1, 3, out_h, out_w, device=self.device))

        return model


    def segment(
        self,
        image: np.ndarray,
        bbox: list,
        prior_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Segment bird parts from image and bounding box.

        Args:
            image: Input image (H, W, 3) in BGR format.
            bbox: Bounding box [x1, y1, x2, y2].
            prior_mask: Optional prior mask for guidance (currently unused).

        Returns:
            Dictionary of part name to binary mask.
        """
        # Extract region of interest
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = (
            max(0, x1),
            max(0, y1),
            min(image.shape[1], x2),
            min(image.shape[0], y2),
        )

        # Crop image
        roi = image[y1:y2, x1:x2].copy()

        if roi.size == 0:
            logger.warning("Empty ROI, returning empty masks")
            return self._empty_masks(image.shape[:2])

        # If no model is loaded, return dummy masks
        if self.model is None:
            logger.debug("No model loaded, returning dummy masks")
            return self._dummy_masks(image.shape[:2], bbox)

        # Preprocess for model
        roi_preprocessed = self._preprocess(roi)

        # Run inference
        with torch.no_grad():
            output = self.model(roi_preprocessed)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Resize prediction back to ROI size
        pred_resized = cv2.resize(
            pred.astype(np.uint8),
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST,
        )

        # Create full-size masks
        full_masks: Dict[str, np.ndarray] = {}
        for class_id, part_name in self.id_to_part.items():
            if class_id == 0:  # Skip background
                continue

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            part_mask = (pred_resized == class_id).astype(np.uint8) * 255
            mask[y1:y2, x1:x2] = part_mask
            full_masks[part_name] = mask

        return full_masks

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference.

        Args:
            image: Input image (H, W, 3) in BGR format.

        Returns:
            Preprocessed tensor (1, 3, H, W).
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to square self.resize × self.resize
        image_resized = cv2.resize(image_rgb, (self.resize, self.resize))

        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_norm = (image_resized / 255.0 - mean) / std

        # Convert to tensor
        image_tensor = torch.from_numpy(image_norm).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        return image_tensor.to(self.device)

    def _dummy_masks(self, shape: Tuple[int, int], bbox: list) -> Dict[str, np.ndarray]:
        """
        Generate dummy masks for testing when no model is available.

        Args:
            shape: Image shape (H, W).
            bbox: Bounding box [x1, y1, x2, y2].

        Returns:
            Dictionary of dummy masks.
        """
        masks: Dict[str, np.ndarray] = {}
        x1, y1, x2, y2 = bbox

        for part_name in ["head", "body", "wing", "tail", "beak", "feet"]:
            mask = np.zeros(shape, dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            masks[part_name] = mask

        logger.debug("Generated dummy masks")
        return masks

    def _empty_masks(self, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Generate empty masks."""
        return {
            part_name: np.zeros(shape, dtype=np.uint8)
            for part_name in ["head", "body", "wing", "tail", "beak", "feet"]
        }
