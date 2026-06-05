# SPDX-License-Identifier: MIT
"""
GroundingDINO detector implementation for text-prompted object detection.
"""

import logging
from typing import List, Dict, Any
import numpy as np

from .base_detector import BaseDetector

# Setup logging
logger = logging.getLogger(__name__)


class GDINODetector(BaseDetector):
    """
    GroundingDINO-based object detector with text prompts.
    
    Uses natural language prompts to detect specific objects.
    """
    
    def __init__(self, 
                 model_config: str,
                 model_weights: str,
                 prompts: str = "bird",
                 box_threshold: float = 0.3,
                 text_threshold: float = 0.25,
                 device: str = "cuda",
                 **kwargs):
        """
        Initialize GroundingDINO detector.
        
        Args:
            model_config: Path to GroundingDINO config file
            model_weights: Path to model weights
            prompts: Text prompts separated by comma (e.g., "bird, wing, head")
            box_threshold: Box confidence threshold
            text_threshold: Text similarity threshold
            device: Device to run inference ('cuda' or 'cpu')
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        try:
            from groundingdino.util.inference import load_model, predict
            import groundingdino.datasets.transforms as T
        except ImportError:
            raise ImportError(
                "GroundingDINO not installed. "
                "Please install GroundingDINO following the official instructions."
            )
        
        self.model_config = model_config
        self.model_weights = model_weights
        self.prompts = prompts
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        
        # Load model
        logger.info(f"Loading GroundingDINO model from {model_weights}")
        self.model = load_model(model_config, model_weights, device=device)
        logger.info(f"GroundingDINO model loaded. Prompts: {prompts}")
        
        # Store transform function
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using GroundingDINO with text prompts.
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR or RGB format
            
        Returns:
            List of detection dictionaries with keys:
                - 'bbox': [x1, y1, x2, y2]
                - 'score': confidence score
                - 'class_id': always 0 (GroundingDINO doesn't use class IDs)
                - 'class_name': matched phrase from prompts
        """
        from groundingdino.util.inference import predict
        from torchvision.ops import box_convert
        from PIL import Image
        
        # Convert BGR to RGB if needed
        if image.shape[2] == 3:
            image_rgb = image[..., ::-1].copy()  # BGR to RGB
        else:
            image_rgb = image
        
        # Convert to PIL Image
        image_pil = Image.fromarray(image_rgb)
        
        # Apply transforms
        image_transformed, _ = self.transform(image_pil, None)
        
        # Run prediction
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=self.prompts,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        detections = []
        for score, bbox, phrase in zip(logits, boxes, phrases):
            # Convert from normalized cxcywh to xyxy
            xyxy = box_convert(boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            xyxy = (xyxy * np.array([w, h, w, h])).astype(int)
            
            detection = {
                'bbox': [int(x) for x in xyxy.tolist()],
                'score': float(score.item()),
                'class_id': 0,  # GroundingDINO uses phrases, not class IDs
                'class_name': phrase
            }
            detections.append(detection)
        
        logger.debug(f"Found {len(detections)} detections")
        return detections
