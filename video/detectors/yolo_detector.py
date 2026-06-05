# SPDX-License-Identifier: MIT
"""
YOLO detector implementation using Ultralytics YOLO.
Supports YOLOv8 and other YOLO variants.
"""

import logging
from typing import List, Dict, Any
import numpy as np

from .base_detector import BaseDetector

# Setup logging
logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """
    YOLO-based object detector using Ultralytics library.
    
    Supports detection and instance segmentation models.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", 
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.7,
                 classes: List[int] = None,
                 device: str = "cuda",
                 imgsz: int = 640,
                 **kwargs):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights (e.g., 'yolov8n.pt')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            classes: List of class IDs to detect (None = all classes)
            device: Device to run inference ('cuda' or 'cpu')
            imgsz: Input image size for inference
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics YOLO not installed. "
                "Please install: pip install ultralytics"
            )
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.device = device
        self.imgsz = imgsz
        
        # Load model
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        
        # Get class names
        self.class_names = self.model.names
        logger.info(f"YOLO model loaded. Available classes: {len(self.class_names)}")
        
        # Check if model supports segmentation
        self.is_seg = getattr(self.model, "task", None) == "segment"
        if self.is_seg:
            logger.info("YOLO segmentation model detected")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using YOLO.
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            
        Returns:
            List of detection dictionaries with keys:
                - 'bbox': [x1, y1, x2, y2]
                - 'score': confidence score
                - 'class_id': class ID
                - 'class_name': class name
                - 'mask': binary mask (if segmentation model)
        """
        # Run YOLO prediction
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False
        )
        
        if not results or len(results) == 0:
            logger.debug("No detections found")
            return []
        
        result = results[0]
        detections = []
        
        # Process detections
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            conf = result.boxes.conf.cpu().numpy()  # Confidence scores
            cls = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
            
            for i in range(len(xyxy)):
                class_id = int(cls[i])
                class_name = self.class_names.get(class_id, str(class_id))
                
                detection = {
                    'bbox': [int(round(x)) for x in xyxy[i].tolist()],
                    'score': float(conf[i]),
                    'class_id': class_id,
                    'class_name': class_name
                }
                
                # Add segmentation mask if available
                if self.is_seg and result.masks is not None:
                    # Get binary mask aligned with input image size
                    mask = self._get_binary_mask(result, i)
                    detection['mask'] = mask
                
                detections.append(detection)
        
        logger.debug(f"Found {len(detections)} detections")
        return detections
    
    def _get_binary_mask(self, result, idx: int, threshold: float = 0.5) -> np.ndarray:
        """
        Extract binary mask from YOLO segmentation result.
        
        Args:
            result: YOLO result object
            idx: Index of the detection
            threshold: Threshold for mask binarization
            
        Returns:
            Binary mask as numpy array (H, W), dtype uint8, values 0 or 255
        """
        import cv2
        
        # Get mask probability map
        mask_prob = result.masks.data[idx].cpu().numpy()
        
        # Resize to original image size
        H, W = result.orig_shape
        mask_resized = cv2.resize(mask_prob, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Binarize
        binary_mask = (mask_resized > threshold).astype(np.uint8) * 255
        
        return binary_mask
