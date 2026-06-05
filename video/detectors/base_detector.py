# SPDX-License-Identifier: MIT
"""
Base class for all detectors in the DTS pipeline.
Provides a common interface for object detection.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for object detectors.
    
    All detectors should inherit from this class and implement the detect() method.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the detector with configuration parameters.
        
        Args:
            **kwargs: Detector-specific configuration parameters
        """
        self.config = kwargs
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            
        Returns:
            List of detection dictionaries, each containing:
                - 'bbox': Bounding box as [x1, y1, x2, y2]
                - 'score': Confidence score (0-1)
                - 'class_id': Class ID
                - 'class_name': Class name (optional)
        """
        pass
    
    def __repr__(self):
        """String representation of the detector."""
        return f"{self.__class__.__name__}(config={self.config})"
