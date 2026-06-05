# SPDX-License-Identifier: MIT
"""
Base class for all trackers in the DTS pipeline.
Provides a common interface for object tracking.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseTracker(ABC):
    """
    Abstract base class for object trackers.
    
    All trackers should inherit from this class and implement the update() method.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the tracker with configuration parameters.
        
        Args:
            **kwargs: Tracker-specific configuration parameters
        """
        self.config = kwargs
        self.frame_count = 0
    
    @abstractmethod
    def update(self, detections: List[Dict[str, Any]], image: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections and return tracked objects.
        
        Args:
            detections: List of detection dictionaries from detector, each containing:
                - 'bbox': [x1, y1, x2, y2]
                - 'score': confidence score
                - 'class_id': class ID
            image: Optional image for appearance-based tracking
            
        Returns:
            List of tracked object dictionaries, each containing:
                - 'bbox': [x1, y1, x2, y2]
                - 'score': confidence score
                - 'class_id': class ID
                - 'track_id': Unique track ID
                - Additional fields from input detections
        """
        pass
    
    def reset(self):
        """Reset the tracker state."""
        self.frame_count = 0
    
    def __repr__(self):
        """String representation of the tracker."""
        return f"{self.__class__.__name__}(config={self.config})"
