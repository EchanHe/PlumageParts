# SPDX-License-Identifier: MIT
"""
Detector module: Provides object detection interfaces for the DTS pipeline.
Supports YOLO and GroundingDINO detectors.
"""

from .base_detector import BaseDetector
from .yolo_detector import YOLODetector
from .gdino_detector import GDINODetector

__all__ = ['BaseDetector', 'YOLODetector', 'GDINODetector']
