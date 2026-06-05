# SPDX-License-Identifier: MIT
"""
Segmentation module: Provides unified interface for bird part segmentation.
Users can replace the segmentation logic with their own models.
"""

from .segmenter import segment_parts, BirdPartSegmenter

__all__ = ['segment_parts', 'BirdPartSegmenter']
