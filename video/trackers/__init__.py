# SPDX-License-Identifier: MIT
"""
Tracker module: Provides object tracking interfaces for the DTS pipeline.
Supports various tracking algorithms including SORT, DeepSORT, and ByteTrack.
"""

from .base_tracker import BaseTracker
from .sort_tracker import SORTTracker

__all__ = ['BaseTracker', 'SORTTracker']
