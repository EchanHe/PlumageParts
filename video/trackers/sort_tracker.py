# SPDX-License-Identifier: MIT
"""
SORT (Simple Online and Realtime Tracking) tracker implementation.
Uses Kalman filter and Hungarian algorithm for tracking.
"""

import logging
from typing import List, Dict, Any
import numpy as np

from .base_tracker import BaseTracker

# Setup logging
logger = logging.getLogger(__name__)


class SORTTracker(BaseTracker):
    """
    SORT-based object tracker.
    
    Simple, efficient tracking using Kalman filter and Hungarian matching.
    Fallback to a simple IoU-based tracker if SORT/FilterPy is not available.
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 **kwargs):
        """
        Initialize SORT tracker.
        
        Args:
            max_age: Maximum frames to keep track alive without detections
            min_hits: Minimum hits to establish a track
            iou_threshold: IoU threshold for matching
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Try to import SORT
        self.use_sort = False
        try:
            # Try to use external SORT implementation
            from .sort_impl import Sort
            self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
            self.use_sort = True
            logger.info("Using SORT tracker with Kalman filter")
        except ImportError:
            logger.warning("SORT not available, using simple IoU-based tracker")
            self.tracker = SimpleIoUTracker(
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold
            )
    
    def update(self, detections: List[Dict[str, Any]], image: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
            image: Optional image (not used in SORT)
            
        Returns:
            List of tracked objects with track IDs
        """
        self.frame_count += 1
        
        if not detections:
            # Update tracker with empty detections
            if self.use_sort:
                tracks = self.tracker.update(np.empty((0, 5)))
            else:
                tracks = self.tracker.update([])
            return self._format_tracks(tracks, [])
        
        # Convert detections to SORT format: [x1, y1, x2, y2, score]
        dets = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['score']] 
                         for d in detections])
        
        # Update tracker
        if self.use_sort:
            tracks = self.tracker.update(dets)
        else:
            tracks = self.tracker.update(detections)
        
        return self._format_tracks(tracks, detections)
    
    def _format_tracks(self, tracks, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tracker output to match expected format.
        
        Args:
            tracks: Tracker output
            detections: Original detections for metadata
            
        Returns:
            List of tracked objects
        """
        tracked_objects = []
        
        if len(tracks) == 0:
            return tracked_objects
        
        for track in tracks:
            if self.use_sort:
                # SORT format: [x1, y1, x2, y2, track_id]
                x1, y1, x2, y2, track_id = track
            else:
                # SimpleIoUTracker format: dict
                x1, y1, x2, y2 = track['bbox']
                track_id = track['track_id']
            
            # Find matching detection for additional metadata
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            matched_det = self._find_matching_detection(bbox, detections)
            
            tracked_obj = {
                'bbox': bbox,
                'track_id': int(track_id),
                'frame': self.frame_count
            }
            
            # Add metadata from matched detection
            if matched_det:
                tracked_obj.update({
                    'score': matched_det.get('score', 0.0),
                    'class_id': matched_det.get('class_id', 0),
                    'class_name': matched_det.get('class_name', 'unknown')
                })
                # Copy additional fields like masks
                for key in ['mask']:
                    if key in matched_det:
                        tracked_obj[key] = matched_det[key]
            
            tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def _find_matching_detection(self, bbox: List[int], detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find the detection that best matches the given bbox using IoU.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            detections: List of detections
            
        Returns:
            Matching detection or None
        """
        if not detections:
            return None
        
        max_iou = 0
        best_match = None
        
        for det in detections:
            iou = self._compute_iou(bbox, det['bbox'])
            if iou > max_iou:
                max_iou = iou
                best_match = det
        
        return best_match if max_iou > 0.1 else None
    
    @staticmethod
    def _compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """
        Compute IoU between two bounding boxes.
        
        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self):
        """Reset tracker state."""
        super().reset()
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()


class SimpleIoUTracker:
    """
    Simple IoU-based tracker as fallback when SORT is not available.
    
    Uses IoU matching without Kalman filtering.
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize simple tracker.
        
        Args:
            max_age: Maximum frames to keep track alive
            min_hits: Minimum hits to establish a track
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []  # List of active tracks
        self.next_id = 1
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of active tracks
        """
        # Match detections to existing tracks
        matched_tracks = []
        unmatched_dets = list(range(len(detections)))
        
        for track in self.tracks:
            best_iou = 0
            best_det_idx = -1
            
            for det_idx in unmatched_dets:
                iou = self._compute_iou(track['bbox'], detections[det_idx]['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                # Update track with matched detection
                track['bbox'] = detections[best_det_idx]['bbox']
                track['score'] = detections[best_det_idx]['score']
                track['hits'] += 1
                track['age'] = 0
                matched_tracks.append(track)
                unmatched_dets.remove(best_det_idx)
            else:
                # No match, age the track
                track['age'] += 1
                if track['age'] < self.max_age:
                    matched_tracks.append(track)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = {
                'track_id': self.next_id,
                'bbox': detections[det_idx]['bbox'],
                'score': detections[det_idx]['score'],
                'hits': 1,
                'age': 0
            }
            self.next_id += 1
            matched_tracks.append(new_track)
        
        self.tracks = matched_tracks
        
        # Return tracks that have enough hits
        return [t for t in self.tracks if t['hits'] >= self.min_hits]
    
    @staticmethod
    def _compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 1
