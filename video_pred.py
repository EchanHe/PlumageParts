#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
DTS Pipeline: Detect → Track → Segment

A modular pipeline for bird detection, tracking, and part segmentation in videos.

Usage Examples
--------------

# Process video with YOLO detector and SORT tracker + DINOv3 MSU segmenter:
python video_pred.py --video input.mp4 --output output_dir \
    --detector yolo --detector_model yolov8n.pt \
    --tracker sort \
    --segment_model path/to/dinov3_msu_segmentation.pth \
    --segment_model_type dinov3_msu \
    --dinov3_weights path/to/dinov3_backbone_weights.pth

# Process video with classic DeepLab segmenter:
python video_pred.py --video input.mp4 --output output_dir \
    --detector yolo --tracker sort \
    --segment_model path/to/deeplab_model.pth \
    --segment_model_type deeplab

# Process image directory:
python video_pred.py --image_dir images/ --output output_dir \
    --detector yolo --no_tracking \
    --segment_model path/to/dinov3_mlf_segmentation.pth \
    --segment_model_type dinov3_mlf \
    --dinov3_weights path/to/dinov3_backbone_weights.pth

# Dry run (detection and tracking only, no segmentation):
python video_pred.py --video input.mp4 --output output_dir \
    --detector yolo --tracker sort --no_segmentation
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
import matplotlib
import json
import torch
from tqdm import tqdm
from datetime import datetime
import time
import csv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import DTS modules
from video.detectors import YOLODetector, GDINODetector
from video.trackers import SORTTracker
from video.segment import segment_parts, BirdPartSegmenter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def get_color_map(part_names):
    n = len(part_names)
    cmap = matplotlib.colormaps.get_cmap("tab20" if n <= 20 else "hsv")
    colors = {}
    for i, name in enumerate(part_names):
        rgb = np.array(cmap(i / n)[:3]) * 255
        colors[name] = tuple(map(int, rgb))
    return colors


class DTSPipeline:
    """
    Complete DTS (Detect → Track → Segment) pipeline for bird analysis in videos.
    """

    def __init__(
        self,
        detector,
        tracker=None,
        segmenter=None,
        output_dir: str = "output",
        save_frames: bool = True,
        save_masks: bool = True,
        visualize: bool = True,
        enable_profiling: bool = False,
        perf_csv_path: Optional[str] = None,
        mask_alpha: float = 0.3,
    ):
        """
        Initialize DTS pipeline.

        Args:
            detector: Detector instance (YOLODetector or GDINODetector).
            tracker: Tracker instance (SORTTracker or None for no tracking).
            segmenter: Segmenter instance (BirdPartSegmenter or None for no segmentation).
            output_dir: Directory to save results.
            save_frames: Whether to save annotated frames.
            save_masks: Whether to save segmentation masks.
            visualize: Whether to create visualization overlays.
            enable_profiling: Whether to enable performance profiling.
            perf_csv_path: Path to save per-frame profiling CSV (None = auto-generate).
            mask_alpha: Alpha blending factor for mask visualization.
        """
        self.detector = detector
        self.tracker = tracker
        self.segmenter = segmenter
        self.label_map = (
            getattr(segmenter, "id_to_part", None) if segmenter is not None else None
        )
        self.output_dir = Path(output_dir)
        self.save_frames = save_frames
        self.save_masks = save_masks
        self.visualize = visualize
        self.enable_profiling = enable_profiling
        self.perf_csv_path = perf_csv_path
        self.mask_alpha = mask_alpha

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if save_frames:
            (self.output_dir / "frames").mkdir(exist_ok=True)
        if save_masks:
            (self.output_dir / "masks").mkdir(exist_ok=True)

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "detections": [],
            "tracks": [],
            "segmentations": [],
        }

        # Profiling data
        self.profiling_data = {
            "per_frame": [],  # List of per-frame timing records
            "summary": {},  # Summary statistics
        }

        logger.info(f"DTS Pipeline initialized. Output: {self.output_dir}")
        logger.info(f"  Detector: {detector}")
        logger.info(f"  Tracker: {tracker if tracker else 'None (disabled)'}")
        logger.info(f"  Segmenter: {segmenter if segmenter else 'None (disabled)'}")
        if enable_profiling:
            logger.info("  Profiling: Enabled")
            if perf_csv_path:
                logger.info(f"  Profiling CSV: {perf_csv_path}")

    def process_video(self, video_path: str, max_frames: Optional[int] = None):
        """
        Process a video file through the DTS pipeline.
        """
        logger.info(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

        if max_frames:
            total_frames = min(total_frames, max_frames)

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Processing frames")

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            result = self.process_frame(frame, frame_idx)

            if self.save_frames and result["visualization"] is not None:
                frame_path = self.output_dir / "frames" / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), result["visualization"])

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        self._save_statistics()

        if self.enable_profiling:
            self._save_profiling_data()

        logger.info(f"Video processing complete. Processed {frame_idx} frames.")

    def process_image_dir(self, image_dir: str):
        """
        Process a directory of images through the DTS pipeline.
        """
        logger.info(f"Processing image directory: {image_dir}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths: List[Path] = []
        for ext in image_extensions:
            image_paths.extend(Path(image_dir).glob(f"*{ext}"))
            image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))

        image_paths = sorted(image_paths)
        logger.info(f"Found {len(image_paths)} images")

        for idx, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue

            result = self.process_frame(image, idx)

            if self.save_frames and result["visualization"] is not None:
                output_name = img_path.stem + "_dts" + img_path.suffix
                frame_path = self.output_dir / "frames" / output_name
                cv2.imwrite(str(frame_path), result["visualization"])

        self._save_statistics()

        if self.enable_profiling:
            self._save_profiling_data()

        logger.info(
            f"Image directory processing complete. Processed {len(image_paths)} images."
        )

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """
        Process a single frame through the DTS pipeline.
        """
        self.stats["frames_processed"] += 1

        profiling_record = {
            "frame_idx": frame_idx,
            "detection_time": 0.0,
            "tracking_time": 0.0,
            "segmentation_time": 0.0,
            "visualization_time": 0.0,
            "total_time": 0.0,
            "num_detections": 0,
            "num_tracks": 0,
            "num_segmentations": 0,
        }

        frame_start_time = time.time()

        # Step 1: Detection
        detection_start = time.time()
        detections = self.detector.detect(frame)
        detection_time = time.time() - detection_start

        logger.debug(f"Frame {frame_idx}: {len(detections)} detections")

        for det in detections:
            det["frame"] = frame_idx
            self.stats["detections"].append(det.copy())

        if self.enable_profiling:
            profiling_record["detection_time"] = detection_time
            profiling_record["num_detections"] = len(detections)

        # Step 2: Tracking (optional)
        tracked_objects = []
        tracking_start = time.time()
        if self.tracker:
            tracked_objects = self.tracker.update(detections, frame)
            logger.debug(f"Frame {frame_idx}: {len(tracked_objects)} tracks")

            for track in tracked_objects:
                self.stats["tracks"].append(track.copy())
        else:
            tracked_objects = detections
        tracking_time = time.time() - tracking_start

        if self.enable_profiling:
            profiling_record["tracking_time"] = tracking_time
            profiling_record["num_tracks"] = len(tracked_objects)

        # Step 3: Segmentation (optional)
        segmentations: Dict[Any, Dict[str, np.ndarray]] = {}
        segmentation_start = time.time()
        if self.segmenter:
            for obj in tracked_objects:
                bbox = obj["bbox"]
                obj_id = obj.get("track_id", f"det_{len(segmentations)}")

                part_masks = self.segmenter.segment(frame, bbox)
                segmentations[obj_id] = part_masks

                if self.save_masks:
                    self._save_part_masks(part_masks, frame_idx, obj_id)

            logger.debug(f"Frame {frame_idx}: {len(segmentations)} segmentations")
        segmentation_time = time.time() - segmentation_start

        if self.enable_profiling:
            profiling_record["segmentation_time"] = segmentation_time
            profiling_record["num_segmentations"] = len(segmentations)

        # Step 4: Visualization (optional)
        visualization = None
        visualization_start = time.time()
        if self.visualize:
            visualization = self._create_visualization(
                frame, tracked_objects, segmentations
            )
        visualization_time = time.time() - visualization_start

        if self.enable_profiling:
            profiling_record["visualization_time"] = visualization_time
            profiling_record["total_time"] = time.time() - frame_start_time
            self.profiling_data["per_frame"].append(profiling_record)

        return {
            "detections": detections,
            "tracks": tracked_objects,
            "segmentations": segmentations,
            "visualization": visualization,
            "profiling": profiling_record if self.enable_profiling else None,
        }

    def _create_visualization(
        self,
        frame: np.ndarray,
        objects: List[Dict[str, Any]],
        segmentations: Dict[int, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """
        Create visualization with bounding boxes and masks.
        """
        vis = frame.copy()

        # Draw segmentation masks with transparency
        if segmentations:
            mask_overlay = np.zeros_like(frame)
            if self.label_map is not None:
                part_names = [v for k, v in self.label_map.items() if str(k) != "0"]
            else:
                part_names = ["head", "body", "wing", "tail", "beak", "feet"]
            colors = get_color_map(part_names)

            for obj_id, part_masks in segmentations.items():
                for part_name, mask in part_masks.items():
                    if part_name in colors:
                        color = colors[part_name]
                        mask_overlay[mask > 0] = color

            alpha = self.mask_alpha
            vis = cv2.addWeighted(vis, 1.0 - alpha, mask_overlay, alpha, 0)

        # Draw bounding boxes and labels
        for obj in objects:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = bbox

            obj_id = obj.get("track_id", "")
            score = obj.get("score", 0.0)
            class_name = obj.get("class_name", "object")

            color = (0, 255, 0) if obj_id else (255, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = (
                f"ID:{obj_id} {class_name} {score:.2f}"
                if obj_id
                else f"{class_name} {score:.2f}"
            )
            cv2.putText(
                vis,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return vis

    def _save_part_masks(
        self,
        part_masks: Dict[str, np.ndarray],
        frame_idx: int,
        obj_id: int,
    ):
        """
        Save individual part masks to disk.
        """
        for part_name, mask in part_masks.items():
            mask_path = (
                self.output_dir
                / "masks"
                / f"frame_{frame_idx:06d}_obj_{obj_id}_{part_name}.png"
            )
            cv2.imwrite(str(mask_path), mask)

    def _save_statistics(self):
        """Save pipeline statistics to JSON file."""
        stats_path = self.output_dir / "statistics.json"

        summary = {
            "total_frames": self.stats["frames_processed"],
            "total_detections": len(self.stats["detections"]),
            "total_tracks": len(
                set(t.get("track_id", -1) for t in self.stats["tracks"])
            ),
            "avg_detections_per_frame": (
                len(self.stats["detections"]) / self.stats["frames_processed"]
                if self.stats["frames_processed"] > 0
                else 0
            ),
        }

        output = {
            "summary": summary,
            "detections": self.stats["detections"][:1000],
            "tracks": self.stats["tracks"][:1000],
        }

        with open(stats_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Statistics saved to {stats_path}")
        logger.info(f"Summary: {summary}")

    def _save_profiling_data(self):
        """Save profiling data to JSON and optionally CSV files."""
        if not self.enable_profiling:
            return

        per_frame = self.profiling_data["per_frame"]

        if len(per_frame) == 0:
            logger.warning("No profiling data to save")
            return

        detection_times = [r["detection_time"] for r in per_frame]
        tracking_times = [r["tracking_time"] for r in per_frame]
        segmentation_times = [r["segmentation_time"] for r in per_frame]
        visualization_times = [r["visualization_time"] for r in per_frame]
        total_times = [r["total_time"] for r in per_frame]

        num_detections = [r["num_detections"] for r in per_frame]
        num_tracks = [r["num_tracks"] for r in per_frame]
        num_segmentations = [r["num_segmentations"] for r in per_frame]

        summary = {
            "total_frames": len(per_frame),
            "detection": {
                "mean_time": np.mean(detection_times),
                "median_time": np.median(detection_times),
                "std_time": np.std(detection_times),
                "min_time": np.min(detection_times),
                "max_time": np.max(detection_times),
                "total_time": np.sum(detection_times),
            },
            "tracking": {
                "mean_time": np.mean(tracking_times),
                "median_time": np.median(tracking_times),
                "std_time": np.std(tracking_times),
                "min_time": np.min(tracking_times),
                "max_time": np.max(tracking_times),
                "total_time": np.sum(tracking_times),
            },
            "segmentation": {
                "mean_time": np.mean(segmentation_times),
                "median_time": np.median(segmentation_times),
                "std_time": np.std(segmentation_times),
                "min_time": np.min(segmentation_times),
                "max_time": np.max(segmentation_times),
                "total_time": np.sum(segmentation_times),
            },
            "visualization": {
                "mean_time": np.mean(visualization_times),
                "median_time": np.median(visualization_times),
                "std_time": np.std(visualization_times),
                "min_time": np.min(visualization_times),
                "max_time": np.max(visualization_times),
                "total_time": np.sum(visualization_times),
            },
            "total_per_frame": {
                "mean_time": np.mean(total_times),
                "median_time": np.median(total_times),
                "std_time": np.std(total_times),
                "min_time": np.min(total_times),
                "max_time": np.max(total_times),
                "total_time": np.sum(total_times),
                "fps": len(per_frame) / np.sum(total_times)
                if np.sum(total_times) > 1e-6
                else 0.0,
            },
            "throughput": {
                "mean_detections_per_frame": np.mean(num_detections),
                "mean_tracks_per_frame": np.mean(num_tracks),
                "mean_segmentations_per_frame": np.mean(num_segmentations),
                "total_detections": np.sum(num_detections),
                "total_tracks": np.sum(num_tracks),
                "total_segmentations": np.sum(num_segmentations),
            },
        }

        self.profiling_data["summary"] = summary

        profiling_json_path = self.output_dir / "profiling.json"
        with open(profiling_json_path, "w") as f:
            serializable_data = convert_numpy_types(self.profiling_data)
            json.dump(serializable_data, f, indent=2)

        logger.info(f"Profiling data saved to {profiling_json_path}")

        if self.perf_csv_path:
            csv_path = Path(self.perf_csv_path)
        else:
            csv_path = self.output_dir / "profiling_per_frame.csv"

        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = [
                "frame_idx",
                "detection_time",
                "tracking_time",
                "segmentation_time",
                "visualization_time",
                "total_time",
                "num_detections",
                "num_tracks",
                "num_segmentations",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in per_frame:
                writer.writerow(record)

        logger.info(f"Per-frame profiling CSV saved to {csv_path}")
        logger.info("=" * 80)
        logger.info("Performance Profiling Summary")
        logger.info("=" * 80)
        logger.info(
            f"Total frames processed: {summary['total_frames']}"
        )
        logger.info(
            f"Overall throughput: {summary['total_per_frame']['fps']:.2f} FPS"
        )
        logger.info(
            f"Average processing time per frame: {summary['total_per_frame']['mean_time']*1000:.2f} ms"
        )
        logger.info(
            f"  Detection:     {summary['detection']['mean_time']*1000:.2f} ms "
            f"(±{summary['detection']['std_time']*1000:.2f} ms)"
        )
        logger.info(
            f"  Tracking:      {summary['tracking']['mean_time']*1000:.2f} ms "
            f"(±{summary['tracking']['std_time']*1000:.2f} ms)"
        )
        logger.info(
            f"  Segmentation:  {summary['segmentation']['mean_time']*1000:.2f} ms "
            f"(±{summary['segmentation']['std_time']*1000:.2f} ms)"
        )
        logger.info(
            f"  Visualization: {summary['visualization']['mean_time']*1000:.2f} ms "
            f"(±{summary['visualization']['std_time']*1000:.2f} ms)"
        )
        logger.info(
            f"Average detections per frame: "
            f"{summary['throughput']['mean_detections_per_frame']:.2f}"
        )
        logger.info(
            f"Average tracks per frame: "
            f"{summary['throughput']['mean_tracks_per_frame']:.2f}"
        )
        logger.info(
            f"Average segmentations per frame: "
            f"{summary['throughput']['mean_segmentations_per_frame']:.2f}"
        )
        logger.info("=" * 80)


def create_detector(args) -> Any:
    """
    Create detector based on command-line arguments.
    """
    logger.info(f"Creating detector: {args.detector}")

    if args.detector == "yolo":
        return YOLODetector(
            model_path=args.detector_model,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            classes=[args.target_class] if args.target_class is not None else None,
            device=args.device,
            imgsz=args.imgsz,
        )
    elif args.detector == "gdino":
        return GDINODetector(
            model_config=args.gdino_config,
            model_weights=args.gdino_weights,
            prompts=args.gdino_prompts,
            box_threshold=args.gdino_box_threshold,
            text_threshold=args.gdino_text_threshold,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown detector: {args.detector}")


def create_tracker(args) -> Optional[Any]:
    """
    Create tracker based on command-line arguments.
    """
    if args.no_tracking:
        logger.info("Tracking disabled")
        return None

    logger.info(f"Creating tracker: {args.tracker}")

    if args.tracker == "sort":
        return SORTTracker(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.track_iou_threshold,
        )
    else:
        raise ValueError(f"Unknown tracker: {args.tracker}")


def create_segmenter(args) -> Optional[Any]:
    """
    Create segmenter based on command-line arguments.
    """
    if args.no_segmentation:
        logger.info("Segmentation disabled")
        return None

    logger.info("Creating segmenter")

    return BirdPartSegmenter(
        model_path=args.segment_model,
        model_type=args.segment_model_type,
        num_classes=args.num_classes,
        device=args.device,
        resize=args.segment_resize,
        label_map=args.label_map,
        # DINOv3-specific parameters
        variant=args.variant,
        dinov3_weights=args.dinov3_weights,
        enhanced_decoder=args.enhanced_decoder,
        take_n=args.take_n,
    )


def main():
    """Main entry point for DTS pipeline."""
    parser = argparse.ArgumentParser(
        description="DTS Pipeline: Detect → Track → Segment birds in videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/Output arguments
    input_group = parser.add_argument_group("Input/Output")
    input_group.add_argument("--video", type=str, help="Path to input video file")
    input_group.add_argument("--image_dir", type=str, help="Path to image directory")
    input_group.add_argument(
        "--output",
        type=str,
        default="output_dts",
        help="Output directory (default: output_dts)",
    )
    input_group.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process",
    )

    # Detector arguments
    detector_group = parser.add_argument_group("Detector")
    detector_group.add_argument(
        "--detector",
        type=str,
        required=True,
        choices=["yolo", "gdino"],
        help="Detector type",
    )
    detector_group.add_argument(
        "--detector_model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path (default: yolov8n.pt)",
    )
    detector_group.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    detector_group.add_argument(
        "--iou_threshold",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7)",
    )
    detector_group.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Target class ID to detect (None = all classes)",
    )
    detector_group.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Detector input image size (default: 640)",
    )

    # GroundingDINO specific arguments
    gdino_group = parser.add_argument_group("GroundingDINO (if --detector=gdino)")
    gdino_group.add_argument(
        "--gdino_config", type=str, help="Path to GroundingDINO config file"
    )
    gdino_group.add_argument(
        "--gdino_weights", type=str, help="Path to GroundingDINO weights"
    )
    gdino_group.add_argument(
        "--gdino_prompts",
        type=str,
        default="bird",
        help="Text prompts for detection (default: 'bird')",
    )
    gdino_group.add_argument(
        "--gdino_box_threshold",
        type=float,
        default=0.3,
        help="Box threshold for GroundingDINO (default: 0.3)",
    )
    gdino_group.add_argument(
        "--gdino_text_threshold",
        type=float,
        default=0.25,
        help="Text threshold for GroundingDINO (default: 0.25)",
    )

    # Tracker arguments
    tracker_group = parser.add_argument_group("Tracker")
    tracker_group.add_argument(
        "--tracker",
        type=str,
        default="sort",
        choices=["sort"],
        help="Tracker type (default: sort)",
    )
    tracker_group.add_argument(
        "--no_tracking", action="store_true", help="Disable tracking"
    )
    tracker_group.add_argument(
        "--max_age",
        type=int,
        default=30,
        help="Max frames to keep track alive (default: 30)",
    )
    tracker_group.add_argument(
        "--min_hits",
        type=int,
        default=3,
        help="Min hits to establish track (default: 3)",
    )
    tracker_group.add_argument(
        "--track_iou_threshold",
        type=float,
        default=0.3,
        help="IoU threshold for tracking (default: 0.3)",
    )

    # Segmentation arguments
    segment_group = parser.add_argument_group("Segmentation")
    segment_group.add_argument(
        "--no_segmentation", action="store_true", help="Disable segmentation"
    )
    segment_group.add_argument(
        "--segment_model",
        type=str,
        help="Path to segmentation model weights (.pth)",
    )
    segment_group.add_argument(
        "--segment_model_type",
        type=str,
        default="deeplab",
        choices=["unet", "deeplab", "sam_unet", "dinov3_msu", "dinov3_mlf"],
        help="Segmentation model type (default: deeplab)",
    )
    segment_group.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of segmentation classes (default: 11)",
    )
    segment_group.add_argument(
        "--segment_resize",
        type=int,
        default=1024,
        help="Segmentation input size for non-DINOv3 models (default: 512)",
    )
    segment_group.add_argument(
        "--label_map",
        type=str,
        default=None,
        help="Path to JSON file mapping class IDs to part names",
    )

    # DINOv3-specific segmentation arguments
    dinov3_group = parser.add_argument_group(
        "DINOv3 Segmentation (if --segment_model_type=dinov3_msu or dinov3_mlf)"
    )
    dinov3_group.add_argument(
        "--variant",
        type=str,
        default="vitl16",
        choices=[
            "vits16",
            "vits16plus",
            "vitb16",
            "vitl16",
            "vith16plus",
            "vit7b16",
        ],
        help="DINOv3 ViT variant (default: vitl16)",
    )
    dinov3_group.add_argument(
        "--dinov3_weights",
        type=str,
        default=None,
        help="Path to DINOv3 backbone weights",
    )
    dinov3_group.add_argument(
        "--enhanced_decoder",
        action="store_true",
        help="Use enhanced decoder (GN+SiLU) for DINOv3 models",
    )
    dinov3_group.add_argument(
        "--take_n",
        type=int,
        default=1,
        help="Number of intermediate layers to take from DINOv3 (default: 1)",
    )

    # Visualization arguments
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument(
        "--no_visualization", action="store_true", help="Disable visualization"
    )
    viz_group.add_argument(
        "--no_save_frames", action="store_true", help="Don't save annotated frames"
    )
    viz_group.add_argument(
        "--no_save_masks", action="store_true", help="Don't save individual masks"
    )
    viz_group.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.3,
        help="Alpha for segmentation overlay in DTS visualization (default: 0.3)",
    )

    # Profiling arguments
    profiling_group = parser.add_argument_group("Profiling")
    profiling_group.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-frame timing and throughput metrics",
    )
    profiling_group.add_argument(
        "--perf_csv",
        type=str,
        default=None,
        help=(
            "Path to save per-frame profiling CSV "
            "(default: <output_dir>/profiling_per_frame.csv)"
        ),
    )

    # General arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input
    if not args.video and not args.image_dir:
        parser.error("Must provide either --video or --image_dir")

    if args.detector == "gdino":
        if not args.gdino_config or not args.gdino_weights:
            parser.error("GroundingDINO requires --gdino_config and --gdino_weights")

    # Validate DINOv3 segmentation parameters
    if (
        not args.no_segmentation
        and args.segment_model_type in ("dinov3_msu", "dinov3_mlf")
    ):
        if args.dinov3_weights is None:
            parser.error(
                "--dinov3_weights is required when using DINOv3-based segmenters "
                "(dinov3_msu or dinov3_mlf)"
            )
        if not os.path.isfile(args.dinov3_weights):
            parser.error(
                f"DINOv3 weights file not found: {args.dinov3_weights}"
            )

    # Create pipeline components
    logger.info("=" * 80)
    logger.info("DTS Pipeline - Detect → Track → Segment")
    logger.info("=" * 80)

    detector = create_detector(args)
    tracker = create_tracker(args)
    segmenter = create_segmenter(args)

    pipeline = DTSPipeline(
        detector=detector,
        tracker=tracker,
        segmenter=segmenter,
        output_dir=args.output,
        save_frames=not args.no_save_frames,
        save_masks=not args.no_save_masks,
        visualize=not args.no_visualization,
        enable_profiling=args.profile,
        perf_csv_path=args.perf_csv,
        mask_alpha=args.overlay_alpha,
    )

    if args.video:
        pipeline.process_video(args.video, max_frames=args.max_frames)
    elif args.image_dir:
        pipeline.process_image_dir(args.image_dir)

    logger.info("=" * 80)
    logger.info("DTS Pipeline completed successfully!")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
