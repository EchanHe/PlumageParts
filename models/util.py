# SPDX-License-Identifier: MIT
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, to_rgb, to_rgba
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import torch
import cv2
from skimage import measure


from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import NotFittedError

neon_palette = [
    "#69F6F6",  # Cyan
    "#FF43FF",  # Magenta
    "#5EFF00",  # Acid Lemon Green
    "#00BCFA",  # Deep Sky Blue
    "#FF4500",  # Orange Red
    "#6C0399",  # Dark Violet
    "#FFD700",  # Bright Gold (distinct from pale yellow)
    "#0CF83B",  # Spring Green
    "#1538FF",  # Dodger Blue
    "#FF1493",  # Deep Pink
    "#32CD32",  # Lime Green
    "#FFA500",  # Orange
    "#3E0376",  # Blue Violet
    "#DC143C",  # Crimson
    "#00CED1",  # Dark Turquoise
]

def apply_colormap(mask_tensor, num_classes):
    """
    Map an integer mask tensor of shape [B, H, W] to an RGB image tensor [B, 3, H, W].

    Args:
        mask_tensor: torch.Tensor, shape [B, H, W], integer class indices.
        num_classes: int, number of classes.

    Returns:
        torch.Tensor: shape [B, 3, H, W], uint8 RGB images.
    """
    # Get a matplotlib colormap (avoid too many classes to prevent color repetition)
    cmap = plt.get_cmap('tab20')  # 20 colors; alternatives: 'viridis', 'nipy_spectral'
    
    colored = []
    for mask in mask_tensor:  # mask: [H, W]
        mask_np = mask.cpu().numpy()
        color_np = cmap(mask_np / (num_classes - 1))[:, :, :3]  # shape: [H, W, 3]
        color_tensor = torch.from_numpy((color_np * 255).astype(np.uint8)).permute(2, 0, 1)  # [3, H, W]
        colored.append(color_tensor)

    return torch.stack(colored)  # [B, 3, H, W]


def overlay_mask_on_image(image, mask_rgb, alpha=0.5):
    """
    image: [3, H, W], float [0,1]
    mask_rgb: [3, H, W], uint8 [0,255]
    returns: overlayed [3, H, W], uint8
    """
    image = (image * 255).to(torch.uint8)
    overlay = (alpha * mask_rgb + (1 - alpha) * image).to(torch.uint8)
    return overlay

def get_edges(mask):
    # mask: 0/1 array
    edges = cv2.Canny(mask.astype(np.uint8)*255, 0, 1)
    return edges > 0


def visualize_segmentation(image, mask, label_map=None, alpha=0.3, cmap='tab20',
                           save_path=None,
                           class_colors=neon_palette,
                           edge_lw=2, edge_alpha=0.7, limit=512):
    """
    Visualizes a segmentation mask overlaid on an image, with optional class labels and legend.
    Args:
        image (np.ndarray): The input image as a NumPy array (H x W x 3), expected in the range [0, 255].
        mask (np.ndarray): The segmentation mask as a NumPy array (H x W), where each pixel value corresponds to a class label. Background should be labeled as 0.
        label_map (dict, optional): A dictionary mapping class indices to human-readable class names. If provided, a legend will be displayed.
        alpha (float, optional): The transparency factor for the mask overlay. Default is 0.5.
        cmap (str, optional): The name of the matplotlib colormap to use for coloring the mask. Default is 'tab20'.
        save_path (str, optional): If provided, the visualization will be saved to this path. Otherwise, the image will be displayed.
        class_colors (list or dict, optional): A list or dictionary specifying colors for each class. If a list is provided, its length should be at least the number of classes. 
        
    Returns:
        None
    """
    
    mask = mask.astype(int)
    unique_classes = np.unique(mask)
    unique_fg = unique_classes[unique_classes != 0]

    # Resize image and mask if either dimension exceeds the limit (e.g., 512)
    h, w = image.shape[:2]
    if max(h, w) > limit:
        scale = limit / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 1) Build colormap
    if class_colors is None:
        # Use standard colormap: default tab20
        base_cmap = plt.get_cmap(cmap, int(mask.max()) + 1)
        # colormap(mask) returns RGBA, we only use the first 3 channels
        color_img = base_cmap(mask)[:, :, :3]
        get_color_for_legend = lambda cls: base_cmap(cls)
    else:
        # Custom list/dict
        max_cls = int(mask.max())
        # Use default cmap as fallback
        fallback = plt.get_cmap(cmap, max_cls + 1)
        # Pre-fill: background color placeholder, will be filtered by mask_nonzero
        palette = [fallback(0)]  # index 0 won't be used
        # Assign color for each class
        for cls in range(1, max_cls + 1):
            if isinstance(class_colors, dict) and cls in class_colors:
                c = class_colors[cls]
            elif isinstance(class_colors, (list, tuple)) and (cls - 1) < len(class_colors):
                c = class_colors[cls - 1]
            else:
                c = fallback(cls)
            # Convert to RGBA
            rgba = to_rgba(c)
            palette.append(rgba)

        listed = ListedColormap(palette, N=max_cls + 1)
        color_img = listed(mask)[:, :, :3]
        get_color_for_legend = lambda cls: listed(cls)

    # 2) Overlay
    img_float = image.astype(np.float32) / 255.0
    mask_nonzero = (mask != 0)
    overlay = img_float.copy()
    overlay[mask_nonzero] = (1 - alpha) * img_float[mask_nonzero] + alpha * color_img[mask_nonzero]


    # overlay_uint8 = (overlay * 255).astype(np.uint8)

    # for cls in unique_fg:
    #     cls_mask = (mask == cls).astype(np.uint8)
        
    #     contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     color_rgba = get_color_for_legend(int(cls))
    #     color_bgr = [c * 255 for c in color_rgba[:3]][::-1] 

    #     cv2.drawContours(overlay_uint8, contours, -1, color_bgr, thickness=edge_lw)

    # 3) Visualization + legend
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay) 
    ax.axis('off')

    for cls in unique_fg:
        cls_mask = (mask == cls).astype(np.uint8)
        contours = measure.find_contours(cls_mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0],
                    color=get_color_for_legend(int(cls)),
                    linewidth=edge_lw,
                    alpha=edge_alpha)

    #legend
    if label_map is not None and len(unique_fg) > 0:
        legend_elements = []
        # iterate over label_map keys
        for cls in label_map.keys():
            if cls in unique_fg:   # Avoid classes not present in the image
                color = get_color_for_legend(int(cls))
                label = f"{cls}: {label_map[cls]}"  
                legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def overlay_mask_on_image_np(image_np, pred_mask_np, alpha=0.5, color_map=None):
    """
    Overlays a segmentation mask on an image using NumPy.

    Args:
        image_np (np.ndarray): The original image as a NumPy array (H, W, 3).
        pred_mask_np (np.ndarray): The predicted segmentation mask as a NumPy array (H, W).
                                   It should contain integer class labels (0, 1, 2, ...).
        alpha (float, optional): The transparency of the mask overlay. Defaults to 0.5.
        color_map (dict or np.ndarray, optional): A mapping from class labels to RGB colors.
                                                  If None, a default colormap will be used.
                                                  Example: {0: [0, 0, 0], 1: [255, 0, 0], ...}

    Returns:
        np.ndarray: The overlaid image as a NumPy array (H, W, 3) in 'uint8' format.
    """
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel RGB NumPy array of shape (H, W, 3).")
    if pred_mask_np.ndim != 2:
        raise ValueError("Input mask must be a single-channel NumPy array of shape (H, W).")

    # Ensure data types are suitable for operations
    image_np = image_np.astype(np.float32)
    pred_mask_np = pred_mask_np.astype(np.int32)
    
    # Get image dimensions
    h, w = image_np.shape[:2]
    
    # Create a blank RGB mask overlay
    mask_overlay = np.zeros((h, w, 3), dtype=np.float32)

    # If no colormap is provided, create a default one
    if color_map is None:
        num_classes = np.max(pred_mask_np) + 1
        # Create a list of colors (excluding black for background)
        # You can customize these colors
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255)]
        color_map = {i: c for i, c in enumerate(colors)}
        # Add black for the background class (0)
        color_map[0] = [0, 0, 0]

    # Apply colors to the mask overlay based on the class labels
    for class_id, color in color_map.items():
        # Get pixels that belong to this class
        class_pixels = (pred_mask_np == class_id)
        # Apply the color to those pixels
        mask_overlay[class_pixels] = color

    # Blend the image and the mask overlay
    # This is the core blending operation
    overlaid_image = (alpha * mask_overlay + (1 - alpha) * image_np)

    # Convert the result back to 'uint8' format
    overlaid_image = overlaid_image.astype(np.uint8)

    return overlaid_image


def build_augmentation_from_config(aug_list, resize_height=None, resize_width=None):
    import inspect
    import albumentations as A

    aug_objs = []

    for aug in aug_list:
        name = aug['name']
        params = dict(aug.get('params', {}))

        # check user_define_height/width
        if any(v in ['user_define_height', 'user_define_width'] for v in params.values()):
            if resize_height is None or resize_width is None:
                raise ValueError(f"Transform {name} uses user_define_* but resize_height/width not provided")

            for key in list(params.keys()):
                if params[key] == 'user_define_height':
                    params[key] = resize_height
                elif params[key] == 'user_define_width':
                    params[key] = resize_width

        # Compatibility check
        if not hasattr(A, name):
            raise ValueError(f"Albumentations has no transform named {name}")
        aug_class = getattr(A, name)
        if name == "PadIfNeeded":
            sig_params = inspect.signature(aug_class.__init__).parameters
            if "fill" in sig_params:
                if "value" in params and "fill" not in params:
                    params["fill"] = params.pop("value")
                else:
                    params.pop("value", None)
            elif "value" in sig_params:
                if "fill" in params and "value" not in params:
                    params["value"] = params.pop("fill")
                else:
                    params.pop("fill", None)

            if "fill_mask" in sig_params:
                if "mask_value" in params and "fill_mask" not in params:
                    params["fill_mask"] = params.pop("mask_value")
                elif "masks_value" in params and "fill_mask" not in params:
                    params["fill_mask"] = params.pop("masks_value")
                else:
                    params.pop("mask_value", None)
                    params.pop("masks_value", None)
            elif "masks_value" in sig_params:
                if "fill_mask" in params and "masks_value" not in params:
                    params["masks_value"] = params.pop("fill_mask")
                elif "mask_value" in params and "masks_value" not in params:
                    params["masks_value"] = params.pop("mask_value")
                else:
                    params.pop("fill_mask", None)
                    params.pop("mask_value", None)
            elif "mask_value" in sig_params:
                if "fill_mask" in params and "mask_value" not in params:
                    params["mask_value"] = params.pop("fill_mask")
                elif "masks_value" in params and "mask_value" not in params:
                    params["mask_value"] = params.pop("masks_value")
                else:
                    params.pop("fill_mask", None)
                    params.pop("masks_value", None)
        aug_objs.append(aug_class(**params))

    return A.Compose(aug_objs, additional_targets={'mask': 'mask'})


def denormalize_tensor(tensor, mean, std):
    """
    Denormalizes a PyTorch tensor.
    Args:
        tensor (torch.Tensor): The normalized tensor (C, H, W).
        mean (tuple or list): The mean values.
        std (tuple or list): The standard deviation values.
    Returns:
        torch.Tensor: The denormalized tensor (C, H, W).
    """
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    denormalized = tensor * std + mean
    # Scale back to 0-255 and clamp values
    denormalized = denormalized.clamp(0, 1) * 255.0
    return denormalized.to(torch.uint8)

import datetime
import os

def get_run_name(args):
    dataset_name = os.path.basename(os.path.normpath(args.img_dir))
    date_str = datetime.datetime.now().strftime('%Y%m%d')

    # if args has model
    if hasattr(args, 'model'):
        model_name = args.model
        name_parts = [
            model_name,
            f"img{dataset_name}",
            f"size{args.resize}",
            f"cls{args.num_classes}",
            f"bs{args.batch_size}",
            f"ep{args.epochs}",
            f"lr{args.lr:.0e}"
        ]
    # if args has model_type
    elif hasattr(args, 'model_type'):
        # it's sam
        model_name = args.model_type
        name_parts = [
            model_name,
            f"img{dataset_name}",
            f"cls{args.num_classes}",
            f"bs{args.batch_size}",
            f"ep{args.epochs}",
            f"lr{args.lr:.0e}"
        ]

    if getattr(args, "aug", False):
        name_parts.append("aug")
    name_parts.append(date_str)
    
    return "_".join(name_parts)


import torch.nn as nn


class DiceCELoss(nn.Module):
    def __init__(self, weight=None, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

    def dice(self, inputs, targets, eps=1e-6):
        num_classes = inputs.shape[1]
        inputs = torch.softmax(inputs, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(inputs * targets_onehot, dims)
        union = torch.sum(inputs + targets_onehot, dims)
        dice = (2. * intersection + eps) / (union + eps)
        return 1 - dice.mean()


def compute_dice(preds, targets, num_classes, per_class=False):
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    dice_scores = []
    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls

        intersection = (pred_inds & target_inds).sum().item()
        pred_sum = pred_inds.sum().item()
        target_sum = target_inds.sum().item()

        denom = pred_sum + target_sum
        dice = round((2. * intersection / denom) if denom > 0 else 1.0, 4)  # round to 4 decimal places
        dice_scores.append(dice)

    if per_class:
        return dice_scores
    else:
        return sum(dice_scores) / len(dice_scores)


def accumulate_per_class_stats(preds, targets, num_classes, stat_dict):
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        intersection = (pred_inds & target_inds).sum().item()
        pred_sum = pred_inds.sum().item()
        target_sum = target_inds.sum().item()

        stat_dict[cls]['intersection'] += intersection
        stat_dict[cls]['pred'] += pred_sum
        stat_dict[cls]['target'] += target_sum


def compute_dice_from_stats(stat_dict):
    dice_scores = []
    for cls, v in stat_dict.items():
        denom = v['pred'] + v['target']
        if denom == 0:
            dice = 1.0  # skip empty class
        else:
            dice = 2 * v['intersection'] / denom
        dice_scores.append(dice)
    return dice_scores


def compute_metrics_from_stats(stat_dict):
    dice_scores = []
    iou_scores = []

    class_ids = list(stat_dict.keys())

    for cls in class_ids:
        v = stat_dict[cls]

        tp = v['intersection']
        pred = v['pred']           # TP + FP
        target = v['target']       # TP + FN

        # ---- Dice ----
        denom_dice = pred + target
        if denom_dice > 0:
            dice = 2 * tp / denom_dice
            dice_scores.append(dice)


        # ---- IoU ----
        union = pred + target - tp  # = TP + FP + FN
        if union > 0:
            iou = tp / union
            iou_scores.append(iou)


    # macro averaging
    mean_dice = sum(dice_scores) / len(dice_scores)
    mean_iou = sum(iou_scores) / len(iou_scores)

    return dice_scores, mean_dice, iou_scores, mean_iou



def compute_custom_class_weight(y_true, all_possible_classes, handle_missing=True, missing_weight=1e-6):
    """
    Compute class weights, gracefully handling classes that do not appear in the dataset.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Array of true labels for all samples in the dataset.

    all_possible_classes : array-like
        An array containing all theoretically possible classes, even if some classes do not appear in y_true.
        For example [0, 1, 2, 3].

    handle_missing : bool, optional (default=True)
        A flag.
        - If True, assigns missing_weight to classes not present in y_true.
        - If False, behaves like sklearn: if any class in all_possible_classes is not present in y_true, raises an error.

    missing_weight : float, optional (default=1e-6)
        The weight assigned to missing classes when handle_missing=True.

    Returns
    -------
    class_weight_vect : ndarray of shape (n_classes,)
        An array of weights, in the same order as all_possible_classes.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    all_possible_classes = np.asarray(all_possible_classes)

    # 1. Get the classes actually present in the dataset
    present_classes = np.unique(y_true)

    # 2. Basic check: labels in y_true must not exceed the range of all_possible_classes
    if not np.all(np.isin(present_classes, all_possible_classes)):
        raise ValueError("y_true contains labels that are not in all_possible_classes.")

    # 3. Core logic
    # Initialize the final weights dict: key is class, value is weight
    final_weights = {}

    if handle_missing:
        # --- Logic for handling missing classes ---
        # a. Compute 'balanced' weights only for classes that actually appear
        balanced_weights = compute_class_weight(
            class_weight='balanced',
            classes=present_classes,
            y=y_true
        )
        present_weight_map = dict(zip(present_classes, balanced_weights))

        weight_array = np.array([
            present_weight_map.get(cls, missing_weight) for cls in all_possible_classes
        ])
        return weight_array

    
    else:
        # --- Logic consistent with sklearn behavior ---
        # Check if all_possible_classes is a subset of present_classes
        if not np.all(np.isin(all_possible_classes, present_classes)):
            missing = np.setdiff1d(all_possible_classes, present_classes)
            raise ValueError(f"The classes {list(missing)} were specified in all_possible_classes but not found in y_true. "
                             "Set handle_missing=True to assign them a default weight.")
        
        # If all classes are present, compute directly
        return compute_class_weight(
            class_weight='balanced',
            classes=all_possible_classes,
            y=y_true
        )


def balanced_class_weight_from_counts(counts, missing_weight=1e-6):
    """
    Given an array of class counts, compute balanced class weights.
    For classes with zero counts, assign them the specified missing_weight.
    """
    counts = torch.as_tensor(counts, dtype=torch.double)
    C = counts.numel()
    N = counts.sum().item()

    
    weights = torch.empty_like(counts, dtype=torch.double)
    mask_zero = counts == 0
    weights[~mask_zero] = N / (C * counts[~mask_zero])
    weights[mask_zero] = float(missing_weight)

    return weights.numpy()  
