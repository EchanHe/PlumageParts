# SPDX-License-Identifier: MIT
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import inspect
try:
    import tifffile
except ImportError:
    print("tifffile is not installed. Please install it to handle TIFF files.")


def make_pad_if_needed(
    min_height: int,
    min_width: int,
    border_mode: int = cv2.BORDER_CONSTANT,
    image_fill=0,
    mask_fill=0,
    **extra
):
    """
    Return A.PadIfNeeded with kwargs adapted to the installed Albumentations version.
    - Newer versions expect: fill, fill_mask
    - Older versions may expect: value, mask_value or masks_value
    """
    sig = inspect.signature(A.PadIfNeeded.__init__)
    params = sig.parameters

    kwargs = dict(min_height=min_height, min_width=min_width, border_mode=border_mode, **extra)

    # image fill
    if "fill" in params:
        kwargs["fill"] = image_fill
    elif "value" in params:
        kwargs["value"] = image_fill

    # mask fill
    if "fill_mask" in params:
        kwargs["fill_mask"] = mask_fill
    elif "masks_value" in params:
        kwargs["masks_value"] = mask_fill
    elif "mask_value" in params:
        kwargs["mask_value"] = mask_fill

    return A.PadIfNeeded(**kwargs)



class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, album_aug = None, transform=None, mask_transform=None, file_list=None, mask_suffix=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # self.img_files = sorted(os.listdir(img_dir))
        # self.mask_files = sorted(os.listdir(mask_dir))
        self.mask_suffix = mask_suffix
        
        self.album_aug = album_aug
        self.transform = transform
        self.mask_transform = mask_transform

        if file_list is not None:
            self.image_files = file_list
        else:
            self.image_files = sorted(os.listdir(self.img_dir))  # assumes all are images
            self.image_files = [f for f in self.image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        assert len(self.image_files) > 0, "No images found!"

        # do assertitons for either albumentations or torchvision transforms
        if self.album_aug is not None:
            assert self.transform is None, "If using albumentations, do not use torchvision transforms"
            assert self.mask_transform is None, "If using albumentations, do not use torchvision mask transforms"
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        has_mask = self.mask_dir is not None
        if has_mask:
            if self.mask_suffix is not None:
                mask_name = os.path.splitext(img_name)[0] + self.mask_suffix + ".png"
            else:
                mask_name = img_name
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = np.array(Image.open(mask_path).convert("L"))

        if self.album_aug:
            if has_mask:
                augmented = self.album_aug(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.album_aug(image=image)
                image = augmented['image']
                height, width = image.shape[:2] if not torch.is_tensor(image) else image.shape[1:3]
                mask = torch.zeros(height, width, dtype=torch.long)
        else:
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            
            if has_mask:
                if self.mask_transform:
                    mask = self.mask_transform(mask)
                else:
                    mask = torch.from_numpy(mask).long()
            else:
                mask = None
                height, width = image.shape[1:3]
                mask = torch.zeros(height, width, dtype=torch.long)

        return image, mask, img_name

class PredictEvalDataset(torch.utils.data.Dataset):
    """
    For prediction/evaluation only:
    - No random augmentation, only deterministic preprocessing (PadIfNeeded/Resize/Normalize/ToTensorV2)
    - Returns original image/original mask (if available)
    - Returns meta info needed to restore predictions to original image size
    """
    def __init__(self, img_dir, mask_dir=None, file_list=None,
                 resize=512, pad_if_needed=True, normalize=True,
                 mask_suffix=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix

        self.files = file_list if file_list is not None else sorted(os.listdir(img_dir))
        # only for files with image extensions
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        assert len(self.files) > 0, "No images found."


        tfs = []

            
        tfs.append(A.LongestMaxSize(max_size=resize))
        tfs.append(make_pad_if_needed(min_height=resize, min_width=resize,
                                    border_mode=0, image_fill=0, mask_fill=0))

        if normalize:
            tfs.append(A.Normalize(mean=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225)))
        tfs.append(ToTensorV2())
        # self.preprocess = A.Compose(tfs, additional_targets={'mask': 'mask'})
        self.preprocess = A.ReplayCompose(tfs, additional_targets={'mask': 'mask'}) 

    def __len__(self): return len(self.files)

    def _mask_name(self, img_name):
        if self.mask_dir is None: return None
        if self.mask_suffix is not None:
            base = os.path.splitext(img_name)[0]
            return f"{base}{self.mask_suffix}.png"
        return img_name

    def __getitem__(self, idx):
        fname = self.files[idx]

        img_path = os.path.join(self.img_dir, fname)
        image_np = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image_np.shape[:2]

        mask_np = None
        mname = self._mask_name(fname)
        if mname is not None:
            mpath = os.path.join(self.mask_dir, mname)
            if os.path.exists(mpath):
                mask_np = np.array(Image.open(mpath).convert("L"))
            else:
                mask_np = None  

        # preprocess
        if mask_np is not None:
            aug = self.preprocess(image=image_np, mask=mask_np)
            image_t = aug["image"]          # [3,H,W] float32
            mask_t  = aug["mask"]           # [H,W]  
            replay = aug["replay"]  # for replaying the augmentation
        else:
            aug = self.preprocess(image=image_np)
            image_t = aug["image"]
            mask_t = None
            replay = aug["replay"]  # for replaying the augmentation


        meta = {
            "orig_size": (orig_h, orig_w),
            "fname": fname,
            "replay": replay
        }

        return image_t,  (mask_t if mask_t is not None else None), image_np, mask_np, meta

import torch.nn.functional as F


def restore_pred_to_orig_replay(pred_logits_or_labels, meta, is_logits=True):
    """
    pred_logits_or_labels:
        logits: [B, C, H, W] or labels: [B, H, W]
    meta: dict, containing:
        - "orig_size": (orig_h, orig_w)
        - "replay": dict produced by ReplayCompose
    Returns: numpy labels, shape [B, orig_h, orig_w]
    """
    orig_h, orig_w = meta["orig_size"]
    replay = meta["replay"]
    tfms = replay["transforms"]

    # default padding
    pad_top = pad_bottom = pad_left = pad_right = 0

    # parse replay transforms to find padding
    for t in tfms:
        if t["__class_fullname__"].endswith("PadIfNeeded"):
            params = t["params"]
            # get padding values
            pad_top    = int(params.get("pad_top", 0))
            pad_bottom = int(params.get("pad_bottom", 0))
            pad_left   = int(params.get("pad_left", 0))
            pad_right  = int(params.get("pad_right", 0))
            break

    padded_h = orig_h + pad_top + pad_bottom
    padded_w = orig_w + pad_left + pad_right

    # 1) reverse the replay transforms
    if is_logits:
        # logits
        up = F.interpolate(pred_logits_or_labels, size=(padded_h, padded_w),
                           mode="bilinear", align_corners=False)
        labels = up.argmax(dim=1)   # [B, padded_h, padded_w]
    else:
        # labels
        if pred_logits_or_labels.ndim == 2:  # [H, W] -> [1, H, W]
            pred_logits_or_labels = pred_logits_or_labels.unsqueeze(0)
        labels = F.interpolate(pred_logits_or_labels.unsqueeze(1).float(),
                               size=(padded_h, padded_w),
                               mode="nearest").squeeze(1).long()

    # 2) remove padding
    labels = labels[:, pad_top:pad_top+orig_h, pad_left:pad_left+orig_w]  # [B, orig_h, orig_w]

    return labels.cpu().numpy()




def restore_pred_to_orig_replay_lmspad(pred_logits_or_labels, meta, is_logits=True):
    """
    Adapted for Albumentations: LongestMaxSize -> PadIfNeeded
    pred_logits_or_labels:
        - logits: [B, C, Hf, Wf]  (Hf=Wf=final network input size, e.g. 512)
        - labels: [B, Hf, Wf]     (already argmaxed integer labels)
    meta: dict, contains
        - "orig_size": (orig_h, orig_w)  original image size
        - "replay": dict produced by ReplayCompose (contains PadIfNeeded's pad_top/bottom/left/right)
    Returns:
        numpy labels, shape [B, orig_h, orig_w]
    """
    orig_h, orig_w = meta["orig_size"]
    replay = meta["replay"]
    tfms = replay["transforms"]

    # Default: no padding
    pad_top = pad_bottom = pad_left = pad_right = 0

    # Parse replay to get actual padding from PadIfNeeded
    for t in tfms:
        if t.get("__class_fullname__", "").endswith("PadIfNeeded"):
            params = t.get("params", {})
            pad_top    = int(params.get("pad_top", 0))
            pad_bottom = int(params.get("pad_bottom", 0))
            pad_left   = int(params.get("pad_left", 0))
            pad_right  = int(params.get("pad_right", 0))
            break

    # Final network input size (after padding, usually 512x512)
    if is_logits:
        # First crop out padding in logits space, then bilinear upsample to original size, then argmax
        cropped = pred_logits_or_labels[
            :,
            :,
            pad_top: pred_logits_or_labels.shape[2] - pad_bottom if pad_bottom > 0 else pred_logits_or_labels.shape[2],
            pad_left: pred_logits_or_labels.shape[3] - pad_right  if pad_right  > 0 else pred_logits_or_labels.shape[3]
        ]
        up = F.interpolate(cropped, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        labels = up.argmax(dim=1)  # [B, orig_h, orig_w]
    else:
        # For integer labels: crop padding, then nearest upsample
        if pred_logits_or_labels.ndim == 2:
            pred_logits_or_labels = pred_logits_or_labels.unsqueeze(0)  # [1,Hf,Wf]
        cropped = pred_logits_or_labels[
            :,
            pad_top: pred_logits_or_labels.shape[-2] - pad_bottom if pad_bottom > 0 else pred_logits_or_labels.shape[-2],
            pad_left: pred_logits_or_labels.shape[-1] - pad_right  if pad_right  > 0 else pred_logits_or_labels.shape[-1]
        ]  # [B, Hs, Ws]
        labels = F.interpolate(cropped.unsqueeze(1).float(),
                               size=(orig_h, orig_w), mode="nearest").squeeze(1).long()

    return labels.cpu().numpy()


def restore_probs_to_orig_replay_lmspad(logits, meta):
    """
    logits: [B, C, Hf, Wf]  (not softmaxed)
    Returns: probs_np: [B, C, Horig, Worig] float32
    Procedure: crop padding according to replay -> bilinear interpolate to original image size -> softmax
    """
    orig_h, orig_w = meta["orig_size"]
    tfms = meta["replay"]["transforms"]

    pad_top = pad_bottom = pad_left = pad_right = 0
    for t in tfms:
        if t.get("__class_fullname__", "").endswith("PadIfNeeded"):
            p = t.get("params", {})
            pad_top    = int(p.get("pad_top", 0))
            pad_bottom = int(p.get("pad_bottom", 0))
            pad_left   = int(p.get("pad_left", 0))
            pad_right  = int(p.get("pad_right", 0))
            break

    # Crop padding (in logits space)
    cropped = logits[
        :,
        :,
        pad_top : logits.shape[2] - pad_bottom if pad_bottom > 0 else logits.shape[2],
        pad_left: logits.shape[3] - pad_right  if pad_right  > 0 else logits.shape[3]
    ]
    # Resize to original image size
    up = F.interpolate(cropped, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    probs = torch.softmax(up, dim=1)  # [B, C, Horig, Worig]
    return probs.cpu().numpy().astype(np.float32)


def restore_pred_to_orig_simple(pred_logits_or_labels, meta, is_logits=False, num_classes=None):
    """
    Restore predictions to original image size by resizing only
    """
    orig_h, orig_w = meta["orig_size"]
    if is_logits:

        up = F.interpolate(pred_logits_or_labels, size=(orig_h, orig_w),
                           mode="bilinear", align_corners=False)
        labels = up.argmax(dim=1)  # [B,H_orig,W_orig]
    else:
        
        labels = F.interpolate(pred_logits_or_labels.unsqueeze(1).float(),
                               size=(orig_h, orig_w), mode="nearest").squeeze(1).long()
    return labels.cpu().numpy()

def cls_weights(train_dataset):
    all_labels = []

    for i in range(len(train_dataset)):
        _, mask ,_ = train_dataset[i]  # return image, mask
        all_labels.append(mask.flatten())  # flat to 1d

    # Combine into a numpy array
    all_labels_flat = torch.cat(all_labels).numpy()

    return all_labels_flat

@torch.no_grad()
def compute_class_counts_loader(loader, num_classes, ignore_index=None, downsample_stride=None):

    counts = torch.zeros(num_classes, dtype=torch.long)

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            masks = batch[1]
        else:
            masks = batch['mask']

        if not torch.is_tensor(masks):
            masks = torch.as_tensor(masks)

        if masks.ndim == 2:
            masks = masks.unsqueeze(0)  # [1,H,W]

        if downsample_stride and downsample_stride > 1:
            masks = masks[:, ::downsample_stride, ::downsample_stride]

        if ignore_index is not None:
            valid = (masks != ignore_index)
            if valid.any():
                vals = masks[valid].to(torch.long)
            else:
                continue
        else:
            vals = masks.to(torch.long)

        local = torch.bincount(vals.flatten(), minlength=num_classes)
        if local.numel() > num_classes:
            local = local[:num_classes]
        counts += local.to(counts.dtype)

    return counts

def save_probs_npz(probs_np, save_path_npz):
    """
    probs_np: [C, H, W] float32
    Save as compressed npz (most universal, lossless)
    """
    np.savez_compressed(save_path_npz, probs=probs_np)

def save_probs_tiff16(probs_np, save_path_tif):
    """
    probs_np: [C, H, W] float32 in [0,1]
    Save as 16-bit multi-page TIFF (one page per class), for interoperability with GIS/scientific tools
    """
    vol = (np.clip(probs_np, 0, 1) * 65535.0).astype(np.uint16)  # [C,H,W]

    
    tifffile.imwrite(save_path_tif, vol)  # [C,H,W]



def save_probs_tiff_float(probs_np, save_path_tif):
    """
    probs_np: [C, H, W] float16 in [0,1]
    Save as float16 multi-page TIFF (one page per class)
    """
    probs_np = np.clip(probs_np, 0, 1).astype(np.float16)
    tifffile.imwrite(save_path_tif, probs_np)  # [C,H,W]

def save_probs_tiff_int(probs_np, save_path_tif):
    """
    probs_np: [C, H, W] float32 in [0,1]
    Save as uint8 multi-page TIFF (one page per class), scaled to [0,100]
    """
    vol = (np.clip(probs_np, 0, 1) * 100).round().astype(np.uint8)  # [C,H,W], values 0-100
    tifffile.imwrite(save_path_tif, vol)


def save_class_heatmap_png(prob_np, save_path_png):
    """
    prob_np: [H, W] float32 in [0,1]
    Save color heatmap (jet colormap), for quick visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    plt.figure(figsize=(6,6))
    plt.imshow(prob_np, cmap='jet', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    plt.close()

def compute_entropy_map(probs_np):
    """
    probs_np: [C, H, W] in [0,1]
    Return pixel-wise uncertainty entropy map [H, W]
    """
    eps = 1e-8
    p = np.clip(probs_np, eps, 1.0)
    ent = -np.sum(p * np.log(p), axis=0)  # [H,W]
    # Normalize to [0,1] (maximum entropy is log(C))
    return (ent / np.log(probs_np.shape[0])).astype(np.float32)
