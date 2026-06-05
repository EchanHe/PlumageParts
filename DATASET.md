# PlumageParts Dataset

This file describes the expected local layout for the PlumageParts annotations,
predictions and external benchmark files. The dataset and trained checkpoints can be found in <TODO>.


## Main Dataset

The main dataset contains 4,705 high-resolution bird specimen images annotated
with fine-grained plumage regions.

```text
plumageparts_dataset/
  train/
    img/
    masks/
  val/
    img/
    masks/
  test/
    img/
    masks/
```

Images are JPG. Masks are single-channel PNG files. By default, masks use the
same filename stem as the corresponding image, with no extra suffix; for example:

```text
img/ABC123.jpg
masks/ABC123.png
```

If your masks include a suffix such as `ABC123_mask.png`, pass
`--mask_suffix _mask` to the training or inference scripts.




## Mask IDs

| ID | Region |
| -- | ------ |
| 0 | Background |
| 1 | Head |
| 2 | Back |
| 3 | Tail |
| 4 | Throat |
| 5 | Breast |
| 6 | Belly |
| 7 | Vent |
| 8 | Coverts |
| 9 | Remiges |

Use `--num_classes 10` for PlumageParts training and inference.

## Predictions

Predicted masks from the best model can be stored as:

```text
plumageparts_test_prediction/
  masks/
  overlay/
```

## Model Checkpoints

The Zenodo release is expected to include two segmentation checkpoints:

| Checkpoint | Description |
| -- | -- |
| `plumageparts_best.pth` | Best PlumageParts model: DINOv3 ViT-H+/16 (`vith16plus`) encoder with a Multi-Stage Upsampling (MSU) decoder, trained/evaluated at 1024 x 1024 with 10 classes including background. |
| `partimagenet_best.pth` | Best PartImageNet benchmark model: DINOv3 ViT-H+/16 (`vith16plus`) encoder with a Multi-Layer Fusion (MLF) decoder, trained/evaluated at 512 x 512 with 41 classes including background. |

These checkpoints store the segmentation model weights used by this repository.
Inference also requires the matching DINOv3 backbone weights, supplied separately
with `--dinov3_weights`.

## External Benchmarks

External datasets should be obtained from their original sources and used under
their own licenses.

```text
cub_masks/
  gt_cub_parts_points.csv
  pred_masks/

partimagenet_bird/
  images/
  gt_masks/
  pred_masks/
```

Original sources:

- CUB-200-2011: https://www.vision.caltech.edu/datasets/cub_200_2011/
- PartImageNet: https://github.com/tacju/partimagenet


