# PlumageParts Dataset

This file describes the PlumageParts annotation release and the expected local
layout for training or evaluation. The PlumageParts annotation masks, metadata,
train/validation/test split information, prediction masks and trained model
checkpoints have been deposited on Zenodo under the reserved DOI:
[doi.org/10.5281/zenodo.21254480](https://doi.org/10.5281/zenodo.21254480).
The Zenodo record is available to editors and reviewers via a private preview
link during peer review and will be made publicly available upon publication.


## Main Dataset

The main annotation set contains segmentation masks for 4,705
bird images annotated with fine-grained plumage regions. Original
source images are not redistributed in the PlumageParts release.

The PlumageParts source images were selected from the iRateBirds Citizen Science
Project, which is based on bird photographs from the Macaulay Library. Original
source images remain subject to the applicable Macaulay Library source-image
licence and reuse terms. To support reproducibility, the released metadata
includes the corresponding Macaulay Library record information, and
`download_dataset.py` helps users retrieve the source images and reconstruct the
dataset locally, subject to those terms.

- The iRateBirds Citizen Science Project: a Dataset on Birds' Visual Aesthetic
  Attractiveness to Humans
- https://www.nature.com/articles/s41597-023-02169-0
- Figshare for iRateBirds data: [link](https://figshare.com/articles/dataset/The_iRateBirds_Citizen_Science_Project_a_Dataset_on_Birds_Visual_Aesthetic_Attractiveness_to_Humans/20170082)


The filename stem follows the format:

`<scientific_name>_<sex>_<macaulay_photo_catalog_id>`

For example:

`Acanthis_cabaret_Female_2043824211`

The released masks use this filename stem. Masks are stored as single-channel
PNG files:

`Acanthis_cabaret_Female_2043824211.png`

If users obtain the corresponding source image locally, the image should use the
same filename stem, usually as a JPG file:

`Acanthis_cabaret_Female_2043824211.jpg`

For local training or evaluation, place downloaded source images and released
masks in matching split folders.


```text
plumageparts_dataset/
  train/
    img/      # source images obtained by the user
    masks/    # released PlumageParts masks
  val/
    img/
    masks/
  test/
    img/
    masks/
```


Note: Original source images are not redistributed in this release.



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

Predicted masks from the best model are stored as:

```text
plumageparts_test_prediction/
  masks/
```

## Model Checkpoints

The Zenodo release includes two segmentation checkpoints:

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
  images/       # not redistributed by PlumageParts
  gt_masks/
  pred_masks/
```

Original sources:
- iRateBirds figshare [link](https://figshare.com/articles/dataset/The_iRateBirds_Citizen_Science_Project_a_Dataset_on_Birds_Visual_Aesthetic_Attractiveness_to_Humans/20170082)
- Macaulay Library: https://www.macaulaylibrary.org/
- CUB-200-2011: https://www.vision.caltech.edu/datasets/cub_200_2011/
- PartImageNet: https://github.com/tacju/partimagenet
