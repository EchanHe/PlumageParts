# PlumageParts
*Fine-grained plumage region segmentation for bird images and video.*

**Manuscript:** PlumageParts: A fine-grained avian plumage segmentation dataset and benchmark for ecological image analysis
**Authors:** Yichen He, Eleftherios Ioannou, Kathryn Harris, Gavin Thomas, Steve Maddock, Julien P. Renoult, Christopher Cooney

PlumageParts contains the code used for training, evaluating and applying the
models described in our manuscript on biologically meaningful avian plumage
patches. The main annotation set provides masks for 4,705 high-resolution bird
specimen images, covering nine regions: head, throat, breast, belly, vent, back,
coverts, remiges and tail. Original source images are not redistributed.



## Repository Structure

```text
PlumageParts/
  dataset/                  Dataset loaders and prediction restoration helpers
  models/                   DINOv3, DINOv2, SAM and classical segmentation models
  models/augs/              Albumentations augmentation configs
  models/seg_label/         Label maps for PlumageParts and PartImageNet
  video/                    Detect-track-segment pipeline modules
  train.py                  DINOv3/DINOv2 training
  train_sam.py              SAM-based training
  train_classic.py          UNet/DeepLab baselines
  pred.py                   Image-level DINOv3 inference and evaluation
  video_pred.py             Video or image-sequence detect-track-segment inference
```

## Installation

We tested the code with Python 3.10+. A CUDA-enabled PyTorch install is
recommended for training and high-resolution inference.

```bash
pip install -r requirements.txt
```

Install PyTorch using the command recommended for your CUDA version at
https://pytorch.org/get-started/locally/ if the default `pip install` does not
match your system.

Optional model families require their own upstream packages and weights:

- DINOv3: https://github.com/facebookresearch/dinov3
- DINOv2: https://github.com/facebookresearch/dinov2
- Segment Anything: https://github.com/facebookresearch/segment-anything
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- Ultralytics YOLO: `pip install ultralytics`

## Data and Checkpoints

See [DATASET.md](DATASET.md) for the detailed dataset information.

Data and checkpoints can be found in [doi.org/10.5281/zenodo.20551408](https://doi.org/10.5281/zenodo.20551408).

The PlumageParts data release contains segmentation masks, split information,
metadata and model checkpoints. It does not include the original source images.

The PlumageParts source images were selected from the iRateBirds project, whose photographs are sourced from the Macaulay Library. Users should
obtain the corresponding images from the original Macaulay Library records and
follow the applicable source-image licence/reuse terms.



Released segmentation checkpoints:

| Checkpoint | Dataset/task | Architecture | DINOv3 backbone | Input size | Classes |
| -- | -- | -- | -- | -- | -- |
| `plumageparts_best.pth` | PlumageParts plumage-patch segmentation | DINOv3 + Multi-Stage Upsampling (MSU) | ViT-H+/16 (`vith16plus`) | 1024 x 1024 | 10 |
| `partimagenet_best.pth` | PartImageNet part-segmentation benchmark | DINOv3 + Multi-Layer Fusion (MLF) | ViT-H+/16 (`vith16plus`) | 512 x 512 | 41 |

These files are segmentation-head checkpoints. Inference also requires the
corresponding DINOv3 backbone weights, passed with `--dinov3_weights`.

## Label Scheme

Use `--num_classes 10` for PlumageParts models because class 0 is background and
classes 1-9 are plumage patches.

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

The same mapping is stored in [`models/seg_label/PlumageParts.json`](models/seg_label/PlumageParts.json) for
visualisation.

## Image Inference

`pred.py` runs inference with trained DINOv3 MSU or MLF models and can also
compute Dice scores when ground-truth masks are supplied.

PlumageParts best checkpoint:

```bash
python pred.py \
  --img_dir /path/to/images \
  --model_path /path/to/weights/plumageparts_best.pth \
  --output_dir ./predictions_dinov3 \
  --model dinov3_msu \
  --num_classes 10 \
  --dinov3_weights /path/to/dinov3_backbone_weights.pth \
  --variant vith16plus \
  --enhanced_decoder \
  --output_size 1024 1024 \
  --label_map models/seg_label/PlumageParts.json \
  --save_overlay
```

PartImageNet benchmark checkpoint:

```bash
python pred.py \
  --img_dir /path/to/partimagenet/images \
  --model_path /path/to/weights/partimagenet_best.pth \
  --output_dir ./predictions_partimagenet \
  --model dinov3_mlf \
  --num_classes 41 \
  --dinov3_weights /path/to/dinov3_backbone_weights.pth \
  --variant vith16plus \
  --output_size 512 512 \
  --label_map models/seg_label/partimagenet.json \
  --background_class 40 \
  --save_overlay
```

Add `--mask_dir /path/to/masks` to compute per-image Dice scores and write CSV
summaries to the output directory. The Dice summary excludes `--background_class`
from the mean; this defaults to 0 for PlumageParts and should be set to 40 for
PartImageNet.

Outputs:

- `masks/`: predicted label masks as PNG files
- `overlay/`: visual overlays when `--save_overlay` is set
- `probs/`: optional 16-bit TIFF probability volumes with `--save_probs`
- `entropy/`: optional entropy maps with `--save_entropy`

## DINOv3 and DINOv2 Training

`train.py` trains MSU or MLF decoders on top of frozen or fine-tuned DINOv3/DINOv2
backbones.

```bash
python train.py \
  --img_dir /path/to/plumageparts_dataset/train/img \
  --mask_dir /path/to/plumageparts_dataset/train/masks \
  --val_img_dir /path/to/plumageparts_dataset/val/img \
  --val_mask_dir /path/to/plumageparts_dataset/val/masks \
  --test_img_dir /path/to/plumageparts_dataset/test/img \
  --test_mask_dir /path/to/plumageparts_dataset/test/masks \
  --model dinov3_msu \
  --dinov3_weights /path/to/dinov3_backbone_weights.pth \
  --variant vith16plus \
  --enhanced_decoder \
  --num_classes 10 \
  --output_size 1024 1024 \
  --batch_size 2 \
  --epochs 50 \
  --lr 1e-3 \
  --log_dir ./runs/dinov3_msu
```

Useful arguments:

- `--model`: `dinov3_msu`, `dinov3_mlf`, `dinov2_msu` or `dinov2_mlf`
- `--full_train`: train the backbone as well as the decoder
- `--weighted_loss`: use class-frequency-based loss weights
- `--loss_fn`: `cross_entropy` or `DiceCE`
- `--aug_config`: augmentation JSON, for example `models/augs/default.json`
- `--mask_suffix`: suffix appended to the image stem when finding masks. The
  default is an empty string, so `image.jpg` maps to `image.png`. Use
  `--mask_suffix _mask` if masks are named like `image_mask.png`.
- `--amp`: enable mixed-precision training on CUDA

Checkpoints are saved under `--log_dir` as `best_loss_model.pth`,
`best_dice_model.pth`, `best_iou_model.pth` and `last_model.pth`.

## SAM and Classical Baselines

SAM-based training:

```bash
python train_sam.py \
  --img_dir /path/to/train/img \
  --mask_dir /path/to/train/masks \
  --val_img_dir /path/to/val/img \
  --val_mask_dir /path/to/val/masks \
  --sam_checkpoint /path/to/sam_vit_b.pth \
  --model_type vit_b \
  --backend sam \
  --num_classes 10 \
  --output_size 1024 1024 \
  --batch_size 2 \
  --epochs 50 \
  --lr 1e-3 \
  --log_dir ./runs/sam_train
```

Classical baselines:

```bash
python train_classic.py \
  --img_dir /path/to/train/img \
  --mask_dir /path/to/train/masks \
  --val_img_dir /path/to/val/img \
  --val_mask_dir /path/to/val/masks \
  --model deeplab_resnet50 \
  --num_classes 10 \
  --resize 512 \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-3 \
  --optimizer adamw \
  --log_dir ./runs/classic
```

## Video Pipeline

`video_pred.py` applies a modular detect-track-segment pipeline to a video or an
image folder.

```bash
python video_pred.py \
  --video /path/to/input.mp4 \
  --output ./output_dts \
  --detector yolo \
  --detector_model yolov8n.pt \
  --target_class 14 \
  --tracker sort \
  --segment_model /path/to/weights/plumageparts_best.pth \
  --segment_model_type dinov3_msu \
  --variant vith16plus \
  --enhanced_decoder \
  --dinov3_weights /path/to/dinov3_backbone_weights.pth \
  --num_classes 10 \
  --label_map models/seg_label/PlumageParts.json
```

For YOLO COCO models, class 14 corresponds to bird.

## External Benchmarks

The manuscript also evaluates on CUB-200-2011 and PartImageNet-derived data.
Those datasets should be obtained from their original sources and used according to their licenses:

- CUB-200-2011: https://www.vision.caltech.edu/datasets/cub_200_2011/
- PartImageNet: https://github.com/tacju/partimagenet

`models/seg_label/partimagenet.json` contains the 41-class PartImageNet label
map used by the benchmark scripts.


## Citation

TODO

## License

The source code in this repository is released under the MIT License.

The PlumageParts annotation masks, metadata, split files and trained model
checkpoints are released through the associated Zenodo record under the licence
specified there.

The original PlumageParts source images are selected from the iRateBirds Citizen
Science Project and are not relicensed by this repository. Please refer to the
iRateBirds dataset paper and its associated reuse terms:

- The iRateBirds Citizen Science Project: a Dataset on Birds' Visual Aesthetic
  Attractiveness to Humans
- https://www.nature.com/articles/s41597-023-02169-0

