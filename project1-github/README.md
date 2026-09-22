# HAM10000 Skin Lesion Classification and Segmentation

## Project Overview

This Final Project implements multi-class skin-lesion classification and binary lesion segmentation on HAM10000. It compares ResNet50, VGG16, VGG19, DenseNet121, EfficientNet-B0 and a custom CNN, then trains a compact U-Net using real segmentation masks. Existing classifier checkpoints were preserved and evaluated fairly on one canonical lesion-level validation split.

## Environment

- macOS on Apple Silicon
- Python 3.13 virtual environment
- PyTorch 2.13.0 / torchvision 0.28.0
- Apple Metal Performance Shaders (MPS)
- Completed inference device: `mps`

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py check
```

Model checkpoints are intentionally excluded from this GitHub-ready repository because GitHub has strict file-size limits. Add them through Git LFS or a release asset if you want to share trained weights; they are compatible with the architecture definitions in `source_code/classification/models.py`.

## Dataset

Download HAM10000 from the [official Harvard Dataverse record](https://doi.org/10.7910/DVN/DBW86T), then create the `data/` layout documented in [dataset_description.md](dataset_description.md). The raw dataset is excluded from this repository because it is approximately 2.6 GB. A complete manifest is included.

## Training

The supplied results do not require retraining. Optional reproducible classifier training writes to a new directory and never overwrites supplied weights:

```bash
source venv/bin/activate
python source_code/training/train_classifier.py --model EfficientNetB0 --data-dir data --epochs 5
```

Supported model names are `ResNet50`, `VGG16`, `VGG19`, `DenseNet121`, `EfficientNetB0`, and `CNN`.

## Evaluation

Evaluate one checkpoint on the canonical lesion-level split:

```bash
source venv/bin/activate
python main.py evaluate --models EfficientNetB0 --data-dir data
```

Evaluate all models:

```bash
python main.py evaluate --models all --data-dir data --batch-size 16
```

Full evaluation requires `torch.backends.mps.is_available() == True`. Results are written to `results/recomputed/` and supplied results remain unchanged.

## Segmentation

The completed compact U-Net run used 10,015 real masks, 128×128 input, five epochs and MPS. Optional retraining:

```bash
python main.py segment --data-dir data --epochs 5 --batch-size 32 --size 128
```

New segmentation runs are stored in `new_training_runs/SegmentationUNet/` and do not overwrite the supplied checkpoint.

## Results

| Model | Accuracy | Macro F1 | Weighted F1 | Parameters |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 87.70% | 0.8075 | 0.8741 | 4,016,515 |
| DenseNet121 | 77.49% | 0.5345 | 0.7626 | 6,961,031 |
| VGG16 | 77.39% | 0.6489 | 0.7774 | 27,563,847 |
| VGG19 | 72.02% | 0.7659 | 0.7394 | 32,873,543 |
| ResNet50 | 68.12% | 0.5435 | 0.7058 | 23,522,375 |
| Custom CNN | 48.29% | 0.3376 | 0.5309 | 423,175 |

Segmentation achieved Dice 0.9176, IoU 0.8613, precision 0.9582 and recall 0.8983. All values come from saved experiment files; no missing metric was invented. The custom CNN has no genuine epoch history, so no training curve is claimed for it.

## Folder Structure

```text
final_project/
├── README.md
├── requirements.txt
├── dataset_description.md
├── main.py
├── Project_Report.docx
├── Final_Project_Presentation.pptx
├── source_code/
│   ├── classification/
│   ├── segmentation/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── dataset_manifest/
├── results/
├── figures/
└── weights/
```

## Reproducibility Notes

- Seed: 42
- Canonical validation: 2,048 images grouped by `lesion_id`
- Supplied classifier checkpoints were not retrained during final comparison
- Original historical training splits differed, so historical best accuracies are not substituted for canonical metrics
- No external test dataset was available; validation results must not be described as external clinical performance
