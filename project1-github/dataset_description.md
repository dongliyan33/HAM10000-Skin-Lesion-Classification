# Dataset Description

## Dataset

- **Name:** Human Against Machine with 10,000 Training Images (HAM10000)
- **Primary publication:** Tschandl, Rosendahl, and Kittler, *Scientific Data*, 2018.
- **Official data source:** Harvard Dataverse
- **Download:** https://doi.org/10.7910/DVN/DBW86T
- **Images used:** 10,015 dermoscopic RGB images
- **Diagnostic classes:** 7
- **Segmentation masks used:** 10,015 real lesion masks supplied with the dataset

## Classes

| Code | Diagnosis |
|---|---|
| akiec | Actinic keratoses and intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

The dataset is strongly imbalanced; `nv` is the majority class. Therefore, the project reports macro and weighted metrics in addition to accuracy.

## Expected Local Layout

The raw 2.6 GB dataset is not duplicated inside the submission ZIP. Download it from the official source and place it as follows:

```text
final_project/data/
├── HAM10000_metadata                 # or HAM10000_metadata.csv
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
└── HAM10000_segmentations_lesion_tschandl/
```

`dataset_manifest/manifest.csv` records every locally used image and mask. `dataset_manifest/class_distribution.csv` records the class counts.

## Data Split

- **Training:** 7,967 images
- **Canonical validation:** 2,048 images
- **Independent test set:** not available in this project
- **Random seed:** 42
- **Grouping:** split by unique `lesion_id`, preventing images from the same lesion from appearing in both training and validation sets

The supplied five pretrained models historically used different split implementations. For fair final comparison, all checkpoints were re-evaluated on the exact same canonical validation image IDs without retraining.

## Classification Preprocessing

- Resize to 224×224
- Convert to tensor
- ImageNet normalization: mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`
- Training augmentation: horizontal/vertical flips, rotations, and mild brightness/contrast jitter
- Validation: deterministic resize and normalization only

## Segmentation Preprocessing

- Paired image/mask loading using matching ISIC identifiers
- Resize images and masks to 128×128 for the completed run
- Nearest-neighbour interpolation for masks
- Binary threshold at 0.5
- Joint flips and rotations so image and mask remain aligned
- The ground-truth masks are genuine dataset annotations; none were generated synthetically

## Licensing and Redistribution

Users must follow the license and citation requirements published on the official Harvard Dataverse record. This submission provides a reproducible download reference and manifest rather than redistributing the full raw dataset.
