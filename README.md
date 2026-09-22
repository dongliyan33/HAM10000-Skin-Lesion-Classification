# HAM10000 Skin Lesion Classification

A deep learning project for multi-class skin lesion classification using the HAM10000 dermatoscopic image dataset. The project explores transfer learning with multiple convolutional neural network architectures and compares their performance on a 7-class classification task.

## Project Overview

The HAM10000 dataset contains 10,015 dermatoscopic images across seven diagnostic categories.

This project builds an end-to-end deep learning pipeline for skin lesion classification, including:

- Image preprocessing and data preparation
- Transfer learning with pretrained CNN architectures
- Model training and validation
- Performance evaluation and visualization
- Comparison across multiple neural network architectures

## Models

The project experiments with several convolutional neural network architectures, including:

- ResNet50
- VGG16
- VGG19

Additional model architectures and experiments are included in the source code.

## Dataset

**HAM10000 — Human Against Machine with 10,000 training images**

- 10,015 dermatoscopic images
- 7 diagnostic categories
- Designed for research in automated skin lesion classification

The original image dataset is not included in this repository.

## Project Structure


project1-github/
├── source_code/
│   ├── classification/
│   ├── segmentation/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── figures/
├── main.py
└── ...



Technologies
Python
Deep Learning
Convolutional Neural Networks (CNNs)
Transfer Learning
Image Classification
Computer Vision
Evaluation

Model performance is evaluated using validation results and visualizations. The figures/ directory contains plots and comparisons generated during the experiments.

Repository Contents

The repository contains the source code required for model construction, training, evaluation, and visualization. Large raw image files and trained model weights are excluded to keep the repository lightweight.

Disclaimer

This project is intended for educational and research purposes only. It is not intended for clinical diagnosis or medical decision-making.



