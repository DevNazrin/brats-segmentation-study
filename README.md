# Brain Tumor Segmentation on BraTS 2021

A study comparing CNN and Transformer-based architectures for 3D brain tumor segmentation from multi-modal MRI scans.

## Overview

Accurate brain tumor segmentation from MRI is critical for diagnosis and treatment planning, but manual delineation is time-consuming and prone to inter-observer variability. This project implements and compares two deep learning approaches for automatic 3D segmentation:

- **3D U-Net** — a convolutional encoder-decoder baseline
- **TransUNet-Tiny** — a lightweight Transformer-based architecture designed for low-resource settings

Both models are trained and evaluated on the [BraTS 2021](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) dataset, which provides four MRI modalities per case (T1, T1ce, T2, FLAIR) with expert tumor annotations.

## Status

🚧 **Work in progress** — second-semester graduation project, Ankara University, Department of Computer Engineering.

The first semester established a baseline pipeline and reproduced standard model comparisons. The second semester focuses on:

- Refactoring into a reproducible experimental framework
- Investigating data efficiency: how segmentation performance scales with training set size for each architecture
- Per-case failure analysis to understand where each model breaks down

## Methods

- **Framework:** PyTorch + MONAI
- **Preprocessing:** Per-modality intensity normalization, foreground cropping, random spatial cropping during training
- **Inference:** Sliding-window inference for full-volume evaluation
- **Loss:** Dice + Cross-Entropy
- **Metrics:** Dice Similarity Coefficient (primary), with planned addition of IoU and Hausdorff distance

## Repository Structure

brats-segmentation-study/
├── configs/         # YAML experiment configurations
├── data/            # Local dataset utilities (BraTS data is not committed)
├── notebooks/       # Exploratory and analysis notebooks
├── src/
│   ├── data/        # Dataset loading, transforms, splits
│   ├── models/      # Model architectures (U-Net, TransUNet-Tiny)
│   ├── training/    # Training loops, schedulers, callbacks
│   └── evaluation/  # Metrics, sliding-window inference, analysis
└── outputs/
├── checkpoints/ # Saved model weights
└── logs/        # Training logs and CSV results

## Setup

Setup instructions will be added once the codebase is migrated.

## Author

**Nazrin Mammadli** — Ankara University, Faculty of Engineering, Department of Computer Engineering
Supervisor: Dr. Öğr. Üyesi İrem Ülkü

## License

This project is released under the [MIT License](LICENSE).