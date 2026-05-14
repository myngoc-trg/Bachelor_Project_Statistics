# Improving Pollen Species Classification Using Deep Learning and Size Features

This repository contains code, notebooks, figures, and experiment material for the bachelor thesis:

**Improving Pollen Species Classification Using Deep Learning and Size Features**  
Nancy Truong, Lund University, 2025.

The project investigates deep-learning models for classifying microscopic pollen-grain images from ten plant species using RGB image crops together with pollen size measurements. The work builds on the ResNet18 image-and-size model of Jurdell (2025) and evaluates imputation, validation-based model selection, class-balancing augmentation, focal loss, reduced image-feature representation, and ConvNeXt-Tiny.

## Main results

The reproduced ResNet18 baseline achieved approximately:

- Test accuracy: 76.8%
- Macro recall: 76.2%

The best-performing model was the augmented ConvNeXt-Tiny model:

- Test accuracy: 85.2%
- Macro recall: 85.3%

The main remaining difficulty was the separation of *Crepis capillaris* and *Hypochaeris radicata*, especially under flower-level test splits.

## Repository structure

```text
.
├── Data_Utility/              # Data loading, splitting, size lookup, and imputation utilities
├── engine/                    # Training and evaluation utilities
├── loss_fn/                   # Loss-function implementations, including focal loss
├── models/                    # Model definitions and saved-model utilities
├── figures/                   # Generated figures used in the thesis
├── Experimental_Reports/      # Experiment outputs and result summaries
├── Report_base.ipynb          # Baseline and model-comparison notebook
├── Report_sensi.ipynb         # Sensitivity-analysis notebook
├── Report.ipynb               # Main experimental notebook
├── train_flower_summary_missing_sizes.csv
├── train_species_flower_sizes.pdf
├── train_test_species_sizes.pdf
└── README.md
