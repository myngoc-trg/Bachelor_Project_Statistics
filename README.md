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
```

## Data availability

The raw pollen image dataset and size measurements may be subject to separate access restrictions and are not automatically covered by the software license of this repository. This repository is intended to document the code, methodology, and experiment structure used in the thesis project.

## Citation

If you use this repository, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

After this repository is archived on Zenodo, the DOI will be added here.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).

The MIT License applies to the code and documentation in this repository. The dataset itself may be subject to separate access restrictions and is not automatically covered by this software license.