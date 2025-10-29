# Tight Generalization Bound for Supervised Quantum Machine Learning

This repository contains the code implementation for the paper "Tight Generalization Bound for Supervised Quantum Machine Learning" by Xin Wang and Rebing Wu.


## Overview

This project investigates tight generalization bounds for supervised quantum machine learning models. The code implements quantum classifiers and regression models that can be used to validate theoretical generalization bounds through experiments. The implementation uses JAX, PennyLane, and Flax to build and train quantum machine learning models.

## Project Structure

```
Tight_Generalization_bound/
├── batch_test/                          # Investigate the effect of Batch size on generalization error
├── classification_phase_datasets/       # Quantum phase classification on multiple sampled datasets
├── datasets/                            # Dataset storage directory
├── datasets_utils/                      # Utility functions for generating and loading datasets
├── label_random/                        # Classification task with completely random labels
├── learning_rate/                       # Investigate the effect of Learning rate on generalization error
├── metric/                              # Metrics computation utilities
├── model/                               # Quantum model
├── optimization_test/                   # Investigate the effect of Optimization Epochs on generalization error
├── optimizer_test/                      # Investigate the effect of Optimizer on generalization error
├── regression_datasets/                 # Regression tasks on multiple sampled datasets
├── regression_special/                  # Regression with special encoding methods
├── scripts/                             # Training Scripts
├── train_utils/                         # Training utilities
├── vis_utils/                           # Visualization utilities
└── env.yml                              # Conda environment file
```

## Dependencies

The project uses a conda environment specified in `env.yml`. Key dependencies include:

- JAX and JAXlib for automatic differentiation and numerical computation
- PennyLane for quantum machine learning
- Flax for neural network implementations
- Optax for optimization algorithms
- PyTorch for dataset loading
- NumPy, SciPy, and Pandas for general data processing
- Matplotlib for visualization

## Setup

**Create Conda Environment**: 
   ```bash
   conda env create -f env.yml
   conda activate quantum
   ```



## Usage


## Data

Datasets are generated through the `.ipynb` files in the datasets_utils directory:

- datasets_utils/classification_phase: Quantum phase classification datasets
- datasets_utils/regression: Regression datasets  
- Regression datasets with special encoding methods



## Models

The core quantum models are defined in the `model/` directory, including:
- `QuantumClassifier`: Quantum machine learning model with configurable ansatz
- Support for different measurement types (Hamiltonian, expectation values)
- Configurable number of qubits and circuit layers

## Training 

Different training task scripts are available in subdirectories of Scripts. For example, for quantum phase classification task with 2000 samples:

```sh
bash /scripts/classification_phase_dataset/classification_phase_2000.sh
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@misc{wang2025tightgeneralizationboundsupervised,
  title={Tight Generalization Bound for Supervised Quantum Machine Learning}, 
  author={Xin Wang and Rebing Wu},
  year={2025},
  eprint={2510.24348},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2510.24348}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
