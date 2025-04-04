# Neural Network Regression Framework

This repository contains a framework for training and evaluating neural network regression models on a simple dataset. The framework is designed to be simple to use, requiring only plug-and-play components.

## Overview

The `NN_regression.py` script provides a complete pipeline for:
- Loading and normalizing regression data
- Training neural network models with customizable architectures
- Evaluating model performance on training, validation, and test sets
- Visualizing regression results and error metrics

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch
- NumPy
- Pandas
- Matplotlib

You can install the required packages using:

```bash
pip install torch numpy pandas matplotlib
```

### Usage

To run the framework, use the following command:

```
python src/NN_regression.py
```

This will train a neural network on the dataset and save the results to the `results` directory.

You are encouraged to modify the `src/NN_regression.py` script to experiment with different neural network architectures and datasets. This is easily done by adding a new architecture to the `ARCHITECTURES` list using the default format.


