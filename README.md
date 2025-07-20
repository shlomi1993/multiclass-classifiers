# Multiclass Classifiers

A simple yet complete Python implementation of four classic multiclass classification algorithms from scratch:

- **K-Nearest Neighbors (KNN)**
- **Perceptron**
- **Support Vector Machine (SVM)**
- **Passive-Aggressive (PA)**

This project also includes:
- **Cross-validation** for hyperparameter tuning.
- **Normalization techniques** (Min-Max and Z-Score).
- **Evaluation pipeline** for comparing model performance.

## Project Structure

```
multiclass-classifiers/
├── calibration.py          # Finds optimal hyperparameters via 5-fold cross-validation
├── classifiers.py          # Core implementations of KNN, Perceptron, SVM, PA
├── ml\_utils.py             # Utility functions: data shuffling, weight creation, predictions
├── normalizations.py       # Z-Score and Min-Max normalization
├── test.py                 # Runs predictions with selected hyperparameters
└── report.pdf              # Project writeup detailing logic, methods, and results
````

## Usage

### 1. Run Prediction

```bash
python test.py <train_x.csv> <train_y.csv> <test_x.csv> <output.txt>
````

### 2. Find Best Hyperparameters

```bash
python calibration.py <train_x.csv> <train_y.csv>
```

## Algorithms Summary

| Algorithm          | Tuned Hyperparameters           | Notes                                         |
| ------------------ | ------------------------------- | --------------------------------------------- |
| KNN                | k = 5                           | Performs best without normalization           |
| Perceptron         | epochs = 90, eta = 0.2          | Learning rate halves on each mistake          |
| SVM                | epochs = 80, eta = 0.1, λ = 0.8 | Regularized margin-based updates              |
| Passive-Aggressive | epochs = 190                    | Aggressive updates using margin loss and norm |

All algorithms are implemented using only NumPy - no external ML libraries are used.

## Methodology

* 5-fold cross-validation for parameter tuning
* Z-Score and Min-Max normalization comparisons
* Shuffle before training to ensure fair splits
* Accuracy-based evaluation

## Report

For detailed explanations and implementation logic, see [`report.pdf`](report.pdf).

## Requirements

* Python 3.x
* NumPy

```bash
pip install numpy
```
