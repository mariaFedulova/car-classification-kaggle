# Vehicle Purchase Risk Assessment

Predicting "lemon" vehicles using custom-built classification models with time-based validation and Gini score evaluation.

## Overview

The goal is to build and compare classification models that help identify problematic used car purchases based on historical transaction data. Key aspects include:

- Time-based train/validation/test splitting to respect temporal order
- Proper preprocessing with categorical encoding and feature scaling
- Implementation of custom classifiers from scratch
- Feature engineering to improve predictive performance
- Hyperparameter tuning and feature selection

## Dataset

Data is sourced from a Kaggle Don’tGetKicked competition. The target variable is binary: indicates whether a purchased vehicle was problematic ("bad buy").

## Models Implemented

### Scikit-learn Models
- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors

### Custom Implementations
- **Logistic Regression**: Trained using Stochastic Gradient Descent
- **Naive Bayes**: With Gaussian assumption for continuous features
- **KNN**: Custom distance-based classifier with configurable k parameter

## Key Metrics

The following metrics are implemented and used for evaluation:
- **Gini Score** (derived from ROC AUC: `2 * AUC - 1`)
- Precision & Recall
- F1 Score
- AUC PR

## Feature Engineering

Additional nonlinear features are created to boost performance:
- Ratios between numerical features
- Target-encoded categorical features using group means

## Feature Selection

Two approaches are explored:
1. Manual selection based on Logistic Regression coefficient magnitudes
2. Automatic selection using L1 regularization 

## Hyperparameter Tuning

The best performing model undergoes hyperparameter optimization to maximize validation Gini score, analyzing which parameters have the greatest impact (using Optuna).

## Validation Strategy

- **Train set**: Earliest 1/3 of purchase dates
- **Validation set**: Middle 1/3 of purchase dates
- **Test set**: Latest 1/3 of purchase dates
- This temporal split prevents future data leakage and better simulates real-world prediction scenarios

## Results Summary

All custom implementations achieve comparable performance to their scikit-learn counterparts. The best model configuration is evaluated on training, validation, and test sets to assess overfitting. Gini scores across all three datasets are reported and analyzed.

## Files Structure
* data/ # Dataset files
* classification.ipynb # Main notebook with all experiments and conclusions
* README.md # This file

## Requirements

- Python
- pandas, numpy, scikit-learn
- optuna
