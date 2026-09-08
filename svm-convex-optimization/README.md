# SVM Convex Optimization Project

## Overview
This project trains and compares three classifiers — Linear SVM, RBF Kernel SVM, 
and Logistic Regression — on the Breast Cancer Wisconsin dataset. It demonstrates 
convex optimization concepts (convex loss functions, margins, regularization) 
applied to a real machine learning task.

## Dataset
Breast Cancer Wisconsin dataset from `sklearn.datasets.load_breast_cancer()`.
Binary classification: malignant (0) vs benign (1). 30 numerical features, 
569 samples.

## Project Structure
```
svm-convex-optimization-task/
├── README.md
├── requirements.txt
├── notebooks/
│   └── svm_training.ipynb      
├── outputs/
│   ├── confusion_matrix.png
│   ├── decision_boundary.png
│   └── metrics_table.csv
└── report/
└── convex_optimization_applications.pdf
```
## How to Run
1. Install dependencies:
`pip install -r requirements.txt`
2. Open `notebooks/svm_training.ipynb` in Jupyter.
3. Run all cells in order, top to bottom.
4. Outputs (plots, metrics table) will be saved automatically to `outputs/`.

## What the Notebook Does
1. Loads and inspects the dataset
2. Splits into train/test sets (80/20, stratified)
3. Scales features with StandardScaler
4. Trains and tunes a Linear SVM (grid search on C)
5. Trains and tunes an RBF Kernel SVM (grid search on C and gamma)
6. Trains a Logistic Regression baseline
7. Evaluates all three models (accuracy, precision, recall, F1, confusion matrix)
8. Visualizes 2D decision boundaries (PCA-reduced) for all three models
9. Explains the best model in terms of margin, support vectors, regularization, 
   and generalization

## Results Summary
All three models achieved high performance (F1 ≈ 0.97–0.99) on the standard 
80/20 split, with results converging closely — consistent with this dataset 
being close to linearly separable. See `outputs/metrics_table.csv` for exact 
numbers and the report for full discussion. But for smaller dataset like 20/80,
40/60 their performance changes well check that

