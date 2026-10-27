# EPFL Machine Learning - Project 1 

## Group Members
* **Member 1**: Marc Sempere Ayala (Sciper: XXXXXX)
* **Member 2**: Alex Garcia (Sciper: XXXXXX)
* **Member 3**: Berta Solé Claur (Sciper: XXXXXX)

---

## Project Overview
This project implements machine learning algorithms from scratch using NumPy to classify our data. Our pipeline includes missing data claning and replacement, median imputation, z-score standardization, feature selection based on correlation thresholds, and class imbalance handling.

### Best Performing Model & Results
* **Model**: Logistic Regression
* **Validation F1-Score**: 0.XX
* **Validation Accuracy**: XX.X%
* **Optimal Decision Threshold**: 0.XX

---

## Repository Structure
```text
.
├── data/
│   ├── sample_submission.csv        # Sample submission format
│   ├── x_train.csv                  # Training feature matrix
│   ├── y_train.csv                  # Training targets {-1, 1}
│   └── x_test.csv                   # Test feature matrix
├── aux_cleaning_functions.py        # Helper functions for NaN & outlier removal
├── aux_explanatory_data_analysis.py # Helper functions for plotting & EDA
├── aux_grid_search.py               # Grid search hyperparameter tuning logic
├── aux_preprocessing.py             # Imputation, standardization & OHE helpers
├── helpers.py                       # EPFL project I/O helper functions
├── implementations.py               # 6 required EPFL algorithm implementations
├── run.ipynb                        # Main pipeline generating submission.csv
├── dataanalysis.ipynb               # Exploratory Data Analysis experiments
├── model_logistic_basic.ipynb       # Logistic Regression experiment notebook
├── model_regression.ipynb           # Least Squares experiment notebook
└── README.md                        # Project documentation
```