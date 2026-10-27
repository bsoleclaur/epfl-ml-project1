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

├── data/
|   ├── sample_submission.csv          # Training feature
│   ├── x_train.csv                    # Training feature matrix
│   ├── y_train.csv                    # Training targets {-1, 1}
│   └── x_test.csv                     # Test feature matrix
├── aux_cleaning_functions.py          # Required 6 EPFL algorithm functions
├── aux_explanatory_data_analysis.py   # Required 6 EPFL algorithm functions
├── aux_grid_search.py                 # Required 6 EPFL algorithm functions
├── aux_preprocessing.py               # Required 6 EPFL algorithm functions
├── helpers.py                         # Required 6 EPFL algorithm functions
├── implementations.py                 # Required 6 EPFL algorithm functions
├── run.ipynb                   # End-to-end pipeline script to build submission.csv
├── dataanalysis.ipynb        # Exploratory Data Analysis & Grid Search experiments
├── model_logistic_basic.ipynb        # Grid Search experiments for Logistic Regression
├── model_regression.ipynb        # Grid Search experiments for Regression
└── README.md                # Project documentation