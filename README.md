

## Group Members
* **Member 1**: Marc Sempere Ayala 
* **Member 2**: Alex Garcia 
* **Member 3**: Berta Solé Claur

---

## Project Overview
This project implements machine learning algorithms from scratch using NumPy to classify our data. 
Our pipeline includes missing data claning and replacement, median imputation, z-score standardization, feature selection based on correlation thresholds, and class imbalance handling. All this data analysis is taking into account the nature of our data which is the prediction of whether a paitaint has coronary heart disease (MICHD) or not. 
To select the best model we compared diferent prediction algorithms, we started with a Regression model but since our target has values {-1,1} we shifted into Logistic Regression and finally we compared weighted Logistic Regression and Logistic Regression using undeersample and overrsample to take into account the imbalance in our data. 
Comparing with F2 score we reached to our best model parameters and we calaulate the final prediction results. 

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
├── aux_explanatory_data_analysis.py # Helper functions for plotting & Data Analysis
├── aux_grid_search.py               # Grid search hyperparameter tuning 
├── aux_preprocessing.py             # Other auxiliar functions
├── helpers.py                       # Given helper functions
├── implementations.py               # 6 required function implementations
├── dataanalysis.ipynb               # Exploratory Data Analysis experiments
├── model_regression.ipynb           # Regresion-Least Squares experiment notebook
├── model_logistic_basic.ipynb       # Undersampled, oversampled and weighted Logistic Regression experiment notebook
├── run.ipynb                        # Main pipeline generating submission.csv
└── README.md                        # Project documentation
```