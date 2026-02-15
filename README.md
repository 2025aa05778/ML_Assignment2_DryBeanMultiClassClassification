# Dry Bean Multi-Class Classification

## Project Overview
This project implements and compares six machine learning models for multi-class classification of Dry Bean varieties.

## Dataset
- Dataset: Dry Bean Dataset
- Features: 16 numerical features
- Target: 7 bean varieties

## Implemented Models
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble Boosting)

## Evaluation Metrics
- Accuracy
- Precision (Macro)
- Recall (Macro)
- F1 Score (Macro)
- Matthews Correlation Coefficient (MCC)
- AUC (One-vs-Rest)

## Best Model
XGBoost achieved the best overall performance across most evaluation metrics.

## Streamlit App Features
- Upload CSV dataset
- Select trained model
- Generate predictions
- Display evaluation metrics
- Display confusion matrix

## How to Run

Install dependencies:
pip install -r requirements.txt

Run Streamlit:
streamlit run app.py

Upload test_data_with_labels.csv to view predictions, metrics, and confusion matrix.