# Dry Bean Multi-Class Classification

---

## a. Problem Statement

The objective of this project is to perform multi-class classification on the Dry Bean dataset using various Machine Learning algorithms.  
The goal is to compare different classification models and evaluate their performance using standard evaluation metrics to determine the most effective model for the dataset.

---

## b. Dataset Description

The Dry Bean dataset contains measurements of different types of dry beans.  
Each instance represents a bean sample described using multiple numerical features derived from its shape and structure.

- The dataset is a **multi-class classification problem**.
- Each class corresponds to a different type of dry bean.
- The features are numerical and represent geometric and morphological properties.
- The target variable represents the bean category.

---

## c. Models Used

The following Machine Learning models were implemented and evaluated:

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (kNN)
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

### Comparison Table of Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.924 | 0.9965 | 0.9373 | 0.9358 | 0.9356 | 0.9087 |
| Decision Tree | 0.904 | 0.9513 | 0.9168 | 0.9195 | 0.9179 | 0.8844 |
| kNN | 0.930 | 0.9853 | 0.9471 | 0.9350 | 0.9394 | 0.9158 |
| Naive Bayes | 0.882 | 0.9934 | 0.9047 | 0.8959 | 0.8966 | 0.8590 |
| Random Forest (Ensemble) | 0.934 | 0.9961 | 0.9488 | 0.9430 | 0.9449 | 0.9206 |
| XGBoost (Ensemble) | 0.932 | 0.9968 | 0.9445 | 0.9396 | 0.9414 | 0.9180 |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| Logistic Regression | Performs very well with high AUC (0.9965) and strong overall metrics, indicating good linear separability in the dataset. |
| Decision Tree | Performs reasonably well but slightly lower accuracy and AUC compared to other models; may suffer from overfitting or variance issues. |
| kNN | Achieves strong accuracy (0.93) and balanced performance across all metrics; benefits from local neighborhood learning. |
| Naive Bayes | Shows comparatively lower accuracy but very high AUC (0.9934), indicating good probability estimation despite independence assumptions. |
| Random Forest (Ensemble) | Achieves the highest accuracy (0.934) and highest MCC (0.9206); ensemble averaging improves stability and performance. |
| XGBoost (Ensemble) | Achieves the highest AUC (0.9968) and strong overall metrics; boosting enhances model performance and handles complex patterns effectively. |

---

## Conclusion

Ensemble models such as Random Forest and XGBoost outperform individual baseline models on the Dry Bean dataset.  
XGBoost achieved the highest AUC, while Random Forest achieved the highest accuracy and MCC, indicating strong predictive performance and robustness.
