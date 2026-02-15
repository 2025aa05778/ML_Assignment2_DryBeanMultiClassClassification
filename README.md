# Dry Bean Multi-Class Classification

## Project Overview

This project performs multi-class classification of dry beans using multiple Machine Learning algorithms.  
Each model is evaluated using standard performance metrics to compare their effectiveness.

---

# Evaluation Metrics Used

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- AUC (Area Under ROC Curve)

---

# Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | MCC | AUC |
|-------|----------|-----------|--------|----------|------|------|
| Model 1 | 0.924 | 0.9373 | 0.9358 | 0.9356 | 0.9087 | 0.9965 |
| Decision Tree | 0.904 | 0.9168 | 0.9195 | 0.9179 | 0.8844 | 0.9513 |
| KNN | 0.930 | 0.9471 | 0.9350 | 0.9394 | 0.9158 | 0.9853 |
| Naive Bayes | 0.882 | 0.9047 | 0.8959 | 0.8966 | 0.8590 | 0.9934 |
| Random Forest | 0.934 | 0.9488 | 0.9430 | 0.9449 | 0.9206 | 0.9961 |
| XGBoost | 0.932 | 0.9445 | 0.9396 | 0.9414 | 0.9180 | 0.9968 |

---

# Performance Highlights

- **Highest Accuracy:** Random Forest (0.934)
- **Highest AUC:** XGBoost (0.9968)
- **Highest MCC:** Random Forest (0.9206)
- Ensemble methods outperform individual baseline models.

---

# How to Run the Project

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
