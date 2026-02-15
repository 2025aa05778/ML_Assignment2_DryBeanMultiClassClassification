import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score


st.set_page_config(page_title="Dry Bean Classification", layout="wide")

# -------------------------
# Title & Description
# -------------------------

st.title("Dry Bean Multi-Class Classification App")
st.write("Upload CSV file and select a trained model to make predictions.")

# -------------------------
# Quick Download Sample Test File
# -------------------------

st.markdown("### Quick Test Dataset Download")

sample_file = "test_data_with_labels.csv"

if os.path.exists(sample_file):
    with open(sample_file, "rb") as f:
        st.download_button(
            label="Download Sample Test CSV File",
            data=f,
            file_name="test_data_with_labels.csv",
            mime="text/csv"
        )
else:
    st.info("Sample test file not found.")

# -------------------------
# Load Saved Models
# -------------------------

model_path = "model"

log_model = joblib.load(os.path.join(model_path, "logistic.pkl"))
dt_model = joblib.load(os.path.join(model_path, "decision_tree.pkl"))
knn_model = joblib.load(os.path.join(model_path, "knn.pkl"))
nb_model = joblib.load(os.path.join(model_path, "naive_bayes.pkl"))
rf_model = joblib.load(os.path.join(model_path, "random_forest.pkl"))
xgb_model = joblib.load(os.path.join(model_path, "xgboost.pkl"))

scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))

# -------------------------
# Upload File
# -------------------------

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_option = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# -------------------------
# Prediction Section
# -------------------------

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    # Separate features and labels
    if "Class" in data.columns:
        X_input = data.drop("Class", axis=1)
        y_true = data["Class"]
    else:
        X_input = data
        y_true = None

    # Model selection
    if model_option == "Logistic Regression":
        model = log_model
        X_processed = scaler.transform(X_input)

    elif model_option == "Decision Tree":
        model = dt_model
        X_processed = X_input

    elif model_option == "KNN":
        model = knn_model
        X_processed = scaler.transform(X_input)

    elif model_option == "Naive Bayes":
        model = nb_model
        X_processed = scaler.transform(X_input)

    elif model_option == "Random Forest":
        model = rf_model
        X_processed = X_input

    else:  # XGBoost
        model = xgb_model
        X_processed = X_input

    predictions = model.predict(X_processed)

    # Decode XGBoost predictions
    if model_option == "XGBoost":
        predictions = label_encoder.inverse_transform(predictions)

    st.subheader("Predictions")
    st.write(predictions)

    # -------------------------
    # Metrics + Confusion Matrix
    # -------------------------

    if y_true is not None:

        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, average="macro")
        recall = recall_score(y_true, predictions, average="macro")
        f1 = f1_score(y_true, predictions, average="macro")
        mcc = matthews_corrcoef(y_true, predictions)

        # AUC Calculation
        auc = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_processed)

                if model_option == "XGBoost":
                    y_true_encoded = label_encoder.transform(y_true)
                    auc = roc_auc_score(
                        y_true_encoded,
                        y_prob,
                        multi_class="ovr",
                        average="macro"
                    )
                else:
                    auc = roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="macro"
                    )
            except:
                auc = None

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(accuracy, 4))
        col2.metric("Precision", round(precision, 4))
        col3.metric("Recall", round(recall, 4))

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", round(f1, 4))
        col5.metric("MCC", round(mcc, 4))

        if auc is not None:
            col6.metric("AUC", round(auc, 4))

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, predictions)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

    else:
        st.warning("Uploaded file does not contain true labels. Metrics and confusion matrix not displayed.")