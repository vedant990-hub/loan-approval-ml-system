# How QuantifyAI Works

This document explains the current notebook-first flow used in this repository and how it connects to inference and deployment.

## 1. End-to-End Flow

The active workflow is:

1. Load data in Section 2.
2. Explore distributions and relationships in Section 3.
3. Build a model-ready matrix in Section 4.
4. Train six classifiers in Section 5.
5. Compare and validate models in Section 6.
6. Inspect feature importance in Section 7.
7. Tune the best model in Section 8.
8. Save artifacts and run sample inference in Section 9.
9. Generate and run the Streamlit app in Section 10.

Primary notebook:

- loan_approval_prediction.ipynb

## 2. Data Loading Behavior

Section 2 uses this fallback logic:

- If loan_train.csv exists, it is loaded.
- If it does not exist, a synthetic loan dataset is generated with realistic distributions and missing values.

This keeps the notebook runnable even when the original CSV is not available.

## 3. EDA Outputs

Section 3 includes:

1. Target class distribution
2. Approval rates by key categories
3. Income and loan amount distributions
4. Correlation heatmap
5. Pairplot for selected numeric fields

The EDA is intended to verify class balance, check skew, and identify informative features before model training.

## 4. Preprocessing Logic

Section 4 performs:

- column cleanup (whitespace, normalization)
- missing-value imputation for categorical and numeric columns
- categorical encoding
- numeric safety conversion
- train/test split with stratification
- feature scaling with StandardScaler

The output of this section defines the feature schema used by the scaler and model.

## 5. Modeling and Selection

Sections 5 to 8:

- train six classifiers
- compute Accuracy, Precision, Recall, and F1-score
- visualize confusion matrices and metric comparison
- select the best model by F1-score
- run GridSearchCV with model-specific parameter grids

The tuned model is assigned to final_model for artifact export.

## 6. Section 9 Artifact and Inference Step

Section 9 saves:

- best_loan_model.pkl
- scaler.pkl
- feature_columns.pkl

It also validates inference with a sample application row. The preprocessing helper in this section now ensures the sample is converted to fully numeric model input before scaling.

## 7. Streamlit Deployment Path

Section 10 writes streamlit_app.py, which:

1. loads model, scaler, and feature order
2. collects user input in a sidebar form
3. applies preprocessing
4. runs prediction and probability
5. displays approval result and confidence

Run with:

streamlit run streamlit_app.py

## 8. Important Compatibility Note

This repository contains two different data schemas:

- notebook schema (mixed-case columns such as LoanAmount)
- script pipeline schema in loan_approval_dataset_pipeline.py (lower-case columns such as loan_amount)

Artifacts should be trained and inferred within the same schema flow to avoid column and encoding mismatches.

## 9. Reproducibility and Design Choices

- random seed fixed to 42
- warnings suppressed for cleaner output
- consistent scaling and feature-order persistence
- artifact-based inference for notebook and app consistency