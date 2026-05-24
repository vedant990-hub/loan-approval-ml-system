# QuantifyAI Loan Approval Prediction

This project predicts loan approval outcomes with scikit-learn, saves reusable model artifacts, and includes a Streamlit app for interactive inference.

## Main Files

- loan_approval_prediction.ipynb: notebook implementation with Sections 1 to 11
- streamlit_app.py: web app for live prediction
- best_loan_model.pkl: trained classifier
- scaler.pkl: fitted StandardScaler
- feature_columns.pkl: saved feature order used during training
- loan_approval_dataset_pipeline.py: script-based pipeline for the lower-case dataset variant
- loan_approval_dataset.csv: dataset used by the script-based pipeline
- HOW_IT_WORKS.md: detailed flow and design notes

## Notebook Workflow

The notebook covers:

1. Setup and imports
2. Dataset loading (uses loan_train.csv when present, otherwise synthetic fallback)
3. Exploratory data analysis
4. Preprocessing
5. Training six models
6. Evaluation and comparison
7. Feature importance
8. Hyperparameter tuning
9. Artifact saving and sample prediction
10. Streamlit app generation cell
11. Summary

## Quick Start

1. Activate your virtual environment.
2. Open loan_approval_prediction.ipynb.
3. Run cells in order from Section 1 through Section 9.
4. Run Streamlit with this command:

streamlit run streamlit_app.py

## Requirements

- Python 3.14
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- streamlit

## Notes

- Section 9 now includes robust preprocessing for the sample row before scaling.
- Saved artifacts are used by both the notebook demo and the Streamlit app.
- The script-based pipeline and the notebook use different dataset schemas, so artifacts should be generated and consumed within the same flow.

## Alternative Script Flow

If you want to run the script pipeline instead of the notebook:

python loan_approval_dataset_pipeline.py
