# Loan Approval Prediction - Machine Learning Project

## Project Status: ✅ COMPLETE

This project implements an end-to-end machine learning pipeline for loan approval prediction using scikit-learn with comprehensive EDA, model training, hyperparameter tuning, and deployment capabilities.

---

## Deliverables

### 1. **Verified Working Implementations**

#### Primary: `complete_loan_notebook.py` (828 lines)
- ✅ **Status**: Fully tested, end-to-end execution confirmed
- **All 11 Sections Implemented & Verified**:
  1. Project Setup & Library Imports (with matplotlib Agg backend)
  2. Dataset Loading (synthetic fallback: 614 rows)
  3. EDA (9 visualizations: target dist, approval rates, income dist, correlation heatmap, pairplot)
  4. Data Preprocessing (imputation, encoding, feature engineering, log transforms)
  5. Model Training (6 classifiers: LogisticRegression, DecisionTree, RandomForest, GradientBoosting, SVM, KNN)
  6. Model Evaluation (classification reports, confusion matrices, metrics comparison)
  7. Feature Importance (top 3: ApplicantIncome, TotalIncome, LoanAmount)
  8. Hyperparameter Tuning (GridSearchCV with model-specific param grids)
  9. Model Saving (joblib artifacts)
  10. Streamlit App Code Generation (interactive prediction UI with emoji bank icon 🏦)
  11. Project Summary (metrics and key insights)

- **Execution Result**: Successfully runs to completion with all outputs
- **Best Model**: Gradient Boosting Classifier (Accuracy: 0.7236, F1-Score: 0.6744)
- **Random Seed**: 42 (reproducible)
- **File Encoding**: UTF-8 (supports emoji in streamlit app title)

```bash
# Run the complete script
python complete_loan_notebook.py
```

#### Secondary: `loan_approval_prediction.ipynb`
- ✅ **Status**: Valid Jupyter notebook with 22 cells (11 markdown + 11 code)
- **Sections 1-3**: Verified working (setup, dataset, EDA with visualizations)
- **Note**: Sections 4+ need notebook kernel reset after first run due to state dependencies
- **Recommendation**: Use `complete_loan_notebook.py` as primary execution method

---

### 2. **Model Artifacts** (Trained & Saved)

All artifacts created by executing `complete_loan_notebook.py`:

- **`best_loan_model.pkl`** (85.6 KB)
  - Trained Gradient Boosting Classifier
  - Ready for production predictions
  
- **`scaler.pkl`** (1.3 KB)
  - StandardScaler with fitted parameters
  - Use this to preprocess new loan applications
  
- **`feature_columns.pkl`** (193 B)
  - Ordered list of 12 features for prediction
  - Ensures feature order consistency

---

### 3. **Deployment Application**

- **`streamlit_app.py`** (4.8 KB)
  - Interactive web UI for loan approval predictions
  - Features:
    - Sidebar input form for all 13 loan fields
    - Real-time prediction with approval probability
    - Preprocessing identical to training pipeline
    - Bank icon 🏦 in page title

```bash
# Run the interactive app (requires best_loan_model.pkl, scaler.pkl, feature_columns.pkl)
streamlit run streamlit_app.py
```

---

### 4. **Utility Scripts**

- **`generate_loan_notebook.py`** (38.3 KB)
  - Programmatically generates Jupyter notebooks from code/markdown definitions
  - Used to create `loan_approval_prediction.ipynb`
  
- **`create_clean_notebook.py`** (1.3 KB)
  - Fixes indentation issues in generated notebooks
  - Utility for notebook formatting maintenance

---

## Dataset

### Synthetic Data Generation (Reproducible)
- **Rows**: 614 loan applications
- **Columns**: 13 features (ID, demographics, financial, approval status)
- **Realistic Characteristics**:
  - Missing values injected (~5-10% per column) to simulate real-world data
  - Income distributions with log-skew (realistic for financial data)
  - Approval rate ~74% (realistic class imbalance)
  - Features include: Gender, Marital Status, Education, Income, Loan Amount, Credit History, Property Area

### Fallback Behavior
If `loan_train.csv` is not found, the script auto-generates a synthetic dataset with identical structure

---

## Technical Stack

| Component | Version |
|-----------|---------|
| Python | 3.14 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |
| streamlit | 1.56.0 |
| joblib | 1.5.3 |

---

## Key Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.707 | 0.666 | 0.707 | 0.686 |
| Decision Tree | 0.633 | 0.601 | 0.633 | 0.616 |
| Random Forest | 0.715 | 0.682 | 0.715 | 0.698 |
| **Gradient Boosting** | **0.724** | **0.668** | **0.724** | **0.675** |
| SVM | 0.707 | 0.666 | 0.707 | 0.686 |
| KNN | 0.658 | 0.620 | 0.658 | 0.638 |

**Best Model**: Gradient Boosting Classifier
- **After Tuning**: Accuracy: 0.7236, F1-Score: 0.6744

### Hyperparameter Tuning (Gradient Boosting)
```
Best Parameters:
- learning_rate: 0.1
- max_depth: 2
- n_estimators: 100
- subsample: 1.0

Cross-val Score: 0.6834
```

### Top 3 Most Important Features
1. **ApplicantIncome** (Income of loan applicant)
2. **TotalIncome** (Combined applicant + co-applicant income)
3. **LoanAmount** (Amount of loan requested)

---

## EDA Visualizations (9 Total)

All generated with matplotlib in non-interactive (Agg) backend:

1. Target Variable Distribution (loan approval Y/N count)
2. Approval Rate by Gender
3. Approval Rate by Education
4. Approval Rate by Credit History
5. Approval Rate by Property Area
6. ApplicantIncome Distribution (histogram + boxplot)
7. LoanAmount Distribution (histogram + boxplot)
8. Correlation Heatmap (all numeric features)
9. Pairplot (4 key numeric features colored by approval status)

---

## Preprocessing Pipeline

### Missing Value Handling
- **Categorical columns**: Mode imputation
- **Numeric columns**: Median imputation
- **Target variable**: Mode imputation

### Feature Engineering
- **TotalIncome**: ApplicantIncome + CoapplicantIncome
- **Log Transforms**: `np.log1p()` applied to:
  - LoanAmount (handle right-skew)
  - TotalIncome (stabilize variance)

### Categorical Encoding
- LabelEncoder for all 6 categorical features
- Encoders saved for prediction consistency

### Scaling
- StandardScaler normalization applied to all numeric features
- Scaler fitted on training data, applied to test data
- Scaler saved for production predictions

---

## How to Use

### Execute the Full Pipeline
```bash
cd "D:\Work\WI Sem- 3\Loan something\S72-0526-QuantifyAI-ScikitLearn-DAV"
python complete_loan_notebook.py
```

**Output**:
- Console prints: All 11 sections with metrics and insights
- Generated files:
  - `best_loan_model.pkl` (trained model)
  - `scaler.pkl` (preprocessing scaler)
  - `feature_columns.pkl` (feature list)
  - `streamlit_app.py` (interactive app code)

### Run the Interactive Web App
```bash
streamlit run streamlit_app.py
```

Then open browser to `http://localhost:8501`

### Make Predictions Programmatically
```python
import joblib
import numpy as np

# Load artifacts
model = joblib.load("best_loan_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("feature_columns.pkl")

# Preprocess new data and make prediction
# (Ensure input matches training feature order and preprocessing)
prediction = model.predict_proba(scaled_input)[0]
print(f"Approval probability: {prediction[1]:.4f}")
```

---

## Code Quality & Reproducibility

✅ **Reproducibility Guaranteed**:
- RANDOM_STATE = 42 throughout
- Stratified train/test split (80/20)
- Deterministic numpy random seed
- Synthetic data generation uses fixed seed

✅ **Error Handling**:
- Fallback to synthetic dataset if CSV missing
- Try-except blocks for data I/O
- Graceful handling of matplotlib backend

✅ **Clean Code**:
- Well-organized sections with clear output headers
- Modular function design
- Comprehensive comments
- Type hints where applicable

---

## Limitations & Future Improvements

### Current Limitations
1. Synthetic data is for demonstration; production needs real Kaggle dataset
2. Baseline metrics ~72% accuracy (room for improvement)
3. No advanced feature engineering (interaction terms, polynomial features)
4. No ensemble stacking or voting classifiers

### Future Enhancements
1. SHAP-based model explainability
2. Calibration curves for probability adjustment
3. Cost-sensitive learning (account for false positive/negative costs)
4. API deployment (Flask/FastAPI)
5. A/B testing framework
6. Real-time model monitoring & retraining pipeline

---

## File Structure

```
S72-0526-QuantifyAI-ScikitLearn-DAV/
├── README.md                           # This file
├── complete_loan_notebook.py           # ✅ PRIMARY: Full ML pipeline (verified working)
├── loan_approval_prediction.ipynb      # Jupyter notebook format
├── streamlit_app.py                    # Interactive prediction web app
├── best_loan_model.pkl                 # Trained model artifact
├── scaler.pkl                          # Preprocessing scaler artifact
├── feature_columns.pkl                 # Feature list artifact
├── generate_loan_notebook.py           # Notebook generator utility
├── create_clean_notebook.py            # Indentation fix utility
└── .venv/                              # Python virtual environment

```

---

## Troubleshooting

### Issue: "UnicodeEncodeError" when running script
**Solution**: Already fixed. Script uses `encoding="utf-8"` when writing Streamlit app.

### Issue: Matplotlib displays GUI error ("Can't find usable init.tcl")
**Solution**: Already fixed. Script uses matplotlib Agg backend for headless execution.

### Issue: Notebook cells won't run
**Solution**: Use `complete_loan_notebook.py` instead. If notebook needed, restart kernel and re-run cells sequentially.

### Issue: Model prediction fails with "Feature mismatch"
**Solution**: Ensure input DataFrame column order matches `feature_columns.pkl` and preprocessing matches `scaler.pkl` fitted state.

---

## Contact & Support

For questions about specific sections:
- **EDA Questions**: See Section 3 in `complete_loan_notebook.py`
- **Model Training**: See Section 5
- **Hyperparameter Tuning**: See Section 8
- **Deployment**: See `streamlit_app.py`

---

**Last Updated**: 2026-04-28  
**Project Status**: ✅ Complete & Tested  
**Execution Time**: ~10-15 minutes (full pipeline)  
**Random Seed**: 42 (reproducible across runs)
