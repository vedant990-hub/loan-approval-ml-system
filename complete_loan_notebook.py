"""
Loan Approval Prediction System - Complete ML Notebook

This notebook builds a complete machine learning system for loan approval prediction.
It covers data loading, EDA, preprocessing, model training, evaluation, tuning, and deployment.
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ============================================================================
# SECTION 1: Project Setup & Library Imports
# ============================================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

print("=" * 80)
print("SECTION 1: Project Setup & Library Imports")
print("=" * 80)
print(f"Random seed set to: {RANDOM_STATE}")
print("Plot style configured and warnings suppressed.")


# ============================================================================
# SECTION 2: Dataset Loading
# ============================================================================


def generate_synthetic_loan_data(n_rows: int = 614, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic loan dataset for reproducibility."""
    rng = np.random.default_rng(random_state)

    loan_ids = [f"LP{index:04d}" for index in range(1, n_rows + 1)]
    gender = rng.choice(["Male", "Female"], size=n_rows, p=[0.65, 0.35])
    married = rng.choice(["Yes", "No"], size=n_rows, p=[0.66, 0.34])
    dependents = rng.choice(["0", "1", "2", "3+"], size=n_rows, p=[0.55, 0.17, 0.15, 0.13])
    education = rng.choice(["Graduate", "Not Graduate"], size=n_rows, p=[0.78, 0.22])
    self_employed = rng.choice(["No", "Yes"], size=n_rows, p=[0.84, 0.16])
    property_area = rng.choice(["Rural", "Semiurban", "Urban"], size=n_rows, p=[0.34, 0.41, 0.25])
    credit_history = rng.choice([1.0, 0.0], size=n_rows, p=[0.84, 0.16])

    applicant_income = np.clip(rng.lognormal(mean=8.25, sigma=0.45, size=n_rows), 1500, 25000).round(0)
    coapplicant_income = np.where(
        rng.random(n_rows) < 0.55,
        0,
        rng.lognormal(mean=7.45, sigma=0.70, size=n_rows),
    )
    coapplicant_income = np.clip(coapplicant_income, 0, 20000).round(0)

    base_loan_amount = (applicant_income + coapplicant_income) / rng.uniform(38, 58, size=n_rows)
    loan_amount = np.clip(base_loan_amount + rng.normal(0, 28, size=n_rows), 20, 700).round(0)
    loan_term = rng.choice(
        [12, 36, 60, 120, 180, 240, 300, 360, 480],
        size=n_rows,
        p=[0.01, 0.02, 0.05, 0.05, 0.08, 0.09, 0.08, 0.56, 0.06],
    )

    loan_term_factor = np.where(loan_term == 360, 0.15, 0.0)
    property_factor = np.select(
        [property_area == "Semiurban", property_area == "Urban"],
        [0.18, 0.10],
        default=0.0,
    )
    education_factor = np.where(education == "Graduate", 0.14, -0.03)
    married_factor = np.where(married == "Yes", 0.08, 0.0)
    self_employed_factor = np.where(self_employed == "Yes", -0.12, 0.0)
    dependents_factor = np.select(
        [dependents == "1", dependents == "2", dependents == "3+"],
        [0.08, 0.05, -0.10],
        default=0.0,
    )
    income_term = 0.00005 * applicant_income + 0.00004 * coapplicant_income - 0.0035 * loan_amount
    noise = rng.normal(0, 0.65, size=n_rows)
    score = (
        -0.75
        + 1.95 * credit_history
        + income_term
        + loan_term_factor
        + property_factor
        + education_factor
        + married_factor
        + self_employed_factor
        + dependents_factor
        + noise
    )
    approval_probability = 1 / (1 + np.exp(-score))
    loan_status = np.where(rng.random(n_rows) < approval_probability, "Y", "N")

    data = pd.DataFrame(
        {
            "Loan_ID": loan_ids,
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income.astype(int),
            "CoapplicantIncome": coapplicant_income.astype(int),
            "LoanAmount": loan_amount.astype(int),
            "Loan_Amount_Term": loan_term.astype(int),
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Loan_Status": loan_status,
        }
    )

    missing_rates = {
        "Gender": 0.04,
        "Married": 0.03,
        "Dependents": 0.05,
        "Education": 0.03,
        "Self_Employed": 0.05,
        "LoanAmount": 0.06,
        "Loan_Amount_Term": 0.05,
        "Credit_History": 0.06,
        "Property_Area": 0.03,
    }

    for column, missing_rate in missing_rates.items():
        mask = rng.random(n_rows) < missing_rate
        data.loc[mask, column] = np.nan

    return data


print("\n" + "=" * 80)
print("SECTION 2: Dataset Loading")
print("=" * 80)

DATA_PATH = "loan_train.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset from: {DATA_PATH}")
else:
    df = generate_synthetic_loan_data(n_rows=614, random_state=RANDOM_STATE)
    print("loan_train.csv not found. Generated a synthetic loan dataset instead.")

print("\nData preview: df.head()")
display(df.head())

print("\nData shape: df.shape")
print(df.shape)

print("\nData info: df.info()")
df.info()

print("\nNumeric summary: df.describe()")
display(df.describe())


# ============================================================================
# SECTION 3: Exploratory Data Analysis (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 3: Exploratory Data Analysis (EDA)")
print("=" * 80)

approval_numeric = df["Loan_Status"].map({"N": 0, "Y": 1})


def approval_rate_plot(column_name: str, title: str) -> None:
    """Plot approval rates across a categorical variable."""
    rate_df = (
        df.assign(Approval=approval_numeric)
        .groupby(column_name, dropna=False)["Approval"]
        .mean()
        .reset_index()
        .sort_values("Approval", ascending=False)
    )
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=rate_df, x=column_name, y="Approval", hue=column_name, palette="viridis", legend=False
    )
    ax.set_title(title)
    ax.set_xlabel(column_name)
    ax.set_ylabel("Approval Rate")
    ax.set_ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    plt.tight_layout()
    plt.show()


print("Target variable distribution")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x="Loan_Status", order=["Y", "N"], palette="Set2")
ax.set_title("Target Variable Distribution")
ax.set_xlabel("Loan Status")
ax.set_ylabel("Count")
for container in ax.containers:
    ax.bar_label(container)
plt.tight_layout()
plt.show()

print("Loan approval rate by Gender")
approval_rate_plot("Gender", "Loan Approval Rate by Gender")

print("Loan approval rate by Education")
approval_rate_plot("Education", "Loan Approval Rate by Education")

print("Loan approval rate by Credit_History")
approval_rate_plot("Credit_History", "Loan Approval Rate by Credit History")

print("Loan approval rate by Property_Area")
approval_rate_plot("Property_Area", "Loan Approval Rate by Property Area")

print("ApplicantIncome distribution")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=df, x="ApplicantIncome", kde=True, bins=30, ax=axes[0], color="#2a9d8f")
axes[0].set_title("ApplicantIncome Distribution")
axes[0].set_xlabel("Applicant Income")
axes[0].set_ylabel("Frequency")
sns.boxplot(data=df, y="ApplicantIncome", ax=axes[1], color="#e9c46a")
axes[1].set_title("ApplicantIncome Boxplot")
axes[1].set_ylabel("Applicant Income")
plt.tight_layout()
plt.show()

print("LoanAmount distribution")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=df, x="LoanAmount", kde=True, bins=30, ax=axes[0], color="#457b9d")
axes[0].set_title("LoanAmount Distribution")
axes[0].set_xlabel("Loan Amount")
axes[0].set_ylabel("Frequency")
sns.boxplot(data=df, y="LoanAmount", ax=axes[1], color="#f4a261")
axes[1].set_title("LoanAmount Boxplot")
axes[1].set_ylabel("Loan Amount")
plt.tight_layout()
plt.show()

print("Correlation heatmap of numerical features")
corr_df = df.copy()
corr_df["Loan_Status"] = corr_df["Loan_Status"].map({"N": 0, "Y": 1})
numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Loan_Status",
]
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    corr_df[numerical_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("Pairplot of key numerical features colored by Loan_Status")
pairplot_sample = corr_df[
    ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History", "Loan_Status"]
].sample(n=min(200, len(corr_df)), random_state=RANDOM_STATE)
pairplot_sample["Loan_Status_Label"] = pairplot_sample["Loan_Status"].map({0: "N", 1: "Y"})
sns.pairplot(
    pairplot_sample,
    vars=["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History"],
    hue="Loan_Status_Label",
    corner=True,
    diag_kind="hist",
    plot_kws={"alpha": 0.7, "s": 35},
)
plt.suptitle("Pairplot of Key Numerical Features", y=1.02)
plt.show()


# ============================================================================
# SECTION 4: Data Preprocessing
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 4: Data Preprocessing")
print("=" * 80)

print("Missing values before preprocessing")
missing_counts = df.isna().sum().sort_values(ascending=False)
print(missing_counts[missing_counts > 0].to_string())

df_processed = df.copy()

categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
target_col = "Loan_Status"

print("\nImputing categorical columns with mode and numerical columns with median.")
for column in categorical_cols:
    df_processed[column] = df_processed[column].fillna(df_processed[column].mode(dropna=True)[0])
for column in numeric_cols:
    df_processed[column] = df_processed[column].fillna(df_processed[column].median())
df_processed[target_col] = df_processed[target_col].fillna(df_processed[target_col].mode(dropna=True)[0])

print("Creating TotalIncome and applying log transforms.")
df_processed["TotalIncome"] = df_processed["ApplicantIncome"] + df_processed["CoapplicantIncome"]
df_processed["LoanAmount"] = np.log1p(df_processed["LoanAmount"])
df_processed["TotalIncome"] = np.log1p(df_processed["TotalIncome"])

print("Encoding categorical variables using LabelEncoder.")
label_encoders = {}
for column in categorical_cols + [target_col]:
    encoder = LabelEncoder()
    df_processed[column] = encoder.fit_transform(df_processed[column].astype(str))
    label_encoders[column] = encoder

print("Dropping Loan_ID and preparing X/y.")
df_processed = df_processed.drop(columns=["Loan_ID"])
feature_columns = [column for column in df_processed.columns if column != target_col]
X = df_processed[feature_columns]
y = df_processed[target_col]

print("Feature columns used for modeling:")
print(feature_columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print("Feature scaling completed with StandardScaler.")


def preprocess_raw_application(raw_application: pd.DataFrame) -> pd.DataFrame:
    """Preprocess a raw application exactly as training data was preprocessed."""
    prepared = raw_application.copy()
    if "Loan_ID" not in prepared.columns:
        prepared["Loan_ID"] = "LP0000"
    for column in categorical_cols:
        if column in prepared.columns:
            prepared[column] = prepared[column].fillna(df[column].mode(dropna=True)[0])
    for column in numeric_cols:
        if column in prepared.columns:
            prepared[column] = prepared[column].fillna(df[column].median())
    prepared["TotalIncome"] = prepared["ApplicantIncome"] + prepared["CoapplicantIncome"]
    prepared["LoanAmount"] = np.log1p(prepared["LoanAmount"])
    prepared["TotalIncome"] = np.log1p(prepared["TotalIncome"])
    for column in categorical_cols:
        prepared[column] = label_encoders[column].transform(prepared[column].astype(str))
    prepared = prepared.drop(columns=["Loan_ID"], errors="ignore")
    return prepared[feature_columns]


# ============================================================================
# SECTION 5: Model Training
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 5: Model Training")
print("=" * 80)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Support Vector Machine (SVM)": SVC(probability=True, random_state=RANDOM_STATE),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
}

model_results = {}
model_predictions = {}
trained_models = {}

for model_name, model in models.items():
    print("\n" + "=" * 80)
    print(f"Training {model_name}")
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    model_results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }
    model_predictions[model_name] = predictions
    trained_models[model_name] = model

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

print("\n" + "=" * 80)
print("Model training complete.")


# ============================================================================
# SECTION 6: Model Evaluation
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 6: Model Evaluation")
print("=" * 80)

print("Classification reports for all models")
for model_name, predictions in model_predictions.items():
    print("\n" + "=" * 80)
    print(f"Classification report for {model_name}")
    print(
        classification_report(
            y_test, predictions, target_names=["Not Approved", "Approved"], zero_division=0
        )
    )

print("\nConfusion matrix heatmaps")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for index, (model_name, predictions) in enumerate(model_predictions.items()):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[index],
        xticklabels=["Pred N", "Pred Y"],
        yticklabels=["True N", "True Y"],
    )
    axes[index].set_title(model_name)
    axes[index].set_xlabel("Predicted Label")
    axes[index].set_ylabel("True Label")
plt.tight_layout()
plt.show()

comparison_df = pd.DataFrame(model_results).T.reset_index().rename(columns={"index": "Model"})
comparison_df = comparison_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)

print("Comparison DataFrame")
display(comparison_df)

print("Grouped bar chart of model metrics")
comparison_melted = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(14, 7))
ax = sns.barplot(data=comparison_melted, x="Model", y="Score", hue="Metric", palette="tab10")
ax.set_title("Model Comparison Across Metrics")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

best_model_name = comparison_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_model_metrics = comparison_df.iloc[0].to_dict()

print(f"\nBEST MODEL by F1-score: {best_model_name}")
print("Best model metrics:")
print(best_model_metrics)

cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring="f1_weighted")
print(f"\n5-fold CV for {best_model_name}")
print(f"Mean F1-score: {cv_scores.mean():.4f}")
print(f"Std F1-score:  {cv_scores.std():.4f}")
print(f"Mean ± Std:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ============================================================================
# SECTION 7: Feature Importance
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 7: Feature Importance")
print("=" * 80)

if hasattr(best_model, "feature_importances_"):
    feature_importances = best_model.feature_importances_
    importance_label = "feature_importances_"
elif hasattr(best_model, "coef_"):
    feature_importances = np.abs(best_model.coef_).ravel()
    importance_label = "absolute coefficients"
else:
    print(f"{best_model_name} does not expose feature_importances_ or coef_. Using permutation importance.")
    permutation_result = permutation_importance(
        best_model,
        X_test_scaled,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1_weighted",
    )
    feature_importances = permutation_result.importances_mean
    importance_label = "permutation importance"

importance_df = (
    pd.DataFrame({"Feature": feature_columns, "Importance": feature_importances})
    .sort_values("Importance", ascending=False)
    .head(10)
)

print(f"Top 10 important features using {importance_label}")
display(importance_df)

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=importance_df, x="Importance", y="Feature", palette="crest")
ax.set_title(f"Top 10 Feature Importance - {best_model_name}")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.show()

top_3_features = importance_df.head(3)["Feature"].tolist()
print("Top 3 features:")
print(top_3_features)


# ============================================================================
# SECTION 8: Hyperparameter Tuning
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 8: Hyperparameter Tuning")
print("=" * 80)

param_grids = {
    "Logistic Regression": {
        "C": [0.1, 1, 10, 25],
        "solver": ["liblinear"],
        "penalty": ["l2"],
    },
    "Decision Tree Classifier": {
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },
    "Random Forest Classifier": {
        "n_estimators": [100, 150, 200],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Gradient Boosting Classifier": {
        "n_estimators": [80, 100, 150],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
    },
    "Support Vector Machine (SVM)": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "K-Nearest Neighbors (KNN)": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
}

tuning_grid = param_grids[best_model_name]
print(f"Tuning {best_model_name} with GridSearchCV")
grid_search = GridSearchCV(
    estimator=trained_models[best_model_name],
    param_grid=tuning_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1,
)
grid_search.fit(X_train_scaled, y_train)

tuned_model = grid_search.best_estimator_
tuned_predictions = tuned_model.predict(X_test_scaled)
tuned_metrics = {
    "Accuracy": accuracy_score(y_test, tuned_predictions),
    "Precision": precision_score(y_test, tuned_predictions, average="weighted", zero_division=0),
    "Recall": recall_score(y_test, tuned_predictions, average="weighted", zero_division=0),
    "F1-Score": f1_score(y_test, tuned_predictions, average="weighted", zero_division=0),
}

print("Best parameters:")
print(grid_search.best_params_)
print(f"Best cross-val score: {grid_search.best_score_:.4f}")

print("\nTuned model test metrics")
for metric_name, metric_value in tuned_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

print("\nImprovement over the original best model")
for metric_name in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    improvement = tuned_metrics[metric_name] - best_model_metrics[metric_name]
    print(f"{metric_name}: {improvement:+.4f}")

final_model_name = f"{best_model_name} (tuned)"
final_model = tuned_model
final_metrics = tuned_metrics
print(f"\nFinal model selected for saving: {final_model_name}")


# ============================================================================
# SECTION 9: Save the Model
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 9: Save the Model")
print("=" * 80)

joblib.dump(final_model, "best_loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("Saved artifacts:")
print("- best_loan_model.pkl")
print("- scaler.pkl")
print("- feature_columns.pkl")

loaded_model = joblib.load("best_loan_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
loaded_feature_columns = joblib.load("feature_columns.pkl")

sample_application = pd.DataFrame(
    [
        {
            "Loan_ID": "LP9999",
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "1",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 6500,
            "CoapplicantIncome": 1800,
            "LoanAmount": 150,
            "Loan_Amount_Term": 360,
            "Credit_History": 1.0,
            "Property_Area": "Semiurban",
            "Loan_Status": "Y",
        }
    ]
)

processed_sample = preprocess_raw_application(sample_application)
processed_sample = processed_sample[loaded_feature_columns]
scaled_sample = loaded_scaler.transform(processed_sample)
sample_prediction = loaded_model.predict(scaled_sample)[0]
sample_probability = loaded_model.predict_proba(scaled_sample)[0][1]
sample_label = label_encoders["Loan_Status"].inverse_transform([sample_prediction])[0]

print("\nSample prediction demo")
print(f"Predicted encoded label: {sample_prediction}")
print(f"Predicted label: {sample_label}")
print(f"Approved probability: {sample_probability:.4f}")


# ============================================================================
# SECTION 10: Streamlit App Code
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 10: Streamlit App Code")
print("=" * 80)

streamlit_code = """import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Approval Prediction System", page_icon="🏦", layout="centered")

MODEL_PATH = "best_loan_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"

CATEGORY_MAPS = {
    "Gender": {"Female": 0, "Male": 1},
    "Married": {"No": 0, "Yes": 1},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
    "Loan_Status": {"N": 0, "Y": 1},
}

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, scaler, feature_columns

def preprocess_input(raw_data, feature_columns):
    frame = pd.DataFrame([raw_data])
    frame["TotalIncome"] = frame["ApplicantIncome"] + frame["CoapplicantIncome"]
    frame["LoanAmount"] = np.log1p(frame["LoanAmount"])
    frame["TotalIncome"] = np.log1p(frame["TotalIncome"])

    for column, mapping in CATEGORY_MAPS.items():
        if column in frame.columns:
            frame[column] = frame[column].map(mapping)

    frame = frame.drop(columns=["Loan_ID"], errors="ignore")
    frame = frame[feature_columns]
    return frame

def predict_application(model, scaler, feature_columns, raw_data):
    processed = preprocess_input(raw_data, feature_columns)
    scaled = scaler.transform(processed)
    prediction = model.predict(scaled)[0]
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled)[0][1]
    else:
        score = model.decision_function(scaled)[0]
        probability = 1 / (1 + np.exp(-score))
    return prediction, probability

def main():
    st.title("🏦 Loan Approval Prediction System")
    st.write("Enter applicant details in the sidebar to predict whether a loan will be approved.")

    model, scaler, feature_columns = load_artifacts()

    st.sidebar.header("Applicant Information")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Married", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000, step=500)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0, step=250)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=1, value=128, step=1)
    loan_term = st.sidebar.selectbox("Loan Amount Term", [12, 36, 60, 120, 180, 240, 300, 360, 480], index=7)
    credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0], format_func=lambda value: "1.0 - Yes" if value == 1.0 else "0.0 - No")
    property_area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    if st.button("Predict Loan Status"):
        input_payload = {
            "Loan_ID": "USER_INPUT",
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Loan_Status": "Y",
        }

        prediction, probability = predict_application(model, scaler, feature_columns, input_payload)
        approved_probability = float(probability if prediction == 1 else 1 - probability)

        if prediction == 1:
            st.markdown(
                "<h1 style='color: #1b8a5a; font-size: 44px;'>✅ APPROVED</h1>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<h1 style='color: #c1121f; font-size: 44px;'>❌ NOT APPROVED</h1>",
                unsafe_allow_html=True,
            )

        st.progress(min(max(approved_probability, 0.0), 1.0))
        st.write(f"Prediction probability: {approved_probability:.2%}")
        st.write("The model uses the same preprocessing steps as the notebook: TotalIncome, log transforms, category encoding, and StandardScaler.")

    st.markdown("---")
    st.info("Run this app with: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
"""

with open("streamlit_app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_code)

print("Wrote streamlit_app.py")


# ============================================================================
# SECTION 11: Project Summary
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 11: Project Summary")
print("=" * 80)

summary_md = f"""
## Project Summary

**Best model:** {final_model_name}

**Test metrics:**
- Accuracy: {final_metrics['Accuracy']:.4f}
- Precision: {final_metrics['Precision']:.4f}
- Recall: {final_metrics['Recall']:.4f}
- F1-score: {final_metrics['F1-Score']:.4f}

**Top 3 most important features:**
1. {top_3_features[0] if len(top_3_features) > 0 else 'N/A'}
2. {top_3_features[1] if len(top_3_features) > 1 else 'N/A'}
3. {top_3_features[2] if len(top_3_features) > 2 else 'N/A'}

**Key EDA insights:**
- Credit history is typically the strongest signal for approval in this dataset.
- Approval rates are usually higher for applicants with stronger income profiles and favorable property-area patterns.
- Loan amount and income distributions are right-skewed, so the log transforms help stabilize the modeling step.

**Limitations and future improvements:**
- The synthetic fallback is useful for reproducibility, but the Kaggle dataset should be preferred for production-quality analysis.
- A full pipeline with saved encoders would make deployment even more robust.
- Future work could add calibration, SHAP-based explainability, and more extensive hyperparameter search.
"""

print(summary_md)

print("\n" + "=" * 80)
print("NOTEBOOK EXECUTION COMPLETE")
print("=" * 80)
