"""
Loan Approval Prediction Pipeline for loan_approval_dataset.csv

This script runs the full machine learning workflow on the attached loan approval
 dataset. It loads the real CSV, performs EDA, trains multiple classifiers,
tunes the best model, saves artifacts, and writes a Streamlit app.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_PATH = Path("loan_approval_dataset.csv")
MODEL_PATH = Path("best_loan_model.pkl")
SCALER_PATH = Path("scaler.pkl")
FEATURE_COLUMNS_PATH = Path("feature_columns.pkl")
STREAMLIT_PATH = Path("streamlit_app.py")

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


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# SECTION 1: Project Setup & Library Imports
# ============================================================================

np.random.seed(RANDOM_STATE)
print_section("SECTION 1: Project Setup & Library Imports")
print(f"Random seed set to: {RANDOM_STATE}")
print("Matplotlib Agg backend configured for non-interactive execution.")


# ============================================================================
# SECTION 2: Dataset Loading
# ============================================================================

print_section("SECTION 2: Dataset Loading")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Place loan_approval_dataset.csv in the project root."
    )

raw_df = pd.read_csv(DATA_PATH)
raw_df.columns = raw_df.columns.str.strip()
for column in raw_df.select_dtypes(include="object").columns:
    raw_df[column] = raw_df[column].astype(str).str.strip()

print(f"Loaded dataset from: {DATA_PATH}")
print("Data preview:")
display(raw_df.head())
print("\nData shape:")
print(raw_df.shape)
print("\nData info:")
raw_df.info()
print("\nNumeric summary:")
display(raw_df.describe())


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared.columns = prepared.columns.str.strip()
    for column in prepared.select_dtypes(include="object").columns:
        prepared[column] = prepared[column].astype(str).str.strip()
    prepared["loan_status"] = prepared["loan_status"].map({"Approved": 1, "Rejected": 0})
    prepared["education"] = prepared["education"].map({"Graduate": 1, "Not Graduate": 0})
    prepared["self_employed"] = prepared["self_employed"].map({"Yes": 1, "No": 0})
    prepared["no_of_dependents"] = pd.to_numeric(prepared["no_of_dependents"], errors="coerce")
    numeric_columns = [
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


# ============================================================================
# SECTION 3: Exploratory Data Analysis (EDA)
# ============================================================================

print_section("SECTION 3: Exploratory Data Analysis (EDA)")

eda_df = prepare_dataframe(raw_df)


def approval_rate_plot(column_name: str, title: str, x_labels: dict[int, str] | None = None) -> None:
    rate_df = (
        eda_df.groupby(column_name, dropna=False)["loan_status"]
        .mean()
        .reset_index()
        .sort_values("loan_status", ascending=False)
    )
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=rate_df, x=column_name, y="loan_status", palette="viridis")
    ax.set_title(title)
    ax.set_xlabel(column_name.replace("_", " ").title())
    ax.set_ylabel("Approval Rate")
    ax.set_ylim(0, 1)
    if x_labels:
        ax.set_xticklabels([x_labels.get(value, str(value)) for value in rate_df[column_name]])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    plt.tight_layout()
    plt.show()

print("Target variable distribution")
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=eda_df, x="loan_status", order=[1, 0], palette="Set2")
ax.set_title("Target Variable Distribution")
ax.set_xlabel("Loan Status")
ax.set_ylabel("Count")
ax.set_xticklabels(["Approved", "Rejected"])
for container in ax.containers:
    ax.bar_label(container)
plt.tight_layout()
plt.show()

print("Approval rate by education")
approval_rate_plot("education", "Approval Rate by Education", {1: "Graduate", 0: "Not Graduate"})

print("Approval rate by self_employed")
approval_rate_plot("self_employed", "Approval Rate by Self Employment", {1: "Yes", 0: "No"})

print("Approval rate by cibil score bands")
eda_df["cibil_band"] = pd.cut(
    eda_df["cibil_score"],
    bins=[0, 350, 450, 550, 650, 750, 900],
    include_lowest=True,
)
approval_rate_plot("cibil_band", "Approval Rate by CIBIL Score Band")

print("Income annuum distribution")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=eda_df, x="income_annum", kde=True, bins=30, ax=axes[0], color="#2a9d8f")
axes[0].set_title("Income Annum Distribution")
axes[0].set_xlabel("Income Annum")
axes[0].set_ylabel("Frequency")
sns.boxplot(data=eda_df, y="income_annum", ax=axes[1], color="#e9c46a")
axes[1].set_title("Income Annum Boxplot")
axes[1].set_ylabel("Income Annum")
plt.tight_layout()
plt.show()

print("Loan amount distribution")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=eda_df, x="loan_amount", kde=True, bins=30, ax=axes[0], color="#457b9d")
axes[0].set_title("Loan Amount Distribution")
axes[0].set_xlabel("Loan Amount")
axes[0].set_ylabel("Frequency")
sns.boxplot(data=eda_df, y="loan_amount", ax=axes[1], color="#f4a261")
axes[1].set_title("Loan Amount Boxplot")
axes[1].set_ylabel("Loan Amount")
plt.tight_layout()
plt.show()

print("Loan term distribution")
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=eda_df, x="loan_term", palette="mako")
ax.set_title("Loan Term Distribution")
ax.set_xlabel("Loan Term")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

print("Correlation heatmap of numerical features")
correlation_columns = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "loan_status",
]
plt.figure(figsize=(10, 8))
ax = sns.heatmap(eda_df[correlation_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("Pairplot of key numerical features colored by loan_status")
pairplot_sample = eda_df[
    ["income_annum", "loan_amount", "loan_term", "cibil_score", "loan_status"]
].sample(n=min(250, len(eda_df)), random_state=RANDOM_STATE)
pairplot_sample["loan_status_label"] = pairplot_sample["loan_status"].map({1: "Approved", 0: "Rejected"})
sns.pairplot(
    pairplot_sample,
    vars=["income_annum", "loan_amount", "loan_term", "cibil_score"],
    hue="loan_status_label",
    corner=True,
    diag_kind="hist",
    plot_kws={"alpha": 0.7, "s": 35},
)
plt.suptitle("Pairplot of Key Numerical Features", y=1.02)
plt.show()


# ============================================================================
# SECTION 4: Data Preprocessing
# ============================================================================

print_section("SECTION 4: Data Preprocessing")

print("Missing values before preprocessing")
missing_counts = eda_df.isna().sum().sort_values(ascending=False)
print(missing_counts[missing_counts > 0].to_string() if (missing_counts > 0).any() else "No missing values found.")

model_df = eda_df.copy()
model_df = model_df.drop(columns=["loan_id", "cibil_band"], errors="ignore")
model_df["loan_status"] = model_df["loan_status"].astype(int)

feature_columns = [column for column in model_df.columns if column != "loan_status"]
X = model_df[feature_columns]
y = model_df["loan_status"]

print("Feature columns used for modeling:")
print(feature_columns)
print("Target balance:")
print(y.value_counts(normalize=True).rename({1: "Approved", 0: "Rejected"}).round(4).to_string())

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

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print("Feature scaling completed with StandardScaler.")


def preprocess_raw_application(raw_application: pd.DataFrame) -> pd.DataFrame:
    prepared = raw_application.copy()
    prepared.columns = prepared.columns.str.strip()
    prepared["education"] = prepared["education"].map({"Graduate": 1, "Not Graduate": 0})
    prepared["self_employed"] = prepared["self_employed"].map({"Yes": 1, "No": 0})
    numeric_columns = [
        "no_of_dependents",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared[feature_columns]


# ============================================================================
# SECTION 5: Model Training
# ============================================================================

print_section("SECTION 5: Model Training")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
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
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

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

print("\nModel training complete.")


# ============================================================================
# SECTION 6: Model Evaluation
# ============================================================================

print_section("SECTION 6: Model Evaluation")

for model_name, predictions in model_predictions.items():
    print("\n" + "=" * 80)
    print(f"Classification report for {model_name}")
    print(classification_report(y_test, predictions, target_names=["Rejected", "Approved"], zero_division=0))

print("Confusion matrix heatmaps")
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
        xticklabels=["Pred Rejected", "Pred Approved"],
        yticklabels=["True Rejected", "True Approved"],
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

print(f"Best model by F1-score: {best_model_name}")
print(best_model_metrics)

cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring="f1")
print(f"5-fold CV mean F1-score: {cv_scores.mean():.4f}")
print(f"5-fold CV std F1-score:  {cv_scores.std():.4f}")


# ============================================================================
# SECTION 7: Feature Importance
# ============================================================================

print_section("SECTION 7: Feature Importance")

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
        scoring="f1",
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

print_section("SECTION 8: Hyperparameter Tuning")

param_grids = {
    "Logistic Regression": {
        "C": [0.1, 1, 10],
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
        "n_estimators": [100, 200],
        "max_depth": [None, 8, 12],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
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
        "n_neighbors": [3, 5, 7, 9],
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
    scoring="f1",
    n_jobs=-1,
)
grid_search.fit(X_train_scaled, y_train)

tuned_model = grid_search.best_estimator_
print("Best parameters:")
print(grid_search.best_params_)
print(f"Best cross-val score: {grid_search.best_score_:.4f}")

tuned_predictions = tuned_model.predict(X_test_scaled)
tuned_accuracy = accuracy_score(y_test, tuned_predictions)
tuned_precision = precision_score(y_test, tuned_predictions, zero_division=0)
tuned_recall = recall_score(y_test, tuned_predictions, zero_division=0)
tuned_f1 = f1_score(y_test, tuned_predictions, zero_division=0)

print("Tuned model test metrics")
print(f"Accuracy:  {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall:    {tuned_recall:.4f}")
print(f"F1-score:  {tuned_f1:.4f}")

print("Improvement over the original best model")
print(f"Accuracy:  {tuned_accuracy - best_model_metrics['Accuracy']:+.4f}")
print(f"Precision: {tuned_precision - best_model_metrics['Precision']:+.4f}")
print(f"Recall:    {tuned_recall - best_model_metrics['Recall']:+.4f}")
print(f"F1-score:  {tuned_f1 - best_model_metrics['F1-Score']:+.4f}")

final_model = tuned_model
final_model_name = f"{best_model_name} (tuned)"
print(f"Final model selected for saving: {final_model_name}")


# ============================================================================
# SECTION 9: Save the Model
# ============================================================================

print_section("SECTION 9: Save the Model")

trained_models[final_model_name] = final_model
ALL_MODELS_PATH = Path("all_models.pkl")
FEATURE_IMPORTANCE_PATH = Path("feature_importance.pkl")

joblib.dump(final_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
joblib.dump(trained_models, ALL_MODELS_PATH)
joblib.dump(importance_df, FEATURE_IMPORTANCE_PATH)

print("Saved artifacts:")
print(f"- {MODEL_PATH}")
print(f"- {SCALER_PATH}")
print(f"- {FEATURE_COLUMNS_PATH}")
print(f"- {ALL_MODELS_PATH}")
print(f"- {FEATURE_IMPORTANCE_PATH}")

sample_application = pd.DataFrame(
    [
        {
            "no_of_dependents": 2,
            "education": 1,
            "self_employed": 0,
            "income_annum": 8500000,
            "loan_amount": 28000000,
            "loan_term": 12,
            "cibil_score": 760,
            "residential_assets_value": 12000000,
            "commercial_assets_value": 7000000,
            "luxury_assets_value": 24000000,
            "bank_asset_value": 7000000,
        }
    ]
)
sample_scaled = scaler.transform(sample_application[feature_columns])
sample_prediction = final_model.predict(sample_scaled)[0]
sample_probability = final_model.predict_proba(sample_scaled)[0][1]

print("Sample prediction demo")
print(f"Predicted encoded label: {sample_prediction}")
print(f"Predicted label: {'Approved' if sample_prediction == 1 else 'Rejected'}")
print(f"Approved probability: {sample_probability:.4f}")


# ============================================================================
# SECTION 10: Streamlit App Code
# ============================================================================

print_section("SECTION 10: Streamlit App Code")

streamlit_code = '''import joblib
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Loan Approval System", page_icon="🏦", layout="wide")

MODEL_PATH = "best_loan_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
ALL_MODELS_PATH = "all_models.pkl"
FEATURE_IMPORTANCE_PATH = "feature_importance.pkl"
DATASET_PATH = "loan_approval_dataset.csv"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    all_models = joblib.load(ALL_MODELS_PATH)
    feature_importance = joblib.load(FEATURE_IMPORTANCE_PATH)
    return model, scaler, feature_columns, all_models, feature_importance

@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df

model, scaler, feature_columns, all_models, feature_importance = load_artifacts()
raw_df = load_data()

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a view:", ["Prediction System", "Exploratory Data Analysis (EDA)"])

if app_mode == "Prediction System":
    st.title("🏦 Loan Approval Prediction System")
    st.caption("Predict approval outcomes using applicant income, CIBIL score, and asset information.")

    with st.sidebar:
        st.header("Applicant Details")
        no_of_dependents = st.number_input("Number of dependents", min_value=0, max_value=10, value=0, step=1)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self employed", ["No", "Yes"])
        income_annum = st.number_input("Annual income", min_value=0, value=5000000, step=100000)
        loan_amount = st.number_input("Loan amount", min_value=0, value=15000000, step=100000)
        loan_term = st.number_input("Loan term (Years)", min_value=1, max_value=40, value=20, step=1)
        cibil_score = st.number_input("CIBIL score", min_value=300, max_value=900, value=700, step=1)
        residential_assets_value = st.number_input("Residential assets value", min_value=0, value=5000000, step=100000)
        commercial_assets_value = st.number_input("Commercial assets value", min_value=0, value=3000000, step=100000)
        luxury_assets_value = st.number_input("Luxury assets value", min_value=0, value=12000000, step=100000)
        bank_asset_value = st.number_input("Bank asset value", min_value=0, value=3000000, step=100000)

        submitted = st.button("Predict Loan Status")

    def encode_inputs() -> pd.DataFrame:
        application = pd.DataFrame(
            [
                {
                    "no_of_dependents": no_of_dependents,
                    "education": 1 if education == "Graduate" else 0,
                    "self_employed": 1 if self_employed == "Yes" else 0,
                    "income_annum": income_annum,
                    "loan_amount": loan_amount,
                    "loan_term": loan_term,
                    "cibil_score": cibil_score,
                    "residential_assets_value": residential_assets_value,
                    "commercial_assets_value": commercial_assets_value,
                    "luxury_assets_value": luxury_assets_value,
                    "bank_asset_value": bank_asset_value,
                }
            ]
        )
        return application[feature_columns]

    if submitted:
        input_df = encode_inputs()
        scaled_input = scaler.transform(input_df)
        
        tab1, tab2, tab3 = st.tabs(["Prediction Result", "Model Comparison", "Feature Importance"])
        
        with tab1:
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

            if prediction == 1:
                st.success(f"Approved with probability {probability:.2%}")
            else:
                st.error(f"Rejected with approval probability {probability:.2%}")

            st.subheader("Prediction Details")
            st.write(input_df)
            st.progress(float(probability))
            
        with tab2:
            st.subheader("Model Probability Comparison")
            st.write("Compare how different machine learning algorithms interpret this exact application:")
            
            results = []
            for name, m in all_models.items():
                if hasattr(m, "predict_proba"):
                    prob = m.predict_proba(scaled_input)[0][1]
                    results.append({"Model": name, "Approval Probability": float(prob)})
                elif hasattr(m, "decision_function"):
                    score = m.decision_function(scaled_input)[0]
                    prob = 1 / (1 + np.exp(-score))
                    results.append({"Model": name, "Approval Probability": float(prob)})
                    
            if results:
                results_df = pd.DataFrame(results).sort_values(by="Approval Probability", ascending=False)
                st.bar_chart(data=results_df.set_index("Model"), y="Approval Probability", height=400)
                
        with tab3:
            st.subheader("What drove this decision?")
            st.write("These are the most important features the model considers when approving or rejecting a loan:")
            
            # Ensure we sort the feature importance dataframe for the best visual presentation
            sorted_importance = feature_importance.sort_values(by="Importance", ascending=True)
            st.bar_chart(data=sorted_importance.set_index("Feature"), y="Importance", height=400, horizontal=True)

elif app_mode == "Exploratory Data Analysis (EDA)":
    st.title("📊 Exploratory Data Analysis (EDA) Dashboard")
    st.write("Visually explore the real-world dataset that powers our machine learning models.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loan Status Distribution")
        st.write("Are there more approved or rejected loans?")
        status_counts = raw_df["loan_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        st.bar_chart(data=status_counts.set_index("Status"), color="#2a9d8f")
        
    with col2:
        st.subheader("Approval by Education")
        st.write("Does having a degree matter?")
        edu_approval = raw_df.groupby("education")["loan_status"].apply(lambda x: (x == "Approved").mean() * 100).reset_index()
        edu_approval.columns = ["Education", "Approval Rate (%)"]
        st.bar_chart(data=edu_approval.set_index("Education"), color="#e9c46a")

    st.divider()
    
    st.subheader("CIBIL Score vs Loan Amount")
    st.write("How does credit score interact with the amount of money requested? (Colored by Approval Status)")
    st.scatter_chart(raw_df, x="cibil_score", y="loan_amount", color="loan_status", size=50)

    st.divider()
    
    st.subheader("Raw Dataset Preview")
    st.write(f"Total Rows: {len(raw_df)}")
    st.dataframe(raw_df, use_container_width=True)
'''

with open(STREAMLIT_PATH, "w", encoding="utf-8") as streamlit_file:
    streamlit_file.write(streamlit_code)

print(f"Wrote {STREAMLIT_PATH}")


# ============================================================================
# SECTION 11: Project Summary
# ============================================================================

print_section("SECTION 11: Project Summary")

summary_md = f'''
## Project Summary

**Dataset:** `loan_approval_dataset.csv`

**Best model:** {final_model_name}

**Test metrics:**
- Accuracy: {tuned_accuracy:.4f}
- Precision: {tuned_precision:.4f}
- Recall: {tuned_recall:.4f}
- F1-score: {tuned_f1:.4f}

**Top 3 most important features:**
1. {top_3_features[0]}
2. {top_3_features[1]}
3. {top_3_features[2]}

**Key EDA insights:**
- CIBIL score is typically the strongest driver of approval.
- Higher income and stronger asset values improve approval probability.
- Loan amount and loan term interact with income and CIBIL score in the final model.

**Limitations and future improvements:**
- The current pipeline uses the attached CSV directly, which is better than the synthetic fallback for realism.
- Future work could add calibration, SHAP explainability, and probability threshold tuning.
- The Streamlit app can be extended with input validation and batch prediction support.
'''

print(summary_md)
print("\n" + "=" * 80)
print("NOTEBOOK EXECUTION COMPLETE")
print("=" * 80)


if __name__ == "__main__":
    pass
