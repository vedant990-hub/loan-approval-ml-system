import json

# Create a fresh notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "md-001",
            "metadata": {},
            "source": ["# Section 1: Setup and Configuration\n", "Initialize the environment with required libraries and configurations."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-001",
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib\n",
                "matplotlib.use('Agg')\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
                "from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.tree import DecisionTreeClassifier\n",
                "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
                "from sklearn.svm import SVC\n",
                "from sklearn.neighbors import KNeighborsClassifier\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
                "from sklearn.metrics import confusion_matrix, classification_report\n",
                "import joblib\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "RANDOM_STATE = 42\n",
                "np.random.seed(RANDOM_STATE)\n",
                "print(f'Random seed set to: {RANDOM_STATE}')\n",
                "\n",
                "sns.set_style('whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (12, 6)\n",
                "print('Plot style configured and warnings suppressed.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "md-002",
            "metadata": {},
            "source": ["# Section 2: Load Dataset\n", "Load the loan approval dataset and display basic information."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-002",
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_csv('loan_approval_dataset.csv')\n",
                "print(f'Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')\n",
                "print(f'\\nColumn names and types:')\n",
                "print(df.dtypes)\n",
                "print(f'\\nFirst 5 rows:')\n",
                "print(df.head())\n",
                "print(f'\\nMissing values:')\n",
                "print(df.isnull().sum())\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "md-003",
            "metadata": {},
            "source": ["# Section 3: Exploratory Data Analysis\n", "Visualize distributions and relationships in the data."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-003",
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                "df['loan_status'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'orange'])\n",
                "axes[0, 0].set_title('Loan Status Distribution')\n",
                "axes[0, 1].hist(df['income_annum'], bins=50, color='teal', edgecolor='black')\n",
                "axes[0, 1].set_title('Income Distribution')\n",
                "axes[1, 0].hist(df['loan_amount'], bins=50, color='purple', edgecolor='black')\n",
                "axes[1, 0].set_title('Loan Amount Distribution')\n",
                "axes[1, 1].hist(df['cibil_score'], bins=50, color='brown', edgecolor='black')\n",
                "axes[1, 1].set_title('CIBIL Score Distribution')\n",
                "plt.tight_layout()\n",
                "plt.savefig('eda.png', dpi=100, bbox_inches='tight')\n",
                "plt.show()\n",
                "print('EDA visualizations saved.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "md-004",
            "metadata": {},
            "source": ["# Section 4: Data Preprocessing\n", "Clean and transform the data for modeling."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-004",
            "metadata": {},
            "outputs": [],
            "source": [
                "df.columns = df.columns.str.strip()\n",
                "for col in df.select_dtypes(include='object').columns:\n",
                "    df[col] = df[col].str.strip()\n",
                "\n",
                "print('Data preprocessing started.')\n",
                "df['education'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})\n",
                "df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})\n",
                "df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})\n",
                "\n",
                "X = df.drop(['loan_id', 'loan_status'], axis=1)\n",
                "y = df['loan_status']\n",
                "\n",
                "feature_columns = X.columns.tolist()\n",
                "print(f'Feature columns: {feature_columns}')\n",
                "print(f'Features shape: {X.shape}')\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)\n",
                "\n",
                "scaler = StandardScaler()\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "print(f'Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}')\n",
                "print('Preprocessing completed.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "md-005",
            "metadata": {},
            "source": ["# Section 5: Model Training\n", "Train 6 classifiers and compare their performance."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-005",
            "metadata": {},
            "outputs": [],
            "source": [
                "models = {\n",
                "    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),\n",
                "    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),\n",
                "    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),\n",
                "    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),\n",
                "    'SVM': SVC(random_state=RANDOM_STATE, probability=True),\n",
                "    'KNN': KNeighborsClassifier()\n",
                "}\n",
                "\n",
                "results = {}\n",
                "for name, model in models.items():\n",
                "    model.fit(X_train_scaled, y_train)\n",
                "    y_pred = model.predict(X_test_scaled)\n",
                "    results[name] = {\n",
                "        'accuracy': accuracy_score(y_test, y_pred),\n",
                "        'precision': precision_score(y_test, y_pred),\n",
                "        'recall': recall_score(y_test, y_pred),\n",
                "        'f1': f1_score(y_test, y_pred)\n",
                "    }\n",
                "    print(f'{name}: Accuracy={results[name][\"accuracy\"]:.4f}')\n",
                "\n",
                "results_df = pd.DataFrame(results).T\n",
                "print('\\nModel Comparison:')\n",
                "print(results_df)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "md-006",
            "metadata": {},
            "source": ["# Section 6: Model Evaluation\n", "Detailed evaluation of best performing model."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-006",
            "metadata": {},
            "outputs": [],
            "source": [
                "best_model_name = results_df['accuracy'].idxmax()\n",
                "best_model = models[best_model_name]\n",
                "\n",
                "y_pred = best_model.predict(X_test_scaled)\n",
                "\n",
                "print(f'Best Model: {best_model_name}')\n",
                "print(f'\\nClassification Report:\\n')\n",
                "print(classification_report(y_test, y_pred))\n",
                "\n",
                "cm = confusion_matrix(y_test, y_pred)\n",
                "plt.figure(figsize=(8, 6))\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)\n",
                "plt.title(f'Confusion Matrix - {best_model_name}')\n",
                "plt.ylabel('Actual')\n",
                "plt.xlabel('Predicted')\n",
                "plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')\n",
                "plt.show()\n",
                "print('Confusion matrix visualization saved.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "md-007",
            "metadata": {},
            "source": ["# Section 7: Feature Importance\n", "Extract and visualize feature importance."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-007",
            "metadata": {},
            "outputs": [],
            "source": [
                "if hasattr(best_model, 'feature_importances_'):\n",
                "    importances = best_model.feature_importances_\n",
                "    importance_df = pd.DataFrame({\n",
                "        'feature': feature_columns,\n",
                "        'importance': importances\n",
                "    }).sort_values('importance', ascending=False)\n",
                "    \n",
                "    print('Top 5 Important Features:')\n",
                "    print(importance_df.head())\n",
                "    \n",
                "    plt.figure(figsize=(10, 6))\n",
                "    plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])\n",
                "    plt.xlabel('Importance')\n",
                "    plt.title(f'Top 10 Feature Importances - {best_model_name}')\n",
                "    plt.tight_layout()\n",
                "    plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')\n",
                "    plt.show()\n",
                "else:\n",
                "    print(f'{best_model_name} does not have feature importance attribute.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "md-008",
            "metadata": {},
            "source": ["# Section 8: Hyperparameter Tuning\n", "Tune best model using GridSearchCV."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-008",
            "metadata": {},
            "outputs": [],
            "source": [
                "if best_model_name == 'Random Forest':\n",
                "    param_grid = {\n",
                "        'n_estimators': [50, 100],\n",
                "        'max_depth': [None, 10],\n",
                "        'min_samples_split': [2, 5]\n",
                "    }\n",
                "    grid_search = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), param_grid, cv=3)\n",
                "    grid_search.fit(X_train_scaled, y_train)\n",
                "    print(f'Best parameters: {grid_search.best_params_}')\n",
                "    print(f'Best cross-validation score: {grid_search.best_score_:.4f}')\n",
                "else:\n",
                "    print('Hyperparameter tuning demonstration skipped for non-RF model')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "md-009",
            "metadata": {},
            "source": ["# Section 9: Cross-Validation\n", "Evaluate model stability with k-fold cross-validation."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-009",
            "metadata": {},
            "outputs": [],
            "source": [
                "cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)\n",
                "print(f'Cross-validation scores: {cv_scores}')\n",
                "print(f'Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "md-010",
            "metadata": {},
            "source": ["# Section 10: Save Artifacts\n", "Persist the trained model and preprocessing objects."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-010",
            "metadata": {},
            "outputs": [],
            "source": [
                "joblib.dump(best_model, 'best_loan_model.pkl')\n",
                "joblib.dump(scaler, 'scaler.pkl')\n",
                "joblib.dump(feature_columns, 'feature_columns.pkl')\n",
                "\n",
                "print('Model artifacts saved:')\n",
                "print('  - best_loan_model.pkl')\n",
                "print('  - scaler.pkl')\n",
                "print('  - feature_columns.pkl')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "md-011",
            "metadata": {},
            "source": ["# Section 11: Sample Prediction\n", "Demonstrate making predictions on new data."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-011",
            "metadata": {},
            "outputs": [],
            "source": [
                "sample_app = pd.DataFrame({\n",
                "    'no_of_dependents': [2],\n",
                "    'education': [1],\n",
                "    'self_employed': [0],\n",
                "    'income_annum': [6500000],\n",
                "    'loan_amount': [500000],\n",
                "    'loan_term': [360],\n",
                "    'cibil_score': [750],\n",
                "    'residential_assets_value': [2500000],\n",
                "    'commercial_assets_value': [1500000],\n",
                "    'luxury_assets_value': [1000000],\n",
                "    'bank_asset_value': [300000]\n",
                "})\n",
                "\n",
                "sample_scaled = scaler.transform(sample_app)\n",
                "prediction = best_model.predict(sample_scaled)[0]\n",
                "probability = best_model.predict_proba(sample_scaled)[0]\n",
                "\n",
                "status = 'Approved' if prediction == 1 else 'Rejected'\n",
                "print(f'Prediction: {status}')\n",
                "print(f'Probability - Rejected: {probability[0]:.2%}, Approved: {probability[1]:.2%}')\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Write the notebook
with open('loan_approval_prediction.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print('✓ Fresh notebook created successfully!')
