import joblib
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
