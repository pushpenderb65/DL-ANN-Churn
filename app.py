import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# Configure page
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")


# Load assets with caching
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("churn_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


model, scaler = load_assets()

st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Enter customer demographic and account details to assess churn risk."
)

# User inputs
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=900, value=650
    )
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=3)

with col2:
    balance = st.number_input(
        "Account Balance", min_value=0.0, value=50000.0, step=1000.0
    )
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
    has_crcard = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input(
        "Estimated Salary", min_value=0.0, value=75000.0, step=1000.0
    )

if st.button("Predict Churn"):
    # Encode categorical values matching training dummies
    # Geography: France is baseline (drop_first=True)
    is_germany = 1 if geography == "Germany" else 0
    is_spain = 1 if geography == "Spain" else 0

    # Gender: Female is baseline (drop_first=True)
    is_male = 1 if gender == "Male" else 0

    # Binary flags
    has_card = 1 if has_crcard == "Yes" else 0
    active_member = 1 if is_active == "Yes" else 0

    # Construct feature array in original column sequence:
    # [CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Germany, Spain, Male]
    features = np.array(
        [[
            credit_score,
            age,
            tenure,
            balance,
            num_products,
            has_card,
            active_member,
            estimated_salary,
            is_germany,
            is_spain,
            is_male,
        ]]
    )

    # Scale and predict
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0][0]

    st.markdown("---")
    st.subheader("Result")
    st.write(f"**Predicted Churn Probability:** `{prediction:.2%}`")

    if prediction >= 0.5:
        st.error("⚠️ High Risk: The customer is likely to churn.")
    else:
        st.success("✅ Low Risk: The customer is likely to stay.")