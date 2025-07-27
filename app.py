import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load saved models and preprocessors
autoencoder = tf.keras.models.load_model("autoencoder_model.keras")
iso_forest = joblib.load("isolation_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")  # should contain {'gender': le, 'age': le, 'category': le}

st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("🔍 Real-Time Transaction Fraud Detection")

# Get encoders and their classes
gender_labels = encoders['gender'].classes_
age_labels = encoders['age'].classes_
category_labels = encoders['category'].classes_

# -------- Streamlit UI Inputs --------
st.subheader("Enter Transaction Details")

step = st.number_input("Step (Time Step)", min_value=0, value=1)
amount = st.number_input("Transaction Amount", min_value=0.01, value=10.0)
gender = st.selectbox("Gender", gender_labels)
age = st.selectbox("Age Group", age_labels)
category = st.selectbox("Transaction Category", category_labels)

if st.button("Predict Fraud"):
    input_data = {
        "step": step,
        "amount": np.log1p(amount),  # log transform as in training
        "gender": gender,
        "age": age,
        "category": category
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical fields
    for col in ['gender', 'age', 'category']:
        input_df[col] = encoders[col].transform(input_df[col])

    # Scale numerical features
    input_df[['step', 'amount']] = scaler.transform(input_df[['step', 'amount']])

    # -------- Predict using Autoencoder --------
    recon = autoencoder.predict(input_df)
    mse = np.mean(np.power(input_df - recon, 2), axis=1)

    # -------- Predict using Isolation Forest --------
    iso_score = -iso_forest.decision_function(input_df)
    iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min())

    # -------- Weighted Ensemble Prediction --------
    combined_score = 0.6 * ((mse - mse.min()) / (mse.max() - mse.min())) + 0.4 * iso_score
    fraud_pred = int(combined_score > 0.5)

    st.subheader("🧾 Prediction Result")
    if fraud_pred == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")
