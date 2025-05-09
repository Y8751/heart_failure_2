import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load models and preprocessing tools
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
keras_model = load_model("keras_model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.title("Heart Disease Prediction App")

st.sidebar.header("Enter Patient Data")

def user_input_features():
    Age = st.sidebar.slider("Age", 20, 100, 50)
    Sex = st.sidebar.selectbox("Sex", ["M", "F"])
    ChestPainType = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    RestingBP = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    Cholesterol = st.sidebar.slider("Cholesterol", 100, 600, 200)
    FastingBS = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    RestingECG = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    MaxHR = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    ExerciseAngina = st.sidebar.selectbox("Exercise Angina", ["Y", "N"])
    Oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
    ST_Slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

    data = {
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope,
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Encode categorical features
cat_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
num_features = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

encoded_cat = encoder.transform(input_df[cat_features])
scaled_num = scaler.transform(input_df[num_features])

# Combine for final input
final_input = np.hstack([scaled_num, encoded_cat])

# Reshape input for Keras
final_input_keras = final_input.reshape(1, -1)

# Predict with models
rf_pred = rf_model.predict(final_input)[0]
xgb_pred = xgb_model.predict(final_input)[0]
keras_pred = (keras_model.predict(final_input_keras)[0][0] > 0.5).astype(int)

st.subheader("Predictions")

st.write(f"**Random Forest** Prediction: {'Heart Disease' if rf_pred else 'No Heart Disease'}")
st.write(f"**XGBoost** Prediction: {'Heart Disease' if xgb_pred else 'No Heart Disease'}")
st.write(f"**Keras Neural Network** Prediction: {'Heart Disease' if keras_pred else 'No Heart Disease'}")
