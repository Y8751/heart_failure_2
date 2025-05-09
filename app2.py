import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Streamlit app title
st.title("Heart Disease Prediction App")

st.sidebar.header("Enter Patient Data")

# User input function
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

# Load pre-trained models (ensure these files exist in the correct path)
rf_model = joblib.load("rf_model.joblib")  # RandomForest model saved as .joblib
xgb_model = joblib.load("xgb_model.joblib")  # XGBoost model saved as .joblib
keras_model = load_model("keras_model.h5")  # Keras model saved as .h5

# Preprocessing setup
cat_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
num_features = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

# Preprocessing for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ]
)

# Preprocess the input features
input_processed = preprocessor.fit_transform(input_df)

# Make predictions with pre-trained models
rf_pred = rf_model.predict(input_processed)[0]
xgb_pred = xgb_model.predict(input_processed)[0]
keras_pred = (keras_model.predict(input_processed)[0][0] > 0.5).astype(int)

# Display predictions
st.subheader("Predictions")

st.write(f"**Random Forest** Prediction: {'Heart Disease' if rf_pred else 'No Heart Disease'}")
st.write(f"**XGBoost** Prediction: {'Heart Disease' if xgb_pred else 'No Heart Disease'}")
st.write(f"**Keras Neural Network** Prediction: {'Heart Disease' if keras_pred else 'No Heart Disease'}")
