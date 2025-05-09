import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Training a model (Here we assume mock data or placeholder data for training)
# Example of mock data (you can replace this with actual dataset if available)
X_mock = np.random.rand(100, 11)  # 100 samples with 11 features
y_mock = np.random.randint(0, 2, 100)  # 100 target labels (0 or 1)

# Preprocessing for categorical and numerical features
cat_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
num_features = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

# Creating encoder and scaler for the features
encoder = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

# Apply encoding to categorical columns and scaling to numerical columns
X_mock_cat = encoder.fit_transform(pd.DataFrame(X_mock[:, :5], columns=cat_features))
X_mock_num = scaler.fit_transform(pd.DataFrame(X_mock[:, 5:], columns=num_features))

# Combine the processed categorical and numerical data
X_processed = np.hstack([X_mock_cat.toarray(), X_mock_num])

# Train a RandomForest model
rf_model = RandomForestClassifier()
rf_model.fit(X_processed, y_mock)

# Train a Keras Neural Network
keras_model = Sequential()
keras_model.add(Dense(32, input_dim=11, activation='relu'))
keras_model.add(Dense(1, activation='sigmoid'))
keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
keras_model.fit(X_processed, y_mock, epochs=10, batch_size=10, verbose=0)

# Make predictions with trained models
rf_pred = rf_model.predict(X_processed[:1])[0]  # Use the first row of data for prediction
keras_pred = (keras_model.predict(X_processed[:1])[0][0] > 0.5).astype(int)

# Display predictions
st.subheader("Predictions")

st.write(f"**Random Forest** Prediction: {'Heart Disease' if rf_pred else 'No Heart Disease'}")
st.write(f"**Keras Neural Network** Prediction: {'Heart Disease' if keras_pred else 'No Heart Disease'}")
