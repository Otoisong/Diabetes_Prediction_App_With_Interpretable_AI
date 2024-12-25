# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 07:20:20 2024

@author: User
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the model
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components  # For rendering HTML/JS

# Load the trained LightGBM model with error handling
try:
    model = joblib.load('lightgbm_Saved_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'lightgbm_Saved_model.pkl' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Define or dynamically retrieve feature names
feature_names = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]

# Set up the Streamlit app
st.title("Diabetes Prediction App")
st.sidebar.header("Input Features")

# Create sliders for user input
input_data = {}
input_data["gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])
input_data["age"] = st.sidebar.slider("Age", 0, 100, 50)
input_data["hypertension"] = st.sidebar.selectbox("Hypertension", [0, 1])
input_data["heart_disease"] = st.sidebar.selectbox("Heart Disease", [0, 1])
input_data["smoking_history"] = st.sidebar.selectbox("Smoking History", ["never", "former", "current"])
input_data["bmi"] = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
input_data["HbA1c_level"] = st.sidebar.slider("HbA1c Level", 0.0, 15.0, 5.0)
input_data["blood_glucose_level"] = st.sidebar.slider("Blood Glucose Level", 0, 300, 100)

# Encode categorical features
def encode_features(data):
    encoded_data = data.copy()
    gender_map = {"Male": 1, "Female": 0}
    smoking_map = {"never": 0, "former": 1, "current": 2}

    try:
        encoded_data["gender"] = gender_map[data["gender"]]
        encoded_data["smoking_history"] = smoking_map[data["smoking_history"]]
    except KeyError as e:
        st.error(f"Invalid input: {e}")
        st.stop()

    return encoded_data

encoded_input_data = encode_features(input_data)
input_df = pd.DataFrame([encoded_input_data])
st.write("### Input Data", input_df)

# Make prediction
prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]
st.write(f"## Predicted Outcome: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
st.write(f"### Probability Scores: Non-Diabetic: {probabilities[0]:.2f}, Diabetic: {probabilities[1]:.2f}")

# Explain the prediction using SHAP
st.write("### SHAP Explanation")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

# Initialize SHAP JS
shap.initjs()

# Check if shap_values is a list (multi-output case)
if isinstance(shap_values, list):
    # For binary classification, select the positive class (index 1)
    shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
else:
    # Single output (e.g., regression or binary classification with single class probability)
    shap_matrix = shap_values

# SHAP Summary Plot (Bar)
st.write("### SHAP Summary Plot (Bar)")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_matrix, input_df, plot_type="bar", feature_names=feature_names, show=False)
plt.title("Feature Importance (Bar Plot)")
st.pyplot(fig)

# SHAP Waterfall Plot
st.write("#### SHAP Waterfall Plot")
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_matrix[0],  # First instance's SHAP values
            base_values=explainer.expected_value if isinstance(explainer.expected_value, float) else explainer.expected_value[0],
            data=input_df.iloc[0],
            feature_names=feature_names
        )
    )
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Error generating SHAP waterfall plot: {e}")

# SHAP Beeswarm Plot
st.write("#### SHAP Beeswarm Plot")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_matrix, input_df, plot_type="dot", feature_names=feature_names, show=False)
plt.title("Feature Importance (Beeswarm Plot)")
st.pyplot(fig)

# Explain the prediction using LIME
st.write("### LIME Explanation")
try:
    # Load or generate training data for LIME
    training_data = pd.DataFrame(
        np.random.rand(100, len(feature_names)), columns=feature_names
    )  # Replace with actual training data if available

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data.values,
        feature_names=feature_names,
        class_names=['Non-Diabetic', 'Diabetic'],
        discretize_continuous=True
    )
    lime_exp = lime_explainer.explain_instance(
        input_df.iloc[0].values, model.predict_proba, num_features=len(feature_names)
    )
    # Render LIME explanation as HTML
    lime_html = lime_exp.as_html()
    components.html(lime_html, height=800)
except Exception as e:
    st.warning(f"Error initializing or generating LIME explanation: {e}")

st.write("This app predicts diabetes and explains the prediction using SHAP and LIME.")
