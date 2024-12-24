import streamlit as st
import pandas as pd
import joblib  # For loading the model
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components  # For rendering HTML/JS
from sklearn.preprocessing import LabelEncoder

# Load the trained LightGBM model
model = joblib.load('lightgbm_Saved_model.pkl')  # Ensure this file path matches your model file

# Load the dataset used for training the model
training_data = pd.read_csv('diabetes_prediction_dataset.csv')  # Replace with the path to your training data

# Encode categorical columns (e.g., Gender, Smoking History) if necessary
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'smoking_history']  # Add other categorical columns here
for column in categorical_columns:
    training_data[column] = label_encoder.fit_transform(training_data[column])

# Extract features and target (Ensure you're using only the features for prediction)
target_column = 'diabetes'  # Replace with your actual target column name
features_data = training_data.drop(columns=[target_column])  # Ensure no target column is included

# Initialize SHAP explainer
@st.cache_resource
def load_shap_explainer():
    return shap.Explainer(model)

shap_explainer = load_shap_explainer()

# Streamlit App Layout

def main():
    # Add a logo at the top of the app
    st.image("logo.png", use_column_width=True)  # Replace "logo.png" with the path to your image

    # Streamlit app title and description
    st.markdown("<h1 style='font-size:30px;'>Diabetes Prediction App with SHAP and LIME Explanations</h1>", unsafe_allow_html=True)
    st.write("Use the sliders to adjust input values for real-time diabetes risk prediction.")

    # **EDA Section**
    st.markdown("<h2 style='font-size:24px;'>Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
    st.write("Below are some key statistics and visualizations for the dataset:")

    # Dataset Overview
    st.write("### Dataset Overview")
    st.write(training_data.describe())

    # Correlation matrix
    st.write("### Correlation Matrix")
    corr_matrix = training_data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Distribution of categorical features (for EDA)
    st.write("### Categorical Feature Distributions")
    for column in categorical_columns:
        fig, ax = plt.subplots()
        training_data[column].value_counts().plot(kind='bar', ax=ax)
        st.write(f"Distribution of {column}")
        st.pyplot(fig)

    # Create layout for input, SHAP, and LIME explanations
    col1, col2, col3 = st.columns(3)

    # Column 1: User input
    with col1:
        st.markdown("<h3 style='font-size:20px;'>User Input</h3>", unsafe_allow_html=True)

        # Use sliders to accept input values
        gender = st.selectbox("Gender (1 = Male, 0 = Female)", [0, 1])
        age = st.slider("Age", 1, 100, 25)
        hypertension = st.selectbox("Hypertension (1 = Yes, 0 = No)", [0, 1])
        heart_disease = st.selectbox("Heart Disease (1 = Yes, 0 = No)", [0, 1])
        smoking_history = st.selectbox("Smoking History (0 = Never, 1 = Formerly, 2 = Current)", [0, 1, 2])
        bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
        hba1c_level = st.slider("HbA1c Level", 4.0, 15.0, 6.5)
        blood_glucose_level = st.slider("Blood Glucose Level", 50.0, 300.0, 120.0)

        # Prepare input data for prediction using sliders (only the features, no target column)
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_history],
            'bmi': [bmi],
            'HbA1c_level': [hba1c_level],
            'blood_glucose_level': [blood_glucose_level]
        })

    # Ensure input data has the same columns as the model was trained on
    input_data = input_data[features_data.columns]  # Match input columns to training data

    # Column 2: SHAP Explanations
    with col2:
        st.markdown("<h3 style='font-size:20px;'>SHAP Explanation</h3>", unsafe_allow_html=True)

        # Make prediction and calculate SHAP values
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
        shap_values = shap_explainer(input_data)

        # SHAP Waterfall Plot
        st.write("### SHAP Waterfall Plot")
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

        # SHAP Summary Plot
        st.write("### SHAP Summary Plot (Bar)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
        st.pyplot(fig)

        # Prediction results
        st.markdown("<h4 style='font-size:18px;'>Prediction Result</h4>", unsafe_allow_html=True)
        if prediction == 1:
            st.write("The model predicts **a higher risk of diabetes**.")
        else:
            st.write("The model predicts **a lower risk of diabetes**.")
        st.write(f"Prediction Probability of Diabetes: {prediction_proba:.2f}")

    # Column 3: LIME Explanation
    with col3:
        st.markdown("<h3 style='font-size:20px;'>LIME Explanation</h3>", unsafe_allow_html=True)
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=features_data.values,
            feature_names=features_data.columns,
            class_names=["No Diabetes", "Diabetes"],
            discretize_continuous=True
        )
        lime_explanation = lime_explainer.explain_instance(
            input_data.iloc[0].values, model.predict_proba, num_features=8
        )
        lime_html = lime_explanation.as_html()
        components.html(lime_html, height=600)

if __name__ == "__main__":
    main()
