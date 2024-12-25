import streamlit as st
import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Title of the app
st.title("Diabetes Prediction Explanation App")

# Upload the trained LightGBM model
uploaded_model = st.file_uploader("Upload your LightGBM model (lgbm file)", type="txt")  # Use .txt for LightGBM models

# If a model is uploaded
if uploaded_model is not None:
    # Load the model
    clf = lgb.Booster(model_file=uploaded_model.name)
    
    # Load your data
    st.header("Upload Data for Explanation")
    uploaded_data = st.file_uploader("Upload your data (CSV file)", type="csv")
    
    if uploaded_data is not None:
        # Read the data
        data = pd.read_csv(uploaded_data)

        # Preprocessing (replace with your actual preprocessing steps)
        categorical_vars = ['gender']
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first') 
        encoded_data = encoder.fit_transform(data[categorical_vars])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_vars))
        data = data.drop(categorical_vars, axis=1)
        data = pd.concat([data, encoded_df], axis=1)

        # Make predictions
        predictions = clf.predict(data)

        # Display predictions
        st.subheader("Predictions")
        st.write(pd.DataFrame({'Prediction': predictions}))

        # SHAP explanation
        st.header("SHAP Analysis")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(data)

        # Display SHAP summary plot
        st.subheader("SHAP Summary Plot")
        st.write("This plot shows the overall importance of features across the entire dataset.")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, data, plot_type="bar", show=False)
        st.pyplot(fig)

        # Display SHAP waterfall plot for the first prediction
        st.subheader("SHAP Waterfall Plot (First Prediction)")
        st.write("This plot shows how each feature contributed to the prediction for the first instance in your data.")
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjust figure size
        shap.waterfall_plot(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
