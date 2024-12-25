# Diabetes Prediction App with Explainable AI (XAI)
This repository contains a machine learning model for predicting diabetes, built using LightGBM (a gradient boosting model). The app includes the use of CTGAN to balance the dataset, threshold adjustment to improve recall/sensitivity, and SHAP and LIME for explainable AI (XAI) to enhance the interpretability of model predictions.

Table of Contents
Introduction
Features
Technologies Used
Model Development

## Introduction
This app predicts the likelihood of a person developing diabetes based on various demographic, clinical, and lifestyle data inputs. The model has been designed to provide not only accurate predictions but also explainability through SHAP and LIME methods, allowing healthcare practitioners to understand how predictions are made.

## Features
Accurate Diabetes Prediction: The app uses a robust LightGBM model to predict the likelihood of diabetes.
Balanced Dataset: The training dataset is balanced using CTGAN (Conditional Generative Adversarial Networks) to avoid class imbalance and improve model performance.
Improved Sensitivity/Recall: Threshold adjustment is applied to improve recall, ensuring that the model identifies as many potential cases as possible.
Explainability with XAI: SHAP and LIME are integrated to provide local and global interpretability of model predictions, ensuring transparency for healthcare professionals.
## Technologies Used
LightGBM: A gradient boosting framework for fast, distributed training.
CTGAN: Conditional Generative Adversarial Network used for balancing the dataset.
SHAP: SHapley Additive exPlanations for local interpretability.
LIME: Local Interpretable Model-Agnostic Explanations for interpreting model predictions.
Streamlit Cloud: For App deployment
Python: Programming language used for model building and deployment.
## Model Development
Data Preprocessing: The dataset includes clinical, demographic, and lifestyle data. It is preprocessed to handle missing values, normalize numerical features, and encode categorical variables.
Balancing the Dataset: Due to class imbalance in diabetes prediction, CTGAN is used to generate synthetic samples to balance the dataset.
LightGBM Model: The final model is a LightGBM classifier, trained on the preprocessed and balanced data.
Threshold Adjustment: A custom threshold is applied to optimize the recall/sensitivity of the model, ensuring it detects more positive cases (diabetes).
Explainable AI: Both SHAP and LIME methods are applied to the trained model to provide explanations for individual predictions.
