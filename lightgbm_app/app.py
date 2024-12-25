import streamlit as st
import pandas as pd
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu  # Install via pip if needed

# Inject custom CSS for sidebar color
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #bde0fe; /* Sky blue */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with icons
with st.sidebar:
    selected = option_menu(
        menu_title=None,  # No menu title
        options=["Home", "EDA", "Prediction"],  # Pages
        icons=["house", "bar-chart", "activity"],  # Corresponding icons
        menu_icon="cast",  # Menu icon
        default_index=0,  # Default selected page
        styles={
            "container": {"padding": "5px", "background-color": "#87CEEB"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "white",
            },
            "nav-link-selected": {"background-color": "#4682B4"},  # Darker blue for active item
        },
    )

    #import os
    #import streamlit as st

    #st.write("Current Directory:", os.getcwd())
    #st.write("Files in Current Directory:", os.listdir())

    
    st.image("lightgbm_app/Diabetes_img.JPG", use_container_width =False, width=120)

# Load the trained LightGBM model
model = joblib.load('lightgbm_app/lightgbm_Saved_model.pkl')

# Load the dataset used for training
training_data = pd.read_csv('lightgbm_app/diabetes_prediction_dataset.csv')

# Encode categorical columns
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'smoking_history']
for column in categorical_columns:
    training_data[column] = label_encoder.fit_transform(training_data[column])

# Prepare features and target
target_column = 'diabetes'
features_data = training_data.drop(columns=[target_column])

# Initialize SHAP explainer
@st.cache_resource
def load_shap_explainer():
    return shap.Explainer(model)

shap_explainer = load_shap_explainer()

# Conditional display for pages
if selected == "Home":
    col1, col2 = st.columns([4, 1])  # Adjust column width ratios as needed

    with col1:
        st.title("Diabetes Prediction App")

    with col2:
        st.image("lightgbm_app/Diabetes_Hrt.Jpg", use_column_width=False, width = 60)

    # Main introduction content
    st.markdown("""
    ### Understanding Diabetes
    Diabetes is a chronic condition that affects the way your body processes blood sugar (glucose).
    
    ### Types of Diabetes:
    - **Type 1 Diabetes**: An autoimmune condition.
    - **Type 2 Diabetes**: Insulin resistance or insufficient production.
    - **Gestational Diabetes**: Occurs during pregnancy.
    
    ### **Risk Factors**

    - 🧬 **Genetics**  
    Family history of diabetes can increase your risk.

    - ⚖️ **Obesity**  
    Excess body weight, especially around the abdomen, is a key risk factor for Type 2 diabetes.

    - 🏃 **Lack of Exercise**  
    Physical inactivity can lead to insulin resistance and increase your diabetes risk.

    - 🍔 **Poor Diet**  
    High consumption of processed foods, sugary drinks, and unhealthy fats contributes to the development of diabetes.
        
        For more information, visit the [American Diabetes Association](https://www.diabetes.org/).
        """)

    # Bio Section
    st.markdown("""
    ## About This App
    Inspired by the MSc dissertation **Explainable Artificial Intelligence for Diabetes Prediction: Insights from SHAP and LIME**, this app was created to deliver real-time diabetes predictions with high accuracy and interpretability.

    Using advanced machine learning techniques, the app combines predictions with explanations through SHAP and LIME. These ensure transparency, helping both healthcare professionals and patients understand the factors influencing each prediction.

    By focusing on explainability and user trust, this app empowers informed decision-making and promotes proactive health management, bridging the gap between AI innovation and practical healthcare solutions.
    """)

    # Bio and Image Section
    st.image("lightgbm_app/Otobong.Jpg", caption="About the Author: Otobong Edemenang", use_column_width=False, width=120)
    
    st.markdown(
        """
            A passionate Machine Learning Engineer dedicated to leveraging AI and data science to enhance healthcare outcomes and improve lives worldwide.
        
        """,
        unsafe_allow_html=True
    )


elif selected == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Below are key insights and visualizations from the dataset:")

    # Dataset overview
    st.write("### Dataset Overview")
    st.write(training_data.describe())

    # Correlation matrix
    st.write("### Correlation Matrix")
    corr_matrix = training_data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Distribution of categorical features
    st.write("### Categorical Feature Distributions")
    for column in categorical_columns:
        fig, ax = plt.subplots()
        training_data[column].value_counts().plot(kind='bar', ax=ax)
        st.write(f"Distribution of {column}")
        st.pyplot(fig)

elif selected == "Prediction":
    st.title("Diabetes Prediction Interface")

    # User input section
    st.sidebar.markdown("## Input Your Data")
    gender = st.sidebar.selectbox("Gender (1 = Male, 0 = Female)", [0, 1])
    age = st.sidebar.slider("Age", 1, 100, 25)
    hypertension = st.sidebar.selectbox("Hypertension (1 = Yes, 0 = No)", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease (1 = Yes, 0 = No)", [0, 1])
    smoking_history = st.sidebar.selectbox("Smoking History (0 = Never, 1 = Formerly, 2 = Current)", [0, 1, 2])
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    hba1c_level = st.sidebar.slider("HbA1c Level", 4.0, 15.0, 6.5)
    blood_glucose_level = st.sidebar.slider("Blood Glucose Level", 50.0, 300.0, 120.0)

    # Create input data for prediction
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
    input_data = input_data[features_data.columns]

    # Prediction and explanations
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    shap_values = shap_explainer(input_data)
    
    # Display Prediction Result
    st.markdown("### Prediction Result")
    if prediction == 1:
        # High risk: Color only the phrase "high risk of diabetes"
        st.markdown(
            "The model predicts **<span style='color:red;'>a higher risk of diabetes</span>**.",
            unsafe_allow_html=True
        )
    else:
        # Low risk: Color only the phrase "low risk of diabetes"
        st.markdown(
            "The model predicts **<span style='color:blue;'>a lower risk of diabetes</span>**.",
            unsafe_allow_html=True
        )
    
    # Display Prediction Probability
    st.write(f"Prediction Probability of Diabetes: {prediction_proba:.2f}")

    # Row 1: SHAP Waterfall Plot and SHAP Bar Plot
    st.write("## SHAP Explanation")

    # Create two columns
    col1, col2 = st.columns(2)

    # Column 1: SHAP Waterfall Plot
    with col1:
        st.markdown("### Waterfall Plot")
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

    # Column 2: SHAP Summary Bar Plot
    with col2:
        st.markdown("### SHAP Summary Bar Plot")
        fig_bar, ax_bar = plt.subplots()
        shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
        st.pyplot(fig_bar)

    # LIME explanation
    st.markdown("## LIME Explanation")
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
