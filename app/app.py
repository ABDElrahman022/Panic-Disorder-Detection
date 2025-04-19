import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('models/xgboost_model1.pkl')

# App Configuration
st.set_page_config(
    page_title="Panic Disorder Prediction",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Header Section
st.title("🙏 Panic Disorder Detection")
st.markdown(
    """<style>body {background-color: #f4f4fc;} h1, h2, h3 {color: #4a90e2;}</style>""",
    unsafe_allow_html=True
)

st.markdown(
    "Welcome to the **Panic Disorder Detection** tool. Please provide your details below to assess the likelihood of panic disorder based on key features. Remember, this tool is for **informational purposes only** and is not a substitute for professional medical advice."
)

# Input Form
with st.form(key="input_form"):
    st.subheader("Provide Your Details")

    age = st.slider("Age", min_value=10, max_value=100, value=25, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    family_history = st.selectbox("Family History", ["No", "Yes"])
    personal_history = st.selectbox("Personal History", ["Yes", "No"])
    current_stressors = st.selectbox("Current Stressors", ["Low", "Moderate", "High"])
    symptoms = st.selectbox(
        "Symptoms", [
            "Shortness of breath",
            "Panic attacks",
            "Chest pain",
            "Dizziness",
            "Fear of losing control"
        ]
    )
    severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
    impact_on_life = st.selectbox("Impact on Life", ["Mild", "Moderate", "Significant"])
    demographics = st.selectbox("Demographics", ["Rural", "Urban"])
    medical_history = st.selectbox(
        "Medical History", ["Diabetes", "Asthma", "Heart disease", "None"]
    )
    psychiatric_history = st.selectbox(
        "Psychiatric History", [
            "Bipolar disorder",
            "Anxiety disorder",
            "Depressive disorder",
            "None"
        ]
    )
    substance_use = st.selectbox("Substance Use", ["None", "Drugs", "Alcohol"])
    coping_mechanisms = st.selectbox(
        "Coping Mechanisms", [
            "Socializing",
            "Exercise",
            "Seeking therapy",
            "Meditation"
        ]
    )
    social_support = st.selectbox("Social Support", ["High", "Moderate", "Low"])
    lifestyle_factors = st.selectbox(
        "Lifestyle Factors", ["Sleep quality", "Exercise", "Diet"]
    )

    # Submit Button
    submit_button = st.form_submit_button(label="Predict")

# Feature Mapping
mapping = {
    "Male": 0, "Female": 1,
    "No": 0, "Yes": 1,
    "Low": 2, "Moderate": 0, "High": 1,
    "Shortness of breath": 0, "Panic attacks": 1, "Chest pain": 2, "Dizziness": 3, "Fear of losing control": 4,
    "Mild": 0, "Moderate": 1, "Severe": 2, "Significant": 1,
    "Rural": 0, "Urban": 1,
    "Diabetes": 0, "Asthma": 1, "Heart disease": 3, "None": 2,
    "Bipolar disorder": 0, "Anxiety disorder": 1, "Depressive disorder": 2,
    "Drugs": 1, "Alcohol": 2,
    "Socializing": 0, "Exercise": 1, "Seeking therapy": 2, "Meditation": 3,
    "High": 0, "Low": 2,
    "Sleep quality": 0, "Diet": 2
}

# Prediction Logic
if submit_button:
    input_features = [
        age,
        mapping[gender],
        mapping[family_history],
        mapping[personal_history],
        mapping[current_stressors],
        mapping[symptoms],
        mapping[severity],
        mapping[impact_on_life],
        mapping[demographics],
        mapping[medical_history],
        mapping[psychiatric_history],
        mapping[substance_use],
        mapping[coping_mechanisms],
        mapping[social_support],
        mapping[lifestyle_factors]
    ]

    input_features = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_features)

    # Display Result
    if prediction == 1:
        st.error("The results suggest a **high likelihood** of panic disorder. Please consult a mental health professional.")
    else:
        st.success("The results suggest a **low likelihood** of panic disorder. Stay healthy and mindful!")
