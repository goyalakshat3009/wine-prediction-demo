import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load Model
with open("wine_quality_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page Configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# Header
st.title("🍷 Wine Quality Prediction")
st.subheader("Machine Learning Project")

st.markdown("""
### 👨‍💻 Developed by Akshat Goyal
**B.Tech Mathematics & Computing | IIIT Ranchi**
""")

st.markdown("---")

# Input Fields
fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
citric_acid = st.number_input("Citric Acid", value=0.00)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

# Prediction Button
if st.button("Predict Wine Quality"):

    input_df = pd.DataFrame(
        [[
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol
        ]],
        columns=[
            'fixed acidity',
            'volatile acidity',
            'citric acid',
            'residual sugar',
            'chlorides',
            'free sulfur dioxide',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol'
        ]
    )

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("🍷 Good Quality Wine")
    else:
        st.error("❌ Bad Quality Wine")

# Footer
st.markdown("---")
st.caption("Developed by Akshat Goyal | Machine Learning Project")