import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Load model and data
model = pickle.load(open('random_forest_diabetes_model.pkl', 'rb'))
df = pd.read_csv('diabetes.csv')

# Sidebar for mode selection
st.sidebar.title("Mode Selection")
mode = st.sidebar.radio("Select Mode", ["Static (Single Patient)", "Dynamic (Multiple Patients)"])

# ------------------------- STATIC PAGE -------------------------
if mode == "Static (Single Patient)":
    st.title("🩺 Diabetes Risk Prediction - Static Mode")

    # Dropdown for selecting parameter for histogram
    st.subheader("📊 Parameter-wise Histogram")
    selected_param = st.selectbox("Select Parameter", ['Glucose', 'BloodPressure', 'BMI', 'Age', 'Insulin'])

    # Plot histogram using Plotly
    fig = px.histogram(df, x=selected_param, nbins=30, title=f"Distribution of {selected_param}")
    st.plotly_chart(fig)

    st.subheader("🔍 Enter Patient Details")

    # Input fields
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.slider("Glucose", 0, 200, 100)
    bp = st.slider("Blood Pressure", 0, 130, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 67.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age", 1, 120, 33)

    # Prediction
    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("❌ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

# ------------------------- DYNAMIC PAGE -------------------------
elif mode == "Dynamic (Multiple Patients)":
    st.title("📁 Diabetes Risk Analysis - Dynamic Mode (Multiple Patients)")

    uploaded_file = st.file_uploader("Upload Patient Data CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if not {'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age'}.issubset(data.columns):
            st.error("CSV file must contain all required columns.")
        else:
            # Predict for all patients
            input_features = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
            data['Prediction'] = model.predict(input_features)

            # Show predictions
            st.subheader("📊 Prediction Results")
            st.dataframe(data)

            # Plot number of diabetic vs non-diabetic
            result_counts = data['Prediction'].value_counts().reset_index()
            result_counts.columns = ['Prediction', 'Count']
            result_counts['Prediction'] = result_counts['Prediction'].map({0: 'Non-Diabetic', 1: 'Diabetic'})

            fig_bar = px.bar(result_counts, x='Prediction', y='Count',
                             color='Prediction', title="Diabetes Prediction Summary",
                             color_discrete_map={'Diabetic': 'red', 'Non-Diabetic': 'green'})
            st.plotly_chart(fig_bar)
