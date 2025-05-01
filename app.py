import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Set Streamlit page config
st.set_page_config(page_title="Diabetes Predictor", layout="wide", page_icon="🩺")

# Load the trained model and check if it's loaded correctly
try:
    model = joblib.load("random_forest_diabetes_model.pkl")
    st.success("Model has been successfully loaded!")
except FileNotFoundError:
    st.error("Model file not found. Please make sure the model file is in the correct directory.")

# Custom Dark Theme CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #412232;
        color: #DB9F75;
    }
    h1, h2, h3, h4 {
        color: #A3E241;
    }
    .stButton>button {
        background-color: #A3E241;
        color: #2F3A32;
        font-weight: bold;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #DB9F75;
        color: #2F3A32;
    }
    .stAlert {
        color: #ffffff !important;
        background-color: #545748 !important;
        border-left: 0.3rem solid #A3E241 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #2F3A32 !important;
        color: white;
        padding: 20px;
    }
    section[data-testid="stSidebar"] h2 {
        color: #DB9F75;
    }
    section[data-testid="stSidebar"] p {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar mode selector and safe ranges
with st.sidebar:
    st.markdown("<h2>📝 Multiple Patient Diabetes Risk</h2>", unsafe_allow_html=True)
    mode = st.radio("📊 Select Data Mode", ["Static (Upload File)", "Dynamic (Multiple Patients)"])
    
    st.markdown("<h3>Normal Ranges</h3>", unsafe_allow_html=True)
    st.markdown("""
    - Pregnancies: 0-10<br>
    - Glucose: 70-99 mg/dL<br>
    - Blood Pressure: 70-120 mm Hg<br>
    - Skin Thickness: 10-30 mm<br>
    - Insulin: 2-25 µU/mL<br>
    - BMI: 18.5-24.9<br>
    - Age: 18-100 years<br>
    - Diabetes Pedigree Function: 0.0-1.0
    """, unsafe_allow_html=True)

# Static Mode
if mode == "Static (Upload File)":
    st.markdown("<h1 style='text-align: center;'>💉 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Upload trained dataset and new dataset to compare</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("📤 Upload Combined Year-wise Patient Data CSV", type=["csv"], key="yearwise_csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        df = df.rename(columns={
            "Blood Pressure": "BloodPressure",
            "Diabetes Pedigree Function": "DiabetesPedigreeFunction"
        })

        if "Year" in df.columns and "Patient Name" in df.columns:
            parameter_options = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            selected_param = st.selectbox("📌 Select Parameter to View Trends", parameter_options)

            fig = px.line(
                df,
                x="Year",
                y=selected_param,
                color="Patient Name",
                markers=True,
                title=f"📈 Year-wise Trend for {selected_param}",
                template="plotly_dark"
            )
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title=selected_param,
                title_font=dict(size=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Uploaded file must contain 'Year' and 'Patient Name' columns.")

# Dynamic Mode
else:
    st.markdown("<h1 style='text-align: center;'>💉 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Compare multiple patients' diabetes risks using AI-powered insights</p>", unsafe_allow_html=True)
    st.markdown("---")

    num_patients = st.selectbox("👨‍👩‍👧 Choose Number of Patients to Compare", range(1, 11), index=1)
    patients_data = []
    names = []

    for i in range(num_patients):
        st.subheader(f"📟 Patient {i+1} Details")
        patient_name = st.text_input(f"Enter Name of Patient {i+1}", f"Patient {i+1}", key=f"name{i}")
        gender = st.radio(f"Select Gender (Patient {i+1})", ["Male", "Female"], horizontal=True, key=f"gender{i}")

        col1, col2 = st.columns(2)
        with col1:
            pregnancies = 0 if gender == "Male" else st.number_input("Pregnancies", 0, 20, 1, key=f"preg{i}")
            glucose = st.number_input("Glucose Level", 0, 200, 120, key=f"glu{i}")
            bp = st.number_input("Blood Pressure", 0, 200, 70, key=f"bp{i}")
            skin = st.number_input("Skin Thickness", 0, 100, 20, key=f"skin{i}")
        with col2:
            insulin = st.number_input("Insulin Level", 0, 900, 80, key=f"insulin{i}")
            bmi = st.number_input("BMI", 0.0, 50.0, 25.0, key=f"bmi{i}")
            age = st.number_input("Age", 18, 100, 25, key=f"age{i}")
            pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, key=f"ped{i}")

        patient_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age
        }
        patients_data.append(patient_data)
        names.append(patient_name)

    if patients_data:
        predict_button = st.button("🔮 Predict Diabetes Risk")

        if predict_button:
            df = pd.DataFrame(patients_data)
            
            # Custom risk logic
            custom_risks = []
            for index, row in df.iterrows():
                risk = 0
                bmi = row["BMI"]
                age = row["Age"]
                insulin = row["Insulin"]
                
                # 50%: BMI > 26 and rest normal
                if bmi > 26 and all([
                    70 <= row["Glucose"] <= 99,
                    70 <= row["BloodPressure"] <= 120,
                    10 <= row["SkinThickness"] <= 30,
                    2 <= insulin <= 25,
                    0 <= row["Pregnancies"] <= 10,
                    0.0 <= row["DiabetesPedigreeFunction"] <= 1.0
                ]):
                    risk = 50
                # 80%: BMI > 26 and Age > 40
                elif bmi > 26 and age > 40:
                    risk = 80
                # 60%: BMI > 26 and Insulin high
                elif insulin > 25 and bmi > 26:
                    risk = 60
                else:
                    # Default: use model prediction
                    model_prediction = model.predict(pd.DataFrame([row]))[0]
                    risk = model_prediction * 100

                custom_risks.append(risk)

            df['Patient Name'] = names
            df['Risk (%)'] = custom_risks

            st.markdown("## 📊 Patient-wise Diabetes Risk Comparison")

            fig = px.bar(df, x='Patient Name', y='Risk (%)', color='Risk (%)',
                         color_continuous_scale='Reds', title="Diabetes Risk per Patient",
                         labels={'Risk (%)': 'Diabetes Risk (%)'}, template="plotly_dark")

            fig.update_layout(
                xaxis_title="Patient",
                yaxis_title="Risk (%)",
                coloraxis_colorbar=dict(title="Risk %"),
                title_font=dict(size=22),
                yaxis_range=[0, 100]
            )

            st.plotly_chart(fig, use_container_width=True)
