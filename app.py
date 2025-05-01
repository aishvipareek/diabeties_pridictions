import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="Diabetes Predictor", layout="wide", page_icon="🩺")

# Data Loading Function
def load_data(file):
    data = pd.read_csv(file)
    return data

# Model Training Function
def train_model(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# Data Preprocessing
def preprocess_data(data):
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Fill missing values with the mean of the column
    X = X.fillna(X.mean())
    
    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Sidebar Configuration
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

    uploaded_file = st.file_uploader("📤 Upload Diabetes Dataset CSV", type=["csv"], key="dataset_csv")

    if uploaded_file:
        data = load_data(uploaded_file)
        st.write(data.head())

        # Preprocess the data
        X_scaled, y = preprocess_data(data)

        # Split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train the XGBoost model
        model = train_model(X_train, y_train)

        # Model Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Confusion Matrix and Classification Report
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        st.write(cm)

        report = classification_report(y_test, y_pred)
        st.write("Classification Report:")
        st.text(report)

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
            # Preprocess the dynamic input data
            X_scaled, _ = preprocess_data(df)

            # Train the XGBoost model (using the data already available)
            model = train_model(X_scaled, y)

            # Predicting diabetes risk
            predictions = model.predict(X_scaled)

            df['Patient Name'] = names
            df['Risk (%)'] = predictions * 100

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
