import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Load the pre-trained model (ensure your model supports predict_proba)
model = pickle.load(open('random_forest_diabetes_model.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🍏", layout="wide")
st.title("🍏 Diabetes Risk Prediction App")

# Sidebar for static mode
st.sidebar.title("📊 Diabetes Risk Prediction")
mode = st.sidebar.radio("Select Mode", ("Dynamic Mode", "Static Mode"))

# Static Mode: Single patient input
if mode == "Static Mode":
    st.markdown("<h1 style='text-align: center;'>💉 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Predict diabetes risk for a single patient</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Input fields for a single patient
    patient_name = st.text_input("Enter Patient Name", "Patient 1")
    gender = st.radio("Select Gender", ["Male", "Female"], horizontal=True)
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin Level", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 50.0, 25.0)
    age = st.number_input("Age", 18, 100, 25)
    pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

    # Create a DataFrame for input data
    input_data = {
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [bp],
        "SkinThickness": [skin],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [pedigree],
        "Age": [age]
    }
    df_input = pd.DataFrame(input_data)

    # Prediction button
    if st.button("🔮 Predict Diabetes Risk"):
        # Model prediction
        prediction = model.predict(df_input)
        probability = model.predict_proba(df_input)[:, 1][0]  # Probability of diabetes

        # Risk Calculation
        if probability > 0.5:
            risk_percent = round(probability * 100, 2)
        else:
            risk_percent = round((1 - probability) * 100, 2)

        # Custom Risk Logic
        if bmi > 26 and insulin <= 25 and age <= 40:
            risk_percent = 50
        elif bmi > 26 and age > 40:
            risk_percent = 80
        elif insulin > 25 and bmi > 26:
            risk_percent = 60

        # Show results
        st.subheader(f"🩺 {patient_name}'s Risk of Diabetes: {risk_percent}%")
        st.write(f"Model Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
        st.write(f"Prediction Probability: {probability * 100:.2f}%")

        # Show a bar chart for risk
        fig = px.bar(x=[patient_name], y=[risk_percent], labels={'x': 'Patient', 'y': 'Risk (%)'}, color=[risk_percent], color_continuous_scale='Reds')
        fig.update_layout(title=f"{patient_name} - Diabetes Risk", xaxis_title="Patient", yaxis_title="Risk (%)", template="plotly_dark")
        st.plotly_chart(fig)

# Dynamic Mode: Multiple patients comparison
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

            # Get model probabilities (this returns a 2D array)
            probabilities = model.predict_proba(df)[:, 1]  # Get probabilities for class 1 (diabetes)

            # Apply custom risk logic
            risk_percentages = []

            for i, patient in df.iterrows():
                bmi = patient["BMI"]
                age = patient["Age"]
                insulin = patient["Insulin"]

                if bmi > 26 and insulin <= 25 and age <= 40:
                    risk_percent = 50
                elif bmi > 26 and age > 40:
                    risk_percent = 80
                elif insulin > 25 and bmi > 26:
                    risk_percent = 60
                else:
                    # Get the risk from model output
                    risk_percent = probabilities[i] * 100

                risk_percentages.append(risk_percent)

            df['Risk (%)'] = risk_percentages
            df['Patient Name'] = names

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
