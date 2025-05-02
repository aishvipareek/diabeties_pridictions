import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load the trained model
model = joblib.load("random_forest_diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Prediction - Dynamic Mode", layout="wide")
st.title("🤖 Dynamic Diabetes Risk Prediction")

st.sidebar.header("📝 Enter Patient Data")

# Safe ranges to be displayed in sidebar
st.sidebar.markdown("### 📊 Safe Ranges")
st.sidebar.markdown("- **BMI**: 18.5 - 24.9")
st.sidebar.markdown("- **Glucose**: 70 - 140 mg/dL")
st.sidebar.markdown("- **Blood Pressure**: 80 - 120 mm Hg")
st.sidebar.markdown("- **Skin Thickness**: ~10 - 50 mm")
st.sidebar.markdown("- **Insulin**: 16 - 166 uU/mL")
st.sidebar.markdown("- **Age**: Ideally < 45")

num_patients = st.sidebar.number_input("Number of Patients", min_value=1, max_value=10, value=1, step=1)

patients_data = []

for i in range(num_patients):
    st.sidebar.subheader(f"Patient {i+1}")
    name = st.sidebar.text_input(f"Name {i+1}", key=f"name_{i}")
    pregnancies = st.sidebar.number_input(f"Pregnancies {i+1}", min_value=0, max_value=20, key=f"pregnancies_{i}")
    glucose = st.sidebar.number_input(f"Glucose {i+1}", min_value=0, max_value=200, key=f"glucose_{i}")
    bp = st.sidebar.number_input(f"Blood Pressure {i+1}", min_value=0, max_value=150, key=f"bp_{i}")
    skin = st.sidebar.number_input(f"Skin Thickness {i+1}", min_value=0, max_value=100, key=f"skin_{i}")
    insulin = st.sidebar.number_input(f"Insulin {i+1}", min_value=0, max_value=900, key=f"insulin_{i}")
    bmi = st.sidebar.number_input(f"BMI {i+1}", min_value=0.0, max_value=70.0, key=f"bmi_{i}")
    dpf = st.sidebar.number_input(f"Diabetes Pedigree Function {i+1}", min_value=0.0, max_value=2.5, key=f"dpf_{i}")
    age = st.sidebar.number_input(f"Age {i+1}", min_value=1, max_value=120, key=f"age_{i}")

    patients_data.append({
        'Patient Name': name or f"Patient {i+1}",
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    })

if st.button("🔍 Predict Diabetes Risk"):
    df = pd.DataFrame(patients_data)
    X = df.drop(['Patient Name'], axis=1)
    predictions = model.predict(X)

    # Custom risk logic
    risks = []
    for i, row in df.iterrows():
        bmi = row['BMI']
        age = row['Age']
        insulin = row['Insulin']

        if bmi > 26 and age > 40:
            risk = 80
        elif insulin > 166 and bmi > 26:
            risk = 60
        elif bmi > 26:
            risk = 50
        else:
            risk = 20
        risks.append(risk)

    df['Risk (%)'] = risks
    df['Prediction'] = ["Diabetic" if r >= 50 else "Non-Diabetic" for r in risks]

    st.subheader("📄 Prediction Summary")
    st.dataframe(df[['Patient Name', 'Risk (%)', 'Prediction']], use_container_width=True)

    st.subheader("📊 Risk Comparison Chart")
    fig = px.bar(df, x='Patient Name', y='Risk (%)', color='Prediction', 
                 color_discrete_map={"Diabetic": "crimson", "Non-Diabetic": "seagreen"},
                 text='Risk (%)', height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 🏃 Suggested Exercises Based on Risk")
    for i, row in df.iterrows():
        name = row['Patient Name']
        risk = row['Risk (%)']
        st.markdown(f"### {name} (Risk: {risk:.2f}%)")

        if risk < 30:
            st.info("🧘 **Suggestion:** Light physical activity like morning walks, gentle yoga, and stretching.")
        elif 30 <= risk <= 70:
            st.warning("🚴 **Suggestion:** Moderate workouts like brisk walking, swimming, cycling, and Zumba.")
        else:
            st.error("🏋️ **Suggestion:** Intensive workouts like cardio, strength training, and doctor-supervised routines.")

        st.markdown("---")
