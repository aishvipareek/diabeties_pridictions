import streamlit as st
import pandas as pd
import plotly.express as px

# Assuming diabetes.csv is already loaded
data = pd.read_csv("diabetes.csv")

# Static mode content
def static_page():
    st.title("Static Page - Diabetes Prediction")
    
    # Dropdown for selecting parameter for histogram
    parameter = st.selectbox("Select Parameter for Histogram", ["BMI", "Glucose", "Age"])

    # Plotting histogram
    fig = px.histogram(data, x=parameter, nbins=20, title=f"{parameter} Distribution")
    st.plotly_chart(fig)

    # You can add other static page content here as well
    # Prediction button, explanation, etc.
    st.write("Use the above histogram to understand the distribution of selected parameter.")

# Your existing Streamlit app code
mode = st.sidebar.selectbox("Select Mode", ["Static (Single Patient)", "Dynamic (Multiple Patients)"])

if mode == "Static (Single Patient)":
    static_page()

elif mode == "Dynamic (Multiple Patients)":
    # Your existing dynamic mode code here...
    pass
