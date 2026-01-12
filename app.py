import streamlit as st
import base64
def get_base64(file_path):
    with open(file_path,"rb")as f:
        return base64.b64encode(f.read()).decode()
leaf_bg = get_base64("picture.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{leaf_bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
    
import matplotlib.pyplot as plt
from soil_crop_model import (
    predict_soil_and_crop,
    get_accuracies,
    get_dataset
)

st.set_page_config(page_title="Soil & Crop Prediction", page_icon="🌱")

st.title("Soil based Crop Prediction System🌱")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📄 Dataset", "📊 Model Accuracy"])

# ---------------- TAB 1: Prediction ----------------
with tab1:
    st.subheader("Enter Input Values")
    N = st.number_input("Nitrogen", 0.0, 100.0)
    P = st.number_input("Phosphorus ", 0.0, 100.0)
    K= st.number_input("Potassium ", 0.0, 100.0)
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0)
    ph = st.number_input("Soil pH", 0.0, 14.0)
    rainfall=st.number_input("Rainfall",0.0,300.0)

    if st.button("Predict"):
        crop = predict_soil_and_crop(
           N,P,K, temperature, humidity,ph,rainfall
        )

        st.success(f"🌾 Suggested Crop: {crop}")

# ---------------- TAB 2: Dataset ----------------
with tab2:
    st.subheader("Training Dataset")
    data = get_dataset()
    st.dataframe(data)

# ---------------- TAB 3: Accuracy Chart ----------------
with tab3:
    st.subheader("Model Accuracy")

    crop_acc = get_accuracies()

    st.write(f"**Crop Prediction Accuracy:** {crop_acc * 100:.2f}%")

    # Plot bar chart
    labels = ['Crop Model']
    accuracies = [ crop_acc * 100]

    fig = plt.figure()
    plt.bar(labels, accuracies)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")

    st.pyplot(fig)
