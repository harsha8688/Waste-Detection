from pathlib import Path
import streamlit as st
import helper
import settings

st.set_page_config(
    page_title="Waste Detection",
    layout="wide",
)

st.sidebar.title("Detect Console")

model_path = Path(settings.DETECTION_MODEL)

st.title("Intelligent waste segregation system")
st.write("Detect multiple objects from webcam or uploaded image.")

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    webcam_on = st.button("Start Webcam Detection")
with col2:
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Webcam Detection
if webcam_on:
    helper.play_webcam(model)

# Image Upload Detection
if uploaded_image is not None:
    helper.process_uploaded_image(uploaded_image, model)

st.sidebar.markdown(
    "This is a demo of the waste detection model.", unsafe_allow_html=True
)
