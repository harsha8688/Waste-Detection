from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
import threading
from PIL import Image
import numpy as np


def sleep_and_clear_success():
    time.sleep(3)
    for key in [
        "recyclable_placeholder",
        "non_recyclable_placeholder",
        "hazardous_placeholder",
    ]:
        if key in st.session_state:
            try:
                st.session_state[key].empty()
            except:
                pass


def load_model(model_path):
    model = YOLO(model_path)
    return model


def classify_waste_type(detected_items):
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    hazardous_items = set(detected_items) & set(settings.HAZARDOUS)
    return recyclable_items, non_recyclable_items, hazardous_items


def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ")


def _display_detected_frames(model, st_frame, image):
    image = cv2.resize(image, (640, int(640 * (9 / 16))))

    for key in [
        "recyclable_placeholder",
        "non_recyclable_placeholder",
        "hazardous_placeholder",
    ]:
        if key not in st.session_state:
            st.session_state[key] = st.sidebar.empty()

    res = model.predict(image, conf=0.6)
    names = model.names
    class_counts = {}

    for result in res:
        for cls_id in result.boxes.cls:
            class_name = names[int(cls_id)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(
        class_counts.keys()
    )

    st.session_state["recyclable_placeholder"].markdown("")
    st.session_state["non_recyclable_placeholder"].markdown("")
    st.session_state["hazardous_placeholder"].markdown("")

    if recyclable_items:
        items_str = "\n- ".join(
            f"{remove_dash_from_class_name(item)}: {class_counts[item]}"
            for item in recyclable_items
        )
        st.session_state["recyclable_placeholder"].markdown(
            f"<div class='stRecyclable'>Recyclable items:\n\n- {items_str}</div>",
            unsafe_allow_html=True,
        )
    if non_recyclable_items:
        items_str = "\n- ".join(
            f"{remove_dash_from_class_name(item)}: {class_counts[item]}"
            for item in non_recyclable_items
        )
        st.session_state["non_recyclable_placeholder"].markdown(
            f"<div class='stNonRecyclable'>Non-Recyclable items:\n\n- {items_str}</div>",
            unsafe_allow_html=True,
        )
    if hazardous_items:
        items_str = "\n- ".join(
            f"{remove_dash_from_class_name(item)}: {class_counts[item]}"
            for item in hazardous_items
        )
        st.session_state["hazardous_placeholder"].markdown(
            f"<div class='stHazardous'>Hazardous items:\n\n- {items_str}</div>",
            unsafe_allow_html=True,
        )

    threading.Thread(target=sleep_and_clear_success).start()

    res_plotted = res[0].plot()
    st_frame.image(res_plotted, channels="BGR")


def play_webcam(model):
    source_webcam = settings.WEBCAM_PATH
    try:
        vid_cap = cv2.VideoCapture(source_webcam)
        st_frame = st.empty()
        while vid_cap.isOpened():
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(model, st_frame, image)
            else:
                break
        vid_cap.release()
    except Exception as e:
        st.sidebar.error("Error loading webcam: " + str(e))


def process_uploaded_image(uploaded_image, model):
    image = Image.open(uploaded_image)
    image_np = np.array(image.convert("RGB"))
    st_frame = st.empty()
    _display_detected_frames(model, st_frame, image_np)
