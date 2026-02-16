import streamlit as st
from image_pipeline import analyze_image
import cv2
from reasoning import ask_llm


st.title("Autonomous Visual AI Assistant")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    data = analyze_image("temp.jpg")

    st.subheader("Annotated Image")
    st.image(data["image"], channels="BGR")

    st.subheader("Detected Objects")
    st.write(data["objects"])

    st.subheader("Extracted Text")
    st.write(data["text"])
