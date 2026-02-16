import streamlit as st
import tempfile
from image_pipeline import analyze_image
from reasoning import ask_llm
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img)
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


st.title("Autonomous Visual AI Assistant")

# Upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
question = st.text_input("Ask a question about the image")

# Webcam section
st.subheader("Capture Image from Webcam")
camera_image = st.camera_input("Take a picture")


# Process uploaded image
if uploaded_file is not None:
    import os, tempfile
    file_extension = os.path.splitext(uploaded_file.name)[1]
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    file_bytes = uploaded_file.read()
    tfile.write(file_bytes)
    tfile.close()

    st.image(file_bytes, caption="Uploaded Image")

    data = analyze_image(tfile.name)

    st.write("Detected Objects:", data["objects"])
    st.write("Extracted Text:", data["text"])

    if question:
        answer = ask_llm(data["objects"], data["text"], question)
        st.success(answer)


# Process webcam image
if camera_image is not None:
    file_bytes = camera_image.read()

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tfile.write(file_bytes)
    tfile.close()

    st.image(file_bytes, caption="Captured Image")

    data = analyze_image(tfile.name)

    st.write("Detected Objects:", data["objects"])
    st.write("Extracted Text:", data["text"])

st.subheader("Live Webcam Detection")

webrtc_streamer(
    key="live-camera",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
