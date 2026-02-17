git add README.md
git commit -m "Added README"
git push
![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ObjectDetection-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Active-success)
 
# Autonomous Visual AI Assistant

## Overview
This project is a real-time AI assistant that can:
- Detect objects
- Extract text from images
- Answer questions about scenes
- Perform live webcam detection

## Live Demo
https://autonomousvisualai-eyh7qt4exhx43yvxunox2k.streamlit.app/

## Features
- Real-time object detection (YOLOv8)
- OCR text extraction (EasyOCR)
- LLM reasoning
- Webcam integration
- Streamlit deployment

## Tech Stack
- Python
- OpenCV
- YOLOv8
- EasyOCR
- Streamlit
- LLM API

## Installation

```bash
git clone https://github.com/Ajay176854/AutonomousVisualAI.git
cd AutonomousVisualAI
pip install -r requirements.txt
streamlit run dashboard.py