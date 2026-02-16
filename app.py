import streamlit as st
from image_pipeline import analyze_image
from reasoning import ask_llm

st.title("Autonomous Visual AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    # Question section
    question = st.text_input("Ask a question about the image")

    # Memory setup
if "history" not in st.session_state:
    st.session_state.history = []

    question = st.text_input("Ask a question about the image")
# Store questions
if question:
    st.session_state.history.append(question)

# Show previous questions
st.write("Previous questions:", st.session_state.history)

#Answer logic below
if question:
    answer = ask_llm(data["objects"], data["text"], question)
    st.write(answer)

if question:
    objects = data["objects"]
    text = data["text"]

    if "object" in question.lower():
        answer = f"I can see {', '.join(objects)}."

    elif "text" in question.lower() or "read" in question.lower():
        answer = f"The text in the image says: {text}"

    elif "summary" in question.lower():
        answer = f"This image contains {len(objects)} objects including {', '.join(objects)}. The text found is: {text}"

    else:
        answer = "I understand the image but cannot answer that yet."

    # Save conversation
    st.session_state.chat_history.append(("You", question))
    st.session_state.chat_history.append(("Assistant", answer))

# Display chat history
st.subheader("Conversation")

for role, message in st.session_state.chat_history:
    st.write(f"**{role}:** {message}")


