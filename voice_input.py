import speech_recognition as sr

def listen_question():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except:
        print("Could not understand audio")
        return ""


def listen_wake_word(wake_word="assistant"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for wake word...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, phrase_time_limit=3)

    try:
        text = recognizer.recognize_google(audio).lower()
        print("Heard:", text)
        return wake_word in text
    except:
        return False

