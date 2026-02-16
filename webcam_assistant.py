import cv2
from ultralytics import YOLO
from reasoning import ask_llm
from voice_input import listen_question, listen_wake_word
from voice_output import speak
from face_module import recognize_faces
import time
import os
from datetime import datetime
import threading

os.makedirs("logs", exist_ok=True)

# -----------------------
# STATE VARIABLES
# -----------------------
last_answer = ""
conversation_history = []

latest_question = None
listening = True

greeted_people = set()
last_scene = set()
last_person_count = 0
last_event_time = 0
event_interval = 3   # calmer speaking

prev_time = 0
status_text = "Monitoring"

short_memory = []
event_log = []

last_summary_time = 0
summary_interval = 20

# -----------------------
# MODELS
# -----------------------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

speak("Visual assistant is ready and monitoring the scene.")

# -----------------------
# VOICE THREAD
# -----------------------
def voice_loop():
    global latest_question, listening
    while listening:
        if listen_wake_word():
            print("Wake word detected")
            q = listen_question()
            if q:
                latest_question = q

voice_thread = threading.Thread(target=voice_loop, daemon=True)
voice_thread.start()

# -----------------------
# ACTION DETECTION
# -----------------------
def detect_actions(results):
    actions = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = r.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0]
            width = x2 - x1
            height = y2 - y1

            if label == "person":
                ratio = height / width if width > 0 else 0
                if ratio < 1.2:
                    actions.append("person sitting")
                else:
                    actions.append("person standing")

            if label in ["laptop", "cell phone"]:
                actions.append(f"someone using {label}")

    return list(set(actions))


# -----------------------
# MAIN LOOP
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # DETECTION
    results = model(frame)
    actions = detect_actions(results)

    objects = []
    tracked_people = set()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = r.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0]
            cx = int((x1 + x2) / 2)

            if cx < w // 3:
                zone = "left"
            elif cx < 2 * w // 3:
                zone = "center"
            else:
                zone = "right"

            objects.append(f"{label} ({zone})")

            if label == "person" and box.id is not None:
                tracked_people.add(int(box.id))

    current_objects = list(set(objects))
    person_count = len(tracked_people)

    # -----------------------
    # EVENT DETECTION
    # -----------------------
    now = time.time()
    current_scene = set(current_objects)

    entered = current_scene - last_scene
    left = last_scene - current_scene

    if now - last_event_time > event_interval:
        now_time = datetime.now().strftime("%H:%M:%S")

        for obj in entered:
            speak(f"I notice {obj}")
            event_log.append(f"{now_time} - {obj} entered")

        for obj in left:
            speak(f"It looks like {obj} has left")
            event_log.append(f"{now_time} - {obj} left")

        event_log = event_log[-10:]
        last_event_time = now

    last_scene = current_scene

    # Speak actions calmly
    if now - last_event_time > event_interval:
        for act in actions:
            speak(f"I observe {act}")

    # -----------------------
    # PEOPLE COUNT
    # -----------------------
    if person_count != last_person_count:
        speak(f"I can see {person_count} people")
        last_person_count = person_count

    # -----------------------
    # FACE RECOGNITION
    # -----------------------
    faces = recognize_faces(frame)

    if len(faces) == 0:
        greeted_people.clear()

    for name in faces:
        if name not in greeted_people and name != "Unknown":
            speak(f"Hello {name}")
            greeted_people.add(name)

    # -----------------------
    # SCENE DESCRIPTION
    # -----------------------
    scene_description = (
        f"I see {person_count} people. "
        + ("Actions: " + ", ".join(actions) if actions else "")
    )

    short_memory.append(scene_description)
    short_memory = short_memory[-5:]

    # Periodic calm summary
    if time.time() - last_summary_time > summary_interval:
        speak(scene_description)
        last_summary_time = time.time()

    # -----------------------
    # DRAW FRAME
    # -----------------------
    annotated_frame = results[0].plot()

    cv2.line(annotated_frame, (w//3, 0), (w//3, h), (255,255,0), 1)
    cv2.line(annotated_frame, (2*w//3, 0), (2*w//3, h), (255,255,0), 1)

    cv2.putText(annotated_frame, "Objects: " + ", ".join(current_objects),
                (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.putText(annotated_frame, "Assistant: " + last_answer[:60],
                (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

    cv2.putText(annotated_frame, f"FPS: {int(fps)}",
                (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2)

    cv2.putText(annotated_frame, "Status: " + status_text,
                (10,120), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

    cv2.imshow("AI Assistant", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------
    # QUESTION HANDLING
    # -----------------------
    if latest_question:
        status_text = "Analyzing scene..."

        timeline_context = "Recent events: " + " | ".join(event_log)

        answer = ask_llm(
            current_objects,
            scene_description + ". " + timeline_context,
            latest_question
        )

        speak(answer)
        last_answer = answer

        conversation_history.append(f"You: {latest_question}")
        conversation_history.append(f"AI: {answer}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"logs/frame_{timestamp}.jpg", frame)

        with open("logs/conversation.txt", "a", encoding="utf-8") as f:
            f.write(f"\nTime: {timestamp}\n")
            f.write(f"Objects: {current_objects}\n")
            f.write(f"Question: {latest_question}\n")
            f.write(f"Answer: {answer}\n")

        latest_question = None
        status_text = "Monitoring"

    if key == ord('q'):
        listening = False
        break

cap.release()
cv2.destroyAllWindows()
