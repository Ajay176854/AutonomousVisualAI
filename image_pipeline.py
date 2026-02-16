from ultralytics import YOLO
import easyocr
import cv2

model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])

def analyze_image(image_path):

    # Object detection
    results = model(image_path)

    objects = []
    for r in results:
        for box in r.boxes:
            objects.append(r.names[int(box.cls)])

    objects = list(set(objects))

    # Get annotated image from YOLO
    annotated_image = results[0].plot()

    # OCR
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ocr_results = reader.readtext(gray)
    texts = [res[1] for res in ocr_results]
    full_text = " ".join(texts)

    return {
        "objects": objects,
        "text": full_text,
        "image": annotated_image
    }
