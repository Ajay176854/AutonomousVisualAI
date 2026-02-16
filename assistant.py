from image_pipeline import analyze_image

data = analyze_image("test01.jpg")

print("Analysis complete.")
question = input("Ask a question about the image: ").lower()

objects = data["objects"]
text = data["text"]

print(f"I analyzed the image and detected {len(objects)} objects. The main items are {', '.join(objects)}.")


# Reasoning layer
if "how many" in question and "person" in question:
    count = objects.count("person")
    print(f"I can see approximately {count} people in the image.")

elif "what objects" in question or "what is in the image" in question:
    print("I can see:", ", ".join(objects))

elif "text" in question or "read" in question or "invoice" in question:
    print("Extracted text:", text)

elif "summary" in question:
    print(f"The image contains {len(objects)} objects including {', '.join(objects)}. The text found is: {text}")

else:
    print("I analyzed the image but I am not trained to answer that yet.")
