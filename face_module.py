from deepface import DeepFace

def recognize_faces(frame):
    try:
        results = DeepFace.find(
            img_path=frame,
            db_path="known_faces",
            enforce_detection=False
        )

        names = []
        for df in results:
            if len(df) > 0:
                identity = df.iloc[0]['identity']
                name = identity.split("\\")[-1].split(".")[0]
                names.append(name)

        return names

    except Exception as e:
        print(f"Error recognizing faces: {e}")
        return []

