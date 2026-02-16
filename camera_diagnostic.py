import cv2
import sys

print("Testing camera access with different backends and indices...\n")

# Test different camera indices
for i in range(5):
    print(f"Testing camera index {i}...")
    
    # Try with default backend
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ SUCCESS: Camera {i} works!")
            print(f"    Frame size: {frame.shape}")
            cap.release()
            break
        else:
            print(f"  ✗ Camera {i} opened but can't read frames")
    else:
        print(f"  ✗ Camera {i} not found")
    cap.release()

print("\nTrying with DirectShow backend (CV_CAP_DSHOW)...")
for i in range(3):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ SUCCESS: Camera {i} with DirectShow!")
            print(f"    Frame size: {frame.shape}")
            cap.release()
            break
        else:
            print(f"  ✗ Camera {i} opened but can't read")
    cap.release()

print("\nDone. Use the working camera index in your script.")
