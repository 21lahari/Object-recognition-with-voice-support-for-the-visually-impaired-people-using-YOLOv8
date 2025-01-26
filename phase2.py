# code for object recog and text 
import random
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import pytesseract

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to speak the text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Set up Tesseract executable path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  # Ensure this is the correct path to tesseract.exe

# Read class list from the COCO file
with open(r"C:\Users\Pasham Lahari\yolov8-silva\utils\coco.txt", "r") as my_file:
    data = my_file.read()

class_list = data.split("\n")

# Generate random colors for each class
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt")

# Values to resize video frames for optimized processing
frame_wid = 640
frame_hyt = 480

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

print("Press '1' for Object Detection, '2' for Text Recognition, 'q' to Quit.")

# Variable to store the current mode
mode = None  # Start with no mode selected

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting...")
        break

    # Resize the frame for better performance
    frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Check for key press without blocking
    key = cv2.waitKey(1) & 0xFF  # Reads key press (non-blocking)

    # Handle key input to toggle modes
    if key == ord('1'):
        mode = 1
        print("Switched to Object Detection Mode")
    elif key == ord('2'):
        mode = 2
        print("Switched to Text Recognition Mode")
    elif key == ord('q'):
        print("Exiting...")
        break

    # Execute actions based on the selected mode
    if mode == 1:  # Object Detection
        results = model.predict(frame, conf=0.45, verbose=False)
        detected_objects = set()

        for r in results:
            for box in r.boxes:  # Iterate through detected boxes
                clsID = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangles around detected objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), detection_colors[clsID], 2)

                # Display class name and confidence
                cv2.putText(
                    frame,
                    f"{class_list[clsID]} {round(conf, 2)}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                detected_objects.add(class_list[clsID])

        # Speak detected objects
        if detected_objects:
            for obj in detected_objects:
                speak(f"Detected {obj}")

    elif mode == 2:  # Text Recognition
        # Convert the frame to grayscale for better OCR performance
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_frame)

        if text.strip():  # If text is detected
            print(f"Detected text: {text.strip()}")
            speak(f"Detected text: {text.strip()}")
        else:
            print("No text detected")

    # Display mode instruction on the frame
    instructions = "Press '1' for Object Detection, '2' for Text Recognition, 'q' to Quit."
    cv2.putText(frame, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Object Detection / Text Recognition", frame)

# Release resources
cap.release()
cv2.destroyAllWindows()

