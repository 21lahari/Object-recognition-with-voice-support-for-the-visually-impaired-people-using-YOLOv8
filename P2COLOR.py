import cv2
import numpy as np

# Function to detect the dominant color in an image or ROI
def detect_dominant_color(roi):
    # Convert the ROI to HSV color space
    hsv_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Create masks for each color with predefined HSV ranges
    color_ranges = {
        "Red": [(0, 50, 50), (10, 255, 255), (160, 50, 50), (180, 255, 255)],
        "Orange": [(10, 100, 100), (25, 255, 255)],
        "Yellow": [(25, 100, 100), (35, 255, 255)],
        "Green": [(35, 50, 50), (85, 255, 255)],
        "Blue": [(85, 50, 50), (135, 255, 255)],
        "Purple": [(135, 50, 50), (160, 255, 255)],
        "Pink": [(300, 50, 50), (330, 255, 255)],
        "White": [(0, 0, 200), (180, 40, 255)],
        "Black": [(0, 0, 0), (180, 40, 50)],
    }

    max_coverage = 0
    detected_color = "Unknown"

    for color, ranges in color_ranges.items():
        if len(ranges) == 2:
            lower, upper = ranges
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        else:  # Red has two separate ranges
            mask1 = cv2.inRange(hsv_frame, np.array(ranges[0]), np.array(ranges[1]))
            mask2 = cv2.inRange(hsv_frame, np.array(ranges[2]), np.array(ranges[3]))
            mask = cv2.bitwise_or(mask1, mask2)

        # Calculate the percentage of pixels matching the color
        coverage = (np.sum(mask > 0) / mask.size) * 100

        if coverage > max_coverage:
            max_coverage = coverage
            detected_color = color

    return detected_color, max_coverage

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        break

    # Define a region of interest (ROI) for color detection
    roi_x1, roi_y1, roi_x2, roi_y2 = 200, 150, 400, 300
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # Detect the dominant color in the ROI
    dominant_color, accuracy = detect_dominant_color(roi)

    # Display the ROI on the main frame
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 2)
    cv2.putText(
        frame,
        f"Color: {dominant_color} ({accuracy:.2f}%)",
        (roi_x1, roi_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Show the frame
    cv2.imshow("Color Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()