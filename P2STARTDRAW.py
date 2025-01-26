#tactile drawing
import speech_recognition as sr
import pyttsx3
import cv2
import numpy as np

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(message):
    """Convert text to speech."""
    engine.say(message)
    engine.runAndWait()

def listen():
    """Listen to user's voice commands."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening for your command...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio)
            return command.lower()
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand that. Please try again.")
            return None
        except sr.WaitTimeoutError:
            speak("No command detected. Please try again.")
            return None

def calculate_similarity(drawing, target):
    """Calculate similarity between the drawing and target using contours."""
    # Convert to grayscale
    drawing_gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Threshold both images
    _, drawing_thresh = cv2.threshold(drawing_gray, 127, 255, cv2.THRESH_BINARY)
    _, target_thresh = cv2.threshold(target_gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours_drawing, _ = cv2.findContours(drawing_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_target, _ = cv2.findContours(target_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_drawing and contours_target:
        # Use the largest contour for comparison
        cnt_drawing = max(contours_drawing, key=cv2.contourArea)
        cnt_target = max(contours_target, key=cv2.contourArea)

        # Compare shapes
        similarity = cv2.matchShapes(cnt_drawing, cnt_target, cv2.CONTOURS_MATCH_I1, 0.0)
        return similarity
    return float('inf')  # No valid contours

def provide_feedback(drawing, target_shape):
    """Provide feedback on drawing accuracy."""
    similarity = calculate_similarity(drawing, target_shape)
    if similarity < 0.1:  # Very similar to target
        speak("Great job! Your drawing is very accurate.")
    elif similarity < 0.3:  # Slight difference, but generally accurate
        speak("You're almost there! Just a few adjustments needed.")
    elif similarity < 0.5:  # Noticeable difference
        speak("You're close, but some parts need improvement.")
    else:  # Significant deviation
        speak("It seems you're off track. Try again.")

def tactile_surface_simulation():
    """Simulate tactile surface drawing."""
    # Initialize the canvas and target shape
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    target_shape = np.zeros_like(canvas)
    cv2.rectangle(target_shape, (100, 100), (400, 400), (255, 255, 255), 2)

    # Drawing variables
    drawing = False
    last_point = None

    def draw_circle(event, x, y, flags, param):
        """Mouse callback for drawing."""
        nonlocal drawing, last_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas, last_point, (x, y), (255, 255, 255), 2)  # Draw white line
            last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            last_point = None

    # Set up the drawing window
    cv2.namedWindow("Tactile Surface")
    cv2.setMouseCallback("Tactile Surface", draw_circle)

    speak("You can start drawing on the tactile surface now.")
    speak("Press 'r' to reset the canvas or 'f' to get feedback on your drawing.")
    while True:
        # Show the canvas
        cv2.imshow("Tactile Surface", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Reset canvas
            canvas[:] = 0
            speak("Canvas reset.")
        elif key == ord('f'):  # Feedback
            provide_feedback(canvas, target_shape)

    cv2.destroyAllWindows()

def main():
    """Main function to run the system."""
    speak("Welcome to the drawing assistant for visually impaired people.")
    while True:
        command = listen()
        if command:
            if "start" in command:
                speak("Let's start drawing.")
                tactile_surface_simulation()
            elif "exit" in command or "quit" in command:
                speak("Goodbye!")
                break
            else:
                speak("I didn't understand the command. Please say 'start' to begin or 'exit' to quit.")

if __name__ == "__main__":
    main()
