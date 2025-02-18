import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, Label, messagebox
from PIL import Image, ImageTk

# Load the YOLO model with the specified weights file
try:
    model = YOLO("yolo11n.pt")  # Upgrade 4: Error handling for model loading
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Initialize the webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():  # Upgrade 4: Check if the webcam is accessible
    messagebox.showerror("Error", "Unable to access webcam. Please check your camera.")
    exit(1)

# Initialize the Tkinter root window
root = Tk()
root.title("YOLO Object Detection")  # Upgrade 5: Add a title to the window
label = Label(root)
label.pack()

# Set a lower resolution for better performance (Upgrade 5: Performance optimization)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480 pixels

def detect_and_display():
    try:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")  # Upgrade 4: Error handling
            return

        # Perform object detection on the captured frame using the YOLO model
        results = model(frame)

        # Annotate the frame with the detection results
        annotated_frame = results[0].plot()

        # Convert the annotated frame from BGR to RGB format
        img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Convert the image to a PIL format
        img = Image.fromarray(img)

        # Convert the PIL image to a format that Tkinter can display
        imgtk = ImageTk.PhotoImage(image=img)

        # Set the frame rate for the display (frames per second)
        frame_rate = 30  # Upgrade 1: Increased from 10 to 30 for smoother video
        delay = int(1000 / frame_rate)

        # Update the Tkinter label with the new image
        label.imgtk = imgtk
        label.configure(image=imgtk)

        # Schedule the next frame to be processed after the specified delay
        label.after(delay, detect_and_display)

    except Exception as e:
        print(f"Error during detection or display: {e}")  # Upgrade 4: Error handling
        cap.release()
        cv2.destroyAllWindows()
        root.quit()

# Start the detection and display loop
detect_and_display()

# Start the Tkinter main loop to keep the window open
root.mainloop()

# Release the webcam and close all OpenCV windows when the application exits
cap.release()
cv2.destroyAllWindows()
