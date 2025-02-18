import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, Label
from PIL import Image, ImageTk

# Load the YOLO model with the specified weights file
model = YOLO("yolo11n.pt")

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Initialize the Tkinter root window
root = Tk()
label = Label(root)
label.pack()

def detect_and_display():
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
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
    frame_rate = 30  # Upgraded from 10 to 30 for smoother video
    delay = int(1000 / frame_rate)

    # Update the Tkinter label with the new image
    label.imgtk = imgtk
    label.configure(image=imgtk)
    
    # Schedule the next frame to be processed after the specified delay
    label.after(delay, detect_and_display)

# Start the detection and display loop
detect_and_display()

# Start the Tkinter main loop to keep the window open
root.mainloop()

# Release the webcam and close all OpenCV windows when the application exits
cap.release()
cv2.destroyAllWindows()
