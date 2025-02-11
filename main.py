import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, Label
from PIL import Image, ImageTk

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

root = Tk()
label = Label(root)
label.pack()

def detect_and_display():
    ret, frame = cap.read()
    if not ret:
        return

    results = model(frame)
    annotated_frame = results[0].plot()

    img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    frame_rate = 10
    delay = int(1000 / frame_rate)

    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(delay, detect_and_display)

detect_and_display()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
