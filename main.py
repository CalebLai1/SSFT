import cv2
import logging
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTextToSpeech import QTextToSpeech
from ultralytics import YOLO
import numpy as np
import threading
from queue import Queue

# Suppress Ultralytics logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Initialize PyQt application
app = QApplication(sys.argv)
tts = QTextToSpeech()

# Queue to manage TTS messages
tts_queue = Queue()
tts_lock = threading.Lock()

def tts_worker():
    while True:
        message = tts_queue.get()
        if message is None:
            break  # Exit signal
        tts.say(message)
        # Wait until the current speech is finished
        while tts.state() == QTextToSpeech.State.Speaking:
            QApplication.processEvents()
        tts_queue.task_done()

# Start TTS worker thread
thread = threading.Thread(target=tts_worker, daemon=True)
thread.start()

# Load YOLO model
model = YOLO('yolo11n.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow('YOLO11 Detection', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Annotate frame with detection results
    annotated_frame = results[0].plot()

    # Display annotated frame
    cv2.imshow('YOLO11 Detection', annotated_frame)

    # Text-to-Speech for detected objects
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # Filter detections by confidence >= 75%
        confidences = results[0].boxes.conf.cpu().numpy()
        high_conf_indices = np.where(confidences >= 0.75)[0]
        if len(high_conf_indices) > 0:
            detected_classes = results[0].boxes.cls.cpu().numpy().astype(int)[high_conf_indices]
            detected_confidences = confidences[high_conf_indices]
            detected_objects = [(model.names[cls_id], confidences[idx]) for idx, cls_id in enumerate(detected_classes)]

            # Prepare the message
            object_names = [f"{name} ({conf*100:.1f}%)" for name, conf in detected_objects]
            message = f"Detected: {', '.join(object_names)}"

            # Add message to TTS queue
            tts_queue.put(message)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

# Stop TTS worker thread
tts_queue.put(None)
thread.join()
