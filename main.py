import cv2
import logging
import time
from ultralytics import YOLO
import numpy as np
import threading
from queue import Queue

# Suppress Ultralytics logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Load YOLO model (using a small model, e.g. YOLO11n)
model = YOLO('yolo11n.pt')

# Reduce input resolution for YOLO (improves speed on Raspberry Pi)
model.args['imgsz'] = 320  # YOLO will work on 320x320 images

# Initialize video capture (for example, a Raspberry Pi camera)
cap = cv2.VideoCapture(0)
# Here you set the camera resolution.
# If your camera’s max width is higher than 320 and you want to capture at maximum resolution,
# you could set these properties accordingly. In your case you seem to want a width of 320:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Maximum width supported by your setup
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Maximum height supported by your setup
cap.set(cv2.CAP_PROP_FPS, 30)  # Maximum FPS supported by the camera

# Queue to manage frames for detection
detection_queue = Queue()

def inference_worker():
    while True:
        # Get frame from the queue (this is the resized 320x320 frame used for inference)
        frame = detection_queue.get()
        if frame is None:
            break  # Exit signal

        # Run YOLO detection on the frame
        results = model(frame)

        # Process detection results
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            confidences = results[0].boxes.conf.cpu().numpy()
            # Find indices of detections with high confidence (>= 0.75)
            high_conf_indices = np.where(confidences >= 0.75)[0]
            if len(high_conf_indices) > 0:
                # For example, take the first high-confidence detection
                idx = high_conf_indices[0]
                bbox = results[0].boxes.xyxy.cpu().numpy()[idx].astype(int)
                x1, y1, x2, y2 = bbox

                # Note: The frame here is 320x320 because you resized it before inference.
                # If you wish to crop from the original full-resolution frame,
                # you must either run detection on the full-resolution image
                # or map the coordinates back to the original size.
                object_frame = frame[y1:y2, x1:x2]

                # Resize the object frame so that its width is 320, keeping the aspect ratio:
                if object_frame.shape[1] > 0:  # Avoid division by zero
                    current_width = object_frame.shape[1]
                    scale = 320 / current_width
                    new_height = int(object_frame.shape[0] * scale)
                    object_frame_resized = cv2.resize(object_frame, (320, new_height))

                    # Display the cropped and resized object frame
                    cv2.imshow("Object Frame", object_frame_resized)

                # (Optional) Print detected object information
                detected_class = results[0].boxes.cls.cpu().numpy().astype(int)[idx]
                print(f"Detected: {model.names[detected_class]}, Confidence: {confidences[idx]}")

        # Mark the task as done
        detection_queue.task_done()

# Start the inference worker thread
thread = threading.Thread(target=inference_worker, daemon=True)
thread.start()

# Main loop for capturing frames
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the captured frame to 320x320 for YOLO inference.
        # (This is why your detection results are in 320x320 coordinate space.)
        resized_frame = cv2.resize(frame, (320, 320))

        # Add the resized frame to the detection queue
        detection_queue.put(resized_frame)

        # Display the original frame (or you could display the resized_frame)
        cv2.imshow('YOLO11 Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # Stop the inference worker thread
    detection_queue.put(None)
    thread.join()
