import cv2
from ultralytics import YOLO
import time
import platform
import subprocess

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

paused = False
speech_delay = 5  
last_spoken_time = {}  

cv2.namedWindow("YOLO Object Detection")

def on_trackbar(val):
    global speech_delay
    speech_delay = val

cv2.createTrackbar("Speech Delay (s)", "YOLO Object Detection", speech_delay, 10, on_trackbar)

def speak_object(class_name):
    system_platform = platform.system()
    if system_platform == "Windows":
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Speak(class_name)
    elif system_platform == "Darwin": 
        subprocess.run(["say", class_name])
    elif system_platform == "Linux":
        subprocess.run(["espeak", class_name])
    else:
        print(f"Text-to-speech not supported on {system_platform}")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        results = model(frame)

        for result in results:
            boxes = result.boxes.xyxy  
            conf = result.boxes.conf   
            cls = result.boxes.cls     
            names = result.names       

            for box, confidence, class_id in zip(boxes, conf, cls):
                x1, y1, x2, y2 = map(int, box)  
                class_name = names[int(class_id)]  

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                current_time = time.time()
                if class_name not in last_spoken_time or (current_time - last_spoken_time[class_name]) >= speech_delay:
                    speak_object(class_name)
                    last_spoken_time[class_name] = current_time

    cv2.imshow("YOLO Object Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  
        break
    elif key == ord('p'):  
        paused = not paused
        print("Paused" if paused else "Resumed")

cap.release()
cv2.destroyAllWindows()
