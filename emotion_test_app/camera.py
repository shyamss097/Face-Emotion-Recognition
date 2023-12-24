# emotion_test_app/camera.py

import cv2
from deepface import DeepFace
import numpy as np
import time

face_cascade = cv2.CascadeClassifier("path/to/haarcascade_frontal_default.xml")
emotion_history = []
timestamp_file_path = "emotion_test_app/emotion_timestamps.txt"

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame_with_emotion(self):
        success, frame = self.video.read()
        emotion = None

        if success:
            # Analyze emotion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            response = DeepFace.analyze(frame, actions=('emotion',), enforce_detection=False)

            for x, y, w, h in faces:
                img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # Display the emotion on the frame
                cv2.putText(frame, text=response[0]['dominant_emotion'], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), org=(x, y))

                current_emotion = response[0]['dominant_emotion']
                if not emotion_history or emotion_history[-1] != current_emotion:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    emotion_history.append(current_emotion)

                    # Append the timestamp and emotion to the file
                    with open(timestamp_file_path, 'a') as timestamp_file:
                        timestamp_file.write(f"{timestamp} - Emotion: {current_emotion}\n")

                emotion = current_emotion

            # Encode the frame in JPEG format
            _, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(), emotion

        return None, None
