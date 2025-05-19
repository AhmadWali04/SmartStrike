import cv2
import os
import mediapipe as mp
import numpy as np
from joblib import load
import datetime
import corrections  

BASE_DIR = os.getenv('SMARTSTRIKE_BASE', '.')
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, 'strike_model.pkl'))
REPORT_PATH = os.getenv('REPORT_PATH', os.path.join(BASE_DIR, 'report.txt'))

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
model = load(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

moves = ["jab", "cross", "left_hook", "right_hook", "uppercut", "kick"]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit(1)

print("Starting real-time strike classification. Press 'q' to quit.")

with open(REPORT_PATH, 'a') as report_file:
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_file.write(f"\n=== Session started at {start_time} ===\n")
    prev_feedback = None
    prev_move = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)  
        if not results.pose_landmarks:
            cv2.putText(frame, "No pose detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('SmartStrike - Live', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)  

        landmarks = results.pose_landmarks.landmark
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])
        features = np.array(features, dtype=np.float32).reshape(1, -1)  

        pred_label = int(model.predict(features)[0])
        pred_move = moves[pred_label] if 0 <= pred_label < len(moves) else "Unknown"

        feedback_list = corrections.get_corrections(pred_move, landmarks)
        feedback_str = "; ".join(feedback_list)
        cv2.putText(frame, f"Move: {pred_move}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if feedback_list:
            for i, msg in enumerate(feedback_list[:2]):
                cv2.putText(frame, f"Fix: {msg}", (10, 60 + 30*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('SmartStrike - Live', frame)

        if feedback_list and feedback_str != prev_feedback:
            log_time = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{log_time}] {pred_move} - Corrections: {feedback_str}\n"
            report_file.write(log_entry)
            report_file.flush()
            print(log_entry.strip())
            prev_feedback = feedback_str
        elif not feedback_list:
            prev_feedback = ""  
        prev_move = pred_move

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Inference session ended. Feedback log saved to", REPORT_PATH)
