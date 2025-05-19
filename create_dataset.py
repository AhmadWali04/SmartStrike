import os
import cv2
import mediapipe as mp
import numpy as np

BASE_DIR = os.getenv('SMARTSTRIKE_BASE', '.')
MOVES_DIR = os.getenv('MOVES_DIR', os.path.join(BASE_DIR, 'Moves'))
DATASET_PATH = os.getenv('DATASET_PATH', os.path.join(BASE_DIR, 'dataset.npz'))

moves = ["jab", "cross", "left_hook", "right_hook", "uppercut", "kick"]

X = []
y = []

mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    for label, move in enumerate(moves):
        move_folder = os.path.join(MOVES_DIR, move)
        if not os.path.isdir(move_folder):
            print(f"Warning: directory {move_folder} does not exist. Skipping {move}.")
            continue
       
        for filename in os.listdir(move_folder):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  
            img_path = os.path.join(move_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: could not read {img_path}, skipping.")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)  
            if not results.pose_landmarks:
                
                continue
           
            landmarks = results.pose_landmarks.landmark
           
            features = []
            for lm in landmarks:
                features.extend([lm.x, lm.y, lm.z])
            X.append(features)
            y.append(label)
        print(f"Processed images for move '{move}'. Total samples so far: {len(X)}")


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f"Extracted pose features for {len(X)} images. Saving dataset...")

os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
np.savez(DATASET_PATH, X=X, y=y)
print(f"Dataset saved to {DATASET_PATH} (X.shape={X.shape}, y.shape={y.shape})")

if __name__ == "__main__":
    pass  
