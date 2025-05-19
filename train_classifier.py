# train_classifier.py
# Trains a Random Forest classifier on the pose dataset and evaluates its accuracy.
# Splits the data into training and testing sets, trains the model, prints performance, and saves the model.

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt

BASE_DIR = os.getenv('SMARTSTRIKE_BASE', '.')
DATASET_PATH = os.getenv('DATASET_PATH', os.path.join(BASE_DIR, 'dataset.npz'))
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, 'strike_model.pkl'))

if not os.path.isfile(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}. Run create_dataset.py first.")

data = np.load(DATASET_PATH)
X = data['X']
y = data['y']
print(f"Loaded dataset from {DATASET_PATH}. Total samples={len(X)}, feature length={X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

plt.figure(figsize=(6,6))
plt.imshow(conf_matrix, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
classes = ["jab", "cross", "left_hook", "right_hook", "uppercut", "kick"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.xlabel("Predicted")
plt.ylabel("True")
plot_path = os.path.join(BASE_DIR, "confusion_matrix.png")
plt.savefig(plot_path)
print(f"Confusion matrix plot saved as {plot_path}")

os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)
dump(clf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    pass
