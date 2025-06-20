import cv2
import numpy as np
from joblib import load
from utils import get_face_embedding # <-- IMPORT THE NEW FUNCTION
import sys
import os

# Path to the image you want to test
image_path = "dataset/Angelina Jolie/012_cfcd4007.jpg" # Use an image from your test set
CONFIDENCE_THRESHOLD = 0.60  # You can likely lower this now

# Check file exists
if not os.path.exists(image_path):
    print(f"[ERROR] Image path '{image_path}' not found.")
    sys.exit()

# Get face embedding
embedding = get_face_embedding(image_path)
if embedding is None:
    print("[ERROR] No face found in the image or could not create embedding.")
    sys.exit()

# The embedding is already a flat vector, but the model expects a 2D array (list of samples)
face_embedding = np.array(embedding).reshape(1, -1)

# Load model (make sure this is the model trained on embeddings)
model = load("encoders/face_model.joblib")

# Predict
probs = model.predict_proba(face_embedding)[0]
best_idx = np.argmax(probs)
best_class = model.classes_[best_idx]
confidence = probs[best_idx]

# Decide if known or unknown
if confidence < CONFIDENCE_THRESHOLD:
    label = "Unknown"
    print(f"[INFO] Prediction: {best_class} (confidence too low: {confidence*100:.2f}%) -> Classified as Unknown")
else:
    label = f"{best_class}"
    print(f"[INFO] Prediction: {best_class} (confidence: {confidence*100:.2f}%)")

# Show image
img = cv2.imread(image_path)
cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if label == "Unknown" else (0, 255, 0), 2)
cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()