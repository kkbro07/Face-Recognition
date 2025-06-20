import numpy as np
from joblib import load
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

print("[INFO] Loading test data...")
data = np.load("encoders/face_embeddings.npz")
testX, testY = data["testX"], data["testY"]

model = load("encoders/face_model.joblib")

print("[INFO] Predicting on test data...")
predictions = [model.predict([x])[0] for x in tqdm(testX, desc="Testing")]

print("[INFO] Step 4 complete: Model evaluation")
print("Accuracy:", accuracy_score(testY, predictions))
print(classification_report(testY, predictions))
print("[INFO] Evaluation complete. Check the accuracy and classification report above.")
# This script evaluates the trained model on the test dataset and prints the accuracy and classification report.