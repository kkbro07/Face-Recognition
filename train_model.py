# Step 3 - Train SVM model with detailed progress

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import dump
from tqdm import tqdm

# print("[INFO] Loading training data...")
# data = np.load("encoders/face_data.npz")
# trainX, trainY = data["trainX"], data["trainY"]

# ... (imports) ...
print("[INFO] Loading training data...")
data = np.load("encoders/face_embeddings.npz") # <-- LOAD THE NEW FILE
trainX, trainY = data["trainX"], data["trainY"]
# ... (rest of the script is identical) ...


print("[INFO] Training data loaded successfully.")
print(f"[INFO] Number of classes: {len(np.unique(trainY))}")
print(f"[INFO] Training samples: {trainX.shape[0]}, Features per sample: {trainX.shape[1]}")
print("[INFO] Performing cross-validation...")

# Optional: Pre-validation estimate using k-fold (progress bar shown)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, test_idx in tqdm(cv.split(trainX, trainY), total=5, desc="Cross-validation"):
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX[train_idx], trainY[train_idx])
    acc = model.score(trainX[test_idx], trainY[test_idx])
    scores.append(acc)

print(f"[INFO] Cross-validation accuracy: {np.mean(scores):.4f}")

# Final training
print("[INFO] Training final model (this may take a few minutes)...")
final_model = SVC(kernel='linear', probability=True, verbose=True)
final_model.fit(trainX, trainY)

dump(final_model, "encoders/face_model.joblib")
print("[INFO] Step 3 complete: Final model trained and saved.")
