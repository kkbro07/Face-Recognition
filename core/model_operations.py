# File: /core/model_operations.py

from sklearn.svm import SVC
from joblib import dump, load
import numpy as np

# These are the best parameters you found with GridSearchCV.
# If you run it again and find better ones, update them here.
BEST_SVM_PARAMS = {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'} # Example: Update with your best params

def train_new_model(embeddings_path, save_path):
    """
    Trains a new SVM model on the full embedding dataset.
    Args:
        embeddings_path (str): Path to the face_embeddings.npz file.
        save_path (str): Path to save the new .joblib model file.
    """
    print(f"[INFO] Loading embeddings from {embeddings_path}...")
    data = np.load(embeddings_path)
    trainX, trainY = data["trainX"], data["trainY"]
    
    print(f"[INFO] Training new SVM model with optimal parameters: {BEST_SVM_PARAMS}")
    
    # Use the best parameters we found earlier
    model = SVC(probability=True, **BEST_SVM_PARAMS)
    model.fit(trainX, trainY)
    
    print(f"[INFO] Training complete. Saving model to {save_path}...")
    dump(model, save_path)
    print("[SUCCESS] New model saved.")
    return model

def load_model(model_path):
    """Loads a pre-trained .joblib model."""
    print(f"[INFO] Loading model from {model_path}...")
    return load(model_path)

def predict_person(face_embedding, model):
    """
    Predicts a person from a single face embedding.
    Args:
        face_embedding (np.array): A single face embedding.
        model: A loaded scikit-learn model.
    
    Returns:
        tuple: (predicted_class, confidence_score)
    """
    # Reshape for the model
    face_embedding = face_embedding.reshape(1, -1)
    
    # Predict probabilities
    probs = model.predict_proba(face_embedding)[0]
    best_idx = np.argmax(probs)
    
    predicted_class = model.classes_[best_idx]
    confidence = probs[best_idx]
    
    return predicted_class, confidence