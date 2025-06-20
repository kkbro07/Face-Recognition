import os
import numpy as np
from tqdm import tqdm
from utils import get_face_embedding # <-- IMPORT THE NEW FUNCTION

def load_data_with_embeddings(data_dir):
    X, y = [], []
    people = os.listdir(data_dir)

    for person in tqdm(people, desc=f"Extracting embeddings from {data_dir}"):
        person_path = os.path.join(data_dir, person)
        if not os.path.isdir(person_path):
            continue
        image_files = os.listdir(person_path)

        for img_name in tqdm(image_files, desc=f"  {person}", leave=False):
            img_path = os.path.join(person_path, img_name)
            
            # Use the new embedding function
            embedding = get_face_embedding(img_path) 
            
            if embedding is not None:
                # The embedding is already a 1D list/vector, no need to flatten
                X.append(embedding) 
                y.append(person)
                
    return np.array(X), np.array(y)

# Re-run the data loading
print("[INFO] Loading training set...")
trainX, trainY = load_data_with_embeddings("dataset_train")
print("[INFO] Loading test set...")
testX, testY = load_data_with_embeddings("dataset_test")

os.makedirs("encoders", exist_ok=True)
# Save the new, powerful embeddings
np.savez_compressed("encoders/face_embeddings.npz", trainX=trainX, trainY=trainY, testX=testX, testY=testY)
print("[INFO] Step 2 complete: Embeddings saved.")