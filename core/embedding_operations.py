import os
import numpy as np
from tqdm import tqdm
from utils import get_face_embedding

def add_new_person_to_embeddings(person_name, images_folder_path, embeddings_file_path):
    new_embeddings = []
    print(f"[INFO] Generating embeddings for new person: {person_name}")
    for img_name in tqdm(os.listdir(images_folder_path)):
        img_path = os.path.join(images_folder_path, img_name)
        embedding = get_face_embedding(img_path)
        if embedding is not None:
            new_embeddings.append(embedding)
    if not new_embeddings:
        print(f"[ERROR] No faces found for {person_name}. Aborting.")
        return False

    new_embeddings = np.array(new_embeddings)
    new_labels = np.array([person_name] * len(new_embeddings))

    print(f"[INFO] Loading existing embeddings from {embeddings_file_path}...")
    if os.path.exists(embeddings_file_path):
        data = np.load(embeddings_file_path, allow_pickle=True)
        all_X = data['trainX']
        all_y = data['trainY']
        all_testX = data['testX']
        all_testY = data['testY']
        all_X = np.concatenate((all_X, new_embeddings))
        all_y = np.concatenate((all_y, new_labels))
    else:
        all_X = new_embeddings
        all_y = new_labels
        all_testX, all_testY = np.array([]), np.array([])

    print("[INFO] Saving updated embeddings file...")
    np.savez_compressed(embeddings_file_path, trainX=all_X, trainY=all_y, testX=all_testX, testY=all_testY)
    print(f"[SUCCESS] Added {len(new_embeddings)} images of {person_name} to the dataset.")
    return True
