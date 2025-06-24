# File: /core/clustering.py
import os
import shutil
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from utils import get_face_embedding

def cluster_unknown_faces(input_dir, output_dir):
    """
    Clusters faces from an input directory using DBSCAN and organizes them into folders.
    
    Returns:
        tuple: (number_of_clusters, number_of_noise_points)
    """
    embeddings = []
    image_paths = []
    
    print(f"[INFO] Generating embeddings for images in {input_dir}...")
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in tqdm(all_files):
        img_path = os.path.join(input_dir, img_name)
        embedding = get_face_embedding(img_path)
        if embedding is not None:
            embeddings.append(embedding)
            image_paths.append(img_path)

    if not embeddings:
        print("[WARNING] No faces found to cluster.")
        return 0, 0
        
    print(f"[INFO] Found {len(embeddings)} faces. Starting DBSCAN clustering...")
    # DBSCAN parameters:
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    # Adjust 'eps' based on the embedding model. For ArcFace/VGG-Face, 0.4-0.6 is a good starting point.
    clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=0.5, min_samples=3)
    clt.fit(embeddings)
    
    # Get unique labels. -1 is for noise (outliers).
    label_ids = np.unique(clt.labels_)
    num_unique_faces = len(np.where(label_ids > -1)[0])
    num_noise_points = len(np.where(clt.labels_ == -1)[0])
    
    print(f"[INFO] Found {num_unique_faces} unique clusters and {num_noise_points} outlier images.")
    
    # Create output directories and copy images
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for label in label_ids:
        # Get paths for all images with the current label
        idxs = np.where(clt.labels_ == label)[0]
        
        # Determine the folder name
        if label == -1:
            cluster_folder_name = "outliers"
        else:
            cluster_folder_name = f"person_{label:02d}"
        
        cluster_path = os.path.join(output_dir, cluster_folder_name)
        os.makedirs(cluster_path, exist_ok=True)
        
        # Copy the images to the new cluster folder
        for i in idxs:
            shutil.copy(image_paths[i], cluster_path)

    return num_unique_faces, num_noise_points