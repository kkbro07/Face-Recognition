# Place this in a utils.py file or at the top of your scripts that need it.
from deepface import DeepFace
import logging

# Suppress excessive logging from deepface
logging.getLogger('deepface').setLevel(logging.ERROR)

def get_face_embedding(image_path):
    """
    Detects a face in an image and returns its deep learning embedding.
    Returns None if no face is detected.
    """
    try:
        # DeepFace.represent will handle face detection and embedding extraction.
        # It uses MTCNN for detection by default, which is better than Haar Cascades.
        # It uses VGGFace for embeddings by default.
        embedding_objs = DeepFace.represent(img_path=image_path, 
                                            model_name='VGG-Face', 
                                            enforce_detection=True) # Fails if no face is found
        
        # The result is a list of objects, one for each face. We take the first one.
        embedding = embedding_objs[0]['embedding']
        return embedding
    except ValueError as e:
        # This error is typically raised by deepface if no face is detected
        # print(f"[WARNING] No face detected in {image_path}: {e}")
        return None
    except Exception as e:
        # print(f"[ERROR] An unexpected error occurred with {image_path}: {e}")
        return None