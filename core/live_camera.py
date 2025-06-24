import cv2
import numpy as np
from deepface import DeepFace
from utils import get_face_embedding

def start_webcam(model, confidence_threshold=0.60):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam.")

    frame_counter = 0
    last_results = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % 5 == 0:
            try:
                faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
                current_frame_results = {}

                for face_obj in faces:
                    if face_obj['confidence'] > 0.9:
                        x, y, w, h = face_obj['facial_area']['x'], face_obj['facial_area']['y'], face_obj['facial_area']['w'], face_obj['facial_area']['h']
                        embedding_obj = DeepFace.represent(face_obj['face'], model_name='VGG-Face', enforce_detection=False)
                        embedding = embedding_obj[0]['embedding']
                        pred_class, confidence = predict_person(np.array(embedding), model)

                        label, color = ("Unknown", (0, 0, 255))
                        if confidence > confidence_threshold:
                            label, color = (pred_class, (0, 255, 0))

                        text = f"{label} ({confidence*100:.0f}%)"
                        current_frame_results[(x, y, w, h)] = (text, color)

                last_results = current_frame_results
            except Exception:
                last_results = {}

        if last_results:
            for (x, y, w, h), (text, color) in last_results.items():
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame_counter += 1
        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def predict_person(face_embedding, model):
    face_embedding = face_embedding.reshape(1, -1)
    probs = model.predict_proba(face_embedding)[0]
    best_idx = np.argmax(probs)
    predicted_class = model.classes_[best_idx]
    confidence = probs[best_idx]
    return predicted_class, confidence
