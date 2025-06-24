import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import os
import sys
import threading
import numpy as np
import cv2
import time
from tqdm import tqdm
from deepface import DeepFace
from core.live_camera import start_webcam
from core.model_operations import load_model, predict_person, train_new_model
from core.embedding_operations import add_new_person_to_embeddings
from core.clustering import cluster_unknown_faces
from utils import get_face_embedding

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

MODEL_PATH = resource_path("encoders/face_model.joblib")
EMBEDDINGS_PATH = resource_path("encoders/face_embeddings.npz")
CONFIDENCE_THRESHOLD = 0.60
UNKNOWN_OUTPUT_DIR = resource_path("outputs/clustered_unknowns")

class FaceRecognitionApp(TkinterDnD.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Face Recognition System")
        self.geometry("800x700")
        self.model = None
        self.webcam_running = False
        self.cap = None
        self.load_app_model()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")
        self.predict_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_tab, text="Predict Image")
        self.setup_predict_tab()
        self.live_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.live_tab, text="Live Recognition")
        self.setup_live_tab()
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="Add New Person")
        self.setup_train_tab()
        self.cluster_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.cluster_tab, text="Cluster Unknowns")
        self.setup_cluster_tab()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.webcam_running = False
        time.sleep(0.5)
        if self.cap is not None:
            self.cap.release()
        self.destroy()

    def load_app_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showwarning("Model Not Found", f"Model not found at {MODEL_PATH}")
            return
        self.model = load_model(MODEL_PATH)

    def run_in_thread(self, target_func, *args):
        thread = threading.Thread(target=target_func, args=args)
        thread.daemon = True
        thread.start()

    def setup_live_tab(self):
        self.live_display_label = tk.Label(self.live_tab, text="Webcam Feed", bg="black")
        self.live_display_label.pack(pady=10, padx=10, expand=True, fill="both")
        button_frame = ttk.Frame(self.live_tab)
        button_frame.pack(pady=10)
        self.cam_button = tk.Button(button_frame, text="Start Camera", command=self.toggle_camera, bg="lightgreen", width=15)
        self.cam_button.pack(side=tk.LEFT, padx=10)
        self.progress_bar = ttk.Progressbar(self.live_tab, orient="horizontal", length=200, mode="indeterminate")
        self.progress_bar.pack(pady=10)

    def toggle_camera(self):
        if not self.webcam_running:
            self.start_webcam()
        else:
            self.stop_webcam()

    def start_webcam(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Cannot start live recognition.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.cap = None
            return
        self.webcam_running = True
        self.cam_button.config(text="Stop Camera", bg="salmon")
        self.progress_bar.start()
        self.run_in_thread(self.video_loop)

    def stop_webcam(self):
        self.webcam_running = False
        self.cam_button.config(text="Start Camera", bg="lightgreen")
        self.progress_bar.stop()
        self.after(200, lambda: self.live_display_label.config(image='', text="Webcam Feed", bg="black"))

    def video_loop(self):
        frame_counter = 0
        last_results = {}
        while self.webcam_running:
            ret, frame = self.cap.read()
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
                            pred_class, confidence = predict_person(np.array(embedding), self.model)
                            label, color = ("Unknown", (0, 0, 255))
                            if confidence > CONFIDENCE_THRESHOLD:
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
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img.thumbnail((700, 500))
            photo = ImageTk.PhotoImage(pil_img)
            self.live_display_label.config(image=photo)
            self.live_display_label.image = photo
        if self.cap:
            self.cap.release()
        self.cap = None

    def setup_predict_tab(self):
        self.display_label = tk.Label(self.predict_tab, text="Drag & Drop Image Here", font=("Helvetica", 14), bg="lightgrey", relief="solid", borderwidth=1)
        self.display_label.pack(pady=20, padx=20, expand=True, fill="both")
        self.result_label = tk.Label(self.predict_tab, text="Prediction: -", font=("Helvetica", 16))
        self.result_label.pack(pady=10)
        tk.Button(self.predict_tab, text="List Detectable Persons", command=self.list_detectable_persons, bg="lightgreen").pack(pady=10)
        self.display_label.drop_target_register(DND_FILES)
        self.display_label.dnd_bind('<<Drop>>', self.handle_drop)
        self.progress_bar = ttk.Progressbar(self.predict_tab, orient="horizontal", length=200, mode="indeterminate")
        self.progress_bar.pack(pady=10)

    def handle_drop(self, event):
        filepath = event.data.strip('{}')
        if not filepath: return
        self.display_label.config(text="Processing...", image='')
        self.progress_bar.start()
        self.run_in_thread(self.process_prediction, filepath)

    def process_prediction(self, filepath):
        if self.model is None:
            self.result_label.config(text="Model not loaded.", fg="red")
            return
        embedding = get_face_embedding(filepath)
        if embedding is None:
            self.result_label.config(text="Error: No face detected!", fg="red")
            self.display_label.config(text="Drag & Drop Image Here", image='')
            return
        predicted_class, confidence = predict_person(np.array(embedding), self.model)
        final_label, color, text_color_bgr = ("Unknown", "red", (0, 0, 255))
        if confidence >= CONFIDENCE_THRESHOLD:
            final_label, color, text_color_bgr = (predicted_class, "green", (0, 255, 0))
        result_text = f"Prediction: {final_label} ({confidence*100:.2f}%)"
        self.result_label.config(text=result_text, fg=color)
        img_bgr = cv2.imread(filepath)
        cv2.putText(img_bgr, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color_bgr, 2, cv2.LINE_AA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((self.display_label.winfo_width(), self.display_label.winfo_height()))
        photo = ImageTk.PhotoImage(pil_img)
        self.display_label.config(image=photo, text="")
        self.display_label.image = photo
        self.progress_bar.stop()

    def setup_train_tab(self):
        tk.Label(self.train_tab, text="Add a New Person to the Dataset", font=("Helvetica", 14)).pack(pady=10)
        tk.Label(self.train_tab, text="Person's Name:").pack()
        self.person_name_entry = tk.Entry(self.train_tab, width=40)
        self.person_name_entry.pack()
        tk.Label(self.train_tab, text="Path to Images Folder:").pack()
        self.folder_path_label = tk.Label(self.train_tab, text="No folder selected", bg="lightgrey")
        self.folder_path_label.pack()
        tk.Button(self.train_tab, text="Select Folder", command=self.select_folder).pack(pady=5)
        tk.Button(self.train_tab, text="Start Training Process", command=self.start_training_process, bg="lightblue").pack(pady=20)
        tk.Button(self.train_tab, text="List Recognizable Persons", command=self.list_recognizable_persons, bg="lightgreen").pack(pady=10)
        self.train_console = tk.Text(self.train_tab, height=15, state='disabled')
        self.train_console.pack(pady=10, padx=10, fill="both", expand=True)
        self.progress_bar = ttk.Progressbar(self.train_tab, orient="horizontal", length=200, mode="indeterminate")
        self.progress_bar.pack(pady=10)

    def log_to_console(self, console, message):
        console.config(state='normal')
        console.insert(tk.END, message + "\n")
        console.config(state='disabled')
        console.see(tk.END)

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path: self.folder_path_label.config(text=folder_path)

    def start_training_process(self):
        name, folder = self.person_name_entry.get().strip(), self.folder_path_label.cget("text")
        if not name or folder == "No folder selected":
            messagebox.showerror("Error", "Please provide a name and select a folder.")
            return
        if self.is_person_already_trained(name):
            messagebox.showerror("Error", f"Person '{name}' is already trained in the model.")
            return
        self.progress_bar.start()
        self.run_in_thread(self.run_training, name, folder)

    def is_person_already_trained(self, person_name):
        if self.model is None:
            return False
        return person_name in self.model.classes_

    def run_training(self, name, folder):
        self.log_to_console(self.train_console, f"--- Starting process for {name} ---")
        if add_new_person_to_embeddings(name, folder, EMBEDDINGS_PATH):
            self.log_to_console(self.train_console, "Embeddings updated. Now retraining model...")
            self.model = train_new_model(EMBEDDINGS_PATH, MODEL_PATH)
            self.log_to_console(self.train_console, "--- Process Complete! Model is updated. ---")
            messagebox.showinfo("Success", "New person added and model retrained!")
        else:
            self.log_to_console(self.train_console, "--- Process Failed. ---")
        self.progress_bar.stop()

    def list_recognizable_persons(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded.")
            return
        recognizable_persons = self.model.classes_
        messagebox.showinfo("Recognizable Persons", f"Recognizable Persons:\n{', '.join(recognizable_persons)}")

    def list_detectable_persons(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded.")
            return
        detectable_persons = self.model.classes_
        messagebox.showinfo("Detectable Persons", f"Detectable Persons:\n{', '.join(detectable_persons)}")

    def setup_cluster_tab(self):
        tk.Label(self.cluster_tab, text="Group Unknown Faces with Unsupervised Learning", font=("Helvetica", 14)).pack(pady=10)
        tk.Label(self.cluster_tab, text="Folder with unlabelled images:").pack()
        self.unlabelled_folder_label = tk.Label(self.cluster_tab, text="No folder selected", bg="lightgrey")
        self.unlabelled_folder_label.pack()
        tk.Button(self.cluster_tab, text="Select Folder", command=self.select_unlabelled_folder).pack(pady=5)
        tk.Button(self.cluster_tab, text="Start Clustering", command=self.start_clustering, bg="lightgreen").pack(pady=20)
        self.cluster_console = tk.Text(self.cluster_tab, height=15, state='disabled')
        self.cluster_console.pack(pady=10, padx=10, fill="both", expand=True)
        self.progress_bar = ttk.Progressbar(self.cluster_tab, orient="horizontal", length=200, mode="indeterminate")
        self.progress_bar.pack(pady=10)

    def select_unlabelled_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path: self.unlabelled_folder_label.config(text=folder_path)

    def start_clustering(self):
        folder = self.unlabelled_folder_label.cget("text")
        if folder == "No folder selected":
            messagebox.showerror("Error", "Please select a folder of unlabelled images.")
            return
        self.progress_bar.start()
        self.run_in_thread(self.run_clustering, folder)

    def run_clustering(self, folder):
        self.log_to_console(self.cluster_console, f"--- Starting clustering in {folder} ---")
        num_clusters, num_noise = cluster_unknown_faces(folder, UNKNOWN_OUTPUT_DIR)
        self.log_to_console(self.cluster_console, f"Found {num_clusters} potential new people and {num_noise} outliers.")
        self.log_to_console(self.cluster_console, f"--- Clustering complete! Check '{UNKNOWN_OUTPUT_DIR}' folder. ---")
        messagebox.showinfo("Success", f"Clustering complete! Found {num_clusters} groups.")
        self.progress_bar.stop()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()
