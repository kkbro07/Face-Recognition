
# Face Recognition with Deep Learning Embeddings and SVM

This project demonstrates a high-accuracy face recognition system built using Python. It leverages state-of-the-art deep learning models for feature extraction and a classic machine learning classifier for identification.

The pipeline takes a dataset of labeled face images, trains a model to recognize these individuals, and provides a script to identify a person in a new, unseen image. With the current implementation using ArcFace embeddings and a tuned SVC, the system achieves ~94% accuracy on the test set.

---

## Table of Contents

- [Features](#features)  
- [How It Works](#how-it-works)  
- [Project Structure](#project-structure)  
- [Setup and Installation](#setup-and-installation)  
- [How to Use](#how-to-use)  
- [Model Performance](#model-performance)  
- [Next Steps and Improvements](#next-steps-and-improvements)  

---

## Features

- ✅ **High Accuracy**: Achieves ~94% accuracy using state-of-the-art ArcFace embeddings.  
- 💡 **Robust Recognition**: Less sensitive to lighting, pose, and expression variations.  
- ⚡ **Fast Identification**: Once trained, predictions are near-instant.  
- 🧩 **Modular Codebase**: Logical script separation for easy extension.  
- 🚫 **"Unknown" Detection**: Uses a confidence threshold to flag unknown persons.  
- 🔍 **Hyperparameter Tuning**: Uses GridSearchCV for optimal SVM performance.

---

## How It Works

The system uses a traditional ML pipeline with modern DL-based embeddings:

1. **Dataset Preparation**: Organize images into subfolders by class (person).  
2. **Feature Extraction**: ArcFace model generates embeddings from face images.  
3. **Model Training**: A Support Vector Machine (SVM) is trained on those embeddings.  
4. **Prediction**: A face in a new image is identified by generating its embedding and comparing it with known ones.

---

## Project Structure

```
FACE-RECOGNITION/
├── dataset/                  # Original dataset: one folder per person
│   ├── Angelina Jolie/
│   └── Brad Pitt/
│
├── dataset_train/           # Auto-generated training set (from split)
├── dataset_test/            # Auto-generated testing set (from split)
│
├── encoders/                # Stores face embeddings and trained SVM model
│   ├── face_embeddings.npz
│   └── face_model_tuned.joblib
│
├── 01_split_dataset.py      # Script to split the dataset into train/test
├── 02_create_embeddings.py  # Extracts face embeddings using ArcFace
├── 03_train_model.py        # Trains the SVM model with GridSearchCV
├── 04_test_model.py         # Evaluates the model's accuracy on test set
│
├── predict_one_image.py     # Script to make a prediction on a single image
├── utils.py                 # Utility functions (face detection, embedding, etc.)
└── README.md                # Project documentation (this file)


## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/kkbro07/Face-Recognition.git
cd Face-Recognition
````

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate       # On Windows
# source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install numpy opencv-python scikit-learn joblib tqdm deepface
```

---

## How to Use

### 1. Split the dataset

```bash
python 01_split_dataset.py
```

### 2. Generate embeddings

```bash
python 02_create_embeddings.py
```

### 3. Train the SVM model

```bash
python 03_train_model.py
```

### 4. Evaluate the model (optional)

```bash
python 04_test_model.py
```

### 5. Predict a single image

```bash
python predict_one_image.py "path/to/image.jpg"
```

Example:

```bash
python predict_one_image.py dataset_test/Tom Hanks/tom_hanks_20.jpg
```

---

## Model Performance

* **Overall Accuracy**: 93.7%
* **Observations**:

  * High accuracy across most classes.
  * Some classes (e.g., "Kate Winslet", "Jennifer Lawrence") show slightly lower precision.
  * Improved curation and class balancing can enhance results further.

---

## Next Steps and Improvements

* 📷 **Real-Time Recognition**: Extend to webcam input using `cv2.VideoCapture`.
* 🔒 **Liveness Detection**: Prevent spoofing via photo attacks.
* 🖥 **GUI Development**: Build a user-friendly interface using Tkinter, Streamlit, or PyQt.
* 🧼 **Data Cleaning**: Manually remove noisy, blurred, or mislabeled images.
* 🧠 **Model Experimentation**: Try other embedding models like VGGFace, Facenet, or Dlib.

---

### License

This project is open-source and free to use for educational purposes.

---

### Author

Created by [kkbro07](https://github.com/kkbro07)

---

```

Let me know if you’d like a version that includes badges, demo images, or setup scripts!
```
