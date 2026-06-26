# Cat & Dog Classifier (API & Model Deployment) 

An end-to-end **Deep Learning** and **Backend API** project that classifies uploaded images as either **cats** or **dogs** using a Convolutional Neural Network (CNN).

This repository demonstrates the complete machine learning workflow, from dataset cleaning and preprocessing to CNN training and deployment through a high-performance asynchronous **FastAPI** service.

---

## Highlights

* ✅ End-to-end Machine Learning pipeline
* ✅ TensorFlow/Keras CNN architecture
* ✅ Data cleaning and validation script
* ✅ Data augmentation to reduce overfitting
* ✅ EarlyStopping + custom manual stop callback
* ✅ FastAPI asynchronous REST API
* ✅ Interactive Swagger API documentation

---

## Features

### Data Sanitization & Cleaning

The `data-cleaning.py` script automatically scans the dataset directory and permanently removes:

* Corrupted images
* Unreadable files
* Invalid image formats

This ensures the training dataset contains only valid images.

---

### Robust CNN Architecture

The CNN is implemented using **TensorFlow/Keras** and includes several techniques to improve generalization:

* RandomFlip
* RandomRotation
* RandomZoom
* Dropout layers

These augmentations help reduce overfitting and improve performance on unseen images.

---

### Training Callbacks

Besides the standard **EarlyStopping** callback, the project implements a custom callback named **ManuelDurdurmaCallback**.

During training, the callback checks whether a temporary `stop.txt` file exists in the project root.

If detected, training stops gracefully at the end of the current epoch while preserving the best model weights.

This allows long training sessions to be interrupted safely without losing progress.

---

### High-Performance API

The trained model is served through an asynchronous **FastAPI** application.

Features include:

* Asynchronous image upload
* Low-latency predictions
* JSON API responses
* Production-ready architecture

---

# Project Structure

```text
cat-dog-classifier/
├── main.py               # FastAPI endpoints & model inference
├── train.py              # CNN architecture & training pipeline
├── data-cleaning.py      # Dataset verification & cleaning
├── requirements.txt      # Project dependencies
└── .gitignore            # Ignores datasets and trained model files
```

---

# 🛠 Tech Stack

### Backend

* FastAPI
* Uvicorn

### Machine Learning

* TensorFlow
* Keras

### Image Processing

* OpenCV
* Pillow (PIL)

### Numerical Computing

* NumPy

### Visualization

* Matplotlib

---

# Installation

## Prerequisites

* Python 3.9 or newer

---

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cat-dog-classifier.git

cd cat-dog-classifier
```

---

## 2. Create and Activate a Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv cnn_env

.\cnn_env\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv cnn_env

source cnn_env/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Model Setup

Since trained model files (`*.keras`) are excluded from the repository, choose one of the following options.

---

## Option A — Train the Model Yourself

### 1. Download the dataset

Download the **Dog and Cat Classification Dataset** from:

https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

---

### 2. Create the dataset structure

```text
data/
├── Cat/
└── Dog/
```

Place the corresponding images inside each folder.

---

### 3. Clean the dataset

```bash
python data-cleaning.py
```

---

### 4. Train the CNN

```bash
python train.py
```

After training finishes, the model will automatically be saved as:

```text
cat_dog_model.keras
```

---

## Option B — Use a Pre-trained Model

Download the trained weights from:

https://drive.google.com/drive/folders/1ubemvkSErh0_5ceMB1NoaUtB8c9wWhyr?usp=sharing

Move the downloaded

```text
cat_dog_model.keras
```

file into the project's root directory (next to `main.py`).

---

# Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

---

# Interactive API Documentation

FastAPI automatically generates interactive Swagger documentation.

Open:

```text
http://localhost:8000/docs
```

Then:

1. Expand **POST /predict**
2. Click **Try it out**
3. Upload a cat or dog image
4. Click **Execute**
5. View the prediction response

---

# API Endpoint

| Method | Endpoint   | Description                                         |
| ------ | ---------- | --------------------------------------------------- |
| POST   | `/predict` | Predict whether an uploaded image is a cat or a dog |

---

# Example Response

```json
{
    "label": "CAT 🐱",
    "confidence": 94.25
}
```

# License

This project is distributed under the **MIT License**.
