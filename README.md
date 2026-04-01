# 🧠 Multimodal Image Understanding Pipeline

🚀 End-to-end AI system for multi-label image classification and image captioning using CNNs and transformer-based models, deployed via FastAPI.

## 📌 Overview

This project implements a **multimodal AI pipeline** that processes an input image to generate:

* **Multi-label predictions** using a pretrained CNN (ResNet50)
* **Natural language captions** using a transformer-based model (BLIP)
* **Real-time inference** via a FastAPI endpoint

The system demonstrates how deep learning models can be integrated into a **production-ready pipeline**.


## 🏗️ Architecture

Input Image
     │
     ▼
CNN (ResNet50) ──► Multi-label Predictions
     │
     ▼
BLIP Model ─────► Caption Generation
     │
     ▼
FastAPI ────────► /predict Endpoint


## ⚙️ Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **HuggingFace Transformers**
* **FastAPI**
* **Uvicorn**



## 🚀 Features

* Multi-label image classification (Top-K predictions)
* Transformer-based image caption generation
* REST API for real-time inference
* Fully local execution (no external API dependency)
* Clean and modular pipeline design


## 🛠️ Installation

pip install -r requirements.txt


## ▶️ Running the Application

Run the FastAPI server locally:
uvicorn main:app --reload

Open in browser:

http://127.0.0.1:8000/docs


## 🧪 API Usage

### Endpoint

POST /predict


### Input

* Image file (multipart/form-data)

### Output
{
  "labels": ["laptop", "keyboard"],
  "caption": "a laptop placed on a desk"
}


## 🧠 Methodology

### Multi-label Classification

A pretrained **ResNet50** model is used to extract the top-k predictions, simulating multi-label classification.

### Image Captioning

The **BLIP (Bootstrapped Language Image Pretraining)** model is used to generate descriptive captions from images.

### Deployment

The pipeline is deployed using **FastAPI**, enabling real-time inference through a RESTful API.


## ⚠️ Design Choices

* Used pretrained models to reduce training time and computational cost
* Focused on pipeline integration and deployment
* Ensured reproducibility without relying on external APIs


## 🔮 Future Improvements

* Fine-tuning on COCO multi-label dataset
* Improved label filtering using confidence thresholds
* Integration with LangGraph-based workflows
* Frontend UI for better interaction
* Dockerization for scalable deployment


## 📸 Demo

Test the API using:

http://127.0.0.1:8000/docs


## 📎 Conclusion

This project demonstrates:

* Integration of CNN and transformer-based models
* Multimodal AI system design
* Deployment of machine learning models using FastAPI


## 👤 Author

**Saadiya Farheen**
