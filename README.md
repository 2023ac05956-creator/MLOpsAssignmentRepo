
# Cats vs Dogs Classification - MLOps Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green)
![MLflow](https://img.shields.io/badge/MLflow-2.9.2-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-Enabled-purple)

---

## 🚀 Quick Start

📁 **Main Project Directory**: `cats-dogs-mlops/`

This project demonstrates a complete end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) designed for a pet adoption platform.

---

## 👥 Contributors

| Name | BITS ID |
|------|---------|
| GOBIND SAH | 2024AA05643 |
| VISHAL SINGH | 2024AA05641 |
| YASH VERMA | 2024AA05640 |
| AVISHI GUPTA | 2024AA05055 |
| ASIT SHUKLA  | 2023AC05956 |

---

## 📋 Project Overview

This project implements an end-to-end MLOps pipeline for classifying images of cats and dogs using deep learning. The solution includes:

- Data preprocessing and augmentation
- CNN-based model training
- Experiment tracking with MLflow
- Data versioning with DVC
- FastAPI-based REST API
- Docker containerization
- CI pipeline using GitHub Actions
- Deployment using Docker Compose
- Logging and monitoring

---

## 📊 Dataset

Source: Kaggle - Cats vs Dogs Dataset  

- Binary image classification (Cat / Dog)
- Images resized to 224x224 RGB
- Data split:
  - 80% Training
  - 10% Validation
  - 10% Testing
- Data augmentation applied

---

## 🏗️ Repository Structure

cats-dogs-mlops/
├── data/
├── models/
├── src/
├── app/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

---

## 🚀 Getting Started

### Installation

git clone <your-repo-link>
cd cats-dogs-mlops

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

---

## 🧠 Model Training

python src/train.py

mlflow ui

---

## 🌐 Run API Locally

uvicorn app.main:app --reload

Health: http://localhost:8000/health

Prediction:
curl -X POST "http://localhost:8000/predict" -F "file=@cat.jpg"

---

## 🧪 Testing

pytest

---

## 🐳 Docker Deployment

docker build -t catsdogs-api .
docker run -p 8000:8000 catsdogs-api

Or:

docker-compose up --build

---

## 🔄 CI Pipeline

- Install dependencies
- Run unit tests
- Build Docker image

---

## 🏆 MLOps Best Practices

- Git versioning
- DVC data tracking
- MLflow experiment tracking
- Automated testing
- Docker containerization
- CI/CD automation
- Logging & monitoring

---

## 📦 Submission Checklist

- Source Code
- DVC files
- Dockerfile
- CI YAML
- Trained model
- Demo recording

