#Cats vs Dogs - End-to-End MLOps Pipeline

##  Project Overview

This project implements a complete end-to-end MLOps pipeline for binary
image classification (Cats vs Dogs) designed for a pet adoption
platform.

The pipeline covers:

-   Model development
-   Experiment tracking
-   Data & code versioning
-   Containerization
-   CI/CD automation
-   Deployment
-   Monitoring & logging

------------------------------------------------------------------------

##  Dataset

Kaggle Cats vs Dogs Dataset\
Images resized to **224x224 RGB**\
Data split: - 80% Training - 10% Validation - 10% Testing

Data augmentation applied for better generalization.

------------------------------------------------------------------------

##  Tech Stack

-   Python 3.10
-   PyTorch
-   MLflow (Experiment Tracking)
-   Git (Code Versioning)
-   DVC (Data Versioning)
-   FastAPI (Inference API)
-   Docker (Containerization)
-   GitHub Actions (CI/CD)
-   Docker Compose (Deployment)

------------------------------------------------------------------------

##  Project Structure

    cats-dogs-mlops/
    │
    ├── data/
    ├── models/
    ├── src/
    ├── app/
    ├── tests/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── .github/workflows/ci.yml
    └── README.md

------------------------------------------------------------------------

##  How to Run

##  #Install Dependencies

    pip install -r requirements.txt

##  #Train Model

    python src/train.py

##  #Launch MLflow UI

    mlflow ui

##  #Run API Locally

    uvicorn app.main:app --reload

Health Check:

    http://localhost:8000/health

Prediction:

    curl -X POST "http://localhost:8000/predict" -F "file=@cat.jpg"

------------------------------------------------------------------------

##  Docker Deployment

Build Image:

    docker build -t catsdogs-api .

Run Container:

    docker run -p 8000:8000 catsdogs-api

Or use Docker Compose:

    docker-compose up --build

------------------------------------------------------------------------

##  CI/CD Pipeline

On every push to main branch: - Install dependencies - Run unit tests
(pytest) - Build Docker image - (Optional) Push to container registry

------------------------------------------------------------------------

##  Monitoring

-   Request logging enabled
-   Request count tracking
-   Inference latency logging

------------------------------------------------------------------------

##  Demonstration Checklist

1.  Train model
2.  Show MLflow experiment tracking
3.  Run unit tests
4.  Build Docker image
5.  Deploy with Docker Compose
6.  Test prediction endpoint

------------------------------------------------------------------------

##  Submission Contents

-   Source code
-   DVC configuration
-   CI/CD pipeline
-   Docker configuration
-   Trained model artifact
-   Screen recording (≤ 5 minutes)

------------------------------------------------------------------------

##  Author

MLOps Assignment Submission\
Binary Image Classification -- Cats vs Dogs
