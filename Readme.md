# Heart Disease MLOps Pipeline
### BITS Pilani WILP | MTech AI & ML | MLOps (S2-25_AMLCSZG523)

---

## Project Overview

An end-to-end MLOps pipeline that:
1. Downloads and preprocesses the **Heart Disease UCI dataset**
2. Trains and tracks two ML models with **MLflow**
3. Serves predictions via a **FastAPI** REST API
4. Containerises the app with **Docker**
5. Deploys to **Kubernetes** (Docker Desktop)
6. Monitors via **Prometheus + Grafana**
7. Automates lint → test → train → build via **GitHub Actions**

---

## Repository Structure

heart-disease-mlops/
├── data/
│ ├── download_data.py # Dataset download script
│ ├── heart_raw.csv # Raw UCI data (auto-generated)
│ └── heart_cleaned.csv # Cleaned dataset (auto-generated)
├── notebooks/
│ ├── 01_EDA.ipynb # Exploratory Data Analysis
│ └── 02_Training.ipynb # Interactive training notebook
├── src/
│ ├── preprocess.py # Preprocessing pipeline
│ ├── train.py # Model training + MLflow
│ └── app.py # FastAPI serving app
├── tests/
│ ├── conftest.py # Pytest config + shared fixtures
│ ├── test_preprocess.py # Unit tests — preprocessing
│ └── test_model.py # Unit tests — model + API
├── deployment/
│ ├── deployment.yaml # Kubernetes Deployment
│ └── service.yaml # Kubernetes Service
├── monitoring/
│ ├── docker-compose.yml # Prometheus + Grafana stack
│ └── prometheus.yml # Prometheus scrape config
├── screenshots/ # Auto-generated plots + screenshots
├── models/ # Saved model files (auto-generated)
├── mlruns/ # MLflow experiment logs
├── .github/
│ └── workflows/
│ └── ci_cd.yml # GitHub Actions pipeline
├── Dockerfile # Container definition
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md

