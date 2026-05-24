# Multi-Label Text Classifier

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.12.2-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=docker&logoColor=white)

## Overview

This project is a production-grade multi-label text classification pipeline that identifies multiple applicable categories for a given text. It provides an end-to-end blueprint covering data ingestion from GitHub, multi-hot encoded preprocessing, scalable cloud training with DistilBERT via Google Colab, experiment tracking with MLflow, and robust API serving using FastAPI and Docker.

## Architecture Diagram

```text
[ Raw Data Source ]
        │
        ▼
   ( Ingestion ) src/ingest.py
        │
        ▼
[ data/raw/arxiv_raw.csv ]
        │
        ▼
 ( Preprocessing ) src/preprocess.py
        │
        ├──► [ data/processed/label_encoder.pkl ]
        └──► [ data/processed/{train,val,test}.csv ]
                   │
                   ▼
      ( Colab Training ) notebooks/train_colab.ipynb  ──► [ MLflow Tracking ]
                   │
                   ▼
   [ models/distilbert_multilabel/ ]
                   │
                   ▼
         ( FastAPI Serving ) src/serve/main.py
                   │
                   ▼
           [ Docker Container ]
```

## Tech Stack

| Component | Technology |
| --- | --- |
| Deep Learning / NLP | PyTorch, Transformers |
| Data Processing | Pandas, Scikit-learn, iterstrat |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI, Uvicorn, Pydantic |
| Evaluation | Evaluate, Scikit-learn |
| Containerization | Docker, Docker Compose |

## Project Structure

```text
multi-label-text-clf/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── README.md
├── requirements.txt
├── data/
│   ├── processed/
│   │   ├── label_encoder.pkl
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── raw/
│       └── arxiv_raw.csv
├── mlruns/
├── models/
│   └── distilbert_multilabel/
├── notebooks/
│   └── train_colab.ipynb
├── scripts/
│   └── test_api.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── ingest.py
    ├── preprocess.py
    └── serve/
        ├── __init__.py
        ├── main.py
        └── model.py
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd multi-label-text-clf
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

1. **Ingest Data**
   Download the datasets and extract the top labels:
   ```bash
   make ingest
   ```
2. **Preprocess Data**
   Clean the data, generate multi-hot encoded labels, and split the dataset:
   ```bash
   make preprocess
   ```
3. **Train the Model (Google Colab)**
   Upload `data/processed/*.csv` to Google Colab and run `notebooks/train_colab.ipynb`.
4. **Download Model Artifacts**
   Once training completes, download the trained model directory from Google Drive into `models/distilbert_multilabel/`.
5. **Start the API**
   Run the FastAPI server locally:
   ```bash
   make serve
   ```
6. **Test the API**
   Ensure the server is running, then execute the test script:
   ```bash
   make test-api
   ```

## Docker

To containerize the application and run it without requiring a local Python environment:

1. **Build the image**:
   ```bash
   make docker-build
   # or
   docker build -t multilabel-clf .
   ```

2. **Run the container**:
   ```bash
   make docker-up
   # or
   docker-compose up
   ```

## MLflow UI

If you train locally or download your `mlruns` directory from Colab, you can visualize the experiment metrics with MLflow:

```bash
make mlflow-ui
# or
mlflow ui --backend-store-uri mlruns
```

## API Reference

| Method | Path | Description | Example Response |
| --- | --- | --- | --- |
| `GET` | `/health` | Health check and model load status | `{"status": "ok", "model_loaded": true}` |
| `GET` | `/labels` | Returns all supported category labels | `["anger", "joy", "sadness", ...]` |
| `POST` | `/predict`| Predicts labels for the input text | `{"text": "...", "labels": ["joy"], "scores": {"joy": 0.98, ...}}` |

## Results

| Metric | Value |
| --- | --- |
| Test F1 Micro | TBD |
| Test F1 Macro | TBD |
| Hamming Loss | TBD |

## License

MIT
