# multi-label-text-clf

A production-grade Multi-Label Text Classification Pipeline. This project handles the end-to-end process of data ingestion, training a multi-label text classification model, tracking experiments, and serving the model via an API. It is designed to be fully scalable and reproducible.

## Tech Stack

| Component | Technology |
| --- | --- |
| Deep Learning / NLP | PyTorch, Transformers, SetFit |
| Machine Learning | scikit-learn, scikit-multilearn, iterstrat |
| Data Processing | Datasets, Pandas, NumPy |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI, Uvicorn, Pydantic |
| Evaluation | Evaluate |

## Directory Structure

```
multi-label-text-clf/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── serve/
├── models/
├── mlruns/
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

_Instructions for setting up the environment, including installing dependencies._

## Training

_Instructions for running the training pipeline._

## Running the API

_Instructions for starting the FastAPI application._

## Docker

_Instructions for building and running the Docker container._
