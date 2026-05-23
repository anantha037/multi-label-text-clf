from dataclasses import dataclass

@dataclass
class Config:
    RAW_DATA_PATH = "data/raw/arxiv_raw.csv"
    PROCESSED_DATA_PATH = "data/processed/arxiv_processed.csv"
    MODEL_SAVE_PATH = "models/distilbert_multilabel"
    MLFLOW_TRACKING_URI = "mlruns"
    MAX_SEQ_LENGTH = 256
    LABEL_ENCODER_PATH = "data/processed/label_encoder.pkl"
