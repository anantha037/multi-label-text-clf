import os
import pickle
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from src.config import Config

class ModelLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.mlb = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

    def load(self):
        """Loads the model, tokenizer, and label binarizer from disk."""
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            raise RuntimeError(f"Model directory not found at {Config.MODEL_SAVE_PATH}. Please train the model first.")
            
        if not os.path.exists(Config.LABEL_ENCODER_PATH):
            raise RuntimeError(f"Label encoder not found at {Config.LABEL_ENCODER_PATH}. Please run preprocessing first.")

        # Load Tokenizer and Model
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(Config.MODEL_SAVE_PATH)
        self.model = DistilBertForSequenceClassification.from_pretrained(Config.MODEL_SAVE_PATH)
        self.model.to(self.device)
        self.model.eval()

        # Load MultiLabelBinarizer
        with open(Config.LABEL_ENCODER_PATH, 'rb') as f:
            self.mlb = pickle.load(f)
            
        self.is_loaded = True

    def predict(self, text: str) -> dict:
        """Runs inference on a single text string."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=Config.MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        # Threshold probabilities at 0.5 for binary label prediction
        preds_binary = (probs > 0.5).astype(int)
        
        # Inverse transform to get string labels
        predicted_labels = self.mlb.inverse_transform(preds_binary.reshape(1, -1))[0]
        
        # Build dictionary of scores for all labels
        label_names = self.mlb.classes_
        scores = {label: float(prob) for label, prob in zip(label_names, probs)}

        return {
            "labels": list(predicted_labels),
            "scores": scores
        }
