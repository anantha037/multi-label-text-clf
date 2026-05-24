import argparse
import pandas as pd
import numpy as np
import logging
import re
import pickle
import os
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from src.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_text(text: str) -> str:
    """
    Cleans the input text by lowercasing, stripping extra whitespace, 
    removing URLs, and removing non-ASCII characters.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Strip extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data():
    logging.info("Starting data preprocessing.")
    
    # 1. Load the raw data
    try:
        df = pd.read_csv(Config.RAW_DATA_PATH)
        logging.info(f"Loaded raw data from {Config.RAW_DATA_PATH} with {len(df)} samples.")
    except Exception as e:
        logging.error(f"Failed to load raw data: {e}")
        return
        
    # 2. Clean the abstract column
    logging.info("Cleaning text in 'abstract' column...")
    df['abstract'] = df['abstract'].apply(clean_text)
    
    # 3. Build multi-hot label matrix
    logging.info("Building multi-hot label matrix...")
    # Labels are pipe-separated in the raw file
    df['labels_list'] = df['labels'].apply(lambda x: x.split('|') if isinstance(x, str) and x else [])
    
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(df['labels_list'])
    
    # Save the fitted binarizer
    os.makedirs(os.path.dirname(Config.LABEL_ENCODER_PATH), exist_ok=True)
    with open(Config.LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(mlb, f)
    logging.info(f"Saved MultiLabelBinarizer to {Config.LABEL_ENCODER_PATH}")
    
    label_classes = mlb.classes_
    logging.info(f"Found {len(label_classes)} unique labels.")
    
    # Create a DataFrame for the encoded labels
    encoded_df = pd.DataFrame(encoded_labels, columns=label_classes)
    
    # Combine id, abstract, and encoded labels
    final_df = pd.concat([df[['id', 'abstract']], encoded_df], axis=1)
    
    # 4. Split into train (70%), val (15%), test (15%)
    logging.info("Splitting dataset into train, val, test using MultilabelStratifiedShuffleSplit...")
    
    # First split: train (70%) and temp (30%)
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    X = final_df.drop(columns=label_classes)
    y = final_df[label_classes].values
    
    for train_index, temp_index in msss1.split(X, y):
        train_df = final_df.iloc[train_index]
        temp_df = final_df.iloc[temp_index]
        
    # Second split: temp (30%) into val (15%) and test (15%) -> 50% of temp
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    X_temp = temp_df.drop(columns=label_classes)
    y_temp = temp_df[label_classes].values
    
    for val_index, test_index in msss2.split(X_temp, y_temp):
        val_df = temp_df.iloc[val_index]
        test_df = temp_df.iloc[test_index]
        
    # 5. Save the splits
    os.makedirs("data/processed", exist_ok=True)
    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"
    test_path = "data/processed/test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logging.info(f"Saved splits to data/processed/: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # 6. Print Summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Total samples: {len(final_df)}")
    print(f"Train samples: {len(train_df)} ({(len(train_df)/len(final_df))*100:.1f}%)")
    print(f"Val samples  : {len(val_df)} ({(len(val_df)/len(final_df))*100:.1f}%)")
    print(f"Test samples : {len(test_df)} ({(len(test_df)/len(final_df))*100:.1f}%)")
    print("\nLabel Columns (Multi-hot encoded):")
    print(", ".join(label_classes))
    
    print("\nClass Balance (Positive counts per split):")
    print(f"{'Label':<20} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
    print("-" * 50)
    
    for label in label_classes:
        train_count = train_df[label].sum()
        val_count = val_df[label].sum()
        test_count = test_df[label].sum()
        print(f"{label:<20} | {train_count:<8} | {val_count:<8} | {test_count:<8}")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for multi-label classification.")
    args = parser.parse_args()
    
    preprocess_data()
