import argparse
import pandas as pd
import logging
from collections import Counter
import sys
from datasets import load_dataset
from src.config import Config
import os

# Set up logging to output to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ingest_data(max_labels: int, max_samples: int):
    """
    Downloads, filters, and formats the dataset for multilabel classification.
    """
    logging.info("Starting data ingestion process.")
    
    # 1. Attempt to load the primary dataset
    try:
        logging.info("Attempting to load 'go_emotions' (simplified) dataset from HuggingFace.")
        dataset = load_dataset("go_emotions", "simplified", split="train")
        label_names = dataset.features["labels"].feature.names
        
        df = dataset.to_pandas()
        
        # Map text to abstract (to maintain original structure)
        df = df.rename(columns={'text': 'abstract'})
        
        # Map integer labels to string names
        df['parsed_labels'] = df['labels'].apply(lambda x: [label_names[i] for i in x])
        
    except Exception as e:
        logging.error(f"Failed to load primary dataset: {e}")
        logging.info("Attempting fallback to 'sem_eval_2018_task_1' (subtask5.english).")
        try:
            # Fallback dataset
            dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english", split="train")
            df = dataset.to_pandas()
            
            # Map Tweet to abstract
            df = df.rename(columns={'Tweet': 'abstract'})
            
            emotion_cols = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
            
            def extract_active_emotions(row):
                return [col for col in emotion_cols if row[col] == 1]
                
            df['parsed_labels'] = df.apply(extract_active_emotions, axis=1)
            
        except Exception as e2:
            logging.error(f"Graceful fallback failed: Could not load HuggingFace datasets. Error: {e2}")
            sys.exit(1)
            
    # 2. Subsample to keep RAM usage safe
    if len(df) > max_samples:
        logging.info(f"Dataset has {len(df)} rows. Subsampling to --max_samples={max_samples} for safe RAM usage.")
        df = df.sample(n=max_samples, random_state=42)
    else:
        logging.info(f"Loaded {len(df)} samples.")
        
    # Ensure necessary columns are present
    if 'abstract' not in df.columns:
        logging.error(f"Could not find 'abstract' column. Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # 3. Process the labels
    logging.info("Counting label frequencies.")
    
    # Count all label frequencies across the entire dataset
    all_labels = [label for sublist in df['parsed_labels'] for label in sublist]
    label_counts = Counter(all_labels)
    
    # Identify the top N labels
    top_labels = [label for label, count in label_counts.most_common(max_labels)]
    logging.info(f"Top {max_labels} labels identified: {top_labels}")
    
    # 4. Filter the dataset based on top labels
    def filter_top_labels(labels_list):
        # Keep only the labels that are in our top N list
        return [label for label in labels_list if label in top_labels]
        
    df['filtered_labels'] = df['parsed_labels'].apply(filter_top_labels)
    
    # Drop rows that do not have any of the top N labels
    initial_len = len(df)
    df = df[df['filtered_labels'].map(len) > 0].copy()
    logging.info(f"Dropped {initial_len - len(df)} samples that did not belong to any of the top {max_labels} labels.")
    
    # 5. Format to a pipe-separated string as required
    df['labels'] = df['filtered_labels'].apply(lambda x: "|".join(x))
    
    # Ensure 'id' column exists, otherwise generate one
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
        
    # Select only the requested columns for the raw output
    final_df = df[['id', 'abstract', 'labels']]
    
    # 6. Save to raw data path
    os.makedirs(os.path.dirname(Config.RAW_DATA_PATH), exist_ok=True)
    final_df.to_csv(Config.RAW_DATA_PATH, index=False)
    logging.info(f"Successfully saved {len(final_df)} samples to {Config.RAW_DATA_PATH}")
    
    # 7. Print Output Summary
    print("\n" + "="*50)
    print("DATA INGESTION SUMMARY")
    print("="*50)
    print(f"Total samples kept: {len(final_df)}")
    print(f"Unique labels kept: {len(top_labels)}")
    print("\nLabel Distribution in final dataset:")
    
    # Calculate label distribution for the printed summary
    final_all_labels = [label for sublist in final_df['labels'].str.split('|') for label in sublist]
    final_label_counts = Counter(final_all_labels)
    for label, count in final_label_counts.most_common():
        print(f"  - {label}: {count}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and preprocess dataset for multi-label classification.")
    parser.add_argument("--max_labels", type=int, default=10, help="Maximum number of top frequent labels to keep.")
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum number of samples to process from HF.")
    
    args = parser.parse_args()
    ingest_data(args.max_labels, args.max_samples)
