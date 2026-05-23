import argparse
import pandas as pd
import logging
from collections import Counter
import sys
import requests
import io
from src.config import Config
import os

# Set up logging to output to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ingest_data(max_labels: int, max_samples: int):
    """
    Downloads, filters, and formats the dataset for multilabel classification.
    """
    logging.info("Starting data ingestion process.")
    
    # 1. Download emotions list
    emotions_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
    try:
        logging.info("Downloading emotions list...")
        response = requests.get(emotions_url)
        response.raise_for_status()
        emotion_names = response.text.strip().split('\n')
    except Exception as e:
        logging.error(f"Failed to download emotions list: {e}")
        sys.exit(1)
        
    # 2. Download dataset splits
    splits = ['train.tsv', 'dev.tsv', 'test.tsv']
    base_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
    
    dfs = []
    for split in splits:
        try:
            logging.info(f"Downloading {split}...")
            url = base_url + split
            response = requests.get(url)
            response.raise_for_status()
            
            # Read TSV into dataframe
            df_split = pd.read_csv(io.StringIO(response.text), sep='\t', header=None, names=['abstract', 'label_ids', 'id'])
            dfs.append(df_split)
        except Exception as e:
            logging.error(f"Failed to download {split}: {e}")
            sys.exit(1)
            
    # Combine all splits
    df = pd.concat(dfs, ignore_index=True)
    
    # Map label indices to string names
    def parse_labels(label_str):
        if pd.isna(label_str):
            return []
        indices = [int(idx) for idx in str(label_str).split(',')]
        return [emotion_names[i] for i in indices]
        
    logging.info("Parsing labels...")
    df['parsed_labels'] = df['label_ids'].apply(parse_labels)
    
    # 3. Subsample to keep RAM usage safe
    if len(df) > max_samples:
        logging.info(f"Dataset has {len(df)} rows. Subsampling to --max_samples={max_samples} for safe RAM usage.")
        df = df.sample(n=max_samples, random_state=42)
    else:
        logging.info(f"Loaded {len(df)} samples.")
        
    # Ensure necessary columns are present
    if 'abstract' not in df.columns:
        logging.error(f"Could not find 'abstract' column. Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # 4. Process the labels
    logging.info("Counting label frequencies.")
    
    # Count all label frequencies across the entire dataset
    all_labels = [label for sublist in df['parsed_labels'] for label in sublist]
    label_counts = Counter(all_labels)
    
    # Identify the top N labels
    top_labels = [label for label, count in label_counts.most_common(max_labels)]
    logging.info(f"Top {max_labels} labels identified: {top_labels}")
    
    # 5. Filter the dataset based on top labels
    def filter_top_labels(labels_list):
        # Keep only the labels that are in our top N list
        return [label for label in labels_list if label in top_labels]
        
    df['filtered_labels'] = df['parsed_labels'].apply(filter_top_labels)
    
    # Drop rows that do not have any of the top N labels
    initial_len = len(df)
    df = df[df['filtered_labels'].map(len) > 0].copy()
    logging.info(f"Dropped {initial_len - len(df)} samples that did not belong to any of the top {max_labels} labels.")
    
    # 6. Format to a pipe-separated string as required
    df['labels'] = df['filtered_labels'].apply(lambda x: "|".join(x))
    
    # Select only the requested columns for the raw output
    final_df = df[['id', 'abstract', 'labels']]
    
    # 7. Save to raw data path
    os.makedirs(os.path.dirname(Config.RAW_DATA_PATH), exist_ok=True)
    final_df.to_csv(Config.RAW_DATA_PATH, index=False)
    logging.info(f"Successfully saved {len(final_df)} samples to {Config.RAW_DATA_PATH}")
    
    # 8. Print Output Summary
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
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum number of samples to process.")
    
    args = parser.parse_args()
    ingest_data(args.max_labels, args.max_samples)
