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
    Downloads, filters, and formats the arxiv dataset for multilabel classification.
    """
    logging.info("Starting data ingestion process.")
    
    # 1. Attempt to load the dataset
    try:
        logging.info("Attempting to load 'nandakishormpai/arxiv-paper-abstracts' dataset from HuggingFace.")
        dataset = load_dataset("nandakishormpai/arxiv-paper-abstracts", split="train")
    except Exception as e:
        logging.error(f"Failed to load primary dataset: {e}")
        logging.info("Attempting fallback to 'scientific_papers' (arxiv config).")
        try:
            # Fallback dataset as requested, though 'scientific_papers' lacks native category labels in its HuggingFace form.
            dataset = load_dataset("scientific_papers", "arxiv", split="train")
            # If we fall back to scientific_papers, it's primarily designed for summarization 
            # and may lack the 'categories' field required for text classification.
            logging.warning("Fallback 'scientific_papers' loaded. Warning: It may lack standard 'categories' labels. Attempting to parse...")
        except Exception as e2:
            logging.error(f"Graceful fallback failed: Could not load HuggingFace datasets. Error: {e2}")
            sys.exit(1)
            
    # Convert to pandas dataframe for easier manipulation
    df = dataset.to_pandas()
    
    # 2. Subsample to keep RAM usage safe
    if len(df) > max_samples:
        logging.info(f"Dataset has {len(df)} rows. Subsampling to --max_samples={max_samples} for safe RAM usage.")
        df = df.sample(n=max_samples, random_state=42)
    else:
        logging.info(f"Loaded {len(df)} samples.")
        
    # Ensure necessary columns are present ('abstract' and 'categories')
    label_col = 'categories'
    if label_col not in df.columns:
        logging.error(f"Could not find label column '{label_col}' in the dataset. Available columns: {df.columns.tolist()}")
        sys.exit(1)
        
    if 'abstract' not in df.columns:
        logging.error(f"Could not find 'abstract' column. Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # 3. Process the labels
    # The 'nandakishormpai/arxiv-paper-abstracts' categories are space-separated strings
    logging.info("Parsing labels and counting frequencies.")
    df['parsed_labels'] = df[label_col].apply(lambda x: str(x).split() if isinstance(x, str) else x)
    
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
    parser = argparse.ArgumentParser(description="Ingest and preprocess ArXiv dataset for multi-label classification.")
    parser.add_argument("--max_labels", type=int, default=10, help="Maximum number of top frequent labels to keep.")
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum number of samples to process from HF.")
    
    args = parser.parse_args()
    ingest_data(args.max_labels, args.max_samples)
