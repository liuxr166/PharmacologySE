# -*- coding: utf-8 -*-
"""
Main entry point for the PharmacologySE framework.

This module orchestrates the data loading, preprocessing, model training,
and result generation for predicting drug-drug interactions.
"""
import os
import csv
import time
import torch
import sqlite3
import numpy as np
import pandas as pd

from .data_sampling import prepare
from .load_config import load_config
from .cross_validation import cross_val

# Load configuration parameters
params, device, results_dir = load_config()
seed = params['seed']
n_attention_heads = params['Att_n_heads']
dropout_rate = params['drop_out_rating']
batch_size = params['batch_size']
learning_rate = params['learn_rating']
num_epochs = params['epo_num']
cross_validation_folds = params['cross_ver_tim']
pair_num = params['pair_num']
weight_decay = params['weight_decay_rate']
feature_list = params['feature_list']


def save_results(directory, result_type, results):
    """
    Save results to a CSV file.
    
    Args:
        directory (str): Directory to save results
        result_type (str): Type of result (e.g., "allFolds", "eachFold")
        results (list): Results to save
        
    Returns:
        int: 0 on success
    """
    file_path = os.path.join(directory, f"{result_type}.csv")
    
    with open(file_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in results:
            writer.writerow(row)
    
    print(f"Results saved to {file_path}")
    return 0


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device.type == 'cuda' or device.type == 'npu':
        torch.cuda.manual_seed_all(seed)
    
    print("Starting PharmacologySE framework...")
    print(f"Using device: {device}")
    print(f"Selected features: {feature_list}")
    
    # Load data from database
    conn = sqlite3.connect('./data/event.db')
    df_drug = pd.read_sql('select * from drug;', conn)  # Contains drug information
    extraction = pd.read_sql('select * from extraction;', conn)  # Contains interaction events
    conn.close()
    
    print(f"Loaded {len(df_drug)} drugs and {len(extraction)} interaction events")
    
    # Extract interaction components
    mechanisms = extraction['mechanism'].tolist()
    actions = extraction['action'].tolist()
    drugA_names = extraction['drugA'].tolist()
    drugB_names = extraction['drugB'].tolist()
    
    # Prepare training data
    print("Preparing training data...")
    features, labels, num_event_types = prepare(
        df_drug, feature_list, mechanisms, actions, drugA_names, drugB_names
    )
    
    print(f"Data preparation completed!")
    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of event types: {num_event_types}")
    
    # Shuffle data while maintaining feature-label correspondence
    print("Shuffling data...")
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]
    
    # Perform cross-validation
    print("Starting cross-validation...")
    start_time = time.time()
    
    all_fold_results, each_fold_results = cross_val(features, labels, num_event_types)
    
    elapsed_time = (time.time() - start_time) / 3600
    print(f"Cross-validation completed in {elapsed_time:.2f} hours")
    
    # Save results
    print("Saving results...")
    save_results(results_dir, "allFolds", all_fold_results)
    save_results(results_dir, "eachFold", each_fold_results)
    
    print("PharmacologySE framework execution completed!")
