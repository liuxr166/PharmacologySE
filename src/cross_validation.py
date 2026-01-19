# -*- coding: utf-8 -*-
"""
Cross-validation implementation for PharmacologySE framework.

This module handles the stratified k-src cross-validation process,
including data sampling, model training, and evaluation.
"""
import os
import time
import torch
import random
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from .model import PharmacologySEModel as Model
from .tools import cos_sim
from .evaluate import evaluate
from .train import train_model
from .load_config import load_config
from .bilstm_encoder import data_processing
from sklearn.model_selection import StratifiedKFold

params, device, results_dir = load_config()
seed = params['seed']
pair_num = params['pair_num']
n_attention_heads = params['Att_n_heads']
cross_validation_folds = params['cross_ver_tim']

# Set random seeds for reproducibility
def set_random_seeds(seed):
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif device.type == 'npu':
        import torch_npu
        torch_npu.npu.manual_seed(seed)
        torch_npu.npu.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

# Apply random seeds
set_random_seeds(seed)

def generate_similar_pairs(features, reference_features, pair_num, exclude_self=False):
    """
    Generate similar pairs of features using cosine similarity.
    
    Args:
        features (np.ndarray): Features for which to find similar pairs
        reference_features (np.ndarray): Reference features to compare against
        pair_num (int): Number of similar pairs to generate per sample
        exclude_self (bool): Whether to exclude the sample itself from similar pairs
        
    Returns:
        np.ndarray: Mean features of similar pairs
    """
    print("Generating similar pairs...")
    pair_indices = np.array(cos_sim(features, reference_features, pair_num))
    
    similar_features = []
    for i in range(len(features)):
        # Select similar indices (exclude self if needed)
        start_idx = 1 if exclude_self else 0
        similar_indices = pair_indices[i, start_idx:start_idx + pair_num]
        
        # Get similar samples and calculate mean
        similar_samples = reference_features[similar_indices]
        mean_features = np.mean(similar_samples, axis=0)
        similar_features.append(mean_features.tolist())
    
    return np.array(similar_features)

def augment_data(x_train, train_pair_features, y_train):
    """
    Augment training data by swapping features and applying BiLSTM encoding.
    
    Args:
        x_train (np.ndarray): Original training features
        train_pair_features (np.ndarray): Original training pair features
        y_train (np.ndarray): Original training labels
        
    Returns:
        tuple:
            - augmented_x_train (np.ndarray): Augmented training features
            - augmented_pair_features (np.ndarray): Augmented pair features
            - augmented_y_train (np.ndarray): Augmented labels
    """
    print("Performing data augmentation...")
    drug_dim = len(x_train[0]) // 2
    
    # Augmentation 1: Swap drugA and drugB features
    x_swapped = np.hstack((x_train[:, drug_dim:], x_train[:, :drug_dim]))
    pair_swapped = np.hstack((train_pair_features[:, drug_dim:], train_pair_features[:, :drug_dim]))
    
    # Augmentation 2: BiLSTM encoding - using appropriate input dimension
    max_feature_value = np.max(x_train) + 1
    
    # Apply BiLSTM encoding to both drugs' features
    x_bilstm_left = data_processing(
        x_train[:, drug_dim:], 
        input_dim=max_feature_value,  # Use actual vocabulary size
        embedding_dim=x_train.shape[1] // 4,
        hidden_dim=x_train.shape[1] // 2,
        output_dim=x_train.shape[1] // 4,
        num_layers=2,
        dropout=0.1,
        batch_size=64
    )
    
    x_bilstm_right = data_processing(
        x_train[:, :drug_dim], 
        input_dim=max_feature_value,  # Use actual vocabulary size
        embedding_dim=x_train.shape[1] // 4,
        hidden_dim=x_train.shape[1] // 2,
        output_dim=x_train.shape[1] // 4,
        num_layers=2,
        dropout=0.1,
        batch_size=64
    )
    
    # Combine BiLSTM outputs
    x_bilstm = np.hstack((x_bilstm_left, x_bilstm_right))
    
    # Augment data by combining original and augmented versions
    augmented_x_train = np.vstack((x_train, x_swapped, x_bilstm))
    augmented_pair_features = np.vstack((
        train_pair_features, 
        pair_swapped, 
        train_pair_features  # Use original pair features for BiLSTM augmented data
    ))
    
    # Augment labels to match
    augmented_y_train = np.hstack((y_train, y_train, y_train))
    
    print(f"After augmentation - Training features: {augmented_x_train.shape}")
    print(f"After augmentation - Pair features: {augmented_pair_features.shape}")
    print(f"After augmentation - Labels: {augmented_y_train.shape}")
    
    return augmented_x_train, augmented_pair_features, augmented_y_train

def cross_val(features, labels, num_event_types):
    """
    Perform stratified k-src cross-validation.
    
    Args:
        features (np.ndarray): Input features
        labels (np.ndarray): Target labels
        num_event_types (int): Number of DDI event types
        
    Returns:
        tuple:
            - result_all (list): Aggregated results across all folds
            - result_eve (list): Results for each individual src
    """
    # Validate input data
    if len(features) != len(labels):
        raise ValueError(f"Mismatch between feature length ({len(features)}) and label length ({len(labels)})")
    
    skf = StratifiedKFold(n_splits=cross_validation_folds, shuffle=True, random_state=seed)
    
    y_true = np.array([])
    y_pred = np.array([])
    y_scores = np.zeros((0, num_event_types), dtype=float)
    
    fold_index = 1
    for train_indices, test_indices in skf.split(features, labels):
        print(f"\n=== Fold {fold_index}/{cross_validation_folds} ===")
        start_time = time.time()
        fold_index += 1
        
        # Split data into train and test sets
        x_train, x_test = features[train_indices], features[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        
        print(f"Training samples: {len(y_train)}")
        print(f"Testing samples: {len(y_test)}")
        
        # Generate similar pairs for training and testing data
        train_pair_features = generate_similar_pairs(x_train, x_train, pair_num, exclude_self=True)
        test_pair_features = generate_similar_pairs(x_test, x_train, pair_num, exclude_self=False)
        
        # Augment training data
        x_train, train_pair_features, y_train = augment_data(x_train, train_pair_features, y_train)
        
        # Initialize and train model
        print("Initializing and training model...")
        model = Model(
            input_dim=x_train.shape[1], 
            n_heads=n_attention_heads, 
            event_num=num_event_types
        )
        
        pred_scores = train_model(
            model, 
            x_train, 
            train_pair_features, 
            y_train, 
            x_test, 
            test_pair_features, 
            y_test, 
            num_event_types
        )
        
        # Update results
        pred_types = np.argmax(pred_scores, axis=1)
        y_pred = np.hstack((y_pred, pred_types))
        y_true = np.hstack((y_true, y_test))
        y_scores = np.row_stack((y_scores, pred_scores))
        
        fold_time = time.time() - start_time
        print(f"Fold completed in {fold_time:.2f} seconds. Total samples processed: {len(y_true)}")
    
    # Evaluate results
    print("\n=== Evaluating cross-validation results ===")
    result_all, result_eve = evaluate(y_pred, y_scores, y_true, num_event_types)
    
    return result_all, result_eve
