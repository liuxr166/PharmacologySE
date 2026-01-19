# -*- coding: utf-8 -*-
"""
Data preprocessing and sampling module for PharmacologySE framework.

This module handles the transformation of raw drug data and interaction events into
structured feature vectors suitable for model training.
"""
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare(df_drug, feature_list, mechanism, action, drugA, drugB):
    """
    Prepare training data from drug information and interaction events.
    
    Args:
        df_drug (pd.DataFrame): Drug information dataframe
        feature_list (list): List of feature names to use (e.g., ['smile', 'target', 'enzyme'])
        mechanism (list): List of interaction mechanisms
        action (list): List of interaction actions
        drugA (list): List of first drugs in each interaction pair
        drugB (list): List of second drugs in each interaction pair
        
    Returns:
        tuple:
            - new_feature (np.ndarray): Feature vectors for drug pairs
            - new_label (np.ndarray): Labels for drug-drug interactions
            - event_num (int): Number of unique interaction types
            
    Raises:
        ValueError: If input data is invalid or inconsistent
        KeyError: If required columns are missing from input data
        MemoryError: If there's insufficient memory to process the data
    """
    try:
        # Validate input data
        _validate_input_data(df_drug, feature_list, mechanism, action, drugA, drugB)
        
        # Transform interaction events to numerical labels
        # Concatenate mechanism and action to form event descriptions
        d_event = [mechanism[i] + " " + action[i] for i in range(len(mechanism))]
        
        # Count occurrences of each interaction event
        event_counts = {}
        for event in d_event:
            event_counts[event] = event_counts.get(event, 0) + 1
        
        event_num = len(event_counts)
        logger.info(f"Found {event_num} unique interaction types")
        
        # Create label dictionary sorted by event frequency
        sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
        event_to_label = {event: idx for idx, (event, _) in enumerate(sorted_events)}
        
        # Initialize feature vector matrix
        num_drugs = len(df_drug)
        feature_matrix = np.zeros((num_drugs, 0), dtype=float)
        
        # Extract and concatenate features for each drug
        for feature_name in feature_list:
            feature_vectors = feature_vector(feature_name, df_drug)
            logger.info(f"{feature_name} feature shape: {feature_vectors.shape}")
            feature_matrix = np.hstack((feature_matrix, feature_vectors))
        
        # Create drug-to-feature mapping
        if 'name' not in df_drug.columns:
            raise KeyError("'name' column missing from drug dataframe")
            
        drug_names = df_drug['name'].tolist()
        drug_feature_map = {drug_names[i]: feature_matrix[i] for i in range(num_drugs)}
        
        # Generate features and labels for drug pairs
        new_feature = []
        new_label = []
        
        for i in range(len(d_event)):
            # Concatenate features for drug pairs
            try:
                drug_a_features = drug_feature_map[drugA[i]]
                drug_b_features = drug_feature_map[drugB[i]]
            except KeyError as e:
                logger.error(f"Drug not found in feature map: {e}")
                raise ValueError(f"Drug '{str(e)}' from input pairs not found in drug database")
                
            pair_features = np.hstack((drug_a_features, drug_b_features))
            
            new_feature.append(pair_features)
            new_label.append(event_to_label[d_event[i]])
        
        logger.info(f"Prepared {len(new_feature)} drug pair samples")
        return np.array(new_feature), np.array(new_label), event_num
        
    except (ValueError, KeyError) as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise
    except MemoryError as e:
        logger.error(f"Memory error during data preparation: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in data preparation: {str(e)}")
        raise


def feature_vector(feature_name, df):
    """
    Generate one-hot encoded feature vectors for a specific feature type.
    
    Args:
        feature_name (str): Name of the feature column in the dataframe
        df (pd.DataFrame): Drug information dataframe
        
    Returns:
        np.ndarray: One-hot encoded feature matrix of shape (num_drugs, num_features)
        
    Raises:
        KeyError: If the specified feature column doesn't exist in the dataframe
        ValueError: If the feature data is invalid
    """
    try:
        # Check if feature column exists
        if feature_name not in df.columns:
            raise KeyError(f"Feature column '{feature_name}' not found in dataframe")
        
        # Collect all unique features across all drugs
        all_features = set()
        drug_features = df[feature_name].tolist()
        
        # Extract all unique feature values
        for i, drug_feature in enumerate(drug_features):
            try:
                drug_feature_str = str(drug_feature)
                if pd.isna(drug_feature_str) or drug_feature_str == 'nan' or drug_feature_str.strip() == '':
                    logger.warning(f"Empty feature value for {feature_name} at drug index {i}")
                    continue
                    
                for feature_value in drug_feature_str.split('|'):
                    if feature_value.strip():
                        all_features.add(feature_value.strip())
            except Exception as e:
                logger.error(f"Error processing feature {feature_name} for drug index {i}: {str(e)}")
                raise ValueError(f"Invalid feature value at index {i} for {feature_name}")
        
        # Convert to sorted list for consistent indexing
        all_features = sorted(list(all_features))
        if not all_features:
            logger.warning(f"No valid feature values found for {feature_name}")
            return np.zeros((len(drug_features), 1), dtype=float)
            
        feature_to_idx = {feature: idx for idx, feature in enumerate(all_features)}
        logger.info(f"Generated {len(all_features)} unique {feature_name} features")
        
        # Create one-hot encoded feature matrix
        num_drugs = len(drug_features)
        num_features = len(all_features)
        feature_matrix = np.zeros((num_drugs, num_features), dtype=float)
        
        # Fill the feature matrix
        for i in range(num_drugs):
            try:
                drug_feature_str = str(df[feature_name].iloc[i])
                if not pd.isna(drug_feature_str) and drug_feature_str != 'nan' and drug_feature_str.strip():
                    for feature_value in drug_feature_str.split('|'):
                        feature_value = feature_value.strip()
                        if feature_value in feature_to_idx:
                            feature_matrix[i, feature_to_idx[feature_value]] = 1
            except Exception as e:
                logger.error(f"Error filling feature matrix at index {i}: {str(e)}")
                raise ValueError(f"Error processing feature value at index {i}")
        
        return feature_matrix
        
    except KeyError as e:
        logger.error(f"Error in feature_vector: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Value error in feature_vector: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in feature_vector: {str(e)}")
        raise


def _validate_input_data(df_drug, feature_list, mechanism, action, drugA, drugB):
    """
    Validate input data for the prepare function.
    
    Args:
        df_drug (pd.DataFrame): Drug information dataframe
        feature_list (list): List of feature names to use
        mechanism (list): List of interaction mechanisms
        action (list): List of interaction actions
        drugA (list): List of first drugs in each interaction pair
        drugB (list): List of second drugs in each interaction pair
        
    Raises:
        ValueError: If input data is invalid or inconsistent
        TypeError: If input data has incorrect types
    """
    # Validate input types
    if not isinstance(df_drug, pd.DataFrame):
        raise TypeError(f"df_drug must be a pandas DataFrame, got {type(df_drug)}")
        
    if not isinstance(feature_list, list):
        raise TypeError(f"feature_list must be a list, got {type(feature_list)}")
        
    for param_name, param in [('mechanism', mechanism), ('action', action), ('drugA', drugA), ('drugB', drugB)]:
        if not isinstance(param, (list, np.ndarray)):
            raise TypeError(f"{param_name} must be a list or numpy array, got {type(param)}")
    
    # Validate input lengths
    if len(mechanism) != len(action):
        raise ValueError(f"Mechanism and action lists must have the same length: {len(mechanism)} vs {len(action)}")
        
    if len(drugA) != len(drugB):
        raise ValueError(f"drugA and drugB lists must have the same length: {len(drugA)} vs {len(drugB)}")
        
    if len(drugA) != len(mechanism):
        raise ValueError(f"Drug pairs and interaction events must have the same length: {len(drugA)} vs {len(mechanism)}")
    
    # Validate feature list content
    if not feature_list:
        raise ValueError("feature_list cannot be empty")
        
    for feature_name in feature_list:
        if not isinstance(feature_name, str):
            raise TypeError(f"Feature names must be strings, got {type(feature_name)}")
        
        if feature_name not in df_drug.columns:
            raise ValueError(f"Feature '{feature_name}' not found in drug dataframe columns")
    
    # Validate dataframe is not empty
    if df_drug.empty:
        raise ValueError("Drug dataframe is empty")
        
    logger.info("All input data validated successfully")
