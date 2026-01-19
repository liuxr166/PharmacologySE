# -*- coding: utf-8 -*-
"""
Model training pipeline for PharmacologySE framework.

This module handles the training process, including dataset preparation,
model optimization, and evaluation.
"""
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .load_config import load_config

params, device, file_path = load_config()
num_epochs = params['epo_num']
batch_size = params['batch_size']
learning_rate = params['learn_rating']
weight_decay = params['weight_decay_rate']


class DDIDataset(Dataset):
    """
    Dataset class for Drug-Drug Interaction data.
    
    Args:
        features (np.ndarray): Primary feature array for drug pairs
        pair_features (np.ndarray): Secondary feature array for drug pairs
        labels (np.ndarray): Labels for drug-drug interactions
    """
    def __init__(self, features, pair_features, labels):
        self.features = torch.from_numpy(np.array(features))
        self.pair_features = torch.from_numpy(np.array(pair_features))
        self.labels = torch.from_numpy(np.array(labels))
        self.num_samples = len(labels)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index (int): Sample index
            
        Returns:
            tuple: Features, pair features, and label for the sample
        """
        return self.features[index], self.pair_features[index], self.labels[index]

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return self.num_samples


def train_model(model, train_features, train_pair_features, train_labels, test_features, test_pair_features, test_labels, num_event_types):
    """
    Train and evaluate the PharmacologySE model.
    
    Args:
        model (torch.nn.Module): PharmacologySE model instance
        train_features (np.ndarray): Training primary features
        train_pair_features (np.ndarray): Training secondary features
        train_labels (np.ndarray): Training labels
        test_features (np.ndarray): Testing primary features
        test_pair_features (np.ndarray): Testing secondary features
        test_labels (np.ndarray): Testing labels
        num_event_types (int): Number of DDI event types
        
    Returns:
        np.ndarray: Prediction scores for the test set
    """
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Move model to device and enable data parallelism
    model = nn.DataParallel(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Create datasets and data loaders
    train_dataset = DDIDataset(train_features, train_pair_features, train_labels)
    test_dataset = DDIDataset(test_features, test_pair_features, test_labels)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        
        for batch_idx, (features, pair_features, labels) in enumerate(train_loader):
            # Move data to device
            features = features.to(device)
            pair_features = pair_features.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features.float(), pair_features.float())
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (features, pair_features, labels) in enumerate(test_loader):
                # Move data to device
                features = features.to(device)
                pair_features = pair_features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(features.float(), pair_features.float())
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = running_loss / len(train_dataset)
        avg_test_loss = test_loss / len(test_dataset)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.6f} Test Loss: {avg_test_loss:.6f}')
    
    # Generate predictions on test set
    model.eval()
    predictions = np.zeros((0, num_event_types), dtype=float)
    
    with torch.no_grad():
        for batch_idx, (features, pair_features, labels) in enumerate(test_loader):
            # Move data to device
            features = features.to(device)
            pair_features = pair_features.to(device)
            
            # Forward pass
            outputs = model(features.float(), pair_features.float())
            
            # Apply softmax and convert to numpy
            batch_predictions = F.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.vstack((predictions, batch_predictions))
    
    return predictions
