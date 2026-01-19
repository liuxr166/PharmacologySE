# -*- coding: utf-8 -*-
"""
BiLSTM Encoder module for PharmacologySE framework.

This module provides a BiLSTM encoder for processing sequential drug features,
converting them into fixed-size vector representations.
"""
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader


try:
    import torch_npu
except ImportError:
    torch_npu = None


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for sequential data.
    
    Args:
        input_dim (int): Vocabulary size for embedding layer
        embedding_dim (int): Dimension of embedding vectors
        hidden_dim (int): Dimension of hidden state in LSTM
        output_dim (int): Dimension of outputs representation
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate for regularization
    """
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(BiLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.bilstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional LSTM
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        """
        Forward pass of the BiLSTM encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        outputs, (hidden, cell) = self.bilstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.layer_norm(hidden)
        out = self.fc(hidden)
        
        return out


def pad_sequences(sequences, max_len, padding_value=0):
    """
    Pad sequences to the same length.
    
    Args:
        sequences (list): List of sequences (each sequence is a list of integers)
        max_len (int): Maximum length for padded sequences
        padding_value (int, optional): Value to use for padding. Defaults to 0.
        
    Returns:
        list: List of padded sequences
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = seq + [padding_value] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]  # Truncate if longer than max_len
        padded_sequences.append(padded_seq)
    return padded_sequences


def data_processing(input_matrix, input_dim, embedding_dim, hidden_dim, output_dim, 
                   num_layers, dropout, batch_size):
    """
    Process input sequences using BiLSTM encoder.
    
    Args:
        input_matrix (np.ndarray): Input matrix of shape (num_samples, variable_length)
        input_dim (int): Vocabulary size for embedding layer
        embedding_dim (int): Dimension of embedding vectors
        hidden_dim (int): Dimension of hidden state in LSTM
        output_dim (int): Dimension of outputs representation
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate for regularization
        batch_size (int): Batch size for processing
        
    Returns:
        np.ndarray: Encoded representations of shape (num_samples, output_dim)
    """
    output_matrix = []
    
    # Determine maximum sequence length
    max_length = max(len(row) for row in input_matrix)
    
    # Pad sequences to the same length
    padded_matrix = pad_sequences(input_matrix, max_length)
    
    # Convert to PyTorch tensor
    padded_matrix = np.array(padded_matrix, dtype=np.long)
    input_tensor = torch.tensor(padded_matrix)
    
    # Create data loader for batch processing
    dataloader = DataLoader(
        input_tensor, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Determine device (NPU, CUDA, or CPU)
    device = torch.device(
        "npu" if (torch_npu and torch_npu.npu.is_available()) else 
        "cuda" if torch.cuda.is_available() else 
        "cpu"
    )
    
    # Initialize model
    model = BiLSTMEncoder(
        input_dim, 
        embedding_dim, 
        hidden_dim, 
        output_dim, 
        num_layers, 
        dropout
    ).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Set random seeds for reproducibility
    np.random.seed(4)
    torch.manual_seed(4)
    
    # Forward pass without gradient computation
    with torch.no_grad():
        for input_batch in dataloader:
            input_batch = input_batch.to(device)
            output_batch = model(input_batch)
            output_matrix.append(output_batch.cpu().numpy())
    
    # Combine all batches into a single array
    return np.vstack(output_matrix)


if __name__ == '__main__':
    """Example usage of the BiLSTM encoder."""
    # Sample sequences for testing
    sequences = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    sequences = np.array(sequences)
    
    # Process sequences
    encoded_sequences = data_processing(
        sequences, 
        input_dim=10, 
        embedding_dim=20, 
        hidden_dim=20, 
        output_dim=10, 
        num_layers=2, 
        dropout=0.1,
        batch_size=2
    )
    
    print("Encoded sequences shape:", encoded_sequences.shape)
    print("Encoded sequences:", encoded_sequences)
