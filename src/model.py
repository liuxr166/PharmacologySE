# -*- coding: utf-8 -*-
"""
Model architecture implementation for PharmacologySE framework.

This module contains the implementation of various attention mechanisms and the
main model architecture for predicting drug-drug interactions.
"""
import torch
import numpy as np

from .tools import gelu
from .load_config import load_config

params, device, file_path = load_config()
dropout_rate = params['drop_out_rating']


class MultiHeadSelfAttentionSem(torch.nn.Module):
    """
    Multi-head self-attention layer using dot-product attention.
    
    Args:
        input_dim (int): Input dimension
        n_heads (int): Number of attention heads
        output_dim (int, optional): Output dimension. Defaults to input_dim.
    """
    def __init__(self, input_dim, n_heads, output_dim=None):
        super(MultiHeadSelfAttentionSem, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # Linear projections for query, key, and value
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.output_dim)

    def forward(self, X):
        """
        Forward pass of the multi-head self-attention layer.
        
        Args:
            X (torch.Tensor): Input tensor of shape (sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (sequence_length, output_dim)
        """
        # Linear projection and split into multiple heads
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        # Calculate attention scores using dot-product
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        
        # Apply softmax to get attention weights
        attn_weights = torch.nn.Softmax(dim=-1)(scores)
        
        # Compute context vectors
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear projection
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        
        return output


class MultiHeadCrossAttentionSem(torch.nn.Module):
    """
    Multi-head cross-attention layer using dot-product attention.
    
    Args:
        input_dim (int): Input dimension
        n_heads (int): Number of attention heads
        output_dim (int, optional): Output dimension. Defaults to input_dim.
    """
    def __init__(self, input_dim, n_heads, output_dim=None):
        super(MultiHeadCrossAttentionSem, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # Linear projections for query, key, and value
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.output_dim)

    def forward(self, X, Y):
        """
        Forward pass of the multi-head cross-attention layer.
        
        Args:
            X (torch.Tensor): Query tensor of shape (sequence_length, input_dim)
            Y (torch.Tensor): Key/Value tensor of shape (sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (sequence_length, output_dim)
        """
        # Linear projection and split into multiple heads
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(Y).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(Y).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        
        # Calculate attention scores using dot-product
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        
        # Apply softmax to get attention weights
        attn_weights = torch.nn.Softmax(dim=-1)(scores)
        
        # Compute context vectors
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear projection
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        
        return output


class MultiHeadSelfAttentionNod(torch.nn.Module):
    """
    Multi-head self-attention layer using linear attention.
    
    Args:
        input_dim (int): Input dimension
        n_heads (int): Number of attention heads
        output_dim (int, optional): Output dimension. Defaults to input_dim.
    """
    def __init__(self, input_dim, n_heads, output_dim=None):
        super(MultiHeadSelfAttentionNod, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # Linear projections for query, key, and value
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.linear_w = torch.nn.Linear(self.d_k, self.d_k)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.output_dim)

    def forward(self, X):
        """
        Forward pass of the multi-head self-attention layer with linear attention.
        
        Args:
            X (torch.Tensor): Input tensor of shape (sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (sequence_length, output_dim)
        """
        # Linear projection and split into multiple heads
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        
        # Calculate attention scores using linear attention
        alpha = torch.matmul(Q, self.linear_w(K).transpose(-1, -2))
        
        # Apply softmax to get attention weights
        attn_weights = torch.nn.Softmax(dim=-1)(alpha)
        
        # Compute context vectors
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear projection
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        
        return output


class MultiHeadCrossAttentionNod(torch.nn.Module):
    """
    Multi-head cross-attention layer using linear attention.
    
    Args:
        input_dim (int): Input dimension
        n_heads (int): Number of attention heads
        output_dim (int, optional): Output dimension. Defaults to input_dim.
    """
    def __init__(self, input_dim, n_heads, output_dim=None):
        super(MultiHeadCrossAttentionNod, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # Linear projections for query, key, and value
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.linear_w = torch.nn.Linear(self.d_k, self.d_k)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.output_dim)

    def forward(self, X, Y):
        """
        Forward pass of the multi-head cross-attention layer with linear attention.
        
        Args:
            X (torch.Tensor): Query tensor of shape (sequence_length, input_dim)
            Y (torch.Tensor): Key/Value tensor of shape (sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (sequence_length, output_dim)
        """
        # Linear projection and split into multiple heads
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(Y).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(Y).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        
        # Calculate attention scores using linear attention
        alpha = torch.matmul(Q, self.linear_w(K).transpose(-1, -2))
        
        # Apply softmax to get attention weights
        attn_weights = torch.nn.Softmax(dim=-1)(alpha)
        
        # Compute context vectors
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear projection
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        
        return output


class EncoderLayerSelfSem(torch.nn.Module):
    """
    Encoder layer with self-attention using dot-product attention.
    
    Args:
        input_dim (int): Input dimension
        n_heads (int): Number of attention heads
    """
    def __init__(self, input_dim, n_heads):
        super(EncoderLayerSelfSem, self).__init__()
        self.attn1 = MultiHeadSelfAttentionSem(input_dim, n_heads)
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.attn2 = MultiHeadSelfAttentionSem(input_dim, n_heads)
        self.norm3 = torch.nn.LayerNorm(input_dim)
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.norm4 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        """
        Forward pass of the encoder layer with self-attention.
        
        Args:
            X (torch.Tensor): Input tensor of shape (sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (sequence_length, input_dim)
        """
        # First attention block
        output = self.attn1(X)
        X = self.norm1(output + X)
        
        # First feed-forward block
        output = self.linear1(X)
        X = self.norm2(output + X)
        
        # Second attention block
        output = self.attn2(X)
        X = self.norm3(output + X)
        
        # Second feed-forward block
        output = self.linear2(X)
        X = self.norm4(output + X)
        
        return X


class EncoderLayerCrossSem(torch.nn.Module):
    """
    Encoder layer with cross-attention using dot-product attention.
    
    Args:
        input_dim (int): Input dimension
        n_heads (int): Number of attention heads
    """
    def __init__(self, input_dim, n_heads):
        super(EncoderLayerCrossSem, self).__init__()
        self.attn = MultiHeadCrossAttentionSem(input_dim, n_heads)
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X, Y):
        """
        Forward pass of the encoder layer with cross-attention.
        
        Args:
            X (torch.Tensor): First input tensor of shape (sequence_length, input_dim)
            Y (torch.Tensor): Second input tensor of shape (sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (sequence_length, input_dim)
        """
        # Cross-attention block
        output = self.attn(X, Y)
        Z = self.norm1(output + X + Y)
        
        # Feed-forward block
        output = self.linear1(Z)
        output = self.norm2(output + Z)
        
        return output


class EncoderLayerSelfNod(torch.nn.Module):
    """
    Encoder layer with self-attention using linear attention.
    
    Args:
        input_dim (int): Input dimension
        n_heads (int): Number of attention heads
    """
    def __init__(self, input_dim, n_heads):
        super(EncoderLayerSelfNod, self).__init__()
        self.attn1 = MultiHeadSelfAttentionNod(input_dim, n_heads)
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.attn2 = MultiHeadSelfAttentionNod(input_dim, n_heads)
        self.norm3 = torch.nn.LayerNorm(input_dim)
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.norm4 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        """
        Forward pass of the encoder layer with self-attention using linear attention.
        
        Args:
            X (torch.Tensor): Input tensor of shape (sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (sequence_length, input_dim)
        """
        # First attention block
        output = self.attn1(X)
        X = self.norm1(output + X)
        
        # First feed-forward block
        output = self.linear1(X)
        X = self.norm2(output + X)
        
        # Second attention block
        output = self.attn2(X)
        X = self.norm3(output + X)
        
        # Second feed-forward block
        output = self.linear2(X)
        X = self.norm4(output + X)
        
        return X


class EncoderLayerCrossNod(torch.nn.Module):
    """
    Encoder layer with cross-attention using linear attention.
    
    Args:
        input_dim (int): Input dimension
        n_heads (int): Number of attention heads
    """
    def __init__(self, input_dim, n_heads):
        super(EncoderLayerCrossNod, self).__init__()
        self.attn = MultiHeadCrossAttentionNod(input_dim, n_heads)
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X, Y):
        """
        Forward pass of the encoder layer with cross-attention using linear attention.
        
        Args:
            X (torch.Tensor): First input tensor of shape (sequence_length, input_dim)
            Y (torch.Tensor): Second input tensor of shape (sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (sequence_length, input_dim)
        """
        # Cross-attention block
        output = self.attn(X, Y)
        Z = self.norm1(output + X + Y)
        
        # Feed-forward block
        output = self.linear1(Z)
        output = self.norm2(output + Z)
        
        return output


class SemModel(torch.nn.Module):
    """
    Model using dot-product attention mechanisms for drug pair representation.
    
    Args:
        input_dim (int): Input dimension for each drug
        n_heads (int): Number of attention heads
    """
    def __init__(self, input_dim, n_heads):
        super(SemModel, self).__init__()
        self.self_attention_A = EncoderLayerSelfSem(input_dim, n_heads)
        self.self_attention_B = EncoderLayerSelfSem(input_dim, n_heads)
        self.cross_attention = EncoderLayerCrossSem(input_dim, n_heads)
        self.linear = torch.nn.Linear(input_dim, input_dim)
        self.batch_norm = torch.nn.BatchNorm1d(input_dim)
        self.activation = gelu
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, X, Y):
        """
        Forward pass of the dot-product attention model.
        
        Args:
            X (torch.Tensor): Feature tensor for first drug
            Y (torch.Tensor): Feature tensor for second drug
            
        Returns:
            torch.Tensor: Combined representation of the drug pair
        """
        # Apply self-attention to each drug separately
        X = self.self_attention_A(X)
        Y = self.self_attention_B(Y)
        
        # Apply cross-attention to capture interactions
        output = self.cross_attention(X, Y)
        
        # Apply linear transformation, activation, and dropout
        output = self.linear(output)
        output = self.batch_norm(output)
        output = self.activation(output)
        output = self.dropout(output)
        
        return output


class NodModel(torch.nn.Module):
    """
    Model using linear attention mechanisms for drug pair representation.
    
    Args:
        input_dim (int): Input dimension for each drug
        n_heads (int): Number of attention heads
    """
    def __init__(self, input_dim, n_heads):
        super(NodModel, self).__init__()
        self.self_attention_A = EncoderLayerSelfNod(input_dim, n_heads)
        self.self_attention_B = EncoderLayerSelfNod(input_dim, n_heads)
        self.cross_attention = EncoderLayerCrossNod(input_dim, n_heads)
        self.linear = torch.nn.Linear(input_dim, input_dim)
        self.batch_norm = torch.nn.BatchNorm1d(input_dim)
        self.activation = gelu
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, X, Y):
        """
        Forward pass of the linear attention model.
        
        Args:
            X (torch.Tensor): Feature tensor for first drug
            Y (torch.Tensor): Feature tensor for second drug
            
        Returns:
            torch.Tensor: Combined representation of the drug pair
        """
        # Apply self-attention to each drug separately
        X = self.self_attention_A(X)
        Y = self.self_attention_B(Y)
        
        # Apply cross-attention to capture interactions
        output = self.cross_attention(X, Y)
        
        # Apply linear transformation, activation, and dropout
        output = self.linear(output)
        output = self.batch_norm(output)
        output = self.activation(output)
        output = self.dropout(output)
        
        return output


class PharmacologySEModel(torch.nn.Module):
    """
    Main PharmacologySE model for predicting drug-drug interactions.
    
    This model combines both dot-product and linear attention mechanisms
    to capture complex drug-drug interactions from multiple feature representations.
    
    Args:
        input_dim (int): Total input dimension (for both drugs combined)
        n_heads (int): Number of attention heads
        event_num (int): Number of DDI event types to predict
    """
    def __init__(self, input_dim, n_heads, event_num):
        super(PharmacologySEModel, self).__init__()
        self.drug_dim = input_dim // 2
        
        # Initialize sub-models
        self.sem_model = SemModel(self.drug_dim, n_heads)
        self.nod_model = NodModel(self.drug_dim, n_heads)
        
        # Prediction layers
        self.fc1 = torch.nn.Linear(self.drug_dim * 5, self.drug_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.drug_dim)
        self.fc2 = torch.nn.Linear(self.drug_dim, (self.drug_dim + event_num) // 2)
        self.bn2 = torch.nn.BatchNorm1d((self.drug_dim + event_num) // 2)
        self.fc3 = torch.nn.Linear((self.drug_dim + event_num) // 2, event_num)
        
        # Activation and dropout
        self.activation = gelu
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, X, X_pair):
        """
        Forward pass of the PharmacologySE model.
        
        Args:
            X (torch.Tensor): Primary feature tensor for drug pairs
            X_pair (torch.Tensor): Secondary feature tensor for drug pairs
            
        Returns:
            torch.Tensor: Prediction scores for each DDI event type
        """
        # Split features for each drug
        drugA_features = X[:, :self.drug_dim]
        drugB_features = X[:, self.drug_dim:]
        drugA_pair_features = X_pair[:, :self.drug_dim]
        drugB_pair_features = X_pair[:, self.drug_dim:]
        
        # Generate multiple representations using different attention mechanisms
        sem_out1 = self.sem_model(drugA_features, drugB_features)
        nod_out1 = self.nod_model(drugA_pair_features, drugB_pair_features)
        sem_out2 = self.sem_model(drugA_pair_features, drugB_pair_features)
        nod_out2 = self.nod_model(drugA_features, drugB_features)
        sem_nod_out = sem_out1 + nod_out2
        
        # Concatenate all representations
        combined_features = torch.cat((sem_out1, nod_out1, sem_out2, nod_out2, sem_nod_out), dim=1)
        
        # Apply prediction layers
        output = self.fc1(combined_features)
        output = self.bn1(output)
        output = self.activation(output)
        output = self.dropout(output)
        
        output = self.fc2(output)
        output = self.bn2(output)
        output = self.activation(output)
        output = self.dropout(output)
        
        # Final prediction layer
        output = self.fc3(output)
        
        return output


# For backward compatibility
Model = PharmacologySEModel
