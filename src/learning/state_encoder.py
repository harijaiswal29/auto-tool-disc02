"""Supervised State Encoder for dimensionality reduction.

This module implements a neural network-based encoder that reduces
the 476-dimensional state space to a lower-dimensional representation
using supervised learning with contrastive loss.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import json

from utils.logger import get_logger

logger = get_logger(__name__)


class StateEncoderDataset(Dataset):
    """Dataset for training the supervised state encoder."""
    
    def __init__(self, states: np.ndarray, labels: np.ndarray):
        """Initialize dataset.
        
        Args:
            states: Array of state vectors (N, 476)
            labels: Array of labels (N,)
        """
        self.states = torch.FloatTensor(states)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]


class SupervisedStateEncoder(nn.Module):
    """Neural network encoder for state dimensionality reduction."""
    
    def __init__(self, input_dim: int = 476, hidden_dims: List[int] = None,
                 latent_dim: int = 50, dropout_rate: float = 0.3,
                 use_batch_norm: bool = True):
        """Initialize the encoder.
        
        Args:
            input_dim: Input state dimension
            hidden_dims: List of hidden layer dimensions
            latent_dim: Output encoded dimension
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(SupervisedStateEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if i < len(hidden_dims) - 1:  # Don't add dropout before last hidden
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final projection to latent space
        layers.append(nn.Dropout(dropout_rate * 0.5))  # Less dropout before output
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Projection head for contrastive learning (removed during inference)
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor, use_projection: bool = False) -> torch.Tensor:
        """Encode states to latent representation.
        
        Args:
            x: Input state tensor
            use_projection: Whether to apply projection head (for training)
            
        Returns:
            Encoded state tensor
        """
        encoded = self.encoder(x)
        
        if use_projection:
            encoded = self.projection_head(encoded)
        
        return encoded
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.
        
        Args:
            x: Input state tensor
            
        Returns:
            Encoded state tensor
        """
        return self.encode(x, use_projection=False)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for supervised encoder training."""
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        """Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for similarity scaling
            margin: Margin for triplet loss
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            embeddings: Encoded state embeddings (batch_size, latent_dim)
            labels: Binary labels (batch_size,)
            
        Returns:
            Contrastive loss value
        """
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive and negative pairs
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float()
        negative_mask = 1 - positive_mask
        
        # Remove diagonal
        mask_diagonal = torch.eye(batch_size, device=embeddings.device).bool()
        positive_mask.masked_fill_(mask_diagonal, 0)
        negative_mask.masked_fill_(mask_diagonal, 0)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Sum of positive similarities
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        
        # Sum of all similarities (excluding self)
        all_sim = (exp_sim * (positive_mask + negative_mask)).sum(dim=1)
        
        # Avoid division by zero
        loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)
        
        return loss.mean()


class TripletLoss(nn.Module):
    """Triplet loss for supervised encoder training."""
    
    def __init__(self, margin: float = 1.0):
        """Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same class as anchor)
            negative: Negative embeddings (different class from anchor)
            
        Returns:
            Triplet loss value
        """
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(distance_positive - distance_negative + self.margin)
        
        return loss.mean()


class EncoderTrainer:
    """Trainer for the supervised state encoder."""
    
    def __init__(self, encoder: SupervisedStateEncoder, 
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        """Initialize trainer.
        
        Args:
            encoder: State encoder model
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')
        """
        self.encoder = encoder.to(device)
        self.device = device
        self.optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        self.contrastive_loss = ContrastiveLoss()
        self.triplet_loss = TripletLoss()
        self.best_loss = float('inf')
        self.training_history = []
    
    def train_epoch(self, dataloader: DataLoader, 
                   use_contrastive: bool = True) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            use_contrastive: Whether to use contrastive loss (vs triplet)
            
        Returns:
            Average loss for the epoch
        """
        self.encoder.train()
        total_loss = 0
        num_batches = 0
        
        for batch_states, batch_labels in dataloader:
            batch_states = batch_states.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Encode states with projection head for training
            embeddings = self.encoder.encode(batch_states, use_projection=True)
            
            if use_contrastive:
                # Contrastive loss
                loss = self.contrastive_loss(embeddings, batch_labels)
            else:
                # Triplet loss (need to create triplets)
                loss = self._compute_triplet_loss(embeddings, batch_labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def _compute_triplet_loss(self, embeddings: torch.Tensor, 
                             labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss by creating triplets from batch.
        
        Args:
            embeddings: Batch embeddings
            labels: Batch labels
            
        Returns:
            Triplet loss value
        """
        # Simple triplet mining within batch
        positive_mask = labels > 0.5
        negative_mask = ~positive_mask
        
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            # Can't form triplets, return zero loss
            return torch.tensor(0.0, device=self.device)
        
        positive_embeddings = embeddings[positive_mask]
        negative_embeddings = embeddings[negative_mask]
        
        # Use all positives as anchors
        anchors = positive_embeddings
        
        # For each anchor, use another positive as positive sample
        if len(positive_embeddings) > 1:
            positives = torch.roll(positive_embeddings, shifts=1, dims=0)
        else:
            positives = positive_embeddings
        
        # Use first negative for all anchors (simple strategy)
        negatives = negative_embeddings[0].unsqueeze(0).expand_as(anchors)
        
        return self.triplet_loss(anchors, positives, negatives)
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the encoder.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.encoder.eval()
        total_loss = 0
        num_batches = 0
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_states, batch_labels in dataloader:
                batch_states = batch_states.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                embeddings = self.encoder.encode(batch_states, use_projection=True)
                loss = self.contrastive_loss(embeddings, batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store for analysis
                all_embeddings.append(embeddings.cpu())
                all_labels.append(batch_labels.cpu())
        
        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Compute embedding statistics
        all_embeddings = torch.cat(all_embeddings)
        all_labels = torch.cat(all_labels)
        
        # Compute intra-class and inter-class distances
        positive_mask = all_labels > 0.5
        negative_mask = ~positive_mask
        
        metrics = {
            'val_loss': avg_loss,
            'embedding_mean': all_embeddings.mean().item(),
            'embedding_std': all_embeddings.std().item()
        }
        
        if positive_mask.sum() > 1 and negative_mask.sum() > 1:
            pos_embeddings = all_embeddings[positive_mask]
            neg_embeddings = all_embeddings[negative_mask]
            
            # Intra-class distance (should be small)
            pos_dist = F.pdist(pos_embeddings, p=2).mean().item()
            neg_dist = F.pdist(neg_embeddings, p=2).mean().item()
            
            # Inter-class distance (should be large)
            inter_dist = torch.cdist(pos_embeddings, neg_embeddings, p=2).mean().item()
            
            metrics.update({
                'intra_class_dist_pos': pos_dist,
                'intra_class_dist_neg': neg_dist,
                'inter_class_dist': inter_dist,
                'separation_ratio': inter_dist / (pos_dist + neg_dist + 1e-8)
            })
        
        return metrics
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict = None):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            metrics: Optional metrics to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': {
                'input_dim': self.encoder.input_dim,
                'hidden_dims': self.encoder.hidden_dims,
                'latent_dim': self.encoder.latent_dim,
                'dropout_rate': self.encoder.dropout_rate,
                'use_batch_norm': self.encoder.use_batch_norm
            }
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        # Create directory if needed
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('epoch', 0)


def load_encoder(model_path: str, device: str = 'cpu') -> SupervisedStateEncoder:
    """Load a trained encoder from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded encoder model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint.get('config', {})
    encoder = SupervisedStateEncoder(
        input_dim=config.get('input_dim', 476),
        hidden_dims=config.get('hidden_dims', [256, 128]),
        latent_dim=config.get('latent_dim', 50),
        dropout_rate=config.get('dropout_rate', 0.3),
        use_batch_norm=config.get('use_batch_norm', True)
    )
    
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.to(device)
    encoder.eval()
    
    logger.info(f"Loaded encoder from {model_path}")
    return encoder


def encode_states(encoder: SupervisedStateEncoder, states: np.ndarray,
                 batch_size: int = 256, device: str = 'cpu') -> np.ndarray:
    """Encode a batch of states using the trained encoder.
    
    Args:
        encoder: Trained encoder model
        states: Array of states to encode (N, 476)
        batch_size: Batch size for encoding
        device: Device to use
        
    Returns:
        Encoded states (N, latent_dim)
    """
    encoder.eval()
    encoder.to(device)
    
    encoded_states = []
    
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            batch = states[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            
            encoded = encoder(batch_tensor)
            encoded_states.append(encoded.cpu().numpy())
    
    return np.vstack(encoded_states)


if __name__ == "__main__":
    # Test the encoder
    print("Testing SupervisedStateEncoder...")
    
    # Create dummy data
    states = np.random.randn(100, 476).astype(np.float32)
    labels = np.random.randint(0, 2, 100).astype(np.float32)
    
    # Create dataset and dataloader
    dataset = StateEncoderDataset(states, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create encoder and trainer
    encoder = SupervisedStateEncoder()
    trainer = EncoderTrainer(encoder)
    
    # Train for one epoch
    loss = trainer.train_epoch(dataloader)
    print(f"Training loss: {loss:.4f}")
    
    # Validate
    metrics = trainer.validate(dataloader)
    print(f"Validation metrics: {metrics}")
    
    # Test encoding
    encoded = encode_states(encoder, states[:10])
    print(f"Encoded shape: {encoded.shape}")
    
    print("Test completed successfully!")