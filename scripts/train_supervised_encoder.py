#!/usr/bin/env python3
"""Train the supervised state encoder using labeled data from training runs.

This script trains the encoder to reduce state dimensionality from 476 to 50
dimensions using contrastive learning on labeled episode data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from src.learning.encoder_data_extractor import EncoderDataExtractor
from src.learning.state_encoder import (
    SupervisedStateEncoder, StateEncoderDataset, 
    EncoderTrainer, encode_states
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_training_history(history: List[Dict], save_path: str = None):
    """Plot training history.
    
    Args:
        history: List of training metrics per epoch
        save_path: Path to save plot
    """
    epochs = range(1, len(history) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training and validation loss
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h.get('val_loss', None) for h in history]
    
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    if any(v is not None for v in val_losses):
        axes[0, 0].plot(epochs, [v for v in val_losses if v is not None], 
                       'r-', label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Separation ratio (if available)
    sep_ratios = [h.get('separation_ratio', None) for h in history]
    if any(s is not None for s in sep_ratios):
        valid_epochs = [e for e, s in zip(epochs, sep_ratios) if s is not None]
        valid_ratios = [s for s in sep_ratios if s is not None]
        axes[0, 1].plot(valid_epochs, valid_ratios, 'g-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Separation Ratio')
        axes[0, 1].set_title('Class Separation (Higher is Better)')
        axes[0, 1].grid(True)
    
    # Intra-class distances
    pos_dists = [h.get('intra_class_dist_pos', None) for h in history]
    neg_dists = [h.get('intra_class_dist_neg', None) for h in history]
    
    if any(p is not None for p in pos_dists):
        valid_epochs = [e for e, p in zip(epochs, pos_dists) if p is not None]
        valid_pos = [p for p in pos_dists if p is not None]
        valid_neg = [n for n in neg_dists if n is not None]
        
        axes[1, 0].plot(valid_epochs, valid_pos, 'b-', label='Positive Class')
        axes[1, 0].plot(valid_epochs, valid_neg, 'r-', label='Negative Class')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Distance')
        axes[1, 0].set_title('Intra-class Distances (Lower is Better)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Inter-class distance
    inter_dists = [h.get('inter_class_dist', None) for h in history]
    if any(i is not None for i in inter_dists):
        valid_epochs = [e for e, i in zip(epochs, inter_dists) if i is not None]
        valid_inter = [i for i in inter_dists if i is not None]
        
        axes[1, 1].plot(valid_epochs, valid_inter, 'm-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].set_title('Inter-class Distance (Higher is Better)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved training plot to {save_path}")
    
    plt.show()


def train_encoder(args):
    """Main training function.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Step 1: Extract or load data
    if args.data_path and os.path.exists(args.data_path):
        logger.info(f"Loading data from {args.data_path}")
        data = np.load(args.data_path)
        states = data['states']
        labels = data['labels']
        statistics = data.get('statistics', {})
    else:
        logger.info("Extracting data from training runs...")
        extractor = EncoderDataExtractor(args.results_dir)
        
        result = extractor.extract_from_directory(
            directory=args.directory,
            labeling_strategy=args.labeling_strategy
        )
        
        if not result['data']:
            logger.error("No data extracted!")
            return
        
        # Balance dataset if requested
        if args.balance:
            result = extractor.create_balanced_dataset(args.balance_ratio)
        
        # Save extracted data
        if args.save_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_path = f"data/encoder_training_data_{timestamp}.npz"
            extractor.save_extracted_data(data_path)
            logger.info(f"Saved extracted data to {data_path}")
        
        # Prepare arrays
        states = np.array([d['state'] for d in result['data']], dtype=np.float32)
        labels = np.array([d['label'] for d in result['data']], dtype=np.float32)
        statistics = result['statistics']
    
    logger.info(f"Data shape: states={states.shape}, labels={labels.shape}")
    logger.info(f"Label statistics: mean={labels.mean():.3f}, std={labels.std():.3f}")
    logger.info(f"Positive ratio: {(labels > 0.5).mean():.3f}")
    
    # Step 2: Create dataset and split
    dataset = StateEncoderDataset(states, labels)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Split data: {train_size} train, {val_size} validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Step 3: Create encoder and trainer
    encoder = SupervisedStateEncoder(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        dropout_rate=args.dropout,
        use_batch_norm=args.batch_norm
    )
    
    trainer = EncoderTrainer(
        encoder,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Step 4: Training loop
    logger.info("Starting training...")
    training_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = trainer.train_epoch(
            train_loader, 
            use_contrastive=args.loss_type == 'contrastive'
        )
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        val_loss = val_metrics['val_loss']
        
        # Record history
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **val_metrics
        }
        training_history.append(history_entry)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if 'separation_ratio' in val_metrics:
            logger.info(f"  Separation Ratio: {val_metrics['separation_ratio']:.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            if args.save_dir:
                save_path = os.path.join(args.save_dir, 'best_encoder.pth')
                trainer.save_checkpoint(save_path, epoch + 1, val_metrics)
                logger.info(f"Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            if args.save_dir:
                save_path = os.path.join(args.save_dir, f'encoder_epoch_{epoch+1}.pth')
                trainer.save_checkpoint(save_path, epoch + 1, val_metrics)
    
    # Step 5: Save final model and results
    if args.save_dir:
        # Save final model
        final_path = os.path.join(args.save_dir, 'final_encoder.pth')
        trainer.save_checkpoint(final_path, epoch + 1, val_metrics)
        
        # Save training history
        history_path = os.path.join(args.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save configuration
        config_path = os.path.join(args.save_dir, 'encoder_config.json')
        config = {
            'input_dim': args.input_dim,
            'hidden_dims': args.hidden_dims,
            'latent_dim': args.latent_dim,
            'dropout_rate': args.dropout,
            'use_batch_norm': args.batch_norm,
            'training_args': vars(args),
            'final_metrics': val_metrics,
            'data_statistics': statistics
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved training results to {args.save_dir}")
    
    # Step 6: Plot training history
    if args.plot:
        plot_path = os.path.join(args.save_dir, 'training_plot.png') if args.save_dir else None
        plot_training_history(training_history, plot_path)
    
    # Step 7: Test encoding
    logger.info("\nTesting encoder on sample data...")
    encoder.eval()
    
    # Encode a few samples
    sample_states = states[:10]
    encoded_states = encode_states(encoder, sample_states, device=device)
    
    logger.info(f"Original shape: {sample_states.shape}")
    logger.info(f"Encoded shape: {encoded_states.shape}")
    logger.info(f"Compression ratio: {sample_states.shape[1] / encoded_states.shape[1]:.1f}x")
    
    return encoder, training_history


def main():
    parser = argparse.ArgumentParser(description="Train supervised state encoder")
    
    # Data arguments
    parser.add_argument('--results-dir', type=str,
                       default='tests/dissertation_test_suite/results',
                       help='Directory containing training results')
    parser.add_argument('--directory', type=str, default=None,
                       help='Specific directory to extract from')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to pre-extracted data (NPZ file)')
    parser.add_argument('--labeling-strategy', type=str, default='episode',
                       choices=['episode', 'trajectory', 'reward'],
                       help='Strategy for labeling data')
    parser.add_argument('--balance', action='store_true',
                       help='Balance the dataset')
    parser.add_argument('--balance-ratio', type=float, default=1.0,
                       help='Ratio of negative to positive samples')
    parser.add_argument('--save-data', action='store_true',
                       help='Save extracted data for later use')
    
    # Model arguments
    parser.add_argument('--input-dim', type=int, default=476,
                       help='Input state dimension')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128],
                       help='Hidden layer dimensions')
    parser.add_argument('--latent-dim', type=int, default=50,
                       help='Latent space dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--batch-norm', action='store_true', default=True,
                       help='Use batch normalization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--loss-type', type=str, default='contrastive',
                       choices=['contrastive', 'triplet'],
                       help='Loss function type')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Checkpoint save interval')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loader workers')
    
    # System arguments
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save-dir', type=str, 
                       default='models/supervised_encoder',
                       help='Directory to save models')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training history')
    
    args = parser.parse_args()
    
    # Create save directory
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Train encoder
    encoder, history = train_encoder(args)
    
    logger.info("\nTraining complete!")
    
    if history:
        final_metrics = history[-1]
        logger.info(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()