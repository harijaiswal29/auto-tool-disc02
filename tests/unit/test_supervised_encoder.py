"""Unit tests for the supervised state encoder module."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path

from src.learning.state_encoder import (
    SupervisedStateEncoder, StateEncoderDataset, 
    EncoderTrainer, ContrastiveLoss, TripletLoss,
    load_encoder, encode_states
)
from src.learning.encoder_data_extractor import EncoderDataExtractor


class TestSupervisedStateEncoder(unittest.TestCase):
    """Test the SupervisedStateEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 476
        self.latent_dim = 50
        self.batch_size = 32
        
        # Create dummy data
        self.states = np.random.randn(100, self.input_dim).astype(np.float32)
        self.labels = np.random.randint(0, 2, 100).astype(np.float32)
        
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        encoder = SupervisedStateEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim
        )
        
        self.assertEqual(encoder.input_dim, self.input_dim)
        self.assertEqual(encoder.latent_dim, self.latent_dim)
        self.assertIsNotNone(encoder.encoder)
        self.assertIsNotNone(encoder.projection_head)
        
    def test_encoder_forward_pass(self):
        """Test encoder forward pass."""
        encoder = SupervisedStateEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim
        )
        
        # Test single sample
        sample = torch.FloatTensor(self.states[0:1])
        output = encoder(sample)
        
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], self.latent_dim)
        
        # Test batch
        batch = torch.FloatTensor(self.states[:self.batch_size])
        output = encoder(batch)
        
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.latent_dim)
        
    def test_encoder_with_projection(self):
        """Test encoder with projection head."""
        encoder = SupervisedStateEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim
        )
        
        sample = torch.FloatTensor(self.states[0:1])
        
        # Without projection
        output_no_proj = encoder.encode(sample, use_projection=False)
        
        # With projection
        output_with_proj = encoder.encode(sample, use_projection=True)
        
        # Both should have same shape
        self.assertEqual(output_no_proj.shape, output_with_proj.shape)
        
        # But different values (due to projection head)
        self.assertFalse(torch.allclose(output_no_proj, output_with_proj))
        
    def test_different_architectures(self):
        """Test encoder with different architectures."""
        architectures = [
            [256, 128],
            [512, 256, 128],
            [128]
        ]
        
        for hidden_dims in architectures:
            encoder = SupervisedStateEncoder(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                latent_dim=self.latent_dim
            )
            
            sample = torch.FloatTensor(self.states[0:1])
            output = encoder(sample)
            
            self.assertEqual(output.shape[1], self.latent_dim)


class TestContrastiveLoss(unittest.TestCase):
    """Test the ContrastiveLoss class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 16
        self.embedding_dim = 50
        
    def test_contrastive_loss_computation(self):
        """Test contrastive loss computation."""
        loss_fn = ContrastiveLoss(temperature=0.07)
        
        # Create dummy embeddings and labels
        embeddings = torch.randn(self.batch_size, self.embedding_dim)
        labels = torch.randint(0, 2, (self.batch_size,)).float()
        
        loss = loss_fn(embeddings, labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # Scalar
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        
    def test_contrastive_loss_with_all_same_labels(self):
        """Test contrastive loss with all same labels."""
        loss_fn = ContrastiveLoss(temperature=0.07)
        
        embeddings = torch.randn(self.batch_size, self.embedding_dim)
        
        # All positive
        labels_pos = torch.ones(self.batch_size)
        loss_pos = loss_fn(embeddings, labels_pos)
        
        # All negative
        labels_neg = torch.zeros(self.batch_size)
        loss_neg = loss_fn(embeddings, labels_neg)
        
        self.assertFalse(torch.isnan(loss_pos))
        self.assertFalse(torch.isnan(loss_neg))


class TestTripletLoss(unittest.TestCase):
    """Test the TripletLoss class."""
    
    def test_triplet_loss_computation(self):
        """Test triplet loss computation."""
        loss_fn = TripletLoss(margin=1.0)
        
        batch_size = 8
        embedding_dim = 50
        
        anchor = torch.randn(batch_size, embedding_dim)
        positive = anchor + torch.randn(batch_size, embedding_dim) * 0.1  # Close to anchor
        negative = torch.randn(batch_size, embedding_dim)  # Random
        
        loss = loss_fn(anchor, positive, negative)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # Scalar
        self.assertFalse(torch.isnan(loss))
        self.assertTrue(loss >= 0)  # Loss should be non-negative


class TestEncoderTrainer(unittest.TestCase):
    """Test the EncoderTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 476
        self.latent_dim = 50
        self.num_samples = 100
        
        # Create dummy data
        self.states = np.random.randn(self.num_samples, self.input_dim).astype(np.float32)
        self.labels = np.random.randint(0, 2, self.num_samples).astype(np.float32)
        
        # Create dataset and dataloader
        self.dataset = StateEncoderDataset(self.states, self.labels)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=16, shuffle=True
        )
        
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        encoder = SupervisedStateEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim
        )
        
        trainer = EncoderTrainer(encoder, learning_rate=1e-3)
        
        self.assertIsNotNone(trainer.encoder)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.contrastive_loss)
        self.assertIsNotNone(trainer.triplet_loss)
        
    def test_train_epoch(self):
        """Test training for one epoch."""
        encoder = SupervisedStateEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim
        )
        
        trainer = EncoderTrainer(encoder, learning_rate=1e-3)
        
        # Train with contrastive loss
        loss_contrastive = trainer.train_epoch(self.dataloader, use_contrastive=True)
        self.assertIsInstance(loss_contrastive, float)
        self.assertFalse(np.isnan(loss_contrastive))
        
        # Train with triplet loss
        loss_triplet = trainer.train_epoch(self.dataloader, use_contrastive=False)
        self.assertIsInstance(loss_triplet, float)
        self.assertFalse(np.isnan(loss_triplet))
        
    def test_validation(self):
        """Test validation."""
        encoder = SupervisedStateEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim
        )
        
        trainer = EncoderTrainer(encoder, learning_rate=1e-3)
        
        metrics = trainer.validate(self.dataloader)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('val_loss', metrics)
        self.assertIn('embedding_mean', metrics)
        self.assertIn('embedding_std', metrics)
        
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        encoder = SupervisedStateEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim
        )
        
        trainer = EncoderTrainer(encoder, learning_rate=1e-3)
        
        # Train for one epoch
        trainer.train_epoch(self.dataloader)
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            checkpoint_path = tmp.name
            trainer.save_checkpoint(checkpoint_path, epoch=1)
            
            # Load checkpoint
            epoch = trainer.load_checkpoint(checkpoint_path)
            self.assertEqual(epoch, 1)
            
            # Clean up
            os.unlink(checkpoint_path)


class TestEncoderDataExtractor(unittest.TestCase):
    """Test the EncoderDataExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = EncoderDataExtractor(results_dir=self.temp_dir)
        
        self.assertEqual(extractor.results_dir, Path(self.temp_dir))
        self.assertEqual(len(extractor.labeled_data), 0)
        
    def test_labeling_strategies(self):
        """Test different labeling strategies."""
        extractor = EncoderDataExtractor()
        
        # Create dummy episode data
        episode = {
            'completion_status': True,
            'reward': 20.0,
            'states': [np.random.randn(476) for _ in range(5)]
        }
        
        # Test episode labeling
        data_episode = extractor._extract_episode_data(episode, 'episode')
        self.assertEqual(len(data_episode), 5)
        for item in data_episode:
            self.assertEqual(item['label'], 1.0)
            
        # Test trajectory labeling
        data_trajectory = extractor._extract_episode_data(episode, 'trajectory')
        self.assertEqual(len(data_trajectory), 5)
        # Check progressive labels
        self.assertTrue(data_trajectory[0]['label'] < data_trajectory[-1]['label'])
        
        # Test reward labeling
        data_reward = extractor._extract_episode_data(episode, 'reward')
        self.assertEqual(len(data_reward), 5)
        for item in data_reward:
            self.assertTrue(-1 <= item['label'] <= 1)  # Normalized with tanh
            
    def test_balanced_dataset_creation(self):
        """Test balanced dataset creation."""
        extractor = EncoderDataExtractor()
        
        # Create imbalanced data
        for i in range(70):
            extractor.labeled_data.append({
                'state': np.random.randn(476),
                'label': 0.0  # Negative
            })
        
        for i in range(30):
            extractor.labeled_data.append({
                'state': np.random.randn(476),
                'label': 1.0  # Positive
            })
        
        # Balance dataset
        balanced = extractor.create_balanced_dataset(balance_ratio=1.0)
        
        # Check balance
        positive_count = sum(1 for d in balanced['data'] if d['label'] > 0.5)
        negative_count = sum(1 for d in balanced['data'] if d['label'] <= 0.5)
        
        self.assertEqual(positive_count, 30)
        self.assertEqual(negative_count, 30)
        
    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestEncoderUtilityFunctions(unittest.TestCase):
    """Test utility functions for encoder."""
    
    def test_load_encoder(self):
        """Test loading encoder from checkpoint."""
        # Create and save a dummy encoder
        encoder = SupervisedStateEncoder(input_dim=476, latent_dim=50)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            checkpoint_path = tmp.name
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': encoder.state_dict(),
                'config': {
                    'input_dim': 476,
                    'hidden_dims': [256, 128],
                    'latent_dim': 50,
                    'dropout_rate': 0.3,
                    'use_batch_norm': True
                }
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load encoder
            loaded_encoder = load_encoder(checkpoint_path)
            
            self.assertIsInstance(loaded_encoder, SupervisedStateEncoder)
            self.assertEqual(loaded_encoder.input_dim, 476)
            self.assertEqual(loaded_encoder.latent_dim, 50)
            
            # Clean up
            os.unlink(checkpoint_path)
            
    def test_encode_states(self):
        """Test batch encoding of states."""
        encoder = SupervisedStateEncoder(input_dim=476, latent_dim=50)
        encoder.eval()
        
        # Create test states
        states = np.random.randn(100, 476).astype(np.float32)
        
        # Encode states
        encoded = encode_states(encoder, states, batch_size=32)
        
        self.assertEqual(encoded.shape[0], 100)
        self.assertEqual(encoded.shape[1], 50)
        self.assertFalse(np.isnan(encoded).any())


class TestStateEncoderDataset(unittest.TestCase):
    """Test the StateEncoderDataset class."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        states = np.random.randn(100, 476).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.float32)
        
        dataset = StateEncoderDataset(states, labels)
        
        self.assertEqual(len(dataset), 100)
        
        # Test getting item
        state, label = dataset[0]
        self.assertIsInstance(state, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(state.shape[0], 476)
        
    def test_dataloader_compatibility(self):
        """Test dataset with DataLoader."""
        states = np.random.randn(100, 476).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.float32)
        
        dataset = StateEncoderDataset(states, labels)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=True
        )
        
        # Test iteration
        for batch_states, batch_labels in dataloader:
            self.assertEqual(batch_states.shape[1], 476)
            self.assertEqual(len(batch_labels), len(batch_states))
            break  # Just test first batch


if __name__ == '__main__':
    unittest.main()