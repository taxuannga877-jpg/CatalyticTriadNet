#!/usr/bin/env python3
"""
Base trainer class with common training functionality.

This module provides a base class for model trainers with:
- Standard training loop
- Validation logic
- Checkpoint management
- Early stopping
- Learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
import logging
from tqdm import tqdm

from ..config import get_config

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.

    Provides common functionality for training PyTorch models:
    - Standard training loop with progress bars
    - Validation and early stopping
    - Checkpoint saving/loading
    - Learning rate scheduling
    - Metric tracking

    Subclasses must implement:
    - compute_loss() method for loss calculation
    - compute_metrics() method for metric calculation (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        config=None
    ):
        """
        Initialize base trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer
            device: Device to train on ('cuda' or 'cpu')
            config: Configuration object (uses global config if None)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config or get_config()

        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Load training parameters from config
        self.max_epochs = self.config.get('training.epochs', 100)
        self.early_stopping_patience = self.config.get('training.early_stopping_patience', 10)
        self.gradient_clip = self.config.get('training.gradient_clip', 1.0)

        # Scheduler (optional)
        self.scheduler = None

    def set_scheduler(self, scheduler):
        """Set learning rate scheduler."""
        self.scheduler = scheduler

    @abstractmethod
    def compute_loss(self, batch: Any, model_output: Any) -> torch.Tensor:
        """
        Compute loss for a batch.

        Subclasses must implement this method.

        Args:
            batch: Input batch
            model_output: Model output

        Returns:
            Loss tensor
        """
        pass

    def compute_metrics(self, batch: Any, model_output: Any) -> Dict[str, float]:
        """
        Compute metrics for a batch.

        Subclasses can override this method.

        Args:
            batch: Input batch
            model_output: Model output

        Returns:
            Dictionary of metrics
        """
        return {}

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            # Move batch to device
            batch = self._move_to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()
            model_output = self.model(batch)

            # Compute loss
            loss = self.compute_loss(batch, model_output)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {'loss': avg_loss}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_to_device(batch)

                # Forward pass
                model_output = self.model(batch)

                # Compute loss
                loss = self.compute_loss(batch, model_output)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {'loss': avg_loss}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs (uses config if None)

        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.max_epochs

        history = {
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])

            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_metrics['loss']:.4f}")

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['loss'])

                logger.info(f"Epoch {epoch + 1}/{num_epochs} - Val loss: {val_metrics['loss']:.4f}")

                # Early stopping
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Learning rate scheduling
            if self.scheduler is not None:
                if val_loader is not None:
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

        return history

    def save_checkpoint(self, path: Path, **kwargs):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            **kwargs: Additional data to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            **kwargs
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Checkpoint loaded: {path}")

        return checkpoint

    def _move_to_device(self, batch: Any) -> Any:
        """
        Move batch to device.

        Args:
            batch: Input batch

        Returns:
            Batch on device
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(item) for item in batch)
        else:
            return batch
