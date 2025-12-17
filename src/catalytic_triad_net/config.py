"""
Unified configuration system for Catalytic Triad Net.

This module provides centralized configuration management for all components,
including data processing, model architecture, training, and inference.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import logging

logger = logging.getLogger(__name__)


class Config:
    """Centralized configuration manager."""

    # Default paths
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "catalytic_triad_net"
    DEFAULT_PDB_DIR = DEFAULT_CACHE_DIR / "pdb"
    DEFAULT_SWISSPROT_CACHE = DEFAULT_CACHE_DIR / "swissprot"

    # Data processing defaults
    DEFAULT_EDGE_CUTOFF = 8.0  # Angstroms
    DEFAULT_MAX_RESIDUES = 1000
    DEFAULT_REQUEST_TIMEOUT = 30  # seconds
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0  # seconds
    DEFAULT_RATE_LIMIT = 0.5  # seconds between requests

    # Model architecture defaults
    DEFAULT_NODE_DIM = 48
    DEFAULT_EDGE_DIM = 14
    DEFAULT_HIDDEN_DIM = 128
    DEFAULT_NUM_LAYERS = 4
    DEFAULT_NUM_HEADS = 4
    DEFAULT_DROPOUT = 0.1

    # Training defaults
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_WEIGHT_DECAY = 1e-5
    DEFAULT_MAX_EPOCHS = 100
    DEFAULT_PATIENCE = 10
    DEFAULT_GRAD_CLIP = 1.0

    # Inference defaults
    DEFAULT_SITE_THRESHOLD = 0.5
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7

    # Generation defaults
    DEFAULT_NUM_DIFFUSION_STEPS = 1000
    DEFAULT_BETA_START = 1e-4
    DEFAULT_BETA_END = 0.02
    DEFAULT_NUM_SAMPLES = 10
    DEFAULT_TEMPERATURE = 1.0

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            config_path: Optional path to YAML config file
        """
        self._config: Dict[str, Any] = {}

        # Load from file if provided
        if config_path and config_path.exists():
            self.load_from_file(config_path)

        # Set defaults for missing values
        self._set_defaults()

    def _set_defaults(self):
        """Set default values for all configuration options."""
        # Paths
        self._config.setdefault('cache_dir', str(self.DEFAULT_CACHE_DIR))
        self._config.setdefault('pdb_dir', str(self.DEFAULT_PDB_DIR))
        self._config.setdefault('swissprot_cache', str(self.DEFAULT_SWISSPROT_CACHE))

        # Data processing
        data_config = self._config.setdefault('data', {})
        data_config.setdefault('edge_cutoff', self.DEFAULT_EDGE_CUTOFF)
        data_config.setdefault('max_residues', self.DEFAULT_MAX_RESIDUES)
        data_config.setdefault('request_timeout', self.DEFAULT_REQUEST_TIMEOUT)
        data_config.setdefault('max_retries', self.DEFAULT_MAX_RETRIES)
        data_config.setdefault('retry_delay', self.DEFAULT_RETRY_DELAY)
        data_config.setdefault('rate_limit', self.DEFAULT_RATE_LIMIT)
        data_config.setdefault('offline_mode', False)
        data_config.setdefault('validate_cache', True)

        # Model architecture
        model_config = self._config.setdefault('model', {})
        model_config.setdefault('node_dim', self.DEFAULT_NODE_DIM)
        model_config.setdefault('edge_dim', self.DEFAULT_EDGE_DIM)
        model_config.setdefault('hidden_dim', self.DEFAULT_HIDDEN_DIM)
        model_config.setdefault('num_layers', self.DEFAULT_NUM_LAYERS)
        model_config.setdefault('num_heads', self.DEFAULT_NUM_HEADS)
        model_config.setdefault('dropout', self.DEFAULT_DROPOUT)

        # Training
        train_config = self._config.setdefault('training', {})
        train_config.setdefault('batch_size', self.DEFAULT_BATCH_SIZE)
        train_config.setdefault('learning_rate', self.DEFAULT_LEARNING_RATE)
        train_config.setdefault('weight_decay', self.DEFAULT_WEIGHT_DECAY)
        train_config.setdefault('max_epochs', self.DEFAULT_MAX_EPOCHS)
        train_config.setdefault('patience', self.DEFAULT_PATIENCE)
        train_config.setdefault('grad_clip', self.DEFAULT_GRAD_CLIP)
        train_config.setdefault('num_workers', 4)

        # Inference
        inference_config = self._config.setdefault('inference', {})
        inference_config.setdefault('site_threshold', self.DEFAULT_SITE_THRESHOLD)
        inference_config.setdefault('confidence_threshold', self.DEFAULT_CONFIDENCE_THRESHOLD)
        inference_config.setdefault('batch_size', 1)

        # Generation
        generation_config = self._config.setdefault('generation', {})
        generation_config.setdefault('num_diffusion_steps', self.DEFAULT_NUM_DIFFUSION_STEPS)
        generation_config.setdefault('beta_start', self.DEFAULT_BETA_START)
        generation_config.setdefault('beta_end', self.DEFAULT_BETA_END)
        generation_config.setdefault('num_samples', self.DEFAULT_NUM_SAMPLES)
        generation_config.setdefault('temperature', self.DEFAULT_TEMPERATURE)
        generation_config.setdefault('max_atoms', 200)

        # Batch screening
        screening_config = self._config.setdefault('screening', {})
        screening_config.setdefault('num_workers', 4)
        screening_config.setdefault('max_retries', 3)
        screening_config.setdefault('timeout', 300)

    def load_from_file(self, config_path: Path):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    self._config.update(loaded_config)
                    logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    def save_to_file(self, config_path: Path):
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save YAML config file
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'model.node_dim')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()

    @property
    def cache_dir(self) -> Path:
        """Get cache directory path."""
        return Path(self._config['cache_dir'])

    @property
    def pdb_dir(self) -> Path:
        """Get PDB directory path."""
        return Path(self._config['pdb_dir'])

    @property
    def swissprot_cache(self) -> Path:
        """Get SwissProt cache directory path."""
        return Path(self._config['swissprot_cache'])


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance.

    Returns:
        Global Config instance
    """
    global _global_config
    if _global_config is None:
        # Try to load from default locations
        default_paths = [
            Path.cwd() / "config.yaml",
            Path.home() / ".catalytic_triad_net" / "config.yaml",
        ]

        config_path = None
        for path in default_paths:
            if path.exists():
                config_path = path
                break

        _global_config = Config(config_path)

    return _global_config


def set_config(config: Config):
    """
    Set global configuration instance.

    Args:
        config: Config instance to set as global
    """
    global _global_config
    _global_config = config


def reset_config():
    """Reset global configuration to defaults."""
    global _global_config
    _global_config = None
