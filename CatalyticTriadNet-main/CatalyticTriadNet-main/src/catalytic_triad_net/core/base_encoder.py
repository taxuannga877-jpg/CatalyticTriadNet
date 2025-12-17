#!/usr/bin/env python3
"""
Base feature encoder class with common functionality.

This module provides a base class for feature encoders with:
- Unified encoding interface
- Common feature computation methods
- Validation and error handling
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging

from ..config import get_config

logger = logging.getLogger(__name__)


class BaseFeatureEncoder(ABC):
    """
    Abstract base class for feature encoders.

    Provides common functionality for encoding molecular features:
    - Unified encoding interface
    - Feature validation
    - Error handling
    - Configuration management

    Subclasses must implement:
    - encode() method for specific feature encoding
    """

    def __init__(self, config=None):
        """
        Initialize base feature encoder.

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self._feature_cache = {}

    @abstractmethod
    def encode(self, *args, **kwargs) -> np.ndarray:
        """
        Encode features.

        Subclasses must implement this method.

        Returns:
            Feature vector as numpy array
        """
        pass

    def validate_features(self, features: np.ndarray) -> bool:
        """
        Validate encoded features.

        Args:
            features: Feature array to validate

        Returns:
            True if valid, False otherwise
        """
        if features is None:
            return False

        if not isinstance(features, np.ndarray):
            return False

        if np.any(np.isnan(features)):
            logger.warning("Features contain NaN values")
            return False

        if np.any(np.isinf(features)):
            logger.warning("Features contain infinite values")
            return False

        return True

    def normalize_features(
        self,
        features: np.ndarray,
        method: str = 'standard'
    ) -> np.ndarray:
        """
        Normalize features.

        Args:
            features: Feature array to normalize
            method: Normalization method ('standard', 'minmax', 'none')

        Returns:
            Normalized feature array
        """
        if method == 'none':
            return features

        if method == 'standard':
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[std == 0] = 1.0  # Avoid division by zero
            return (features - mean) / std

        elif method == 'minmax':
            min_val = np.min(features, axis=0)
            max_val = np.max(features, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0  # Avoid division by zero
            return (features - min_val) / range_val

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def clear_cache(self):
        """Clear feature cache."""
        self._feature_cache.clear()

    def get_feature_dim(self) -> int:
        """
        Get feature dimension.

        Subclasses should override this method.

        Returns:
            Feature dimension
        """
        return 0
