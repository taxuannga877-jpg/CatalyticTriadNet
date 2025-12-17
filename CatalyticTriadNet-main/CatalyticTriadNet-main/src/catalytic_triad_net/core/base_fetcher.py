#!/usr/bin/env python3
"""
Base data fetcher class with common functionality.

This module provides a base class for data fetchers with:
- Rate limiting
- Retry logic with exponential backoff
- Cache validation with MD5 checksums
- Offline mode support
"""

import time
import hashlib
import tempfile
import requests
import json
from pathlib import Path
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod
import logging

from ..config import get_config, Config

logger = logging.getLogger(__name__)


class BaseDataFetcher(ABC):
    """
    Abstract base class for data fetchers.

    Provides common functionality for fetching data from remote APIs:
    - Rate limiting to respect API constraints
    - Exponential backoff retry logic
    - Cache validation with MD5 checksums
    - Atomic file writes
    - Offline mode support

    Subclasses must implement:
    - fetch_* methods for specific data retrieval
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize base data fetcher.

        Args:
            cache_dir: Cache directory path
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self.cache_dir = cache_dir

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration parameters
        self.timeout = self.config.get('data.request_timeout', 30)
        self.max_retries = self.config.get('data.max_retries', 3)
        self.retry_delay = self.config.get('data.retry_delay', 1.0)
        self.rate_limit = self.config.get('data.rate_limit', 0.5)
        self.offline_mode = self.config.get('data.offline_mode', False)
        self._validate_cache_enabled = self.config.get('data.validate_cache', True)

        self._last_request_time = 0.0

    def _rate_limit_wait(self) -> None:
        """
        Apply rate limiting between requests.

        Ensures minimum time between consecutive API requests
        to respect API rate limits.
        """
        if self.rate_limit > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _request_with_retry(
        self,
        url: str,
        method: str = 'GET',
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with exponential backoff retry logic.

        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional arguments for requests

        Returns:
            Response object

        Raises:
            requests.RequestException: If all retries fail
        """
        if self.offline_mode:
            raise requests.RequestException(
                "Offline mode enabled - cannot make network requests"
            )

        self._rate_limit_wait()

        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response

            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Request failed after {self.max_retries} attempts: {url}")
                    raise

                # Exponential backoff
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)

        # Note: This line is unreachable as the loop always returns or raises
        # Kept for defensive programming
        raise requests.RequestException(f"Failed to fetch {url}")

    def _compute_md5(self, file_path: Path) -> str:
        """
        Compute MD5 checksum of a file.

        Args:
            file_path: Path to file

        Returns:
            MD5 checksum as hex string
        """
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def _save_cache_with_checksum(
        self,
        data: Any,
        cache_file: Path,
        checksum_file: Optional[Path] = None
    ) -> None:
        """
        Save data to cache with atomic write and MD5 checksum.

        Args:
            data: Data to save (will be JSON serialized)
            cache_file: Cache file path
            checksum_file: Checksum file path (auto-generated if None)
        """
        if checksum_file is None:
            checksum_file = cache_file.with_suffix(cache_file.suffix + '.md5')

        # Atomic write using temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=cache_file.parent,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_path = Path(tmp_file.name)

        # Compute checksum
        checksum = self._compute_md5(tmp_path)

        # Move to final location
        tmp_path.replace(cache_file)

        # Save checksum
        checksum_file.write_text(checksum)

        logger.debug(f"Saved cache: {cache_file} (MD5: {checksum[:8]}...)")

    def _load_cache_with_validation(
        self,
        cache_file: Path,
        checksum_file: Optional[Path] = None
    ) -> Optional[Any]:
        """
        Load data from cache with checksum validation.

        Args:
            cache_file: Cache file path
            checksum_file: Checksum file path (auto-generated if None)

        Returns:
            Cached data or None if validation fails
        """
        if not cache_file.exists():
            return None

        if checksum_file is None:
            checksum_file = cache_file.with_suffix(cache_file.suffix + '.md5')

        # Skip validation if disabled or checksum file missing
        if not self._validate_cache_enabled or not checksum_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}")
                return None

        # Validate checksum
        try:
            expected_checksum = checksum_file.read_text().strip()
            actual_checksum = self._compute_md5(cache_file)

            if expected_checksum != actual_checksum:
                logger.warning(
                    f"Cache checksum mismatch for {cache_file}: "
                    f"expected {expected_checksum[:8]}..., got {actual_checksum[:8]}..."
                )
                return None

            with open(cache_file, 'r') as f:
                data = json.load(f)

            logger.debug(f"Loaded validated cache: {cache_file}")
            return data

        except Exception as e:
            logger.warning(f"Cache validation failed for {cache_file}: {e}")
            return None

    def validate_cache(self, cache_file: Path) -> bool:
        """
        Validate cache file integrity.

        Args:
            cache_file: Cache file path

        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_file.exists():
            return False

        checksum_file = cache_file.with_suffix(cache_file.suffix + '.md5')

        if not self._validate_cache_enabled or not checksum_file.exists():
            return cache_file.exists()

        try:
            expected_checksum = checksum_file.read_text().strip()
            actual_checksum = self._compute_md5(cache_file)
            return expected_checksum == actual_checksum
        except Exception:
            return False

    @abstractmethod
    def fetch_data(self, *args, **kwargs) -> Any:
        """
        Fetch data from remote source.

        Subclasses must implement this method.
        """
        pass
