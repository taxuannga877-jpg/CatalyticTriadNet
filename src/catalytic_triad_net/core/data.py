#!/usr/bin/env python3
"""
M-CSA data fetching and processing module.

This module provides functionality to fetch catalytic site data from the M-CSA API,
with robust error handling, caching, rate limiting, and retry logic.
"""

import json
import requests
import numpy as np
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
import logging

from ..config import get_config, Config

logger = logging.getLogger(__name__)


# =============================================================================
# M-CSA API数据获取
# =============================================================================

class MCSADataFetcher:
    """
    M-CSA API data fetcher with retry logic, rate limiting, and cache validation.

    This class handles fetching catalytic site data from the M-CSA database with:
    - Exponential backoff retry logic
    - Rate limiting to respect API constraints
    - Atomic cache file writes with validation
    - MD5 checksums for cache integrity

    Args:
        cache_dir: Directory for caching downloaded data
        config: Optional Config instance (uses global config if not provided)
    """

    BASE_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the M-CSA data fetcher.

        Args:
            cache_dir: Optional cache directory path (overrides config)
            config: Optional Config instance
        """
        self.config = config or get_config()

        # Use provided cache_dir or fall back to config
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.config.cache_dir / "mcsa_cache"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration parameters
        self.timeout = self.config.get('data.request_timeout', 30)
        self.max_retries = self.config.get('data.max_retries', 3)
        self.retry_delay = self.config.get('data.retry_delay', 1.0)
        self.rate_limit = self.config.get('data.rate_limit', 0.5)
        self.offline_mode = self.config.get('data.offline_mode', False)
        self.validate_cache = self.config.get('data.validate_cache', True)

        self._last_request_time = 0.0

    def _rate_limit_wait(self) -> None:
        """Apply rate limiting between requests."""
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
        kwargs.setdefault('timeout', self.timeout)

        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()

                if method.upper() == 'GET':
                    response = requests.get(url, **kwargs)
                elif method.upper() == 'POST':
                    response = requests.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff: delay * 2^attempt
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                    raise

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute MD5 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            MD5 hash as hex string
        """
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def _write_cache_atomic(
        self,
        data: Any,
        cache_file: Path,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Write cache file atomically with validation metadata.

        Uses temporary file + atomic rename to prevent corruption.

        Args:
            data: Data to cache (will be JSON serialized)
            cache_file: Target cache file path
            metadata: Optional metadata to store alongside data
        """
        # Create cache structure with metadata
        cache_data = {
            'data': data,
            'metadata': metadata or {},
            'timestamp': time.time(),
        }

        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=cache_file.parent,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            json.dump(cache_data, tmp_file, indent=2, ensure_ascii=False)

        # Compute hash for validation
        file_hash = self._compute_file_hash(tmp_path)
        file_size = tmp_path.stat().st_size

        # Add validation info to metadata
        cache_data['metadata']['hash'] = file_hash
        cache_data['metadata']['size'] = file_size

        # Rewrite with validation metadata
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        # Atomic rename
        tmp_path.replace(cache_file)
        logger.debug(f"Cache written atomically: {cache_file}")

    def _read_cache_validated(self, cache_file: Path) -> Optional[Any]:
        """
        Read and validate cache file.

        Args:
            cache_file: Cache file path

        Returns:
            Cached data if valid, None otherwise
        """
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Extract data and metadata
            data = cache_data.get('data')
            metadata = cache_data.get('metadata', {})

            # Validate if enabled
            if self.validate_cache and 'hash' in metadata:
                # Recompute hash on data portion only
                temp_data = json.dumps(data, ensure_ascii=False)
                current_hash = hashlib.md5(temp_data.encode()).hexdigest()

                stored_hash = metadata.get('hash')
                if current_hash != stored_hash:
                    logger.warning(f"Cache validation failed for {cache_file}: hash mismatch")
                    return None

            logger.info(f"Loaded from cache: {cache_file}")
            return data

        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.warning(f"Failed to read cache {cache_file}: {e}")
            return None

    def fetch_all_entries(self, force_refresh: bool = False) -> List[Dict]:
        """
        Fetch all M-CSA entries with caching and retry logic.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            List of entry dictionaries

        Raises:
            RuntimeError: If offline mode is enabled and cache is unavailable
        """
        cache_file = self.cache_dir / "mcsa_entries_full.json"

        # Try to load from cache
        if not force_refresh:
            cached_data = self._read_cache_validated(cache_file)
            if cached_data is not None:
                return cached_data

        # Check offline mode
        if self.offline_mode:
            raise RuntimeError(
                f"Offline mode enabled but cache unavailable: {cache_file}"
            )

        logger.info("Fetching M-CSA entries from API...")
        url = f"{self.BASE_URL}/entries/?format=json"
        all_results = []
        total_count = None

        while url:
            try:
                response = self._request_with_retry(url)
                data = response.json()

                all_results.extend(data["results"])
                url = data.get("next")

                if total_count is None:
                    total_count = data.get('count', '?')

                logger.info(f"Fetched {len(all_results)}/{total_count} entries")

            except Exception as e:
                logger.error(f"Failed to fetch entries: {e}")
                if all_results:
                    logger.warning("Returning partial results")
                    break
                else:
                    raise

        # Write to cache atomically
        metadata = {
            'source': 'M-CSA API',
            'endpoint': 'entries',
            'count': len(all_results)
        }
        self._write_cache_atomic(all_results, cache_file, metadata)

        return all_results

    def fetch_all_residues(self, force_refresh: bool = False) -> List[Dict]:
        """
        Fetch all M-CSA catalytic residues with caching and retry logic.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            List of residue dictionaries

        Raises:
            RuntimeError: If offline mode is enabled and cache is unavailable
        """
        cache_file = self.cache_dir / "mcsa_residues_full.json"

        # Try to load from cache
        if not force_refresh:
            cached_data = self._read_cache_validated(cache_file)
            if cached_data is not None:
                return cached_data

        # Check offline mode
        if self.offline_mode:
            raise RuntimeError(
                f"Offline mode enabled but cache unavailable: {cache_file}"
            )

        logger.info("Fetching M-CSA residues from API...")
        url = f"{self.BASE_URL}/residues/?format=json"
        all_results = []
        total_count = None

        while url:
            try:
                response = self._request_with_retry(url)
                data = response.json()

                all_results.extend(data["results"])
                url = data.get("next")

                if total_count is None:
                    total_count = data.get('count', '?')

                logger.info(f"Fetched {len(all_results)}/{total_count} residues")

            except Exception as e:
                logger.error(f"Failed to fetch residues: {e}")
                if all_results:
                    logger.warning("Returning partial results")
                    break
                else:
                    raise

        # Write to cache atomically
        metadata = {
            'source': 'M-CSA API',
            'endpoint': 'residues',
            'count': len(all_results)
        }
        self._write_cache_atomic(all_results, cache_file, metadata)

        return all_results


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class CatalyticResidue:
    """催化残基"""
    residue_name: str
    residue_number: int
    chain_id: str
    pdb_id: str
    roles: List[str] = field(default_factory=list)


@dataclass
class EnzymeEntry:
    """酶条目"""
    mcsa_id: str
    enzyme_name: str
    ec_numbers: List[str]
    pdb_id: str
    catalytic_residues: List[CatalyticResidue] = field(default_factory=list)


# =============================================================================
# M-CSA数据解析
# =============================================================================

class MCSADataParser:
    """M-CSA数据解析器"""

    ROLE_MAPPING = {
        'nucleophile': 'nucleophile',
        'proton donor': 'proton_donor',
        'proton acceptor': 'proton_acceptor',
        'electrostatic stabiliser': 'electrostatic_stabilizer',
        'metal ligand': 'metal_binding',
        'covalent catalyst': 'covalent_catalyst',
        'activator': 'activator',
        'steric role': 'steric_role',
    }

    def parse_entries(self, entries_json: List[Dict]) -> List[EnzymeEntry]:
        """解析entries"""
        parsed = []
        for entry in tqdm(entries_json, desc="解析M-CSA"):
            try:
                enzyme = self._parse_entry(entry)
                if enzyme and enzyme.catalytic_residues:
                    parsed.append(enzyme)
            except Exception as e:
                logger.debug(f"解析失败: {e}")

        logger.info(f"✓ 解析 {len(parsed)} 个酶条目")
        return parsed

    def _parse_entry(self, entry: Dict) -> Optional[EnzymeEntry]:
        """解析单个entry"""
        residues_data = entry.get('residues', [])
        if not residues_data:
            return None

        chains = residues_data[0].get('residue_chains', [])
        if not chains:
            return None

        pdb_id = chains[0].get('pdb_id', '').lower()
        if not pdb_id:
            return None

        catalytic_residues = []
        for res in residues_data:
            cat_res = self._parse_residue(res, pdb_id)
            if cat_res:
                catalytic_residues.append(cat_res)

        return EnzymeEntry(
            mcsa_id=entry.get('mcsa_id', ''),
            enzyme_name=entry.get('enzyme_name', ''),
            ec_numbers=entry.get('all_ecs', []),
            pdb_id=pdb_id,
            catalytic_residues=catalytic_residues
        )

    def _parse_residue(self, res_data: Dict, pdb_id: str) -> Optional[CatalyticResidue]:
        """解析催化残基"""
        chains = res_data.get('residue_chains', [])
        if not chains:
            return None

        c = chains[0]
        roles = []
        for role in res_data.get('roles', []):
            func = role.get('function', '').lower()
            for k, v in self.ROLE_MAPPING.items():
                if k in func:
                    roles.append(v)
                    break

        return CatalyticResidue(
            residue_name=c.get('code', ''),
            residue_number=c.get('resid', 0),
            chain_id=c.get('chain_name', 'A'),
            pdb_id=pdb_id,
            roles=roles or ['other']
        )

    def get_statistics(self, entries: List[EnzymeEntry]) -> Dict:
        """统计信息"""
        stats = {
            'total_entries': len(entries),
            'total_catalytic': sum(len(e.catalytic_residues) for e in entries),
            'unique_pdbs': len(set(e.pdb_id for e in entries)),
            'ec_dist': defaultdict(int),
            'res_dist': defaultdict(int),
        }

        for e in entries:
            for ec in e.ec_numbers:
                ec1 = ec.split('.')[0] if '.' in ec else ec
                stats['ec_dist'][ec1] += 1
            for r in e.catalytic_residues:
                stats['res_dist'][r.residue_name] += 1

        return stats
