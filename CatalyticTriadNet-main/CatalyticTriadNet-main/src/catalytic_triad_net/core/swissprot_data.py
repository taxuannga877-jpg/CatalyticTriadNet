#!/usr/bin/env python3
"""
Swiss-Prot data fetching and parsing module.

Swiss-Prot is the manually curated section of UniProtKB, containing 570,000+
high-quality protein sequences. Compared to M-CSA's ~1,000 entries, Swiss-Prot
provides much larger-scale training data.

Data source:
- UniProt REST API: https://rest.uniprot.org/
- Data volume: 570,000+ protein sequences
- Enzyme data: ~200,000 entries
"""

import requests
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SwissProtEntry:
    """Swiss-Prot entry data structure."""
    uniprot_id: str
    protein_name: str
    organism: str
    sequence: str
    ec_number: Optional[str]
    pdb_ids: List[str]
    active_sites: List[Dict]  # {'position': int, 'residue': str, 'description': str}
    binding_sites: List[Dict]
    metal_binding: List[Dict]
    catalytic_activity: Optional[str]
    function: Optional[str]


class SwissProtDataFetcher:
    """
    Swiss-Prot data fetcher with robust error handling.

    Features:
    - Automatic retry with exponential backoff
    - Rate limiting for API requests
    - Cache validation with checksums
    - Offline mode support
    - Comprehensive error handling

    Usage example:
        fetcher = SwissProtDataFetcher()

        # Fetch all hydrolases (EC 3)
        entries = fetcher.fetch_enzymes_by_ec_class(ec_class='3', limit=1000)

        # Fetch enzymes by specific EC number
        entries = fetcher.fetch_enzymes_by_ec_number('3.4.21.4')

        # Fetch enzymes with PDB structures
        entries = fetcher.fetch_enzymes_with_structure(ec_class='3', limit=500)
    """

    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    def __init__(self, cache_dir: Optional[str] = None, config=None):
        """
        Initialize Swiss-Prot data fetcher.

        Args:
            cache_dir: Cache directory (uses config if None)
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()

        if cache_dir is None:
            self.cache_dir = self.config.swissprot_cache
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration parameters
        self.timeout = self.config.get('data.request_timeout', 30)
        self.max_retries = self.config.get('data.max_retries', 3)
        self.retry_delay = self.config.get('data.retry_delay', 1.0)
        self.rate_limit = self.config.get('data.rate_limit', 0.5)
        self.offline_mode = self.config.get('data.offline_mode', False)
        self.validate_cache = self.config.get('data.validate_cache', True)

        self._last_request_time = 0

        logger.info(f"Swiss-Prot data fetcher initialized: cache_dir={self.cache_dir}, "
                   f"offline_mode={self.offline_mode}")

    def fetch_enzymes_by_ec_class(
        self,
        ec_class: str,
        limit: int = 1000,
        reviewed: bool = True,
        has_structure: bool = False,
        has_active_site: bool = False,
        has_catalytic_activity: bool = False
    ) -> List[Dict]:
        """
        Fetch enzyme data by EC classification with fine-grained filtering.

        Args:
            ec_class: EC classification ('1', '2', '3', '4', '5', '6', '7')
            limit: Maximum number of entries
            reviewed: Only fetch Swiss-Prot reviewed entries (recommended True)
            has_structure: Only fetch entries with PDB structures (recommended True for training)
            has_active_site: Only fetch entries with active site annotations (recommended True)
            has_catalytic_activity: Only fetch entries with catalytic activity descriptions (recommended True)

        Returns:
            List of entry dictionaries

        Recommended configurations:
            # High-quality training data (most strict)
            fetch_enzymes_by_ec_class(
                ec_class='3',
                reviewed=True,
                has_structure=True,
                has_active_site=True,
                has_catalytic_activity=True
            )

            # Medium quality (balance quantity and quality)
            fetch_enzymes_by_ec_class(
                ec_class='3',
                reviewed=True,
                has_structure=True,
                has_active_site=True
            )

            # Large-scale data (only require EC number)
            fetch_enzymes_by_ec_class(
                ec_class='3',
                reviewed=True
            )
        """
        cache_file = self.cache_dir / f"ec{ec_class}_limit{limit}_struct{has_structure}.json"
        metadata_file = cache_file.with_suffix('.meta.json')

        # Check cache
        if cache_file.exists():
            if self.validate_cache and not self._validate_cache_file(cache_file, metadata_file):
                logger.warning(f"Cache validation failed for {cache_file.name}, re-fetching")
                cache_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)
            else:
                logger.info(f"Loading from cache: {cache_file}")
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load cache: {e}")
                    cache_file.unlink(missing_ok=True)
                    metadata_file.unlink(missing_ok=True)

        # Offline mode check
        if self.offline_mode:
            logger.error(f"Offline mode enabled and no cache found for EC {ec_class}")
            return []

        # Build query
        query_parts = [f"(ec:{ec_class}.*)"]

        if reviewed:
            query_parts.append("(reviewed:true)")

        if has_structure:
            query_parts.append("(structure_3d:true)")

        query = " AND ".join(query_parts)

        logger.info(f"Querying Swiss-Prot: EC {ec_class}, limit={limit}")
        logger.info(f"Query string: {query}")

        # Send request
        params = {
            'query': query,
            'format': 'json',
            'size': min(limit * 2, 500),  # Fetch more for filtering
            'fields': 'accession,id,protein_name,organism_name,sequence,ec,xref_pdb,ft_act_site,ft_binding,ft_metal,cc_catalytic_activity,cc_function'
        }

        raw_entries = []
        cursor = None

        # Fetch raw data (may need filtering)
        while len(raw_entries) < limit * 2:
            if cursor:
                params['cursor'] = cursor

            try:
                # Apply rate limiting
                self._apply_rate_limit()

                response = self._request_with_retry(
                    f"{self.BASE_URL}/search",
                    params=params
                )

                if response is None:
                    break

                data = response.json()
                results = data.get('results', [])

                if not results:
                    break

                raw_entries.extend(results)
                logger.info(f"Fetched {len(raw_entries)} raw entries...")

                # Get next page cursor
                cursor = response.headers.get('x-next-cursor')
                if not cursor:
                    break

            except Exception as e:
                logger.error(f"Request failed: {e}")
                break

        # Post-filtering: filter by additional conditions
        filtered_entries = []

        for entry in raw_entries:
            # Check if all conditions are met
            if has_active_site:
                # Must have active site annotation
                has_act_site = False
                if entry.get('features'):
                    for feat in entry['features']:
                        if feat.get('type') == 'Active site':
                            has_act_site = True
                            break
                if not has_act_site:
                    continue

            if has_catalytic_activity:
                # Must have catalytic activity description
                has_cat_activity = False
                if entry.get('comments'):
                    for comment in entry['comments']:
                        if comment.get('commentType') == 'CATALYTIC ACTIVITY':
                            has_cat_activity = True
                            break
                if not has_cat_activity:
                    continue

            # Passed all filtering conditions
            filtered_entries.append(entry)

            if len(filtered_entries) >= limit:
                break

        logger.info(f"After filtering: {len(filtered_entries)}/{len(raw_entries)} entries")

        # Save cache with metadata
        if filtered_entries:
            self._save_cache_with_metadata(cache_file, metadata_file, filtered_entries)

        logger.info(f"Fetch complete: {len(filtered_entries)} high-quality entries")

        return filtered_entries

    def fetch_enzymes_by_ec_number(
        self,
        ec_number: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch enzyme data by exact EC number.

        Args:
            ec_number: Complete EC number (e.g., '3.4.21.4')
            limit: Maximum number of entries

        Returns:
            List of entry dictionaries
        """
        cache_file = self.cache_dir / f"ec_{ec_number.replace('.', '_')}.json"
        metadata_file = cache_file.with_suffix('.meta.json')

        # Check cache
        if cache_file.exists():
            if self.validate_cache and not self._validate_cache_file(cache_file, metadata_file):
                logger.warning(f"Cache validation failed, re-fetching")
                cache_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)
            else:
                logger.info(f"Loading from cache: {cache_file}")
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load cache: {e}")

        # Offline mode check
        if self.offline_mode:
            logger.error(f"Offline mode enabled and no cache found for EC {ec_number}")
            return []

        query = f"(ec:{ec_number}) AND (reviewed:true)"

        params = {
            'query': query,
            'format': 'json',
            'size': limit,
            'fields': 'accession,id,protein_name,organism_name,sequence,ec,xref_pdb,ft_act_site,ft_binding,ft_metal,cc_catalytic_activity,cc_function'
        }

        try:
            # Apply rate limiting
            self._apply_rate_limit()

            response = self._request_with_retry(
                f"{self.BASE_URL}/search",
                params=params
            )

            if response is None:
                return []

            data = response.json()
            entries = data.get('results', [])

            # Save cache with metadata
            if entries:
                self._save_cache_with_metadata(cache_file, metadata_file, entries)

            logger.info(f"Fetched EC {ec_number}: {len(entries)} entries")

            return entries

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return []

    def fetch_enzymes_with_structure(
        self,
        ec_class: str,
        limit: int = 500
    ) -> List[Dict]:
        """
        Fetch enzyme data with PDB structures.

        Args:
            ec_class: EC classification
            limit: Maximum number of entries

        Returns:
            List of entry dictionaries
        """
        return self.fetch_enzymes_by_ec_class(
            ec_class=ec_class,
            limit=limit,
            reviewed=True,
            has_structure=True
        )

    def get_statistics(self, entries: List[Dict]) -> Dict:
        """
        Compute statistics for Swiss-Prot data.

        Args:
            entries: List of entry dictionaries

        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_entries': len(entries),
            'with_pdb': 0,
            'with_active_site': 0,
            'with_binding_site': 0,
            'with_metal': 0,
            'ec_distribution': {},
            'organism_distribution': {}
        }

        for entry in entries:
            # PDB structures
            if entry.get('uniProtKBCrossReferences'):
                pdb_refs = [ref for ref in entry['uniProtKBCrossReferences']
                           if ref.get('database') == 'PDB']
                if pdb_refs:
                    stats['with_pdb'] += 1

            # Active sites
            if entry.get('features'):
                for feat in entry['features']:
                    if feat.get('type') == 'Active site':
                        stats['with_active_site'] += 1
                        break

            # Binding sites
            if entry.get('features'):
                for feat in entry['features']:
                    if feat.get('type') == 'Binding site':
                        stats['with_binding_site'] += 1
                        break

            # Metal binding
            if entry.get('features'):
                for feat in entry['features']:
                    if feat.get('type') == 'Metal binding':
                        stats['with_metal'] += 1
                        break

            # EC distribution
            if entry.get('proteinDescription', {}).get('recommendedName', {}).get('ecNumbers'):
                ec_numbers = entry['proteinDescription']['recommendedName']['ecNumbers']
                for ec in ec_numbers:
                    ec_value = ec.get('value', '')
                    ec_class = ec_value.split('.')[0] if ec_value else 'unknown'
                    stats['ec_distribution'][ec_class] = stats['ec_distribution'].get(ec_class, 0) + 1

            # Organism distribution
            organism = entry.get('organism', {}).get('scientificName', 'unknown')
            stats['organism_distribution'][organism] = stats['organism_distribution'].get(organism, 0) + 1

        return stats

    def _request_with_retry(self, url: str, params: Dict) -> Optional[requests.Response]:
        """
        Make HTTP request with retry mechanism.

        Args:
            url: Request URL
            params: Request parameters

        Returns:
            Response object or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response

            except requests.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
            except requests.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}/{self.max_retries}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")

            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)

        logger.error(f"Failed to fetch data after {self.max_retries} attempts")
        return None

    def _apply_rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _validate_cache_file(self, cache_file: Path, metadata_file: Path) -> bool:
        """
        Validate cached file using metadata.

        Args:
            cache_file: Path to cache file
            metadata_file: Path to metadata file

        Returns:
            True if file is valid, False otherwise
        """
        if not metadata_file.exists():
            return False

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check file size
            actual_size = cache_file.stat().st_size
            if actual_size != metadata.get('size', -1):
                logger.warning(f"Size mismatch for {cache_file.name}")
                return False

            # Check checksum
            with open(cache_file, 'r') as f:
                content = f.read()
            actual_checksum = hashlib.md5(content.encode()).hexdigest()
            if actual_checksum != metadata.get('checksum', ''):
                logger.warning(f"Checksum mismatch for {cache_file.name}")
                return False

            return True

        except Exception as e:
            logger.warning(f"Failed to validate cache for {cache_file.name}: {e}")
            return False

    def _save_cache_with_metadata(self, cache_file: Path, metadata_file: Path, data: List[Dict]):
        """
        Save cache file with metadata atomically.

        Args:
            cache_file: Path to cache file
            metadata_file: Path to metadata file
            data: Data to save
        """
        temp_file = cache_file.with_suffix('.tmp')

        try:
            # Write data to temporary file
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Create metadata
            with open(temp_file, 'r') as f:
                content = f.read()

            metadata = {
                'timestamp': time.time(),
                'size': len(content),
                'checksum': hashlib.md5(content.encode()).hexdigest(),
                'num_entries': len(data)
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)

            # Atomic rename
            temp_file.rename(cache_file)
            logger.debug(f"Saved cache: {cache_file}")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            temp_file.unlink(missing_ok=True)
            raise


class SwissProtDataParser:
    """
    Swiss-Prot data parser.

    Parses Swiss-Prot JSON data into structured objects.
    """

    @staticmethod
    def parse_entry(entry: Dict) -> SwissProtEntry:
        """
        Parse single Swiss-Prot entry.

        Args:
            entry: Swiss-Prot JSON entry

        Returns:
            SwissProtEntry object
        """
        # Basic information
        uniprot_id = entry.get('primaryAccession', '')
        protein_name = entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
        organism = entry.get('organism', {}).get('scientificName', '')
        sequence = entry.get('sequence', {}).get('value', '')

        # EC number
        ec_number = None
        if entry.get('proteinDescription', {}).get('recommendedName', {}).get('ecNumbers'):
            ec_numbers = entry['proteinDescription']['recommendedName']['ecNumbers']
            if ec_numbers:
                ec_number = ec_numbers[0].get('value')

        # PDB IDs
        pdb_ids = []
        if entry.get('uniProtKBCrossReferences'):
            for ref in entry['uniProtKBCrossReferences']:
                if ref.get('database') == 'PDB':
                    pdb_ids.append(ref.get('id'))

        # Active sites
        active_sites = []
        binding_sites = []
        metal_binding = []

        if entry.get('features'):
            for feat in entry['features']:
                feat_type = feat.get('type')
                location = feat.get('location', {})
                start = location.get('start', {}).get('value')
                description = feat.get('description', '')

                if feat_type == 'Active site':
                    active_sites.append({
                        'position': start,
                        'description': description
                    })
                elif feat_type == 'Binding site':
                    binding_sites.append({
                        'position': start,
                        'description': description
                    })
                elif feat_type == 'Metal binding':
                    metal_binding.append({
                        'position': start,
                        'description': description
                    })

        # Catalytic activity
        catalytic_activity = None
        if entry.get('comments'):
            for comment in entry['comments']:
                if comment.get('commentType') == 'CATALYTIC ACTIVITY':
                    reaction = comment.get('reaction', {})
                    catalytic_activity = reaction.get('name', '')
                    break

        # Function
        function = None
        if entry.get('comments'):
            for comment in entry['comments']:
                if comment.get('commentType') == 'FUNCTION':
                    texts = comment.get('texts', [])
                    if texts:
                        function = texts[0].get('value', '')
                    break

        return SwissProtEntry(
            uniprot_id=uniprot_id,
            protein_name=protein_name,
            organism=organism,
            sequence=sequence,
            ec_number=ec_number,
            pdb_ids=pdb_ids,
            active_sites=active_sites,
            binding_sites=binding_sites,
            metal_binding=metal_binding,
            catalytic_activity=catalytic_activity,
            function=function
        )

    @staticmethod
    def parse_entries(entries: List[Dict]) -> List[SwissProtEntry]:
        """
        Parse multiple Swiss-Prot entries.

        Args:
            entries: List of Swiss-Prot JSON entries

        Returns:
            List of SwissProtEntry objects
        """
        parsed_entries = []

        for entry in entries:
            try:
                parsed = SwissProtDataParser.parse_entry(entry)
                parsed_entries.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse entry: {e}")

        return parsed_entries


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'SwissProtEntry',
    'SwissProtDataFetcher',
    'SwissProtDataParser',
]


if __name__ == "__main__":
    # Test code
    print("Swiss-Prot Data Fetcher - Test")
    print("="*60)

    fetcher = SwissProtDataFetcher()

    # Fetch hydrolase data
    print("\nFetching hydrolase (EC 3) data...")
    entries = fetcher.fetch_enzymes_by_ec_class(ec_class='3', limit=100)

    print(f"\nFetched {len(entries)} entries")

    # Statistics
    stats = fetcher.get_statistics(entries)
    print(f"\nStatistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  With PDB structure: {stats['with_pdb']}")
    print(f"  With active site annotation: {stats['with_active_site']}")
    print(f"  With binding site annotation: {stats['with_binding_site']}")
    print(f"  With metal binding annotation: {stats['with_metal']}")

    # Parse first entry
    if entries:
        print(f"\nParsing first entry...")
        parser = SwissProtDataParser()
        parsed = parser.parse_entry(entries[0])

        print(f"\nUniProt ID: {parsed.uniprot_id}")
        print(f"Protein name: {parsed.protein_name}")
        print(f"EC number: {parsed.ec_number}")
        print(f"PDB IDs: {parsed.pdb_ids}")
        print(f"Number of active sites: {len(parsed.active_sites)}")
        print(f"Sequence length: {len(parsed.sequence)}")
