#!/usr/bin/env python3
"""
PDB structure processing and feature encoding module.

This module provides functionality for downloading, parsing, and encoding
protein structures from PDB files, with support for both BioPython and
simplified parsing methods.
"""

import numpy as np
import requests
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from ..config import get_config
from .constants import (
    AA_PROPERTIES, CATALYTIC_PRIOR, NON_STANDARD_AA_MAP,
    BACKBONE_ATOMS, SIDECHAIN_ATOMS_BY_AA
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

AA_LIST = list(AA_3TO1.keys())

# Catalytic role mapping
ROLE_MAPPING = {
    'nucleophile': 0, 'proton_donor': 1, 'proton_acceptor': 2,
    'electrostatic_stabilizer': 3, 'metal_binding': 4, 'covalent_catalyst': 5,
    'activator': 6, 'steric_role': 7, 'other': 8
}


# =============================================================================
# PDB Processor
# =============================================================================

class PDBProcessor:
    """
    PDB structure processor with robust downloading, caching, and parsing.

    Features:
    - Automatic retry with exponential backoff
    - Cache validation with checksums
    - Rate limiting for API requests
    - Support for both BioPython and simplified parsing
    - Non-standard residue mapping
    - Missing atom handling
    """

    def __init__(self, pdb_dir: Optional[str] = None, config=None):
        """
        Initialize PDB processor.

        Args:
            pdb_dir: Directory for PDB file cache (uses config if None)
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()

        if pdb_dir is None:
            self.pdb_dir = self.config.pdb_dir
        else:
            self.pdb_dir = Path(pdb_dir)

        self.pdb_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration parameters
        self.timeout = self.config.get('data.request_timeout', 30)
        self.max_retries = self.config.get('data.max_retries', 3)
        self.retry_delay = self.config.get('data.retry_delay', 1.0)
        self.rate_limit = self.config.get('data.rate_limit', 0.5)
        self.validate_cache = self.config.get('data.validate_cache', True)

        self._last_request_time = 0

        # Try to import BioPython
        try:
            from Bio.PDB import PDBParser
            self.parser = PDBParser(QUIET=True)
            self.biopython = True
            logger.info("BioPython available, using BioPython parser")
        except ImportError:
            self.biopython = False
            logger.warning("BioPython not installed, using simplified parser")

    def download_pdb(self, pdb_id: str) -> Optional[Path]:
        """
        Download PDB file with retry mechanism and cache validation.

        Args:
            pdb_id: PDB identifier (e.g., '1ABC')

        Returns:
            Path to downloaded PDB file, or None if download failed

        Error codes (logged):
            - CACHE_VALID: File exists and passes validation
            - DOWNLOAD_SUCCESS: Successfully downloaded
            - DOWNLOAD_FAILED: All retry attempts failed
            - VALIDATION_FAILED: Downloaded file failed validation
        """
        pdb_id = pdb_id.lower()
        pdb_file = self.pdb_dir / f"{pdb_id}.pdb"
        metadata_file = self.pdb_dir / f"{pdb_id}.meta.json"

        # Check if file exists and is valid
        if pdb_file.exists():
            if self.validate_cache and not self._validate_cache_file(pdb_file, metadata_file):
                logger.warning(f"Cache validation failed for {pdb_id}, re-downloading")
                pdb_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)
            else:
                logger.debug(f"Using cached PDB file: {pdb_id}")
                return pdb_file

        # Download with retry
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                self._apply_rate_limit()

                logger.info(f"Downloading PDB {pdb_id} (attempt {attempt + 1}/{self.max_retries})")
                response = requests.get(url, timeout=self.timeout)

                if response.status_code == 200:
                    content = response.text

                    # Validate content
                    if not self._validate_pdb_content(content):
                        logger.error(f"Invalid PDB content for {pdb_id}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        return None

                    # Write file atomically with lock
                    temp_file = pdb_file.with_suffix('.tmp')
                    try:
                        with open(temp_file, 'w') as f:
                            f.write(content)

                        # Create metadata
                        metadata = {
                            'pdb_id': pdb_id,
                            'download_time': time.time(),
                            'size': len(content),
                            'checksum': hashlib.md5(content.encode()).hexdigest()
                        }
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f)

                        # Atomic rename
                        temp_file.rename(pdb_file)
                        logger.info(f"Successfully downloaded PDB {pdb_id}")
                        return pdb_file

                    except Exception as e:
                        logger.error(f"Failed to write PDB file {pdb_id}: {e}")
                        temp_file.unlink(missing_ok=True)
                        raise

                elif response.status_code == 404:
                    logger.error(f"PDB {pdb_id} not found (404)")
                    return None
                else:
                    logger.warning(f"Download failed with status {response.status_code}")

            except requests.Timeout:
                logger.warning(f"Timeout downloading PDB {pdb_id}")
            except requests.RequestException as e:
                logger.warning(f"Request error downloading PDB {pdb_id}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error downloading PDB {pdb_id}: {e}")

            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)

        logger.error(f"Failed to download PDB {pdb_id} after {self.max_retries} attempts")
        return None

    def _apply_rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _validate_cache_file(self, pdb_file: Path, metadata_file: Path) -> bool:
        """
        Validate cached PDB file using metadata.

        Args:
            pdb_file: Path to PDB file
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
            actual_size = pdb_file.stat().st_size
            if actual_size != metadata.get('size', -1):
                logger.warning(f"Size mismatch for {pdb_file.name}")
                return False

            # Check checksum
            with open(pdb_file, 'r') as f:
                content = f.read()
            actual_checksum = hashlib.md5(content.encode()).hexdigest()
            if actual_checksum != metadata.get('checksum', ''):
                logger.warning(f"Checksum mismatch for {pdb_file.name}")
                return False

            return True

        except Exception as e:
            logger.warning(f"Failed to validate cache for {pdb_file.name}: {e}")
            return False

    def _validate_pdb_content(self, content: str) -> bool:
        """
        Validate PDB file content.

        Args:
            content: PDB file content

        Returns:
            True if content appears valid, False otherwise
        """
        if not content or len(content) < 100:
            return False

        # Check for essential PDB records
        lines = content.split('\n')
        has_atom = any(line.startswith('ATOM') for line in lines[:1000])

        return has_atom

    def parse_pdb(self, pdb_path: Path) -> Dict:
        """
        Parse PDB file with error handling and validation.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Dictionary containing parsed structure data

        Raises:
            ValueError: If PDB file is invalid or empty
        """
        if not pdb_path.exists():
            raise ValueError(f"PDB file not found: {pdb_path}")

        try:
            if self.biopython:
                result = self._parse_biopython(pdb_path)
            else:
                result = self._parse_simple(pdb_path)

            # Validate result
            if result['num_residues'] == 0:
                raise ValueError(f"No valid residues found in {pdb_path}")

            # Add shape assertions
            assert len(result['residues']) == result['num_residues']
            assert len(result['sequence']) == result['num_residues']

            logger.info(f"Parsed {pdb_path.name}: {result['num_residues']} residues")
            return result

        except Exception as e:
            logger.error(f"Failed to parse PDB {pdb_path}: {e}")
            raise

    def _parse_biopython(self, pdb_path: Path) -> Dict:
        """
        Parse PDB using BioPython with non-standard residue mapping.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Dictionary containing parsed structure data
        """
        structure = self.parser.get_structure('protein', str(pdb_path))

        residues = []
        sequence = []
        skipped_residues = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    het, resseq, icode = residue.get_id()
                    if het != ' ':
                        continue

                    res_name = residue.get_resname()

                    # Map non-standard residues
                    if res_name in NON_STANDARD_AA_MAP:
                        mapped_name = NON_STANDARD_AA_MAP[res_name]
                        logger.debug(f"Mapping non-standard residue {res_name} -> {mapped_name}")
                        res_name_3letter = mapped_name
                        # Convert 1-letter to 3-letter
                        res_name_3letter_full = [k for k, v in AA_3TO1.items() if v == mapped_name][0]
                        res_name = res_name_3letter_full
                    elif res_name not in AA_3TO1:
                        skipped_residues.append((chain.get_id(), resseq, res_name))
                        continue

                    atoms = {}
                    ca_coord = cb_coord = n_coord = c_coord = None

                    for atom in residue:
                        coord = atom.get_coord().tolist()
                        atoms[atom.get_name()] = coord
                        if atom.get_name() == 'CA':
                            ca_coord = coord
                        elif atom.get_name() == 'CB':
                            cb_coord = coord
                        elif atom.get_name() == 'N':
                            n_coord = coord
                        elif atom.get_name() == 'C':
                            c_coord = coord

                    # Check for missing backbone atoms
                    if not ca_coord:
                        logger.warning(f"Missing CA atom for {chain.get_id()}:{resseq}:{res_name}")
                        continue

                    # Handle missing CB (use CA as fallback)
                    if not cb_coord:
                        if res_name != 'GLY':  # Glycine doesn't have CB
                            logger.debug(f"Missing CB atom for {chain.get_id()}:{resseq}:{res_name}, using CA")
                        cb_coord = ca_coord

                    residues.append({
                        'name': res_name,
                        'number': resseq,
                        'chain': chain.get_id(),
                        'ca_coord': ca_coord,
                        'cb_coord': cb_coord,
                        'n_coord': n_coord,
                        'c_coord': c_coord,
                        'atoms': atoms
                    })
                    sequence.append(AA_3TO1[res_name])

        if skipped_residues:
            logger.info(f"Skipped {len(skipped_residues)} non-standard residues: "
                       f"{set(r[2] for r in skipped_residues)}")

        return {
            'pdb_id': pdb_path.stem,
            'residues': residues,
            'sequence': ''.join(sequence),
            'num_residues': len(residues)
        }

    def _parse_simple(self, pdb_path: Path) -> Dict:
        """
        Simplified PDB parser without BioPython dependency.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Dictionary containing parsed structure data
        """
        residues = []
        sequence = []
        current = None
        skipped_residues = []

        with open(pdb_path, 'r') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue

                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21]
                try:
                    res_num = int(line[22:26].strip())
                except ValueError:
                    continue

                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                except ValueError:
                    logger.warning(f"Invalid coordinates in {pdb_path.name} at line: {line[:30]}")
                    continue

                # Map non-standard residues
                if res_name in NON_STANDARD_AA_MAP:
                    mapped_name = NON_STANDARD_AA_MAP[res_name]
                    logger.debug(f"Mapping non-standard residue {res_name} -> {mapped_name}")
                    # Convert 1-letter to 3-letter
                    res_name = [k for k, v in AA_3TO1.items() if v == mapped_name][0]
                elif res_name not in AA_3TO1:
                    if res_name not in [r[2] for r in skipped_residues]:
                        skipped_residues.append((chain, res_num, res_name))
                    continue

                key = (chain, res_num)
                if current is None or (current['chain'], current['number']) != key:
                    if current and current.get('ca_coord'):
                        residues.append(current)
                        sequence.append(AA_3TO1[current['name']])
                    current = {
                        'name': res_name, 'number': res_num, 'chain': chain,
                        'ca_coord': None, 'cb_coord': None,
                        'n_coord': None, 'c_coord': None, 'atoms': {}
                    }

                current['atoms'][atom_name] = [x, y, z]
                if atom_name == 'CA':
                    current['ca_coord'] = [x, y, z]
                elif atom_name == 'CB':
                    current['cb_coord'] = [x, y, z]
                elif atom_name == 'N':
                    current['n_coord'] = [x, y, z]
                elif atom_name == 'C':
                    current['c_coord'] = [x, y, z]

        if current and current.get('ca_coord'):
            residues.append(current)
            sequence.append(AA_3TO1[current['name']])

        # Fill missing CB coordinates
        for r in residues:
            if not r['cb_coord']:
                if r['name'] != 'GLY':
                    logger.debug(f"Missing CB for {r['chain']}:{r['number']}:{r['name']}, using CA")
                r['cb_coord'] = r['ca_coord']

        if skipped_residues:
            logger.info(f"Skipped {len(skipped_residues)} non-standard residues: "
                       f"{set(r[2] for r in skipped_residues)}")

        return {
            'pdb_id': pdb_path.stem,
            'residues': residues,
            'sequence': ''.join(sequence),
            'num_residues': len(residues)
        }


# =============================================================================
# Feature Encoder
# =============================================================================

class FeatureEncoder:
    """
    Residue and structure feature encoder.

    Encodes amino acid residues and protein structures into numerical features
    suitable for machine learning models. Uses biochemical properties from
    the constants module.
    """

    def __init__(self, config=None):
        """
        Initialize feature encoder.

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self.aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}

        # Get edge cutoff from config
        self.default_cutoff = self.config.get('data.edge_cutoff', 8.0)

    def encode_residue(self, res_name: str) -> np.ndarray:
        """
        Encode single residue into feature vector.

        The feature vector contains:
        - One-hot encoding (20 dimensions)
        - Biochemical properties from constants (8 dimensions)

        Args:
            res_name: Three-letter amino acid code (e.g., 'ALA')

        Returns:
            Feature vector of shape (28,) with dtype float32

        Note:
            Uses AA_PROPERTIES and CATALYTIC_PRIOR from constants module
        """
        features = []

        # One-hot encoding (20 dimensions)
        one_hot = [0] * 20
        if res_name in self.aa_to_idx:
            one_hot[self.aa_to_idx[res_name]] = 1
        features.extend(one_hot)

        # Get 1-letter code for property lookup
        aa_1letter = AA_3TO1.get(res_name, 'A')

        # Biochemical properties from constants (8 dimensions)
        if aa_1letter in AA_PROPERTIES:
            props = AA_PROPERTIES[aa_1letter]
            # Properties are already normalized in constants
            features.extend(props.tolist())
        else:
            # Fallback to alanine properties
            logger.warning(f"Unknown residue {res_name}, using alanine properties")
            features.extend(AA_PROPERTIES['A'].tolist())

        # Shape assertion
        feature_array = np.array(features, dtype=np.float32)
        assert feature_array.shape == (28,), f"Expected shape (28,), got {feature_array.shape}"

        return feature_array

    def encode_structure(self, structure_data: Dict, cutoff: Optional[float] = None) -> Dict:
        """
        Encode entire protein structure into graph representation.

        Args:
            structure_data: Dictionary from parse_pdb containing residue information
            cutoff: Distance cutoff for edge construction (uses config default if None)

        Returns:
            Dictionary containing:
                - node_features: (N, 33) array of node features
                - ca_coords: (N, 3) array of CA coordinates
                - cb_coords: (N, 3) array of CB coordinates
                - edge_index: (2, E) array of edge indices
                - edge_attr: (E, 8) array of edge features
                - residue_info: List of (chain, number, name) tuples
                - sequence: Protein sequence string

        Raises:
            ValueError: If structure_data is invalid
        """
        if cutoff is None:
            cutoff = self.default_cutoff

        # Ensure cutoff is float
        cutoff = float(cutoff)

        residues = structure_data['residues']
        n = len(residues)

        if n == 0:
            raise ValueError("No residues in structure_data")

        # Node features
        node_features = np.zeros((n, 28), dtype=np.float32)
        ca_coords = np.zeros((n, 3), dtype=np.float32)
        cb_coords = np.zeros((n, 3), dtype=np.float32)
        residue_info = []

        for i, res in enumerate(residues):
            node_features[i] = self.encode_residue(res['name'])
            ca_coords[i] = res['ca_coord']
            cb_coords[i] = res.get('cb_coord', res['ca_coord'])
            residue_info.append((res['chain'], res['number'], res['name']))

        # Build edges and edge features
        edge_index, edge_attr = self._build_edges(ca_coords, cb_coords, cutoff)

        # Spatial features
        spatial_features = self._compute_spatial_features(ca_coords)
        node_features = np.concatenate([node_features, spatial_features], axis=1)

        # Shape assertions
        assert node_features.shape == (n, 33), f"Expected node_features shape ({n}, 33), got {node_features.shape}"
        assert ca_coords.shape == (n, 3), f"Expected ca_coords shape ({n}, 3), got {ca_coords.shape}"
        assert cb_coords.shape == (n, 3), f"Expected cb_coords shape ({n}, 3), got {cb_coords.shape}"
        assert edge_index.shape[0] == 2, f"Expected edge_index shape (2, E), got {edge_index.shape}"
        if edge_attr.shape[0] > 0:
            assert edge_attr.shape[1] == 8, f"Expected edge_attr shape (E, 8), got {edge_attr.shape}"

        return {
            'node_features': node_features,
            'ca_coords': ca_coords,
            'cb_coords': cb_coords,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'residue_info': residue_info,
            'sequence': structure_data.get('sequence', '')
        }

    def _build_edges(self, ca_coords: np.ndarray, cb_coords: np.ndarray,
                     cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edges with geometric features.

        Args:
            ca_coords: (N, 3) array of CA coordinates
            cb_coords: (N, 3) array of CB coordinates
            cutoff: Distance cutoff for edge construction

        Returns:
            Tuple of:
                - edge_index: (2, E) array of edge indices
                - edge_attr: (E, 8) array of edge features containing:
                    [normalized_ca_dist, normalized_cb_dist, dist_inverse,
                     rbf_encoding, normalized_seq_dist, direction_x, direction_y, direction_z]
        """
        n = len(ca_coords)

        # Compute CA distance matrix
        diff = ca_coords[:, None, :] - ca_coords[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Find edges within cutoff
        src, dst = np.where((dist_matrix < cutoff) & (dist_matrix > 0))

        edge_attr = []
        for s, d in zip(src, dst):
            dist = dist_matrix[s, d]
            direction = (ca_coords[d] - ca_coords[s]) / (dist + 1e-8)
            cb_dist = np.linalg.norm(cb_coords[d] - cb_coords[s])
            seq_dist = abs(d - s)

            edge_attr.append([
                dist / cutoff,                    # Normalized CA distance
                cb_dist / cutoff,                 # Normalized CB distance
                1.0 / (dist + 1.0),              # Distance inverse
                np.exp(-dist**2 / 32),           # RBF encoding
                min(seq_dist, 20) / 20.0,        # Normalized sequence distance
                direction[0], direction[1], direction[2]  # Direction vector
            ])

        edge_index = np.array([src, dst], dtype=np.int64)
        edge_attr = np.array(edge_attr, dtype=np.float32) if edge_attr else np.zeros((0, 8), dtype=np.float32)

        return edge_index, edge_attr

    def _compute_spatial_features(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute spatial environment features for each residue.

        Features include:
        - Local density at 8Å and 12Å
        - Average neighbor distance
        - Burial depth (distance to centroid)
        - Local curvature estimate

        Args:
            coords: (N, 3) array of coordinates

        Returns:
            (N, 5) array of spatial features
        """
        n = len(coords)
        features = np.zeros((n, 5), dtype=np.float32)

        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        centroid = coords.mean(axis=0)

        for i in range(n):
            # Local density (8Å, 12Å)
            features[i, 0] = np.sum(dist_matrix[i] < 8.0) / n
            features[i, 1] = np.sum(dist_matrix[i] < 12.0) / n

            # Average neighbor distance
            sorted_dists = np.sort(dist_matrix[i])
            features[i, 2] = np.mean(sorted_dists[1:min(11, n)]) / 20.0 if n > 1 else 0

            # Burial depth (distance to centroid)
            depth = np.linalg.norm(coords[i] - centroid)
            max_depth = np.max(np.linalg.norm(coords - centroid, axis=1))
            features[i, 3] = depth / (max_depth + 1e-8)

            # Local curvature estimate
            if n > 10:
                neighbors = np.argsort(dist_matrix[i])[1:11]
                neighbor_coords = coords[neighbors]
                local_centroid = neighbor_coords.mean(axis=0)
                features[i, 4] = np.linalg.norm(coords[i] - local_centroid) / 5.0

        return features
