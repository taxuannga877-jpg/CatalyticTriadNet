#!/usr/bin/env python3
"""
Dataset module for catalytic site prediction.

This module provides PyTorch Dataset classes with support for:
- Lazy loading for memory efficiency
- Multiprocessing for faster preprocessing
- Proper batch collation with shape assertions
- Configuration-based parameter management
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

from .data import EnzymeEntry
from .structure import PDBProcessor, FeatureEncoder, ROLE_MAPPING
from ..config import get_config, Config

logger = logging.getLogger(__name__)


class CatalyticSiteDataset(Dataset):
    """
    Catalytic site dataset with lazy loading and multiprocessing support.

    This dataset handles protein structures and their catalytic site annotations,
    with options for eager or lazy loading and parallel preprocessing.

    Args:
        entries: List of enzyme entries to process
        pdb_processor: PDB structure processor
        feature_encoder: Feature encoder for structures
        max_residues: Maximum number of residues per structure
        edge_cutoff: Distance cutoff for edge construction (Angstroms)
        lazy_loading: If True, process structures on-the-fly (saves memory)
        num_workers: Number of workers for parallel preprocessing (0 = no parallelism)
        config: Optional Config instance
    """

    def __init__(
        self,
        entries: List[EnzymeEntry],
        pdb_processor: PDBProcessor,
        feature_encoder: FeatureEncoder,
        max_residues: Optional[int] = None,
        edge_cutoff: Optional[float] = None,
        lazy_loading: bool = False,
        num_workers: int = 0,
        config: Optional[Config] = None
    ):
        self.entries = entries
        self.pdb_proc = pdb_processor
        self.feat_enc = feature_encoder
        self.lazy_loading = lazy_loading

        # Load configuration
        self.config = config or get_config()
        self.max_residues = max_residues or self.config.get('data.max_residues', 1000)
        self.edge_cutoff = edge_cutoff or self.config.get('data.edge_cutoff', 8.0)
        self.num_workers = num_workers if num_workers > 0 else self.config.get('training.num_workers', 4)

        # Storage for processed data
        self.processed: List[Dict] = []

        # Preprocess if not lazy loading
        if not lazy_loading:
            self._preprocess()
        else:
            logger.info(f"Lazy loading enabled for {len(entries)} entries")

    def _preprocess(self):
        """
        Preprocess all entries with optional multiprocessing.

        Uses parallel processing if num_workers > 1, otherwise sequential.
        """
        if self.num_workers > 1:
            self._preprocess_parallel()
        else:
            self._preprocess_sequential()

        logger.info(f"Processed {len(self.processed)}/{len(self.entries)} samples")

    def _preprocess_sequential(self):
        """Sequential preprocessing of entries."""
        for entry in tqdm(self.entries, desc="Processing structures"):
            try:
                data = self._process_entry(entry)
                if data:
                    self.processed.append(data)
            except Exception as e:
                logger.debug(f"Failed to process {entry.pdb_id}: {e}")

    def _preprocess_parallel(self):
        """Parallel preprocessing of entries using multiprocessing."""
        logger.info(f"Using {self.num_workers} workers for parallel preprocessing")

        # Create partial function with fixed parameters
        process_func = partial(
            _process_entry_wrapper,
            pdb_proc=self.pdb_proc,
            feat_enc=self.feat_enc,
            max_residues=self.max_residues,
            edge_cutoff=self.edge_cutoff
        )

        # Process in parallel
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, self.entries),
                total=len(self.entries),
                desc="Processing structures (parallel)"
            ))

        # Filter out None results
        self.processed = [r for r in results if r is not None]

    def _process_entry(self, entry: EnzymeEntry) -> Optional[Dict]:
        """
        Process a single enzyme entry into a training sample.

        Args:
            entry: EnzymeEntry to process

        Returns:
            Dictionary containing processed data, or None if processing fails
        """
        # Download PDB
        pdb_path = self.pdb_proc.download_pdb(entry.pdb_id)
        if not pdb_path:
            return None

        # Parse structure
        struct = self.pdb_proc.parse_pdb(pdb_path)
        if struct['num_residues'] > self.max_residues or struct['num_residues'] < 20:
            return None

        # Encode features
        encoded = self.feat_enc.encode_structure(struct, self.edge_cutoff)

        # Create labels
        cat_set = {(r.chain_id, r.residue_number) for r in entry.catalytic_residues}
        labels = [1 if (c, n) in cat_set else 0
                  for c, n, _ in encoded['residue_info']]

        # Role labels (multi-label)
        role_labels = torch.zeros(len(encoded['residue_info']), 9, dtype=torch.float32)
        for res in entry.catalytic_residues:
            for i, (c, n, _) in enumerate(encoded['residue_info']):
                if c == res.chain_id and n == res.residue_number:
                    for role in res.roles:
                        role_idx = ROLE_MAPPING.get(role, 8)
                        role_labels[i, role_idx] = 1.0

        # EC label
        ec1 = 0
        if entry.ec_numbers:
            try:
                ec1 = int(entry.ec_numbers[0].split('.')[0]) - 1
            except (ValueError, IndexError):
                pass

        # Convert to tensors with explicit dtypes
        node_features = torch.tensor(encoded['node_features'], dtype=torch.float32)
        edge_index = torch.tensor(encoded['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(encoded['edge_attr'], dtype=torch.float32)
        ca_coords = torch.tensor(encoded['ca_coords'], dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Masks (all ones for fully labelled M-CSA样本)
        labels_mask = torch.ones(len(encoded['residue_info']), dtype=torch.float32)
        role_mask = torch.ones(len(encoded['residue_info']), dtype=torch.float32)
        ec1_mask = torch.tensor(1, dtype=torch.float32)

        # Shape assertions for debugging
        num_nodes = node_features.shape[0]
        assert edge_index.shape[0] == 2, f"edge_index should have shape [2, num_edges], got {edge_index.shape}"
        assert labels_tensor.shape[0] == num_nodes, f"labels shape mismatch: {labels_tensor.shape[0]} vs {num_nodes}"
        assert role_labels.shape[0] == num_nodes, f"role_labels shape mismatch: {role_labels.shape[0]} vs {num_nodes}"

        return {
            'pdb_id': entry.pdb_id,
            'mcsa_id': entry.mcsa_id,
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'ca_coords': ca_coords,
            'labels': labels_tensor,
            'role_labels': role_labels,
            'ec1_label': ec1,
            'labels_mask': labels_mask,
            'role_mask': role_mask,
            'ec1_mask': ec1_mask,
            'residue_info': encoded['residue_info'],
            'sequence': encoded['sequence'],
        }

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.lazy_loading:
            return len(self.entries)
        else:
            return len(self.processed) if self.processed is not None else 0

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing sample data
        """
        if self.lazy_loading:
            # Process on-the-fly
            entry = self.entries[idx]
            data = self._process_entry(entry)
            if data is None:
                raise ValueError(f"Failed to process entry {idx}: {entry.pdb_id}")
            return data
        else:
            return self.processed[idx]

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Collate function for batching with proper tensor handling.

        This function creates proper batches with:
        - Consistent tensor dtypes and devices
        - Proper batch dimension handling
        - Shape assertions for validation

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched dictionary with proper tensor shapes
        """
        if len(batch) == 0:
            return {}

        # Collect metadata (non-tensor fields)
        pdb_ids = [item['pdb_id'] for item in batch]
        mcsa_ids = [item['mcsa_id'] for item in batch]
        residue_infos = [item['residue_info'] for item in batch]
        sequences = [item['sequence'] for item in batch]

        # Stack node-level features with batch indices
        node_features_list = []
        labels_list = []
        role_labels_list = []
        ca_coords_list = []
        labels_mask_list = []
        role_mask_list = []
        batch_indices = []

        # Stack edge-level features with offset edge indices
        edge_index_list = []
        edge_attr_list = []

        node_offset = 0
        for i, item in enumerate(batch):
            num_nodes = item['node_features'].shape[0]

            # Node features
            node_features_list.append(item['node_features'])
            labels_list.append(item['labels'])
            role_labels_list.append(item['role_labels'])
            ca_coords_list.append(item['ca_coords'])
            labels_mask_list.append(item.get('labels_mask', torch.ones(num_nodes, dtype=torch.float32)))
            role_mask_list.append(item.get('role_mask', torch.ones(num_nodes, dtype=torch.float32)))

            # Batch indices for this graph
            batch_indices.append(torch.full((num_nodes,), i, dtype=torch.long))

            # Edge features with offset
            edge_index = item['edge_index'] + node_offset
            edge_index_list.append(edge_index)
            edge_attr_list.append(item['edge_attr'])

            node_offset += num_nodes

        # Concatenate all tensors
        batched = {
            'pdb_id': pdb_ids,
            'mcsa_id': mcsa_ids,
            'node_features': torch.cat(node_features_list, dim=0),
            'edge_index': torch.cat(edge_index_list, dim=1),
            'edge_attr': torch.cat(edge_attr_list, dim=0),
            'ca_coords': torch.cat(ca_coords_list, dim=0),
            'labels': torch.cat(labels_list, dim=0),
            'role_labels': torch.cat(role_labels_list, dim=0),
            'labels_mask': torch.cat(labels_mask_list, dim=0),
            'role_mask': torch.cat(role_mask_list, dim=0),
            'batch': torch.cat(batch_indices, dim=0),
            'ec1_label': torch.tensor([item['ec1_label'] for item in batch], dtype=torch.long),
            'ec1_mask': torch.tensor([item.get('ec1_mask', 1) for item in batch], dtype=torch.float32),
            'residue_info': residue_infos,
            'sequence': sequences,
        }

        # Shape assertions
        total_nodes = batched['node_features'].shape[0]
        assert batched['labels'].shape[0] == total_nodes, "Labels shape mismatch"
        assert batched['role_labels'].shape[0] == total_nodes, "Role labels shape mismatch"
        assert batched['batch'].shape[0] == total_nodes, "Batch indices shape mismatch"
        assert batched['edge_index'].shape[0] == 2, "Edge index should have shape [2, num_edges]"

        return batched


# Helper function for multiprocessing
def _process_entry_wrapper(
    entry: EnzymeEntry,
    pdb_proc: PDBProcessor,
    feat_enc: FeatureEncoder,
    max_residues: int,
    edge_cutoff: float
) -> Optional[Dict]:
    """
    Wrapper function for processing entries in parallel.

    This function is defined at module level to be picklable for multiprocessing.

    Args:
        entry: EnzymeEntry to process
        pdb_proc: PDB processor instance
        feat_enc: Feature encoder instance
        max_residues: Maximum residues threshold
        edge_cutoff: Edge distance cutoff

    Returns:
        Processed data dictionary or None
    """
    try:
        # Create temporary dataset instance for processing
        temp_dataset = CatalyticSiteDataset.__new__(CatalyticSiteDataset)
        temp_dataset.pdb_proc = pdb_proc
        temp_dataset.feat_enc = feat_enc
        temp_dataset.max_residues = max_residues
        temp_dataset.edge_cutoff = edge_cutoff

        return temp_dataset._process_entry(entry)
    except Exception as e:
        logger.debug(f"Failed to process {entry.pdb_id}: {e}")
        return None
