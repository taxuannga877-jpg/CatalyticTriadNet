#!/usr/bin/env python3
"""
Swiss-Prot dataset wrapper for transfer learning.

Only EC标签可靠，催化位点/角色标签缺失，因此通过掩码屏蔽位点/角色损失。
"""

import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Union
import logging

from .structure import PDBProcessor, FeatureEncoder
from .swissprot_data import SwissProtEntry
from ..config import get_config, Config

logger = logging.getLogger(__name__)


class SwissProtDataset(Dataset):
    """
    Swiss-Prot 数据集（仅EC标签有效）

    生成的样本格式与 CatalyticSiteDataset 对齐，但：
    - labels_mask=0，role_mask=0（不参与位点/角色损失）
    - ec1_mask=1（参与EC损失）
    """

    def __init__(
        self,
        entries: List[Union[SwissProtEntry, Dict]],
        pdb_processor: PDBProcessor,
        feature_encoder: FeatureEncoder,
        max_residues: Optional[int] = None,
        edge_cutoff: Optional[float] = None,
        config: Optional[Config] = None
    ):
        self.entries = entries
        self.pdb_proc = pdb_processor
        self.feat_enc = feature_encoder
        self.config = config or get_config()
        self.max_residues = max_residues or self.config.get('data.max_residues', 1000)
        self.edge_cutoff = edge_cutoff or self.config.get('data.edge_cutoff', 8.0)

        self.processed: List[Dict] = []
        self._preprocess()

    def _preprocess(self):
        for entry in self.entries:
            try:
                data = self._process_entry(entry)
                if data:
                    self.processed.append(data)
            except Exception as e:
                logger.debug(f"Failed to process Swiss-Prot entry: {e}")

        logger.info(f"SwissProtDataset processed {len(self.processed)}/{len(self.entries)} samples")

    def _get_ec1(self, entry) -> int:
        ec_number = None
        if isinstance(entry, SwissProtEntry):
            ec_number = entry.ec_number
        elif isinstance(entry, dict):
            ec_number = entry.get('ec_number') or entry.get('ec') or entry.get('ec_numbers')

        if isinstance(ec_number, list):
            ec_number = ec_number[0] if ec_number else None

        ec1 = 0
        if ec_number:
            try:
                ec1 = int(str(ec_number).split('.')[0]) - 1
            except (ValueError, IndexError):
                ec1 = 0
        return ec1

    def _get_pdb_ids(self, entry) -> List[str]:
        if isinstance(entry, SwissProtEntry):
            return entry.pdb_ids or []
        if isinstance(entry, dict):
            return entry.get('pdb_ids', []) or entry.get('pdb', []) or []
        return []

    def _process_entry(self, entry) -> Optional[Dict]:
        pdb_ids = self._get_pdb_ids(entry)
        if not pdb_ids:
            return None

        pdb_id = pdb_ids[0]
        pdb_path = self.pdb_proc.download_pdb(pdb_id)
        if not pdb_path:
            return None

        struct = self.pdb_proc.parse_pdb(pdb_path)
        if struct['num_residues'] > self.max_residues or struct['num_residues'] < 20:
            return None

        encoded = self.feat_enc.encode_structure(struct, self.edge_cutoff)
        n = len(encoded['residue_info'])

        # 空标签 + 掩码
        labels = torch.zeros(n, dtype=torch.long)
        role_labels = torch.zeros(n, 9, dtype=torch.float32)
        labels_mask = torch.zeros(n, dtype=torch.float32)
        role_mask = torch.zeros(n, dtype=torch.float32)
        ec1_label = torch.tensor(self._get_ec1(entry), dtype=torch.long)
        ec1_mask = torch.tensor(1, dtype=torch.float32)

        return {
            'pdb_id': pdb_id,
            'mcsa_id': None,
            'node_features': torch.tensor(encoded['node_features'], dtype=torch.float32),
            'edge_index': torch.tensor(encoded['edge_index'], dtype=torch.long),
            'edge_attr': torch.tensor(encoded['edge_attr'], dtype=torch.float32),
            'ca_coords': torch.tensor(encoded['ca_coords'], dtype=torch.float32),
            'labels': labels,
            'labels_mask': labels_mask,
            'role_labels': role_labels,
            'role_mask': role_mask,
            'ec1_label': ec1_label,
            'ec1_mask': ec1_mask,
            'residue_info': encoded['residue_info'],
            'sequence': encoded.get('sequence', '')
        }

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx: int) -> Dict:
        return self.processed[idx]

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        # 重用 CatalyticSiteDataset 的 collate 逻辑
        from .dataset import CatalyticSiteDataset
        return CatalyticSiteDataset.collate_fn(batch)
