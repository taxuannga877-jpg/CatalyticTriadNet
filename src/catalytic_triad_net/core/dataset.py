#!/usr/bin/env python3
"""
数据集模块
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

from .data import EnzymeEntry
from .structure import PDBProcessor, FeatureEncoder, ROLE_MAPPING

logger = logging.getLogger(__name__)


class CatalyticSiteDataset(Dataset):
    """催化位点数据集"""

    def __init__(self, entries: List[EnzymeEntry], pdb_processor: PDBProcessor,
                 feature_encoder: FeatureEncoder, max_residues: int = 1000,
                 edge_cutoff: float = 10.0):
        self.entries = entries
        self.pdb_proc = pdb_processor
        self.feat_enc = feature_encoder
        self.max_residues = max_residues
        self.edge_cutoff = edge_cutoff
        self.processed = []
        self._preprocess()

    def _preprocess(self):
        """预处理所有条目"""
        for entry in tqdm(self.entries, desc="处理结构"):
            try:
                data = self._process_entry(entry)
                if data:
                    self.processed.append(data)
            except Exception as e:
                logger.debug(f"处理 {entry.pdb_id} 失败: {e}")

        logger.info(f"✓ 处理 {len(self.processed)}/{len(self.entries)} 个样本")

    def _process_entry(self, entry: EnzymeEntry) -> Optional[Dict]:
        """处理单个条目"""
        # 下载PDB
        pdb_path = self.pdb_proc.download_pdb(entry.pdb_id)
        if not pdb_path:
            return None

        # 解析结构
        struct = self.pdb_proc.parse_pdb(pdb_path)
        if struct['num_residues'] > self.max_residues or struct['num_residues'] < 20:
            return None

        # 编码特征
        encoded = self.feat_enc.encode_structure(struct, self.edge_cutoff)

        # 创建标签
        cat_set = {(r.chain_id, r.residue_number) for r in entry.catalytic_residues}
        labels = [1 if (c, n) in cat_set else 0
                  for c, n, _ in encoded['residue_info']]

        # 角色标签 (多标签)
        role_labels = torch.zeros(len(encoded['residue_info']), 9, dtype=torch.float32)
        for res in entry.catalytic_residues:
            for i, (c, n, _) in enumerate(encoded['residue_info']):
                if c == res.chain_id and n == res.residue_number:
                    for role in res.roles:
                        role_idx = ROLE_MAPPING.get(role, 8)
                        role_labels[i, role_idx] = 1.0

        # EC标签
        ec1 = 0
        if entry.ec_numbers:
            try:
                ec1 = int(entry.ec_numbers[0].split('.')[0]) - 1
            except (ValueError, IndexError):
                pass

        return {
            'pdb_id': entry.pdb_id,
            'mcsa_id': entry.mcsa_id,
            'node_features': torch.tensor(encoded['node_features'], dtype=torch.float32),
            'edge_index': torch.tensor(encoded['edge_index'], dtype=torch.long),
            'edge_attr': torch.tensor(encoded['edge_attr'], dtype=torch.float32),
            'ca_coords': torch.tensor(encoded['ca_coords'], dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'role_labels': role_labels,
            'ec1_label': ec1,
            'residue_info': encoded['residue_info'],
            'sequence': encoded['sequence'],
        }

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        return self.processed[idx]

    @staticmethod
    def collate_fn(batch):
        """批次整理函数"""
        return batch
