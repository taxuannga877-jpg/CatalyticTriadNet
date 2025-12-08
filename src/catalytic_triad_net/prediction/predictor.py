#!/usr/bin/env python3
"""
催化位点预测推理模块（完整版）
整合：三联体检测、金属中心分析、氢键网络
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

from .models import CatalyticTriadPredictor
from .analysis import TriadDetector, BimetallicCenterDetector, HydrogenBondAnalyzer
from .features import EnhancedFeatureEncoder, SubstrateAwareEncoder
from ..core.structure import PDBProcessor, FeatureEncoder, ROLE_MAPPING

logger = logging.getLogger(__name__)


class EnhancedCatalyticSiteInference:
    """增强版催化位点推理器 - 整合所有分析功能"""

    def __init__(self, model_path: str = None, config: Dict = None, device: str = None):
        self.config = config or {}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path and Path(model_path).exists():
            self.model = CatalyticTriadPredictor.load_from_checkpoint(model_path, self.device)
        else:
            logger.warning("未提供模型，使用随机初始化")
            self.model = CatalyticTriadPredictor().to(self.device)
            self.model.eval()

        self.pdb_proc = PDBProcessor()
        self.feat_enc = FeatureEncoder()
        self.triad_detector = TriadDetector()
        self.bimetallic_detector = BimetallicCenterDetector()
        self.hbond_analyzer = HydrogenBondAnalyzer()
        self.substrate_encoder = SubstrateAwareEncoder()

    @torch.no_grad()
    def predict(self, pdb_path: str, site_threshold: float = 0.5) -> Dict:
        """完整预测"""
        pdb_path = Path(pdb_path)
        if not pdb_path.exists() and len(str(pdb_path)) == 4:
            pdb_path = self.pdb_proc.download_pdb(str(pdb_path))
        
        struct = self.pdb_proc.parse_pdb(pdb_path)
        encoded = self.feat_enc.encode_structure(struct)

        x = torch.tensor(encoded['node_features'], dtype=torch.float32).to(self.device)
        ei = torch.tensor(encoded['edge_index'], dtype=torch.long).to(self.device)
        ea = torch.tensor(encoded['edge_attr'], dtype=torch.float32).to(self.device)

        outputs = self.model(x, ei, ea)
        site_probs = torch.sigmoid(outputs['site_logits']).cpu().numpy().flatten()
        ec1_probs = F.softmax(outputs['ec1_logits'], dim=-1).cpu().numpy()[0]
        pred_ec1 = int(np.argmax(ec1_probs)) + 1

        catalytic_residues = []
        for i, (chain, num, name) in enumerate(encoded['residue_info']):
            if site_probs[i] >= site_threshold:
                catalytic_residues.append({
                    'index': i, 'chain': chain, 'resseq': num,
                    'resname': name, 'site_prob': float(site_probs[i])
                })

        return {
            'pdb_id': struct['pdb_id'],
            'ec1_prediction': pred_ec1,
            'ec1_confidence': float(ec1_probs[pred_ec1 - 1]),
            'catalytic_residues': catalytic_residues,
            'triads': self.triad_detector.detect_triads(
                struct['residues'], encoded['ca_coords'], catalytic_residues, pred_ec1
            ),
            'metal_centers': [self.bimetallic_detector.analyze_coordination_geometry(
                m, struct['residues'], encoded['ca_coords']
            ) for m in struct.get('metals', [])],
            'bimetallic_centers': self.bimetallic_detector.detect_bimetallic_centers(
                struct.get('metals', []), struct['residues'], encoded['ca_coords']
            ),
        }


# 兼容旧接口
CatalyticSiteInference = EnhancedCatalyticSiteInference
