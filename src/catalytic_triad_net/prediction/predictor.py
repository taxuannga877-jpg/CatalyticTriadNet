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

    def print_results(self, results: Dict, top_k: int = 10):
        """打印预测结果"""
        print(f"\n{'='*60}")
        print(f"PDB: {results['pdb_id']}")
        print(f"预测EC类别: EC {results['ec1_prediction']} (置信度: {results['ec1_confidence']:.3f})")
        print(f"{'='*60}")

        print(f"\n催化残基 (Top {top_k}):")
        print(f"{'序号':<6} {'链':<4} {'残基':<8} {'编号':<6} {'概率':<8}")
        print("-" * 40)

        for i, res in enumerate(results['catalytic_residues'][:top_k], 1):
            print(f"{i:<6} {res['chain']:<4} {res['resname']:<8} {res['resseq']:<6} {res['site_prob']:.4f}")

        if results.get('triads'):
            print(f"\n检测到的催化三联体: {len(results['triads'])} 个")
            for i, triad in enumerate(results['triads'][:3], 1):
                print(f"  三联体 {i}: {triad.get('type', 'unknown')}")

        if results.get('bimetallic_centers'):
            print(f"\n双金属中心: {len(results['bimetallic_centers'])} 个")

    def export_pymol(self, results: Dict, output_path: str, threshold: float = 0.5):
        """导出PyMOL脚本"""
        with open(output_path, 'w') as f:
            f.write("# PyMOL script for catalytic site visualization\n")
            f.write(f"# PDB: {results['pdb_id']}\n\n")

            f.write("# 加载结构\n")
            f.write(f"load {results['pdb_id']}.pdb\n")
            f.write("hide everything\n")
            f.write("show cartoon\n")
            f.write("color gray80\n\n")

            f.write("# 催化残基\n")
            for res in results['catalytic_residues']:
                if res['site_prob'] >= threshold:
                    f.write(f"select cat_{res['chain']}{res['resseq']}, "
                           f"chain {res['chain']} and resi {res['resseq']}\n")
                    f.write(f"show sticks, cat_{res['chain']}{res['resseq']}\n")
                    f.write(f"color red, cat_{res['chain']}{res['resseq']}\n")

            f.write("\n# 视图设置\n")
            f.write("zoom\n")
            f.write("bg_color white\n")

        print(f"✓ PyMOL脚本: {output_path}")

    def export_for_proteinmpnn(self, results: Dict, output_path: str):
        """导出ProteinMPNN格式"""
        import json

        # ProteinMPNN 需要固定的残基位置
        fixed_positions = []
        for res in results['catalytic_residues']:
            fixed_positions.append({
                'chain': res['chain'],
                'position': res['resseq'],
                'residue': res['resname']
            })

        mpnn_data = {
            'pdb_id': results['pdb_id'],
            'fixed_positions': fixed_positions,
            'design_mode': 'fixed_backbone',
            'temperature': 0.1
        }

        with open(output_path, 'w') as f:
            json.dump(mpnn_data, f, indent=2)

        print(f"✓ ProteinMPNN格式: {output_path}")

    def export_for_rfdiffusion(self, results: Dict, output_path: str):
        """导出RFdiffusion格式"""
        import json

        # RFdiffusion 需要约束信息
        constraints = []
        for res in results['catalytic_residues']:
            constraints.append({
                'chain': res['chain'],
                'residue': res['resseq'],
                'type': 'catalytic',
                'weight': float(res['site_prob'])
            })

        rfd_data = {
            'pdb_id': results['pdb_id'],
            'constraints': constraints,
            'diffusion_steps': 50,
            'guidance_scale': 1.0
        }

        with open(output_path, 'w') as f:
            json.dump(rfd_data, f, indent=2)

        print(f"✓ RFdiffusion格式: {output_path}")


# 兼容旧接口
CatalyticSiteInference = EnhancedCatalyticSiteInference
