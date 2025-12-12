#!/usr/bin/env python3
"""
催化位点预测推理模块（完整版）
整合：三联体检测、金属中心分析、氢键网络
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json

from .models import CatalyticTriadPredictor
from .analysis import TriadDetector, BimetallicCenterDetector, HydrogenBondAnalyzer
from .features import EnhancedFeatureEncoder, SubstrateAwareEncoder
from ..core.structure import PDBProcessor, FeatureEncoder, ROLE_MAPPING
from ..config import get_config

logger = logging.getLogger(__name__)


class EnhancedCatalyticSiteInference:
    """
    增强版催化位点推理器 - 整合所有分析功能

    提供催化位点预测、三联体检测、金属中心分析等功能。
    """

    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None,
                 device: Optional[str] = None):
        """
        初始化催化位点推理器。

        Args:
            model_path: 模型权重路径（可选）
            config: 配置字典（可选）
            device: 设备（'cuda' 或 'cpu'，可选）
        """
        # 从配置读取参数
        if config is None:
            global_config = get_config()
            self.config = global_config.to_dict()
        else:
            self.config = config

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        if model_path and Path(model_path).exists():
            self.model = CatalyticTriadPredictor.load_from_checkpoint(model_path, self.device)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No model provided, using random initialization")
            self.model = CatalyticTriadPredictor(config=self.config).to(self.device)
            self.model.eval()

        # 初始化子模块
        self.pdb_proc = PDBProcessor()
        self.feat_enc = FeatureEncoder()
        self.triad_detector = TriadDetector()
        self.bimetallic_detector = BimetallicCenterDetector()
        self.hbond_analyzer = HydrogenBondAnalyzer()
        self.substrate_encoder = SubstrateAwareEncoder()

        # 从配置读取阈值参数
        inference_config = self.config.get('inference', {})
        self.default_site_threshold = inference_config.get('site_threshold', 0.5)
        self.default_confidence_threshold = inference_config.get('confidence_threshold', 0.7)

        logger.info(f"EnhancedCatalyticSiteInference initialized on device: {self.device}")

    @torch.no_grad()
    def predict(self, pdb_path: str, site_threshold: Optional[float] = None) -> Dict:
        """
        完整预测流程。

        Args:
            pdb_path: PDB 文件路径或 PDB ID
            site_threshold: 催化位点阈值（如果为 None，使用默认值）

        Returns:
            Dict: 包含预测结果的字典
        """
        # 使用默认阈值
        if site_threshold is None:
            site_threshold = self.default_site_threshold

        # 验证阈值合法性
        if not 0.0 <= site_threshold <= 1.0:
            logger.warning(f"Invalid site_threshold {site_threshold}, using default {self.default_site_threshold}")
            site_threshold = self.default_site_threshold

        # 处理 PDB 路径
        pdb_path_obj = Path(pdb_path)
        if not pdb_path_obj.exists() and len(str(pdb_path)) == 4:
            logger.info(f"Downloading PDB: {pdb_path}")
            pdb_path_obj = self.pdb_proc.download_pdb(str(pdb_path))

        # 解析结构
        struct = self.pdb_proc.parse_pdb(pdb_path_obj)
        encoded = self.feat_enc.encode_structure(struct)

        # 准备输入
        x = torch.tensor(encoded['node_features'], dtype=torch.float32).to(self.device)
        ei = torch.tensor(encoded['edge_index'], dtype=torch.long).to(self.device)
        ea = torch.tensor(encoded['edge_attr'], dtype=torch.float32).to(self.device)

        # 模型推理
        outputs = self.model(x, ei, ea)
        site_probs = torch.sigmoid(outputs['site_logits']).cpu().numpy().flatten()
        ec1_probs = F.softmax(outputs['ec1_logits'], dim=-1).cpu().numpy()[0]
        pred_ec1 = int(np.argmax(ec1_probs)) + 1

        # 验证 EC 预测合法性
        if not 1 <= pred_ec1 <= 7:
            logger.warning(f"Invalid EC1 prediction {pred_ec1}, clamping to [1, 7]")
            pred_ec1 = max(1, min(7, pred_ec1))

        # 收集催化残基
        catalytic_residues = []
        for i, (chain, num, name) in enumerate(encoded['residue_info']):
            if site_probs[i] >= site_threshold:
                catalytic_residues.append({
                    'index': i, 'chain': chain, 'resseq': num,
                    'resname': name, 'site_prob': float(site_probs[i])
                })

        logger.info(f"Found {len(catalytic_residues)} catalytic residues above threshold {site_threshold}")

        # 分析三联体和金属中心
        triads = self.triad_detector.detect_triads(
            struct['residues'], encoded['ca_coords'], catalytic_residues, pred_ec1
        )

        metal_centers = [
            self.bimetallic_detector.analyze_coordination_geometry(
                m, struct['residues'], encoded['ca_coords']
            ) for m in struct.get('metals', [])
        ]

        bimetallic_centers = self.bimetallic_detector.detect_bimetallic_centers(
            struct.get('metals', []), struct['residues'], encoded['ca_coords']
        )

        return {
            'pdb_id': struct['pdb_id'],
            'ec1_prediction': pred_ec1,
            'ec1_confidence': float(ec1_probs[pred_ec1 - 1]),
            'catalytic_residues': catalytic_residues,
            'triads': triads,
            'metal_centers': metal_centers,
            'bimetallic_centers': bimetallic_centers,
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
