#!/usr/bin/env python3
"""
高级特征编码模块：电子特征、底物感知、保守性分析
"""

import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 氨基酸理化性质（扩展版）
# =============================================================================

AA_PROPERTIES = {
    'ALA': {'hydro': 1.8, 'volume': 88.6, 'charge': 0, 'polar': 0, 'aromatic': 0,
            'pka': 0, 'electronegativity': 0.0, 'polarizability': 0.046},
    'ARG': {'hydro': -4.5, 'volume': 173.4, 'charge': 1, 'polar': 1, 'aromatic': 0,
            'pka': 12.5, 'electronegativity': 0.5, 'polarizability': 0.291},
    'ASN': {'hydro': -3.5, 'volume': 114.1, 'charge': 0, 'polar': 1, 'aromatic': 0,
            'pka': 0, 'electronegativity': 0.2, 'polarizability': 0.134},
    'ASP': {'hydro': -3.5, 'volume': 111.1, 'charge': -1, 'polar': 1, 'aromatic': 0,
            'pka': 3.9, 'electronegativity': 0.6, 'polarizability': 0.105},
    'CYS': {'hydro': 2.5, 'volume': 108.5, 'charge': 0, 'polar': 0, 'aromatic': 0,
            'pka': 8.3, 'electronegativity': 0.3, 'polarizability': 0.128},
    'GLN': {'hydro': -3.5, 'volume': 143.8, 'charge': 0, 'polar': 1, 'aromatic': 0,
            'pka': 0, 'electronegativity': 0.2, 'polarizability': 0.180},
    'GLU': {'hydro': -3.5, 'volume': 138.4, 'charge': -1, 'polar': 1, 'aromatic': 0,
            'pka': 4.1, 'electronegativity': 0.6, 'polarizability': 0.151},
    'GLY': {'hydro': -0.4, 'volume': 60.1, 'charge': 0, 'polar': 0, 'aromatic': 0,
            'pka': 0, 'electronegativity': 0.0, 'polarizability': 0.000},
    'HIS': {'hydro': -3.2, 'volume': 153.2, 'charge': 0.5, 'polar': 1, 'aromatic': 1,
            'pka': 6.0, 'electronegativity': 0.4, 'polarizability': 0.230},
    'ILE': {'hydro': 4.5, 'volume': 166.7, 'charge': 0, 'polar': 0, 'aromatic': 0,
            'pka': 0, 'electronegativity': 0.0, 'polarizability': 0.186},
    'LEU': {'hydro': 3.8, 'volume': 166.7, 'charge': 0, 'polar': 0, 'aromatic': 0,
            'pka': 0, 'electronegativity': 0.0, 'polarizability': 0.186},
    'LYS': {'hydro': -3.9, 'volume': 168.6, 'charge': 1, 'polar': 1, 'aromatic': 0,
            'pka': 10.5, 'electronegativity': 0.4, 'polarizability': 0.243},
    'MET': {'hydro': 1.9, 'volume': 162.9, 'charge': 0, 'polar': 0, 'aromatic': 0,
            'pka': 0, 'electronegativity': 0.1, 'polarizability': 0.221},
    'PHE': {'hydro': 2.8, 'volume': 189.9, 'charge': 0, 'polar': 0, 'aromatic': 1,
            'pka': 0, 'electronegativity': 0.1, 'polarizability': 0.290},
    'PRO': {'hydro': -1.6, 'volume': 112.7, 'charge': 0, 'polar': 0, 'aromatic': 0,
            'pka': 0, 'electronegativity': 0.0, 'polarizability': 0.131},
    'SER': {'hydro': -0.8, 'volume': 89.0, 'charge': 0, 'polar': 1, 'aromatic': 0,
            'pka': 0, 'electronegativity': 0.3, 'polarizability': 0.062},
    'THR': {'hydro': -0.7, 'volume': 116.1, 'charge': 0, 'polar': 1, 'aromatic': 0,
            'pka': 0, 'electronegativity': 0.2, 'polarizability': 0.108},
    'TRP': {'hydro': -0.9, 'volume': 227.8, 'charge': 0, 'polar': 0, 'aromatic': 1,
            'pka': 0, 'electronegativity': 0.2, 'polarizability': 0.409},
    'TYR': {'hydro': -1.3, 'volume': 193.6, 'charge': 0, 'polar': 1, 'aromatic': 1,
            'pka': 10.1, 'electronegativity': 0.3, 'polarizability': 0.298},
    'VAL': {'hydro': 4.2, 'volume': 140.0, 'charge': 0, 'polar': 0, 'aromatic': 0,
            'pka': 0, 'electronegativity': 0.0, 'polarizability': 0.140}
}


# =============================================================================
# 电子特征编码器
# =============================================================================

class ElectronicFeatureEncoder:
    """
    电子结构特征编码器
    参考: xtb (https://github.com/grimme-lab/xtb)
    """

    # 氨基酸侧链部分电荷 (预计算, 基于GFN2-xTB)
    AA_PARTIAL_CHARGES = {
        'ALA': {'CA': 0.15, 'CB': -0.18},
        'ARG': {'CA': 0.12, 'CZ': 0.64, 'NH1': -0.36, 'NH2': -0.36},
        'ASN': {'CA': 0.10, 'CG': 0.55, 'OD1': -0.50, 'ND2': -0.30},
        'ASP': {'CA': 0.08, 'CG': 0.62, 'OD1': -0.55, 'OD2': -0.55},
        'CYS': {'CA': 0.12, 'SG': -0.23},
        'GLN': {'CA': 0.10, 'CD': 0.52, 'OE1': -0.48, 'NE2': -0.32},
        'GLU': {'CA': 0.08, 'CD': 0.60, 'OE1': -0.54, 'OE2': -0.54},
        'GLY': {'CA': 0.20},
        'HIS': {'CA': 0.10, 'ND1': -0.20, 'NE2': -0.20, 'CE1': 0.25},
        'ILE': {'CA': 0.12, 'CB': -0.08, 'CG1': -0.15},
        'LEU': {'CA': 0.12, 'CB': -0.10, 'CG': -0.05},
        'LYS': {'CA': 0.10, 'NZ': -0.30},
        'MET': {'CA': 0.12, 'SD': -0.10, 'CE': -0.15},
        'PHE': {'CA': 0.10, 'CG': -0.05, 'CZ': -0.08},
        'PRO': {'CA': 0.08, 'N': -0.25},
        'SER': {'CA': 0.12, 'OG': -0.38},
        'THR': {'CA': 0.10, 'OG1': -0.36, 'CG2': -0.18},
        'TRP': {'CA': 0.10, 'NE1': -0.22, 'CE2': 0.05},
        'TYR': {'CA': 0.10, 'OH': -0.40, 'CZ': 0.15},
        'VAL': {'CA': 0.12, 'CB': -0.05},
    }

    def __init__(self, xtb_path: str = None):
        self.xtb_path = xtb_path
        self.xtb_available = self._check_xtb()

    def _check_xtb(self) -> bool:
        """检查xTB是否可用"""
        if self.xtb_path and Path(self.xtb_path).exists():
            return True
        try:
            result = subprocess.run(['xtb', '--version'], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def compute_features(self, residue: Dict, all_coords: np.ndarray = None) -> np.ndarray:
        """
        计算单个残基的电子特征 (6维)

        特征:
        - 侧链净电荷
        - 最大部分电荷
        - 电负性
        - 极化率
        - 氧化还原活性
        - 反应活性指数
        """
        res_name = residue.get('name', 'ALA')
        props = AA_PROPERTIES.get(res_name, AA_PROPERTIES['ALA'])
        charges = self.AA_PARTIAL_CHARGES.get(res_name, {})

        # 侧链净电荷
        sidechain_charge = sum(charges.values()) if charges else 0.0

        # 最大部分电荷绝对值
        max_partial = max(abs(v) for v in charges.values()) if charges else 0.0

        # 电负性
        electronegativity = props.get('electronegativity', 0.0)

        # 极化率
        polarizability = props.get('polarizability', 0.0)

        # 氧化还原活性
        pka = props.get('pka', 7.0)
        charge = props.get('charge', 0)
        redox_activity = abs(charge) * 0.5 + (1.0 - abs(pka - 7.0) / 7.0) * 0.5 if pka > 0 else 0.0

        # 反应活性指数
        reactivity = (electronegativity + polarizability) / 2.0

        return np.array([
            sidechain_charge,
            max_partial,
            electronegativity,
            polarizability,
            redox_activity,
            reactivity
        ], dtype=np.float32)


# =============================================================================
# 底物感知编码器
# =============================================================================

class SubstrateAwareEncoder:
    """
    底物感知特征编码器
    参考: P2Rank, fpocket
    """

    def __init__(self, pocket_probe_radius: float = 3.0):
        self.pocket_probe_radius = pocket_probe_radius

    def compute_features(self, residue_idx: int, ca_coords: np.ndarray,
                         ligands: List[Dict], metals: List[Dict]) -> np.ndarray:
        """
        计算底物感知特征 (6维)

        特征:
        - 到最近配体的距离
        - 到最近配体的归一化距离
        - 配体邻居数量
        - 是否在结合口袋内
        - 口袋暴露度
        - 底物相互作用潜力
        """
        coord = ca_coords[residue_idx]
        features = np.zeros(6, dtype=np.float32)

        # 合并配体和金属
        all_ligands = ligands + metals

        if not all_ligands:
            features[0] = 999.0
            features[1] = 1.0
            return features

        ligand_coords = np.array([l['coord'] for l in all_ligands])

        # 到最近配体的距离
        distances = np.linalg.norm(ligand_coords - coord, axis=1)
        min_dist = np.min(distances)
        features[0] = min_dist
        features[1] = min(min_dist / 20.0, 1.0)

        # 配体邻居数量 (8Å内)
        features[2] = np.sum(distances < 8.0) / max(len(all_ligands), 1)

        # 是否在结合口袋内 (6Å内有配体)
        features[3] = 1.0 if min_dist < 6.0 else 0.0

        # 口袋暴露度
        n_residues = len(ca_coords)
        all_dists = np.linalg.norm(ca_coords - coord, axis=1)
        local_density = np.sum(all_dists < 8.0) / n_residues
        features[4] = 1.0 - local_density

        # 底物相互作用潜力
        features[5] = np.sum(np.exp(-distances / 5.0))

        return features

    def detect_binding_pockets(self, ca_coords: np.ndarray,
                               ligands: List[Dict]) -> List[Dict]:
        """简化版口袋检测"""
        if not ligands:
            return []

        pockets = []
        for i, lig in enumerate(ligands):
            lig_coord = lig['coord']

            distances = np.linalg.norm(ca_coords - lig_coord, axis=1)
            pocket_residues = np.where(distances < 8.0)[0].tolist()

            if len(pocket_residues) >= 5:
                pocket_coords = ca_coords[pocket_residues]
                centroid = pocket_coords.mean(axis=0)

                pockets.append({
                    'id': i,
                    'ligand': lig,
                    'residue_indices': pocket_residues,
                    'centroid': centroid,
                    'volume_estimate': len(pocket_residues) * 50.0,
                })

        return pockets


# =============================================================================
# 保守性分析器
# =============================================================================

class ConservationAnalyzer:
    """
    序列保守性分析接口
    参考: ConSurf, EVcouplings
    """

    # 简化的BLOSUM62保守性
    CONSERVATION_SCORES = {
        'ALA': 0.4, 'ARG': 0.6, 'ASN': 0.5, 'ASP': 0.7, 'CYS': 0.8,
        'GLN': 0.5, 'GLU': 0.7, 'GLY': 0.9, 'HIS': 0.8, 'ILE': 0.4,
        'LEU': 0.4, 'LYS': 0.6, 'MET': 0.5, 'PHE': 0.5, 'PRO': 0.7,
        'SER': 0.6, 'THR': 0.5, 'TRP': 0.9, 'TYR': 0.6, 'VAL': 0.4,
    }

    def __init__(self):
        self.msa_cache = {}

    def get_conservation_score(self, residue_name: str) -> float:
        """
        获取保守性分数 (简化版)

        完整实现需要:
        1. 使用BLAST/HHblits搜索同源序列
        2. 构建多序列比对
        3. 计算位置特异性保守性
        """
        return self.CONSERVATION_SCORES.get(residue_name, 0.5)

    def compute_features(self, residue_name: str) -> float:
        """计算保守性特征"""
        return self.get_conservation_score(residue_name)


# =============================================================================
# 增强版特征编码器（整合所有高级特征）
# =============================================================================

class EnhancedFeatureEncoder:
    """
    增强版特征编码器 - 整合所有子模块

    节点特征 (48维):
    - AA one-hot: 20
    - 理化性质: 8
    - 空间特征: 5
    - 金属环境: 3
    - 电子特征: 6
    - 底物感知: 6
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # 子编码器
        ext_tools = self.config.get('external_tools', {})
        self.electronic_encoder = ElectronicFeatureEncoder(ext_tools.get('xtb_path'))
        self.substrate_encoder = SubstrateAwareEncoder()
        self.conservation_analyzer = ConservationAnalyzer()

    def encode_metal_env(self, idx: int, coords: np.ndarray, metals: List[Dict],
                         metal_shell_cutoff: float = 5.0,
                         metal_neighbor_cutoff: float = 8.0) -> np.ndarray:
        """单个残基的金属环境特征 (3维)"""
        features = np.zeros(3, dtype=np.float32)

        if not metals:
            features[0] = 1.0
            return features

        coord = coords[idx]
        metal_coords = np.array([m['coord'] for m in metals])
        dists = np.linalg.norm(metal_coords - coord, axis=1)

        min_dist = np.min(dists)
        features[0] = min(min_dist / 20.0, 1.0)
        features[1] = np.sum(dists < metal_neighbor_cutoff) / max(len(metals), 1)
        features[2] = 1.0 if min_dist < metal_shell_cutoff else 0.0

        return features

    def compute_hbond_edge_features(self, i: int, j: int, hbond_pairs: set,
                                     hbonds: List[Dict]) -> np.ndarray:
        """氢键边特征 (3维)"""
        has_hbond = 1.0 if (i, j) in hbond_pairs else 0.0

        hbond_dist = 0.0
        hbond_strength = 0.0
        for hb in hbonds:
            if (hb['donor_residue'] == i and hb['acceptor_residue'] == j) or \
               (hb['donor_residue'] == j and hb['acceptor_residue'] == i):
                hbond_dist = hb['distance'] / 3.5
                hbond_strength = max(0, 1.0 - (hb['distance'] - 2.5) / 1.0)
                break

        return np.array([has_hbond, hbond_dist, hbond_strength], dtype=np.float32)
