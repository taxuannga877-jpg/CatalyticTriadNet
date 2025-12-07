#!/usr/bin/env python3
"""
催化三联体识别与反应预测模型 V2.0
==========================================================

新增功能 (相比V1):
1. 电子特征编码 - 部分电荷、电负性 (参考xtb)
2. 底物感知特征 - 配体距离、结合口袋 (参考P2Rank, fpocket)
3. 智能三联体检测 - 模式匹配 + 几何约束
4. 双金属中心检测 - M-M距离、桥连配体 (参考MetalCoord)
5. 氢键网络分析 (参考MDAnalysis)
6. 保守性分数接口 (参考ConSurf/EVcouplings)
7. 下游设计接口 (ProteinMPNN/RFdiffusion格式)

GitHub参考项目:
- xtb: https://github.com/grimme-lab/xtb
- P2Rank: https://github.com/rdk/p2rank
- fpocket: https://github.com/Discngine/fpocket
- MetalCoord: https://github.com/sb-ncbr/MetalCoord
- MDAnalysis: https://github.com/MDAnalysis/mdanalysis
- ProteinMPNN: https://github.com/dauparas/ProteinMPNN
- RFdiffusion: https://github.com/RosettaCommons/RFdiffusion
"""

import os
import sys
import json
import math
import argparse
import subprocess
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union, Any, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader

try:
    from torch_geometric.data import Data, Dataset, DataLoader
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    Data = object

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 常量定义
# =============================================================================

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

AA_LIST = list(AA_3TO1.keys())

# 氨基酸扩展理化性质 (增加电子相关)
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

# 催化残基先验
CATALYTIC_PRIOR = {
    'HIS': 0.25, 'ASP': 0.20, 'GLU': 0.18, 'CYS': 0.15, 'SER': 0.12,
    'LYS': 0.10, 'ARG': 0.08, 'TYR': 0.06, 'ASN': 0.04, 'THR': 0.03
}

# ★ 新增: 经典催化三联体模式 (参考M-CSA数据库)
TRIAD_PATTERNS = {
    'serine_protease': {
        'residues': [('SER', 'nucleophile'), ('HIS', 'general_base'), ('ASP', 'electrostatic')],
        'distances': {'SER-HIS': (2.5, 4.0), 'HIS-ASP': (2.5, 4.0), 'SER-ASP': (6.0, 10.0)},
        'ec_class': [3, 4],  # 水解酶, 裂解酶
    },
    'cysteine_protease': {
        'residues': [('CYS', 'nucleophile'), ('HIS', 'general_base'), ('ASN', 'electrostatic')],
        'distances': {'CYS-HIS': (3.0, 4.5), 'HIS-ASN': (2.5, 4.0), 'CYS-ASN': (6.0, 10.0)},
        'ec_class': [3],
    },
    'cysteine_protease_asp': {
        'residues': [('CYS', 'nucleophile'), ('HIS', 'general_base'), ('ASP', 'electrostatic')],
        'distances': {'CYS-HIS': (3.0, 4.5), 'HIS-ASP': (2.5, 4.0), 'CYS-ASP': (6.0, 10.0)},
        'ec_class': [3],
    },
    'threonine_protease': {
        'residues': [('THR', 'nucleophile'), ('LYS', 'general_base'), ('ASP', 'electrostatic')],
        'distances': {'THR-LYS': (3.0, 5.0), 'LYS-ASP': (3.0, 5.0), 'THR-ASP': (5.0, 9.0)},
        'ec_class': [3],
    },
    'aspartic_protease': {  # 双Asp机制
        'residues': [('ASP', 'nucleophile'), ('ASP', 'general_acid'), ('THR', 'stabilizer')],
        'distances': {'ASP-ASP': (2.5, 4.0), 'ASP-THR': (3.0, 5.0)},
        'ec_class': [3],
    },
}

# ★ 新增: 金属离子配位几何 (参考MetalCoord)
METAL_COORDINATION = {
    'ZN': {'coord_number': [4, 5, 6], 'geometry': ['tetrahedral', 'trigonal_bipyramidal', 'octahedral'],
           'common_ligands': ['HIS', 'CYS', 'ASP', 'GLU'], 'ideal_distance': 2.0},
    'MG': {'coord_number': [6], 'geometry': ['octahedral'],
           'common_ligands': ['ASP', 'GLU', 'SER', 'THR'], 'ideal_distance': 2.1},
    'MN': {'coord_number': [6], 'geometry': ['octahedral'],
           'common_ligands': ['HIS', 'ASP', 'GLU'], 'ideal_distance': 2.2},
    'FE': {'coord_number': [4, 5, 6], 'geometry': ['tetrahedral', 'square_pyramidal', 'octahedral'],
           'common_ligands': ['HIS', 'CYS', 'TYR'], 'ideal_distance': 2.0},
    'CU': {'coord_number': [4, 5], 'geometry': ['square_planar', 'trigonal_bipyramidal'],
           'common_ligands': ['HIS', 'CYS', 'MET'], 'ideal_distance': 2.0},
    'CA': {'coord_number': [6, 7, 8], 'geometry': ['octahedral', 'pentagonal_bipyramidal'],
           'common_ligands': ['ASP', 'GLU', 'ASN'], 'ideal_distance': 2.4},
}

# 双金属中心模式
BIMETALLIC_PATTERNS = {
    'phosphodiesterase': {
        'metals': ['MG', 'MG'], 'distance_range': (3.4, 4.2),
        'bridging_ligands': ['ASP', 'GLU', 'HOH'],
        'ec_class': [3, 1],
    },
    'purple_acid_phosphatase': {
        'metals': ['FE', 'ZN'], 'distance_range': (3.0, 3.5),
        'bridging_ligands': ['ASP', 'HOH'],
        'ec_class': [3],
    },
    'urease': {
        'metals': ['NI', 'NI'], 'distance_range': (3.5, 3.7),
        'bridging_ligands': ['LYS', 'HOH'],  # carbamylated lysine
        'ec_class': [3],
    },
    'metallo_beta_lactamase': {
        'metals': ['ZN', 'ZN'], 'distance_range': (3.4, 4.5),
        'bridging_ligands': ['ASP', 'HIS', 'HOH'],
        'ec_class': [3],
    },
}

METAL_NAMES = {"ZN", "MG", "MN", "FE", "CU", "CO", "NI", "CA", "NA", "K", "CD", "MO", "W", "V",
               "FE2", "FE3", "ZN2", "MG2", "MN2", "CU2", "CO2", "NI2", "CA2"}
WATER_NAMES = {"HOH", "WAT", "H2O"}

ROLE_MAPPING = {
    0: "non_catalytic", 1: "nucleophile", 2: "general_base", 3: "general_acid",
    4: "metal_ligand", 5: "transition_state_stabilizer", 6: "proton_donor",
    7: "proton_acceptor", 8: "electrostatic_stabilizer", 9: "other"
}
ROLE_NAME_TO_ID = {v: k for k, v in ROLE_MAPPING.items()}


# =============================================================================
# 配置
# =============================================================================

DEFAULT_CONFIG = {
    'data_dir': './data',
    'cache_dir': './data/mcsa_cache',
    'pdb_dir': './data/pdb_structures',
    'model_dir': './models',
    'results_dir': './results',
    
    'model': {
        'node_dim': 48,  # 扩展: 28 + 5 + 3 + 6(electronic) + 6(substrate)
        'edge_dim': 14,  # 扩展: 8 + 3 + 3(hbond_detail)
        'hidden_dim': 256,
        'num_gnn_layers': 6,
        'num_heads': 8,
        'dropout': 0.2,
        'use_esm': False,
        'esm_dim': 1280,
    },
    
    'training': {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 4,
        'epochs': 100,
        'patience': 15,
        'pos_weight': 10.0,
        'focal_gamma': 2.0,
    },
    
    'data': {
        'max_residues': 1000,
        'edge_cutoff': 10.0,
        'metal_shell_cutoff': 5.0,
        'metal_neighbor_cutoff': 8.0,
        'hbond_cutoff': 3.5,
        'hbond_angle_cutoff': 120.0,
    },
    
    'ec_prediction': {
        'num_ec1_classes': 7,
        'num_ec2_classes': 70,
        'num_ec3_classes': 300,
    },
    
    # ★ 新增: 外部工具路径
    'external_tools': {
        'xtb_path': None,  # xTB可执行文件路径
        'fpocket_path': None,  # fpocket可执行文件路径
        'p2rank_path': None,  # P2Rank jar路径
    }
}

def get_config():
    return DEFAULT_CONFIG.copy()


# =============================================================================
# 1. 电子特征编码器 (参考xtb: https://github.com/grimme-lab/xtb)
# =============================================================================

class ElectronicFeatureEncoder:
    """
    电子结构特征编码器
    
    参考项目:
    - xtb (https://github.com/grimme-lab/xtb): GFN2-xTB快速计算部分电荷
    - RDKit: 分子描述符计算
    
    功能:
    - 部分电荷 (Gasteiger或xTB)
    - 电负性
    - 极化率
    - 氧化还原活性指标
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
        'LYS': {'CA': 0.10, 'NZ': -0.30},  # protonated: +0.33
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
        except:
            return False
    
    def compute_features(self, residue: Dict, all_coords: np.ndarray = None) -> np.ndarray:
        """
        计算单个残基的电子特征 (6维)
        
        特征:
        - 侧链净电荷
        - 最大部分电荷
        - 电负性
        - 极化率
        - 氧化还原活性 (基于pKa和电荷)
        - 反应活性指数 (基于HOMO/LUMO近似)
        """
        res_name = residue.get('name', 'ALA')
        props = AA_PROPERTIES.get(res_name, AA_PROPERTIES['ALA'])
        charges = self.AA_PARTIAL_CHARGES.get(res_name, {})
        
        # 侧链净电荷
        sidechain_charge = sum(charges.values()) if charges else 0.0
        
        # 最大部分电荷绝对值
        max_partial = max(abs(v) for v in charges.values()) if charges else 0.0
        
        # 电负性 (0-1归一化)
        electronegativity = props.get('electronegativity', 0.0)
        
        # 极化率 (0-1归一化)
        polarizability = props.get('polarizability', 0.0)
        
        # 氧化还原活性: 基于pKa和电荷
        pka = props.get('pka', 7.0)
        charge = props.get('charge', 0)
        redox_activity = abs(charge) * 0.5 + (1.0 - abs(pka - 7.0) / 7.0) * 0.5 if pka > 0 else 0.0
        
        # 反应活性指数: HOMO/LUMO近似 (基于电负性和极化率)
        reactivity = (electronegativity + polarizability) / 2.0
        
        return np.array([
            sidechain_charge,
            max_partial,
            electronegativity,
            polarizability,
            redox_activity,
            reactivity
        ], dtype=np.float32)
    
    def compute_xtb_charges(self, pdb_path: str) -> Optional[Dict[int, float]]:
        """
        使用xTB计算精确部分电荷 (可选)
        
        需要安装xtb: conda install -c conda-forge xtb
        """
        if not self.xtb_available:
            return None
        
        try:
            # 运行xTB计算
            result = subprocess.run(
                ['xtb', str(pdb_path), '--chrg', '0', '--gfn', '2', '--json'],
                capture_output=True, timeout=300, cwd=Path(pdb_path).parent
            )
            
            # 解析结果
            json_file = Path(pdb_path).parent / 'xtbout.json'
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)
                return {i: q for i, q in enumerate(data.get('partial charges', []))}
        except Exception as e:
            logger.warning(f"xTB计算失败: {e}")
        
        return None


# =============================================================================
# 2. 底物/配体感知编码器 (参考P2Rank, fpocket)
# =============================================================================

class SubstrateAwareEncoder:
    """
    底物感知特征编码器
    
    参考项目:
    - P2Rank (https://github.com/rdk/p2rank): 配体结合位点预测
    - fpocket (https://github.com/Discngine/fpocket): 口袋检测算法
    
    功能:
    - 残基到配体距离
    - 结合口袋识别
    - 底物相互作用类型
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or get_config()
        self.pocket_probe_radius = 3.0  # fpocket默认
        self.pocket_min_alpha_spheres = 30
    
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
            features[0] = 999.0  # 无配体
            features[1] = 1.0
            return features
        
        ligand_coords = np.array([l['coord'] for l in all_ligands])
        
        # 到最近配体的距离
        distances = np.linalg.norm(ligand_coords - coord, axis=1)
        min_dist = np.min(distances)
        features[0] = min_dist
        features[1] = min(min_dist / 20.0, 1.0)  # 归一化
        
        # 配体邻居数量 (8Å内)
        features[2] = np.sum(distances < 8.0) / max(len(all_ligands), 1)
        
        # 是否在结合口袋内 (简化: 6Å内有配体)
        features[3] = 1.0 if min_dist < 6.0 else 0.0
        
        # 口袋暴露度 (基于邻居密度)
        n_residues = len(ca_coords)
        all_dists = np.linalg.norm(ca_coords - coord, axis=1)
        local_density = np.sum(all_dists < 8.0) / n_residues
        features[4] = 1.0 - local_density  # 低密度 = 高暴露
        
        # 底物相互作用潜力 (距离加权)
        features[5] = np.sum(np.exp(-distances / 5.0))
        
        return features
    
    def detect_binding_pockets(self, ca_coords: np.ndarray, 
                               ligands: List[Dict]) -> List[Dict]:
        """
        简化版口袋检测 (参考fpocket的alpha sphere方法)
        
        完整实现建议使用fpocket:
        fpocket -f protein.pdb
        """
        if not ligands:
            return []
        
        pockets = []
        for i, lig in enumerate(ligands):
            lig_coord = lig['coord']
            
            # 找附近残基
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
                    'volume_estimate': len(pocket_residues) * 50.0,  # 简化估计
                })
        
        return pockets


# =============================================================================
# 3. 氢键网络分析 (参考MDAnalysis)
# =============================================================================

class HydrogenBondAnalyzer:
    """
    氢键网络分析
    
    参考项目:
    - MDAnalysis (https://github.com/MDAnalysis/mdanalysis): hbond分析
    - BioPython: 结构分析
    
    功能:
    - 检测氢键
    - 计算氢键网络特征
    - 识别质子转移通路
    """
    
    # 氢键供体/受体原子
    DONORS = {'N', 'NZ', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'OG', 'OG1', 'OH', 'SG'}
    ACCEPTORS = {'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'ND1', 'NE2', 'SD'}
    
    def __init__(self, distance_cutoff: float = 3.5, angle_cutoff: float = 120.0):
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff
    
    def find_hbonds(self, residues: List[Dict]) -> List[Dict]:
        """
        查找氢键
        
        简化版: 只考虑侧链极性原子之间
        """
        hbonds = []
        
        for i, res_i in enumerate(residues):
            atoms_i = res_i.get('atoms', {})
            
            for j, res_j in enumerate(residues):
                if abs(i - j) < 2:  # 跳过序列上太近的
                    continue
                
                atoms_j = res_j.get('atoms', {})
                
                # 检查供体-受体对
                for donor_name in self.DONORS:
                    if donor_name not in atoms_i:
                        continue
                    donor_coord = np.array(atoms_i[donor_name])
                    
                    for acceptor_name in self.ACCEPTORS:
                        if acceptor_name not in atoms_j:
                            continue
                        acceptor_coord = np.array(atoms_j[acceptor_name])
                        
                        dist = np.linalg.norm(donor_coord - acceptor_coord)
                        
                        if dist < self.distance_cutoff:
                            hbonds.append({
                                'donor_residue': i,
                                'acceptor_residue': j,
                                'donor_atom': donor_name,
                                'acceptor_atom': acceptor_name,
                                'distance': float(dist),
                            })
        
        return hbonds
    
    def compute_hbond_features(self, residue_idx: int, hbonds: List[Dict], 
                               n_residues: int) -> np.ndarray:
        """
        计算氢键相关特征 (3维)
        
        特征:
        - 作为供体的氢键数
        - 作为受体的氢键数
        - 氢键网络中心性 (简化)
        """
        n_donor = sum(1 for hb in hbonds if hb['donor_residue'] == residue_idx)
        n_acceptor = sum(1 for hb in hbonds if hb['acceptor_residue'] == residue_idx)
        
        # 简化的中心性: 总连接数 / 最大可能
        centrality = (n_donor + n_acceptor) / max(len(hbonds), 1)
        
        return np.array([
            n_donor / 5.0,  # 归一化
            n_acceptor / 5.0,
            centrality
        ], dtype=np.float32)


# =============================================================================
# 4. 双金属中心检测器 (参考MetalCoord)
# =============================================================================

class BimetallicCenterDetector:
    """
    双金属活性中心检测器
    
    参考项目:
    - MetalCoord (https://github.com/sb-ncbr/MetalCoord): 金属配位分析
    - CheckMyMetal: 金属配位验证
    
    功能:
    - 识别双金属中心
    - 分析桥连配体
    - 计算配位几何
    """
    
    def __init__(self):
        self.bimetallic_patterns = BIMETALLIC_PATTERNS
        self.metal_coordination = METAL_COORDINATION
    
    def detect_bimetallic_centers(self, metals: List[Dict], residues: List[Dict],
                                   coords: np.ndarray) -> List[Dict]:
        """
        检测双金属中心
        """
        if len(metals) < 2:
            return []
        
        bimetallic_centers = []
        metal_coords = np.array([m['coord'] for m in metals])
        
        # 检查所有金属对
        for i in range(len(metals)):
            for j in range(i + 1, len(metals)):
                m1, m2 = metals[i], metals[j]
                m1_name = m1['name'].upper().rstrip('0123456789')
                m2_name = m2['name'].upper().rstrip('0123456789')
                
                dist = np.linalg.norm(metal_coords[i] - metal_coords[j])
                
                # 匹配双金属模式
                matched_pattern = None
                for pattern_name, pattern in self.bimetallic_patterns.items():
                    req_metals = set(pattern['metals'])
                    actual_metals = {m1_name, m2_name}
                    
                    if req_metals == actual_metals or (len(req_metals) == 1 and m1_name == m2_name):
                        d_min, d_max = pattern['distance_range']
                        if d_min <= dist <= d_max:
                            matched_pattern = pattern_name
                            break
                
                # 查找桥连配体
                bridging_residues = self._find_bridging_ligands(
                    metal_coords[i], metal_coords[j], residues, coords
                )
                
                if matched_pattern or (3.0 <= dist <= 5.0 and bridging_residues):
                    bimetallic_centers.append({
                        'metal1': m1,
                        'metal2': m2,
                        'distance': float(dist),
                        'pattern': matched_pattern,
                        'bridging_residues': bridging_residues,
                        'midpoint': (metal_coords[i] + metal_coords[j]) / 2,
                    })
        
        return bimetallic_centers
    
    def _find_bridging_ligands(self, coord1: np.ndarray, coord2: np.ndarray,
                                residues: List[Dict], coords: np.ndarray,
                                cutoff: float = 3.0) -> List[Dict]:
        """
        查找桥连配体 (同时配位两个金属的残基)
        """
        bridging = []
        midpoint = (coord1 + coord2) / 2
        
        for i, res in enumerate(residues):
            ca = coords[i]
            d1 = np.linalg.norm(ca - coord1)
            d2 = np.linalg.norm(ca - coord2)
            
            # 同时接近两个金属
            if d1 < cutoff and d2 < cutoff:
                bridging.append({
                    'index': i,
                    'resname': res['name'],
                    'resseq': res['number'],
                    'chain': res['chain'],
                    'dist_to_m1': float(d1),
                    'dist_to_m2': float(d2),
                })
        
        return bridging
    
    def analyze_coordination_geometry(self, metal: Dict, residues: List[Dict],
                                       coords: np.ndarray, cutoff: float = 3.0) -> Dict:
        """
        分析单金属配位几何
        """
        metal_coord = metal['coord']
        metal_name = metal['name'].upper().rstrip('0123456789')
        
        # 找配位残基
        distances = np.linalg.norm(coords - metal_coord, axis=1)
        ligand_indices = np.where(distances < cutoff)[0]
        
        ligand_info = []
        for idx in ligand_indices:
            ligand_info.append({
                'index': int(idx),
                'resname': residues[idx]['name'],
                'distance': float(distances[idx]),
            })
        
        # 确定配位数和几何
        coord_number = len(ligand_indices)
        expected = self.metal_coordination.get(metal_name, {})
        
        geometry = 'unknown'
        if coord_number == 4:
            geometry = 'tetrahedral'
        elif coord_number == 5:
            geometry = 'trigonal_bipyramidal'
        elif coord_number == 6:
            geometry = 'octahedral'
        
        return {
            'metal': metal,
            'coordination_number': coord_number,
            'geometry': geometry,
            'ligands': ligand_info,
            'expected_coord': expected.get('coord_number', []),
            'ideal_distance': expected.get('ideal_distance', 2.0),
        }


# =============================================================================
# 5. 智能三联体检测器
# =============================================================================

class TriadDetector:
    """
    智能催化三联体检测器
    
    结合:
    - 经典三联体模式 (M-CSA数据库)
    - 几何约束
    - 角色预测
    """
    
    def __init__(self):
        self.patterns = TRIAD_PATTERNS
    
    def detect_triads(self, residues: List[Dict], coords: np.ndarray,
                      catalytic_residues: List[Dict], 
                      predicted_ec1: int = None) -> List[Dict]:
        """
        检测催化三联体
        
        Args:
            residues: 所有残基信息
            coords: CA坐标
            catalytic_residues: 预测的催化残基
            predicted_ec1: 预测的EC1类别
        """
        triads = []
        
        # 按模式检测
        for pattern_name, pattern in self.patterns.items():
            # EC类别过滤
            if predicted_ec1 and predicted_ec1 not in pattern.get('ec_class', []):
                continue
            
            matched = self._match_pattern(
                pattern, residues, coords, catalytic_residues
            )
            triads.extend(matched)
        
        # 去重和排序
        triads = self._deduplicate_triads(triads)
        triads.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return triads
    
    def _match_pattern(self, pattern: Dict, residues: List[Dict],
                       coords: np.ndarray, catalytic_residues: List[Dict]) -> List[Dict]:
        """
        匹配单个三联体模式
        """
        matches = []
        required = pattern['residues']
        distances = pattern.get('distances', {})
        
        # 创建残基名到索引的映射
        cat_by_name = defaultdict(list)
        for cat_res in catalytic_residues:
            cat_by_name[cat_res['resname']].append(cat_res)
        
        # 尝试所有组合
        candidates = [[], [], []]
        for i, (req_name, req_role) in enumerate(required):
            for cat_res in cat_by_name.get(req_name, []):
                candidates[i].append(cat_res)
        
        # 检查所有组合
        from itertools import product
        for combo in product(*candidates):
            if len(set(c['index'] for c in combo)) != 3:  # 确保是不同残基
                continue
            
            # 检查距离约束
            valid = True
            actual_distances = {}
            
            idx0, idx1, idx2 = combo[0]['index'], combo[1]['index'], combo[2]['index']
            c0, c1, c2 = coords[idx0], coords[idx1], coords[idx2]
            
            d01 = np.linalg.norm(c0 - c1)
            d12 = np.linalg.norm(c1 - c2)
            d02 = np.linalg.norm(c0 - c2)
            
            # 构建距离键
            for (r1, _), (r2, _), d in [(required[0], required[1], d01),
                                         (required[1], required[2], d12),
                                         (required[0], required[2], d02)]:
                key = f"{r1}-{r2}"
                alt_key = f"{r2}-{r1}"
                
                if key in distances:
                    d_min, d_max = distances[key]
                    if not (d_min <= d <= d_max):
                        valid = False
                        break
                elif alt_key in distances:
                    d_min, d_max = distances[alt_key]
                    if not (d_min <= d <= d_max):
                        valid = False
                        break
                
                actual_distances[key] = d
            
            if valid:
                # 计算置信度
                avg_prob = np.mean([c['site_prob'] for c in combo])
                confidence = avg_prob * 0.7 + 0.3  # 基础置信度 + 概率加权
                
                matches.append({
                    'pattern': pattern,
                    'residues': list(combo),
                    'distances': actual_distances,
                    'confidence': float(confidence),
                })
        
        return matches
    
    def _deduplicate_triads(self, triads: List[Dict]) -> List[Dict]:
        """去重"""
        seen = set()
        unique = []
        
        for t in triads:
            key = tuple(sorted([r['index'] for r in t['residues']]))
            if key not in seen:
                seen.add(key)
                unique.append(t)
        
        return unique


# =============================================================================
# 6. 保守性分析接口 (参考ConSurf/EVcouplings)
# =============================================================================

class ConservationAnalyzer:
    """
    序列保守性分析接口
    
    参考项目:
    - ConSurf (https://consurf.tau.ac.il/): 保守性评分
    - EVcouplings (https://github.com/debbiemarkslab/EVcouplings): 共进化分析
    
    功能:
    - 计算保守性分数
    - 识别保守位点
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
# 7. 增强版特征编码器
# =============================================================================

class EnhancedFeatureEncoder:
    """
    增强版特征编码器 - 整合所有子模块
    
    节点特征 (48维):
    - AA one-hot: 20
    - 理化性质: 8
    - 空间特征: 5
    - 金属环境: 3
    - 电子特征: 6 (新增)
    - 底物感知: 6 (新增)
    
    边特征 (14维):
    - 几何特征: 8
    - interaction type: 3
    - 氢键特征: 3 (新增)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or get_config()
        self.aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}
        
        data_cfg = self.config.get('data', {})
        self.distance_cutoff = data_cfg.get('edge_cutoff', 10.0)
        self.metal_shell_cutoff = data_cfg.get('metal_shell_cutoff', 5.0)
        self.metal_neighbor_cutoff = data_cfg.get('metal_neighbor_cutoff', 8.0)
        
        # 子编码器
        ext_tools = self.config.get('external_tools', {})
        self.electronic_encoder = ElectronicFeatureEncoder(ext_tools.get('xtb_path'))
        self.substrate_encoder = SubstrateAwareEncoder(self.config)
        self.hbond_analyzer = HydrogenBondAnalyzer()
        self.conservation_analyzer = ConservationAnalyzer()
        
        # 维度
        self.node_dim = 48
        self.edge_dim = 14
    
    def encode_structure(self, structure_data: Dict) -> Dict:
        """编码完整结构"""
        residues = structure_data['residues']
        coords = structure_data.get('coords')
        metals = structure_data.get('metals', [])
        ligands = structure_data.get('ligands', [])
        n = len(residues)
        
        if n == 0:
            raise ValueError("No residues in structure")
        
        if coords is None:
            coords = np.array([r['ca_coord'] for r in residues], dtype=float)
        elif not isinstance(coords, np.ndarray):
            coords = np.array(coords, dtype=float)
        
        # 预计算氢键
        hbonds = self.hbond_analyzer.find_hbonds(residues)
        
        # 编码每个残基
        node_features = np.zeros((n, self.node_dim), dtype=np.float32)
        ca_coords = np.zeros((n, 3), dtype=np.float32)
        cb_coords = np.zeros((n, 3), dtype=np.float32)
        residue_info = []
        
        for i, res in enumerate(residues):
            # 1) AA one-hot (20维)
            aa_onehot = np.zeros(20)
            if res['name'] in self.aa_to_idx:
                aa_onehot[self.aa_to_idx[res['name']]] = 1
            
            # 2) 理化性质 (8维)
            props = AA_PROPERTIES.get(res['name'], AA_PROPERTIES['ALA'])
            physchem = np.array([
                props['hydro'] / 5.0,
                props['volume'] / 230.0,
                props['charge'],
                props['polar'],
                props['aromatic'],
                props['pka'] / 14.0 if props['pka'] > 0 else 0,
                CATALYTIC_PRIOR.get(res['name'], 0.01),
                self.conservation_analyzer.compute_features(res['name'])
            ])
            
            # 3) 空间特征 (5维)
            spatial = self._compute_spatial_features_single(i, coords)
            
            # 4) 金属环境 (3维)
            metal_env = self._encode_metal_env_single(i, coords, metals)
            
            # 5) 电子特征 (6维)
            electronic = self.electronic_encoder.compute_features(res, coords)
            
            # 6) 底物感知 (6维)
            substrate = self.substrate_encoder.compute_features(i, coords, ligands, metals)
            
            # 合并
            node_features[i] = np.concatenate([
                aa_onehot, physchem, spatial, metal_env, electronic, substrate
            ])
            
            # 坐标
            ca_coords[i] = res['ca_coord'] if isinstance(res['ca_coord'], np.ndarray) else np.array(res['ca_coord'])
            cb = res.get('cb_coord')
            cb_coords[i] = cb if cb is not None else ca_coords[i]
            
            residue_info.append((res['chain'], res['number'], res['name']))
        
        # 边特征
        edge_index, edge_attr = self._build_edges_enhanced(residues, ca_coords, cb_coords, hbonds)
        
        return {
            'node_features': node_features,
            'ca_coords': ca_coords,
            'cb_coords': cb_coords,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'residue_info': residue_info,
            'metals': metals,
            'ligands': ligands,
            'hbonds': hbonds,
            'sequence': structure_data.get('sequence', '')
        }
    
    def _compute_spatial_features_single(self, idx: int, coords: np.ndarray) -> np.ndarray:
        """单个残基的空间特征"""
        n = len(coords)
        coord = coords[idx]
        
        dists = np.linalg.norm(coords - coord, axis=1)
        centroid = coords.mean(axis=0)
        
        density_8 = np.sum(dists < 8.0) / n
        density_12 = np.sum(dists < 12.0) / n
        
        sorted_dists = np.sort(dists)
        avg_neighbor = np.mean(sorted_dists[1:min(11, n)]) / 20.0 if n > 1 else 0
        
        depth = np.linalg.norm(coord - centroid)
        max_depth = np.max(np.linalg.norm(coords - centroid, axis=1))
        rel_depth = depth / (max_depth + 1e-8)
        
        curvature = 0.0
        if n > 10:
            neighbors = np.argsort(dists)[1:11]
            local_centroid = coords[neighbors].mean(axis=0)
            curvature = np.linalg.norm(coord - local_centroid) / 5.0
        
        return np.array([density_8, density_12, avg_neighbor, rel_depth, curvature], dtype=np.float32)
    
    def _encode_metal_env_single(self, idx: int, coords: np.ndarray, metals: List[Dict]) -> np.ndarray:
        """单个残基的金属环境特征"""
        features = np.zeros(3, dtype=np.float32)
        
        if not metals:
            features[0] = 1.0  # 无金属时归一化距离设为1
            return features
        
        coord = coords[idx]
        metal_coords = np.array([m['coord'] for m in metals])
        dists = np.linalg.norm(metal_coords - coord, axis=1)
        
        min_dist = np.min(dists)
        features[0] = min(min_dist / 20.0, 1.0)
        features[1] = np.sum(dists < self.metal_neighbor_cutoff) / max(len(metals), 1)
        features[2] = 1.0 if min_dist < self.metal_shell_cutoff else 0.0
        
        return features
    
    def _build_edges_enhanced(self, residues: List[Dict], ca_coords: np.ndarray,
                              cb_coords: np.ndarray, hbonds: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """增强版边构建"""
        n = len(ca_coords)
        if n == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, self.edge_dim), dtype=np.float32)
        
        # 构建氢键查找
        hbond_pairs = set()
        for hb in hbonds:
            hbond_pairs.add((hb['donor_residue'], hb['acceptor_residue']))
            hbond_pairs.add((hb['acceptor_residue'], hb['donor_residue']))
        
        diff = ca_coords[:, None, :] - ca_coords[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        src, dst = np.where((dist_matrix < self.distance_cutoff) & (dist_matrix > 0))
        
        edge_attr = []
        for s, d in zip(src, dst):
            dist = dist_matrix[s, d]
            direction = (ca_coords[d] - ca_coords[s]) / (dist + 1e-8)
            cb_dist = np.linalg.norm(cb_coords[d] - cb_coords[s])
            seq_dist = abs(d - s)
            
            # 8维几何
            geom = [
                dist / self.distance_cutoff,
                cb_dist / self.distance_cutoff,
                1.0 / (dist + 1.0),
                np.exp(-dist**2 / 32),
                min(seq_dist, 20) / 20.0,
                direction[0], direction[1], direction[2]
            ]
            
            # 3维interaction type
            aa1, aa2 = residues[s]['name'], residues[d]['name']
            itype = self._classify_interaction(aa1, aa2, dist)
            
            # 3维氢键特征 (新增)
            hbond_feat = self._compute_hbond_edge_features(s, d, hbond_pairs, hbonds)
            
            edge_attr.append(np.concatenate([geom, itype, hbond_feat]))
        
        edge_index = np.array([src, dst], dtype=np.int64)
        edge_attr = np.array(edge_attr, dtype=np.float32) if edge_attr else np.zeros((0, self.edge_dim), dtype=np.float32)
        
        return edge_index, edge_attr
    
    def _classify_interaction(self, aa1: str, aa2: str, dist: float) -> np.ndarray:
        """interaction type分类"""
        prop1 = AA_PROPERTIES.get(aa1, {})
        prop2 = AA_PROPERTIES.get(aa2, {})
        
        hbond = ionic = aromatic = 0.0
        
        if prop1.get('charge', 0) * prop2.get('charge', 0) < 0 and dist <= 4.5:
            ionic = 1.0
        
        if (prop1.get('polar', 0) or prop2.get('polar', 0)) and 2.5 <= dist <= 4.0:
            hbond = 1.0
        
        if prop1.get('aromatic', 0) and prop2.get('aromatic', 0) and 3.5 <= dist <= 6.0:
            aromatic = 1.0
        
        return np.array([hbond, ionic, aromatic], dtype=np.float32)
    
    def _compute_hbond_edge_features(self, i: int, j: int, hbond_pairs: Set,
                                      hbonds: List[Dict]) -> np.ndarray:
        """氢键边特征"""
        has_hbond = 1.0 if (i, j) in hbond_pairs else 0.0
        
        hbond_dist = 0.0
        hbond_strength = 0.0
        for hb in hbonds:
            if (hb['donor_residue'] == i and hb['acceptor_residue'] == j) or \
               (hb['donor_residue'] == j and hb['acceptor_residue'] == i):
                hbond_dist = hb['distance'] / 3.5
                hbond_strength = 1.0 - (hb['distance'] - 2.5) / 1.0
                break
        
        return np.array([has_hbond, hbond_dist, max(0, hbond_strength)], dtype=np.float32)


# =============================================================================
# 8. PDB处理器 (沿用V1，略微增强)
# =============================================================================

class PDBProcessor:
    """PDB结构处理器"""
    
    def __init__(self, pdb_dir: str = "./data/pdb_structures"):
        self.pdb_dir = Path(pdb_dir)
        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from Bio.PDB import PDBParser
            self.parser = PDBParser(QUIET=True)
            self.biopython = True
        except ImportError:
            self.biopython = False
    
    def download_pdb(self, pdb_id: str) -> Optional[Path]:
        pdb_id = pdb_id.lower()
        pdb_file = self.pdb_dir / f"{pdb_id}.pdb"
        
        if pdb_file.exists():
            return pdb_file
        
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(pdb_file, 'w') as f:
                    f.write(r.text)
                return pdb_file
        except Exception as e:
            logger.warning(f"下载PDB {pdb_id} 失败: {e}")
        return None
    
    def parse_pdb(self, pdb_path: Path) -> Dict[str, Any]:
        pdb_path = Path(pdb_path)
        if self.biopython:
            return self._parse_biopython(pdb_path)
        return self._parse_simple(pdb_path)
    
    def _parse_biopython(self, pdb_path: Path) -> Dict[str, Any]:
        structure = self.parser.get_structure('protein', str(pdb_path))
        
        residues, sequence, ca_coords = [], [], []
        metals, ligands = [], []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    het, resseq, icode = residue.get_id()
                    res_name = residue.get_resname().strip()
                    
                    if het == ' ':
                        if res_name not in AA_3TO1:
                            continue
                        
                        aa = AA_3TO1[res_name]
                        sequence.append(aa)
                        
                        atoms = {}
                        ca_coord = cb_coord = n_coord = c_coord = None
                        
                        for atom in residue:
                            coord = atom.get_coord().astype(float).tolist()
                            atoms[atom.get_name()] = coord
                            if atom.get_name() == 'CA':
                                ca_coord = coord
                            elif atom.get_name() == 'CB':
                                cb_coord = coord
                            elif atom.get_name() == 'N':
                                n_coord = coord
                            elif atom.get_name() == 'C':
                                c_coord = coord
                        
                        if ca_coord:
                            residues.append({
                                'name': res_name, 'aa': aa, 'number': resseq,
                                'chain': chain.get_id(),
                                'icode': icode if icode != ' ' else '',
                                'ca_coord': np.array(ca_coord, dtype=float),
                                'cb_coord': np.array(cb_coord if cb_coord else ca_coord, dtype=float),
                                'n_coord': n_coord, 'c_coord': c_coord,
                                'atoms': atoms
                            })
                            ca_coords.append(ca_coord)
                    else:
                        atom_list = list(residue.get_atoms())
                        if not atom_list:
                            continue
                        
                        coords_arr = np.stack([a.get_coord().astype(float) for a in atom_list], axis=0)
                        center = coords_arr.mean(axis=0)
                        
                        is_metal = (res_name.upper() in METAL_NAMES or
                                   any(getattr(a, "element", "").upper() in METAL_NAMES for a in atom_list))
                        
                        if is_metal:
                            metals.append({
                                'name': res_name, 'chain': chain.get_id(),
                                'resseq': int(resseq), 'coord': center,
                            })
                        elif res_name not in WATER_NAMES:
                            ligands.append({
                                'name': res_name, 'chain': chain.get_id(),
                                'resseq': int(resseq), 'coord': center,
                            })
        
        coords = np.array(ca_coords, dtype=float) if ca_coords else np.zeros((0, 3), dtype=float)
        
        return {
            'pdb_id': pdb_path.stem, 'sequence': ''.join(sequence),
            'residues': residues, 'coords': coords,
            'metals': metals, 'ligands': ligands,
            'num_residues': len(residues)
        }
    
    def _parse_simple(self, pdb_path: Path) -> Dict[str, Any]:
        """简化解析 (无BioPython)"""
        residues, ca_coords, sequence = [], [], []
        current = None
        het_groups = {}
        
        with open(pdb_path, 'r') as f:
            for line in f:
                record = line[0:6].strip()
                
                if record == 'ATOM':
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    chain = line[21]
                    try:
                        res_num = int(line[22:26].strip())
                    except ValueError:
                        continue
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    
                    if res_name not in AA_3TO1:
                        continue
                    
                    key = (chain, res_num)
                    if current is None or (current['chain'], current['number']) != key:
                        if current and current.get('ca_coord') is not None:
                            residues.append(current)
                            ca_coords.append(current['ca_coord'].tolist())
                            sequence.append(AA_3TO1.get(current['name'], 'X'))
                        current = {
                            'name': res_name, 'aa': AA_3TO1.get(res_name, 'X'),
                            'number': res_num, 'chain': chain, 'icode': '',
                            'ca_coord': None, 'cb_coord': None,
                            'n_coord': None, 'c_coord': None, 'atoms': {}
                        }
                    
                    current['atoms'][atom_name] = [x, y, z]
                    if atom_name == 'CA':
                        current['ca_coord'] = np.array([x, y, z], dtype=float)
                    elif atom_name == 'CB':
                        current['cb_coord'] = np.array([x, y, z], dtype=float)
                    elif atom_name == 'N':
                        current['n_coord'] = [x, y, z]
                    elif atom_name == 'C':
                        current['c_coord'] = [x, y, z]
                
                elif record == 'HETATM':
                    res_name = line[17:20].strip()
                    chain = line[21]
                    try:
                        res_num = int(line[22:26].strip())
                    except ValueError:
                        continue
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    
                    key = (chain, res_num, res_name)
                    het_groups.setdefault(key, []).append(np.array([x, y, z], dtype=float))
        
        if current and current.get('ca_coord') is not None:
            residues.append(current)
            ca_coords.append(current['ca_coord'].tolist())
            sequence.append(AA_3TO1.get(current['name'], 'X'))
        
        for r in residues:
            if r['cb_coord'] is None:
                r['cb_coord'] = r['ca_coord']
        
        metals, ligands = [], []
        for (chain, resseq, resname), coords_list in het_groups.items():
            center = np.stack(coords_list, axis=0).mean(axis=0)
            if resname.upper() in METAL_NAMES:
                metals.append({'name': resname, 'chain': chain, 'resseq': resseq, 'coord': center})
            elif resname not in WATER_NAMES:
                ligands.append({'name': resname, 'chain': chain, 'resseq': resseq, 'coord': center})
        
        coords = np.array(ca_coords, dtype=float) if ca_coords else np.zeros((0, 3), dtype=float)
        
        return {
            'pdb_id': pdb_path.stem, 'sequence': ''.join(sequence),
            'residues': residues, 'coords': coords,
            'metals': metals, 'ligands': ligands,
            'num_residues': len(residues)
        }


# =============================================================================
# 9. 模型定义 (简化，专注推理)
# =============================================================================

class GeometricMessagePassing(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        edge_bias = self.edge_proj(edge_attr)
        
        attn_scores = (q[src] * k[dst]).sum(dim=-1) / math.sqrt(self.head_dim) + edge_bias
        attn_weights = self._scatter_softmax(attn_scores, dst, x.size(0))
        attn_weights = self.dropout(attn_weights)
        
        weighted_v = v[src] * attn_weights.unsqueeze(-1)
        out = torch.zeros(x.size(0), self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, dst, weighted_v)
        out = self.out_proj(out.view(-1, self.hidden_dim))
        
        x = self.layer_norm(x + self.dropout(out))
        x = self.ffn_norm(x + self.ffn(x))
        return x
    
    def _scatter_softmax(self, src, index, num_nodes):
        max_val = torch.zeros(num_nodes, src.size(1), device=src.device)
        max_val.index_reduce_(0, index, src, 'amax', include_self=False)
        exp_src = torch.exp(src - max_val[index])
        sum_exp = torch.zeros(num_nodes, src.size(1), device=src.device)
        sum_exp.index_add_(0, index, exp_src)
        return exp_src / (sum_exp[index] + 1e-8)


class GeometricGNN(nn.Module):
    def __init__(self, node_dim: int = 48, edge_dim: int = 14, hidden_dim: int = 256,
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.layers = nn.ModuleList([
            GeometricMessagePassing(hidden_dim, hidden_dim // 2, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, node_features, edge_index, edge_attr):
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        return self.final_norm(h)


class HierarchicalECPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_ec1: int = 7, num_ec2: int = 70,
                 num_ec3: int = 300, dropout: float = 0.2):
        super().__init__()
        self.global_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.ec1_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_ec1)
        )
        self.ec2_classifier = nn.Sequential(
            nn.Linear(hidden_dim + num_ec1, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_ec2)
        )
        self.ec3_classifier = nn.Sequential(
            nn.Linear(hidden_dim + num_ec1 + num_ec2, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_ec3)
        )
    
    def forward(self, node_embeddings, batch_idx=None):
        if batch_idx is None:
            global_feat = node_embeddings.mean(dim=0, keepdim=True)
        else:
            num_graphs = batch_idx.max().item() + 1
            global_feat = torch.zeros(num_graphs, node_embeddings.size(1), device=node_embeddings.device)
            for i in range(num_graphs):
                global_feat[i] = node_embeddings[batch_idx == i].mean(dim=0)
        
        global_feat = self.global_pool(global_feat)
        ec1_logits = self.ec1_classifier(global_feat)
        ec1_probs = F.softmax(ec1_logits, dim=-1)
        ec2_logits = self.ec2_classifier(torch.cat([global_feat, ec1_probs], dim=-1))
        ec2_probs = F.softmax(ec2_logits, dim=-1)
        ec3_logits = self.ec3_classifier(torch.cat([global_feat, ec1_probs, ec2_probs], dim=-1))
        
        return {'ec1_logits': ec1_logits, 'ec2_logits': ec2_logits, 'ec3_logits': ec3_logits,
                'ec1_probs': ec1_probs, 'global_feat': global_feat}


class CatalyticSitePredictor(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_roles: int = 10, 
                 ec_cond_dim: int = 7, dropout: float = 0.2):
        super().__init__()
        self.cond_proj = nn.Linear(hidden_dim + ec_cond_dim, hidden_dim)
        self.site_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.role_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_roles)
        )
    
    def forward(self, node_emb, ec_global=None, batch_idx=None):
        if ec_global is not None:
            if batch_idx is None:
                ec_expand = ec_global.expand(node_emb.size(0), -1)
            else:
                ec_expand = ec_global[batch_idx]
            h = self.cond_proj(torch.cat([node_emb, ec_expand], dim=-1))
        else:
            h = node_emb
        
        return self.site_classifier(h), self.role_classifier(h)


class CatalyticTriadPredictorV2(nn.Module):
    """V2完整模型"""
    
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or get_config()
        
        model_cfg = self.config['model']
        ec_cfg = self.config['ec_prediction']
        
        self.gnn = GeometricGNN(
            node_dim=model_cfg['node_dim'], edge_dim=model_cfg['edge_dim'],
            hidden_dim=model_cfg['hidden_dim'], num_layers=model_cfg['num_gnn_layers'],
            num_heads=model_cfg['num_heads'], dropout=model_cfg['dropout']
        )
        self.ec_predictor = HierarchicalECPredictor(
            model_cfg['hidden_dim'], ec_cfg['num_ec1_classes'],
            ec_cfg['num_ec2_classes'], ec_cfg['num_ec3_classes'], model_cfg['dropout']
        )
        self.site_predictor = CatalyticSitePredictor(
            model_cfg['hidden_dim'], num_roles=len(ROLE_MAPPING),
            ec_cond_dim=ec_cfg['num_ec1_classes'], dropout=model_cfg['dropout']
        )
    
    def forward(self, node_features, edge_index, edge_attr, batch_idx=None):
        node_emb = self.gnn(node_features, edge_index, edge_attr)
        ec_outputs = self.ec_predictor(node_emb, batch_idx)
        site_logits, role_logits = self.site_predictor(node_emb, ec_outputs['ec1_probs'], batch_idx)
        return {'site_logits': site_logits, 'role_logits': role_logits, 
                'node_embeddings': node_emb, **ec_outputs}
    
    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = 'cpu'):
        ckpt = torch.load(path, map_location=device)
        config = ckpt.get('config', get_config())
        model = cls(config)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        return model


# =============================================================================
# 10. 增强版推理器
# =============================================================================

class EnhancedCatalyticSiteInference:
    """
    增强版推理器
    
    新增功能:
    - 智能三联体检测
    - 双金属中心检测
    - 下游设计接口输出
    """
    
    def __init__(self, model_path: str = None, model: CatalyticTriadPredictorV2 = None,
                 config: Dict = None, device: str = None):
        self.config = config or get_config()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model is not None:
            self.model = model.to(self.device)
        elif model_path and Path(model_path).exists():
            self.model = CatalyticTriadPredictorV2.load_from_checkpoint(model_path, self.device)
        else:
            logger.warning("未提供模型，使用默认初始化")
            self.model = CatalyticTriadPredictorV2(self.config).to(self.device)
        
        self.model.eval()
        self.pdb_proc = PDBProcessor()
        self.feat_enc = EnhancedFeatureEncoder(self.config)
        self.triad_detector = TriadDetector()
        self.bimetallic_detector = BimetallicCenterDetector()
    
    @torch.no_grad()
    def predict(self, pdb_path: str, target_ec1: int = None,
                site_threshold: float = 0.5, role_threshold: float = 0.3) -> Dict[str, Any]:
        """完整预测"""
        pdb_path = Path(pdb_path)
        
        if not pdb_path.exists() and len(str(pdb_path)) == 4:
            pdb_path = self.pdb_proc.download_pdb(str(pdb_path))
            if not pdb_path:
                raise ValueError(f"无法下载PDB: {pdb_path}")
        
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB文件不存在: {pdb_path}")
        
        # 解析和编码
        struct = self.pdb_proc.parse_pdb(pdb_path)
        encoded = self.feat_enc.encode_structure(struct)
        
        x = torch.tensor(encoded['node_features'], dtype=torch.float32).to(self.device)
        ei = torch.tensor(encoded['edge_index'], dtype=torch.long).to(self.device)
        ea = torch.tensor(encoded['edge_attr'], dtype=torch.float32).to(self.device)
        
        # 模型预测
        outputs = self.model(x, ei, ea)
        
        site_probs = torch.sigmoid(outputs['site_logits']).cpu().numpy().flatten()
        role_logits = outputs['role_logits'].cpu().numpy()
        ec1_probs = F.softmax(outputs['ec1_logits'], dim=-1).cpu().numpy()[0]
        
        pred_ec1 = int(np.argmax(ec1_probs)) + 1
        residues = struct['residues']
        metals = struct.get('metals', [])
        coords = struct['coords']
        
        # 1) 催化残基
        catalytic_residues = []
        for i, res in enumerate(residues):
            p_site = site_probs[i]
            if p_site < site_threshold:
                continue
            
            role_prob = F.softmax(torch.tensor(role_logits[i]), dim=-1).numpy()
            role_id = int(np.argmax(role_prob))
            
            roles = [{'role_id': j, 'role_name': ROLE_MAPPING.get(j, 'unknown'), 'prob': float(role_prob[j])}
                     for j in range(len(role_prob)) if role_prob[j] >= role_threshold]
            
            catalytic_residues.append({
                'index': i, 'chain': res['chain'], 'resseq': res['number'],
                'resname': res['name'], 'aa': res.get('aa', AA_3TO1.get(res['name'], 'X')),
                'site_prob': float(p_site), 'role_id': role_id,
                'role_name': ROLE_MAPPING.get(role_id, 'unknown'),
                'role_prob': float(role_prob[role_id]), 'roles': roles
            })
        
        # 2) 单金属中心
        metal_centers = []
        for m in metals:
            coord_info = self.bimetallic_detector.analyze_coordination_geometry(m, residues, coords)
            metal_centers.append(coord_info)
        
        # 3) 双金属中心
        bimetallic_centers = self.bimetallic_detector.detect_bimetallic_centers(metals, residues, coords)
        
        # 4) 智能三联体检测
        triads = self.triad_detector.detect_triads(residues, coords, catalytic_residues, pred_ec1)
        
        # 5) 结合口袋
        pockets = self.feat_enc.substrate_encoder.detect_binding_pockets(
            coords, struct.get('ligands', [])
        )
        
        return {
            'pdb_id': struct['pdb_id'],
            'sequence': struct.get('sequence', ''),
            'num_residues': len(residues),
            'ec1_prediction': pred_ec1,
            'ec1_confidence': float(ec1_probs[pred_ec1 - 1]),
            'ec1_probs': ec1_probs.tolist(),
            'catalytic_residues': catalytic_residues,
            'metal_centers': metal_centers,
            'bimetallic_centers': bimetallic_centers,
            'triads': triads,
            'binding_pockets': pockets,
            'hbonds': encoded.get('hbonds', []),
            'metals': metals,
            'ligands': struct.get('ligands', [])
        }
    
    def export_for_proteinmpnn(self, results: Dict, output_path: str):
        """
        导出ProteinMPNN格式
        
        参考: https://github.com/dauparas/ProteinMPNN
        """
        fixed_positions = {}
        for cat_res in results['catalytic_residues']:
            chain = cat_res['chain']
            if chain not in fixed_positions:
                fixed_positions[chain] = []
            fixed_positions[chain].append(cat_res['resseq'])
        
        mpnn_input = {
            'pdb_id': results['pdb_id'],
            'fixed_positions': fixed_positions,
            'catalytic_info': {
                'triads': [{'residues': [r['resseq'] for r in t['residues']]} for t in results['triads'][:3]],
                'metal_sites': [{'metal': mc['metal']['name'], 'ligands': [l['index'] for l in mc['ligands']]}
                               for mc in results['metal_centers']],
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(mpnn_input, f, indent=2)
        logger.info(f"✓ ProteinMPNN格式: {output_path}")
    
    def export_for_rfdiffusion(self, results: Dict, output_path: str):
        """
        导出RFdiffusion格式
        
        参考: https://github.com/RosettaCommons/RFdiffusion
        """
        hotspot_residues = []
        for cat_res in results['catalytic_residues'][:10]:
            hotspot_residues.append(f"{cat_res['chain']}{cat_res['resseq']}")
        
        for mc in results['metal_centers']:
            for lig in mc.get('ligands', [])[:3]:
                hotspot_residues.append(f"{results['catalytic_residues'][0]['chain'] if results['catalytic_residues'] else 'A'}{lig['index']+1}")
        
        rfd_input = {
            'pdb_id': results['pdb_id'],
            'hotspot_residues': list(set(hotspot_residues)),
            'contigs': [],  # 需要用户指定
            'ec_class': results['ec1_prediction'],
            'catalytic_mechanism': {
                'triads': len(results['triads']),
                'metal_centers': len(results['metal_centers']),
                'bimetallic': len(results['bimetallic_centers']),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(rfd_input, f, indent=2)
        logger.info(f"✓ RFdiffusion格式: {output_path}")
    
    def export_nanozyme_design_input(self, results: Dict, output_path: str):
        """
        导出纳米酶设计输入格式
        
        包含:
        - 催化三联体几何
        - 金属配位信息
        - 关键距离约束
        """
        design_input = {
            'source_enzyme': results['pdb_id'],
            'ec_class': results['ec1_prediction'],
            
            # 催化几何约束
            'catalytic_geometry': {
                'triads': [],
                'metal_centers': [],
                'bimetallic_centers': [],
            },
            
            # 关键距离
            'distance_constraints': [],
            
            # 电子性质要求
            'electronic_requirements': {
                'nucleophile_types': [],
                'base_types': [],
                'metal_ligand_types': [],
            }
        }
        
        # 填充三联体信息
        for triad in results['triads'][:3]:
            triad_info = {
                'residues': [{'name': r['resname'], 'role': r['role_name']} for r in triad['residues']],
                'distances': triad['distances'],
                'pattern': triad.get('pattern', {}).get('residues', []) if isinstance(triad.get('pattern'), dict) else None,
            }
            design_input['catalytic_geometry']['triads'].append(triad_info)
            
            # 添加距离约束
            for key, dist in triad['distances'].items():
                design_input['distance_constraints'].append({
                    'type': 'triad_distance',
                    'pair': key,
                    'target': dist,
                    'tolerance': 0.5
                })
        
        # 填充金属中心
        for mc in results['metal_centers']:
            metal_info = {
                'metal_type': mc['metal']['name'],
                'coordination_number': mc['coordination_number'],
                'geometry': mc['geometry'],
                'ligand_types': [l['resname'] for l in mc['ligands']],
            }
            design_input['catalytic_geometry']['metal_centers'].append(metal_info)
        
        # 双金属
        for bmc in results['bimetallic_centers']:
            bimetal_info = {
                'metals': [bmc['metal1']['name'], bmc['metal2']['name']],
                'distance': bmc['distance'],
                'pattern': bmc.get('pattern'),
                'bridging_ligands': [r['resname'] for r in bmc['bridging_residues']],
            }
            design_input['catalytic_geometry']['bimetallic_centers'].append(bimetal_info)
        
        # 电子性质
        for cat_res in results['catalytic_residues']:
            role = cat_res['role_name']
            if 'nucleophile' in role:
                design_input['electronic_requirements']['nucleophile_types'].append(cat_res['resname'])
            elif 'base' in role:
                design_input['electronic_requirements']['base_types'].append(cat_res['resname'])
            elif 'metal' in role:
                design_input['electronic_requirements']['metal_ligand_types'].append(cat_res['resname'])
        
        # 去重
        for key in design_input['electronic_requirements']:
            design_input['electronic_requirements'][key] = list(set(design_input['electronic_requirements'][key]))
        
        with open(output_path, 'w') as f:
            json.dump(design_input, f, indent=2)
        logger.info(f"✓ 纳米酶设计输入: {output_path}")
    
    def print_results(self, results: Dict, top_k: int = 15):
        """打印结果"""
        print("\n" + "="*70)
        print(f"催化位点预测结果 V2.0 - PDB: {results['pdb_id']}")
        print("="*70)
        
        ec_names = {1: '氧化还原酶', 2: '转移酶', 3: '水解酶',
                   4: '裂解酶', 5: '异构酶', 6: '连接酶', 7: '转位酶'}
        ec1 = results['ec1_prediction']
        print(f"\nEC分类预测: EC {ec1} ({ec_names.get(ec1, '未知')}) "
              f"[置信度: {results['ec1_confidence']:.2%}]")
        
        # 双金属中心
        if results['bimetallic_centers']:
            print(f"\n★ 发现 {len(results['bimetallic_centers'])} 个双金属中心:")
            for i, bmc in enumerate(results['bimetallic_centers']):
                pattern = bmc.get('pattern', '未知模式')
                print(f"  {i+1}. {bmc['metal1']['name']}-{bmc['metal2']['name']} "
                      f"(距离: {bmc['distance']:.2f}Å, 模式: {pattern})")
                if bmc['bridging_residues']:
                    bridging = ', '.join([f"{r['resname']}{r['resseq']}" for r in bmc['bridging_residues'][:3]])
                    print(f"     桥连配体: {bridging}")
        
        # 单金属中心
        if results['metal_centers']:
            print(f"\n发现 {len(results['metal_centers'])} 个金属中心:")
            for i, mc in enumerate(results['metal_centers']):
                m = mc['metal']
                print(f"  {i+1}. {m['name']} - 配位数: {mc['coordination_number']}, "
                      f"几何: {mc['geometry']}")
        
        # 三联体
        if results['triads']:
            print(f"\n★ 发现 {len(results['triads'])} 个催化三联体:")
            for i, t in enumerate(results['triads'][:5]):
                res_str = '-'.join([f"{r['resname']}{r['resseq']}" for r in t['residues']])
                pattern_name = list(t.get('pattern', {}).keys())[0] if isinstance(t.get('pattern'), dict) else '未知'
                print(f"  {i+1}. {res_str} [置信度: {t['confidence']:.2f}]")
        
        # 催化残基
        catalytic = results['catalytic_residues']
        print(f"\n找到 {len(catalytic)} 个预测催化残基:\n")
        print(f"{'排名':<5} {'链':<4} {'残基':<12} {'概率':<8} {'角色'}")
        print("-"*60)
        
        for i, r in enumerate(catalytic[:top_k]):
            roles_str = ', '.join([f"{ro['role_name']}" for ro in r['roles'][:2]]) or '-'
            print(f"{i+1:<5} {r['chain']:<4} {r['resname']}{r['resseq']:<8} "
                  f"{r['site_prob']:.4f}  {roles_str}")
        
        print("="*70)


# =============================================================================
# 命令行接口
# =============================================================================

def cmd_predict(args):
    """预测"""
    config = get_config()
    
    if args.model and Path(args.model).exists():
        predictor = EnhancedCatalyticSiteInference(model_path=args.model, config=config)
    else:
        logger.warning("模型文件不存在，使用随机初始化模型（仅用于测试）")
        predictor = EnhancedCatalyticSiteInference(config=config)
    
    target_ec1 = int(args.ec1) if args.ec1 else None
    results = predictor.predict(args.pdb, target_ec1=target_ec1, site_threshold=args.threshold)
    predictor.print_results(results, top_k=args.top)
    
    if args.output:
        # 导出各种格式
        predictor.export_nanozyme_design_input(results, args.output + '_nanozyme.json')
        predictor.export_for_proteinmpnn(results, args.output + '_mpnn.json')
        predictor.export_for_rfdiffusion(results, args.output + '_rfd.json')
        
        # CSV
        import pandas as pd
        df = pd.DataFrame(results['catalytic_residues'])
        df.to_csv(args.output + '.csv', index=False)
        logger.info(f"✓ CSV: {args.output}.csv")


def cmd_analyze(args):
    """分析PDB的金属和三联体"""
    pdb_proc = PDBProcessor()
    
    pdb_path = Path(args.pdb)
    if not pdb_path.exists() and len(str(args.pdb)) == 4:
        pdb_path = pdb_proc.download_pdb(args.pdb)
    
    struct = pdb_proc.parse_pdb(pdb_path)
    
    print(f"\nPDB: {struct['pdb_id']}")
    print(f"残基数: {struct['num_residues']}")
    print(f"金属: {[m['name'] for m in struct['metals']]}")
    print(f"配体: {[l['name'] for l in struct['ligands']]}")
    
    # 双金属检测
    detector = BimetallicCenterDetector()
    bimetals = detector.detect_bimetallic_centers(struct['metals'], struct['residues'], struct['coords'])
    
    if bimetals:
        print(f"\n双金属中心:")
        for bm in bimetals:
            print(f"  {bm['metal1']['name']}-{bm['metal2']['name']}: {bm['distance']:.2f}Å ({bm.get('pattern', 'unknown')})")


def main():
    parser = argparse.ArgumentParser(description="催化三联体识别与反应预测模型 V2.0")
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # predict
    p_pred = subparsers.add_parser('predict', help='预测催化残基')
    p_pred.add_argument('--pdb', required=True, help='PDB文件或ID')
    p_pred.add_argument('--model', default='models/best_model.pt', help='模型路径')
    p_pred.add_argument('--threshold', type=float, default=0.5, help='阈值')
    p_pred.add_argument('--top', type=int, default=15, help='显示前N个')
    p_pred.add_argument('--output', help='输出文件前缀')
    p_pred.add_argument('--ec1', help='目标EC1类别')
    
    # analyze
    p_ana = subparsers.add_parser('analyze', help='分析PDB结构')
    p_ana.add_argument('--pdb', required=True, help='PDB文件或ID')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
