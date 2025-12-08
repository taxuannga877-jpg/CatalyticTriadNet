#!/usr/bin/env python3
"""
扩散模型输出适配器
支持: RFdiffusion, ProteinMPNN, 自定义图数据, PyG Data
"""

import numpy as np
import json
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

"""
纳米酶可视化完整模块 v2.0
=====================================
适配: catalytic-triad-predictor-enhanced.py + 扩散模型生成结构

功能:
├── 2D可视化 (分子图、三联体、金属中心、完整报告)
├── 3D可视化 (空间分布、配位多面体、交互式、动画)
├── 专业软件导出 (PyMOL, ChimeraX, VMD)
└── 扩散模型适配 (RFdiffusion, ProteinMPNN, 自定义图数据)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ============================================================================
# 颜色配置
# ============================================================================
AA_COLORS = {
    'SER': '#FF6B6B', 'CYS': '#FF8E8E', 'THR': '#FFA5A5',  # 亲核
    'HIS': '#4ECDC4', 'LYS': '#45B7AA', 'ARG': '#3DAA9E',  # 碱性
    'ASP': '#FFE66D', 'GLU': '#FFD93D',                    # 酸性
    'PHE': '#C9B1FF', 'TYR': '#B8A0FF', 'TRP': '#A78BFA',  # 芳香
    'ALA': '#E8E8E8', 'VAL': '#E0E0E0', 'LEU': '#D8D8D8', 'ILE': '#D0D0D0',
    'MET': '#C8C8C8', 'PRO': '#C0C0C0', 'GLY': '#F0F0F0',
    'ASN': '#A8E6CF', 'GLN': '#98D9BE', 'UNK': '#CCCCCC'
}

METAL_COLORS = {
    'ZN': '#5C7AEA', 'MG': '#06D6A0', 'FE': '#EF476F', 'MN': '#9B5DE5',
    'CU': '#F4A261', 'CA': '#2EC4B6', 'NI': '#8338EC', 'CO': '#FB5607',
    'ZR': '#00B4D8', 'CE': '#90BE6D', 'DEFAULT': '#6C757D'
}

ROLE_COLORS = {
    'nucleophile': '#FF6B6B', 'general_base': '#4ECDC4', 'general_acid': '#FFE66D',
    'electrostatic': '#FFD93D', 'metal_ligand': '#5C7AEA',
    'transition_state_stabilizer': '#9B5DE5', 'non_catalytic': '#CCCCCC', 'other': '#888888'
}

# 配位几何模板
COORD_GEOMETRIES = {
    'tetrahedral': {'v': np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]])/np.sqrt(3),
                    'f': [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]},
    'octahedral': {'v': np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]),
                   'f': [[0,2,4],[0,4,3],[0,3,5],[0,5,2],[1,2,4],[1,4,3],[1,3,5],[1,5,2]]},
    'square_planar': {'v': np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]])/np.sqrt(2),
                      'f': [[0,1,2,3]]},
    'trigonal_bipyramidal': {'v': np.array([[0,0,1],[0,0,-1],[1,0,0],[-0.5,0.866,0],[-0.5,-0.866,0]]),
                              'f': [[0,2,3],[0,3,4],[0,4,2],[1,2,3],[1,3,4],[1,4,2]]}
}
# ============================================================================
# 扩散模型适配器
# ============================================================================
class DiffusionModelAdapter:
    """
    扩散模型输出适配器
    支持: RFdiffusion, ProteinMPNN, 自定义图数据, PyG Data
    """
    
    ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H', 'UNK']
    BOND_TYPES = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    
    @staticmethod
    def from_rfdiffusion(output_path: str) -> Dict:
        """解析RFdiffusion输出"""
        with open(output_path) as f:
            data = json.load(f)
        
        return {
            'pdb_id': data.get('name', 'RFdiff_output'),
            'coords': np.array(data.get('coords', [])),
            'sequence': data.get('sequence', ''),
            'catalytic_residues': [
                {'index': int(h[1:])-1, 'resseq': int(h[1:]), 'chain': h[0],
                 'resname': 'UNK', 'site_prob': 1.0, 'role_name': 'other'}
                for h in data.get('hotspot_residues', [])
            ],
            'triads': [], 'metals': [], 'metal_centers': [], 'bimetallic_centers': []
        }
    
    @staticmethod
    def from_proteinmpnn(output_path: str) -> Dict:
        """解析ProteinMPNN输出"""
        with open(output_path) as f:
            data = json.load(f)
        
        fixed = data.get('fixed_positions', {})
        cat_res = []
        for chain, positions in fixed.items():
            for pos in positions:
                cat_res.append({
                    'index': pos-1, 'resseq': pos, 'chain': chain,
                    'resname': 'UNK', 'site_prob': 1.0, 'role_name': 'other'
                })
        
        return {
            'pdb_id': data.get('pdb_id', 'MPNN_output'),
            'catalytic_residues': cat_res,
            'triads': [], 'metals': [], 'metal_centers': [], 'bimetallic_centers': []
        }
    
    @staticmethod
    def from_graph_data(node_types: np.ndarray, edge_index: np.ndarray,
                        coords: np.ndarray = None, edge_types: np.ndarray = None,
                        atom_list: List[str] = None, **kwargs) -> Dict:
        """
        从图数据构建可视化输入
        
        Args:
            node_types: [N] 或 [N, num_types] 节点类型
            edge_index: [2, E] 边索引
            coords: [N, 3] 坐标（可选）
            edge_types: [E] 或 [E, num_bond_types] 边类型（可选）
            atom_list: 原子类型映射表
        """
        atom_list = atom_list or DiffusionModelAdapter.ATOM_TYPES
        
        # 解析节点
        node_types = np.array(node_types)
        if node_types.ndim == 2:
            node_indices = np.argmax(node_types, axis=1)
        else:
            node_indices = node_types.astype(int)
        
        symbols = [atom_list[i] if i < len(atom_list) else 'UNK' for i in node_indices]
        
        # 生成坐标
        n_nodes = len(symbols)
        if coords is None:
            t = np.linspace(0, 4*np.pi, n_nodes)
            coords = np.column_stack([8*np.cos(t), 8*np.sin(t), t])
        
        # 解析边
        edge_index = np.array(edge_index)
        if edge_index.shape[0] != 2:
            edge_index = edge_index.T
        
        edges = []
        seen = set()
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            key = (min(u,v), max(u,v))
            if key not in seen:
                seen.add(key)
                bond_type = 0
                if edge_types is not None:
                    et = np.array(edge_types)
                    if et.ndim == 2:
                        bond_type = int(np.argmax(et[i]))
                    elif i < len(et):
                        bond_type = int(et[i])
                edges.append({'src': u, 'dst': v, 'type': bond_type})
        
        # 构建残基列表
        residues = []
        for i, sym in enumerate(symbols):
            residues.append({
                'index': i, 'resseq': i+1, 'chain': 'A', 'resname': sym,
                'site_prob': kwargs.get('site_probs', [0.5]*n_nodes)[i] if i < len(kwargs.get('site_probs', [])) else 0.5,
                'role_name': 'other', 'ca_coord': coords[i]
            })
        
        return {
            'pdb_id': kwargs.get('name', 'Generated'),
            'coords': coords, 'sequence': ''.join(symbols),
            'catalytic_residues': residues, 'edges': edges,
            'node_symbols': symbols, 'triads': [], 'metals': [],
            'metal_centers': [], 'bimetallic_centers': [],
            '_is_molecule': True  # 标记为小分子
        }
    
    @staticmethod
    def from_pyg_data(data, atom_list: List[str] = None) -> Dict:
        """从PyG Data对象构建"""
        import torch
        x = data.x.cpu().numpy() if torch.is_tensor(data.x) else np.array(data.x)
        ei = data.edge_index.cpu().numpy() if torch.is_tensor(data.edge_index) else np.array(data.edge_index)
        ea = data.edge_attr.cpu().numpy() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        
        coords = None
        if hasattr(data, 'pos') and data.pos is not None:
            coords = data.pos.cpu().numpy() if torch.is_tensor(data.pos) else np.array(data.pos)
        
        return DiffusionModelAdapter.from_graph_data(x, ei, coords, ea, atom_list)
    
    @staticmethod
    def from_nanozyme_design(design_path: str) -> Dict:
        """从纳米酶设计输入文件构建"""
        with open(design_path) as f:
            data = json.load(f)
        
        cat_geom = data.get('catalytic_geometry', {})
        
        # 提取三联体
        triads = []
        for t in cat_geom.get('triads', []):
            triads.append({
                'residues': [{'resname': r['name'], 'resseq': i, 'index': i, 'role_name': r.get('role', 'other')}
                            for i, r in enumerate(t.get('residues', []))],
                'distances': t.get('distances', {}),
                'confidence': 0.9
            })
        
        # 提取金属中心
        metals = []
        metal_centers = []
        for i, mc in enumerate(cat_geom.get('metal_centers', [])):
            metals.append({'name': mc['metal_type'], 'coord': [i*5, 0, 0]})
            metal_centers.append({
                'metal': {'name': mc['metal_type']},
                'coordination_number': mc.get('coordination_number', 4),
                'geometry': mc.get('geometry', 'tetrahedral'),
                'ligands': [{'resname': lt, 'distance': 2.0} for lt in mc.get('ligand_types', [])]
            })
        
        return {
            'pdb_id': data.get('source_enzyme', 'Nanozyme'),
            'ec1_prediction': data.get('ec_class', 3),
            'catalytic_residues': [],
            'triads': triads, 'metals': metals,
            'metal_centers': metal_centers,
            'bimetallic_centers': cat_geom.get('bimetallic_centers', [])
        }
# ============================================================================
# 专业软件导出器
# ============================================================================
