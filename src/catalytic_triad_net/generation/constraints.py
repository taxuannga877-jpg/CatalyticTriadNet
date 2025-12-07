#!/usr/bin/env python3
"""
CatalyticDiff: 基于催化位点几何约束的条件分子扩散生成模型
================================================================================

输入: CatalyticTriadNet提取的催化位点模板
    - 三联体几何约束 (残基类型、距离、角度)
    - 金属配位环境 (配位数、配位几何、配体类型)
    - 电子性质要求 (亲核/亲电位点、电荷分布)

输出: 满足催化几何约束的全新3D分子结构
    - 原子类型 (C, N, O, S, P, 金属等)
    - 3D坐标
    - 键连接关系

核心方法:
    E(3)-等变扩散模型 + 几何约束条件化

参考项目:
    - EDM: https://github.com/ehoogeboom/e3_diffusion_for_molecules
    - DiffLinker: https://github.com/igashov/DiffLinker
    - TargetDiff: https://github.com/guanjq/targetdiff
    - PMDM: https://github.com/tencent-ailab/MDM
    - GeoLDM: https://github.com/MinkaiXu/GeoLDM

作者: Claude
"""

import os
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. 原子与分子表示
# =============================================================================

# 支持的原子类型 (纳米酶相关)
ATOM_TYPES = [
    'H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',  # 有机元素
    'Fe', 'Cu', 'Zn', 'Mn', 'Co', 'Ni', 'Mo', 'V',       # 过渡金属
    'Ce', 'Pt', 'Pd', 'Au', 'Ag', 'Ru', 'Rh', 'Ir',      # 稀土/贵金属
    'Mg', 'Ca', 'B', 'Si', 'Se', 'Te',                   # 其他
]
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES)

# 原子特征 (电负性、共价半径、vdW半径、价电子数)
ATOM_FEATURES = {
    'H':  [2.20, 0.31, 1.20, 1], 'C':  [2.55, 0.76, 1.70, 4],
    'N':  [3.04, 0.71, 1.55, 5], 'O':  [3.44, 0.66, 1.52, 6],
    'S':  [2.58, 1.05, 1.80, 6], 'P':  [2.19, 1.07, 1.80, 5],
    'Fe': [1.83, 1.32, 2.05, 8], 'Cu': [1.90, 1.32, 1.96, 11],
    'Zn': [1.65, 1.22, 2.01, 12], 'Mn': [1.55, 1.39, 2.05, 7],
    'Co': [1.88, 1.26, 2.00, 9], 'Ni': [1.91, 1.24, 1.97, 10],
    'Ce': [1.12, 1.85, 2.70, 4], 'Pt': [2.28, 1.36, 2.13, 10],
    'Au': [2.54, 1.36, 2.14, 11], 'Pd': [2.20, 1.39, 2.10, 10],
}

# 催化功能团模板
FUNCTIONAL_GROUP_TEMPLATES = {
    'nucleophile_ser': {'atoms': ['O', 'H'], 'geometry': 'terminal'},
    'nucleophile_cys': {'atoms': ['S', 'H'], 'geometry': 'terminal'},
    'general_base_his': {'atoms': ['C', 'N', 'C', 'N', 'C'], 'geometry': 'imidazole'},
    'acid_asp': {'atoms': ['C', 'O', 'O'], 'geometry': 'carboxyl'},
    'acid_glu': {'atoms': ['C', 'C', 'O', 'O'], 'geometry': 'carboxyl'},
    'metal_n4': {'atoms': ['N', 'N', 'N', 'N'], 'geometry': 'square_planar'},
    'metal_o4': {'atoms': ['O', 'O', 'O', 'O'], 'geometry': 'tetrahedral'},
    'metal_n2o2': {'atoms': ['N', 'N', 'O', 'O'], 'geometry': 'square_planar'},
    'oxyanion_hole': {'atoms': ['N', 'H', 'N', 'H'], 'geometry': 'nh_pair'},
}


# =============================================================================
# 2. 催化位点约束解析器
# =============================================================================

@dataclass
class GeometricConstraint:
    """几何约束"""
    constraint_type: str  # 'distance', 'angle', 'dihedral', 'coordination'
    atom_indices: List[int]  # 涉及的原子索引
    target_value: float      # 目标值
    tolerance: float = 0.5   # 容差
    weight: float = 1.0      # 损失权重


@dataclass  
class CatalyticConstraints:
    """从CatalyticTriadNet输出解析的催化约束"""
    # 功能位点定义
    anchor_atoms: List[Dict]  # 锚定原子 (必须存在的关键原子)
    
    # 几何约束
    distance_constraints: List[GeometricConstraint]
    angle_constraints: List[GeometricConstraint]
    coordination_constraints: List[Dict]
    
    # 电子性质约束
    charge_requirements: Dict[int, float]  # atom_idx -> target_charge
    
    # 化学约束
    required_elements: List[str]
    forbidden_elements: List[str]
    
    @classmethod
    def from_catalytic_triad_output(cls, json_path: str) -> 'CatalyticConstraints':
        """从CatalyticTriadNet的nanozyme_design.json解析约束"""
        with open(json_path) as f:
            data = json.load(f)
        
        anchor_atoms = []
        distance_constraints = []
        angle_constraints = []
        coordination_constraints = []
        required_elements = set()
        
        # 解析三联体
        for triad in data.get('catalytic_geometry', {}).get('triads', []):
            residues = triad.get('residues', [])
            distances = triad.get('distances', {})
            
            # 每个三联体残基转换为锚定原子
            for i, res in enumerate(residues):
                role = res.get('role', 'unknown')
                # 根据角色确定关键原子类型
                if 'nucleophile' in role:
                    anchor_atoms.append({
                        'idx': len(anchor_atoms),
                        'role': 'nucleophile',
                        'preferred_elements': ['O', 'S', 'N'],
                        'geometry': 'terminal'
                    })
                    required_elements.update(['O', 'S'])
                elif 'base' in role:
                    anchor_atoms.append({
                        'idx': len(anchor_atoms),
                        'role': 'general_base', 
                        'preferred_elements': ['N'],
                        'geometry': 'sp2'
                    })
                    required_elements.add('N')
                elif 'acid' in role or 'electrostatic' in role:
                    anchor_atoms.append({
                        'idx': len(anchor_atoms),
                        'role': 'electrostatic',
                        'preferred_elements': ['O'],
                        'geometry': 'carboxyl'
                    })
                    required_elements.add('O')
            
            # 距离约束
            for pair, dist in distances.items():
                parts = pair.split('-')
                if len(parts) == 2:
                    distance_constraints.append(GeometricConstraint(
                        constraint_type='distance',
                        atom_indices=[int(parts[0]) if parts[0].isdigit() else 0,
                                     int(parts[1]) if parts[1].isdigit() else 1],
                        target_value=dist if isinstance(dist, float) else dist,
                        tolerance=0.5,
                        weight=1.0
                    ))
        
        # 解析金属中心
        for mc in data.get('catalytic_geometry', {}).get('metal_centers', []):
            metal = mc.get('metal_type', 'Fe')
            coord_num = mc.get('coordination_number', 4)
            geometry = mc.get('geometry', 'tetrahedral')
            
            anchor_atoms.append({
                'idx': len(anchor_atoms),
                'role': 'metal_center',
                'preferred_elements': [metal],
                'geometry': geometry
            })
            required_elements.add(metal)
            
            coordination_constraints.append({
                'metal_idx': len(anchor_atoms) - 1,
                'coordination_number': coord_num,
                'geometry': geometry,
                'ligand_types': mc.get('ligand_types', ['N', 'O'])
            })
        
        # 解析双金属中心
        for bmc in data.get('catalytic_geometry', {}).get('bimetallic_centers', []):
            metals = bmc.get('metals', ['Fe', 'Fe'])
            dist = bmc.get('distance', 3.5)
            
            m1_idx = len(anchor_atoms)
            anchor_atoms.append({
                'idx': m1_idx,
                'role': 'metal_center',
                'preferred_elements': [metals[0]],
                'geometry': 'octahedral'
            })
            m2_idx = len(anchor_atoms)
            anchor_atoms.append({
                'idx': m2_idx,
                'role': 'metal_center', 
                'preferred_elements': [metals[1]],
                'geometry': 'octahedral'
            })
            
            distance_constraints.append(GeometricConstraint(
                constraint_type='distance',
                atom_indices=[m1_idx, m2_idx],
                target_value=dist,
                tolerance=0.3,
                weight=2.0  # 双金属距离更重要
            ))
            
            required_elements.update(metals)
        
        # 从distance_constraints解析
        for dc in data.get('distance_constraints', []):
            pair = dc.get('pair', '')
            target = dc.get('target', 3.0)
            tol = dc.get('tolerance', 0.5)
            # 简化处理
            distance_constraints.append(GeometricConstraint(
                constraint_type='distance',
                atom_indices=[0, 1],  # 需要后续映射
                target_value=target,
                tolerance=tol,
                weight=1.0
            ))
        
        return cls(
            anchor_atoms=anchor_atoms,
            distance_constraints=distance_constraints,
            angle_constraints=angle_constraints,
            coordination_constraints=coordination_constraints,
            charge_requirements={},
            required_elements=list(required_elements),
            forbidden_elements=[]
        )
    
    def to_condition_tensor(self, device='cpu') -> Dict[str, torch.Tensor]:
        """转换为条件张量供模型使用"""
        # 锚定原子特征
        n_anchors = len(self.anchor_atoms)
        anchor_features = torch.zeros(n_anchors, 16, device=device)
        
        for i, anchor in enumerate(self.anchor_atoms):
            # 编码角色
            role_map = {'nucleophile': 0, 'general_base': 1, 'electrostatic': 2, 
                       'metal_center': 3, 'proton_donor': 4, 'proton_acceptor': 5}
            role_idx = role_map.get(anchor['role'], 6)
            anchor_features[i, role_idx] = 1.0
            
            # 编码首选元素
            for elem in anchor.get('preferred_elements', []):
                if elem in ATOM_TO_IDX:
                    anchor_features[i, 7 + min(ATOM_TO_IDX[elem], 8)] = 1.0
        
        # 距离约束矩阵
        n_dist = len(self.distance_constraints)
        dist_matrix = torch.zeros(n_dist, 4, device=device)  # [i, j, target, weight]
        for k, dc in enumerate(self.distance_constraints):
            if len(dc.atom_indices) >= 2:
                dist_matrix[k, 0] = dc.atom_indices[0]
                dist_matrix[k, 1] = dc.atom_indices[1]
                dist_matrix[k, 2] = dc.target_value
                dist_matrix[k, 3] = dc.weight
        
        # 配位约束
        coord_features = []
        for cc in self.coordination_constraints:
            coord_features.append([
                cc['metal_idx'],
                cc['coordination_number'],
                {'tetrahedral': 0, 'square_planar': 1, 'octahedral': 2}.get(cc['geometry'], 0)
            ])
        coord_tensor = torch.tensor(coord_features, device=device) if coord_features else torch.zeros(0, 3, device=device)
        
        return {
            'anchor_features': anchor_features,
            'distance_constraints': dist_matrix,
            'coordination_constraints': coord_tensor,
            'n_anchors': n_anchors,
            'required_elements': self.required_elements
        }


# =============================================================================
# 3. E(3)-等变图神经网络层 (参考EGNN)
# =============================================================================

