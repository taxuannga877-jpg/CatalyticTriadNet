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

class E3EquivariantLayer(nn.Module):
    """
    E(3)-等变消息传递层
    
    保证输出在旋转、平移、反射下等变
    参考: https://github.com/vgsatorras/egnn
    """
    def __init__(self, hidden_dim: int, edge_dim: int = 0, 
                 act_fn: nn.Module = nn.SiLU(), residual: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual
        
        # 边缘MLP: 计算消息
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        
        # 节点MLP: 更新节点特征
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 坐标MLP: 更新坐标 (标量输出保证等变性)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # 注意力
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, h: torch.Tensor, x: torch.Tensor, 
                edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        """
        Args:
            h: 节点特征 [N, hidden_dim]
            x: 节点坐标 [N, 3]
            edge_index: 边索引 [2, E]
            edge_attr: 边特征 [E, edge_dim]
        
        Returns:
            h_new: 更新后的节点特征
            x_new: 更新后的坐标
        """
        row, col = edge_index
        
        # 计算相对位置和距离
        rel_pos = x[row] - x[col]  # [E, 3]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        
        # 边缘输入
        edge_input = [h[row], h[col], dist]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        edge_input = torch.cat(edge_input, dim=-1)
        
        # 计算消息
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_dim]
        
        # 注意力权重
        att = self.attention(m_ij)  # [E, 1]
        
        # 聚合消息到节点
        m_i = torch.zeros_like(h)
        m_i.index_add_(0, row, att * m_ij)
        
        # 更新节点特征
        h_input = torch.cat([h, m_i], dim=-1)
        h_new = self.node_mlp(h_input)
        if self.residual:
            h_new = h + h_new
        
        # 更新坐标 (等变)
        coord_diff = rel_pos / (dist + 1e-8)  # 归一化方向向量
        coord_weight = self.coord_mlp(m_ij)   # 标量权重
        
        coord_update = torch.zeros_like(x)
        coord_update.index_add_(0, row, coord_weight * coord_diff)
        x_new = x + coord_update
        
        return h_new, x_new


class EquivariantGNN(nn.Module):
    """E(3)-等变图神经网络"""
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 6,
                 edge_dim: int = 0, residual: bool = True):
        super().__init__()
        
        self.embedding = nn.Linear(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            E3EquivariantLayer(hidden_dim, edge_dim, residual=residual)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        h = self.embedding(h)
        
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        
        h = self.final_norm(h)
        return h, x


# =============================================================================
# 4. 条件扩散模型
# =============================================================================

class CatalyticDiffusionModel(nn.Module):
    """
    催化位点条件化扩散模型
    
    基于DDPM框架，使用E(3)-等变网络作为去噪网络
    条件信息通过cross-attention注入
    """
    def __init__(self, config: Dict = None):
        super().__init__()
        config = config or {}
        
        self.hidden_dim = config.get('hidden_dim', 256)
        self.n_layers = config.get('n_layers', 6)
        self.num_atom_types = NUM_ATOM_TYPES
        self.num_timesteps = config.get('num_timesteps', 1000)
        
        # 原子类型嵌入
        self.atom_embed = nn.Embedding(self.num_atom_types + 1, self.hidden_dim)  # +1 for mask
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 条件编码器
        self.condition_encoder = ConditionEncoder(
            anchor_dim=16,
            hidden_dim=self.hidden_dim
        )
        
        # E(3)-等变去噪网络
        self.denoiser = EquivariantDenoiser(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            condition_dim=self.hidden_dim
        )
        
        # 输出头
        self.atom_type_head = nn.Linear(self.hidden_dim, self.num_atom_types)
        self.coord_head = nn.Linear(self.hidden_dim, 3)
        
        # 扩散参数
        self._setup_diffusion_params()
    
    def _setup_diffusion_params(self):
        """设置扩散过程参数"""
        # 线性beta schedule
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, 
                 noise: torch.Tensor = None) -> torch.Tensor:
        """前向扩散: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def forward(self, atom_types: torch.Tensor, coords: torch.Tensor,
                edge_index: torch.Tensor, condition: Dict[str, torch.Tensor],
                t: torch.Tensor = None):
        """
        训练前向传播
        
        Args:
            atom_types: 原子类型 [B, N]
            coords: 原子坐标 [B, N, 3]
            edge_index: 边索引 [2, E]
            condition: 条件张量字典
            t: 时间步 [B]
        """
        B, N = atom_types.shape
        device = atom_types.device
        
        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=device)
        
        # 添加噪声
        noise = torch.randn_like(coords)
        noisy_coords = self.q_sample(coords, t, noise)
        
        # 编码
        h = self.atom_embed(atom_types)  # [B, N, hidden]
        t_emb = self.time_embed(t)       # [B, hidden]
        cond_emb = self.condition_encoder(condition)  # [B, hidden] or [B, M, hidden]
        
        # 去噪
        h_out, coord_pred = self.denoiser(
            h.view(B * N, -1),
            noisy_coords.view(B * N, 3),
            edge_index,
            t_emb,
            cond_emb
        )
        
        # 预测噪声
        pred_noise = self.coord_head(h_out).view(B, N, 3)
        pred_atom_logits = self.atom_type_head(h_out).view(B, N, -1)
        
        return {
            'pred_noise': pred_noise,
            'target_noise': noise,
            'pred_atom_logits': pred_atom_logits,
            'target_atom_types': atom_types
        }
    
    @torch.no_grad()
    def sample(self, condition: Dict[str, torch.Tensor], 
               n_atoms: int, n_samples: int = 1,
               guidance_scale: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        条件采样生成分子
        
        Args:
            condition: 催化约束条件
            n_atoms: 生成原子数
            n_samples: 生成样本数
            guidance_scale: 条件引导强度
        """
        device = next(self.parameters()).device
        
        # 初始化噪声
        x = torch.randn(n_samples, n_atoms, 3, device=device)
        atom_types = torch.randint(0, self.num_atom_types, (n_samples, n_atoms), device=device)
        
        # 构建全连接图
        edge_index = self._build_fc_edges(n_atoms, device)
        
        # 编码条件
        cond_emb = self.condition_encoder(condition)
        
        # 反向扩散采样
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            t_emb = self.time_embed(t_tensor)
            
            # 预测噪声
            h = self.atom_embed(atom_types).view(n_samples * n_atoms, -1)
            h_out, _ = self.denoiser(
                h, x.view(n_samples * n_atoms, 3),
                edge_index, t_emb, cond_emb
            )
            
            pred_noise = self.coord_head(h_out).view(n_samples, n_atoms, 3)
            pred_atom_logits = self.atom_type_head(h_out).view(n_samples, n_atoms, -1)
            
            # DDPM采样步
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            ) + torch.sqrt(beta) * noise
            
            # 更新原子类型 (采样)
            if t % 100 == 0:
                atom_types = pred_atom_logits.argmax(dim=-1)
        
        return {
            'atom_types': atom_types,
            'coords': x,
            'atom_type_logits': pred_atom_logits
        }
    
    def _build_fc_edges(self, n_atoms: int, device) -> torch.Tensor:
        """构建全连接边"""
        rows, cols = [], []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        return torch.tensor([rows, cols], dtype=torch.long, device=device)


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码 (用于时间步)"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionEncoder(nn.Module):
    """催化约束条件编码器"""
    def __init__(self, anchor_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        
        # 锚定原子编码
        self.anchor_encoder = nn.Sequential(
            nn.Linear(anchor_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 距离约束编码
        self.dist_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 配位约束编码
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, condition: Dict[str, torch.Tensor]) -> torch.Tensor:
        """编码条件"""
        anchor_feat = condition['anchor_features']
        dist_const = condition['distance_constraints']
        coord_const = condition['coordination_constraints']
        
        # 编码各部分
        anchor_emb = self.anchor_encoder(anchor_feat).mean(dim=0, keepdim=True)  # [1, hidden]
        
        if dist_const.shape[0] > 0:
            dist_emb = self.dist_encoder(dist_const).mean(dim=0, keepdim=True)
        else:
            dist_emb = torch.zeros_like(anchor_emb)
        
        if coord_const.shape[0] > 0:
            coord_emb = self.coord_encoder(coord_const.float()).mean(dim=0, keepdim=True)
        else:
            coord_emb = torch.zeros_like(anchor_emb)
        
        # 融合
        combined = torch.cat([anchor_emb, dist_emb, coord_emb], dim=-1)
        return self.fusion(combined)


class EquivariantDenoiser(nn.Module):
    """E(3)-等变去噪网络"""
    def __init__(self, hidden_dim: int, n_layers: int, condition_dim: int):
        super().__init__()
        
        # 时间嵌入投影
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 条件投影
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        
        # 等变层
        self.layers = nn.ModuleList([
            E3EquivariantLayer(hidden_dim, edge_dim=0, residual=True)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, t_emb: torch.Tensor,
                cond_emb: torch.Tensor):
        """
        Args:
            h: 节点特征 [N, hidden]
            x: 节点坐标 [N, 3]
            edge_index: 边索引
            t_emb: 时间嵌入 [B, hidden]
            cond_emb: 条件嵌入 [B, hidden] or [1, hidden]
        """
        # 注入时间和条件信息
        t_proj = self.time_proj(t_emb)  # [B, hidden]
        c_proj = self.cond_proj(cond_emb)  # [B, hidden]
        
        # 广播到所有节点
        N = h.shape[0]
        B = t_emb.shape[0]
        nodes_per_sample = N // B if B > 0 else N
        
        # 简化: 假设batch_size=1或均匀分布
        h = h + t_proj.repeat(nodes_per_sample, 1) + c_proj.repeat(nodes_per_sample, 1)
        
        # 等变层
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        
        return self.norm(h), x


# =============================================================================
# 5. 约束损失函数
# =============================================================================

class ConstraintLoss(nn.Module):
    """几何约束损失"""
    def __init__(self, distance_weight: float = 1.0,
                 coordination_weight: float = 1.0):
        super().__init__()
        self.distance_weight = distance_weight
        self.coordination_weight = coordination_weight
    
    def forward(self, coords: torch.Tensor, 
                constraints: CatalyticConstraints) -> torch.Tensor:
        """
        计算约束违反损失
        
        Args:
            coords: 生成的坐标 [N, 3]
            constraints: 催化约束
        """
        total_loss = 0.0
        
        # 距离约束损失
        for dc in constraints.distance_constraints:
            i, j = dc.atom_indices
            if i < coords.shape[0] and j < coords.shape[0]:
                actual_dist = torch.norm(coords[i] - coords[j])
                target_dist = dc.target_value
                tolerance = dc.tolerance
                
                # Huber-like损失
                diff = torch.abs(actual_dist - target_dist)
                if diff > tolerance:
                    loss = dc.weight * (diff - tolerance) ** 2
                else:
                    loss = 0.0
                total_loss += loss
        
        return total_loss * self.distance_weight


# =============================================================================
# 6. 主生成器接口
# =============================================================================

class CatalyticNanozymeGenerator:
    """
    纳米酶结构生成器主接口
    
    使用方法:
        generator = CatalyticNanozymeGenerator()
        
        # 加载催化位点模板
        constraints = CatalyticConstraints.from_catalytic_triad_output(
            'catalytic_triad_output_nanozyme.json'
        )
        
        # 生成纳米酶结构
        structures = generator.generate(
            constraints,
            n_samples=10,
            n_atoms=50
        )
    """
    def __init__(self, model_path: str = None, config: Dict = None, device: str = None):
        self.config = config or {
            'hidden_dim': 256,
            'n_layers': 6,
            'num_timesteps': 1000
        }
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = CatalyticDiffusionModel(self.config).to(self.device)
        
        # 加载预训练权重
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
        
        self.model.eval()
        self.constraint_loss = ConstraintLoss()
    
    def generate(self, constraints: CatalyticConstraints,
                 n_samples: int = 10,
                 n_atoms: int = None,
                 guidance_scale: float = 2.0,
                 refine_steps: int = 100) -> List[Dict]:
        """
        生成满足催化约束的分子结构
        
        Args:
            constraints: 催化位点约束
            n_samples: 生成样本数
            n_atoms: 原子数 (None则自动估计)
            guidance_scale: 条件引导强度
            refine_steps: 后处理优化步数
        
        Returns:
            生成的分子结构列表
        """
        # 自动估计原子数
        if n_atoms is None:
            n_atoms = max(20, len(constraints.anchor_atoms) * 5)
        
        # 转换条件
        condition = constraints.to_condition_tensor(self.device)
        
        # 扩散采样
        with torch.no_grad():
            samples = self.model.sample(
                condition, n_atoms, n_samples, guidance_scale
            )
        
        # 后处理: 约束优化
        if refine_steps > 0:
            samples = self._refine_with_constraints(samples, constraints, refine_steps)
        
        # 转换为输出格式
        results = []
        for i in range(n_samples):
            atom_types = samples['atom_types'][i].cpu().numpy()
            coords = samples['coords'][i].cpu().numpy()
            
            # 转换原子类型索引到符号
            atom_symbols = [ATOM_TYPES[idx] if idx < len(ATOM_TYPES) else 'C' 
                          for idx in atom_types]
            
            result = {
                'atom_types': atom_symbols,
                'coords': coords.tolist(),
                'n_atoms': n_atoms,
                'constraint_satisfaction': self._evaluate_constraints(coords, constraints),
                'validity_scores': self._compute_validity(atom_symbols, coords)
            }
            results.append(result)
        
        return results
    
    def _refine_with_constraints(self, samples: Dict, 
                                  constraints: CatalyticConstraints,
                                  n_steps: int) -> Dict:
        """使用约束损失优化生成结构"""
        coords = samples['coords'].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([coords], lr=0.01)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # 计算约束损失
            loss = 0.0
            for i in range(coords.shape[0]):
                loss += self.constraint_loss(coords[i], constraints)
            
            if loss > 0:
                loss.backward()
                optimizer.step()
            
            if step % 20 == 0:
                logger.debug(f"Refine step {step}, loss: {loss.item():.4f}")
        
        samples['coords'] = coords.detach()
        return samples
    
    def _evaluate_constraints(self, coords: np.ndarray, 
                              constraints: CatalyticConstraints) -> Dict:
        """评估约束满足度"""
        results = {'distance': [], 'coordination': []}
        
        for dc in constraints.distance_constraints:
            i, j = dc.atom_indices
            if i < len(coords) and j < len(coords):
                actual = np.linalg.norm(coords[i] - coords[j])
                target = dc.target_value
                error = abs(actual - target)
                satisfied = error <= dc.tolerance
                results['distance'].append({
                    'pair': (i, j),
                    'target': target,
                    'actual': actual,
                    'error': error,
                    'satisfied': satisfied
                })
        
        return results
    
    def _compute_validity(self, atom_types: List[str], 
                          coords: np.ndarray) -> Dict:
        """计算结构有效性分数"""
        # 检查原子间距
        n = len(coords)
        min_dist = float('inf')
        clash_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                min_dist = min(min_dist, d)
                if d < 0.8:  # 太近,冲突
                    clash_count += 1
        
        # 检查连通性 (简化)
        connected = min_dist < 5.0  # 至少有些原子在合理距离内
        
        return {
            'min_distance': min_dist,
            'clash_count': clash_count,
            'has_clashes': clash_count > 0,
            'connected': connected
        }
    
    def to_xyz(self, result: Dict, filepath: str):
        """导出为XYZ格式"""
        atoms = result['atom_types']
        coords = result['coords']
        
        with open(filepath, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"Generated nanozyme structure\n")
            for atom, coord in zip(atoms, coords):
                f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    
    def to_mol(self, result: Dict, filepath: str):
        """导出为MOL格式 (需要推断键)"""
        # 简化版: 基于距离推断键
        atoms = result['atom_types']
        coords = np.array(result['coords'])
        n = len(atoms)
        
        # 推断键 (基于共价半径)
        bonds = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                # 简化: 1.8Å内视为成键
                if d < 2.5:
                    bonds.append((i + 1, j + 1, 1))  # MOL索引从1开始
        
        with open(filepath, 'w') as f:
            f.write("Generated Nanozyme\n")
            f.write("  CatalyticDiff\n\n")
            f.write(f"{n:3d}{len(bonds):3d}  0  0  0  0  0  0  0  0999 V2000\n")
            
            for atom, coord in zip(atoms, coords):
                f.write(f"{coord[0]:10.4f}{coord[1]:10.4f}{coord[2]:10.4f} "
                       f"{atom:>3s}  0  0  0  0  0  0  0  0  0  0  0  0\n")
            
            for b in bonds:
                f.write(f"{b[0]:3d}{b[1]:3d}{b[2]:3d}  0  0  0  0\n")
            
            f.write("M  END\n")


# =============================================================================
# 7. 训练接口
# =============================================================================

class NanozymeDataset(Dataset):
    """纳米酶/催化分子数据集"""
    def __init__(self, data_dir: str, max_atoms: int = 100):
        self.data_dir = Path(data_dir)
        self.max_atoms = max_atoms
        self.samples = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """加载数据"""
        samples = []
        # 支持多种格式
        for ext in ['*.xyz', '*.mol', '*.json']:
            for f in self.data_dir.glob(ext):
                try:
                    sample = self._parse_file(f)
                    if sample and len(sample['atom_types']) <= self.max_atoms:
                        samples.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to parse {f}: {e}")
        return samples
    
    def _parse_file(self, filepath: Path) -> Optional[Dict]:
        """解析分子文件"""
        if filepath.suffix == '.xyz':
            return self._parse_xyz(filepath)
        elif filepath.suffix == '.json':
            with open(filepath) as f:
                return json.load(f)
        return None
    
    def _parse_xyz(self, filepath: Path) -> Dict:
        """解析XYZ文件"""
        with open(filepath) as f:
            lines = f.readlines()
        
        n_atoms = int(lines[0].strip())
        atoms = []
        coords = []
        
        for line in lines[2:2 + n_atoms]:
            parts = line.split()
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        return {
            'atom_types': atoms,
            'coords': coords,
            'name': filepath.stem
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 转换原子类型到索引
        atom_indices = [ATOM_TO_IDX.get(a, 0) for a in sample['atom_types']]
        coords = np.array(sample['coords'], dtype=np.float32)
        
        # 中心化坐标
        coords = coords - coords.mean(axis=0)
        
        return {
            'atom_types': torch.tensor(atom_indices, dtype=torch.long),
            'coords': torch.tensor(coords, dtype=torch.float32)
        }


class Trainer:
    """模型训练器"""
    def __init__(self, model: CatalyticDiffusionModel, 
                 train_dataset: NanozymeDataset,
                 config: Dict = None):
        self.model = model
        self.dataset = train_dataset
        self.config = config or {}
        
        self.lr = self.config.get('lr', 1e-4)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 100)
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
    
    def train(self):
        """训练循环"""
        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, 
            shuffle=True, collate_fn=self._collate_fn
        )
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch in dataloader:
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(
                    batch['atom_types'],
                    batch['coords'],
                    batch['edge_index'],
                    batch['condition']
                )
                
                # 计算损失
                loss = F.mse_loss(outputs['pred_noise'], outputs['target_noise'])
                loss += F.cross_entropy(
                    outputs['pred_atom_logits'].view(-1, NUM_ATOM_TYPES),
                    outputs['target_atom_types'].view(-1)
                )
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            self.scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def _collate_fn(self, batch):
        """批次整理函数"""
        # 简化版: 假设同一batch内原子数相同或进行padding
        atom_types = torch.stack([b['atom_types'] for b in batch])
        coords = torch.stack([b['coords'] for b in batch])
        
        # 构建边
        n_atoms = atom_types.shape[1]
        edge_index = self._build_edges(n_atoms, atom_types.device)
        
        # 空条件 (训练时无监督)
        condition = {
            'anchor_features': torch.zeros(1, 16),
            'distance_constraints': torch.zeros(0, 4),
            'coordination_constraints': torch.zeros(0, 3)
        }
        
        return {
            'atom_types': atom_types,
            'coords': coords,
            'edge_index': edge_index,
            'condition': condition
        }
    
    def _build_edges(self, n, device):
        rows, cols = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        return torch.tensor([rows, cols], dtype=torch.long, device=device)


# =============================================================================
# 8. 命令行接口
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CatalyticDiff: 催化位点条件化纳米酶分子生成'
    )
    subparsers = parser.add_subparsers(dest='command')
    
    # 生成命令
    gen_parser = subparsers.add_parser('generate', help='生成纳米酶结构')
    gen_parser.add_argument('--constraints', required=True, 
                           help='催化约束JSON文件 (CatalyticTriadNet输出)')
    gen_parser.add_argument('--model', default=None, help='模型检查点路径')
    gen_parser.add_argument('--n_samples', type=int, default=10, help='生成样本数')
    gen_parser.add_argument('--n_atoms', type=int, default=None, help='原子数')
    gen_parser.add_argument('--output', default='./generated', help='输出目录')
    gen_parser.add_argument('--format', choices=['xyz', 'mol', 'json'], 
                           default='xyz', help='输出格式')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data_dir', required=True, help='训练数据目录')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--save_path', default='./models/catalytic_diff.pt')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        # 加载约束
        constraints = CatalyticConstraints.from_catalytic_triad_output(args.constraints)
        logger.info(f"Loaded constraints: {len(constraints.anchor_atoms)} anchor atoms, "
                   f"{len(constraints.distance_constraints)} distance constraints")
        
        # 初始化生成器
        generator = CatalyticNanozymeGenerator(model_path=args.model)
        
        # 生成
        results = generator.generate(
            constraints,
            n_samples=args.n_samples,
            n_atoms=args.n_atoms
        )
        
        # 保存
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results):
            if args.format == 'xyz':
                generator.to_xyz(result, output_dir / f'nanozyme_{i:03d}.xyz')
            elif args.format == 'mol':
                generator.to_mol(result, output_dir / f'nanozyme_{i:03d}.mol')
            else:
                with open(output_dir / f'nanozyme_{i:03d}.json', 'w') as f:
                    json.dump(result, f, indent=2)
        
        logger.info(f"Generated {len(results)} structures in {output_dir}")
        
        # 打印约束满足统计
        for i, r in enumerate(results):
            sat = r['constraint_satisfaction']
            n_sat = sum(1 for d in sat.get('distance', []) if d['satisfied'])
            n_total = len(sat.get('distance', []))
            logger.info(f"  Sample {i}: {n_sat}/{n_total} distance constraints satisfied")
    
    elif args.command == 'train':
        dataset = NanozymeDataset(args.data_dir)
        logger.info(f"Loaded {len(dataset)} training samples")
        
        model = CatalyticDiffusionModel()
        trainer = Trainer(model, dataset, {'epochs': args.epochs, 'lr': args.lr})
        trainer.train()
        
        # 保存模型
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        logger.info(f"Model saved to {args.save_path}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
