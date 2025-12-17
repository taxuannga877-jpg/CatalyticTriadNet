#!/usr/bin/env python3
"""
催化约束模块

定义催化位点的几何约束、化学约束和电子性质约束，
用于条件化分子生成模型。

包含:
- 原子类型和特征定义
- 几何约束类（距离、角度、配位）
- 催化约束解析器
- 约束损失函数
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import re

import torch
import torch.nn as nn

# 导入配置系统
try:
    from ..config import get_config
except ImportError:
    get_config = None

logger = logging.getLogger(__name__)


# =============================================================================
# 原子类型和特征定义
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
# 几何约束类
# =============================================================================

@dataclass
class GeometricConstraint:
    """
    几何约束数据类

    Attributes:
        constraint_type: 约束类型 ('distance', 'angle', 'dihedral', 'coordination')
        atom_indices: 涉及的原子索引列表
        target_value: 目标值（距离单位：Å，角度单位：度）
        tolerance: 容差
        weight: 损失权重
    """
    constraint_type: str
    atom_indices: List[int]
    target_value: float
    tolerance: float = 0.5
    weight: float = 1.0

    def __post_init__(self):
        """验证约束参数"""
        assert self.constraint_type in ['distance', 'angle', 'dihedral', 'coordination'], \
            f"Invalid constraint type: {self.constraint_type}"
        assert len(self.atom_indices) >= 2, "At least 2 atom indices required"
        assert self.target_value > 0, "Target value must be positive"
        assert self.tolerance >= 0, "Tolerance must be non-negative"
        assert self.weight >= 0, "Weight must be non-negative"


@dataclass
class CatalyticConstraints:
    """
    催化约束集合

    从CatalyticTriadNet输出解析的催化位点约束，包括：
    - 锚定原子（必须存在的关键原子）
    - 几何约束（距离、角度、配位）
    - 电子性质约束
    - 化学约束（必需/禁止元素）

    Attributes:
        anchor_atoms: 锚定原子列表，每个为包含role、preferred_elements等的字典
        distance_constraints: 距离约束列表
        angle_constraints: 角度约束列表
        coordination_constraints: 配位约束列表
        charge_requirements: 电荷要求字典 {atom_idx: target_charge}
        required_elements: 必需元素列表
        forbidden_elements: 禁止元素列表
    """
    anchor_atoms: List[Dict]
    distance_constraints: List[GeometricConstraint]
    angle_constraints: List[GeometricConstraint]
    coordination_constraints: List[Dict]
    charge_requirements: Dict[int, float]
    required_elements: List[str]
    forbidden_elements: List[str]

    @classmethod
    def from_catalytic_triad_output(cls, json_path: Union[str, Path]) -> 'CatalyticConstraints':
        """
        从CatalyticTriadNet的输出JSON文件解析约束

        Args:
            json_path: JSON文件路径

        Returns:
            CatalyticConstraints实例

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON格式错误
            KeyError: 缺少必需字段
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Constraint file not found: {json_path}")

        try:
            with open(json_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {json_path}: {e}")
            raise

        anchor_atoms = []
        distance_constraints = []
        angle_constraints = []
        coordination_constraints = []
        required_elements = set()

        # 解析三联体
        triads = data.get('catalytic_geometry', {}).get('triads', [])
        for triad_idx, triad in enumerate(triads):
            residues = triad.get('residues', [])
            distances = triad.get('distances', {})

            # 每个三联体残基转换为锚定原子
            for res_idx, res in enumerate(residues):
                role = res.get('role', 'unknown')

                # 根据角色确定关键原子类型
                if 'nucleophile' in role.lower():
                    anchor_atoms.append({
                        'idx': len(anchor_atoms),
                        'role': 'nucleophile',
                        'preferred_elements': ['O', 'S', 'N'],
                        'geometry': 'terminal',
                        'source': f'triad_{triad_idx}_res_{res_idx}'
                    })
                    required_elements.update(['O', 'S'])
                elif 'base' in role.lower():
                    anchor_atoms.append({
                        'idx': len(anchor_atoms),
                        'role': 'general_base',
                        'preferred_elements': ['N'],
                        'geometry': 'sp2',
                        'source': f'triad_{triad_idx}_res_{res_idx}'
                    })
                    required_elements.add('N')
                elif 'acid' in role.lower() or 'electrostatic' in role.lower():
                    anchor_atoms.append({
                        'idx': len(anchor_atoms),
                        'role': 'electrostatic',
                        'preferred_elements': ['O'],
                        'geometry': 'carboxyl',
                        'source': f'triad_{triad_idx}_res_{res_idx}'
                    })
                    required_elements.add('O')

            # 距离约束
            for pair, dist in distances.items():
                parts = pair.split('-')
                if len(parts) == 2:
                    try:
                        idx1 = int(parts[0]) if parts[0].isdigit() else 0
                        idx2 = int(parts[1]) if parts[1].isdigit() else 1
                        dist_value = float(dist) if isinstance(dist, (int, float)) else 3.0

                        distance_constraints.append(GeometricConstraint(
                            constraint_type='distance',
                            atom_indices=[idx1, idx2],
                            target_value=dist_value,
                            tolerance=0.5,
                            weight=1.0
                        ))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse distance constraint {pair}: {e}")

        # 解析金属中心
        metal_centers = data.get('catalytic_geometry', {}).get('metal_centers', [])
        for mc_idx, mc in enumerate(metal_centers):
            metal = mc.get('metal_type', 'Fe')
            coord_num = mc.get('coordination_number', 4)
            geometry = mc.get('geometry', 'tetrahedral')

            anchor_atoms.append({
                'idx': len(anchor_atoms),
                'role': 'metal_center',
                'preferred_elements': [metal],
                'geometry': geometry,
                'source': f'metal_center_{mc_idx}'
            })
            required_elements.add(metal)

            coordination_constraints.append({
                'metal_idx': len(anchor_atoms) - 1,
                'coordination_number': coord_num,
                'geometry': geometry,
                'ligand_types': mc.get('ligand_types', ['N', 'O'])
            })

        # 解析双金属中心
        bimetallic_centers = data.get('catalytic_geometry', {}).get('bimetallic_centers', [])
        for bmc_idx, bmc in enumerate(bimetallic_centers):
            metals = bmc.get('metals', ['Fe', 'Fe'])
            dist = bmc.get('distance', 3.5)

            m1_idx = len(anchor_atoms)
            anchor_atoms.append({
                'idx': m1_idx,
                'role': 'metal_center',
                'preferred_elements': [metals[0]],
                'geometry': 'octahedral',
                'source': f'bimetallic_{bmc_idx}_metal1'
            })

            m2_idx = len(anchor_atoms)
            anchor_atoms.append({
                'idx': m2_idx,
                'role': 'metal_center',
                'preferred_elements': [metals[1]],
                'geometry': 'octahedral',
                'source': f'bimetallic_{bmc_idx}_metal2'
            })

            distance_constraints.append(GeometricConstraint(
                constraint_type='distance',
                atom_indices=[m1_idx, m2_idx],
                target_value=dist,
                tolerance=0.3,
                weight=2.0  # 双金属距离更重要
            ))

            required_elements.update(metals)

        # 从distance_constraints字段解析
        for dc_idx, dc in enumerate(data.get('distance_constraints', [])):
            pair = dc.get('pair', '')
            target = dc.get('target', 3.0)
            tol = dc.get('tolerance', 0.5)

            # 优先使用显式的 atom_indices
            atom_indices = dc.get('atom_indices')

            # 如果未提供，则尝试从 pair 字符串解析，例如 "0-1" 或 "2_3"
            if atom_indices is None and isinstance(pair, str):
                parts = re.split(r'[-_]', pair)
                if len(parts) == 2 and all(p.isdigit() for p in parts):
                    atom_indices = [int(parts[0]), int(parts[1])]

            # 如果仍未得到有效索引，则跳过该约束
            if not atom_indices or len(atom_indices) < 2:
                logger.warning(f"Skipping distance constraint {dc_idx}: invalid atom indices ({pair})")
                continue

            i_idx, j_idx = atom_indices[0], atom_indices[1]
            if i_idx >= len(anchor_atoms) or j_idx >= len(anchor_atoms) or i_idx < 0 or j_idx < 0:
                logger.warning(
                    f"Skipping distance constraint {dc_idx}: indices out of range "
                    f"({i_idx}, {j_idx}) for {len(anchor_atoms)} anchors"
                )
                continue

            distance_constraints.append(GeometricConstraint(
                constraint_type='distance',
                atom_indices=[i_idx, j_idx],
                target_value=target,
                tolerance=tol,
                weight=1.0
            ))

        logger.info(f"Parsed constraints: {len(anchor_atoms)} anchors, "
                   f"{len(distance_constraints)} distance constraints, "
                   f"{len(coordination_constraints)} coordination constraints")

        return cls(
            anchor_atoms=anchor_atoms,
            distance_constraints=distance_constraints,
            angle_constraints=angle_constraints,
            coordination_constraints=coordination_constraints,
            charge_requirements={},
            required_elements=list(required_elements),
            forbidden_elements=[]
        )

    def to_condition_tensor(self, device: Union[str, torch.device] = 'cpu',
                           dtype: torch.dtype = torch.float32,
                           batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        转换为条件张量供模型使用

        Args:
            device: 目标设备
            dtype: 数据类型
            batch_size: 批次大小（如果提供，则复制到批次维度）

        Returns:
            条件张量字典，包含：
                - anchor_features: [n_anchors, 16] 或 [B, n_anchors, 16]
                - distance_constraints: [n_dist, 4] 或 [B, n_dist, 4]
                - coordination_constraints: [n_coord, 3] 或 [B, n_coord, 3]
                - n_anchors: int
                - required_elements: List[str]
        """
        # 锚定原子特征
        n_anchors = len(self.anchor_atoms)
        anchor_features = torch.zeros(n_anchors, 16, device=device, dtype=dtype)

        role_map = {
            'nucleophile': 0, 'general_base': 1, 'electrostatic': 2,
            'metal_center': 3, 'proton_donor': 4, 'proton_acceptor': 5
        }

        for i, anchor in enumerate(self.anchor_atoms):
            # 编码角色
            role_idx = role_map.get(anchor.get('role', 'unknown'), 6)
            anchor_features[i, role_idx] = 1.0

            # 编码首选元素
            for elem in anchor.get('preferred_elements', []):
                if elem in ATOM_TO_IDX:
                    elem_idx = min(ATOM_TO_IDX[elem], 8)
                    anchor_features[i, 7 + elem_idx] = 1.0

        # 距离约束矩阵
        n_dist = len(self.distance_constraints)
        dist_matrix = torch.zeros(n_dist, 4, device=device, dtype=dtype)

        for k, dc in enumerate(self.distance_constraints):
            if len(dc.atom_indices) >= 2:
                dist_matrix[k, 0] = float(dc.atom_indices[0])
                dist_matrix[k, 1] = float(dc.atom_indices[1])
                dist_matrix[k, 2] = float(dc.target_value)
                dist_matrix[k, 3] = float(dc.weight)

        # 配位约束
        coord_features = []
        geometry_map = {'tetrahedral': 0, 'square_planar': 1, 'octahedral': 2}

        for cc in self.coordination_constraints:
            coord_features.append([
                float(cc['metal_idx']),
                float(cc['coordination_number']),
                float(geometry_map.get(cc['geometry'], 0))
            ])

        coord_tensor = torch.tensor(
            coord_features, device=device, dtype=dtype
        ) if coord_features else torch.zeros(0, 3, device=device, dtype=dtype)

        # 如果指定批次大小，则扩展到批次维度
        if batch_size is not None and batch_size > 1:
            anchor_features = anchor_features.unsqueeze(0).expand(batch_size, -1, -1)
            if n_dist > 0:
                dist_matrix = dist_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            if coord_tensor.shape[0] > 0:
                coord_tensor = coord_tensor.unsqueeze(0).expand(batch_size, -1, -1)

        return {
            'anchor_features': anchor_features,
            'distance_constraints': dist_matrix,
            'coordination_constraints': coord_tensor,
            'n_anchors': n_anchors,
            'required_elements': self.required_elements
        }


# =============================================================================
# 约束损失函数
# =============================================================================

class ConstraintLoss(nn.Module):
    """
    几何约束损失函数

    计算生成结构违反约束的程度，用于优化和评估。
    使用张量安全的操作，避免布尔比较问题。

    Args:
        distance_weight: 距离约束权重
        coordination_weight: 配位约束权重
        angle_weight: 角度约束权重
    """

    def __init__(self,
                 distance_weight: float = 1.0,
                 coordination_weight: float = 1.0,
                 angle_weight: float = 1.0):
        super().__init__()
        self.distance_weight = distance_weight
        self.coordination_weight = coordination_weight
        self.angle_weight = angle_weight

        # 从配置读取权重
        if get_config:
            config = get_config()
            gen_config = config.get('generation', {})
            self.distance_weight = gen_config.get('constraint_distance_weight', distance_weight)
            self.coordination_weight = gen_config.get('constraint_coordination_weight', coordination_weight)
            self.angle_weight = gen_config.get('constraint_angle_weight', angle_weight)

    def forward(self, coords: torch.Tensor,
                constraints: CatalyticConstraints) -> torch.Tensor:
        """
        计算约束违反损失

        Args:
            coords: 生成的坐标 [N, 3] 或 [B, N, 3]
            constraints: 催化约束

        Returns:
            总损失标量
        """
        # 处理批次维度
        if coords.dim() == 3:
            # 批次模式: [B, N, 3]
            batch_size = coords.shape[0]
            total_loss = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

            for b in range(batch_size):
                total_loss = total_loss + self._compute_single_loss(coords[b], constraints)

            return total_loss / batch_size
        else:
            # 单样本模式: [N, 3]
            return self._compute_single_loss(coords, constraints)

    def _compute_single_loss(self, coords: torch.Tensor,
                            constraints: CatalyticConstraints) -> torch.Tensor:
        """
        计算单个样本的约束损失

        Args:
            coords: 坐标 [N, 3]
            constraints: 催化约束

        Returns:
            损失标量
        """
        distance_loss = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        angle_loss = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        coordination_loss = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        n_atoms = coords.shape[0]

        # 距离约束损失
        for dc in constraints.distance_constraints:
            i, j = dc.atom_indices[0], dc.atom_indices[1]

            # 边界检查
            if i >= n_atoms or j >= n_atoms or i < 0 or j < 0:
                logger.warning(f"Invalid atom indices in constraint: {i}, {j} (n_atoms={n_atoms})")
                continue

            # 计算实际距离
            actual_dist = torch.norm(coords[i] - coords[j])
            target_dist = torch.tensor(dc.target_value, device=coords.device, dtype=coords.dtype)
            tolerance = torch.tensor(dc.tolerance, device=coords.device, dtype=coords.dtype)

            # 使用 torch.clamp 替代 if 语句，确保张量安全
            # 只有当距离偏差超过容差时才计算损失
            diff = torch.abs(actual_dist - target_dist)
            violation = torch.clamp(diff - tolerance, min=0.0)
            loss = dc.weight * (violation ** 2)

            distance_loss = distance_loss + loss

        # 角度约束损失（如果有）
        for ac in constraints.angle_constraints:
            if len(ac.atom_indices) >= 3:
                i, j, k = ac.atom_indices[0], ac.atom_indices[1], ac.atom_indices[2]

                if i >= n_atoms or j >= n_atoms or k >= n_atoms or i < 0 or j < 0 or k < 0:
                    continue

                # 计算角度
                v1 = coords[i] - coords[j]
                v2 = coords[k] - coords[j]

                # 避免除零
                v1_norm = torch.norm(v1) + 1e-8
                v2_norm = torch.norm(v2) + 1e-8

                cos_angle = torch.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # 数值稳定性
                actual_angle = torch.acos(cos_angle) * 180.0 / 3.14159265359

                target_angle = torch.tensor(ac.target_value, device=coords.device, dtype=coords.dtype)
                tolerance = torch.tensor(ac.tolerance, device=coords.device, dtype=coords.dtype)

                diff = torch.abs(actual_angle - target_angle)
                violation = torch.clamp(diff - tolerance, min=0.0)
                loss = ac.weight * (violation ** 2)

                angle_loss = angle_loss + loss

        # 配位约束损失（简单计数约束）
        if constraints.coordination_constraints:
            # 简单的几何映射用于选择邻域半径
            geometry_cutoff = {
                'tetrahedral': 2.8,
                'square_planar': 2.6,
                'octahedral': 3.0
            }

            for cc in constraints.coordination_constraints:
                metal_idx = cc.get('metal_idx')
                target_coord = cc.get('coordination_number', 0)
                geom = cc.get('geometry', 'octahedral')

                if metal_idx is None or metal_idx < 0 or metal_idx >= n_atoms:
                    logger.warning(f"Invalid metal_idx in coordination constraint: {metal_idx}")
                    continue

                cutoff = geometry_cutoff.get(geom, 3.0)

                # 计算邻居数量（不包含自身）
                dists = torch.norm(coords - coords[metal_idx], dim=1)
                neighbor_mask = (dists < cutoff) & (dists > 1e-6)
                neighbor_count = neighbor_mask.sum().float()

                diff = torch.abs(neighbor_count - torch.tensor(float(target_coord), device=coords.device, dtype=coords.dtype))
                coordination_loss = coordination_loss + diff ** 2

        return (
            self.distance_weight * distance_loss +
            self.angle_weight * angle_loss +
            self.coordination_weight * coordination_loss
        )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'ATOM_TYPES',
    'ATOM_TO_IDX',
    'NUM_ATOM_TYPES',
    'ATOM_FEATURES',
    'FUNCTIONAL_GROUP_TEMPLATES',
    'GeometricConstraint',
    'CatalyticConstraints',
    'ConstraintLoss',
]
