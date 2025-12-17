#!/usr/bin/env python3
"""
纳米酶生成器模块
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from .models import CatalyticDiffusionModel, ConstraintLoss
from .constraints import CatalyticConstraints, ATOM_TYPES

logger = logging.getLogger(__name__)

# 常量定义
DEFAULT_HIDDEN_DIM = 256
DEFAULT_N_LAYERS = 6
DEFAULT_NUM_TIMESTEPS = 1000
DEFAULT_GUIDANCE_SCALE = 2.0
DEFAULT_REFINE_STEPS = 100
DEFAULT_ATOMS_PER_ANCHOR = 5
MIN_ATOMS = 20

# 结构有效性检查常量
MIN_ATOM_DISTANCE = 0.8  # Å - 原子间最小距离（避免冲突）
MAX_BOND_DISTANCE = 2.5  # Å - 推断键的最大距离
MAX_CONNECTIVITY_DISTANCE = 5.0  # Å - 判断连通性的最大距离

# 优化参数
REFINE_LEARNING_RATE = 0.01
REFINE_LOG_INTERVAL = 20

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
        """
        初始化纳米酶生成器

        Args:
            model_path: 预训练模型路径
            config: 模型配置字典
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.config = config or {
            'hidden_dim': DEFAULT_HIDDEN_DIM,
            'n_layers': DEFAULT_N_LAYERS,
            'num_timesteps': DEFAULT_NUM_TIMESTEPS
        }
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        self.model = CatalyticDiffusionModel(self.config).to(self.device)

        # 加载预训练权重
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"已从 {model_path} 加载模型")
        elif model_path:
            logger.warning(f"模型文件不存在: {model_path}")

        self.model.eval()
        self.constraint_loss = ConstraintLoss()
    
    def generate(self, constraints: CatalyticConstraints,
                 n_samples: int = 10,
                 n_atoms: int = None,
                 guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
                 refine_steps: int = DEFAULT_REFINE_STEPS) -> List[Dict]:
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
            n_atoms = max(MIN_ATOMS, len(constraints.anchor_atoms) * DEFAULT_ATOMS_PER_ANCHOR)
        
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
        """
        使用约束损失优化生成结构

        Args:
            samples: 生成的样本字典
            constraints: 催化约束
            n_steps: 优化步数

        Returns:
            优化后的样本字典
        """
        coords = samples['coords'].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([coords], lr=REFINE_LEARNING_RATE)

        for step in range(n_steps):
            optimizer.zero_grad()

            # 计算约束损失（保持张量类型，避免布尔歧义）
            loss = torch.zeros(1, device=coords.device, dtype=coords.dtype)
            for i in range(coords.shape[0]):
                loss += self.constraint_loss(coords[i], constraints)

            loss.backward()
            optimizer.step()

            if step % REFINE_LOG_INTERVAL == 0:
                logger.debug(f"优化步骤 {step}, 损失: {loss.item():.4f}")

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
        """
        计算结构有效性分数

        Args:
            atom_types: 原子类型列表
            coords: 原子坐标数组

        Returns:
            包含有效性指标的字典
        """
        # 检查原子间距
        n = len(coords)
        min_dist = float('inf')
        clash_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                min_dist = min(min_dist, d)
                if d < MIN_ATOM_DISTANCE:  # 原子冲突
                    clash_count += 1

        # 检查连通性（简化版：至少有些原子在合理距离内）
        connected = min_dist < MAX_CONNECTIVITY_DISTANCE

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
        """
        导出为MOL格式（需要推断键）

        Args:
            result: 生成结果字典
            filepath: 输出文件路径
        """
        # 简化版：基于距离推断键
        atoms = result['atom_types']
        coords = np.array(result['coords'])
        n = len(atoms)

        # 推断键（基于共价半径）
        bonds = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                # 在最大键距离内视为成键
                if d < MAX_BOND_DISTANCE:
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
