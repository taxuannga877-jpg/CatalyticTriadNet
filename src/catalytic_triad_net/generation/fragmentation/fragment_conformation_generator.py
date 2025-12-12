#!/usr/bin/env python3
"""
片段构象生成器
使用扩散模型为纳米酶片段生成多个构象
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

from .fragment_definitions import NanozymeFragment, FragmentType
from .models import CatalyticDiffusionModel
from .constraints import CatalyticConstraints, ATOM_TYPES, ATOM_TO_IDX

logger = logging.getLogger(__name__)


class FragmentConformationGenerator:
    """
    片段构象生成器

    为每个纳米酶片段生成多个3D构象
    核心思想：借鉴 StoL，对小片段分别生成构象，而不是一次性生成整个纳米酶
    """

    def __init__(self,
                 model: CatalyticDiffusionModel = None,
                 model_path: str = None,
                 device: str = None,
                 config: Dict = None):
        """
        Args:
            model: 已初始化的扩散模型
            model_path: 模型权重路径
            device: 计算设备
            config: 模型配置
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if model is not None:
            self.model = model
        else:
            # 初始化新模型
            config = config or {
                'hidden_dim': 256,
                'n_layers': 6,
                'num_timesteps': 1000,
                'beta_start': 1e-4,
                'beta_end': 0.02
            }
            self.model = CatalyticDiffusionModel(config).to(self.device)

            # 加载权重
            if model_path and Path(model_path).exists():
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                logger.info(f"Loaded model from {model_path}")

        self.model.eval()

    def generate_fragment_conformations(self,
                                       fragment: NanozymeFragment,
                                       n_conformations: int = 50,
                                       temperature: float = 1.0) -> List[NanozymeFragment]:
        """
        为单个片段生成多个构象

        Args:
            fragment: 输入片段（作为参考）
            n_conformations: 生成构象数量
            temperature: 采样温度（控制多样性）

        Returns:
            生成的构象列表（每个都是 NanozymeFragment）
        """
        logger.info(f"Generating {n_conformations} conformations for fragment {fragment.fragment_id}")
        logger.info(f"Fragment type: {fragment.fragment_type.value}, n_atoms: {fragment.n_atoms}")

        # 1. 构建条件（基于片段的接口原子和类型）
        condition = self._build_fragment_condition(fragment)

        # 2. 使用扩散模型采样
        with torch.no_grad():
            samples = self.model.sample(
                condition=condition,
                n_atoms=fragment.n_atoms,
                n_samples=n_conformations,
                guidance_scale=temperature
            )

        # 3. 转换为 NanozymeFragment 列表
        conformations = []
        for i in range(n_conformations):
            atom_types_idx = samples['atom_types'][i].cpu().numpy()
            coords = samples['coords'][i].cpu().numpy()

            # 转换原子类型索引到符号
            atom_types = [ATOM_TYPES[idx] if idx < len(ATOM_TYPES) else 'C'
                         for idx in atom_types_idx]

            # 创建新片段
            new_fragment = NanozymeFragment(
                fragment_id=f"{fragment.fragment_id}_conf_{i}",
                fragment_type=fragment.fragment_type,
                atom_types=atom_types,
                coords=coords,
                atom_indices=fragment.atom_indices.copy(),
                interface_atoms=fragment.interface_atoms.copy(),
                metadata={
                    **fragment.metadata,
                    'conformation_id': i,
                    'parent_fragment': fragment.fragment_id,
                    'generation_method': 'diffusion_sampling'
                }
            )
            conformations.append(new_fragment)

        logger.info(f"Generated {len(conformations)} conformations successfully")
        return conformations

    def generate_all_fragment_conformations(self,
                                           fragments: List[NanozymeFragment],
                                           n_conformations_per_fragment: int = 50,
                                           temperature: float = 1.0) -> Dict[str, List[NanozymeFragment]]:
        """
        为所有片段生成构象

        Args:
            fragments: 片段列表
            n_conformations_per_fragment: 每个片段生成的构象数
            temperature: 采样温度

        Returns:
            字典 {fragment_id: [conformations]}
        """
        logger.info(f"Generating conformations for {len(fragments)} fragments")

        all_conformations = {}

        for fragment in fragments:
            conformations = self.generate_fragment_conformations(
                fragment=fragment,
                n_conformations=n_conformations_per_fragment,
                temperature=temperature
            )
            all_conformations[fragment.fragment_id] = conformations

        # 统计
        total_conformations = sum(len(confs) for confs in all_conformations.values())
        logger.info(f"Total conformations generated: {total_conformations}")

        return all_conformations

    def _build_fragment_condition(self, fragment: NanozymeFragment) -> Dict[str, torch.Tensor]:
        """
        为片段构建条件张量

        条件包括：
        1. 接口原子位置（锚定点）
        2. 片段类型信息
        3. 原子类型分布
        """
        device = self.device

        # 1. 锚定特征：接口原子的位置和类型
        if len(fragment.interface_atoms) > 0:
            interface_coords = fragment.get_interface_coords()
            interface_types = [fragment.atom_types[i] for i in fragment.interface_atoms]

            # 编码接口原子
            anchor_features = []
            for coord, atom_type in zip(interface_coords, interface_types):
                # 特征：[x, y, z, atom_type_onehot...]
                atom_idx = ATOM_TO_IDX.get(atom_type, 0)
                onehot = np.zeros(len(ATOM_TYPES))
                if atom_idx < len(ATOM_TYPES):
                    onehot[atom_idx] = 1.0

                feature = np.concatenate([coord, onehot])
                anchor_features.append(feature)

            anchor_features = np.array(anchor_features)
        else:
            # 没有接口原子，使用质心作为锚定
            center = fragment.center_of_mass
            anchor_features = np.concatenate([
                center,
                np.zeros(len(ATOM_TYPES))
            ]).reshape(1, -1)

        # 填充到固定大小（16维）
        anchor_dim = 16
        if anchor_features.shape[1] > anchor_dim:
            anchor_features = anchor_features[:, :anchor_dim]
        elif anchor_features.shape[1] < anchor_dim:
            padding = np.zeros((anchor_features.shape[0],
                               anchor_dim - anchor_features.shape[1]))
            anchor_features = np.concatenate([anchor_features, padding], axis=1)

        anchor_tensor = torch.tensor(anchor_features, dtype=torch.float32, device=device)

        # 2. 距离约束（片段内部原子间的典型距离）
        # 简化：使用片段大小作为约束
        fragment_size = np.max(fragment.coords.max(axis=0) - fragment.coords.min(axis=0))
        distance_constraints = torch.tensor(
            [[0, 1, fragment_size * 0.5, fragment_size * 0.2]],  # [i, j, target, tolerance]
            dtype=torch.float32,
            device=device
        )

        # 3. 配位约束（片段类型编码）
        fragment_type_encoding = {
            FragmentType.CATALYTIC_CENTER: [1, 0, 0],
            FragmentType.SCAFFOLD_CORE: [0, 1, 0],
            FragmentType.LINKER: [0, 0, 1],
            FragmentType.SUBSTRATE_POCKET: [0, 0, 0]
        }
        coord_constraints = torch.tensor(
            [fragment_type_encoding.get(fragment.fragment_type, [0, 0, 0])],
            dtype=torch.float32,
            device=device
        )

        condition = {
            'anchor_features': anchor_tensor,
            'distance_constraints': distance_constraints,
            'coordination_constraints': coord_constraints
        }

        return condition

    def batch_generate_with_diversity(self,
                                      fragment: NanozymeFragment,
                                      n_conformations: int = 50,
                                      diversity_weight: float = 0.5) -> List[NanozymeFragment]:
        """
        生成多样化的构象（使用多样性增强采样）

        Args:
            fragment: 输入片段
            n_conformations: 构象数量
            diversity_weight: 多样性权重（0-1）

        Returns:
            多样化的构象列表
        """
        # 使用不同的温度生成多批次
        temperatures = [0.8, 1.0, 1.2, 1.5]
        n_per_temp = n_conformations // len(temperatures)

        all_conformations = []

        for temp in temperatures:
            conformations = self.generate_fragment_conformations(
                fragment=fragment,
                n_conformations=n_per_temp,
                temperature=temp
            )
            all_conformations.extend(conformations)

        # 补充到目标数量
        remaining = n_conformations - len(all_conformations)
        if remaining > 0:
            extra = self.generate_fragment_conformations(
                fragment=fragment,
                n_conformations=remaining,
                temperature=1.0
            )
            all_conformations.extend(extra)

        return all_conformations[:n_conformations]


class ProgressiveFragmentGenerator:
    """
    渐进式片段生成器

    借鉴 StoL 的渐进式采样：先快速生成粗糙构象，再精细化优化
    """

    def __init__(self, generator: FragmentConformationGenerator):
        """
        Args:
            generator: 基础构象生成器
        """
        self.generator = generator

    def generate_progressive(self,
                            fragment: NanozymeFragment,
                            n_coarse: int = 100,
                            n_refined: int = 20,
                            coarse_steps: int = 200,
                            refined_steps: int = 1000) -> List[NanozymeFragment]:
        """
        渐进式生成：粗糙 -> 筛选 -> 精细化

        Args:
            fragment: 输入片段
            n_coarse: 粗糙构象数量
            n_refined: 精细化构象数量
            coarse_steps: 粗糙采样步数
            refined_steps: 精细化采样步数

        Returns:
            精细化的构象列表
        """
        logger.info(f"Progressive generation: {n_coarse} coarse -> {n_refined} refined")

        # 阶段1：快速生成粗糙构象
        logger.info("Stage 1: Generating coarse conformations...")

        # 临时修改模型的采样步数
        original_timesteps = self.generator.model.num_timesteps
        self.generator.model.num_timesteps = coarse_steps

        coarse_conformations = self.generator.generate_fragment_conformations(
            fragment=fragment,
            n_conformations=n_coarse,
            temperature=1.2  # 更高温度增加多样性
        )

        # 恢复原始步数
        self.generator.model.num_timesteps = original_timesteps

        # 阶段2：筛选有潜力的构象
        logger.info("Stage 2: Filtering promising conformations...")
        scored_conformations = self._score_conformations(coarse_conformations, fragment)

        # 选择前 n_refined 个
        top_conformations = scored_conformations[:n_refined]

        # 阶段3：精细化优化
        logger.info("Stage 3: Refining top conformations...")
        self.generator.model.num_timesteps = refined_steps

        refined_conformations = []
        for conf in top_conformations:
            # 使用低温精细化
            refined = self.generator.generate_fragment_conformations(
                fragment=conf,
                n_conformations=1,
                temperature=0.8
            )
            refined_conformations.extend(refined)

        # 恢复原始步数
        self.generator.model.num_timesteps = original_timesteps

        logger.info(f"Progressive generation completed: {len(refined_conformations)} refined conformations")
        return refined_conformations

    def _score_conformations(self,
                            conformations: List[NanozymeFragment],
                            reference: NanozymeFragment) -> List[NanozymeFragment]:
        """
        评分并排序构象

        评分标准：
        1. 与参考片段的RMSD（不要太远）
        2. 内部几何合理性（键长、键角）
        3. 接口原子位置保持
        """
        scored = []

        for conf in conformations:
            score = 0.0

            # 1. RMSD 评分（适中的RMSD更好）
            rmsd = self._calculate_rmsd(conf.coords, reference.coords)
            rmsd_score = np.exp(-rmsd / 2.0)  # 偏好 RMSD ~ 2Å
            score += rmsd_score

            # 2. 几何合理性（检查最小原子间距）
            min_dist = self._min_pairwise_distance(conf.coords)
            if min_dist > 0.8:  # 没有严重冲突
                score += 1.0

            # 3. 接口原子位置保持
            if len(conf.interface_atoms) > 0:
                interface_rmsd = self._calculate_rmsd(
                    conf.get_interface_coords(),
                    reference.get_interface_coords()
                )
                interface_score = np.exp(-interface_rmsd / 1.0)
                score += interface_score

            scored.append((score, conf))

        # 按分数排序
        scored.sort(key=lambda x: x[0], reverse=True)
        return [conf for _, conf in scored]

    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """计算RMSD"""
        if coords1.shape != coords2.shape:
            return float('inf')
        return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))

    def _min_pairwise_distance(self, coords: np.ndarray) -> float:
        """计算最小原子间距"""
        n = len(coords)
        min_dist = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                min_dist = min(min_dist, dist)
        return min_dist


def create_fragment_generator(model_path: str = None,
                              device: str = None) -> FragmentConformationGenerator:
    """创建片段构象生成器的便捷函数"""
    return FragmentConformationGenerator(
        model_path=model_path,
        device=device
    )
