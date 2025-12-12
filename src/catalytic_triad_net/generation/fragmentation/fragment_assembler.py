#!/usr/bin/env python3
"""
片段组装器
将生成的纳米酶片段组装成完整结构
借鉴 StoL 的 Kabsch 对齐和接口原子匹配算法
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
import logging

from .fragment_definitions import NanozymeFragment, FragmentType

logger = logging.getLogger(__name__)


class FragmentAssembler:
    """
    片段组装器

    核心功能：
    1. Kabsch 算法对齐片段
    2. 接口原子匹配
    3. 片段融合和冲突解决
    """

    def __init__(self,
                 overlap_threshold: float = 2.0,
                 clash_threshold: float = 0.8,
                 interface_weight: float = 2.0):
        """
        Args:
            overlap_threshold: 接口原子重叠距离阈值（Å）
            clash_threshold: 原子冲突距离阈值（Å）
            interface_weight: 接口原子匹配权重
        """
        self.overlap_threshold = overlap_threshold
        self.clash_threshold = clash_threshold
        self.interface_weight = interface_weight

    def assemble_fragments(self,
                          fragments: List[NanozymeFragment],
                          assembly_order: Optional[List[str]] = None) -> Dict:
        """
        组装多个片段成完整纳米酶

        Args:
            fragments: 片段列表
            assembly_order: 组装顺序（fragment_id列表），None则自动确定

        Returns:
            组装后的纳米酶结构字典
        """
        if len(fragments) == 0:
            raise ValueError("No fragments to assemble")

        if len(fragments) == 1:
            return self._single_fragment_to_nanozyme(fragments[0])

        logger.info(f"Assembling {len(fragments)} fragments")

        # 确定组装顺序
        if assembly_order is None:
            assembly_order = self._determine_assembly_order(fragments)

        # 按顺序组装
        assembled = None
        for i, frag_id in enumerate(assembly_order):
            fragment = next((f for f in fragments if f.fragment_id == frag_id), None)
            if fragment is None:
                logger.warning(f"Fragment {frag_id} not found, skipping")
                continue

            if assembled is None:
                # 第一个片段作为基准
                assembled = self._fragment_to_dict(fragment)
                logger.info(f"[1/{len(assembly_order)}] Base fragment: {frag_id}")
            else:
                # 对齐并融合后续片段
                logger.info(f"[{i+1}/{len(assembly_order)}] Assembling fragment: {frag_id}")
                assembled = self._align_and_merge(assembled, fragment)

        # 后处理：解决冲突
        assembled = self._resolve_clashes(assembled)

        logger.info(f"Assembly completed: {assembled['n_atoms']} atoms")
        return assembled

    def assemble_fragment_pair(self,
                              fragment1: NanozymeFragment,
                              fragment2: NanozymeFragment) -> Dict:
        """
        组装两个片段（简化版）

        Args:
            fragment1: 第一个片段（基准）
            fragment2: 第二个片段（待对齐）

        Returns:
            组装后的结构
        """
        logger.info(f"Assembling pair: {fragment1.fragment_id} + {fragment2.fragment_id}")

        # 转换第一个片段为字典
        assembled = self._fragment_to_dict(fragment1)

        # 对齐并融合第二个片段
        assembled = self._align_and_merge(assembled, fragment2)

        # 解决冲突
        assembled = self._resolve_clashes(assembled)

        return assembled

    def _determine_assembly_order(self, fragments: List[NanozymeFragment]) -> List[str]:
        """
        自动确定组装顺序

        策略：
        1. 催化中心优先（作为基准）
        2. 按片段大小排序（大的先组装）
        """
        # 按类型和大小排序
        def sort_key(frag):
            type_priority = {
                FragmentType.CATALYTIC_CENTER: 0,
                FragmentType.SCAFFOLD_CORE: 1,
                FragmentType.LINKER: 2,
                FragmentType.SUBSTRATE_POCKET: 3
            }
            return (type_priority.get(frag.fragment_type, 99), -frag.n_atoms)

        sorted_fragments = sorted(fragments, key=sort_key)
        return [f.fragment_id for f in sorted_fragments]

    def _align_and_merge(self,
                        assembled: Dict,
                        new_fragment: NanozymeFragment) -> Dict:
        """
        对齐新片段到已组装结构并融合

        核心步骤：
        1. 找到接口原子对应关系
        2. Kabsch 对齐
        3. 融合坐标
        """
        # 1. 找到接口原子
        assembled_coords = assembled['coords']
        new_coords = new_fragment.coords

        # 找到接口原子对应关系
        interface_pairs = self._find_interface_correspondences(
            assembled_coords,
            new_coords,
            new_fragment.interface_atoms
        )

        if len(interface_pairs) == 0:
            logger.warning("No interface correspondences found, using centroid alignment")
            # 使用质心对齐
            rotation, translation = self._align_by_centroid(
                assembled_coords,
                new_coords
            )
        else:
            # 2. Kabsch 对齐
            logger.info(f"Found {len(interface_pairs)} interface correspondences")
            rotation, translation = self._kabsch_align(
                assembled_coords[interface_pairs[:, 0]],
                new_coords[interface_pairs[:, 1]]
            )

        # 3. 应用变换
        aligned_coords = (new_coords @ rotation.T) + translation

        # 4. 融合片段
        merged = self._merge_coordinates(
            assembled,
            aligned_coords,
            new_fragment.atom_types,
            interface_pairs
        )

        return merged

    def _find_interface_correspondences(self,
                                       coords1: np.ndarray,
                                       coords2: np.ndarray,
                                       interface_atoms2: List[int]) -> np.ndarray:
        """
        找到接口原子对应关系

        使用匈牙利算法（类似 StoL 的 Sinkhorn）
        """
        if len(interface_atoms2) == 0:
            return np.array([]).reshape(0, 2)

        # 计算距离矩阵
        interface_coords2 = coords2[interface_atoms2]

        # 找到 coords1 中最近的原子
        distances = np.linalg.norm(
            coords1[:, None, :] - interface_coords2[None, :, :],
            axis=2
        )

        # 使用匈牙利算法找最优匹配
        row_ind, col_ind = linear_sum_assignment(distances)

        # 过滤距离过大的匹配
        valid_matches = distances[row_ind, col_ind] < self.overlap_threshold
        row_ind = row_ind[valid_matches]
        col_ind = col_ind[valid_matches]

        # 转换为全局索引
        correspondences = np.column_stack([
            row_ind,
            np.array(interface_atoms2)[col_ind]
        ])

        return correspondences

    def _kabsch_align(self,
                     coords_ref: np.ndarray,
                     coords_mobile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kabsch 算法对齐两组坐标

        Args:
            coords_ref: 参考坐标 [n, 3]
            coords_mobile: 待对齐坐标 [n, 3]

        Returns:
            (rotation_matrix, translation_vector)
        """
        assert coords_ref.shape == coords_mobile.shape

        # 中心化
        centroid_ref = np.mean(coords_ref, axis=0)
        centroid_mobile = np.mean(coords_mobile, axis=0)

        coords_ref_centered = coords_ref - centroid_ref
        coords_mobile_centered = coords_mobile - centroid_mobile

        # 计算协方差矩阵
        H = coords_mobile_centered.T @ coords_ref_centered

        # SVD 分解
        U, S, Vt = np.linalg.svd(H)

        # 计算旋转矩阵
        rotation = Vt.T @ U.T

        # 确保是旋转（行列式为1）
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = Vt.T @ U.T

        # 计算平移
        translation = centroid_ref - (centroid_mobile @ rotation.T)

        return rotation, translation

    def _align_by_centroid(self,
                          coords_ref: np.ndarray,
                          coords_mobile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用质心对齐（无旋转）"""
        centroid_ref = np.mean(coords_ref, axis=0)
        centroid_mobile = np.mean(coords_mobile, axis=0)

        rotation = np.eye(3)
        translation = centroid_ref - centroid_mobile

        return rotation, translation

    def _merge_coordinates(self,
                          assembled: Dict,
                          new_coords: np.ndarray,
                          new_atom_types: List[str],
                          interface_pairs: np.ndarray) -> Dict:
        """
        融合坐标

        策略：
        - 接口原子：取平均
        - 非接口原子：直接添加
        """
        old_coords = assembled['coords']
        old_atom_types = assembled['atom_types']

        # 标记接口原子
        interface_old = set(interface_pairs[:, 0]) if len(interface_pairs) > 0 else set()
        interface_new = set(interface_pairs[:, 1]) if len(interface_pairs) > 0 else set()

        # 合并坐标
        merged_coords = []
        merged_atom_types = []

        # 1. 保留旧片段的所有原子
        for i, (coord, atom_type) in enumerate(zip(old_coords, old_atom_types)):
            if i in interface_old:
                # 接口原子：与新片段对应原子取平均
                new_idx = interface_pairs[interface_pairs[:, 0] == i, 1][0]
                averaged_coord = (coord + new_coords[new_idx]) / 2.0
                merged_coords.append(averaged_coord)
            else:
                merged_coords.append(coord)
            merged_atom_types.append(atom_type)

        # 2. 添加新片段的非接口原子
        for i, (coord, atom_type) in enumerate(zip(new_coords, new_atom_types)):
            if i not in interface_new:
                merged_coords.append(coord)
                merged_atom_types.append(atom_type)

        return {
            'coords': np.array(merged_coords),
            'atom_types': merged_atom_types,
            'n_atoms': len(merged_coords),
            'elements': merged_atom_types  # 兼容性
        }

    def _resolve_clashes(self, assembled: Dict) -> Dict:
        """
        解决原子冲突

        策略：移除距离过近的重复原子
        """
        coords = assembled['coords']
        atom_types = assembled['atom_types']
        n = len(coords)

        # 找到冲突原子对
        to_remove = set()
        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue

                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < self.clash_threshold:
                    # 冲突：移除第二个原子
                    to_remove.add(j)
                    logger.debug(f"Removing clashing atom {j} (dist={dist:.2f}Å)")

        # 移除冲突原子
        if len(to_remove) > 0:
            logger.info(f"Removed {len(to_remove)} clashing atoms")
            keep_indices = [i for i in range(n) if i not in to_remove]
            assembled['coords'] = coords[keep_indices]
            assembled['atom_types'] = [atom_types[i] for i in keep_indices]
            assembled['elements'] = assembled['atom_types']
            assembled['n_atoms'] = len(keep_indices)

        return assembled

    def _fragment_to_dict(self, fragment: NanozymeFragment) -> Dict:
        """将片段转换为字典格式"""
        return {
            'coords': fragment.coords.copy(),
            'atom_types': fragment.atom_types.copy(),
            'elements': fragment.atom_types.copy(),
            'n_atoms': fragment.n_atoms,
            'fragment_ids': [fragment.fragment_id]
        }

    def _single_fragment_to_nanozyme(self, fragment: NanozymeFragment) -> Dict:
        """单个片段转换为纳米酶"""
        return self._fragment_to_dict(fragment)


class MultiConformationAssembler:
    """
    多构象组装器

    为多个片段的多个构象生成所有可能的组合
    """

    def __init__(self, assembler: FragmentAssembler):
        """
        Args:
            assembler: 基础片段组装器
        """
        self.assembler = assembler

    def assemble_all_combinations(self,
                                  fragment_conformations: Dict[str, List[NanozymeFragment]],
                                  max_combinations: int = 100) -> List[Dict]:
        """
        组装所有片段构象组合

        Args:
            fragment_conformations: {fragment_id: [conformations]}
            max_combinations: 最大组合数

        Returns:
            组装后的纳米酶列表
        """
        fragment_ids = list(fragment_conformations.keys())
        n_fragments = len(fragment_ids)

        logger.info(f"Assembling combinations from {n_fragments} fragments")
        for frag_id, confs in fragment_conformations.items():
            logger.info(f"  {frag_id}: {len(confs)} conformations")

        # 计算总组合数
        total_combinations = 1
        for confs in fragment_conformations.values():
            total_combinations *= len(confs)

        logger.info(f"Total possible combinations: {total_combinations}")

        if total_combinations > max_combinations:
            logger.warning(f"Too many combinations, sampling {max_combinations}")
            return self._sample_combinations(
                fragment_conformations,
                max_combinations
            )
        else:
            return self._enumerate_all_combinations(fragment_conformations)

    def _enumerate_all_combinations(self,
                                   fragment_conformations: Dict[str, List[NanozymeFragment]]) -> List[Dict]:
        """枚举所有组合"""
        from itertools import product

        fragment_ids = list(fragment_conformations.keys())
        conformation_lists = [fragment_conformations[fid] for fid in fragment_ids]

        assembled_list = []

        for combination in product(*conformation_lists):
            try:
                assembled = self.assembler.assemble_fragments(
                    list(combination),
                    assembly_order=fragment_ids
                )
                assembled['combination_id'] = len(assembled_list)
                assembled_list.append(assembled)
            except Exception as e:
                logger.warning(f"Failed to assemble combination: {e}")

        logger.info(f"Successfully assembled {len(assembled_list)} combinations")
        return assembled_list

    def _sample_combinations(self,
                            fragment_conformations: Dict[str, List[NanozymeFragment]],
                            n_samples: int) -> List[Dict]:
        """随机采样组合"""
        import random

        fragment_ids = list(fragment_conformations.keys())
        assembled_list = []

        for i in range(n_samples):
            # 随机选择每个片段的一个构象
            combination = [
                random.choice(fragment_conformations[fid])
                for fid in fragment_ids
            ]

            try:
                assembled = self.assembler.assemble_fragments(
                    combination,
                    assembly_order=fragment_ids
                )
                assembled['combination_id'] = i
                assembled_list.append(assembled)
            except Exception as e:
                logger.warning(f"Failed to assemble combination {i}: {e}")

        logger.info(f"Sampled and assembled {len(assembled_list)} combinations")
        return assembled_list


def create_assembler(overlap_threshold: float = 2.0,
                    clash_threshold: float = 0.8) -> FragmentAssembler:
    """创建片段组装器的便捷函数"""
    return FragmentAssembler(
        overlap_threshold=overlap_threshold,
        clash_threshold=clash_threshold
    )
