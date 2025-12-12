#!/usr/bin/env python3
"""
纳米酶片段定义模块
定义纳米酶的片段类型、切分规则和组装接口
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from enum import Enum


class FragmentType(Enum):
    """片段类型枚举"""
    CATALYTIC_CENTER = "catalytic_center"  # 催化中心（功能团簇）
    SCAFFOLD_CORE = "scaffold_core"        # 支架核心（金属纳米颗粒/碳骨架核心）
    LINKER = "linker"                      # 连接片段（连接催化中心和支架）
    SUBSTRATE_POCKET = "substrate_pocket"  # 底物结合口袋


@dataclass
class NanozymeFragment:
    """
    纳米酶片段数据结构

    Attributes:
        fragment_id: 片段唯一标识
        fragment_type: 片段类型
        atom_types: 原子类型列表 ['C', 'N', 'O', ...]
        coords: 原子坐标 [n_atoms, 3]
        atom_indices: 在完整纳米酶中的原子索引
        interface_atoms: 接口原子索引（用于与其他片段连接）
        metadata: 额外元数据
    """
    fragment_id: str
    fragment_type: FragmentType
    atom_types: List[str]
    coords: np.ndarray
    atom_indices: List[int]
    interface_atoms: List[int]  # 接口原子的局部索引
    metadata: Dict

    def __post_init__(self):
        """验证数据一致性"""
        assert len(self.atom_types) == len(self.coords), \
            f"atom_types length {len(self.atom_types)} != coords length {len(self.coords)}"
        assert self.coords.shape[1] == 3, \
            f"coords should have shape [n_atoms, 3], got {self.coords.shape}"
        assert all(0 <= idx < len(self.atom_types) for idx in self.interface_atoms), \
            "interface_atoms indices out of range"

    @property
    def n_atoms(self) -> int:
        """片段原子数"""
        return len(self.atom_types)

    @property
    def center_of_mass(self) -> np.ndarray:
        """质心坐标"""
        return np.mean(self.coords, axis=0)

    def get_interface_coords(self) -> np.ndarray:
        """获取接口原子坐标"""
        return self.coords[self.interface_atoms]

    def translate(self, vector: np.ndarray):
        """平移片段"""
        self.coords = self.coords + vector

    def rotate(self, rotation_matrix: np.ndarray):
        """旋转片段（绕质心）"""
        center = self.center_of_mass
        self.coords = (self.coords - center) @ rotation_matrix.T + center

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'fragment_id': self.fragment_id,
            'fragment_type': self.fragment_type.value,
            'atom_types': self.atom_types,
            'coords': self.coords.tolist(),
            'atom_indices': self.atom_indices,
            'interface_atoms': self.interface_atoms,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'NanozymeFragment':
        """从字典创建"""
        return cls(
            fragment_id=data['fragment_id'],
            fragment_type=FragmentType(data['fragment_type']),
            atom_types=data['atom_types'],
            coords=np.array(data['coords']),
            atom_indices=data['atom_indices'],
            interface_atoms=data['interface_atoms'],
            metadata=data['metadata']
        )


@dataclass
class FragmentationRule:
    """
    片段化规则

    定义如何将纳米酶切分成片段
    """
    name: str
    description: str
    min_catalytic_atoms: int = 5      # 催化中心最小原子数
    max_catalytic_atoms: int = 20     # 催化中心最大原子数
    min_scaffold_atoms: int = 10      # 支架最小原子数
    max_scaffold_atoms: int = 100     # 支架最大原子数
    interface_distance: float = 3.0   # 接口原子距离阈值（Å）

    def validate_fragment(self, fragment: NanozymeFragment) -> bool:
        """验证片段是否符合规则"""
        if fragment.fragment_type == FragmentType.CATALYTIC_CENTER:
            return self.min_catalytic_atoms <= fragment.n_atoms <= self.max_catalytic_atoms
        elif fragment.fragment_type == FragmentType.SCAFFOLD_CORE:
            return self.min_scaffold_atoms <= fragment.n_atoms <= self.max_scaffold_atoms
        return True


class NanozymeFragmentizer:
    """
    纳米酶片段化器

    将完整的纳米酶结构切分成片段
    """

    def __init__(self, rule: FragmentationRule = None):
        """
        Args:
            rule: 片段化规则
        """
        self.rule = rule or FragmentationRule(
            name="default",
            description="Default fragmentation rule for nanozymes"
        )

    def fragment_nanozyme(self, nanozyme: Dict) -> List[NanozymeFragment]:
        """
        将纳米酶切分成片段

        Args:
            nanozyme: 纳米酶结构字典（来自 NanozymeAssembler）
                - elements: 元素列表
                - coords: 坐标数组 [n_atoms, 3]
                - functional_group_indices: 功能团原子索引列表
                - scaffold_indices: 支架原子索引列表

        Returns:
            片段列表
        """
        fragments = []

        # 1. 提取催化中心片段
        catalytic_fragment = self._extract_catalytic_center(nanozyme)
        if catalytic_fragment:
            fragments.append(catalytic_fragment)

        # 2. 提取支架片段
        scaffold_fragments = self._extract_scaffold_fragments(nanozyme)
        fragments.extend(scaffold_fragments)

        # 3. 识别接口原子
        self._identify_interface_atoms(fragments)

        return fragments

    def _extract_catalytic_center(self, nanozyme: Dict) -> Optional[NanozymeFragment]:
        """提取催化中心片段"""
        fg_indices = nanozyme.get('functional_group_indices', [])
        if not fg_indices:
            return None

        # 合并所有功能团原子
        all_fg_atoms = []
        for fg_atom_list in fg_indices:
            all_fg_atoms.extend(fg_atom_list)
        all_fg_atoms = sorted(set(all_fg_atoms))

        if len(all_fg_atoms) < self.rule.min_catalytic_atoms:
            return None

        # 提取坐标和原子类型
        coords = nanozyme['coords'][all_fg_atoms]
        atom_types = [nanozyme['elements'][i] for i in all_fg_atoms]

        return NanozymeFragment(
            fragment_id="catalytic_center_0",
            fragment_type=FragmentType.CATALYTIC_CENTER,
            atom_types=atom_types,
            coords=coords.copy(),
            atom_indices=all_fg_atoms,
            interface_atoms=[],  # 稍后识别
            metadata={
                'n_functional_groups': len(fg_indices),
                'source': 'nanozyme_assembly'
            }
        )

    def _extract_scaffold_fragments(self, nanozyme: Dict) -> List[NanozymeFragment]:
        """
        提取支架片段

        策略：将支架切分成多个小片段（类似 StoL 的分子片段化）
        """
        scaffold_indices = nanozyme.get('scaffold_indices', [])
        if not scaffold_indices:
            return []

        coords = nanozyme['coords'][scaffold_indices]
        atom_types = [nanozyme['elements'][i] for i in scaffold_indices]

        # 简单策略：按空间聚类切分支架
        # 这里使用基于距离的聚类
        fragments = []

        # 如果支架较小，作为单个片段
        if len(scaffold_indices) <= self.rule.max_scaffold_atoms:
            fragment = NanozymeFragment(
                fragment_id="scaffold_core_0",
                fragment_type=FragmentType.SCAFFOLD_CORE,
                atom_types=atom_types,
                coords=coords.copy(),
                atom_indices=scaffold_indices,
                interface_atoms=[],
                metadata={'scaffold_type': nanozyme.get('scaffold_type', 'unknown')}
            )
            fragments.append(fragment)
        else:
            # 大支架：切分成多个片段
            sub_fragments = self._cluster_scaffold_atoms(
                coords, atom_types, scaffold_indices
            )
            fragments.extend(sub_fragments)

        return fragments

    def _cluster_scaffold_atoms(self, coords: np.ndarray,
                                atom_types: List[str],
                                global_indices: List[int]) -> List[NanozymeFragment]:
        """使用空间聚类切分支架原子"""
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        # 层次聚类
        distances = pdist(coords)
        Z = linkage(distances, method='average')

        # 切分成簇（目标：每簇 20-50 个原子）
        n_clusters = max(2, len(coords) // 30)
        labels = fcluster(Z, n_clusters, criterion='maxclust')

        fragments = []
        for cluster_id in range(1, n_clusters + 1):
            mask = labels == cluster_id
            cluster_indices = np.where(mask)[0]

            if len(cluster_indices) < 5:  # 太小的簇跳过
                continue

            fragment = NanozymeFragment(
                fragment_id=f"scaffold_core_{cluster_id}",
                fragment_type=FragmentType.SCAFFOLD_CORE,
                atom_types=[atom_types[i] for i in cluster_indices],
                coords=coords[cluster_indices].copy(),
                atom_indices=[global_indices[i] for i in cluster_indices],
                interface_atoms=[],
                metadata={'cluster_id': cluster_id}
            )
            fragments.append(fragment)

        return fragments

    def _identify_interface_atoms(self, fragments: List[NanozymeFragment]):
        """
        识别片段间的接口原子

        接口原子：距离其他片段原子较近的原子
        """
        for i, frag_i in enumerate(fragments):
            interface_atoms = []

            for j, frag_j in enumerate(fragments):
                if i == j:
                    continue

                # 计算片段间距离
                for atom_idx, coord_i in enumerate(frag_i.coords):
                    min_dist = np.min(np.linalg.norm(
                        frag_j.coords - coord_i, axis=1
                    ))

                    if min_dist < self.rule.interface_distance:
                        interface_atoms.append(atom_idx)

            frag_i.interface_atoms = sorted(set(interface_atoms))


def create_default_fragmentizer() -> NanozymeFragmentizer:
    """创建默认片段化器"""
    rule = FragmentationRule(
        name="nanozyme_default",
        description="Default fragmentation for nanozyme structures",
        min_catalytic_atoms=5,
        max_catalytic_atoms=30,
        min_scaffold_atoms=10,
        max_scaffold_atoms=80,
        interface_distance=3.5
    )
    return NanozymeFragmentizer(rule)
