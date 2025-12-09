#!/usr/bin/env python3
"""
纳米酶骨架构建器
用碳骨架/金属配位框架连接催化功能团
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from .functional_group_extractor import FunctionalGroup

logger = logging.getLogger(__name__)


@dataclass
class ScaffoldAtom:
    """骨架原子"""
    element: str
    coords: np.ndarray
    atom_type: str  # 'scaffold' 或 'functional_group'
    connected_to: List[int] = None  # 连接的原子索引

    def __post_init__(self):
        if self.connected_to is None:
            self.connected_to = []


class ScaffoldBuilder:
    """
    骨架构建器

    支持三种骨架类型:
    1. 碳链骨架 (carbon_chain) - 烷基链连接
    2. 芳香环骨架 (aromatic_ring) - 苯环/萘环连接
    3. 金属配位骨架 (metal_framework) - MOF风格金属-有机框架

    使用示例:
        builder = ScaffoldBuilder(scaffold_type='carbon_chain')

        # 构建纳米酶
        nanozyme = builder.build_nanozyme(
            functional_groups=[fg1, fg2, fg3],
            target_distances={'0-1': 10.0, '1-2': 12.0}
        )

        # 导出结构
        builder.export_to_xyz(nanozyme, 'nanozyme.xyz')
        builder.export_to_pdb(nanozyme, 'nanozyme.pdb')
    """

    def __init__(self, scaffold_type: str = 'carbon_chain',
                 scaffold_params: Optional[Dict] = None):
        """
        Args:
            scaffold_type: 骨架类型
                - 'carbon_chain': 碳链骨架
                - 'aromatic_ring': 芳香环骨架
                - 'metal_framework': 金属配位框架
            scaffold_params: 骨架参数
        """
        self.scaffold_type = scaffold_type
        self.scaffold_params = scaffold_params or {}

        # 默认参数
        self.default_params = {
            'carbon_chain': {
                'chain_length': 3,  # 功能团间碳链长度
                'bond_length': 1.54,  # C-C键长 (Å)
                'flexibility': 0.2  # 柔性
            },
            'aromatic_ring': {
                'ring_size': 6,  # 苯环
                'bond_length': 1.40,  # 芳香键长
                'substitution_pattern': 'meta'  # 取代模式
            },
            'metal_framework': {
                'metal_type': 'Fe',
                'linker_type': 'carboxylate',  # 连接基团
                'coordination_number': 6
            }
        }

        self.params = {**self.default_params.get(scaffold_type, {}),
                      **self.scaffold_params}

        logger.info(f"骨架构建器初始化: {scaffold_type}")

    def build_nanozyme(self, functional_groups: List[FunctionalGroup],
                      target_distances: Optional[Dict[str, float]] = None,
                      optimize: bool = True) -> Dict:
        """
        构建纳米酶结构

        Args:
            functional_groups: 功能团列表
            target_distances: 目标距离约束 {'0-1': 10.0, '1-2': 12.0}
            optimize: 是否优化几何

        Returns:
            纳米酶结构字典
        """
        if len(functional_groups) < 2:
            raise ValueError("至少需要2个功能团")

        logger.info(f"开始构建纳米酶: {len(functional_groups)} 个功能团")

        # 步骤1: 放置功能团
        positioned_groups = self._position_functional_groups(
            functional_groups, target_distances
        )

        # 步骤2: 生成连接骨架
        scaffold_atoms = self._generate_scaffold(
            positioned_groups, target_distances
        )

        # 步骤3: 合并功能团和骨架
        nanozyme = self._merge_structure(positioned_groups, scaffold_atoms)

        # 步骤4: 几何优化
        if optimize:
            nanozyme = self._optimize_geometry(nanozyme, target_distances)

        # 步骤5: 添加氢原子（简化版）
        nanozyme = self._add_hydrogens(nanozyme)

        logger.info(f"纳米酶构建完成: {nanozyme['n_atoms']} 个原子")

        return nanozyme

    def _position_functional_groups(self, functional_groups: List[FunctionalGroup],
                                   target_distances: Optional[Dict[str, float]]) -> List[FunctionalGroup]:
        """
        放置功能团到合适的空间位置

        策略:
        - 第一个功能团放在原点
        - 后续功能团根据目标距离放置
        """
        positioned = []

        for i, fg in enumerate(functional_groups):
            # 复制功能团
            fg_copy = FunctionalGroup(
                group_id=fg.group_id,
                group_type=fg.group_type,
                source_residue=fg.source_residue,
                source_pdb=fg.source_pdb,
                role=fg.role,
                atom_names=fg.atom_names.copy(),
                atom_elements=fg.atom_elements.copy(),
                coords=fg.coords.copy(),
                key_atom_indices=fg.key_atom_indices.copy(),
                site_prob=fg.site_prob,
                ec_class=fg.ec_class,
                metadata=fg.metadata.copy() if fg.metadata else {}
            )

            if i == 0:
                # 第一个功能团: 移到原点
                center = fg_copy.get_center()
                fg_copy.translate(-center)
            else:
                # 后续功能团: 根据目标距离放置
                target_dist = None
                if target_distances:
                    key = f"{i-1}-{i}"
                    target_dist = target_distances.get(key)

                if target_dist is None:
                    # 默认距离: 10-15Å
                    target_dist = 10.0 + i * 2.0

                # 放置在前一个功能团的某个方向
                prev_center = positioned[-1].get_key_atoms_center()

                # 随机方向（可以改进为更智能的放置）
                angle = (i - 1) * (2 * np.pi / len(functional_groups))
                direction = np.array([
                    np.cos(angle),
                    np.sin(angle),
                    0.2 * (i % 3 - 1)  # 轻微的z方向变化
                ])
                direction = direction / np.linalg.norm(direction)

                # 移动功能团
                fg_center = fg_copy.get_center()
                fg_copy.translate(-fg_center)  # 先移到原点
                target_pos = prev_center + direction * target_dist
                fg_copy.translate(target_pos)

            positioned.append(fg_copy)

        return positioned

    def _generate_scaffold(self, functional_groups: List[FunctionalGroup],
                          target_distances: Optional[Dict[str, float]]) -> List[ScaffoldAtom]:
        """生成连接骨架"""
        if self.scaffold_type == 'carbon_chain':
            return self._generate_carbon_chain_scaffold(functional_groups)
        elif self.scaffold_type == 'aromatic_ring':
            return self._generate_aromatic_scaffold(functional_groups)
        elif self.scaffold_type == 'metal_framework':
            return self._generate_metal_framework_scaffold(functional_groups)
        else:
            raise ValueError(f"未知骨架类型: {self.scaffold_type}")

    def _generate_carbon_chain_scaffold(self, functional_groups: List[FunctionalGroup]) -> List[ScaffoldAtom]:
        """生成碳链骨架"""
        scaffold_atoms = []
        bond_length = self.params['bond_length']
        chain_length = self.params['chain_length']

        for i in range(len(functional_groups) - 1):
            # 获取两个功能团的连接点
            fg1 = functional_groups[i]
            fg2 = functional_groups[i + 1]

            # 使用关键原子作为连接点
            start = fg1.coords[fg1.key_atom_indices[0]]
            end = fg2.coords[fg2.key_atom_indices[0]]

            # 生成碳链
            direction = end - start
            distance = np.linalg.norm(direction)
            direction = direction / distance

            # 计算需要的碳原子数
            n_carbons = max(2, int(distance / bond_length) - 1)

            for j in range(n_carbons):
                t = (j + 1) / (n_carbons + 1)
                pos = start + direction * distance * t

                # 添加一些随机扰动（模拟柔性）
                if self.params.get('flexibility', 0) > 0:
                    noise = np.random.randn(3) * self.params['flexibility']
                    pos += noise

                scaffold_atoms.append(ScaffoldAtom(
                    element='C',
                    coords=pos,
                    atom_type='scaffold'
                ))

        logger.info(f"生成碳链骨架: {len(scaffold_atoms)} 个碳原子")
        return scaffold_atoms

    def _generate_aromatic_scaffold(self, functional_groups: List[FunctionalGroup]) -> List[ScaffoldAtom]:
        """生成芳香环骨架（简化版：苯环）"""
        scaffold_atoms = []
        ring_size = self.params['ring_size']
        bond_length = self.params['bond_length']

        # 为每对功能团生成一个苯环连接
        for i in range(len(functional_groups) - 1):
            fg1 = functional_groups[i]
            fg2 = functional_groups[i + 1]

            # 苯环中心位置
            center = (fg1.get_center() + fg2.get_center()) / 2

            # 生成苯环
            for j in range(ring_size):
                angle = 2 * np.pi * j / ring_size
                x = center[0] + bond_length * np.cos(angle)
                y = center[1] + bond_length * np.sin(angle)
                z = center[2]

                scaffold_atoms.append(ScaffoldAtom(
                    element='C',
                    coords=np.array([x, y, z]),
                    atom_type='scaffold'
                ))

        logger.info(f"生成芳香环骨架: {len(scaffold_atoms)} 个碳原子")
        return scaffold_atoms

    def _generate_metal_framework_scaffold(self, functional_groups: List[FunctionalGroup]) -> List[ScaffoldAtom]:
        """生成金属配位框架骨架"""
        scaffold_atoms = []
        metal_type = self.params['metal_type']

        # 在功能团之间放置金属中心
        for i in range(len(functional_groups) - 1):
            fg1 = functional_groups[i]
            fg2 = functional_groups[i + 1]

            # 金属位置（两个功能团中间）
            metal_pos = (fg1.get_center() + fg2.get_center()) / 2

            scaffold_atoms.append(ScaffoldAtom(
                element=metal_type,
                coords=metal_pos,
                atom_type='scaffold'
            ))

            # 添加连接配体（羧酸根）
            # 简化版：在金属周围添加O原子
            for j in range(4):
                angle = 2 * np.pi * j / 4
                offset = np.array([
                    2.0 * np.cos(angle),
                    2.0 * np.sin(angle),
                    0
                ])
                ligand_pos = metal_pos + offset

                scaffold_atoms.append(ScaffoldAtom(
                    element='O',
                    coords=ligand_pos,
                    atom_type='scaffold'
                ))

        logger.info(f"生成金属框架骨架: {len(scaffold_atoms)} 个原子")
        return scaffold_atoms

    def _merge_structure(self, functional_groups: List[FunctionalGroup],
                        scaffold_atoms: List[ScaffoldAtom]) -> Dict:
        """合并功能团和骨架"""
        all_elements = []
        all_coords = []
        all_atom_types = []
        functional_group_indices = []

        # 添加功能团原子
        for fg in functional_groups:
            start_idx = len(all_elements)
            all_elements.extend(fg.atom_elements)
            all_coords.extend(fg.coords.tolist())
            all_atom_types.extend(['functional_group'] * len(fg.atom_elements))
            functional_group_indices.append(list(range(start_idx, len(all_elements))))

        # 添加骨架原子
        scaffold_start_idx = len(all_elements)
        for atom in scaffold_atoms:
            all_elements.append(atom.element)
            all_coords.append(atom.coords.tolist())
            all_atom_types.append('scaffold')

        nanozyme = {
            'elements': all_elements,
            'coords': np.array(all_coords),
            'atom_types': all_atom_types,
            'n_atoms': len(all_elements),
            'functional_group_indices': functional_group_indices,
            'scaffold_indices': list(range(scaffold_start_idx, len(all_elements))),
            'functional_groups': [fg.to_dict() for fg in functional_groups],
            'scaffold_type': self.scaffold_type,
            'metadata': {
                'n_functional_groups': len(functional_groups),
                'n_scaffold_atoms': len(scaffold_atoms)
            }
        }

        return nanozyme

    def _optimize_geometry(self, nanozyme: Dict,
                          target_distances: Optional[Dict[str, float]]) -> Dict:
        """
        几何优化（简化版）
        使用力场最小化避免原子冲突
        """
        coords = nanozyme['coords'].copy()
        n_atoms = len(coords)

        def energy_function(flat_coords):
            """简化的能量函数"""
            coords_3d = flat_coords.reshape(-1, 3)
            energy = 0.0

            # 1. 排斥能（避免原子冲突）
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    dist = np.linalg.norm(coords_3d[i] - coords_3d[j])
                    if dist < 1.0:  # 太近
                        energy += 100.0 * (1.0 - dist) ** 2

            # 2. 目标距离约束
            if target_distances:
                for key, target_dist in target_distances.items():
                    i, j = map(int, key.split('-'))
                    if i < len(nanozyme['functional_group_indices']) and \
                       j < len(nanozyme['functional_group_indices']):
                        # 功能团中心距离
                        fg_i_indices = nanozyme['functional_group_indices'][i]
                        fg_j_indices = nanozyme['functional_group_indices'][j]

                        center_i = coords_3d[fg_i_indices].mean(axis=0)
                        center_j = coords_3d[fg_j_indices].mean(axis=0)

                        actual_dist = np.linalg.norm(center_i - center_j)
                        energy += 10.0 * (actual_dist - target_dist) ** 2

            return energy

        # 优化（只优化骨架原子，保持功能团相对刚性）
        scaffold_indices = nanozyme['scaffold_indices']
        if scaffold_indices:
            scaffold_coords = coords[scaffold_indices].flatten()

            result = minimize(
                lambda x: energy_function(
                    np.concatenate([coords[:scaffold_indices[0]].flatten(), x,
                                  coords[scaffold_indices[-1]+1:].flatten()])
                ),
                scaffold_coords,
                method='L-BFGS-B',
                options={'maxiter': 100}
            )

            if result.success:
                coords[scaffold_indices] = result.x.reshape(-1, 3)
                logger.info("几何优化完成")
            else:
                logger.warning("几何优化未收敛")

        nanozyme['coords'] = coords
        return nanozyme

    def _add_hydrogens(self, nanozyme: Dict) -> Dict:
        """
        添加氢原子（简化版）
        只在碳原子上添加
        """
        # 简化版：跳过氢原子添加
        # 实际应用中可以使用RDKit或OpenBabel
        logger.info("跳过氢原子添加（需要RDKit）")
        return nanozyme

    def export_to_xyz(self, nanozyme: Dict, output_path: str):
        """导出为XYZ格式"""
        with open(output_path, 'w') as f:
            f.write(f"{nanozyme['n_atoms']}\n")
            f.write(f"Nanozyme structure - {nanozyme['scaffold_type']}\n")

            for element, coord in zip(nanozyme['elements'], nanozyme['coords']):
                f.write(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

        logger.info(f"已导出XYZ: {output_path}")

    def export_to_pdb(self, nanozyme: Dict, output_path: str):
        """导出为PDB格式"""
        with open(output_path, 'w') as f:
            f.write("HEADER    NANOZYME STRUCTURE\n")
            f.write(f"TITLE     {nanozyme['scaffold_type'].upper()} SCAFFOLD\n")

            for i, (element, coord, atom_type) in enumerate(zip(
                nanozyme['elements'],
                nanozyme['coords'],
                nanozyme['atom_types']
            ), 1):
                # PDB ATOM格式
                resname = 'FGR' if atom_type == 'functional_group' else 'SCF'
                f.write(f"ATOM  {i:5d}  {element:<3s} {resname} A{i:4d}    "
                       f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                       f"  1.00  0.00          {element:>2s}\n")

            f.write("END\n")

        logger.info(f"已导出PDB: {output_path}")

    def export_to_mol2(self, nanozyme: Dict, output_path: str):
        """导出为MOL2格式（包含键信息）"""
        # 简化版：推断键
        bonds = self._infer_bonds(nanozyme)

        with open(output_path, 'w') as f:
            f.write("@<TRIPOS>MOLECULE\n")
            f.write("Nanozyme\n")
            f.write(f"{nanozyme['n_atoms']} {len(bonds)} 0 0 0\n")
            f.write("SMALL\n")
            f.write("USER_CHARGES\n\n")

            f.write("@<TRIPOS>ATOM\n")
            for i, (element, coord) in enumerate(zip(
                nanozyme['elements'],
                nanozyme['coords']
            ), 1):
                f.write(f"{i:7d} {element}{i:<4d} {coord[0]:10.4f} {coord[1]:10.4f} "
                       f"{coord[2]:10.4f} {element:<5s} 1 RES1 0.0000\n")

            f.write("@<TRIPOS>BOND\n")
            for i, (a1, a2, bond_type) in enumerate(bonds, 1):
                f.write(f"{i:6d} {a1:5d} {a2:5d} {bond_type}\n")

        logger.info(f"已导出MOL2: {output_path}")

    def _infer_bonds(self, nanozyme: Dict) -> List[Tuple[int, int, str]]:
        """推断键连接（基于距离）"""
        coords = nanozyme['coords']
        elements = nanozyme['elements']
        bonds = []

        # 共价半径（简化）
        covalent_radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66,
            'S': 1.05, 'P': 1.07, 'Fe': 1.32, 'Cu': 1.32,
            'Zn': 1.22, 'Mg': 1.41
        }

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])

                # 估计键长
                r1 = covalent_radii.get(elements[i], 1.0)
                r2 = covalent_radii.get(elements[j], 1.0)
                expected_bond_length = r1 + r2

                # 如果距离在合理范围内，认为成键
                if dist < expected_bond_length * 1.3:
                    bonds.append((i + 1, j + 1, '1'))  # 单键

        return bonds

    def visualize_with_pymol(self, nanozyme: Dict, output_script: str):
        """生成PyMOL可视化脚本"""
        with open(output_script, 'w') as f:
            f.write("# PyMOL visualization script\n")
            f.write("bg_color white\n")
            f.write("set sphere_scale, 0.3\n")
            f.write("set stick_radius, 0.15\n\n")

            # 功能团着色
            for i, fg_indices in enumerate(nanozyme['functional_group_indices']):
                color = ['red', 'blue', 'green', 'yellow', 'orange'][i % 5]
                indices_str = '+'.join(map(str, [idx + 1 for idx in fg_indices]))
                f.write(f"select fg{i}, id {indices_str}\n")
                f.write(f"color {color}, fg{i}\n")
                f.write(f"show spheres, fg{i}\n\n")

            # 骨架着色
            scaffold_indices_str = '+'.join(map(str, [idx + 1 for idx in nanozyme['scaffold_indices']]))
            f.write(f"select scaffold, id {scaffold_indices_str}\n")
            f.write("color gray, scaffold\n")
            f.write("show sticks, scaffold\n\n")

            f.write("zoom\n")

        logger.info(f"已生成PyMOL脚本: {output_script}")
