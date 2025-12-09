#!/usr/bin/env python3
"""
催化功能团提取器
从天然酶催化残基中提取化学功能团用于纳米酶设计
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    from Bio.PDB import PDBParser, Selection
except ImportError:
    PDBParser = None
    Selection = None

logger = logging.getLogger(__name__)


# 催化功能团模板定义
FUNCTIONAL_GROUP_TEMPLATES = {
    'HIS': {
        'name': 'imidazole',
        'atoms': ['CG', 'ND1', 'CD2', 'CE1', 'NE2'],  # 咪唑环
        'role': 'general_base',
        'key_atoms': ['ND1', 'NE2'],  # 关键原子（质子转移位点）
        'description': '组氨酸咪唑环 - 质子转移'
    },
    'ASP': {
        'name': 'carboxylate',
        'atoms': ['CG', 'OD1', 'OD2'],  # 羧基
        'role': 'electrostatic',
        'key_atoms': ['OD1', 'OD2'],
        'description': '天冬氨酸羧基 - 静电稳定'
    },
    'GLU': {
        'name': 'carboxylate',
        'atoms': ['CD', 'OE1', 'OE2'],  # 羧基
        'role': 'electrostatic',
        'key_atoms': ['OE1', 'OE2'],
        'description': '谷氨酸羧基 - 静电稳定'
    },
    'SER': {
        'name': 'hydroxyl',
        'atoms': ['CB', 'OG'],  # 羟基
        'role': 'nucleophile',
        'key_atoms': ['OG'],
        'description': '丝氨酸羟基 - 亲核试剂'
    },
    'CYS': {
        'name': 'thiol',
        'atoms': ['CB', 'SG'],  # 巯基
        'role': 'nucleophile',
        'key_atoms': ['SG'],
        'description': '半胱氨酸巯基 - 亲核试剂'
    },
    'THR': {
        'name': 'hydroxyl',
        'atoms': ['CB', 'OG1'],  # 羟基
        'role': 'nucleophile',
        'key_atoms': ['OG1'],
        'description': '苏氨酸羟基 - 亲核试剂'
    },
    'LYS': {
        'name': 'amino',
        'atoms': ['CE', 'NZ'],  # 氨基
        'role': 'general_base',
        'key_atoms': ['NZ'],
        'description': '赖氨酸氨基 - 碱催化'
    },
    'TYR': {
        'name': 'phenol',
        'atoms': ['CZ', 'OH'],  # 酚羟基
        'role': 'proton_donor',
        'key_atoms': ['OH'],
        'description': '酪氨酸酚羟基 - 质子供体'
    },
    'ASN': {
        'name': 'amide',
        'atoms': ['CG', 'OD1', 'ND2'],  # 酰胺
        'role': 'hydrogen_bond',
        'key_atoms': ['OD1', 'ND2'],
        'description': '天冬酰胺酰胺基 - 氢键'
    },
    'GLN': {
        'name': 'amide',
        'atoms': ['CD', 'OE1', 'NE2'],  # 酰胺
        'role': 'hydrogen_bond',
        'key_atoms': ['OE1', 'NE2'],
        'description': '谷氨酰胺酰胺基 - 氢键'
    }
}


@dataclass
class FunctionalGroup:
    """功能团数据结构"""
    group_id: str  # 唯一标识
    group_type: str  # 功能团类型 (imidazole, carboxylate等)
    source_residue: str  # 来源残基 (HIS57)
    source_pdb: str  # 来源PDB
    role: str  # 催化角色

    # 几何信息
    atom_names: List[str]  # 原子名称
    atom_elements: List[str]  # 元素符号
    coords: np.ndarray  # 坐标 [N_atoms, 3]

    # 关键原子索引
    key_atom_indices: List[int]

    # 质量信息
    site_prob: float  # 催化位点概率
    ec_class: int  # EC类别

    # 额外信息
    metadata: Dict = None

    def get_center(self) -> np.ndarray:
        """获取功能团几何中心"""
        return self.coords.mean(axis=0)

    def get_key_atoms_center(self) -> np.ndarray:
        """获取关键原子中心"""
        key_coords = self.coords[self.key_atom_indices]
        return key_coords.mean(axis=0)

    def translate(self, vector: np.ndarray):
        """平移功能团"""
        self.coords += vector

    def rotate(self, rotation_matrix: np.ndarray, center: Optional[np.ndarray] = None):
        """旋转功能团"""
        if center is None:
            center = self.get_center()
        self.coords = (self.coords - center) @ rotation_matrix.T + center

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'group_id': self.group_id,
            'group_type': self.group_type,
            'source_residue': self.source_residue,
            'source_pdb': self.source_pdb,
            'role': self.role,
            'atom_names': self.atom_names,
            'atom_elements': self.atom_elements,
            'coords': self.coords.tolist(),
            'key_atom_indices': self.key_atom_indices,
            'site_prob': self.site_prob,
            'ec_class': self.ec_class,
            'metadata': self.metadata or {}
        }


class FunctionalGroupExtractor:
    """
    功能团提取器

    从PDB结构中提取催化功能团的实际3D坐标

    使用示例:
        extractor = FunctionalGroupExtractor()

        # 从筛选结果提取
        functional_groups = extractor.extract_from_screening_results(
            screening_results,
            pdb_dir='data/pdbs/'
        )

        # 查看提取的功能团
        for fg in functional_groups:
            print(f"{fg.group_type} from {fg.source_residue} - prob: {fg.site_prob:.3f}")
    """

    def __init__(self):
        if PDBParser is None:
            raise ImportError("需要安装 Biopython: pip install biopython")

        self.parser = PDBParser(QUIET=True)
        self.templates = FUNCTIONAL_GROUP_TEMPLATES

    def extract_from_screening_results(self,
                                      screening_results: List[Dict],
                                      pdb_dir: Optional[str] = None,
                                      top_n: int = 50) -> List[FunctionalGroup]:
        """
        从批量筛选结果中提取功能团

        Args:
            screening_results: BatchCatalyticScreener的输出
            pdb_dir: PDB文件目录（如果需要从本地读取）
            top_n: 提取前N个高分残基

        Returns:
            功能团列表
        """
        functional_groups = []

        # 按PDB分组
        from collections import defaultdict
        by_pdb = defaultdict(list)
        for result in screening_results[:top_n]:
            by_pdb[result['pdb_id']].append(result)

        logger.info(f"从 {len(by_pdb)} 个PDB中提取功能团...")

        for pdb_id, residues in by_pdb.items():
            try:
                # 加载PDB结构
                if pdb_dir:
                    pdb_path = Path(pdb_dir) / f"{pdb_id}.pdb"
                else:
                    # 尝试下载
                    pdb_path = self._download_pdb(pdb_id)

                if not pdb_path.exists():
                    logger.warning(f"找不到PDB文件: {pdb_id}")
                    continue

                structure = self.parser.get_structure(pdb_id, pdb_path)

                # 提取每个催化残基的功能团
                for res_info in residues:
                    fg = self._extract_functional_group(
                        structure, res_info, pdb_id
                    )
                    if fg:
                        functional_groups.append(fg)

            except Exception as e:
                logger.error(f"处理 {pdb_id} 失败: {e}")

        logger.info(f"成功提取 {len(functional_groups)} 个功能团")
        return functional_groups

    def _extract_functional_group(self, structure, res_info: Dict,
                                  pdb_id: str) -> Optional[FunctionalGroup]:
        """从PDB结构中提取单个功能团"""
        chain_id = res_info['chain']
        resseq = res_info['resseq']
        resname = res_info['resname']

        # 检查是否有模板
        if resname not in self.templates:
            logger.debug(f"跳过无模板残基: {resname}")
            return None

        template = self.templates[resname]

        try:
            # 获取残基
            chain = structure[0][chain_id]
            residue = chain[resseq]

            # 提取功能团原子
            atom_names = []
            atom_elements = []
            coords = []
            key_atom_indices = []

            for atom_name in template['atoms']:
                if atom_name in residue:
                    atom = residue[atom_name]
                    atom_names.append(atom_name)
                    atom_elements.append(atom.element)
                    coords.append(atom.coord)

                    # 标记关键原子
                    if atom_name in template['key_atoms']:
                        key_atom_indices.append(len(atom_names) - 1)

            if len(coords) < len(template['atoms']) * 0.8:
                logger.warning(f"功能团原子不完整: {resname}{resseq}")
                return None

            # 创建功能团对象
            fg = FunctionalGroup(
                group_id=f"{pdb_id}_{chain_id}{resseq}_{resname}",
                group_type=template['name'],
                source_residue=f"{resname}{resseq}",
                source_pdb=pdb_id,
                role=template['role'],
                atom_names=atom_names,
                atom_elements=atom_elements,
                coords=np.array(coords),
                key_atom_indices=key_atom_indices,
                site_prob=res_info['site_prob'],
                ec_class=res_info['ec1_prediction'],
                metadata={
                    'chain': chain_id,
                    'description': template['description']
                }
            )

            return fg

        except Exception as e:
            logger.debug(f"提取功能团失败 {resname}{resseq}: {e}")
            return None

    def _download_pdb(self, pdb_id: str) -> Path:
        """下载PDB文件"""
        from urllib.request import urlretrieve

        cache_dir = Path.home() / '.cache' / 'catalytic_triad_net' / 'pdbs'
        cache_dir.mkdir(parents=True, exist_ok=True)

        pdb_path = cache_dir / f"{pdb_id}.pdb"

        if not pdb_path.exists():
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            try:
                urlretrieve(url, pdb_path)
                logger.info(f"已下载: {pdb_id}")
            except Exception as e:
                logger.error(f"下载失败 {pdb_id}: {e}")

        return pdb_path

    def filter_by_type(self, functional_groups: List[FunctionalGroup],
                      group_types: List[str]) -> List[FunctionalGroup]:
        """按功能团类型过滤"""
        filtered = [fg for fg in functional_groups if fg.group_type in group_types]
        logger.info(f"类型过滤: {len(functional_groups)} -> {len(filtered)}")
        return filtered

    def filter_by_role(self, functional_groups: List[FunctionalGroup],
                      roles: List[str]) -> List[FunctionalGroup]:
        """按催化角色过滤"""
        filtered = [fg for fg in functional_groups if fg.role in roles]
        logger.info(f"角色过滤: {len(functional_groups)} -> {len(filtered)}")
        return filtered

    def deduplicate(self, functional_groups: List[FunctionalGroup],
                   distance_threshold: float = 2.0) -> List[FunctionalGroup]:
        """
        去重：移除空间上非常接近的功能团

        Args:
            functional_groups: 功能团列表
            distance_threshold: 距离阈值 (Å)

        Returns:
            去重后的列表
        """
        if len(functional_groups) <= 1:
            return functional_groups

        # 计算所有功能团中心
        centers = np.array([fg.get_center() for fg in functional_groups])

        # 计算距离矩阵
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(centers, centers)

        # 贪心去重：保留高分的
        keep_indices = []
        sorted_indices = sorted(range(len(functional_groups)),
                              key=lambda i: functional_groups[i].site_prob,
                              reverse=True)

        for idx in sorted_indices:
            # 检查是否与已保留的太近
            too_close = False
            for kept_idx in keep_indices:
                if dist_matrix[idx, kept_idx] < distance_threshold:
                    too_close = True
                    break

            if not too_close:
                keep_indices.append(idx)

        deduplicated = [functional_groups[i] for i in sorted(keep_indices)]
        logger.info(f"去重: {len(functional_groups)} -> {len(deduplicated)}")

        return deduplicated

    def export_to_xyz(self, functional_groups: List[FunctionalGroup],
                     output_path: str):
        """导出为XYZ格式（用于可视化）"""
        with open(output_path, 'w') as f:
            for fg in functional_groups:
                f.write(f"{len(fg.coords)}\n")
                f.write(f"{fg.group_id} - {fg.group_type} - prob:{fg.site_prob:.3f}\n")
                for atom_name, element, coord in zip(fg.atom_names,
                                                     fg.atom_elements,
                                                     fg.coords):
                    f.write(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

        logger.info(f"已导出XYZ: {output_path}")

    def export_to_json(self, functional_groups: List[FunctionalGroup],
                      output_path: str):
        """导出为JSON格式"""
        import json

        data = {
            'n_groups': len(functional_groups),
            'functional_groups': [fg.to_dict() for fg in functional_groups]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"已导出JSON: {output_path}")

    def get_statistics(self, functional_groups: List[FunctionalGroup]) -> Dict:
        """获取统计信息"""
        from collections import Counter

        stats = {
            'total_groups': len(functional_groups),
            'type_distribution': dict(Counter(fg.group_type for fg in functional_groups)),
            'role_distribution': dict(Counter(fg.role for fg in functional_groups)),
            'ec_distribution': dict(Counter(fg.ec_class for fg in functional_groups)),
            'avg_site_prob': np.mean([fg.site_prob for fg in functional_groups]),
            'unique_pdbs': len(set(fg.source_pdb for fg in functional_groups))
        }

        return stats

    def print_statistics(self, functional_groups: List[FunctionalGroup]):
        """打印统计信息"""
        stats = self.get_statistics(functional_groups)

        print("\n" + "="*60)
        print("功能团提取统计")
        print("="*60)
        print(f"总功能团数: {stats['total_groups']}")
        print(f"来源PDB数: {stats['unique_pdbs']}")
        print(f"平均催化概率: {stats['avg_site_prob']:.3f}")

        print(f"\n功能团类型分布:")
        for gtype, count in sorted(stats['type_distribution'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  {gtype}: {count} ({count/stats['total_groups']*100:.1f}%)")

        print(f"\n催化角色分布:")
        for role, count in sorted(stats['role_distribution'].items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"  {role}: {count} ({count/stats['total_groups']*100:.1f}%)")

        print("="*60 + "\n")
