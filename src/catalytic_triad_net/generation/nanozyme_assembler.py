#!/usr/bin/env python3
"""
纳米酶组装器 - 主接口
整合催化中心筛选、功能团提取、骨架构建的完整工作流
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import json

from ..prediction.batch_screener import BatchCatalyticScreener
from .functional_group_extractor import FunctionalGroupExtractor
from .scaffold_builder import ScaffoldBuilder

logger = logging.getLogger(__name__)


class NanozymeAssembler:
    """
    纳米酶组装器 - 完整工作流

    这是您需要的核心功能！

    工作流程:
    1. 批量筛选天然酶PDB，识别高分催化中心
    2. 提取催化功能团（His咪唑环、Asp羧基等）
    3. 用碳骨架/金属框架连接功能团
    4. 生成纳米酶结构

    使用示例:
        # 初始化组装器
        assembler = NanozymeAssembler(
            model_path='models/best_model.pt',
            scaffold_type='carbon_chain'
        )

        # 方式1: 从PDB ID列表组装
        nanozyme = assembler.assemble_from_pdb_list(
            pdb_ids=['1acb', '4cha', '1hiv'],
            n_functional_groups=3,
            site_threshold=0.7
        )

        # 方式2: 从文件夹组装
        nanozyme = assembler.assemble_from_directory(
            pdb_dir='data/pdbs/',
            n_functional_groups=4
        )

        # 导出结果
        assembler.export_nanozyme(nanozyme, 'output/nanozyme')
    """

    def __init__(self, model_path: str,
                 scaffold_type: str = 'carbon_chain',
                 scaffold_params: Optional[Dict] = None,
                 device: str = None):
        """
        Args:
            model_path: 训练好的CatalyticTriadNet模型路径
            scaffold_type: 骨架类型
                - 'carbon_chain': 碳链骨架
                - 'aromatic_ring': 芳香环骨架
                - 'metal_framework': 金属配位框架
            scaffold_params: 骨架参数
            device: 'cuda' 或 'cpu'
        """
        logger.info("="*60)
        logger.info("初始化纳米酶组装器")
        logger.info("="*60)

        # 初始化各个模块
        self.screener = BatchCatalyticScreener(
            model_path=model_path,
            device=device
        )
        self.extractor = FunctionalGroupExtractor()
        self.builder = ScaffoldBuilder(
            scaffold_type=scaffold_type,
            scaffold_params=scaffold_params
        )

        self.scaffold_type = scaffold_type
        logger.info(f"骨架类型: {scaffold_type}")
        logger.info("组装器初始化完成\n")

    def assemble_from_pdb_list(self,
                               pdb_ids: List[str],
                               n_functional_groups: int = 3,
                               site_threshold: float = 0.7,
                               ec_filter: Optional[int] = None,
                               target_distances: Optional[Dict[str, float]] = None,
                               filter_by_type: Optional[List[str]] = None,
                               filter_by_role: Optional[List[str]] = None) -> Dict:
        """
        从PDB ID列表组装纳米酶

        Args:
            pdb_ids: PDB ID列表
            n_functional_groups: 使用的功能团数量
            site_threshold: 催化位点概率阈值 (0-1)
            ec_filter: 只使用特定EC类别的酶
            target_distances: 功能团间目标距离 {'0-1': 10.0, '1-2': 12.0}
            filter_by_type: 只使用特定功能团类型 ['imidazole', 'carboxylate']
            filter_by_role: 只使用特定催化角色 ['nucleophile', 'general_base']

        Returns:
            纳米酶结构字典
        """
        logger.info("\n" + "="*60)
        logger.info("开始纳米酶组装流程")
        logger.info("="*60)
        logger.info(f"输入PDB数: {len(pdb_ids)}")
        logger.info(f"目标功能团数: {n_functional_groups}")
        logger.info(f"催化位点阈值: {site_threshold}")

        # 步骤1: 批量筛选催化中心
        logger.info("\n[步骤1/4] 批量筛选催化中心...")
        screening_results = self.screener.screen_pdb_list(
            pdb_ids=pdb_ids,
            site_threshold=site_threshold,
            top_k=10,
            ec_filter=ec_filter
        )

        if not screening_results:
            raise ValueError("未找到符合条件的催化中心")

        self.screener.print_statistics(screening_results)

        # 步骤2: 提取功能团
        logger.info("\n[步骤2/4] 提取催化功能团...")
        functional_groups = self.extractor.extract_from_screening_results(
            screening_results,
            top_n=n_functional_groups * 3  # 提取更多，后续过滤
        )

        if not functional_groups:
            raise ValueError("未能提取到功能团")

        # 过滤功能团
        if filter_by_type:
            functional_groups = self.extractor.filter_by_type(
                functional_groups, filter_by_type
            )

        if filter_by_role:
            functional_groups = self.extractor.filter_by_role(
                functional_groups, filter_by_role
            )

        # 去重
        functional_groups = self.extractor.deduplicate(functional_groups)

        # 选择前N个
        functional_groups = functional_groups[:n_functional_groups]

        if len(functional_groups) < 2:
            raise ValueError(f"功能团数量不足: {len(functional_groups)} < 2")

        self.extractor.print_statistics(functional_groups)

        # 步骤3: 构建骨架并组装
        logger.info("\n[步骤3/4] 构建骨架并组装纳米酶...")
        nanozyme = self.builder.build_nanozyme(
            functional_groups=functional_groups,
            target_distances=target_distances,
            optimize=True
        )

        # 步骤4: 添加元数据
        logger.info("\n[步骤4/4] 添加元数据...")
        nanozyme['assembly_info'] = {
            'source_pdbs': list(set(fg.source_pdb for fg in functional_groups)),
            'n_source_pdbs': len(set(fg.source_pdb for fg in functional_groups)),
            'functional_groups_used': [fg.group_id for fg in functional_groups],
            'scaffold_type': self.scaffold_type,
            'site_threshold': site_threshold,
            'ec_filter': ec_filter
        }

        logger.info("\n" + "="*60)
        logger.info("纳米酶组装完成!")
        logger.info("="*60)
        logger.info(f"总原子数: {nanozyme['n_atoms']}")
        logger.info(f"功能团数: {len(functional_groups)}")
        logger.info(f"骨架原子数: {len(nanozyme['scaffold_indices'])}")
        logger.info(f"来源PDB: {', '.join(nanozyme['assembly_info']['source_pdbs'])}")
        logger.info("="*60 + "\n")

        return nanozyme

    def assemble_from_directory(self,
                               pdb_dir: str,
                               n_functional_groups: int = 3,
                               site_threshold: float = 0.7,
                               pattern: str = "*.pdb",
                               **kwargs) -> Dict:
        """
        从PDB文件夹组装纳米酶

        Args:
            pdb_dir: PDB文件夹路径
            n_functional_groups: 功能团数量
            site_threshold: 催化位点阈值
            pattern: 文件匹配模式
            **kwargs: 其他参数传递给assemble_from_pdb_list

        Returns:
            纳米酶结构字典
        """
        logger.info(f"从目录组装: {pdb_dir}")

        # 步骤1: 筛选
        screening_results = self.screener.screen_directory(
            pdb_dir=pdb_dir,
            site_threshold=site_threshold,
            pattern=pattern
        )

        if not screening_results:
            raise ValueError("未找到符合条件的催化中心")

        # 步骤2-4: 提取、构建、组装
        functional_groups = self.extractor.extract_from_screening_results(
            screening_results,
            pdb_dir=pdb_dir,
            top_n=n_functional_groups * 3
        )

        # 应用过滤器
        if kwargs.get('filter_by_type'):
            functional_groups = self.extractor.filter_by_type(
                functional_groups, kwargs['filter_by_type']
            )

        if kwargs.get('filter_by_role'):
            functional_groups = self.extractor.filter_by_role(
                functional_groups, kwargs['filter_by_role']
            )

        functional_groups = self.extractor.deduplicate(functional_groups)
        functional_groups = functional_groups[:n_functional_groups]

        if len(functional_groups) < 2:
            raise ValueError(f"功能团数量不足: {len(functional_groups)}")

        nanozyme = self.builder.build_nanozyme(
            functional_groups=functional_groups,
            target_distances=kwargs.get('target_distances'),
            optimize=True
        )

        nanozyme['assembly_info'] = {
            'source_pdbs': list(set(fg.source_pdb for fg in functional_groups)),
            'n_source_pdbs': len(set(fg.source_pdb for fg in functional_groups)),
            'functional_groups_used': [fg.group_id for fg in functional_groups],
            'scaffold_type': self.scaffold_type,
            'site_threshold': site_threshold
        }

        logger.info("纳米酶组装完成!")
        return nanozyme

    def assemble_from_screening_results(self,
                                       screening_results: List[Dict],
                                       n_functional_groups: int = 3,
                                       pdb_dir: Optional[str] = None,
                                       **kwargs) -> Dict:
        """
        从已有的筛选结果组装纳米酶

        Args:
            screening_results: BatchCatalyticScreener的输出
            n_functional_groups: 功能团数量
            pdb_dir: PDB文件目录
            **kwargs: 其他参数

        Returns:
            纳米酶结构字典
        """
        logger.info("从筛选结果组装纳米酶...")

        # 提取功能团
        functional_groups = self.extractor.extract_from_screening_results(
            screening_results,
            pdb_dir=pdb_dir,
            top_n=n_functional_groups * 3
        )

        # 过滤
        if kwargs.get('filter_by_type'):
            functional_groups = self.extractor.filter_by_type(
                functional_groups, kwargs['filter_by_type']
            )

        if kwargs.get('filter_by_role'):
            functional_groups = self.extractor.filter_by_role(
                functional_groups, kwargs['filter_by_role']
            )

        functional_groups = self.extractor.deduplicate(functional_groups)
        functional_groups = functional_groups[:n_functional_groups]

        # 构建
        nanozyme = self.builder.build_nanozyme(
            functional_groups=functional_groups,
            target_distances=kwargs.get('target_distances'),
            optimize=True
        )

        nanozyme['assembly_info'] = {
            'source_pdbs': list(set(fg.source_pdb for fg in functional_groups)),
            'functional_groups_used': [fg.group_id for fg in functional_groups],
            'scaffold_type': self.scaffold_type
        }

        return nanozyme

    def export_nanozyme(self, nanozyme: Dict, output_prefix: str,
                       formats: List[str] = None):
        """
        导出纳米酶结构

        Args:
            nanozyme: 纳米酶结构字典
            output_prefix: 输出文件前缀（不含扩展名）
            formats: 导出格式列表 ['xyz', 'pdb', 'mol2', 'json']
                    None表示导出所有格式
        """
        if formats is None:
            formats = ['xyz', 'pdb', 'mol2', 'json', 'pymol']

        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n导出纳米酶结构: {output_prefix}")

        if 'xyz' in formats:
            self.builder.export_to_xyz(nanozyme, f"{output_prefix}.xyz")

        if 'pdb' in formats:
            self.builder.export_to_pdb(nanozyme, f"{output_prefix}.pdb")

        if 'mol2' in formats:
            self.builder.export_to_mol2(nanozyme, f"{output_prefix}.mol2")

        if 'json' in formats:
            self._export_to_json(nanozyme, f"{output_prefix}.json")

        if 'pymol' in formats:
            self.builder.visualize_with_pymol(nanozyme, f"{output_prefix}.pml")

        logger.info("导出完成!\n")

    def _export_to_json(self, nanozyme: Dict, output_path: str):
        """导出为JSON格式"""
        # 转换numpy数组为列表
        export_data = {
            'elements': nanozyme['elements'],
            'coords': nanozyme['coords'].tolist(),
            'atom_types': nanozyme['atom_types'],
            'n_atoms': nanozyme['n_atoms'],
            'functional_group_indices': nanozyme['functional_group_indices'],
            'scaffold_indices': nanozyme['scaffold_indices'],
            'functional_groups': nanozyme['functional_groups'],
            'scaffold_type': nanozyme['scaffold_type'],
            'metadata': nanozyme['metadata'],
            'assembly_info': nanozyme.get('assembly_info', {})
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"已导出JSON: {output_path}")

    def generate_report(self, nanozyme: Dict, output_path: str):
        """生成组装报告"""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("纳米酶组装报告\n")
            f.write("="*70 + "\n\n")

            # 基本信息
            f.write("## 基本信息\n")
            f.write(f"总原子数: {nanozyme['n_atoms']}\n")
            f.write(f"功能团数: {len(nanozyme['functional_group_indices'])}\n")
            f.write(f"骨架原子数: {len(nanozyme['scaffold_indices'])}\n")
            f.write(f"骨架类型: {nanozyme['scaffold_type']}\n\n")

            # 来源信息
            if 'assembly_info' in nanozyme:
                info = nanozyme['assembly_info']
                f.write("## 来源信息\n")
                f.write(f"来源PDB数: {info.get('n_source_pdbs', 'N/A')}\n")
                f.write(f"来源PDB: {', '.join(info.get('source_pdbs', []))}\n")
                f.write(f"催化位点阈值: {info.get('site_threshold', 'N/A')}\n")
                if info.get('ec_filter'):
                    f.write(f"EC类别过滤: EC{info['ec_filter']}\n")
                f.write("\n")

            # 功能团详情
            f.write("## 功能团详情\n")
            for i, fg_dict in enumerate(nanozyme['functional_groups'], 1):
                f.write(f"\n功能团 {i}:\n")
                f.write(f"  ID: {fg_dict['group_id']}\n")
                f.write(f"  类型: {fg_dict['group_type']}\n")
                f.write(f"  来源: {fg_dict['source_residue']} ({fg_dict['source_pdb']})\n")
                f.write(f"  角色: {fg_dict['role']}\n")
                f.write(f"  催化概率: {fg_dict['site_prob']:.3f}\n")
                f.write(f"  EC类别: EC{fg_dict['ec_class']}\n")
                f.write(f"  原子数: {len(fg_dict['atom_names'])}\n")

            # 元素统计
            f.write("\n## 元素统计\n")
            from collections import Counter
            element_counts = Counter(nanozyme['elements'])
            for element, count in sorted(element_counts.items()):
                f.write(f"  {element}: {count}\n")

            f.write("\n" + "="*70 + "\n")

        logger.info(f"已生成报告: {output_path}")

    def batch_assemble(self, pdb_lists: List[List[str]],
                      output_dir: str,
                      **kwargs) -> List[Dict]:
        """
        批量组装多个纳米酶

        Args:
            pdb_lists: PDB ID列表的列表，每个列表生成一个纳米酶
            output_dir: 输出目录
            **kwargs: 传递给assemble_from_pdb_list的参数

        Returns:
            纳米酶列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        nanozymes = []

        for i, pdb_list in enumerate(pdb_lists, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"批量组装 {i}/{len(pdb_lists)}")
            logger.info(f"{'='*60}")

            try:
                nanozyme = self.assemble_from_pdb_list(
                    pdb_ids=pdb_list,
                    **kwargs
                )

                # 导出
                output_prefix = output_dir / f"nanozyme_{i:03d}"
                self.export_nanozyme(nanozyme, str(output_prefix))
                self.generate_report(nanozyme, f"{output_prefix}_report.txt")

                nanozymes.append(nanozyme)

            except Exception as e:
                logger.error(f"组装失败: {e}")

        logger.info(f"\n批量组装完成: {len(nanozymes)}/{len(pdb_lists)} 成功")
        return nanozymes
