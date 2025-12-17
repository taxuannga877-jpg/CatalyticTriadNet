#!/usr/bin/env python3
"""
片段化纳米酶生成流程
整合所有模块，实现完整的 StoL 启发的片段化生成流程
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np

from .fragment_definitions import (
    NanozymeFragment, FragmentType, NanozymeFragmentizer,
    create_default_fragmentizer
)
from .fragment_conformation_generator import (
    FragmentConformationGenerator, ProgressiveFragmentGenerator,
    create_fragment_generator
)
from .fragment_assembler import (
    FragmentAssembler, MultiConformationAssembler,
    create_assembler
)
from .conformation_analysis import (
    ChemicalValidator, ConformationAnalyzer,
    create_analyzer
)

logger = logging.getLogger(__name__)


class FragmentedNanozymePipeline:
    """
    片段化纳米酶生成流程

    完整流程：
    1. 片段化：将纳米酶切分成片段
    2. 构象生成：为每个片段生成多个构象
    3. 片段组装：组合片段构象
    4. 化学验证：过滤无效结构
    5. 聚类分析：选择代表性结构

    这是 StoL 思想在纳米酶生成中的应用！
    """

    def __init__(self,
                 model_path: str = None,
                 device: str = None,
                 config: Dict = None):
        """
        Args:
            model_path: 扩散模型权重路径
            device: 计算设备
            config: 配置字典
        """
        logger.info("="*70)
        logger.info("Initializing Fragmented Nanozyme Generation Pipeline")
        logger.info("="*70)

        # 初始化各个模块
        self.fragmentizer = create_default_fragmentizer()
        logger.info("✓ Fragmentizer initialized")

        self.conformation_generator = create_fragment_generator(
            model_path=model_path,
            device=device
        )
        logger.info("✓ Conformation generator initialized")

        self.progressive_generator = ProgressiveFragmentGenerator(
            self.conformation_generator
        )
        logger.info("✓ Progressive generator initialized")

        self.assembler = create_assembler()
        logger.info("✓ Fragment assembler initialized")

        self.multi_assembler = MultiConformationAssembler(self.assembler)
        logger.info("✓ Multi-conformation assembler initialized")

        self.validator = ChemicalValidator()
        logger.info("✓ Chemical validator initialized")

        self.analyzer = create_analyzer(self.validator)
        logger.info("✓ Conformation analyzer initialized")

        logger.info("="*70 + "\n")

    def generate_from_nanozyme(self,
                               nanozyme: Dict,
                               n_conformations_per_fragment: int = 50,
                               n_clusters: int = 10,
                               max_combinations: int = 100,
                               output_dir: str = None,
                               use_progressive: bool = True) -> Dict:
        """
        从已有纳米酶生成多样化构象

        Args:
            nanozyme: 输入纳米酶结构（来自 NanozymeAssembler）
            n_conformations_per_fragment: 每个片段生成的构象数
            n_clusters: 聚类数量
            max_combinations: 最大组合数
            output_dir: 输出目录
            use_progressive: 是否使用渐进式生成

        Returns:
            生成结果字典
        """
        logger.info("\n" + "="*70)
        logger.info("FRAGMENTED NANOZYME GENERATION PIPELINE")
        logger.info("="*70)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # ===== 阶段 1: 片段化 =====
        logger.info("\n[STAGE 1/5] Fragmenting nanozyme...")
        fragments = self.fragmentizer.fragment_nanozyme(nanozyme)
        logger.info(f"✓ Generated {len(fragments)} fragments:")
        for frag in fragments:
            logger.info(f"  - {frag.fragment_id}: {frag.fragment_type.value} "
                       f"({frag.n_atoms} atoms, {len(frag.interface_atoms)} interface atoms)")

        # ===== 阶段 2: 构象生成 =====
        logger.info("\n[STAGE 2/5] Generating conformations for each fragment...")

        if use_progressive:
            logger.info("Using progressive generation strategy")
            fragment_conformations = {}
            for fragment in fragments:
                conformations = self.progressive_generator.generate_progressive(
                    fragment=fragment,
                    n_coarse=n_conformations_per_fragment * 2,
                    n_refined=n_conformations_per_fragment
                )
                fragment_conformations[fragment.fragment_id] = conformations
        else:
            logger.info("Using standard generation strategy")
            fragment_conformations = self.conformation_generator.generate_all_fragment_conformations(
                fragments=fragments,
                n_conformations_per_fragment=n_conformations_per_fragment
            )

        total_conformations = sum(len(confs) for confs in fragment_conformations.values())
        logger.info(f"✓ Generated {total_conformations} total conformations")

        # ===== 阶段 3: 片段组装 =====
        logger.info("\n[STAGE 3/5] Assembling fragment combinations...")
        assembled_nanozymes = self.multi_assembler.assemble_all_combinations(
            fragment_conformations=fragment_conformations,
            max_combinations=max_combinations
        )
        logger.info(f"✓ Assembled {len(assembled_nanozymes)} nanozyme structures")

        # ===== 阶段 4: 化学验证 =====
        logger.info("\n[STAGE 4/5] Validating chemical validity...")
        valid_nanozymes, validation_results = self.validator.batch_validate(
            assembled_nanozymes
        )
        logger.info(f"✓ Valid structures: {len(valid_nanozymes)}/{len(assembled_nanozymes)}")

        if len(valid_nanozymes) == 0:
            logger.error("No valid structures generated!")
            return {
                'fragments': fragments,
                'fragment_conformations': fragment_conformations,
                'assembled_nanozymes': assembled_nanozymes,
                'valid_nanozymes': [],
                'error': 'No valid structures'
            }

        # ===== 阶段 5: 聚类分析 =====
        logger.info("\n[STAGE 5/5] Clustering and selecting representatives...")
        analysis_results = self.analyzer.analyze_conformations(
            nanozymes=valid_nanozymes,
            n_clusters=n_clusters,
            output_dir=str(output_dir) if output_dir else None
        )

        representatives = analysis_results['representatives']
        logger.info(f"✓ Selected {len(representatives)} representative structures")

        # ===== 导出结果 =====
        if output_dir:
            self._export_results(
                fragments=fragments,
                fragment_conformations=fragment_conformations,
                representatives=representatives,
                analysis_results=analysis_results,
                output_dir=output_dir
            )

        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Summary:")
        logger.info(f"  - Input fragments: {len(fragments)}")
        logger.info(f"  - Total conformations: {total_conformations}")
        logger.info(f"  - Assembled structures: {len(assembled_nanozymes)}")
        logger.info(f"  - Valid structures: {len(valid_nanozymes)}")
        logger.info(f"  - Representative structures: {len(representatives)}")
        logger.info(f"  - Validity rate: {len(valid_nanozymes)/len(assembled_nanozymes):.1%}")
        logger.info("="*70 + "\n")

        return {
            'fragments': fragments,
            'fragment_conformations': fragment_conformations,
            'assembled_nanozymes': assembled_nanozymes,
            'valid_nanozymes': valid_nanozymes,
            'validation_results': validation_results,
            'analysis_results': analysis_results,
            'representatives': representatives,
            'statistics': analysis_results['statistics']
        }

    def generate_diverse_nanozymes(self,
                                  base_nanozyme: Dict,
                                  n_variants: int = 10,
                                  diversity_level: str = 'medium',
                                  output_dir: str = None) -> List[Dict]:
        """
        生成多样化的纳米酶变体

        Args:
            base_nanozyme: 基础纳米酶结构
            n_variants: 变体数量
            diversity_level: 多样性级别 ('low', 'medium', 'high')
            output_dir: 输出目录

        Returns:
            纳米酶变体列表
        """
        diversity_configs = {
            'low': {'n_conformations': 20, 'n_clusters': 5, 'max_combinations': 50},
            'medium': {'n_conformations': 50, 'n_clusters': 10, 'max_combinations': 100},
            'high': {'n_conformations': 100, 'n_clusters': 20, 'max_combinations': 200}
        }

        config = diversity_configs.get(diversity_level, diversity_configs['medium'])

        logger.info(f"Generating {n_variants} diverse nanozyme variants (diversity: {diversity_level})")

        results = self.generate_from_nanozyme(
            nanozyme=base_nanozyme,
            n_conformations_per_fragment=config['n_conformations'],
            n_clusters=config['n_clusters'],
            max_combinations=config['max_combinations'],
            output_dir=output_dir
        )

        representatives = results['representatives']

        # 选择前 n_variants 个
        selected = representatives[:min(n_variants, len(representatives))]

        logger.info(f"Selected {len(selected)} diverse variants")
        return selected

    def _export_results(self,
                       fragments: List[NanozymeFragment],
                       fragment_conformations: Dict[str, List[NanozymeFragment]],
                       representatives: List[Dict],
                       analysis_results: Dict,
                       output_dir: Path):
        """导出结果"""
        logger.info(f"\nExporting results to {output_dir}")

        # 1. 导出片段信息
        fragments_data = [frag.to_dict() for frag in fragments]
        with open(output_dir / 'fragments.json', 'w') as f:
            json.dump(fragments_data, f, indent=2)
        logger.info(f"✓ Saved fragments to fragments.json")

        # 2. 导出代表性结构为 XYZ
        for i, rep in enumerate(representatives):
            xyz_path = output_dir / f'representative_{i:03d}.xyz'
            self._save_xyz(rep, xyz_path)
        logger.info(f"✓ Saved {len(representatives)} representative structures as XYZ")

        # 3. 导出统计信息
        stats = analysis_results['statistics']
        with open(output_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✓ Saved statistics to statistics.json")

        # 4. 导出聚类标签
        labels = analysis_results['labels']
        np.save(output_dir / 'cluster_labels.npy', labels)
        logger.info(f"✓ Saved cluster labels")

        # 5. 生成报告
        self._generate_report(
            fragments=fragments,
            representatives=representatives,
            stats=stats,
            output_path=output_dir / 'generation_report.txt'
        )
        logger.info(f"✓ Generated report")

    def _save_xyz(self, nanozyme: Dict, filepath: Path):
        """保存为 XYZ 格式"""
        coords = nanozyme['coords']
        atom_types = nanozyme['atom_types']

        with open(filepath, 'w') as f:
            f.write(f"{len(atom_types)}\n")
            f.write(f"Generated nanozyme structure\n")
            for atom, coord in zip(atom_types, coords):
                f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

    def _generate_report(self,
                        fragments: List[NanozymeFragment],
                        representatives: List[Dict],
                        stats: Dict,
                        output_path: Path):
        """生成文本报告"""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FRAGMENTED NANOZYME GENERATION REPORT\n")
            f.write("="*70 + "\n\n")

            # 片段信息
            f.write("## Fragment Information\n")
            f.write(f"Total fragments: {len(fragments)}\n\n")
            for frag in fragments:
                f.write(f"Fragment: {frag.fragment_id}\n")
                f.write(f"  Type: {frag.fragment_type.value}\n")
                f.write(f"  Atoms: {frag.n_atoms}\n")
                f.write(f"  Interface atoms: {len(frag.interface_atoms)}\n")
                f.write("\n")

            # 统计信息
            f.write("\n## Generation Statistics\n")
            f.write(f"Total structures generated: {stats['n_total']}\n")
            f.write(f"Valid structures: {stats['n_valid']}\n")
            f.write(f"Validity rate: {stats['validity_rate']:.1%}\n")
            f.write(f"Number of clusters: {stats['n_clusters']}\n")
            f.write(f"Average geometry score: {stats['avg_geometry_score']:.3f}\n")
            f.write(f"Average clash count: {stats['avg_clash_count']:.2f}\n")
            f.write(f"Average min distance: {stats['avg_min_distance']:.2f} Å\n")

            # 簇大小
            f.write("\n## Cluster Sizes\n")
            for cluster_id, size in stats['cluster_sizes'].items():
                f.write(f"  Cluster {cluster_id}: {size} structures\n")

            # 代表性结构
            f.write("\n## Representative Structures\n")
            for i, rep in enumerate(representatives):
                f.write(f"\nRepresentative {i}:\n")
                f.write(f"  Cluster ID: {rep.get('cluster_id', 'N/A')}\n")
                f.write(f"  Cluster size: {rep.get('cluster_size', 'N/A')}\n")
                f.write(f"  Atoms: {rep['n_atoms']}\n")
                f.write(f"  Distance to centroid: {rep.get('distance_to_centroid', 'N/A'):.3f}\n")

            f.write("\n" + "="*70 + "\n")


def create_pipeline(model_path: str = None,
                   device: str = None,
                   config: Dict = None) -> FragmentedNanozymePipeline:
    """创建片段化纳米酶生成流程的便捷函数"""
    return FragmentedNanozymePipeline(
        model_path=model_path,
        device=device,
        config=config
    )
