#!/usr/bin/env python3
"""
片段化纳米酶生成示例
展示如何使用 StoL 启发的片段化生成流程
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import logging
from catalytic_triad_net.generation.fragmented_nanozyme_pipeline import create_pipeline
from catalytic_triad_net.generation.nanozyme_assembler import NanozymeAssembler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_fragmented_generation():
    """
    示例 1: 基础片段化生成流程

    从已有纳米酶生成多样化构象
    """
    print("\n" + "="*80)
    print("示例 1: 基础片段化纳米酶生成")
    print("="*80)

    # 步骤 1: 使用 NanozymeAssembler 创建基础纳米酶
    logger.info("Step 1: Creating base nanozyme using NanozymeAssembler...")

    assembler = NanozymeAssembler(
        model_path='models/best_model.pt',  # 替换为实际模型路径
        scaffold_type='carbon_chain'
    )

    # 从 PDB 列表组装纳米酶
    base_nanozyme = assembler.assemble_from_pdb_list(
        pdb_ids=['1acb', '4cha', '1hiv'],
        n_functional_groups=3,
        site_threshold=0.7
    )

    logger.info(f"✓ Base nanozyme created: {base_nanozyme['n_atoms']} atoms")

    # 步骤 2: 初始化片段化生成流程
    logger.info("\nStep 2: Initializing fragmented generation pipeline...")

    pipeline = create_pipeline(
        model_path='models/diffusion_model.pt',  # 替换为实际扩散模型路径
        device='cuda'  # 或 'cpu'
    )

    # 步骤 3: 运行完整流程
    logger.info("\nStep 3: Running fragmented generation pipeline...")

    results = pipeline.generate_from_nanozyme(
        nanozyme=base_nanozyme,
        n_conformations_per_fragment=50,  # 每个片段生成 50 个构象
        n_clusters=10,                     # 聚类成 10 个簇
        max_combinations=100,              # 最多组装 100 个组合
        output_dir='output/fragmented_generation',
        use_progressive=True               # 使用渐进式生成
    )

    # 步骤 4: 查看结果
    logger.info("\nStep 4: Results summary...")

    print(f"\n生成结果:")
    print(f"  - 片段数量: {len(results['fragments'])}")
    print(f"  - 总构象数: {sum(len(confs) for confs in results['fragment_conformations'].values())}")
    print(f"  - 组装结构数: {len(results['assembled_nanozymes'])}")
    print(f"  - 有效结构数: {len(results['valid_nanozymes'])}")
    print(f"  - 代表性结构数: {len(results['representatives'])}")
    print(f"  - 有效率: {results['statistics']['validity_rate']:.1%}")

    # 步骤 5: 导出代表性结构
    logger.info("\nStep 5: Exporting representative structures...")

    for i, rep in enumerate(results['representatives'][:3]):  # 导出前 3 个
        output_path = f"output/representative_{i}.xyz"
        # 结构已经在 pipeline 中导出
        print(f"  ✓ Representative {i}: cluster {rep['cluster_id']}, "
              f"{rep['n_atoms']} atoms, "
              f"cluster size {rep['cluster_size']}")

    print("\n✓ 示例 1 完成！")
    return results


def example_2_diverse_variants_generation():
    """
    示例 2: 生成多样化变体

    快速生成多个多样化的纳米酶变体
    """
    print("\n" + "="*80)
    print("示例 2: 生成多样化纳米酶变体")
    print("="*80)

    # 步骤 1: 创建基础纳米酶（简化版）
    logger.info("Step 1: Creating base nanozyme...")

    # 这里使用模拟数据，实际使用时替换为真实纳米酶
    import numpy as np
    base_nanozyme = {
        'coords': np.random.randn(30, 3) * 5,
        'elements': ['C'] * 20 + ['N'] * 5 + ['O'] * 5,
        'atom_types': ['C'] * 20 + ['N'] * 5 + ['O'] * 5,
        'n_atoms': 30,
        'functional_group_indices': [[0, 1, 2], [10, 11, 12], [20, 21, 22]],
        'scaffold_indices': list(range(3, 10)) + list(range(13, 20)) + list(range(23, 30)),
        'scaffold_type': 'carbon_chain'
    }

    # 步骤 2: 初始化流程
    logger.info("\nStep 2: Initializing pipeline...")

    pipeline = create_pipeline(device='cpu')  # 使用 CPU 进行演示

    # 步骤 3: 生成多样化变体
    logger.info("\nStep 3: Generating diverse variants...")

    variants = pipeline.generate_diverse_nanozymes(
        base_nanozyme=base_nanozyme,
        n_variants=10,
        diversity_level='medium',  # 'low', 'medium', 'high'
        output_dir='output/diverse_variants'
    )

    # 步骤 4: 查看变体
    print(f"\n生成了 {len(variants)} 个多样化变体:")
    for i, variant in enumerate(variants):
        print(f"  变体 {i}: {variant['n_atoms']} 原子, "
              f"簇 {variant.get('cluster_id', 'N/A')}, "
              f"簇大小 {variant.get('cluster_size', 'N/A')}")

    print("\n✓ 示例 2 完成！")
    return variants


def example_3_custom_fragmentation():
    """
    示例 3: 自定义片段化规则

    展示如何自定义片段化和生成参数
    """
    print("\n" + "="*80)
    print("示例 3: 自定义片段化规则")
    print("="*80)

    from catalytic_triad_net.generation.fragment_definitions import (
        NanozymeFragmentizer, FragmentationRule
    )

    # 步骤 1: 创建自定义片段化规则
    logger.info("Step 1: Creating custom fragmentation rule...")

    custom_rule = FragmentationRule(
        name="custom_nanozyme_rule",
        description="Custom rule for large nanozyme structures",
        min_catalytic_atoms=8,      # 更大的催化中心
        max_catalytic_atoms=40,
        min_scaffold_atoms=20,      # 更大的支架片段
        max_scaffold_atoms=150,
        interface_distance=4.0      # 更宽松的接口定义
    )

    fragmentizer = NanozymeFragmentizer(rule=custom_rule)

    # 步骤 2: 创建测试纳米酶
    import numpy as np
    test_nanozyme = {
        'coords': np.random.randn(100, 3) * 10,
        'elements': ['C'] * 60 + ['N'] * 20 + ['O'] * 15 + ['Fe'] * 5,
        'atom_types': ['C'] * 60 + ['N'] * 20 + ['O'] * 15 + ['Fe'] * 5,
        'n_atoms': 100,
        'functional_group_indices': [
            list(range(0, 10)),
            list(range(30, 40)),
            list(range(60, 70))
        ],
        'scaffold_indices': list(range(10, 30)) + list(range(40, 60)) + list(range(70, 100)),
        'scaffold_type': 'metal_framework'
    }

    # 步骤 3: 应用自定义片段化
    logger.info("\nStep 2: Applying custom fragmentation...")

    fragments = fragmentizer.fragment_nanozyme(test_nanozyme)

    print(f"\n自定义片段化结果:")
    print(f"  - 总片段数: {len(fragments)}")
    for frag in fragments:
        print(f"  - {frag.fragment_id}: {frag.fragment_type.value}")
        print(f"    原子数: {frag.n_atoms}")
        print(f"    接口原子: {len(frag.interface_atoms)}")

    print("\n✓ 示例 3 完成！")
    return fragments


def example_4_progressive_generation():
    """
    示例 4: 渐进式生成策略

    展示粗糙->精细的两阶段生成
    """
    print("\n" + "="*80)
    print("示例 4: 渐进式生成策略")
    print("="*80)

    from catalytic_triad_net.generation.fragment_conformation_generator import (
        ProgressiveFragmentGenerator, create_fragment_generator
    )
    from catalytic_triad_net.generation.fragment_definitions import NanozymeFragment, FragmentType

    # 步骤 1: 创建测试片段
    logger.info("Step 1: Creating test fragment...")

    import numpy as np
    test_fragment = NanozymeFragment(
        fragment_id="test_catalytic_center",
        fragment_type=FragmentType.CATALYTIC_CENTER,
        atom_types=['C', 'N', 'O', 'C', 'N', 'O', 'C', 'C', 'N', 'O'],
        coords=np.random.randn(10, 3) * 2,
        atom_indices=list(range(10)),
        interface_atoms=[0, 5, 9],
        metadata={'source': 'test'}
    )

    # 步骤 2: 初始化渐进式生成器
    logger.info("\nStep 2: Initializing progressive generator...")

    base_generator = create_fragment_generator(device='cpu')
    progressive_gen = ProgressiveFragmentGenerator(base_generator)

    # 步骤 3: 渐进式生成
    logger.info("\nStep 3: Running progressive generation...")

    refined_conformations = progressive_gen.generate_progressive(
        fragment=test_fragment,
        n_coarse=100,        # 阶段1: 100 个粗糙构象
        n_refined=20,        # 阶段2: 20 个精细构象
        coarse_steps=200,    # 粗糙采样步数
        refined_steps=1000   # 精细采样步数
    )

    print(f"\n渐进式生成结果:")
    print(f"  - 粗糙构象: 100")
    print(f"  - 筛选后: 20")
    print(f"  - 精细化构象: {len(refined_conformations)}")

    print("\n✓ 示例 4 完成！")
    return refined_conformations


def example_5_validation_and_clustering():
    """
    示例 5: 化学验证和聚类分析

    展示如何单独使用验证和聚类模块
    """
    print("\n" + "="*80)
    print("示例 5: 化学验证和聚类分析")
    print("="*80)

    from catalytic_triad_net.generation.conformation_analysis import (
        ChemicalValidator, ConformationAnalyzer
    )

    # 步骤 1: 创建测试纳米酶结构
    logger.info("Step 1: Creating test nanozyme structures...")

    import numpy as np
    test_nanozymes = []

    for i in range(50):
        nanozyme = {
            'coords': np.random.randn(30, 3) * 5 + np.random.randn(3) * 2,
            'atom_types': ['C'] * 20 + ['N'] * 5 + ['O'] * 5,
            'elements': ['C'] * 20 + ['N'] * 5 + ['O'] * 5,
            'n_atoms': 30
        }
        test_nanozymes.append(nanozyme)

    # 步骤 2: 化学验证
    logger.info("\nStep 2: Chemical validation...")

    validator = ChemicalValidator(
        min_bond_length=0.8,
        max_bond_length=2.5,
        clash_threshold=0.8
    )

    valid_nanozymes, validation_results = validator.batch_validate(test_nanozymes)

    print(f"\n验证结果:")
    print(f"  - 总结构数: {len(test_nanozymes)}")
    print(f"  - 有效结构: {len(valid_nanozymes)}")
    print(f"  - 有效率: {len(valid_nanozymes)/len(test_nanozymes):.1%}")
    print(f"  - 平均几何分数: {np.mean([r.geometry_score for r in validation_results]):.3f}")
    print(f"  - 平均冲突数: {np.mean([r.clash_count for r in validation_results]):.1f}")

    # 步骤 3: 聚类分析
    logger.info("\nStep 3: Clustering analysis...")

    analyzer = ConformationAnalyzer(validator=validator)

    analysis_results = analyzer.analyze_conformations(
        nanozymes=valid_nanozymes,
        n_clusters=5,
        output_dir='output/clustering_analysis'
    )

    print(f"\n聚类结果:")
    print(f"  - 簇数量: {analysis_results['statistics']['n_clusters']}")
    print(f"  - 代表性结构: {len(analysis_results['representatives'])}")
    print(f"  - 簇大小分布: {analysis_results['statistics']['cluster_sizes']}")

    print("\n✓ 示例 5 完成！")
    return analysis_results


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("片段化纳米酶生成系统 - 完整示例")
    print("基于 StoL 的片段化生成思想")
    print("="*80)

    try:
        # 示例 1: 基础流程（需要真实模型）
        # results_1 = example_1_basic_fragmented_generation()

        # 示例 2: 多样化变体生成
        variants = example_2_diverse_variants_generation()

        # 示例 3: 自定义片段化
        fragments = example_3_custom_fragmentation()

        # 示例 4: 渐进式生成
        refined = example_4_progressive_generation()

        # 示例 5: 验证和聚类
        analysis = example_5_validation_and_clustering()

        print("\n" + "="*80)
        print("所有示例运行完成！")
        print("="*80)

    except Exception as e:
        logger.error(f"示例运行失败: {e}", exc_info=True)
        print(f"\n错误: {e}")
        print("请确保已安装所有依赖: torch, numpy, scipy, sklearn, umap-learn, matplotlib")


if __name__ == '__main__':
    main()
