#!/usr/bin/env python3
"""
纳米酶组装完整示例
展示从天然酶PDB筛选催化中心到组装纳米酶的完整工作流
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from catalytic_triad_net.generation import NanozymeAssembler


def example_1_basic_assembly():
    """
    示例1: 基础纳米酶组装
    从几个丝氨酸蛋白酶中提取催化中心，组装纳米酶
    """
    print("\n" + "="*70)
    print("示例1: 基础纳米酶组装")
    print("="*70)

    # 初始化组装器
    assembler = NanozymeAssembler(
        model_path='models/best_model.pt',  # 训练好的模型
        scaffold_type='carbon_chain',  # 使用碳链骨架
        device='cuda'  # 或 'cpu'
    )

    # 从PDB ID列表组装
    # 这些都是经典的丝氨酸蛋白酶
    pdb_ids = [
        '1acb',  # 胰凝乳蛋白酶
        '4cha',  # α-胰凝乳蛋白酶
        '1hne',  # 人中性粒细胞弹性蛋白酶
    ]

    nanozyme = assembler.assemble_from_pdb_list(
        pdb_ids=pdb_ids,
        n_functional_groups=3,  # 使用3个功能团
        site_threshold=0.7,  # 只使用高分催化残基
        ec_filter=3  # 只使用水解酶(EC3)
    )

    # 导出结果
    assembler.export_nanozyme(
        nanozyme,
        output_prefix='output/nanozyme_basic',
        formats=['xyz', 'pdb', 'json', 'pymol']
    )

    # 生成报告
    assembler.generate_report(
        nanozyme,
        'output/nanozyme_basic_report.txt'
    )

    print("\n✓ 示例1完成! 查看 output/nanozyme_basic.*")


def example_2_custom_distances():
    """
    示例2: 自定义功能团间距离
    精确控制催化中心的空间排列
    """
    print("\n" + "="*70)
    print("示例2: 自定义功能团间距离")
    print("="*70)

    assembler = NanozymeAssembler(
        model_path='models/best_model.pt',
        scaffold_type='carbon_chain',
        scaffold_params={
            'chain_length': 4,  # 更长的碳链
            'bond_length': 1.54,
            'flexibility': 0.3  # 更大的柔性
        }
    )

    # 指定功能团间的目标距离
    target_distances = {
        '0-1': 8.0,   # 功能团0和1之间: 8Å
        '1-2': 10.0,  # 功能团1和2之间: 10Å
        '0-2': 12.0   # 功能团0和2之间: 12Å
    }

    nanozyme = assembler.assemble_from_pdb_list(
        pdb_ids=['1acb', '4cha', '1hne'],
        n_functional_groups=3,
        site_threshold=0.7,
        target_distances=target_distances  # 👈 关键参数
    )

    assembler.export_nanozyme(nanozyme, 'output/nanozyme_custom_dist')
    assembler.generate_report(nanozyme, 'output/nanozyme_custom_dist_report.txt')

    print("\n✓ 示例2完成! 查看 output/nanozyme_custom_dist.*")


def example_3_filter_by_type():
    """
    示例3: 按功能团类型过滤
    只使用特定类型的催化功能团
    """
    print("\n" + "="*70)
    print("示例3: 按功能团类型过滤")
    print("="*70)

    assembler = NanozymeAssembler(
        model_path='models/best_model.pt',
        scaffold_type='aromatic_ring',  # 使用芳香环骨架
        scaffold_params={
            'ring_size': 6,
            'substitution_pattern': 'meta'
        }
    )

    nanozyme = assembler.assemble_from_pdb_list(
        pdb_ids=['1acb', '4cha', '1hne', '1ppf', '1sgc'],
        n_functional_groups=3,
        site_threshold=0.7,
        # 只使用咪唑环(His)和羧基(Asp/Glu)
        filter_by_type=['imidazole', 'carboxylate']  # 👈 类型过滤
    )

    assembler.export_nanozyme(nanozyme, 'output/nanozyme_filtered_type')
    assembler.generate_report(nanozyme, 'output/nanozyme_filtered_type_report.txt')

    print("\n✓ 示例3完成! 查看 output/nanozyme_filtered_type.*")


def example_4_filter_by_role():
    """
    示例4: 按催化角色过滤
    构建特定催化机制的纳米酶
    """
    print("\n" + "="*70)
    print("示例4: 按催化角色过滤")
    print("="*70)

    assembler = NanozymeAssembler(
        model_path='models/best_model.pt',
        scaffold_type='carbon_chain'
    )

    nanozyme = assembler.assemble_from_pdb_list(
        pdb_ids=['1acb', '4cha', '1hne'],
        n_functional_groups=3,
        site_threshold=0.7,
        # 只使用亲核试剂和广义碱
        filter_by_role=['nucleophile', 'general_base']  # 👈 角色过滤
    )

    assembler.export_nanozyme(nanozyme, 'output/nanozyme_filtered_role')
    assembler.generate_report(nanozyme, 'output/nanozyme_filtered_role_report.txt')

    print("\n✓ 示例4完成! 查看 output/nanozyme_filtered_role.*")


def example_5_metal_framework():
    """
    示例5: 金属配位框架骨架
    构建MOF风格的金属-有机纳米酶
    """
    print("\n" + "="*70)
    print("示例5: 金属配位框架骨架")
    print("="*70)

    assembler = NanozymeAssembler(
        model_path='models/best_model.pt',
        scaffold_type='metal_framework',  # 👈 金属框架
        scaffold_params={
            'metal_type': 'Fe',  # 铁中心
            'linker_type': 'carboxylate',
            'coordination_number': 6
        }
    )

    # 使用含金属中心的酶
    pdb_ids = [
        '1a5t',  # 磷酸三酯酶 (Zn-Zn双金属)
        '1hdh',  # 肝醇脱氢酶 (Zn)
        '1mbo',  # 肌红蛋白 (Fe)
    ]

    nanozyme = assembler.assemble_from_pdb_list(
        pdb_ids=pdb_ids,
        n_functional_groups=4,
        site_threshold=0.6,
        ec_filter=1  # 氧化还原酶
    )

    assembler.export_nanozyme(nanozyme, 'output/nanozyme_metal_framework')
    assembler.generate_report(nanozyme, 'output/nanozyme_metal_framework_report.txt')

    print("\n✓ 示例5完成! 查看 output/nanozyme_metal_framework.*")


def example_6_from_directory():
    """
    示例6: 从PDB文件夹批量组装
    处理本地PDB文件
    """
    print("\n" + "="*70)
    print("示例6: 从PDB文件夹批量组装")
    print("="*70)

    assembler = NanozymeAssembler(
        model_path='models/best_model.pt',
        scaffold_type='carbon_chain'
    )

    # 假设你有一个包含PDB文件的文件夹
    pdb_dir = 'data/my_pdbs/'

    if Path(pdb_dir).exists():
        nanozyme = assembler.assemble_from_directory(
            pdb_dir=pdb_dir,
            n_functional_groups=3,
            site_threshold=0.7,
            pattern="*.pdb"  # 匹配所有.pdb文件
        )

        assembler.export_nanozyme(nanozyme, 'output/nanozyme_from_dir')
        assembler.generate_report(nanozyme, 'output/nanozyme_from_dir_report.txt')

        print("\n✓ 示例6完成! 查看 output/nanozyme_from_dir.*")
    else:
        print(f"\n⚠ 跳过示例6: 目录不存在 {pdb_dir}")


def example_7_batch_assembly():
    """
    示例7: 批量组装多个纳米酶
    一次性生成多个不同的纳米酶设计
    """
    print("\n" + "="*70)
    print("示例7: 批量组装多个纳米酶")
    print("="*70)

    assembler = NanozymeAssembler(
        model_path='models/best_model.pt',
        scaffold_type='carbon_chain'
    )

    # 定义多组PDB列表
    pdb_lists = [
        ['1acb', '4cha', '1hne'],  # 丝氨酸蛋白酶组
        ['1ppf', '1sgc', '1ela'],  # 另一组蛋白酶
        ['1a5t', '1hdh', '1mbo'],  # 金属酶组
    ]

    nanozymes = assembler.batch_assemble(
        pdb_lists=pdb_lists,
        output_dir='output/batch_nanozymes/',
        n_functional_groups=3,
        site_threshold=0.7
    )

    print(f"\n✓ 示例7完成! 成功组装 {len(nanozymes)} 个纳米酶")
    print("  查看 output/batch_nanozymes/")


def example_8_advanced_workflow():
    """
    示例8: 高级工作流
    完全自定义的组装流程
    """
    print("\n" + "="*70)
    print("示例8: 高级工作流")
    print("="*70)

    from catalytic_triad_net.prediction import BatchCatalyticScreener
    from catalytic_triad_net.generation import (
        FunctionalGroupExtractor,
        ScaffoldBuilder
    )

    # 步骤1: 独立使用筛选器
    print("\n[步骤1] 筛选催化中心...")
    screener = BatchCatalyticScreener(
        model_path='models/best_model.pt'
    )

    screening_results = screener.screen_pdb_list(
        pdb_ids=['1acb', '4cha', '1hne'],
        site_threshold=0.7,
        top_k=10
    )

    # 打印统计
    screener.print_statistics(screening_results)

    # 导出筛选结果
    screener.export_summary(screening_results, 'output/screening_summary.csv')

    # 步骤2: 独立使用功能团提取器
    print("\n[步骤2] 提取功能团...")
    extractor = FunctionalGroupExtractor()

    functional_groups = extractor.extract_from_screening_results(
        screening_results,
        top_n=10
    )

    # 打印统计
    extractor.print_statistics(functional_groups)

    # 导出功能团
    extractor.export_to_json(functional_groups, 'output/functional_groups.json')
    extractor.export_to_xyz(functional_groups, 'output/functional_groups.xyz')

    # 步骤3: 独立使用骨架构建器
    print("\n[步骤3] 构建骨架...")
    builder = ScaffoldBuilder(
        scaffold_type='carbon_chain',
        scaffold_params={'chain_length': 3}
    )

    # 选择前3个功能团
    selected_groups = functional_groups[:3]

    nanozyme = builder.build_nanozyme(
        functional_groups=selected_groups,
        target_distances={'0-1': 10.0, '1-2': 10.0},
        optimize=True
    )

    # 导出
    builder.export_to_xyz(nanozyme, 'output/nanozyme_advanced.xyz')
    builder.export_to_pdb(nanozyme, 'output/nanozyme_advanced.pdb')
    builder.export_to_mol2(nanozyme, 'output/nanozyme_advanced.mol2')
    builder.visualize_with_pymol(nanozyme, 'output/nanozyme_advanced.pml')

    print("\n✓ 示例8完成! 查看 output/nanozyme_advanced.*")


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("纳米酶组装示例集")
    print("="*70)
    print("\n这些示例展示了如何使用CatalyticTriadNet进行纳米酶设计")
    print("从天然酶中提取催化中心，用骨架连接，生成纳米酶结构\n")

    # 创建输出目录
    Path('output').mkdir(exist_ok=True)

    # 运行示例（根据需要注释/取消注释）
    try:
        example_1_basic_assembly()
    except Exception as e:
        print(f"示例1失败: {e}")

    try:
        example_2_custom_distances()
    except Exception as e:
        print(f"示例2失败: {e}")

    try:
        example_3_filter_by_type()
    except Exception as e:
        print(f"示例3失败: {e}")

    try:
        example_4_filter_by_role()
    except Exception as e:
        print(f"示例4失败: {e}")

    try:
        example_5_metal_framework()
    except Exception as e:
        print(f"示例5失败: {e}")

    try:
        example_6_from_directory()
    except Exception as e:
        print(f"示例6失败: {e}")

    try:
        example_7_batch_assembly()
    except Exception as e:
        print(f"示例7失败: {e}")

    try:
        example_8_advanced_workflow()
    except Exception as e:
        print(f"示例8失败: {e}")

    print("\n" + "="*70)
    print("所有示例完成!")
    print("="*70)
    print("\n查看 output/ 目录获取生成的纳米酶结构")


if __name__ == "__main__":
    # 运行单个示例
    # example_1_basic_assembly()

    # 或运行所有示例
    main()
