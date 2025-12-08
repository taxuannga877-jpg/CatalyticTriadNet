#!/usr/bin/env python3
"""
CatalyticTriadNet v2.0 - 快速开始示例
演示如何使用项目的核心功能
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def example_1_predict_catalytic_site():
    """示例1: 预测催化位点"""
    print("\n" + "="*60)
    print("示例1: 预测催化位点")
    print("="*60)

    from catalytic_triad_net.prediction.predictor import EnhancedCatalyticSiteInference

    # 注意：这需要预训练模型
    # predictor = EnhancedCatalyticSiteInference(
    #     model_path='models/best_model.pt',
    #     device='cpu'
    # )
    #
    # results = predictor.predict(
    #     pdb_path='1acb.pdb',
    #     site_threshold=0.5
    # )
    #
    # predictor.print_results(results)

    print("✓ 预测器类导入成功")
    print("  提示: 需要预训练模型才能运行预测")
    print("  使用方法: predictor.predict('1acb.pdb')")


def example_2_generate_nanozyme():
    """示例2: 生成纳米酶"""
    print("\n" + "="*60)
    print("示例2: 生成纳米酶")
    print("="*60)

    from catalytic_triad_net.generation.constraints import CatalyticConstraints, GeometricConstraint
    from catalytic_triad_net.generation.generator import CatalyticNanozymeGenerator

    # 创建简单的催化约束
    constraints = CatalyticConstraints(
        anchor_atoms=[
            {
                'idx': 0,
                'role': 'nucleophile',
                'preferred_elements': ['O', 'S'],
                'geometry': 'terminal'
            },
            {
                'idx': 1,
                'role': 'general_base',
                'preferred_elements': ['N'],
                'geometry': 'sp2'
            }
        ],
        distance_constraints=[
            GeometricConstraint(
                constraint_type='distance',
                atom_indices=[0, 1],
                target_value=3.5,
                tolerance=0.5,
                weight=1.0
            )
        ],
        angle_constraints=[],
        coordination_constraints=[],
        charge_requirements={},
        required_elements=['O', 'N', 'C'],
        forbidden_elements=[]
    )

    print("✓ 催化约束创建成功")
    print(f"  - 锚点原子数: {len(constraints.anchor_atoms)}")
    print(f"  - 距离约束数: {len(constraints.distance_constraints)}")
    print(f"  - 必需元素: {constraints.required_elements}")

    # 注意：生成器需要预训练的扩散模型
    # generator = CatalyticNanozymeGenerator(model_path='models/diffusion_model.pt')
    # structures = generator.generate(constraints, n_samples=5)

    print("\n  提示: 需要预训练的扩散模型才能生成纳米酶")
    print("  使用方法: generator.generate(constraints, n_samples=5)")


def example_3_visualize_results():
    """示例3: 可视化结果"""
    print("\n" + "="*60)
    print("示例3: 可视化结果")
    print("="*60)

    from catalytic_triad_net.visualization.visualizer import NanozymeVisualizer
    import numpy as np

    # 创建可视化器
    visualizer = NanozymeVisualizer()

    # 模拟预测结果
    mock_results = {
        'pdb_id': 'example',
        'catalytic_residues': [
            {
                'index': 0,
                'resname': 'SER',
                'resseq': 195,
                'chain': 'A',
                'site_prob': 0.95,
                'role_name': 'nucleophile'
            },
            {
                'index': 1,
                'resname': 'HIS',
                'resseq': 57,
                'chain': 'A',
                'site_prob': 0.92,
                'role_name': 'general_base'
            }
        ],
        'triads': [],
        'metal_centers': [],
        'metals': []
    }

    print("✓ 可视化器创建成功")
    print("  支持的可视化模式:")
    print("    - 2D分子图")
    print("    - 2D三联体图")
    print("    - 3D活性位点")
    print("    - 交互式3D (需要Plotly)")

    # 实际使用示例:
    # coords = np.random.randn(100, 3) * 10
    # visualizer.visualize(mock_results, coords, output_dir='./output')


def example_4_use_constraints():
    """示例4: 使用约束系统"""
    print("\n" + "="*60)
    print("示例4: 使用约束系统")
    print("="*60)

    from catalytic_triad_net.generation.constraints import (
        ATOM_TYPES, NUM_ATOM_TYPES, FUNCTIONAL_GROUP_TEMPLATES
    )

    print(f"✓ 支持的原子类型数: {NUM_ATOM_TYPES}")
    print(f"  原子类型: {', '.join(ATOM_TYPES[:10])}...")

    print(f"\n✓ 催化功能团模板数: {len(FUNCTIONAL_GROUP_TEMPLATES)}")
    print("  可用模板:")
    for name, template in list(FUNCTIONAL_GROUP_TEMPLATES.items())[:5]:
        print(f"    - {name}: {template['geometry']}")


def example_5_import_test():
    """示例5: 测试所有核心导入"""
    print("\n" + "="*60)
    print("示例5: 测试所有核心导入")
    print("="*60)

    try:
        from catalytic_triad_net import (
            CatalyticTriadPredictor,
            EnhancedCatalyticSiteInference,
            CatalyticNanozymeGenerator,
            CatalyticConstraints,
            NanozymeVisualizer,
            __version__
        )

        print(f"✓ CatalyticTriadNet v{__version__} 导入成功")
        print("✓ 所有核心类可用:")
        print("  - CatalyticTriadPredictor")
        print("  - EnhancedCatalyticSiteInference")
        print("  - CatalyticNanozymeGenerator")
        print("  - CatalyticConstraints")
        print("  - NanozymeVisualizer")

        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("CatalyticTriadNet v2.0 - 快速开始示例")
    print("="*70)

    # 测试导入
    if not example_5_import_test():
        print("\n错误: 导入失败，请检查安装")
        return

    # 运行示例
    example_1_predict_catalytic_site()
    example_2_generate_nanozyme()
    example_3_visualize_results()
    example_4_use_constraints()

    print("\n" + "="*70)
    print("✓ 所有示例运行完成！")
    print("="*70)
    print("\n下一步:")
    print("  1. 训练模型: python -m catalytic_triad_net.cli train")
    print("  2. 预测位点: python -m catalytic_triad_net.cli predict --pdb 1acb")
    print("  3. 查看文档: cat README.md")
    print()


if __name__ == "__main__":
    main()
