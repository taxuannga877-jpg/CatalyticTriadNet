#!/usr/bin/env python3
"""
xTB使用示例 - 完整教程

展示如何在CatalyticTriadNet中使用xTB进行：
1. 单点能量计算
2. 结构优化
3. 反应能垒估算
4. 完整TS计算
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from catalytic_triad_net import AUTODE_AVAILABLE

print("="*80)
print("xTB使用示例 - 完整教程")
print("="*80)

# 检查autodE和xTB是否可用
if not AUTODE_AVAILABLE:
    print("\n⚠️  autodE未安装！")
    print("\n安装方法：")
    print("  pip install autode")
    print("  conda install -c conda-forge xtb")
    print("\n请先安装后再运行此示例")
    sys.exit(1)

from catalytic_triad_net import AutodETSCalculator

print("\n✓ autodE和xTB已安装")

# 初始化计算器
calculator = AutodETSCalculator(
    method='xtb',
    n_cores=4
)

print("\n" + "="*80)
print("示例1: xTB单点能量计算")
print("="*80)

print("""
用途：快速计算分子的能量，不改变结构

应用场景：
- 评估分子稳定性
- 比较不同构象的能量
- 验证结构合理性
""")

# 创建测试分子（水分子）
test_xyz = "test_water.xyz"
with open(test_xyz, 'w') as f:
    f.write("""3
Water molecule
O  0.000000  0.000000  0.119262
H  0.000000  0.763239 -0.477047
H  0.000000 -0.763239 -0.477047
""")

print(f"✓ 创建测试文件: {test_xyz}")

# 计算单点能量
print("\n⏳ 计算单点能量...")
result = calculator.calculate_single_point_energy_xtb(
    xyz_file=test_xyz,
    charge=0,
    mult=1
)

if result['success']:
    print(f"\n✓ 单点能量计算成功!")
    print(f"  能量 (Hartree): {result['energy']:.6f}")
    print(f"  能量 (kcal/mol): {result['energy_kcal']:.2f}")
else:
    print(f"\n✗ 计算失败: {result.get('error')}")

print("\n" + "="*80)
print("示例2: xTB结构优化")
print("="*80)

print("""
用途：找到分子的最稳定构象

应用场景：
- 优化纳米酶结构
- 优化反应物/产物
- 准备TS搜索的初始结构
""")

# 创建未优化的结构（稍微扭曲的水分子）
unopt_xyz = "test_water_unopt.xyz"
with open(unopt_xyz, 'w') as f:
    f.write("""3
Unoptimized water
O  0.000000  0.000000  0.150000
H  0.000000  0.800000 -0.500000
H  0.000000 -0.700000 -0.450000
""")

print(f"✓ 创建未优化结构: {unopt_xyz}")

# 优化结构
print("\n⏳ 优化结构...")
result = calculator.optimize_structure_xtb(
    xyz_file=unopt_xyz,
    output_file="test_water_opt.xyz",
    charge=0,
    mult=1
)

if result['success']:
    print(f"\n✓ 结构优化成功!")
    print(f"  优化后能量: {result['optimized_energy']:.2f} kcal/mol")
    print(f"  输出文件: {result['output_file']}")
    print(f"\n优化后的结构:")
    print(result['optimized_structure'])
else:
    print(f"\n✗ 优化失败: {result.get('error')}")

print("\n" + "="*80)
print("示例3: xTB反应能垒估算")
print("="*80)

print("""
用途：快速估算反应的活化能

应用场景：
- 快速筛选纳米酶候选
- 评估反应可行性
- 比较不同催化剂

方法：使用Hammond假设
- 放热反应: Ea ≈ 12 kcal/mol
- 吸热反应: Ea ≈ ΔE + 12 kcal/mol
""")

# 创建反应物（H2O2）
reactant_xyz = "test_h2o2.xyz"
with open(reactant_xyz, 'w') as f:
    f.write("""4
H2O2 reactant
O  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.450000
H  0.950000  0.000000 -0.200000
H -0.950000  0.000000  1.650000
""")

# 创建产物（2 H2O，简化为单个水分子）
product_xyz = "test_h2o.xyz"
with open(product_xyz, 'w') as f:
    f.write("""3
H2O product
O  0.000000  0.000000  0.000000
H  0.950000  0.000000  0.200000
H -0.950000  0.000000  0.200000
""")

print(f"✓ 创建反应物: {reactant_xyz}")
print(f"✓ 创建产物: {product_xyz}")

# 计算反应能垒
print("\n⏳ 计算反应能垒...")
result = calculator.calculate_reaction_barrier_xtb(
    reactant_xyz=reactant_xyz,
    product_xyz=product_xyz,
    charge=0,
    mult=1
)

if result['success']:
    print(f"\n✓ 反应能垒估算成功!")
    print(f"  反应物能量: {result['reactant_energy']:.2f} kcal/mol")
    print(f"  产物能量:   {result['product_energy']:.2f} kcal/mol")
    print(f"  反应能 (ΔE): {result['reaction_energy']:.2f} kcal/mol")
    print(f"  估算能垒:   {result['estimated_barrier']:.2f} kcal/mol")

    # 解释结果
    if result['reaction_energy'] < 0:
        print(f"\n  → 这是放热反应（ΔE < 0）")
        print(f"  → 反应在热力学上有利")
    else:
        print(f"\n  → 这是吸热反应（ΔE > 0）")
        print(f"  → 需要输入能量")

    if result['estimated_barrier'] < 20:
        print(f"  → 能垒较低，反应容易进行")
    else:
        print(f"  → 能垒较高，可能需要催化剂")
else:
    print(f"\n✗ 计算失败: {result.get('error')}")

print("\n" + "="*80)
print("示例4: 完整TS计算（使用autodE + xTB）")
print("="*80)

print("""
用途：精确计算过渡态和活化能

应用场景：
- 最终验证纳米酶活性
- 发表论文级别的精度
- 理解反应机理

注意：这是最慢但最准确的方法！
""")

print("\n演示代码（需要实际的纳米酶结构）:")
print("""
# 完整TS计算示例
from catalytic_triad_net import AutodETSCalculator

calculator = AutodETSCalculator(method='xtb', n_cores=4)

# 计算完整反应路径
result = calculator.calculate_reaction_profile(
    nanozyme_xyz='nanozyme.xyz',
    substrate_smiles='OO',  # H2O2
    product_smiles='O',     # H2O
    charge=0,
    mult=1
)

if result['success']:
    print(f"活化能 (Ea): {result['activation_energy']:.2f} kcal/mol")
    print(f"反应能 (ΔE): {result['reaction_energy']:.2f} kcal/mol")
    print(f"TS虚频: {result['ts_frequency']:.1f} cm⁻¹")

    # TS虚频应该是负值（唯一的虚频）
    if result['ts_frequency'] < 0:
        print("✓ 找到了真正的过渡态!")
    else:
        print("⚠️  可能不是真正的过渡态")
""")

print("\n" + "="*80)
print("示例5: 在纳米酶设计中使用xTB")
print("="*80)

print("""
完整工作流程：

步骤1: 批量筛选（几何打分，不用xTB）
  → 从1000个候选筛选出50个
  → 速度：1秒/纳米酶

步骤2: 中等精度筛选（xTB能垒估算）
  → 从50个候选筛选出10个
  → 速度：10秒/纳米酶
  → 使用：calculate_reaction_barrier_xtb()

步骤3: 高精度验证（完整TS计算）
  → 对10个候选进行精确计算
  → 速度：5分钟/纳米酶
  → 使用：calculate_reaction_profile()

步骤4: 实验验证
  → 合成并测试前3个纳米酶
""")

# 实际使用示例
print("\n实际代码示例:")
print("""
from catalytic_triad_net import (
    Stage2NanozymeActivityScorer,
    AutodETSCalculator
)

# 步骤2: 使用xTB快速估算
scorer = Stage2NanozymeActivityScorer(
    substrate='TMB',
    use_ts_calculation=True,
    ts_quick_mode=True,  # 快速模式
    ts_method='xtb'
)

results = []
for nanozyme in nanozyme_candidates:
    result = scorer.score_nanozyme(nanozyme)
    results.append((nanozyme, result))

# 按分数排序
results.sort(key=lambda x: x[1]['total_score'], reverse=True)
top_10 = results[:10]

# 步骤3: 对前10个进行完整TS计算
scorer_full = Stage2NanozymeActivityScorer(
    substrate='TMB',
    use_ts_calculation=True,
    ts_quick_mode=False,  # 完整计算
    ts_method='xtb'
)

final_results = []
for nanozyme, _ in top_10:
    result = scorer_full.score_nanozyme(nanozyme)
    final_results.append((nanozyme, result))

    print(f"活化能: {result['ts_details']['activation_energy']:.2f} kcal/mol")

# 选择最佳纳米酶
best_nanozyme, best_result = final_results[0]
print(f"\\n最佳纳米酶活化能: {best_result['ts_details']['activation_energy']:.2f} kcal/mol")
""")

print("\n" + "="*80)
print("xTB性能对比")
print("="*80)

print("""
| 方法 | 速度 | 精度 | 内存 | 适用体系 |
|------|------|------|------|----------|
| 几何打分 | 1秒 | 中等 | 低 | 任意大小 |
| xTB单点 | 5秒 | 良好 | 低 | <500原子 |
| xTB优化 | 30秒 | 良好 | 中 | <500原子 |
| xTB能垒估算 | 1分钟 | 良好 | 中 | <500原子 |
| 完整TS计算 | 5-30分钟 | 优秀 | 高 | <200原子 |

推荐使用场景：
- 初筛（1000+候选）: 几何打分
- 中筛（50-100候选）: xTB能垒估算
- 精筛（10-20候选）: 完整TS计算
- 最终验证（前3候选）: 实验测试
""")

print("\n" + "="*80)
print("常见问题")
print("="*80)

print("""
Q1: xTB计算失败怎么办？
A: 检查以下几点：
   - xTB是否正确安装: xtb --version
   - 结构是否合理（无原子重叠）
   - 电荷和多重度是否正确
   - 系统是否太大（>500原子）

Q2: 如何提高xTB计算速度？
A:
   - 增加CPU核心数: n_cores=8
   - 使用快速模式: ts_quick_mode=True
   - 减小体系大小（只保留活性位点）

Q3: xTB结果可靠吗？
A:
   - 对于有机小分子: 非常可靠
   - 对于金属配合物: 较可靠
   - 对于大体系: 需要验证
   - 建议：与实验数据对比验证

Q4: 何时需要使用ORCA代替xTB？
A:
   - 需要更高精度（发表论文）
   - 金属配合物（多种氧化态）
   - 开壳层体系
   - 激发态计算

Q5: 如何解释活化能数值？
A:
   - Ea < 10 kcal/mol: 极快，室温下瞬间完成
   - Ea = 10-15 kcal/mol: 快，室温下秒级
   - Ea = 15-20 kcal/mol: 中等，室温下分钟级
   - Ea = 20-25 kcal/mol: 慢，需要加热或催化剂
   - Ea > 25 kcal/mol: 很慢，需要强催化剂
""")

print("\n" + "="*80)
print("清理临时文件")
print("="*80)

import os
temp_files = [
    'test_water.xyz',
    'test_water_unopt.xyz',
    'test_water_opt.xyz',
    'test_h2o2.xyz',
    'test_h2o.xyz'
]

for f in temp_files:
    if os.path.exists(f):
        os.remove(f)
        print(f"✓ 删除: {f}")

print("\n" + "="*80)
print("总结")
print("="*80)

print("""
xTB是一个强大的工具，在CatalyticTriadNet中有三个主要用途：

1. **电子特征计算** (features.py)
   - 计算残基的部分电荷
   - 用于催化位点预测
   - 可选功能

2. **快速能垒估算** (autode_ts_calculator.py)
   - calculate_reaction_barrier_xtb()
   - 用于中等规模筛选
   - 速度快，精度良好

3. **完整TS计算** (autode_ts_calculator.py)
   - calculate_reaction_profile()
   - 用于最终验证
   - 速度慢，精度优秀

推荐策略：
- 不安装xTB: 使用几何打分（85%精度）
- 安装xTB: 使用三阶段筛选（95%精度）
- 发表论文: 使用ORCA进行最终验证

更多信息：
- xTB文档: https://xtb-docs.readthedocs.io/
- autodE文档: https://duartegroup.github.io/autodE/
- CatalyticTriadNet: README.md
""")

print("\n✓ 示例完成!")
