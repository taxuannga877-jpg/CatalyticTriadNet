# 片段化纳米酶生成系统

## 概述

本系统实现了基于 **StoL (Structure-to-Ligand)** 思想的片段化纳米酶生成方法，这是对传统一次性生成整个纳米酶结构的重大改进。

### 核心创新

借鉴 StoL 论文的"分而治之"策略：

1. **片段化**：将纳米酶切分成小片段（催化中心 + 支架片段）
2. **独立生成**：为每个片段生成多个 3D 构象（使用扩散模型）
3. **智能组装**：使用 Kabsch 对齐和接口原子匹配组装片段
4. **多样性探索**：生成 50+ 个候选构象，通过聚类选择代表性结构
5. **化学验证**：自动过滤无效结构，确保化学合理性

### 为什么使用片段化生成？

| 传统方法 | 片段化方法 |
|---------|-----------|
| 一次性生成整个纳米酶（50-100 原子） | 分别生成小片段（10-30 原子） |
| 生成空间巨大，难以探索 | 降低复杂度，更容易生成合理结构 |
| 单一构象 | 多样化构象（50+ 个候选） |
| 难以满足复杂约束 | 片段级约束更容易满足 |
| 无质量控制 | 自动化学验证 + 聚类筛选 |

---

## 系统架构

```
片段化纳米酶生成流程
├── 1. 片段化 (Fragmentization)
│   ├── NanozymeFragmentizer
│   └── FragmentationRule
│
├── 2. 构象生成 (Conformation Generation)
│   ├── FragmentConformationGenerator
│   └── ProgressiveFragmentGenerator (粗糙→精细)
│
├── 3. 片段组装 (Fragment Assembly)
│   ├── FragmentAssembler (Kabsch 对齐)
│   └── MultiConformationAssembler
│
├── 4. 化学验证 (Chemical Validation)
│   └── ChemicalValidator (键长、冲突检查)
│
└── 5. 聚类分析 (Clustering Analysis)
    ├── UMAP 降维
    ├── K-means 聚类
    └── 代表性结构选择
```

---

## 快速开始

### 安装依赖

```bash
pip install torch numpy scipy scikit-learn umap-learn matplotlib
```

### 基础使用

```python
from catalytic_triad_net.generation import create_pipeline
from catalytic_triad_net.generation import NanozymeAssembler

# 1. 创建基础纳米酶（从天然酶提取）
assembler = NanozymeAssembler(
    model_path='models/best_model.pt',
    scaffold_type='carbon_chain'
)

base_nanozyme = assembler.assemble_from_pdb_list(
    pdb_ids=['1acb', '4cha', '1hiv'],
    n_functional_groups=3
)

# 2. 初始化片段化生成流程
pipeline = create_pipeline(
    model_path='models/diffusion_model.pt',
    device='cuda'
)

# 3. 生成多样化构象
results = pipeline.generate_from_nanozyme(
    nanozyme=base_nanozyme,
    n_conformations_per_fragment=50,  # 每个片段 50 个构象
    n_clusters=10,                     # 聚类成 10 个簇
    max_combinations=100,              # 最多 100 个组合
    output_dir='output/nanozymes',
    use_progressive=True               # 使用渐进式生成
)

# 4. 获取代表性结构
representatives = results['representatives']
print(f"生成了 {len(representatives)} 个代表性纳米酶结构")

# 5. 查看统计信息
stats = results['statistics']
print(f"有效率: {stats['validity_rate']:.1%}")
print(f"平均几何分数: {stats['avg_geometry_score']:.3f}")
```

---

## 核心模块详解

### 1. 片段定义 (`fragment_definitions.py`)

定义纳米酶片段的数据结构和切分规则。

```python
from catalytic_triad_net.generation import (
    NanozymeFragment, FragmentType, NanozymeFragmentizer
)

# 创建片段化器
fragmentizer = NanozymeFragmentizer()

# 切分纳米酶
fragments = fragmentizer.fragment_nanozyme(nanozyme)

# 查看片段
for frag in fragments:
    print(f"{frag.fragment_id}: {frag.fragment_type.value}")
    print(f"  原子数: {frag.n_atoms}")
    print(f"  接口原子: {len(frag.interface_atoms)}")
```

**片段类型：**
- `CATALYTIC_CENTER`: 催化中心（功能团簇）
- `SCAFFOLD_CORE`: 支架核心（金属纳米颗粒/碳骨架）
- `LINKER`: 连接片段
- `SUBSTRATE_POCKET`: 底物结合口袋

### 2. 构象生成 (`fragment_conformation_generator.py`)

使用扩散模型为每个片段生成多个构象。

```python
from catalytic_triad_net.generation import create_fragment_generator

# 创建生成器
generator = create_fragment_generator(
    model_path='models/diffusion_model.pt',
    device='cuda'
)

# 为单个片段生成构象
conformations = generator.generate_fragment_conformations(
    fragment=fragment,
    n_conformations=50,
    temperature=1.0  # 控制多样性
)

# 渐进式生成（粗糙→精细）
from catalytic_triad_net.generation import ProgressiveFragmentGenerator

progressive_gen = ProgressiveFragmentGenerator(generator)
refined = progressive_gen.generate_progressive(
    fragment=fragment,
    n_coarse=100,      # 阶段1: 100 个粗糙构象
    n_refined=20,      # 阶段2: 20 个精细构象
    coarse_steps=200,
    refined_steps=1000
)
```

### 3. 片段组装 (`fragment_assembler.py`)

使用 Kabsch 算法对齐并组装片段。

```python
from catalytic_triad_net.generation import create_assembler

# 创建组装器
assembler = create_assembler(
    overlap_threshold=2.0,   # 接口原子重叠阈值
    clash_threshold=0.8      # 原子冲突阈值
)

# 组装两个片段
assembled = assembler.assemble_fragment_pair(
    fragment1=catalytic_center,
    fragment2=scaffold_core
)

# 组装多个片段
assembled = assembler.assemble_fragments(
    fragments=[frag1, frag2, frag3],
    assembly_order=['frag1', 'frag2', 'frag3']
)
```

**核心算法：**
- **Kabsch 对齐**：最小化 RMSD 的刚体变换
- **匈牙利算法**：接口原子最优匹配
- **冲突解决**：自动移除距离过近的原子

### 4. 化学验证 (`conformation_analysis.py`)

验证生成结构的化学合理性。

```python
from catalytic_triad_net.generation import ChemicalValidator

# 创建验证器
validator = ChemicalValidator(
    min_bond_length=0.8,
    max_bond_length=2.5,
    clash_threshold=0.8
)

# 验证单个结构
result = validator.validate_structure(nanozyme)
print(f"有效: {result.is_valid}")
print(f"几何分数: {result.geometry_score:.3f}")
print(f"冲突数: {result.clash_count}")

# 批量验证
valid_nanozymes, results = validator.batch_validate(nanozymes)
print(f"有效率: {len(valid_nanozymes)}/{len(nanozymes)}")
```

**验证标准：**
- ✓ 键长合理性（基于共价半径）
- ✓ 无原子冲突（距离 > 0.8 Å）
- ✓ 几何分数 > 0.5
- ✓ 键长违规 < 10%

### 5. 聚类分析 (`conformation_analysis.py`)

使用 UMAP + K-means 选择代表性结构。

```python
from catalytic_triad_net.generation import create_analyzer

# 创建分析器
analyzer = create_analyzer()

# 完整分析流程
analysis_results = analyzer.analyze_conformations(
    nanozymes=valid_nanozymes,
    n_clusters=10,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    output_dir='output/analysis'
)

# 获取代表性结构
representatives = analysis_results['representatives']

# 查看聚类统计
stats = analysis_results['statistics']
print(f"簇大小分布: {stats['cluster_sizes']}")
```

**输出：**
- `conformation_analysis.png`: UMAP 可视化 + 聚类结果
- `validation_statistics.png`: 验证统计图表
- `cluster_labels.npy`: 聚类标签
- `representative_*.xyz`: 代表性结构

---

## 完整流程示例

### 示例 1: 从 PDB 生成多样化纳米酶

```python
from catalytic_triad_net.generation import NanozymeAssembler, create_pipeline

# 步骤 1: 从天然酶提取催化位点
assembler = NanozymeAssembler(
    model_path='models/best_model.pt',
    scaffold_type='metal_framework'
)

base_nanozyme = assembler.assemble_from_pdb_list(
    pdb_ids=['1acb', '4cha', '1hiv', '2gbp'],
    n_functional_groups=4,
    site_threshold=0.7
)

# 步骤 2: 片段化生成
pipeline = create_pipeline(
    model_path='models/diffusion_model.pt',
    device='cuda'
)

results = pipeline.generate_from_nanozyme(
    nanozyme=base_nanozyme,
    n_conformations_per_fragment=50,
    n_clusters=10,
    output_dir='output/diverse_nanozymes'
)

# 步骤 3: 导出结果
for i, rep in enumerate(results['representatives']):
    print(f"代表性结构 {i}:")
    print(f"  簇 ID: {rep['cluster_id']}")
    print(f"  簇大小: {rep['cluster_size']}")
    print(f"  原子数: {rep['n_atoms']}")
```

### 示例 2: 快速生成变体

```python
# 快速生成 10 个多样化变体
variants = pipeline.generate_diverse_nanozymes(
    base_nanozyme=base_nanozyme,
    n_variants=10,
    diversity_level='high',  # 'low', 'medium', 'high'
    output_dir='output/variants'
)

print(f"生成了 {len(variants)} 个变体")
```

---

## 性能优化建议

### 1. 使用渐进式生成

```python
# 粗糙→精细，节省计算资源
results = pipeline.generate_from_nanozyme(
    nanozyme=base_nanozyme,
    use_progressive=True  # 推荐！
)
```

### 2. 调整采样参数

```python
# 快速模式（低质量）
results = pipeline.generate_from_nanozyme(
    nanozyme=base_nanozyme,
    n_conformations_per_fragment=20,  # 减少构象数
    n_clusters=5,
    max_combinations=50
)

# 高质量模式（慢）
results = pipeline.generate_from_nanozyme(
    nanozyme=base_nanozyme,
    n_conformations_per_fragment=100,
    n_clusters=20,
    max_combinations=200
)
```

### 3. 并行处理

```python
# 多 GPU 并行（未来支持）
pipeline = create_pipeline(
    model_path='models/diffusion_model.pt',
    device='cuda:0'  # 指定 GPU
)
```

---

## 输出文件说明

运行完整流程后，`output_dir` 包含：

```
output/
├── fragments.json                    # 片段信息
├── representative_000.xyz            # 代表性结构 0
├── representative_001.xyz            # 代表性结构 1
├── ...
├── statistics.json                   # 统计信息
├── cluster_labels.npy                # 聚类标签
├── conformation_analysis.png         # UMAP + 聚类可视化
├── validation_statistics.png         # 验证统计图表
└── generation_report.txt             # 文本报告
```

---

## 与 StoL 的对比

| 特性 | StoL (分子生成) | 本系统 (纳米酶生成) |
|------|----------------|-------------------|
| **目标** | 生成小分子药物构象 | 生成纳米酶结构 |
| **片段化** | BRICS/Recap 切分 | 催化中心 + 支架切分 |
| **构象生成** | 扩散模型 | 扩散模型（相同） |
| **片段组装** | Kabsch + Sinkhorn | Kabsch + 匈牙利算法 |
| **验证** | 键长 + RDKit | 键长 + 几何分数 |
| **聚类** | t-SNE/UMAP + K-means | UMAP + K-means（相同） |
| **应用** | 药物发现 | 纳米酶设计 |

---

## 常见问题

### Q1: 为什么片段化生成比传统方法更好？

**A:** 片段化降低了生成复杂度。生成 10 个原子的片段比生成 100 个原子的完整结构容易得多，且更容易满足局部约束。

### Q2: 如何选择片段数量？

**A:** 建议：
- 小纳米酶（< 50 原子）：2-3 个片段
- 中等纳米酶（50-100 原子）：3-5 个片段
- 大纳米酶（> 100 原子）：5-10 个片段

### Q3: 生成需要多长时间？

**A:** 取决于参数：
- 快速模式：~5 分钟（20 构象/片段，50 组合）
- 标准模式：~15 分钟（50 构象/片段，100 组合）
- 高质量模式：~30 分钟（100 构象/片段，200 组合）

### Q4: 如何提高有效率？

**A:**
1. 使用渐进式生成（`use_progressive=True`）
2. 调整验证阈值（`clash_threshold`, `bond_tolerance`）
3. 增加构象数量（更多候选）
4. 使用更好的基础纳米酶

### Q5: 可以用于其他类型的纳米材料吗？

**A:** 可以！只需修改：
1. `FragmentType`：定义新的片段类型
2. `FragmentationRule`：自定义切分规则
3. `ChemicalValidator`：调整验证标准

---

## 引用

如果使用本系统，请引用：

```bibtex
@software{catalytic_triad_net_fragmented,
  title={Fragmented Nanozyme Generation System},
  author={CatalyticTriadNet Team},
  year={2025},
  note={Inspired by StoL: Structure-to-Ligand molecular generation}
}
```

---

## 联系与支持

- **文档**: `docs/FRAGMENTED_GENERATION.md`
- **示例**: `examples/fragmented_nanozyme_generation_example.py`
- **问题**: 提交 GitHub Issue

---

## 更新日志

### v2.1.0 (2025-12-11)
- ✨ 新增片段化纳米酶生成系统
- ✨ 实现 StoL 启发的片段化策略
- ✨ 添加 Kabsch 对齐和接口原子匹配
- ✨ 集成 UMAP 降维和 K-means 聚类
- ✨ 实现化学有效性自动验证
- ✨ 支持渐进式生成（粗糙→精细）
- 📝 完整的文档和示例

---

**祝你生成出优秀的纳米酶结构！** 🚀
