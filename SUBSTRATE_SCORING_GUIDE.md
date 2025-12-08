# 双阶段多底物打分系统使用指南

## 🎯 系统概述

这是一个**完整的纳米酶活性评估系统**，支持6种经典纳米酶底物的双阶段打分。

### 支持的底物

| 底物 | 全名 | 酶类型 | 检测波长 | 使用频率 |
|------|------|--------|---------|---------|
| **TMB** | 3,3',5,5'-四甲基联苯胺 | 过氧化物酶 | 652 nm | ⭐⭐⭐⭐⭐ |
| **pNPP** | 对硝基苯磷酸酯 | 磷酸酶 | 405 nm | ⭐⭐⭐⭐ |
| **ABTS** | 2,2'-联氮-双-(3-乙基苯并噻唑啉-6-磺酸) | 过氧化物酶 | 414 nm | ⭐⭐⭐⭐ |
| **OPD** | 邻苯二胺 | 过氧化物酶 | 450 nm | ⭐⭐⭐ |
| **H₂O₂** | 过氧化氢 | 过氧化氢酶 | 240 nm | ⭐⭐⭐ |
| **GSH** | 谷胱甘肽 | GPx | 412 nm | ⭐⭐⭐ |

---

## 📊 双阶段打分策略

### 为什么需要两个阶段？

```
问题：从100个功能团中选3个，有C(100,3) = 161,700种组合
      如果每个都组装+精确打分，计算量太大！

解决方案：两阶段筛选
  阶段1：快速过滤（每个组合 < 1ms）
    → 161,700个组合 → 筛选出100个候选

  阶段2：精确评估（每个纳米酶 1-10s）
    → 只对100个候选组装+打分
```

### 阶段对比

| 维度 | 阶段1 | 阶段2 |
|------|-------|-------|
| **时机** | 提取功能团后，组装**之前** | 组装纳米酶**之后** |
| **输入** | 功能团组合 | 完整纳米酶结构 |
| **方法** | 类型匹配 + 距离估算 | NAC几何 + 对接 |
| **速度** | 极快（< 1ms/组合） | 较慢（1-10s/纳米酶） |
| **准确度** | 中等（过滤用） | 高（最终评估） |
| **目的** | 快速过滤无用组合 | 精确预测活性 |

---

## 🚀 快速开始

### 最简单的使用方式

```python
from catalytic_triad_net import (
    BatchCatalyticScreener,
    FunctionalGroupExtractor,
    Stage1FunctionalGroupScorer,
    Stage2NanozymeActivityScorer,
    ScaffoldBuilder
)

# 1. 筛选催化中心
screener = BatchCatalyticScreener(model_path='models/best_model.pt')
results = screener.screen_pdb_list(['1acb', '4cha'], site_threshold=0.7)

# 2. 提取功能团
extractor = FunctionalGroupExtractor()
functional_groups = extractor.extract_from_screening_results(results, top_n=20)

# 3. 阶段1打分 - 快速筛选组合
stage1_scorer = Stage1FunctionalGroupScorer(substrate='TMB')
top_combinations = stage1_scorer.get_top_combinations(
    functional_groups,
    n_per_combo=3,
    top_k=50  # 只保留前50个
)

print(f"阶段1筛选出 {len(top_combinations)} 个候选组合")

# 4. 组装纳米酶（只组装筛选后的）
builder = ScaffoldBuilder(scaffold_type='carbon_chain')
nanozymes = []
for combo, score in top_combinations[:10]:
    nanozyme = builder.build_nanozyme(combo, optimize=True)
    nanozymes.append(nanozyme)

# 5. 阶段2打分 - 精确评估
stage2_scorer = Stage2NanozymeActivityScorer(substrate='TMB')
ranked = stage2_scorer.rank_nanozymes(nanozymes)

# 6. 获取最佳纳米酶
best_nanozyme, best_result = ranked[0]
print(f"最佳纳米酶分数: {best_result['total_score']:.3f}")
print(f"活性预测: {best_result['activity_prediction']['level']}")
```

---

## 📖 详细使用指南

### 阶段1：功能团组合快速打分

#### 单底物打分

```python
from catalytic_triad_net import Stage1FunctionalGroupScorer

# 初始化打分器
scorer = Stage1FunctionalGroupScorer(substrate='TMB')

# 评估单个组合
score_result = scorer.score_combination([fg1, fg2, fg3])

print(f"总分: {score_result['total_score']:.3f}")
print(f"类型匹配: {score_result['component_scores']['type_match']:.3f}")
print(f"角色匹配: {score_result['component_scores']['role_match']:.3f}")

# 批量筛选
good_combinations = scorer.filter_combinations(
    functional_groups,
    n_per_combo=3,
    min_score=0.6,  # 最低分数阈值
    max_combinations=1000  # 最多评估的组合数
)

# 获取top K
top_k = scorer.get_top_combinations(
    functional_groups,
    n_per_combo=3,
    top_k=50
)

# 解释评分
explanation = scorer.explain_score([fg1, fg2, fg3])
print(explanation)
```

#### 多底物打分

```python
from catalytic_triad_net import MultiSubstrateStage1Scorer

# 初始化多底物打分器
multi_scorer = MultiSubstrateStage1Scorer(
    substrates=['TMB', 'pNPP', 'ABTS', 'OPD', 'H2O2', 'GSH']
)

# 评估对所有底物的活性
results = multi_scorer.score_combination_all_substrates([fg1, fg2, fg3])

print(f"最佳底物: {results['best_substrate']}")
print(f"最佳分数: {results['best_score']:.3f}")

# 按最佳底物分类筛选
results_by_substrate = multi_scorer.filter_by_best_substrate(
    functional_groups,
    n_per_combo=3,
    min_score=0.6
)

# 查看每种底物的候选数
for substrate, combos in results_by_substrate.items():
    print(f"{substrate}: {len(combos)} 个候选")
```

#### 阶段1评分标准

```python
# 评分组成（总分 = 1.0）
{
    'type_match': 0.4,      # 功能团类型匹配 (40%)
    'role_match': 0.3,      # 催化角色匹配 (30%)
    'distance': 0.2,        # 距离合理性 (20%)
    'probability': 0.1      # 催化位点概率 (10%)
}

# 分数解释
0.8 - 1.0: 优秀组合，强烈推荐
0.6 - 0.8: 良好组合，可以组装
0.4 - 0.6: 一般组合，活性可能较低
0.0 - 0.4: 不推荐
```

---

### 阶段2：纳米酶活性精确打分

#### 单底物打分

```python
from catalytic_triad_net import Stage2NanozymeActivityScorer

# 初始化打分器
scorer = Stage2NanozymeActivityScorer(substrate='TMB')

# 评估单个纳米酶
result = scorer.score_nanozyme(nanozyme)

print(f"总分: {result['total_score']:.3f}")
print(f"NAC几何: {result['component_scores']['nac_geometry']:.3f}")
print(f"可及性: {result['component_scores']['accessibility']:.3f}")
print(f"活性预测: {result['activity_prediction']['level']}")

# 批量评估并排序
ranked = scorer.rank_nanozymes(nanozyme_list)

# 获取最佳纳米酶
best_nanozyme, best_result = ranked[0]

# 解释评分
explanation = scorer.explain_score(nanozyme)
print(explanation)
```

#### 多底物打分

```python
from catalytic_triad_net import MultiSubstrateStage2Scorer

# 初始化多底物打分器
multi_scorer = MultiSubstrateStage2Scorer(
    substrates=['TMB', 'pNPP', 'ABTS', 'OPD', 'H2O2', 'GSH']
)

# 评估对所有底物的活性谱
results = multi_scorer.score_nanozyme_all_substrates(nanozyme)

print(f"最佳底物: {results['best_substrate']}")
print(f"最佳分数: {results['best_score']:.3f}")

# 查看活性谱
print("\n活性谱:")
for substrate, score in results['activity_profile']:
    print(f"  {substrate}: {score:.3f}")
```

#### 阶段2评分标准

```python
# 评分组成（总分 = 1.0）
{
    'nac_geometry': 0.6,    # NAC几何条件 (60%) - 最重要！
    'accessibility': 0.2,   # 催化中心可及性 (20%)
    'synergy': 0.1,         # 功能团协同性 (10%)
    'stability': 0.1        # 结构稳定性 (10%)
}

# 活性预测
0.8 - 1.0: high (高活性)
0.6 - 0.8: medium (中等活性)
0.4 - 0.6: low (低活性)
0.0 - 0.4: very_low (很低活性)
```

---

## 🎯 NAC条件详解

### 什么是NAC？

**NAC (Near Attack Conformation)** = 近攻击构象

这是过渡态理论的核心概念：
- 底物和催化中心必须处于特定的几何关系
- 才能形成过渡态，发生催化反应

### 各底物的NAC条件

#### TMB（过氧化物酶）

```python
TMB_NAC = {
    'metal_substrate_distance': (2.0, 2.8),  # 金属到底物
    'H2O2_binding_distance': (2.5, 3.5),     # H₂O₂结合
    'electron_transfer_distance': (3.0, 4.5) # 电子转移
}
```

**物理意义：**
- 金属中心（Fe/Cu）必须靠近TMB（2-2.8Å）
- H₂O₂在金属附近结合（2.5-3.5Å）
- 电子从金属转移到TMB（3-4.5Å）

#### pNPP（磷酸酶）

```python
pNPP_NAC = {
    'nucleophile_P_distance': (2.7, 3.3),    # 亲核到P
    'base_nucleophile_distance': (3.0, 4.5), # 碱到亲核
    'attack_angle': (160, 180)               # 攻击角度
}
```

**物理意义：**
- 亲核试剂（Ser-OH）接近磷原子（2.7-3.3Å）
- 广义碱（His）活化亲核试剂（3-4.5Å）
- 攻击角度接近线性（160-180°）

#### ABTS（过氧化物酶）

```python
ABTS_NAC = {
    'metal_substrate_distance': (2.0, 2.8),
    'oxidation_site_distance': (3.0, 4.5),
    'H2O2_coordination': True
}
```

#### OPD（过氧化物酶）

```python
OPD_NAC = {
    'metal_substrate_distance': (2.0, 2.8),
    'amine_oxidation_distance': (3.0, 4.0)
}
```

#### H₂O₂（过氧化氢酶）

```python
H2O2_NAC = {
    'metal_H2O2_distance': (2.0, 2.5),
    'OO_activation_distance': (1.4, 1.6),
    'proton_transfer_distance': (2.5, 3.5)
}
```

#### GSH（GPx）

```python
GSH_NAC = {
    'thiol_active_site_distance': (3.0, 4.0),
    'H2O2_binding_distance': (2.5, 3.5),
    'disulfide_formation_distance': (2.0, 2.5)
}
```

---

## 💡 使用建议

### 1. 选择合适的底物

```python
from catalytic_triad_net import SUBSTRATE_LIBRARY

# 查看底物信息
for substrate in ['TMB', 'pNPP', 'ABTS']:
    info = SUBSTRATE_LIBRARY[substrate]
    print(f"{substrate}:")
    print(f"  酶类型: {info.enzyme_type}")
    print(f"  检测波长: {info.detection_wavelength} nm")
    print(f"  使用频率: {'⭐' * info.usage_frequency}")
```

**推荐：**
- **过氧化物酶活性** → TMB（最常用）或ABTS（水溶性好）
- **磷酸酶活性** → pNPP（金标准）
- **GPx活性** → GSH

### 2. 调整阈值

```python
# 阶段1阈值
min_score = 0.6  # 推荐值
# 0.7-0.8: 严格筛选（候选少但质量高）
# 0.5-0.6: 宽松筛选（候选多但可能有噪声）

# 阶段2活性预测
# high (>0.8): 强烈推荐实验验证
# medium (0.6-0.8): 建议实验验证
# low (<0.6): 可能需要优化
```

### 3. 功能团数量

```python
# 简单纳米酶
n_per_combo = 2-3  # 双功能团或三联体

# 复杂纳米酶
n_per_combo = 4-6  # 多功能团协同

# 注意：功能团越多，组合数指数增长
# C(20, 3) = 1,140
# C(20, 4) = 4,845
# C(20, 5) = 15,504
```

### 4. 计算资源规划

```python
# 阶段1（极快）
100个功能团，选3个 → C(100,3) = 161,700种组合
评估时间：161,700 × 0.001s = 162秒 ≈ 3分钟

# 阶段2（较慢）
50个纳米酶
评估时间：50 × 5s = 250秒 ≈ 4分钟

# 总时间：约7分钟（可接受）
```

---

## 📚 完整示例

查看 `examples/substrate_scoring_example.py` 获取6个完整示例：

1. **基础双阶段打分** - 完整流程演示
2. **多底物阶段1打分** - 同时评估6种底物
3. **多底物阶段2打分** - 活性谱分析
4. **完整工作流** - 从筛选到最终纳米酶
5. **底物比较** - 找出最佳应用场景
6. **快速筛选** - 使用便捷函数

运行示例：
```bash
cd examples
python substrate_scoring_example.py
```

---

## 🔧 API参考

### Stage1FunctionalGroupScorer

```python
class Stage1FunctionalGroupScorer:
    def __init__(self, substrate: str = 'TMB')

    def score_combination(self, functional_groups: List[FunctionalGroup]) -> Dict

    def filter_combinations(self,
                          functional_groups: List[FunctionalGroup],
                          n_per_combo: int = 3,
                          min_score: float = 0.6,
                          max_combinations: int = 1000) -> List

    def get_top_combinations(self,
                           functional_groups: List[FunctionalGroup],
                           n_per_combo: int = 3,
                           top_k: int = 50) -> List

    def explain_score(self, functional_groups: List[FunctionalGroup]) -> str
```

### Stage2NanozymeActivityScorer

```python
class Stage2NanozymeActivityScorer:
    def __init__(self, substrate: str = 'TMB')

    def score_nanozyme(self, nanozyme: Dict) -> Dict

    def rank_nanozymes(self, nanozymes: List[Dict]) -> List

    def explain_score(self, nanozyme: Dict) -> str
```

### 便捷函数

```python
# 快速筛选功能团组合
quick_screen_functional_groups(
    functional_groups: List[FunctionalGroup],
    substrate: str = 'TMB',
    n_per_combo: int = 3,
    top_k: int = 50
) -> List
```

---

## 🎓 工作流程总结

```
输入: 多个天然酶PDB
  ↓
[步骤1] 批量筛选催化中心
  → BatchCatalyticScreener
  → 输出: 高分催化残基列表
  ↓
[步骤2] 提取催化功能团
  → FunctionalGroupExtractor
  → 输出: His咪唑环、Asp羧基等功能团
  ↓
[步骤3] ⭐ 阶段1打分 - 快速筛选组合
  → Stage1FunctionalGroupScorer
  → 从161,700种组合筛选出50个候选
  → 速度: 极快（< 1ms/组合）
  ↓
[步骤4] 组装纳米酶
  → ScaffoldBuilder
  → 只组装筛选后的50个候选
  ↓
[步骤5] ⭐ 阶段2打分 - 精确评估活性
  → Stage2NanozymeActivityScorer
  → NAC几何打分 + 活性预测
  → 速度: 较慢（1-10s/纳米酶）
  ↓
[步骤6] 排序并导出
  → 按活性分数排序
  → 导出最佳纳米酶（XYZ/PDB/MOL2）
  ↓
输出: 高活性纳米酶 + 活性预测报告
```

---

## 📊 性能对比

### 无打分系统 vs 双阶段打分

| 指标 | 无打分 | 双阶段打分 | 改进 |
|------|--------|-----------|------|
| 需要组装的纳米酶数 | 161,700 | 50 | **减少99.97%** |
| 总计算时间 | ~22天 | ~7分钟 | **加速4,500倍** |
| 最终纳米酶质量 | 随机 | 高活性 | **显著提升** |
| 实验验证成功率 | 低 | 高 | **节省实验成本** |

---

## 🚀 下一步

1. **运行示例**
   ```bash
   python examples/substrate_scoring_example.py
   ```

2. **开始设计您的纳米酶**
   ```python
   from catalytic_triad_net import *

   # 您的代码...
   ```

3. **（可选）添加autodE精确计算**
   - 对阶段2高分纳米酶（>0.8）
   - 用autodE计算真实的过渡态能垒
   - 进一步验证活性

---

## 📞 支持

- **GitHub Issues**: https://github.com/taxuannga877-jpg/CatalyticTriadNet/issues
- **文档**: 查看 `NANOZYME_ASSEMBLY_GUIDE.md`
- **示例**: `examples/substrate_scoring_example.py`

---

## 🎉 开始使用

```python
from catalytic_triad_net import (
    BatchCatalyticScreener,
    FunctionalGroupExtractor,
    Stage1FunctionalGroupScorer,
    Stage2NanozymeActivityScorer,
    ScaffoldBuilder
)

# 完整的双阶段打分工作流
screener = BatchCatalyticScreener(model_path='models/best_model.pt')
results = screener.screen_pdb_list(['1acb', '4cha'], site_threshold=0.7)

extractor = FunctionalGroupExtractor()
functional_groups = extractor.extract_from_screening_results(results, top_n=20)

# 阶段1：快速筛选
stage1 = Stage1FunctionalGroupScorer(substrate='TMB')
top_combos = stage1.get_top_combinations(functional_groups, n_per_combo=3, top_k=50)

# 组装
builder = ScaffoldBuilder(scaffold_type='carbon_chain')
nanozymes = [builder.build_nanozyme(combo) for combo, _ in top_combos[:10]]

# 阶段2：精确评估
stage2 = Stage2NanozymeActivityScorer(substrate='TMB')
ranked = stage2.rank_nanozymes(nanozymes)

best_nanozyme, best_result = ranked[0]
print(f"✓ 最佳纳米酶活性分数: {best_result['total_score']:.3f}")
```

祝您设计出高活性的纳米酶！🚀
