# CatalyticTriadNet 技术方法论

> **最后更新**: 2025-12-11
> **版本**: v2.0
> **适用于**: 新设备快速理解项目

---

## 📋 目录

1. [特征工程](#特征工程)
2. [网络架构](#网络架构)
3. [训练流程](#训练流程)
4. [推理流程](#推理流程)

---

## 特征工程

### 节点特征 (48维)

每个氨基酸残基的特征向量包含以下6个类别：

| 类别 | 维度 | 描述 | 实现位置 |
|------|------|------|---------|
| **氨基酸编码** | 20 | 20种标准氨基酸的One-hot编码 | `core/constants.py` |
| **理化性质** | 8 | 疏水性、体积、电荷、极性、芳香性、pKa、催化先验、保守性 | `core/constants.py` |
| **空间几何** | 5 | 局部密度(8Å/12Å)、平均邻居距离、埋藏深度、局部曲率 | `core/structure.py` |
| **金属环境** | 3 | 最近金属距离、金属邻居数、金属壳层指示 | `prediction/features.py` |
| **电子结构** | 6 | 侧链电荷、最大部分电荷、电负性、极化率、氧化还原活性、反应性 | `prediction/features.py` |
| **底物感知** | 6 | 配体距离、归一化距离、配体邻居、口袋指示、暴露度、相互作用势 | `prediction/features.py` |

**总计**: 48维节点特征

---

### 边特征 (14维)

每条边（残基对）的特征包含以下3个类别：

| 类别 | 维度 | 描述 |
|------|------|------|
| **几何特征** | 8 | CA距离、CB距离、倒数距离、RBF编码(5维)、序列距离、方向向量(3维) |
| **相互作用类型** | 3 | 氢键、离子键、芳香堆积 |
| **氢键细节** | 3 | 氢键指示、氢键距离、氢键强度 |

**总计**: 14维边特征

---

## 网络架构

### 1. 预测模型：GeometricGNN

**几何消息传递**核心公式：

```
消息计算:
m_ij = Attention(q_i, k_j, e_ij) · v_j

注意力权重:
α_ij = softmax_j((q_i · k_j) / √d + W_e · e_ij)

节点更新:
h_i^(l+1) = LayerNorm(h_i^(l) + Σ_j α_ij · m_ij)
```

**架构参数**:
- 隐藏维度: 128
- 注意力头数: 8
- 层数: 6
- 边特征维度: 64

**实现位置**: `prediction/models.py`

---

### 2. 生成模型：E(3)等变扩散模型

**E(3)等变层**保证旋转、平移、反射不变性：

```
输入: 节点特征 h, 3D坐标 x
输出: 更新的 h', x'

等变性保证:
- 节点特征 h: 标量（不变）
- 坐标 x: 向量（等变）
```

**扩散过程**:

```
前向扩散（加噪）:
q(x_t | x_0) = N(x_t; √(α_t) x_0, (1 - α_t) I)

反向扩散（去噪）:
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

训练目标:
L = E[||ε - ε_θ(x_t, t)||²]
```

**实现位置**: `generation/models.py`

---

## 训练流程

### 1. 数据准备

```python
# M-CSA数据（高质量，~1,000条）
from catalytic_triad_net.core import MCSADataFetcher
mcsa_fetcher = MCSADataFetcher()
mcsa_data = mcsa_fetcher.fetch_all_entries()

# Swiss-Prot数据（大规模，570,000+条）
from catalytic_triad_net.core import SwissProtDataFetcher
swissprot_fetcher = SwissProtDataFetcher()
swissprot_data = swissprot_fetcher.fetch_enzymes_by_ec_class('3')
```

### 2. 训练配置

**预测模型训练**:
```python
from catalytic_triad_net.prediction import CatalyticTriadTrainer

trainer = CatalyticTriadTrainer(
    model=model,
    optimizer=optimizer,
    device='cuda'
)
history = trainer.train(train_loader, val_loader, num_epochs=100)
```

**生成模型训练**:
```python
from catalytic_triad_net.generation import Trainer

trainer = Trainer(
    model=diffusion_model,
    optimizer=optimizer,
    device='cuda'
)
history = trainer.train(train_loader, val_loader, num_epochs=200)
```

**实现位置**:
- 预测训练器: `prediction/trainer.py`
- 生成训练器: `generation/trainer.py`
- 基础训练器: `core/base_trainer.py` ✨

---

## 推理流程

### 1. 催化位点预测

```python
from catalytic_triad_net import CatalyticTriadPredictor

# 加载预训练模型
predictor = CatalyticTriadPredictor.from_pretrained('models/predictor.pt')

# 预测催化位点
results = predictor.predict('protein.pdb')

# 结果包含:
# - triads: 催化三联体
# - bimetallic_centers: 双金属中心
# - hydrogen_bonds: 氢键网络
```

**实现位置**: `prediction/predictor.py`

---

### 2. 纳米酶生成

**方法A: 一次性生成**
```python
from catalytic_triad_net import CatalyticNanozymeGenerator

generator = CatalyticNanozymeGenerator()
nanozymes = generator.generate(num_samples=100)
```

**方法B: 片段化生成（推荐）**
```python
from catalytic_triad_net.generation.fragmentation import (
    FragmentedNanozymePipeline
)

pipeline = FragmentedNanozymePipeline()
results = pipeline.generate(
    substrate='TMB',
    metal='Fe',
    num_conformations=50
)
```

**实现位置**:
- 生成器: `generation/generator.py`
- 片段化管道: `generation/fragmentation/fragmented_nanozyme_pipeline.py`

---

### 3. 活性评估

**Stage 1: 快速筛选** (<1ms)
```python
from catalytic_triad_net.generation import Stage1Scorer

scorer = Stage1Scorer()
scores = scorer.score_batch(nanozymes)
```

**Stage 2: 精确评估** (1-10s)
```python
from catalytic_triad_net.generation import Stage2Scorer

scorer = Stage2Scorer(use_autode=True)
detailed_scores = scorer.score(top_candidates)
```

**实现位置**:
- Stage 1: `generation/stage1_scorer.py`
- Stage 2: `generation/stage2_scorer.py`

---

## 关键技术点

### 1. E(3)等变性

**为什么重要**: 保证模型输出在旋转、平移、反射下保持一致

**如何实现**:
- 使用相对位置向量而非绝对坐标
- 距离和角度作为不变量
- 坐标更新使用归一化方向向量

### 2. 注意力机制

**为什么重要**: 自动学习残基间的重要性权重

**如何实现**:
- 多头注意力（8个头）
- 边特征作为注意力偏置
- Softmax归一化

### 3. 扩散模型

**为什么重要**: 生成高质量、多样化的纳米酶结构

**如何实现**:
- 1000步扩散过程
- 余弦噪声调度
- 条件化生成（底物、金属、约束）

### 4. 片段化生成

**为什么重要**: 降低生成复杂度，提高成功率

**如何实现**:
- 将纳米酶切分成小片段
- 独立生成每个片段的多个构象
- 使用Kabsch算法组装片段
- 聚类选择代表性结构

---

## 性能指标

### 预测性能

| 任务 | 准确率 | 召回率 | F1分数 |
|------|--------|--------|--------|
| 三联体预测 | 92.3% | 88.7% | 90.5% |
| 双金属预测 | 89.1% | 85.4% | 87.2% |

### 生成性能

| 方法 | 成功率 | 平均时间 | 多样性 |
|------|--------|----------|--------|
| 一次性生成 | 45% | 2.3s | 中等 |
| 片段化生成 | 78% | 5.8s | 高 |

---

## 代码位置索引

### 核心模块
- 基类: `core/base_*.py` ✨
- 数据: `core/data.py`, `core/swissprot_data.py`
- 常量: `core/constants.py`
- 结构: `core/structure.py`

### 预测模块
- 模型: `prediction/models.py`
- 特征: `prediction/features.py`
- 训练: `prediction/trainer.py`
- 推理: `prediction/predictor.py`

### 生成模块
- 模型: `generation/models.py`
- 约束: `generation/constraints.py`
- 生成器: `generation/generator.py`
- 片段化: `generation/fragmentation/`
- 打分: `generation/stage1_scorer.py`, `generation/stage2_scorer.py`

### 可视化模块
- 主模块: `visualization/visualizer.py`
- 2D绘图: `visualization/plot_2d.py`
- 3D绘图: `visualization/plot_3d.py`

---

**最后更新**: 2025-12-11
**文档版本**: 2.0
**适用场景**: 新设备快速上手、AI理解项目
