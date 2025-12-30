# CatalyticTriadNet

基于EC号的纳米酶数据库与催化位点提取系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目概述

CatalyticTriadNet 是一个用于纳米酶设计的计算工具，实现从天然酶到纳米酶的催化位点迁移。

**核心功能：**
- 根据 EC 号从 UniProt 获取酶数据
- 使用 EasIFA 模型预测活性位点
- 提取催化 Motif 用于下游 STOL 组装

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    CatalyticTriadNet                        │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: Database Layer                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ UniProt API │  │  M-CSA API  │  │ AlphaFold   │         │
│  │  (EC查询)   │  │ (催化位点)  │  │  (PDB下载)  │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              NanozymeDatabase (SQLite)                │ │
│  │   EC号 → 纳米酶类型映射 (POD/CAT/SOD/GSH/OXD/LAC)    │ │
│  └───────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: Dual-Track Processing (两手抓策略)               │
│                                                             │
│  ┌─────────────────┐      ┌─────────────────┐              │
│  │   有标注数据    │      │   无标注数据    │              │
│  │ (UniProt/M-CSA) │      │  (需要预测)     │              │
│  └────────┬────────┘      └────────┬────────┘              │
│           │                        │                        │
│           ▼                        ▼                        │
│  ┌─────────────────┐      ┌─────────────────┐              │
│  │   直接使用      │      │  EasIFA 预测    │              │
│  │   已知标注      │      │  活性位点       │              │
│  └────────┬────────┘      └────────┬────────┘              │
│           └────────────┬───────────┘                        │
│                        ▼                                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              MotifExtractor                           │ │
│  │   提取: AnchorAtoms + Geometry + ChemistryTag        │ │
│  └───────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Output: CatalyticMotif JSON → STOL Assembly               │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
CatalyticTriadNet/
├── nanozyme_mining/          # 核心包
│   ├── __init__.py
│   ├── database/             # Stage 1: 数据库层
│   │   ├── nanozyme_db.py    # SQLite 数据库
│   │   ├── uniprot_fetcher.py # UniProt API
│   │   └── mcsa_fetcher.py   # M-CSA API
│   ├── prediction/           # EasIFA 预测模块
│   │   └── easifa_predictor.py
│   ├── extraction/           # Stage 2: Motif 提取
│   │   ├── extractor.py
│   │   └── motif.py
│   ├── core/                 # 双轨处理器
│   │   └── __init__.py
│   └── utils/                # 工具函数
│       ├── constants.py
│       └── ec_mappings.py
├── models/                   # 模型检查点
│   └── easifa/checkpoints/
├── examples/                 # 示例代码
├── tests/                    # 测试
├── requirements.txt
├── setup.py
└── README.md
```

## 安装

### 基础安装

```bash
git clone https://github.com/taxuannga877-jpg/CatalyticTriadNet.git
cd CatalyticTriadNet
pip install -e .
```

### 完整安装（包含 EasIFA 依赖）

```bash
pip install -e ".[full]"
```

### 模型文件

EasIFA 模型检查点需要单独下载并放置在 `models/easifa/checkpoints/` 目录下：

```
models/easifa/checkpoints/
├── enzyme_site_type_predition_model/
│   └── train_in_uniprot_ecreact_.../global_step_284000/model.pth
└── reaction_attn_net/
    └── model-ReactionMGMTurnNet_.../model.pth
```

## 快速开始

### 完整流程示例

```python
from nanozyme_mining import (
    UniProtFetcher,
    DualTrackProcessor,
    MotifExtractor,
)
from nanozyme_mining.utils import NanozymeType

# Step 1: 下载 PDB 文件并分类
fetcher = UniProtFetcher(cache_dir="./data/cache")
annotated, unannotated = fetcher.fetch_and_classify(
    ec_number="1.11.1.7",  # Peroxidase
    nanozyme_type=NanozymeType.POD
)

# Step 2: 对未标注数据进行 EasIFA 预测
processor = DualTrackProcessor(
    output_dir="./data/processed",
    device="cpu"
)
predicted = processor.predict_unannotated_batch(unannotated)

# Step 3: 提取催化 Motif
extractor = MotifExtractor(output_dir="./data/motifs")
# ... 提取 motif
```

## 支持的纳米酶类型

| 类型 | EC 号 | 描述 |
|------|-------|------|
| POD | 1.11.1.7 | 过氧化物酶样 |
| CAT | 1.11.1.6 | 过氧化氢酶样 |
| SOD | 1.15.1.1 | 超氧化物歧化酶样 |
| GSH | 1.11.1.9 | 谷胱甘肽过氧化物酶样 |
| OXD | 1.4.3.4 | 氧化酶样 |
| LAC | 1.10.3.2 | 漆酶样 |
| GOX | 1.1.3.4 | 葡萄糖氧化酶样 |

## 输出格式

### CatalyticMotif JSON

```json
{
  "motif_id": "P00433_1.11.1.7_POD",
  "source_uniprot_id": "P00433",
  "source_ec_number": "1.11.1.7",
  "nanozyme_type": "POD",
  "anchor_atoms": [
    {
      "atom_name": "NE2",
      "residue_name": "HIS",
      "residue_number": 42,
      "coordinates": [10.5, 20.3, 15.2]
    }
  ],
  "geometry_constraints": [
    {
      "constraint_type": "distance",
      "atom_indices": [0, 1],
      "value": 3.5,
      "unit": "angstrom"
    }
  ]
}
```

## 参考项目

- [ChemEnzyRetroPlanner](https://github.com/example/ChemEnzyRetroPlanner) - EasIFA 模型来源
- [STOL](https://github.com/example/STOL) - 下游组装工具

## License

MIT License - 详见 [LICENSE](LICENSE) 文件
