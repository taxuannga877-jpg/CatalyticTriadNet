#!/usr/bin/env python3
"""
Swiss-Prot数据使用示例

展示如何使用Swiss-Prot的570,000+条目扩展训练数据
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from catalytic_triad_net.core.swissprot_data import (
    SwissProtDataFetcher,
    SwissProtDataParser
)

print("="*80)
print("Swiss-Prot数据使用示例")
print("="*80)

print("""
Swiss-Prot vs M-CSA 对比:

| 数据库 | 条目数量 | 催化位点标注 | 数据质量 |
|--------|---------|-------------|---------|
| M-CSA  | ~1,000  | ✅ 精确标注  | ⭐⭐⭐⭐⭐ |
| Swiss-Prot | 570,000+ | ⚠️ 部分标注 | ⭐⭐⭐⭐ |

策略：结合使用
- M-CSA: 高质量训练数据（精确标注）
- Swiss-Prot: 大规模预训练数据（扩展覆盖）
""")

# 初始化数据获取器
fetcher = SwissProtDataFetcher(cache_dir='./data/swissprot_cache')

print("\n" + "="*80)
print("示例1: 获取水解酶数据（EC 3）")
print("="*80)

print("\n⏳ 获取前100个水解酶条目...")
entries = fetcher.fetch_enzymes_by_ec_class(
    ec_class='3',
    limit=100,
    reviewed=True,
    has_structure=False
)

print(f"\n✓ 获取了 {len(entries)} 个条目")

# 统计信息
stats = fetcher.get_statistics(entries)

print(f"\n统计信息:")
print(f"  总条目数: {stats['total_entries']}")
print(f"  有PDB结构: {stats['with_pdb']} ({stats['with_pdb']/stats['total_entries']*100:.1f}%)")
print(f"  有活性位点标注: {stats['with_active_site']} ({stats['with_active_site']/stats['total_entries']*100:.1f}%)")
print(f"  有结合位点标注: {stats['with_binding_site']} ({stats['with_binding_site']/stats['total_entries']*100:.1f}%)")
print(f"  有金属结合标注: {stats['with_metal']} ({stats['with_metal']/stats['total_entries']*100:.1f}%)")

print("\n" + "="*80)
print("示例2: 获取有PDB结构的酶")
print("="*80)

print("\n⏳ 获取有PDB结构的水解酶...")
entries_with_pdb = fetcher.fetch_enzymes_with_structure(
    ec_class='3',
    limit=50
)

print(f"\n✓ 获取了 {len(entries_with_pdb)} 个有结构的条目")

# 解析第一个条目
if entries_with_pdb:
    parser = SwissProtDataParser()
    parsed = parser.parse_entry(entries_with_pdb[0])

    print(f"\n示例条目:")
    print(f"  UniProt ID: {parsed.uniprot_id}")
    print(f"  蛋白质名称: {parsed.protein_name}")
    print(f"  物种: {parsed.organism}")
    print(f"  EC号: {parsed.ec_number}")
    print(f"  PDB IDs: {', '.join(parsed.pdb_ids[:5])}")
    print(f"  序列长度: {len(parsed.sequence)} aa")
    print(f"  活性位点数: {len(parsed.active_sites)}")
    print(f"  结合位点数: {len(parsed.binding_sites)}")
    print(f"  金属结合位点数: {len(parsed.metal_binding)}")

    if parsed.active_sites:
        print(f"\n  活性位点详情:")
        for site in parsed.active_sites[:3]:
            print(f"    位置 {site['position']}: {site['description']}")

print("\n" + "="*80)
print("示例3: 获取特定EC号的酶")
print("="*80)

print("\n⏳ 获取胰蛋白酶（EC 3.4.21.4）...")
trypsin_entries = fetcher.fetch_enzymes_by_ec_number(
    ec_number='3.4.21.4',
    limit=20
)

print(f"\n✓ 获取了 {len(trypsin_entries)} 个胰蛋白酶条目")

if trypsin_entries:
    parser = SwissProtDataParser()
    parsed_list = parser.parse_entries(trypsin_entries)

    print(f"\n胰蛋白酶家族:")
    for i, entry in enumerate(parsed_list[:5], 1):
        print(f"  {i}. {entry.protein_name} ({entry.organism})")
        print(f"     UniProt: {entry.uniprot_id}, PDBs: {len(entry.pdb_ids)}")

print("\n" + "="*80)
print("示例4: 批量获取所有EC类别")
print("="*80)

print("""
推荐策略：分批获取所有EC类别的数据

EC分类：
1. 氧化还原酶 (Oxidoreductases)
2. 转移酶 (Transferases)
3. 水解酶 (Hydrolases)
4. 裂解酶 (Lyases)
5. 异构酶 (Isomerases)
6. 连接酶 (Ligases)
7. 转位酶 (Translocases)
""")

print("\n示例代码（批量获取）:")
print("""
from catalytic_triad_net.core.swissprot_data import SwissProtDataFetcher

fetcher = SwissProtDataFetcher()

# 获取所有EC类别的数据
all_entries = {}
for ec_class in ['1', '2', '3', '4', '5', '6', '7']:
    print(f"获取 EC {ec_class}...")
    entries = fetcher.fetch_enzymes_by_ec_class(
        ec_class=ec_class,
        limit=5000,  # 每个类别5000条
        reviewed=True,
        has_structure=True  # 仅获取有结构的
    )
    all_entries[ec_class] = entries
    print(f"  ✓ EC {ec_class}: {len(entries)} 条目")

# 总计
total = sum(len(entries) for entries in all_entries.values())
print(f"\\n总计: {total} 条目")
""")

print("\n" + "="*80)
print("示例5: 结合M-CSA和Swiss-Prot")
print("="*80)

print("""
推荐的训练策略：

阶段1: 使用M-CSA精确训练
  - 数据：M-CSA ~1,000条目
  - 标注：精确的催化残基和机制
  - 目的：学习精确的催化模式

阶段2: 使用Swiss-Prot扩展训练
  - 数据：Swiss-Prot 有结构的酶 ~50,000条目
  - 标注：活性位点、结合位点（较粗糙）
  - 目的：提高泛化能力，覆盖更多酶家族

阶段3: 迁移学习
  - 在M-CSA上预训练
  - 在Swiss-Prot上微调
  - 最终在M-CSA上精调

代码示例:
""")

print("""
from catalytic_triad_net import (
    MCSADataFetcher,
    SwissProtDataFetcher,
    CatalyticTriadTrainer
)

# 1. 获取M-CSA数据（高质量）
mcsa_fetcher = MCSADataFetcher()
mcsa_entries = mcsa_fetcher.fetch_all_entries()
print(f"M-CSA: {len(mcsa_entries)} 条目")

# 2. 获取Swiss-Prot数据（大规模）
sp_fetcher = SwissProtDataFetcher()
sp_entries = []
for ec_class in ['1', '2', '3', '4', '5', '6', '7']:
    entries = sp_fetcher.fetch_enzymes_with_structure(
        ec_class=ec_class,
        limit=5000
    )
    sp_entries.extend(entries)
print(f"Swiss-Prot: {len(sp_entries)} 条目")

# 3. 创建数据集
from catalytic_triad_net import CatalyticSiteDataset

# M-CSA数据集（精确标注）
mcsa_dataset = CatalyticSiteDataset(
    mcsa_entries,
    pdb_processor,
    feature_encoder
)

# Swiss-Prot数据集（扩展数据）
swissprot_dataset = CatalyticSiteDataset(
    sp_entries,
    pdb_processor,
    feature_encoder,
    use_swissprot_labels=True  # 使用Swiss-Prot的标注
)

# 4. 训练策略
trainer = CatalyticTriadTrainer(model)

# 阶段1: 在M-CSA上预训练
trainer.train(mcsa_dataset, epochs=50)

# 阶段2: 在Swiss-Prot上扩展训练
trainer.train(swissprot_dataset, epochs=20, learning_rate=1e-5)

# 阶段3: 在M-CSA上精调
trainer.train(mcsa_dataset, epochs=10, learning_rate=1e-6)
""")

print("\n" + "="*80)
print("数据质量对比")
print("="*80)

print("""
| 特性 | M-CSA | Swiss-Prot |
|------|-------|-----------|
| 催化残基标注 | ✅ 精确 | ⚠️ 部分 |
| 催化机制 | ✅ 详细 | ❌ 无 |
| 三联体标注 | ✅ 有 | ❌ 无 |
| 活性位点 | ✅ 精确 | ✅ 有 |
| 结合位点 | ✅ 有 | ✅ 有 |
| 金属中心 | ✅ 详细 | ✅ 有 |
| 数据量 | ⭐ 少 | ⭐⭐⭐⭐⭐ 多 |
| 更新频率 | ⭐⭐ 慢 | ⭐⭐⭐⭐ 快 |

建议：
- 核心训练：使用M-CSA（精确）
- 扩展训练：使用Swiss-Prot（覆盖）
- 最终验证：在M-CSA测试集上评估
""")

print("\n" + "="*80)
print("性能提升预期")
print("="*80)

print("""
使用Swiss-Prot扩展后的预期提升：

| 指标 | 仅M-CSA | +Swiss-Prot | 提升 |
|------|---------|------------|------|
| 训练数据量 | 1,000 | 50,000+ | **50倍** |
| 酶家族覆盖 | 有限 | 全面 | **显著** |
| 泛化能力 | 中等 | 优秀 | **+15%** |
| 罕见酶识别 | 较差 | 良好 | **+30%** |
| 新酶预测 | 中等 | 优秀 | **+20%** |

注意事项：
1. Swiss-Prot标注不如M-CSA精确
2. 需要数据清洗和质量控制
3. 建议使用半监督学习策略
4. 在M-CSA上验证最终性能
""")

print("\n" + "="*80)
print("总结")
print("="*80)

print("""
Swiss-Prot集成的优势：

1. **数据量大**
   - 570,000+ 蛋白质序列
   - ~200,000 酶条目
   - ~50,000 有PDB结构的酶

2. **覆盖全面**
   - 所有EC类别
   - 多种物种
   - 罕见酶家族

3. **更新频繁**
   - 每月更新
   - 新酶快速收录
   - 数据质量持续改进

4. **标注丰富**
   - 活性位点
   - 结合位点
   - 金属结合
   - 催化活性描述

推荐使用方式：
- M-CSA: 精确训练（核心）
- Swiss-Prot: 扩展训练（覆盖）
- 结合使用: 最佳性能

下一步：
1. 运行此示例获取数据
2. 清洗和预处理Swiss-Prot数据
3. 创建混合数据集
4. 训练和评估模型
""")

print("\n✓ 示例完成!")
print("\n更多信息:")
print("  - UniProt: https://www.uniprot.org/")
print("  - Swiss-Prot文档: https://www.uniprot.org/help/uniprotkb")
print("  - REST API: https://www.uniprot.org/help/api")
