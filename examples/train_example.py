#!/usr/bin/env python3
"""
训练示例 - 使用M-CSA数据集
"""

import torch
from torch.utils.data import DataLoader

from catalytic_triad_net import (
    MCSADataFetcher,
    MCSADataParser,
    PDBProcessor,
    FeatureEncoder,
    CatalyticSiteDataset,
    CatalyticTriadPredictor,
    CatalyticTriadTrainer
)

# 1. 获取M-CSA数据
print("获取M-CSA数据...")
fetcher = MCSADataFetcher(cache_dir='./data/mcsa_cache')
entries_json = fetcher.fetch_all_entries()

parser = MCSADataParser()
entries = parser.parse_entries(entries_json)
print(f"✓ 获取 {len(entries)} 个酶条目")

# 2. 创建数据集
print("\n创建数据集...")
pdb_proc = PDBProcessor(pdb_dir='./data/pdb_structures')
feat_enc = FeatureEncoder()

dataset = CatalyticSiteDataset(
    entries[:200],  # 使用前200个样本
    pdb_proc,
    feat_enc,
    max_residues=1000,
    edge_cutoff=10.0
)
print(f"✓ 处理 {len(dataset)} 个样本")

# 3. 划分数据
n = len(dataset)
train_n = int(0.8 * n)
val_n = n - train_n

train_ds, val_ds = torch.utils.data.random_split(dataset, [train_n, val_n])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, 
                          collate_fn=dataset.collate_fn)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                        collate_fn=dataset.collate_fn)

# 4. 创建模型
print("\n创建模型...")
model = CatalyticTriadPredictor(
    node_dim=33,
    edge_dim=8,
    hidden_dim=256,
    num_gnn_layers=6,
    num_heads=8,
    dropout=0.2
)

# 5. 训练
print("\n开始训练...")
trainer = CatalyticTriadTrainer(
    model,
    learning_rate=1e-4,
    weight_decay=1e-5,
    patience=15
)

best_f1 = trainer.train(
    train_loader,
    val_loader,
    epochs=50,
    save_dir='./models'
)

print(f"\n✓ 训练完成! 最佳 F1: {best_f1:.4f}")
