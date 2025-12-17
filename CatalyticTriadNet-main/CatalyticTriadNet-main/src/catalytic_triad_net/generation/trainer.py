#!/usr/bin/env python3
"""
扩散模型训练器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Dict
import json

from .models import CatalyticDiffusionModel, ConstraintLoss
from .dataset import NanozymeDataset
from .constraints import NUM_ATOM_TYPES, CatalyticConstraints
from .generator import CatalyticNanozymeGenerator

logger = logging.getLogger(__name__)

class Trainer:
    """模型训练器"""
    def __init__(self, model: CatalyticDiffusionModel, 
                 train_dataset: NanozymeDataset,
                 config: Dict = None):
        self.model = model
        self.dataset = train_dataset
        self.config = config or {}
        
        self.lr = self.config.get('lr', 1e-4)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 100)
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
    
    def train(self):
        """训练循环"""
        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, 
            shuffle=True, collate_fn=self._collate_fn
        )
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch in dataloader:
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(
                    batch['atom_types'],
                    batch['coords'],
                    batch['edge_index'],
                    batch['condition']
                )
                
                # 计算损失
                loss = F.mse_loss(outputs['pred_noise'], outputs['target_noise'])
                loss += F.cross_entropy(
                    outputs['pred_atom_logits'].view(-1, NUM_ATOM_TYPES),
                    outputs['target_atom_types'].view(-1)
                )
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            self.scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def _collate_fn(self, batch):
        """批次整理函数"""
        # 简化版: 假设同一batch内原子数相同或进行padding
        atom_types = torch.stack([b['atom_types'] for b in batch])
        coords = torch.stack([b['coords'] for b in batch])

        # 构建批次边索引（为每个样本创建独立的边）
        batch_size = len(batch)
        n_atoms = atom_types.shape[1]
        edge_index = self._build_batch_edges(n_atoms, batch_size, atom_types.device)

        # 空条件 (训练时无监督) - 扩展到批次大小
        condition = {
            'anchor_features': torch.zeros(batch_size, 16, device=atom_types.device),
            'distance_constraints': torch.zeros(0, 4, device=atom_types.device),
            'coordination_constraints': torch.zeros(0, 3, device=atom_types.device)
        }

        return {
            'atom_types': atom_types,
            'coords': coords,
            'edge_index': edge_index,
            'condition': condition
        }

    def _build_edges(self, n, device):
        """构建单个图的全连接边"""
        rows, cols = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        return torch.tensor([rows, cols], dtype=torch.long, device=device)

    def _build_batch_edges(self, n_atoms, batch_size, device):
        """构建批次全连接边（为每个样本创建独立的边索引）"""
        rows, cols = [], []
        for batch_idx in range(batch_size):
            offset = batch_idx * n_atoms
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j:
                        rows.append(offset + i)
                        cols.append(offset + j)
        return torch.tensor([rows, cols], dtype=torch.long, device=device)


# =============================================================================
# 8. 命令行接口
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CatalyticDiff: 催化位点条件化纳米酶分子生成'
    )
    subparsers = parser.add_subparsers(dest='command')
    
    # 生成命令
    gen_parser = subparsers.add_parser('generate', help='生成纳米酶结构')
    gen_parser.add_argument('--constraints', required=True, 
                           help='催化约束JSON文件 (CatalyticTriadNet输出)')
    gen_parser.add_argument('--model', default=None, help='模型检查点路径')
    gen_parser.add_argument('--n_samples', type=int, default=10, help='生成样本数')
    gen_parser.add_argument('--n_atoms', type=int, default=None, help='原子数')
    gen_parser.add_argument('--output', default='./generated', help='输出目录')
    gen_parser.add_argument('--format', choices=['xyz', 'mol', 'json'], 
                           default='xyz', help='输出格式')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data_dir', required=True, help='训练数据目录')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--save_path', default='./models/catalytic_diff.pt')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        # 加载约束
        constraints = CatalyticConstraints.from_catalytic_triad_output(args.constraints)
        logger.info(f"Loaded constraints: {len(constraints.anchor_atoms)} anchor atoms, "
                   f"{len(constraints.distance_constraints)} distance constraints")
        
        # 初始化生成器
        generator = CatalyticNanozymeGenerator(model_path=args.model)
        
        # 生成
        results = generator.generate(
            constraints,
            n_samples=args.n_samples,
            n_atoms=args.n_atoms
        )
        
        # 保存
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results):
            if args.format == 'xyz':
                generator.to_xyz(result, output_dir / f'nanozyme_{i:03d}.xyz')
            elif args.format == 'mol':
                generator.to_mol(result, output_dir / f'nanozyme_{i:03d}.mol')
            else:
                with open(output_dir / f'nanozyme_{i:03d}.json', 'w') as f:
                    json.dump(result, f, indent=2)
        
        logger.info(f"Generated {len(results)} structures in {output_dir}")
        
        # 打印约束满足统计
        for i, r in enumerate(results):
            sat = r['constraint_satisfaction']
            n_sat = sum(1 for d in sat.get('distance', []) if d['satisfied'])
            n_total = len(sat.get('distance', []))
            logger.info(f"  Sample {i}: {n_sat}/{n_total} distance constraints satisfied")
    
    elif args.command == 'train':
        dataset = NanozymeDataset(args.data_dir)
        logger.info(f"Loaded {len(dataset)} training samples")
        
        model = CatalyticDiffusionModel()
        trainer = Trainer(model, dataset, {'epochs': args.epochs, 'lr': args.lr})
        trainer.train()
        
        # 保存模型
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        logger.info(f"Model saved to {args.save_path}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
