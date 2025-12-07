#!/usr/bin/env python3
"""
CatalyticTriadNet 命令行接口
"""

import argparse
import sys
from pathlib import Path

from .core.data import MCSADataFetcher, MCSADataParser
from .core.structure import PDBProcessor, FeatureEncoder
from .core.dataset import CatalyticSiteDataset
from .prediction.models import CatalyticTriadPredictor
from .prediction.trainer import CatalyticTriadTrainer
from .prediction.predictor import EnhancedCatalyticSiteInference


def cmd_explore(args):
    """探索M-CSA数据集"""
    print("\n" + "="*60)
    print("探索 M-CSA 数据集")
    print("="*60)
    
    fetcher = MCSADataFetcher(cache_dir=args.cache_dir)
    entries_json = fetcher.fetch_all_entries(force_refresh=args.refresh)
    
    parser = MCSADataParser()
    entries = parser.parse_entries(entries_json)
    stats = parser.get_statistics(entries)
    
    print(f"\n总酶条目数: {stats['total_entries']}")
    print(f"总催化残基数: {stats['total_catalytic']}")
    print(f"唯一PDB数: {stats['unique_pdbs']}")
    
    print("\nEC分类分布:")
    ec_names = {'1': '氧化还原酶', '2': '转移酶', '3': '水解酶',
               '4': '裂解酶', '5': '异构酶', '6': '连接酶', '7': '转位酶'}
    for ec, cnt in sorted(stats['ec_dist'].items()):
        print(f"  EC {ec} ({ec_names.get(ec, '')}): {cnt}")
    
    print("\n催化残基类型 (前10):")
    for res, cnt in sorted(stats['res_dist'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {res}: {cnt} ({cnt/stats['total_catalytic']*100:.1f}%)")
    print("="*60)


def cmd_train(args):
    """训练模型"""
    print("\n" + "="*60)
    print("训练 CatalyticTriadNet 模型")
    print("="*60)
    
    # 创建目录
    for d in [args.cache_dir, args.pdb_dir, args.model_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # 获取数据
    fetcher = MCSADataFetcher(cache_dir=args.cache_dir)
    entries_json = fetcher.fetch_all_entries()
    
    parser = MCSADataParser()
    entries = parser.parse_entries(entries_json)
    
    if args.samples > 0:
        entries = entries[:args.samples]
        print(f"使用 {len(entries)} 个样本")
    
    # 创建数据集
    pdb_proc = PDBProcessor(pdb_dir=args.pdb_dir)
    feat_enc = FeatureEncoder()
    dataset = CatalyticSiteDataset(entries, pdb_proc, feat_enc,
                                   max_residues=args.max_residues,
                                   edge_cutoff=args.edge_cutoff)
    
    if len(dataset) < 10:
        print("错误: 数据集太小，无法训练")
        return
    
    # 划分数据
    import torch
    n = len(dataset)
    train_n = int(0.8 * n)
    val_n = n - train_n
    
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_n, val_n])
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                             shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                           shuffle=False, collate_fn=dataset.collate_fn)
    
    # 创建模型和训练器
    model = CatalyticTriadPredictor(
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_layers,
        dropout=args.dropout
    )
    
    trainer = CatalyticTriadTrainer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience
    )
    
    # 训练
    best_f1 = trainer.train(train_loader, val_loader,
                           epochs=args.epochs,
                           save_dir=args.model_dir)
    
    print(f"\n✓ 训练完成! 最佳 F1: {best_f1:.4f}")


def cmd_predict(args):
    """预测催化残基"""
    print("\n" + "="*60)
    print("预测催化位点")
    print("="*60)
    
    if not Path(args.model).exists():
        print(f"错误: 模型不存在: {args.model}")
        return
    
    predictor = EnhancedCatalyticSiteInference(
        model_path=args.model,
        device=args.device
    )
    
    results = predictor.predict(
        args.pdb,
        site_threshold=args.threshold
    )
    
    predictor.print_results(results, top_k=args.top)
    
    # 导出
    if args.output:
        import json
        import pandas as pd
        
        # JSON
        json_path = f"{args.output}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ JSON: {json_path}")
        
        # CSV
        if results['catalytic_residues']:
            df = pd.DataFrame(results['catalytic_residues'])
            csv_path = f"{args.output}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ CSV: {csv_path}")
        
        # PyMOL
        pml_path = f"{args.output}.pml"
        predictor.export_pymol(results, pml_path, args.threshold)
        
        # ProteinMPNN
        if args.export_mpnn:
            mpnn_path = f"{args.output}_mpnn.json"
            predictor.export_for_proteinmpnn(results, mpnn_path)
        
        # RFdiffusion
        if args.export_rfd:
            rfd_path = f"{args.output}_rfd.json"
            predictor.export_for_rfdiffusion(results, rfd_path)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="CatalyticTriadNet v2.0 - 催化位点识别与纳米酶设计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 探索M-CSA数据集
  catalytic-triad-net explore
  
  # 训练模型
  catalytic-triad-net train --epochs 50 --samples 200
  
  # 预测催化残基
  catalytic-triad-net predict --pdb 1acb --model models/best_model.pt
  
  # 导出多种格式
  catalytic-triad-net predict --pdb 1acb --output results/1acb --export-mpnn --export-rfd
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # explore
    p_explore = subparsers.add_parser('explore', help='探索M-CSA数据集')
    p_explore.add_argument('--cache-dir', default='./data/mcsa_cache', help='缓存目录')
    p_explore.add_argument('--refresh', action='store_true', help='刷新缓存')
    
    # train
    p_train = subparsers.add_parser('train', help='训练模型')
    p_train.add_argument('--epochs', type=int, default=50, help='训练轮数')
    p_train.add_argument('--samples', type=int, default=200, help='样本数 (-1=全部)')
    p_train.add_argument('--batch-size', type=int, default=4, help='批次大小')
    p_train.add_argument('--lr', type=float, default=1e-4, help='学习率')
    p_train.add_argument('--weight-decay', type=float, default=1e-5, help='权重衰减')
    p_train.add_argument('--hidden-dim', type=int, default=256, help='隐藏层维度')
    p_train.add_argument('--num-layers', type=int, default=6, help='GNN层数')
    p_train.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    p_train.add_argument('--patience', type=int, default=15, help='早停patience')
    p_train.add_argument('--max-residues', type=int, default=1000, help='最大残基数')
    p_train.add_argument('--edge-cutoff', type=float, default=10.0, help='边截断距离')
    p_train.add_argument('--cache-dir', default='./data/mcsa_cache', help='缓存目录')
    p_train.add_argument('--pdb-dir', default='./data/pdb_structures', help='PDB目录')
    p_train.add_argument('--model-dir', default='./models', help='模型保存目录')
    
    # predict
    p_pred = subparsers.add_parser('predict', help='预测催化残基')
    p_pred.add_argument('--pdb', required=True, help='PDB文件或ID')
    p_pred.add_argument('--model', default='models/best_model.pt', help='模型路径')
    p_pred.add_argument('--threshold', type=float, default=0.5, help='阈值')
    p_pred.add_argument('--top', type=int, default=15, help='显示前N个')
    p_pred.add_argument('--output', help='输出文件前缀')
    p_pred.add_argument('--device', default='cpu', help='设备 (cpu/cuda)')
    p_pred.add_argument('--export-mpnn', action='store_true', help='导出ProteinMPNN格式')
    p_pred.add_argument('--export-rfd', action='store_true', help='导出RFdiffusion格式')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    commands = {
        'explore': cmd_explore,
        'train': cmd_train,
        'predict': cmd_predict,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
