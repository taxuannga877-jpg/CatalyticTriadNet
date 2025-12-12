#!/usr/bin/env python3
"""
CatalyticTriadNet 命令行接口

提供完整的命令行工具，包括数据探索、模型训练和催化位点预测。
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies() -> dict:
    """
    检查必要的依赖是否已安装。

    Returns:
        包含依赖检查结果的字典
    """
    dependencies = {
        'torch': False,
        'torch_geometric': False,
        'biopython': False,
        'rdkit': False,
        'numpy': False,
        'pandas': False,
    }

    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass

    try:
        import torch_geometric
        dependencies['torch_geometric'] = True
    except ImportError:
        pass

    try:
        import Bio
        dependencies['biopython'] = True
    except ImportError:
        pass

    try:
        import rdkit
        dependencies['rdkit'] = True
    except ImportError:
        pass

    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass

    try:
        import pandas
        dependencies['pandas'] = True
    except ImportError:
        pass

    return dependencies


def validate_file_path(path: str, must_exist: bool = True) -> Optional[Path]:
    """
    验证文件路径。

    Args:
        path: 文件路径字符串
        must_exist: 是否必须存在

    Returns:
        验证后的Path对象，如果验证失败返回None
    """
    try:
        file_path = Path(path).resolve()

        if must_exist and not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return None

        if must_exist and not file_path.is_file():
            logger.error(f"路径不是文件: {file_path}")
            return None

        return file_path
    except Exception as e:
        logger.error(f"路径验证失败 '{path}': {e}")
        return None


def validate_pdb_id(pdb_id: str) -> bool:
    """
    验证PDB ID格式。

    Args:
        pdb_id: PDB ID字符串

    Returns:
        是否为有效的PDB ID
    """
    if not pdb_id:
        return False

    # PDB ID应该是4个字符
    if len(pdb_id) != 4:
        return False

    # 第一个字符应该是数字
    if not pdb_id[0].isdigit():
        return False

    # 其余字符应该是字母或数字
    if not pdb_id[1:].isalnum():
        return False

    return True


def cmd_explore(args):
    """探索M-CSA数据集"""
    logger.info("="*60)
    logger.info("探索 M-CSA 数据集")
    logger.info("="*60)

    try:
        from .core.data import MCSADataFetcher, MCSADataParser
        from .config import get_config

        # 获取配置
        config = get_config()
        if args.config:
            config_path = validate_file_path(args.config, must_exist=True)
            if config_path:
                config.load_from_file(config_path)

        cache_dir = args.cache_dir or str(config.cache_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"使用缓存目录: {cache_dir}")

        fetcher = MCSADataFetcher(cache_dir=cache_dir)
        entries_json = fetcher.fetch_all_entries(force_refresh=args.refresh)

        parser = MCSADataParser()
        entries = parser.parse_entries(entries_json)
        stats = parser.get_statistics(entries)

        print(f"\n总酶条目数: {stats['total_entries']}")
        print(f"总催化残基数: {stats['total_catalytic']}")
        print(f"唯一PDB数: {stats['unique_pdbs']}")

        print("\nEC分类分布:")
        ec_names = {
            '1': '氧化还原酶', '2': '转移酶', '3': '水解酶',
            '4': '裂解酶', '5': '异构酶', '6': '连接酶', '7': '转位酶'
        }
        for ec, cnt in sorted(stats['ec_dist'].items()):
            print(f"  EC {ec} ({ec_names.get(ec, '')}): {cnt}")

        print("\n催化残基类型 (前10):")
        for res, cnt in sorted(stats['res_dist'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {res}: {cnt} ({cnt/stats['total_catalytic']*100:.1f}%)")

        logger.info("="*60)

    except ImportError as e:
        logger.error(f"导入错误: {e}")
        logger.error("请确保已安装所有必要的依赖")
        sys.exit(1)
    except Exception as e:
        logger.error(f"探索数据集时出错: {e}", exc_info=args.verbose)
        sys.exit(1)


def cmd_train(args):
    """训练模型"""
    logger.info("="*60)
    logger.info("训练 CatalyticTriadNet 模型")
    logger.info("="*60)

    try:
        from .core.data import MCSADataFetcher, MCSADataParser
        from .core.structure import PDBProcessor, FeatureEncoder
        from .core.dataset import CatalyticSiteDataset
        from .prediction.models import CatalyticTriadPredictor
        from .prediction.trainer import CatalyticTriadTrainer
        from .config import get_config
        import torch
        from torch.utils.data import DataLoader

        # 获取配置
        config = get_config()
        if args.config:
            config_path = validate_file_path(args.config, must_exist=True)
            if config_path:
                config.load_from_file(config_path)

        # 创建目录
        cache_dir = args.cache_dir or str(config.cache_dir)
        pdb_dir = args.pdb_dir or str(config.pdb_dir)
        model_dir = args.model_dir or './models'

        for d in [cache_dir, pdb_dir, model_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

        logger.info(f"缓存目录: {cache_dir}")
        logger.info(f"PDB目录: {pdb_dir}")
        logger.info(f"模型保存目录: {model_dir}")

        # 获取数据
        logger.info("获取M-CSA数据...")
        fetcher = MCSADataFetcher(cache_dir=cache_dir)
        entries_json = fetcher.fetch_all_entries()

        parser = MCSADataParser()
        entries = parser.parse_entries(entries_json)

        if args.samples > 0:
            entries = entries[:args.samples]
            logger.info(f"使用 {len(entries)} 个样本")
        else:
            logger.info(f"使用全部 {len(entries)} 个样本")

        # 创建数据集
        logger.info("创建数据集...")
        pdb_proc = PDBProcessor(pdb_dir=pdb_dir)
        feat_enc = FeatureEncoder()
        dataset = CatalyticSiteDataset(
            entries, pdb_proc, feat_enc,
            max_residues=args.max_residues,
            edge_cutoff=args.edge_cutoff
        )

        if len(dataset) < 10:
            logger.error("错误: 数据集太小，无法训练")
            logger.error(f"当前数据集大小: {len(dataset)}")
            sys.exit(1)

        logger.info(f"数据集大小: {len(dataset)}")

        # 划分数据
        n = len(dataset)
        train_n = int(0.8 * n)
        val_n = n - train_n

        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_n, val_n])
        logger.info(f"训练集: {train_n}, 验证集: {val_n}")

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, collate_fn=dataset.collate_fn,
            num_workers=0  # 避免多进程问题
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            shuffle=False, collate_fn=dataset.collate_fn,
            num_workers=0
        )

        # 创建模型和训练器
        logger.info("创建模型...")
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
        logger.info("开始训练...")
        best_f1 = trainer.train(
            train_loader, val_loader,
            epochs=args.epochs,
            save_dir=model_dir
        )

        logger.info(f"\n训练完成! 最佳 F1: {best_f1:.4f}")
        logger.info(f"模型已保存到: {model_dir}")

    except ImportError as e:
        logger.error(f"导入错误: {e}")
        logger.error("请确保已安装所有必要的依赖")
        sys.exit(1)
    except Exception as e:
        logger.error(f"训练时出错: {e}", exc_info=args.verbose)
        sys.exit(1)


def cmd_predict(args):
    """预测催化残基"""
    logger.info("="*60)
    logger.info("预测催化位点")
    logger.info("="*60)

    try:
        from .prediction.predictor import EnhancedCatalyticSiteInference
        from .config import get_config
        import json

        # 获取配置
        config = get_config()
        if args.config:
            config_path = validate_file_path(args.config, must_exist=True)
            if config_path:
                config.load_from_file(config_path)

        # 验证模型路径
        model_path = validate_file_path(args.model, must_exist=True)
        if not model_path:
            logger.error(f"模型文件不存在: {args.model}")
            sys.exit(1)

        logger.info(f"使用模型: {model_path}")

        # 验证PDB输入
        pdb_input = args.pdb
        is_pdb_id = False

        # 检查是否是PDB ID
        if len(pdb_input) == 4 and validate_pdb_id(pdb_input):
            is_pdb_id = True
            logger.info(f"输入为PDB ID: {pdb_input}")
        else:
            # 检查是否是文件路径
            pdb_path = validate_file_path(pdb_input, must_exist=True)
            if not pdb_path:
                logger.error(f"无效的PDB输入: {pdb_input}")
                logger.error("请提供有效的PDB ID (4个字符) 或PDB文件路径")
                sys.exit(1)
            logger.info(f"输入为PDB文件: {pdb_path}")

        # 创建预测器
        logger.info("加载预测器...")
        predictor = EnhancedCatalyticSiteInference(
            model_path=str(model_path),
            device=args.device
        )

        # 执行预测
        logger.info("执行预测...")
        results = predictor.predict(
            pdb_input,
            site_threshold=args.threshold
        )

        # 打印结果
        predictor.print_results(results, top_k=args.top)

        # 导出结果
        if args.output:
            output_prefix = Path(args.output)
            output_prefix.parent.mkdir(parents=True, exist_ok=True)

            # JSON
            json_path = f"{args.output}.json"
            with open(json_path, 'w') as f:
                # 转换numpy数组为列表以便JSON序列化
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, list):
                        json_results[key] = value
                    else:
                        json_results[key] = value
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON已保存: {json_path}")

            # CSV
            if results.get('catalytic_residues'):
                try:
                    import pandas as pd
                    df = pd.DataFrame(results['catalytic_residues'])
                    csv_path = f"{args.output}.csv"
                    df.to_csv(csv_path, index=False)
                    logger.info(f"CSV已保存: {csv_path}")
                except ImportError:
                    logger.warning("pandas未安装，跳过CSV导出")

            # PyMOL脚本
            pml_path = f"{args.output}.pml"
            predictor.export_pymol(results, pml_path, args.threshold)
            logger.info(f"PyMOL脚本已保存: {pml_path}")

            # ProteinMPNN格式
            if args.export_mpnn:
                mpnn_path = f"{args.output}_mpnn.json"
                predictor.export_for_proteinmpnn(results, mpnn_path)
                logger.info(f"ProteinMPNN格式已保存: {mpnn_path}")

            # RFdiffusion格式
            if args.export_rfd:
                rfd_path = f"{args.output}_rfd.json"
                predictor.export_for_rfdiffusion(results, rfd_path)
                logger.info(f"RFdiffusion格式已保存: {rfd_path}")

        logger.info("预测完成!")

    except ImportError as e:
        logger.error(f"导入错误: {e}")
        logger.error("请确保已安装所有必要的依赖")
        sys.exit(1)
    except Exception as e:
        logger.error(f"预测时出错: {e}", exc_info=args.verbose)
        sys.exit(1)


def cmd_check_deps(args):
    """检查依赖"""
    logger.info("="*60)
    logger.info("检查依赖")
    logger.info("="*60)

    deps = check_dependencies()

    print("\n必需依赖:")
    required = ['torch', 'torch_geometric', 'biopython', 'numpy']
    all_required_ok = True

    for dep in required:
        status = "✓" if deps[dep] else "✗"
        print(f"  {status} {dep}")
        if not deps[dep]:
            all_required_ok = False

    print("\n可选依赖:")
    optional = ['rdkit', 'pandas']
    for dep in optional:
        status = "✓" if deps[dep] else "✗"
        print(f"  {status} {dep}")

    if not all_required_ok:
        print("\n警告: 缺少必需依赖，某些功能可能无法使用")
        print("\n安装说明:")
        print("  pip install torch torch-geometric biopython numpy")
        sys.exit(1)
    else:
        print("\n所有必需依赖已安装!")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="CatalyticTriadNet v2.0 - 催化位点识别与纳米酶设计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查依赖
  catalytic-triad-net check-deps

  # 探索M-CSA数据集
  catalytic-triad-net explore

  # 训练模型
  catalytic-triad-net train --epochs 50 --samples 200

  # 预测催化残基 (使用PDB ID)
  catalytic-triad-net predict --pdb 1acb --model models/best_model.pt

  # 预测催化残基 (使用PDB文件)
  catalytic-triad-net predict --pdb /path/to/protein.pdb --model models/best_model.pt

  # 导出多种格式
  catalytic-triad-net predict --pdb 1acb --output results/1acb --export-mpnn --export-rfd

  # 使用自定义配置文件
  catalytic-triad-net train --config config.yaml
        """
    )

    # 全局参数
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--config', type=str, help='配置文件路径 (YAML)')

    subparsers = parser.add_subparsers(dest='command', help='命令')

    # check-deps
    p_check = subparsers.add_parser('check-deps', help='检查依赖')

    # explore
    p_explore = subparsers.add_parser('explore', help='探索M-CSA数据集')
    p_explore.add_argument('--cache-dir', type=str, help='缓存目录')
    p_explore.add_argument('--refresh', action='store_true', help='刷新缓存')

    # train
    p_train = subparsers.add_parser('train', help='训练模型')
    p_train.add_argument('--epochs', type=int, default=50, help='训练轮数 (默认: 50)')
    p_train.add_argument('--samples', type=int, default=200, help='样本数，-1表示全部 (默认: 200)')
    p_train.add_argument('--batch-size', type=int, default=4, help='批次大小 (默认: 4)')
    p_train.add_argument('--lr', type=float, default=1e-4, help='学习率 (默认: 1e-4)')
    p_train.add_argument('--weight-decay', type=float, default=1e-5, help='权重衰减 (默认: 1e-5)')
    p_train.add_argument('--hidden-dim', type=int, default=256, help='隐藏层维度 (默认: 256)')
    p_train.add_argument('--num-layers', type=int, default=6, help='GNN层数 (默认: 6)')
    p_train.add_argument('--dropout', type=float, default=0.2, help='Dropout (默认: 0.2)')
    p_train.add_argument('--patience', type=int, default=15, help='早停patience (默认: 15)')
    p_train.add_argument('--max-residues', type=int, default=1000, help='最大残基数 (默认: 1000)')
    p_train.add_argument('--edge-cutoff', type=float, default=10.0, help='边截断距离 (默认: 10.0)')
    p_train.add_argument('--cache-dir', type=str, help='缓存目录')
    p_train.add_argument('--pdb-dir', type=str, help='PDB目录')
    p_train.add_argument('--model-dir', type=str, default='./models', help='模型保存目录 (默认: ./models)')

    # predict
    p_pred = subparsers.add_parser('predict', help='预测催化残基')
    p_pred.add_argument('--pdb', required=True, help='PDB文件路径或PDB ID (4个字符)')
    p_pred.add_argument('--model', default='models/best_model.pt', help='模型路径 (默认: models/best_model.pt)')
    p_pred.add_argument('--threshold', type=float, default=0.5, help='阈值 (默认: 0.5)')
    p_pred.add_argument('--top', type=int, default=15, help='显示前N个 (默认: 15)')
    p_pred.add_argument('--output', type=str, help='输出文件前缀')
    p_pred.add_argument('--device', default='cpu', help='设备 (cpu/cuda, 默认: cpu)')
    p_pred.add_argument('--export-mpnn', action='store_true', help='导出ProteinMPNN格式')
    p_pred.add_argument('--export-rfd', action='store_true', help='导出RFdiffusion格式')

    args = parser.parse_args()

    # 设置日志级别
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command is None:
        parser.print_help()
        return

    commands = {
        'check-deps': cmd_check_deps,
        'explore': cmd_explore,
        'train': cmd_train,
        'predict': cmd_predict,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
