#!/usr/bin/env python3
"""
批量催化中心筛选器
用于从多个天然酶PDB中筛选高分催化中心
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .predictor import EnhancedCatalyticSiteInference
from ..config import get_config

logger = logging.getLogger(__name__)


class BatchCatalyticScreener:
    """
    批量催化中心筛选器

    功能:
    1. 批量处理多个PDB文件
    2. 按分数排序催化中心
    3. 过滤和聚类相似催化中心
    4. 导出用于纳米酶设计的模板

    使用示例:
        screener = BatchCatalyticScreener(model_path='models/best_model.pt')

        # 方式1: 从PDB ID列表筛选
        results = screener.screen_pdb_list(
            pdb_ids=['1acb', '4cha', '1hiv'],
            site_threshold=0.7,
            top_k=5
        )

        # 方式2: 从文件夹筛选
        results = screener.screen_directory(
            pdb_dir='data/pdbs/',
            site_threshold=0.7
        )

        # 导出结果
        screener.export_summary(results, 'screening_results.csv')
        screener.export_for_nanozyme_design(results, 'nanozyme_templates/')
    """

    def __init__(self, model_path: str, device: Optional[str] = None,
                 n_workers: Optional[int] = None, max_retries: Optional[int] = None,
                 timeout: Optional[int] = None, config: Optional[Dict] = None):
        """
        初始化批量筛选器。

        Args:
            model_path: 训练好的模型路径
            device: 'cuda' 或 'cpu'（可选）
            n_workers: 并行处理的线程数（如果为 None，从 config 读取）
            max_retries: 最大重试次数（如果为 None，从 config 读取）
            timeout: 超时时间（秒）（如果为 None，从 config 读取）
            config: 配置字典（可选）
        """
        # 从配置读取参数
        if config is None:
            global_config = get_config()
            screening_config = global_config.get('screening', {})
        else:
            screening_config = config.get('screening', {})

        self.n_workers = n_workers if n_workers is not None else screening_config.get('num_workers', 4)
        self.max_retries = max_retries if max_retries is not None else screening_config.get('max_retries', 3)
        self.timeout = timeout if timeout is not None else screening_config.get('timeout', 300)

        self.predictor = EnhancedCatalyticSiteInference(
            model_path=model_path,
            device=device,
            config=config
        )

        # 跟踪失败的任务
        self.failed_tasks: List[Dict[str, Any]] = []

        logger.info(f"BatchCatalyticScreener initialized: workers={self.n_workers}, "
                   f"max_retries={self.max_retries}, timeout={self.timeout}s")

    def screen_pdb_list(self, pdb_ids: List[str],
                       site_threshold: float = 0.7,
                       top_k: int = 10,
                       ec_filter: Optional[int] = None) -> List[Dict]:
        """
        从PDB ID列表筛选催化中心

        Args:
            pdb_ids: PDB ID列表
            site_threshold: 催化位点概率阈值 (0-1)
            top_k: 每个PDB保留前k个高分残基
            ec_filter: 只保留特定EC类别 (1-7), None表示不过滤

        Returns:
            筛选结果列表，按分数降序排列
        """
        logger.info(f"开始筛选 {len(pdb_ids)} 个PDB...")

        all_results = []

        # 并行处理（带异常捕获和重试）
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._screen_single_pdb_with_retry, pdb_id,
                              site_threshold, ec_filter): pdb_id
                for pdb_id in pdb_ids
            }

            for future in tqdm(as_completed(futures), total=len(pdb_ids),
                             desc="筛选进度"):
                pdb_id = futures[future]
                try:
                    result = future.result(timeout=self.timeout)
                    if result:
                        all_results.append(result)
                except TimeoutError:
                    error_msg = f"Timeout after {self.timeout}s"
                    logger.error(f"处理 {pdb_id} 超时: {error_msg}")
                    self.failed_tasks.append({
                        'pdb_id': pdb_id,
                        'error': error_msg,
                        'type': 'timeout'
                    })
                except Exception as e:
                    logger.error(f"处理 {pdb_id} 失败: {e}")
                    self.failed_tasks.append({
                        'pdb_id': pdb_id,
                        'error': str(e),
                        'type': 'exception'
                    })

        # 展开所有催化残基
        flattened = []
        for result in all_results:
            for res in result['catalytic_residues'][:top_k]:
                flattened.append({
                    'pdb_id': result['pdb_id'],
                    'ec1_prediction': result['ec1_prediction'],
                    'ec1_confidence': result['ec1_confidence'],
                    'chain': res['chain'],
                    'resseq': res['resseq'],
                    'resname': res['resname'],
                    'site_prob': res['site_prob'],
                    'triads': result.get('triads', []),
                    'metal_centers': result.get('metal_centers', []),
                    'bimetallic_centers': result.get('bimetallic_centers', [])
                })

        # 按分数排序
        flattened.sort(key=lambda x: x['site_prob'], reverse=True)

        logger.info(f"筛选完成! 共找到 {len(flattened)} 个高分催化残基")
        return flattened

    def screen_directory(self, pdb_dir: str,
                        site_threshold: float = 0.7,
                        pattern: str = "*.pdb",
                        top_k: int = 10) -> List[Dict]:
        """
        从文件夹批量筛选PDB文件

        Args:
            pdb_dir: PDB文件夹路径
            site_threshold: 催化位点阈值
            pattern: 文件匹配模式
            top_k: 每个PDB保留前k个

        Returns:
            筛选结果列表
        """
        pdb_dir = Path(pdb_dir)
        pdb_files = list(pdb_dir.glob(pattern))

        logger.info(f"在 {pdb_dir} 中找到 {len(pdb_files)} 个PDB文件")

        all_results = []

        for pdb_file in tqdm(pdb_files, desc="筛选进度"):
            try:
                result = self.predictor.predict(
                    pdb_path=str(pdb_file),
                    site_threshold=site_threshold
                )

                if result['catalytic_residues']:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"处理 {pdb_file.name} 失败: {e}")

        # 展开并排序
        flattened = []
        for result in all_results:
            for res in result['catalytic_residues'][:top_k]:
                flattened.append({
                    'pdb_id': result['pdb_id'],
                    'ec1_prediction': result['ec1_prediction'],
                    'ec1_confidence': result['ec1_confidence'],
                    'chain': res['chain'],
                    'resseq': res['resseq'],
                    'resname': res['resname'],
                    'site_prob': res['site_prob'],
                    'triads': result.get('triads', []),
                    'metal_centers': result.get('metal_centers', []),
                    'bimetallic_centers': result.get('bimetallic_centers', [])
                })

        flattened.sort(key=lambda x: x['site_prob'], reverse=True)

        logger.info(f"筛选完成! 共找到 {len(flattened)} 个高分催化残基")
        return flattened

    def _screen_single_pdb(self, pdb_id: str,
                          site_threshold: float,
                          ec_filter: Optional[int]) -> Optional[Dict]:
        """
        处理单个 PDB。

        Args:
            pdb_id: PDB ID
            site_threshold: 催化位点阈值
            ec_filter: EC 类别过滤器

        Returns:
            Optional[Dict]: 筛选结果，如果失败或不符合条件则返回 None
        """
        try:
            result = self.predictor.predict(
                pdb_path=pdb_id,
                site_threshold=site_threshold
            )

            # EC过滤
            if ec_filter and result['ec1_prediction'] != ec_filter:
                logger.debug(f"Filtered {pdb_id}: EC{result['ec1_prediction']} != EC{ec_filter}")
                return None

            # 只保留有催化残基的结果
            if not result['catalytic_residues']:
                logger.debug(f"Filtered {pdb_id}: no catalytic residues found")
                return None

            return result
        except Exception as e:
            logger.warning(f"跳过 {pdb_id}: {e}")
            raise  # 重新抛出异常以便重试机制处理

    def _screen_single_pdb_with_retry(self, pdb_id: str,
                                     site_threshold: float,
                                     ec_filter: Optional[int]) -> Optional[Dict]:
        """
        带重试机制的单个 PDB 处理。

        Args:
            pdb_id: PDB ID
            site_threshold: 催化位点阈值
            ec_filter: EC 类别过滤器

        Returns:
            Optional[Dict]: 筛选结果
        """
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return self._screen_single_pdb(pdb_id, site_threshold, ec_filter)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    logger.debug(f"Retry {attempt + 1}/{self.max_retries} for {pdb_id}")
                else:
                    logger.error(f"Failed after {self.max_retries} attempts for {pdb_id}: {e}")

        # 所有重试都失败
        return None

    def filter_by_residue_type(self, results: List[Dict],
                               residue_types: List[str]) -> List[Dict]:
        """
        按残基类型过滤

        Args:
            results: 筛选结果
            residue_types: 保留的残基类型，如 ['HIS', 'ASP', 'SER']

        Returns:
            过滤后的结果
        """
        filtered = [
            r for r in results
            if r['resname'] in residue_types
        ]
        logger.info(f"残基类型过滤: {len(results)} -> {len(filtered)}")
        return filtered

    def cluster_by_similarity(self, results: List[Dict],
                             distance_threshold: float = 5.0) -> List[List[Dict]]:
        """
        按空间相似性聚类催化中心
        (简化版：基于残基类型和EC类别)

        Args:
            results: 筛选结果
            distance_threshold: 距离阈值 (Å)

        Returns:
            聚类列表
        """
        # 简化版：按 (EC类别, 残基类型) 分组
        from collections import defaultdict
        clusters = defaultdict(list)

        for r in results:
            key = (r['ec1_prediction'], r['resname'])
            clusters[key].append(r)

        cluster_list = list(clusters.values())
        logger.info(f"聚类完成: {len(results)} 个残基 -> {len(cluster_list)} 个簇")

        return cluster_list

    def export_summary(self, results: List[Dict], output_path: str):
        """导出CSV摘要"""
        df = pd.DataFrame(results)

        # 选择关键列
        columns = ['pdb_id', 'chain', 'resseq', 'resname',
                  'site_prob', 'ec1_prediction', 'ec1_confidence']
        df = df[columns]

        df.to_csv(output_path, index=False)
        logger.info(f"摘要已导出: {output_path}")

    def export_for_nanozyme_design(self, results: List[Dict],
                                   output_dir: str,
                                   top_n: int = 20):
        """
        导出用于纳米酶设计的模板。

        为每个高分催化中心生成约束文件，并校验必需字段。

        Args:
            results: 筛选结果
            output_dir: 输出目录
            top_n: 导出前N个
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # 按PDB分组
        from collections import defaultdict
        by_pdb = defaultdict(list)
        for r in results[:top_n]:
            by_pdb[r['pdb_id']].append(r)

        exported_count = 0
        for pdb_id, residues in by_pdb.items():
            try:
                # 校验必需字段
                if not residues:
                    logger.warning(f"Skipping {pdb_id}: no residues")
                    continue

                first_residue = residues[0]
                required_fields = ['ec1_prediction', 'chain', 'resseq', 'resname', 'site_prob']
                missing_fields = [f for f in required_fields if f not in first_residue]
                if missing_fields:
                    logger.warning(f"Skipping {pdb_id}: missing fields {missing_fields}")
                    continue

                # 构建约束文件
                template = {
                    'source_pdb': pdb_id,
                    'ec_class': first_residue['ec1_prediction'],
                    'catalytic_residues': [
                        {
                            'chain': r['chain'],
                            'resseq': r['resseq'],
                            'resname': r['resname'],
                            'site_prob': r['site_prob']
                        }
                        for r in residues
                    ],
                    'triads': first_residue.get('triads', []),
                    'metal_centers': first_residue.get('metal_centers', []),
                    'bimetallic_centers': first_residue.get('bimetallic_centers', [])
                }

                output_file = output_dir_path / f"{pdb_id}_template.json"
                with open(output_file, 'w') as f:
                    json.dump(template, f, indent=2)

                exported_count += 1
                logger.debug(f"Exported template for {pdb_id}")

            except Exception as e:
                logger.error(f"Failed to export template for {pdb_id}: {e}")

        logger.info(f"已导出 {exported_count} 个纳米酶模板到 {output_dir_path}")

        # 如果有失败的任务，导出失败列表
        if self.failed_tasks:
            failed_file = output_dir_path / "failed_tasks.json"
            with open(failed_file, 'w') as f:
                json.dump(self.failed_tasks, f, indent=2)
            logger.info(f"已导出 {len(self.failed_tasks)} 个失败任务到 {failed_file}")

    def get_statistics(self, results: List[Dict]) -> Dict:
        """获取筛选统计信息"""
        from collections import Counter

        stats = {
            'total_residues': len(results),
            'unique_pdbs': len(set(r['pdb_id'] for r in results)),
            'ec_distribution': dict(Counter(r['ec1_prediction'] for r in results)),
            'residue_type_distribution': dict(Counter(r['resname'] for r in results)),
            'avg_site_prob': sum(r['site_prob'] for r in results) / len(results) if results else 0,
            'max_site_prob': max(r['site_prob'] for r in results) if results else 0,
            'min_site_prob': min(r['site_prob'] for r in results) if results else 0
        }

        return stats

    def print_statistics(self, results: List[Dict]):
        """打印统计信息"""
        stats = self.get_statistics(results)

        print("\n" + "="*60)
        print("催化中心筛选统计")
        print("="*60)
        print(f"总催化残基数: {stats['total_residues']}")
        print(f"涉及PDB数: {stats['unique_pdbs']}")
        print(f"平均分数: {stats['avg_site_prob']:.3f}")
        print(f"最高分数: {stats['max_site_prob']:.3f}")
        print(f"\nEC类别分布:")
        for ec, count in sorted(stats['ec_distribution'].items()):
            print(f"  EC{ec}: {count} ({count/stats['total_residues']*100:.1f}%)")
        print(f"\n残基类型分布:")
        for resname, count in sorted(stats['residue_type_distribution'].items(),
                                     key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {resname}: {count} ({count/stats['total_residues']*100:.1f}%)")
        print("="*60 + "\n")
