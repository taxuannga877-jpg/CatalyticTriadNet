#!/usr/bin/env python3
"""
纳米酶组装器 - 使用扩散模型生成

🆕 v3.0: 完全使用扩散模型生成纳米酶，集成 StoL 的球谐函数几何编码
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import torch

from ..prediction.batch_screener import BatchCatalyticScreener
from .functional_group_extractor import FunctionalGroupExtractor
from .generator import CatalyticNanozymeGenerator
from .constraints import CatalyticConstraints

logger = logging.getLogger(__name__)


class NanozymeAssembler:
    """
    纳米酶组装器 - 使用扩散模型生成

    🆕 v3.0: 完全使用扩散模型生成纳米酶，集成 StoL 的球谐函数几何编码

    工作流程:
    1. 批量筛选天然酶PDB，识别高分催化中心
    2. 提取催化功能团（His咪唑环、Asp羧基等）
    3. 🆕 使用扩散模型生成纳米酶结构（包含功能团+骨架）
    4. 后处理优化和验证

    使用示例:
        # 初始化组装器
        assembler = NanozymeAssembler(
            model_path='models/diffusion_model.pt',
            device='cuda'
        )

        # 从PDB ID列表生成
        nanozymes = assembler.generate_from_pdb_list(
            pdb_ids=['1acb', '4cha', '1hiv'],
            n_functional_groups=3,
            n_samples=10,  # 生成10个候选
            site_threshold=0.7
        )

        # 导出最佳结果
        best_nanozyme = nanozymes[0]
        assembler.export_nanozyme(best_nanozyme, 'output/nanozyme')
    """

    def __init__(self, model_path: str,
                 device: str = None,
                 config: Optional[Dict] = None):
        """
        Args:
            model_path: 训练好的扩散模型路径
            device: 'cuda' 或 'cpu'
            config: 扩散模型配置
        """
        logger.info("="*60)
        logger.info("初始化纳米酶组装器 (扩散模型)")
        logger.info("="*60)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化各个模块
        self.screener = BatchCatalyticScreener(
            model_path=model_path,
            device=self.device
        )
        self.extractor = FunctionalGroupExtractor()

        # 🆕 使用扩散模型生成器
        self.generator = CatalyticNanozymeGenerator(
            model_path=model_path,
            config=config,
            device=self.device
        )

        logger.info(f"设备: {self.device}")
        logger.info("✓ 扩散模型已加载（集成 StoL 球谐函数编码）")
        logger.info("组装器初始化完成\n")

    def generate_from_pdb_list(self,
                               pdb_ids: List[str],
                               n_functional_groups: int = 3,
                               n_samples: int = 10,
                               site_threshold: float = 0.7,
                               ec_filter: Optional[int] = None,
                               filter_by_type: Optional[List[str]] = None,
                               filter_by_role: Optional[List[str]] = None,
                               n_atoms: int = None,
                               guidance_scale: float = 2.0) -> List[Dict]:
        """
        🆕 使用扩散模型从PDB ID列表生成纳米酶

        Args:
            pdb_ids: PDB ID列表
            n_functional_groups: 使用的功能团数量
            n_samples: 生成样本数（每个样本是一个候选纳米酶）
            site_threshold: 催化位点概率阈值 (0-1)
            ec_filter: 只使用特定EC类别的酶
            filter_by_type: 只使用特定功能团类型 ['imidazole', 'carboxylate']
            filter_by_role: 只使用特定催化角色 ['nucleophile', 'general_base']
            n_atoms: 生成的原子数（None则自动估计）
            guidance_scale: 条件引导强度

        Returns:
            纳米酶结构列表（按质量排序）
        """
        logger.info("\n" + "="*60)
        logger.info("开始纳米酶生成流程 (扩散模型)")
        logger.info("="*60)
        logger.info(f"输入PDB数: {len(pdb_ids)}")
        logger.info(f"目标功能团数: {n_functional_groups}")
        logger.info(f"生成样本数: {n_samples}")
        logger.info(f"催化位点阈值: {site_threshold}")

        # 步骤1: 批量筛选催化中心
        logger.info("\n[步骤1/4] 批量筛选催化中心...")
        screening_results = self.screener.screen_pdb_list(
            pdb_ids=pdb_ids,
            site_threshold=site_threshold,
            top_k=10,
            ec_filter=ec_filter
        )

        if not screening_results:
            raise ValueError("未找到符合条件的催化中心")

        self.screener.print_statistics(screening_results)

        # 步骤2: 提取功能团
        logger.info("\n[步骤2/4] 提取催化功能团...")
        functional_groups = self.extractor.extract_from_screening_results(
            screening_results,
            top_n=n_functional_groups * 3  # 提取更多，后续过滤
        )

        if not functional_groups:
            raise ValueError("未能提取到功能团")

        # 过滤功能团
        if filter_by_type:
            functional_groups = self.extractor.filter_by_type(
                functional_groups, filter_by_type
            )

        if filter_by_role:
            functional_groups = self.extractor.filter_by_role(
                functional_groups, filter_by_role
            )

        # 去重
        functional_groups = self.extractor.deduplicate(functional_groups)

        # 选择前N个
        functional_groups = functional_groups[:n_functional_groups]

        if len(functional_groups) < 2:
            raise ValueError(f"功能团数量不足: {len(functional_groups)} < 2")

        self.extractor.print_statistics(functional_groups)

        # 步骤3: 🆕 使用扩散模型生成纳米酶
        logger.info(f"\n[步骤3/4] 使用扩散模型生成纳米酶 ({n_samples} 个候选)...")
        logger.info("✓ 使用 StoL 球谐函数几何编码")

        # 构建催化约束
        constraints = CatalyticConstraints.from_functional_groups(functional_groups)

        # 扩散模型生成
        nanozymes = self.generator.generate(
            constraints=constraints,
            n_samples=n_samples,
            n_atoms=n_atoms,
            guidance_scale=guidance_scale
        )

        # 步骤4: 添加元数据
        logger.info("\n[步骤4/4] 添加元数据和排序...")
        for i, nanozyme in enumerate(nanozymes):
            nanozyme['assembly_info'] = {
                'source_pdbs': list(set(fg.source_pdb for fg in functional_groups)),
                'n_source_pdbs': len(set(fg.source_pdb for fg in functional_groups)),
                'functional_groups_used': [fg.group_id for fg in functional_groups],
                'generation_method': 'diffusion_model',
                'site_threshold': site_threshold,
                'ec_filter': ec_filter,
                'sample_id': i
            }

        # 按约束满足度和有效性排序
        nanozymes.sort(
            key=lambda x: (
                -sum(c['satisfied'] for c in x['constraint_satisfaction']['distance']),
                -x['validity_scores']['connected'],
                x['validity_scores']['clash_count']
            )
        )

        logger.info("\n" + "="*60)
        logger.info("纳米酶生成完成!")
        logger.info("="*60)
        logger.info(f"成功生成: {len(nanozymes)} 个候选")
        logger.info(f"最佳候选原子数: {nanozymes[0]['n_atoms']}")
        logger.info(f"最佳候选约束满足: {sum(c['satisfied'] for c in nanozymes[0]['constraint_satisfaction']['distance'])}/{len(nanozymes[0]['constraint_satisfaction']['distance'])}")
        logger.info(f"来源PDB: {', '.join(nanozymes[0]['assembly_info']['source_pdbs'])}")
        logger.info("="*60 + "\n")

        return nanozymes

    def export_nanozyme(self, nanozyme: Dict, output_path: str):
        """
        导出纳米酶结构

        Args:
            nanozyme: 纳米酶结构字典
            output_path: 输出路径（不含扩展名）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 导出为 XYZ 格式
        xyz_path = output_path.with_suffix('.xyz')
        self.generator.to_xyz(nanozyme, str(xyz_path))
        logger.info(f"✓ 已导出 XYZ: {xyz_path}")

        # 导出为 MOL 格式
        mol_path = output_path.with_suffix('.mol')
        self.generator.to_mol(nanozyme, str(mol_path))
        logger.info(f"✓ 已导出 MOL: {mol_path}")

        # 导出元数据
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'assembly_info': nanozyme['assembly_info'],
                'constraint_satisfaction': nanozyme['constraint_satisfaction'],
                'validity_scores': nanozyme['validity_scores'],
                'n_atoms': nanozyme['n_atoms']
            }, f, indent=2)
        logger.info(f"✓ 已导出元数据: {json_path}")
