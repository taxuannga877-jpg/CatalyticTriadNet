"""
双轨处理器 - 两手抓策略核心模块
================================

策略：
1. 标注数据 (Annotated): 直接使用 UniProt/M-CSA 的活性位点注释
2. 未标注数据 (Unannotated): 使用 EasIFA 模型预测活性位点

此模块整合两种数据流，统一输出格式。
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

from ..database import UniProtFetcher, MCSAFetcher, NanozymeDatabase, EnzymeEntry
from ..prediction import EasIFAPredictor, ActiveSiteResult, PredictedActiveSite
from ..utils.constants import NanozymeType


@dataclass
class ProcessedEnzyme:
    """处理后的酶数据（统一格式）"""
    uniprot_id: str
    ec_number: str
    nanozyme_type: str
    pdb_path: str
    sequence: str

    # 活性位点信息
    active_sites: List[Dict] = field(default_factory=list)
    source: str = ""  # "annotated" 或 "predicted"

    # 元数据
    confidence: float = 0.0
    raw_labels: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DualTrackProcessor:
    """
    双轨处理器 - 两手抓策略

    自动判断酶数据是否有标注：
    - 有标注：直接使用
    - 无标注：调用 EasIFA 预测
    """

    def __init__(
        self,
        output_dir: str = "./processed",
        device: str = "cpu",
        enzyme_model_path: Optional[str] = None,
        reaction_model_path: Optional[str] = None
    ):
        """
        初始化双轨处理器。

        Args:
            output_dir: 输出目录
            device: EasIFA 运行设备
            enzyme_model_path: 酶位点预测模型路径，None则使用默认
            reaction_model_path: 反应注意力模型路径，None则使用默认
        """
        self.output_dir = Path(output_dir)
        self.annotated_dir = self.output_dir / "annotated"
        self.predicted_dir = self.output_dir / "predicted"

        for d in [self.annotated_dir, self.predicted_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 初始化 EasIFA 预测器（必须）
        print("[DualTrack] 初始化 EasIFA 预测器...")
        self.predictor = EasIFAPredictor(
            device=device,
            enzyme_model_path=enzyme_model_path,
            reaction_model_path=reaction_model_path
        )
        print("[DualTrack] 双轨处理器初始化完成")

    def process_enzyme(
        self,
        uniprot_id: str,
        ec_number: str,
        nanozyme_type: str,
        pdb_path: str,
        sequence: str,
        known_sites: Optional[List[Dict]] = None,
        reaction_smiles: str = "C>>C"
    ) -> ProcessedEnzyme:
        """
        处理单个酶数据。

        Args:
            uniprot_id: UniProt ID
            ec_number: EC 号
            nanozyme_type: 纳米酶类型
            pdb_path: PDB 文件路径
            sequence: 氨基酸序列
            known_sites: 已知活性位点（来自 UniProt/M-CSA）
            reaction_smiles: 反应 SMILES

        Returns:
            ProcessedEnzyme 对象
        """
        # 判断是否有标注
        if known_sites and len(known_sites) > 0:
            # 标注数据：直接使用
            return self._process_annotated(
                uniprot_id, ec_number, nanozyme_type,
                pdb_path, sequence, known_sites
            )
        else:
            # 未标注数据：使用 EasIFA 预测
            return self._process_unannotated(
                uniprot_id, ec_number, nanozyme_type,
                pdb_path, sequence, reaction_smiles
            )

    def _process_annotated(
        self,
        uniprot_id: str,
        ec_number: str,
        nanozyme_type: str,
        pdb_path: str,
        sequence: str,
        known_sites: List[Dict]
    ) -> ProcessedEnzyme:
        """处理标注数据"""
        print(f"[Annotated] {uniprot_id}: 使用已知标注")

        result = ProcessedEnzyme(
            uniprot_id=uniprot_id,
            ec_number=ec_number,
            nanozyme_type=nanozyme_type,
            pdb_path=pdb_path,
            sequence=sequence,
            active_sites=known_sites,
            source="annotated",
            confidence=1.0
        )

        # 保存结果
        output_path = self.annotated_dir / f"{uniprot_id}.json"
        result.to_json(str(output_path))

        return result

    def _process_unannotated(
        self,
        uniprot_id: str,
        ec_number: str,
        nanozyme_type: str,
        pdb_path: str,
        sequence: str,
        reaction_smiles: str
    ) -> ProcessedEnzyme:
        """处理未标注数据 - 使用 EasIFA 预测"""
        print(f"[Predicted] {uniprot_id}: 调用 EasIFA 预测")

        # 调用 EasIFA 预测
        pred_result = self.predictor.predict_with_details(
            pdb_path=pdb_path,
            uniprot_id=uniprot_id,
            reaction_smiles=reaction_smiles
        )

        if pred_result is None:
            print(f"[Predicted] {uniprot_id}: 预测失败")
            return ProcessedEnzyme(
                uniprot_id=uniprot_id,
                ec_number=ec_number,
                nanozyme_type=nanozyme_type,
                pdb_path=pdb_path,
                sequence=sequence,
                active_sites=[],
                source="predicted_failed"
            )

        # 转换预测结果
        sites = [
            {
                "residue_index": s.residue_index,
                "residue_name": s.residue_name,
                "site_type": s.site_type,
                "coordinates": s.coordinates
            }
            for s in pred_result.sites
        ]

        result = ProcessedEnzyme(
            uniprot_id=uniprot_id,
            ec_number=ec_number,
            nanozyme_type=nanozyme_type,
            pdb_path=pdb_path,
            sequence=sequence,
            active_sites=sites,
            source="predicted",
            raw_labels=pred_result.raw_labels
        )

        # 保存结果
        output_path = self.predicted_dir / f"{uniprot_id}.json"
        result.to_json(str(output_path))

        return result

    def process_batch(
        self,
        entries: List[Dict],
        reaction_smiles: str = "C>>C"
    ) -> Tuple[List[ProcessedEnzyme], List[ProcessedEnzyme]]:
        """
        批量处理酶数据。

        Args:
            entries: 酶数据列表
            reaction_smiles: 反应 SMILES

        Returns:
            (标注结果列表, 预测结果列表)
        """
        annotated_results = []
        predicted_results = []

        for entry in entries:
            result = self.process_enzyme(
                uniprot_id=entry.get("uniprot_id", ""),
                ec_number=entry.get("ec_number", ""),
                nanozyme_type=entry.get("nanozyme_type", ""),
                pdb_path=entry.get("pdb_path", ""),
                sequence=entry.get("sequence", ""),
                known_sites=entry.get("active_sites"),
                reaction_smiles=reaction_smiles
            )

            if result.source == "annotated":
                annotated_results.append(result)
            else:
                predicted_results.append(result)

        print(f"[DualTrack] 处理完成: {len(annotated_results)} 标注, {len(predicted_results)} 预测")
        return annotated_results, predicted_results

    def predict_unannotated_batch(
        self,
        unannotated_entries: List[Dict],
        reaction_smiles: str = "C>>C"
    ) -> List[ProcessedEnzyme]:
        """
        批量预测未标注数据的活性位点。

        流程：下载完成 -> 整理完成 -> 调用此方法批量预测

        Args:
            unannotated_entries: 未标注的酶数据列表
            reaction_smiles: 反应 SMILES

        Returns:
            预测结果列表
        """
        print(f"[DualTrack] 开始批量预测 {len(unannotated_entries)} 条未标注数据...")

        results = []
        for i, entry in enumerate(unannotated_entries):
            print(f"[{i+1}/{len(unannotated_entries)}] 预测 {entry.get('uniprot_id', '')}...")

            result = self._process_unannotated(
                uniprot_id=entry.get("uniprot_id", ""),
                ec_number=entry.get("ec_number", ""),
                nanozyme_type=entry.get("nanozyme_type", ""),
                pdb_path=entry.get("pdb_path", ""),
                sequence=entry.get("sequence", ""),
                reaction_smiles=reaction_smiles
            )
            results.append(result)

        print(f"[DualTrack] 批量预测完成: {len(results)} 条")
        return results
