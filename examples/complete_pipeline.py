"""
Complete Pipeline Example: 正确的三步流程
==========================================

流程：
Step 1: 批量下载 PDB 文件
Step 2: 整理分类（标注 vs 未标注）
Step 3: 对未标注数据批量调用 EasIFA 预测
Step 4: 提取催化 motif
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanozyme_mining import (
    NanozymeDatabase,
    UniProtFetcher,
    DualTrackProcessor,
    MotifExtractor,
)
from nanozyme_mining.utils import NanozymeType


def run_pipeline():
    """运行完整的三步流程 pipeline"""

    # 设置目录
    os.makedirs("./data", exist_ok=True)

    # ============================================================
    # Step 1: 批量下载 PDB 文件 + 整理分类
    # ============================================================
    print("=" * 60)
    print("Step 1: 批量下载 PDB 文件 + 整理分类")
    print("=" * 60)

    fetcher = UniProtFetcher(cache_dir="./data/cache")

    ec_number = "1.11.1.7"  # Peroxidase
    print(f"\n下载 EC {ec_number} 的 PDB 文件...")

    # 下载并分类
    annotated, unannotated = fetcher.fetch_and_classify(
        ec_number=ec_number,
        nanozyme_type=NanozymeType.POD
    )

    print(f"\n下载完成:")
    print(f"  有标注: {len(annotated)} 条")
    print(f"  无标注: {len(unannotated)} 条")

    # ============================================================
    # Step 2: 对未标注数据批量调用 EasIFA 预测
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: 对未标注数据批量调用 EasIFA 预测")
    print("=" * 60)

    # 初始化处理器（加载 EasIFA 模型）
    processor = DualTrackProcessor(
        output_dir="./data/processed",
        device="cpu"
    )

    # 批量预测未标注数据
    predicted_results = processor.predict_unannotated_batch(unannotated)

    print(f"\n预测完成: {len(predicted_results)} 条")

    print("\n" + "=" * 60)
    print("Step 3: 提取催化 Motif")
    print("=" * 60)

    extractor = MotifExtractor(output_dir="./data/motifs")

    # 合并所有结果（标注 + 预测）
    all_results = []

    # 处理标注数据
    for entry in annotated:
        all_results.append({
            "uniprot_id": entry["uniprot_id"],
            "pdb_path": entry["pdb_path"],
            "ec_number": entry["ec_number"],
            "nanozyme_type": entry["nanozyme_type"],
            "active_sites": entry["active_sites"],
            "source": "annotated"
        })

    # 处理预测数据
    for result in predicted_results:
        all_results.append({
            "uniprot_id": result.uniprot_id,
            "pdb_path": result.pdb_path,
            "ec_number": result.ec_number,
            "nanozyme_type": result.nanozyme_type,
            "active_sites": result.active_sites,
            "source": result.source
        })

    for result in all_results:
        pdb_path = result.get("pdb_path")
        if not pdb_path or not os.path.exists(pdb_path):
            continue

        # 获取活性位点索引
        active_sites = result.get("active_sites", [])
        site_indices = []
        for s in active_sites:
            if isinstance(s, dict):
                idx = s.get("residue_index") or s.get("start")
                if idx:
                    site_indices.append(idx)

        motif = extractor.extract_motif(
            pdb_path=pdb_path,
            uniprot_id=result["uniprot_id"],
            ec_number=result["ec_number"],
            nanozyme_type=result["nanozyme_type"],
            active_site_indices=site_indices
        )

        if motif:
            print(f"\n{result['uniprot_id']} [{result['source']}]:")
            print(f"  Anchor atoms: {len(motif.anchor_atoms)}")
            print(f"  Constraints: {len(motif.geometry_constraints)}")
            motif.to_json(f"./data/motifs/{motif.motif_id}.json")

    print("\n" + "=" * 60)
    print("Pipeline 完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
