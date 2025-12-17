#!/usr/bin/env python3
"""
双阶段多底物打分系统完整示例
支持6种经典纳米酶底物：TMB, pNPP, ABTS, OPD, H₂O₂, GSH
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from catalytic_triad_net import (
    # 筛选和提取
    BatchCatalyticScreener,
    FunctionalGroupExtractor,

    # 组装
    NanozymeAssembler,
    ScaffoldBuilder,

    # 底物定义
    SUBSTRATE_LIBRARY,
    get_all_substrate_names,

    # 阶段1打分
    Stage1FunctionalGroupScorer,
    MultiSubstrateStage1Scorer,
    quick_screen_functional_groups,

    # 阶段2打分
    Stage2NanozymeActivityScorer,
    MultiSubstrateStage2Scorer,
)


def example_1_basic_two_stage_scoring():
    """
    示例1: 基础双阶段打分
    完整展示从筛选到打分的全流程
    """
    print("\n" + "="*70)
    print("示例1: 基础双阶段打分流程")
    print("="*70)

    # ========== 步骤1: 筛选催化中心 ==========
    print("\n[步骤1] 筛选催化中心...")
    screener = BatchCatalyticScreener(
        model_path='models/best_model.pt',
        device='cuda'
    )

    results = screener.screen_pdb_list(
        pdb_ids=['1acb', '4cha', '1hiv'],
        site_threshold=0.7,
        top_k=10
    )

    print(f"找到 {len(results)} 个高分催化残基")

    # ========== 步骤2: 提取功能团 ==========
    print("\n[步骤2] 提取功能团...")
    extractor = FunctionalGroupExtractor()

    functional_groups = extractor.extract_from_screening_results(
        results,
        top_n=20
    )

    print(f"提取了 {len(functional_groups)} 个功能团")

    # ========== 步骤3: 阶段1打分 - 快速筛选功能团组合 ==========
    print("\n[步骤3] 阶段1打分 - 快速筛选功能团组合...")

    # 针对TMB底物打分
    stage1_scorer = Stage1FunctionalGroupScorer(substrate='TMB')

    # 筛选前50个最佳组合
    top_combinations = stage1_scorer.get_top_combinations(
        functional_groups,
        n_per_combo=3,
        top_k=50
    )

    print(f"阶段1筛选出 {len(top_combinations)} 个候选组合")
    print(f"最高分: {top_combinations[0][1]:.3f}")

    # 打印前5个组合
    print("\n前5个最佳组合:")
    for i, (combo, score) in enumerate(top_combinations[:5], 1):
        print(f"  {i}. 分数={score:.3f}, 功能团: {[fg.group_id for fg in combo]}")

    # ========== 步骤4: 组装纳米酶 ==========
    print("\n[步骤4] 组装纳米酶（只组装top 10）...")

    builder = ScaffoldBuilder(scaffold_type='carbon_chain')

    nanozymes = []
    for combo, score in top_combinations[:10]:
        nanozyme = builder.build_nanozyme(
            functional_groups=combo,
            optimize=True
        )
        nanozyme['stage1_score'] = score
        nanozymes.append(nanozyme)

    print(f"成功组装 {len(nanozymes)} 个纳米酶")

    # ========== 步骤5: 阶段2打分 - 精确评估活性 ==========
    print("\n[步骤5] 阶段2打分 - 精确评估活性...")

    stage2_scorer = Stage2NanozymeActivityScorer(substrate='TMB')

    ranked_nanozymes = stage2_scorer.rank_nanozymes(nanozymes)

    print(f"\n阶段2评估完成!")
    print(f"最高分: {ranked_nanozymes[0][1]['total_score']:.3f}")

    # 打印前3个纳米酶
    print("\n前3个最佳纳米酶:")
    for i, (nanozyme, result) in enumerate(ranked_nanozymes[:3], 1):
        print(f"\n  {i}. 总分={result['total_score']:.3f}")
        print(f"     阶段1分数: {nanozyme['stage1_score']:.3f}")
        print(f"     NAC几何: {result['component_scores']['nac_geometry']:.3f}")
        print(f"     活性预测: {result['activity_prediction']['level']}")

    # 导出最佳纳米酶
    best_nanozyme = ranked_nanozymes[0][0]
    builder.export_to_xyz(best_nanozyme, 'output/best_nanozyme_TMB.xyz')
    builder.export_to_pdb(best_nanozyme, 'output/best_nanozyme_TMB.pdb')

    print("\n✓ 示例1完成! 最佳纳米酶已导出到 output/")


def example_2_multi_substrate_stage1():
    """
    示例2: 多底物阶段1打分
    同时评估功能团组合对6种底物的活性
    """
    print("\n" + "="*70)
    print("示例2: 多底物阶段1打分")
    print("="*70)

    # 准备功能团（简化：使用示例数据）
    print("\n准备功能团...")
    screener = BatchCatalyticScreener(model_path='models/best_model.pt')
    results = screener.screen_pdb_list(['1acb', '4cha'], site_threshold=0.7)

    extractor = FunctionalGroupExtractor()
    functional_groups = extractor.extract_from_screening_results(results, top_n=15)

    # 多底物打分
    print("\n[阶段1] 多底物打分...")
    multi_scorer = MultiSubstrateStage1Scorer(
        substrates=['TMB', 'pNPP', 'ABTS', 'OPD', 'H2O2', 'GSH']
    )

    # 按最佳底物分类筛选
    results_by_substrate = multi_scorer.filter_by_best_substrate(
        functional_groups,
        n_per_combo=3,
        min_score=0.6
    )

    # 打印结果
    print("\n各底物的候选组合数:")
    for substrate, combos in results_by_substrate.items():
        print(f"  {substrate}: {len(combos)} 个")

    # 打印每种底物的最佳组合
    print("\n每种底物的最佳组合:")
    for substrate, combos in results_by_substrate.items():
        if combos:
            best_combo, best_score = combos[0]
            print(f"\n  {substrate}:")
            print(f"    分数: {best_score:.3f}")
            print(f"    功能团: {[fg.group_id for fg in best_combo]}")

    print("\n✓ 示例2完成!")


def example_3_multi_substrate_stage2():
    """
    示例3: 多底物阶段2打分
    评估纳米酶对所有底物的活性谱
    """
    print("\n" + "="*70)
    print("示例3: 多底物阶段2打分 - 活性谱分析")
    print("="*70)

    # 准备纳米酶（简化：组装一个示例纳米酶）
    print("\n准备纳米酶...")
    screener = BatchCatalyticScreener(model_path='models/best_model.pt')
    results = screener.screen_pdb_list(['1acb'], site_threshold=0.7)

    extractor = FunctionalGroupExtractor()
    functional_groups = extractor.extract_from_screening_results(results, top_n=3)

    builder = ScaffoldBuilder(scaffold_type='carbon_chain')
    nanozyme = builder.build_nanozyme(functional_groups, optimize=True)

    # 多底物阶段2打分
    print("\n[阶段2] 评估对所有底物的活性...")
    multi_scorer = MultiSubstrateStage2Scorer(
        substrates=['TMB', 'pNPP', 'ABTS', 'OPD', 'H2O2', 'GSH']
    )

    results = multi_scorer.score_nanozyme_all_substrates(nanozyme)

    # 打印活性谱
    print("\n纳米酶活性谱:")
    print(f"最佳底物: {results['best_substrate']} (分数: {results['best_score']:.3f})")

    print("\n所有底物的活性:")
    for substrate, score in results['activity_profile']:
        activity_level = results[substrate]['activity_prediction']['level']
        print(f"  {substrate:8s}: {score:.3f} ({activity_level})")

    # 详细分析最佳底物
    best_substrate = results['best_substrate']
    best_result = results[best_substrate]

    print(f"\n最佳底物 ({best_substrate}) 详细分析:")
    print(f"  NAC几何: {best_result['component_scores']['nac_geometry']:.3f}")
    print(f"  可及性: {best_result['component_scores']['accessibility']:.3f}")
    print(f"  协同性: {best_result['component_scores']['synergy']:.3f}")
    print(f"  稳定性: {best_result['component_scores']['stability']:.3f}")

    print("\n✓ 示例3完成!")


def example_4_complete_workflow_with_scoring():
    """
    示例4: 完整工作流（集成双阶段打分）
    从筛选到最终纳米酶，全程打分指导
    """
    print("\n" + "="*70)
    print("示例4: 完整工作流 - 双阶段打分指导")
    print("="*70)

    # 目标底物
    target_substrate = 'pNPP'  # 磷酸酶活性

    print(f"\n目标底物: {target_substrate}")
    print(f"底物信息: {SUBSTRATE_LIBRARY[target_substrate].full_name}")
    print(f"酶类型: {SUBSTRATE_LIBRARY[target_substrate].enzyme_type}")

    # ========== 步骤1: 筛选 ==========
    print("\n[步骤1] 筛选催化中心...")
    screener = BatchCatalyticScreener(model_path='models/best_model.pt')

    # 针对磷酸酶，筛选水解酶
    results = screener.screen_pdb_list(
        pdb_ids=['1acb', '4cha', '1hiv', '1ppf'],
        site_threshold=0.7,
        ec_filter=3  # 水解酶
    )

    print(f"找到 {len(results)} 个催化残基")

    # ========== 步骤2: 提取 ==========
    print("\n[步骤2] 提取功能团...")
    extractor = FunctionalGroupExtractor()
    functional_groups = extractor.extract_from_screening_results(results, top_n=30)

    # 针对pNPP，过滤出合适的功能团
    functional_groups = extractor.filter_by_role(
        functional_groups,
        roles=['nucleophile', 'general_base', 'electrostatic']
    )

    print(f"过滤后: {len(functional_groups)} 个功能团")

    # ========== 步骤3: 阶段1打分 ==========
    print("\n[步骤3] 阶段1打分 - 筛选功能团组合...")
    stage1_scorer = Stage1FunctionalGroupScorer(substrate=target_substrate)

    top_combinations = stage1_scorer.get_top_combinations(
        functional_groups,
        n_per_combo=3,
        top_k=20
    )

    print(f"筛选出 {len(top_combinations)} 个候选组合")

    # 解释最佳组合的分数
    if top_combinations:
        best_combo, best_score = top_combinations[0]
        explanation = stage1_scorer.explain_score(best_combo)
        print(explanation)

    # ========== 步骤4: 组装 ==========
    print("\n[步骤4] 组装纳米酶...")
    builder = ScaffoldBuilder(
        scaffold_type='carbon_chain',
        scaffold_params={'chain_length': 3}
    )

    nanozymes = []
    for combo, stage1_score in top_combinations[:10]:
        nanozyme = builder.build_nanozyme(combo, optimize=True)
        nanozyme['stage1_score'] = stage1_score
        nanozymes.append(nanozyme)

    print(f"组装了 {len(nanozymes)} 个纳米酶")

    # ========== 步骤5: 阶段2打分 ==========
    print("\n[步骤5] 阶段2打分 - 精确评估...")
    stage2_scorer = Stage2NanozymeActivityScorer(substrate=target_substrate)

    ranked = stage2_scorer.rank_nanozymes(nanozymes)

    # 解释最佳纳米酶的分数
    best_nanozyme, best_result = ranked[0]
    explanation = stage2_scorer.explain_score(best_nanozyme)
    print(explanation)

    # ========== 步骤6: 导出 ==========
    print("\n[步骤6] 导出最佳纳米酶...")
    builder.export_to_xyz(best_nanozyme, f'output/best_nanozyme_{target_substrate}.xyz')
    builder.export_to_pdb(best_nanozyme, f'output/best_nanozyme_{target_substrate}.pdb')
    builder.export_to_mol2(best_nanozyme, f'output/best_nanozyme_{target_substrate}.mol2')

    # 生成报告
    with open(f'output/nanozyme_{target_substrate}_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"纳米酶设计报告 - {target_substrate}\n")
        f.write("="*70 + "\n\n")

        f.write(f"目标底物: {SUBSTRATE_LIBRARY[target_substrate].full_name}\n")
        f.write(f"酶类型: {SUBSTRATE_LIBRARY[target_substrate].enzyme_type}\n")
        f.write(f"检测波长: {SUBSTRATE_LIBRARY[target_substrate].detection_wavelength} nm\n\n")

        f.write("评分结果:\n")
        f.write(f"  阶段1分数: {best_nanozyme['stage1_score']:.3f}\n")
        f.write(f"  阶段2分数: {best_result['total_score']:.3f}\n")
        f.write(f"  活性预测: {best_result['activity_prediction']['level']}\n")
        f.write(f"  置信度: {best_result['activity_prediction']['confidence']:.2%}\n\n")

        f.write("分项得分:\n")
        for key, value in best_result['component_scores'].items():
            f.write(f"  {key}: {value:.3f}\n")

        f.write("\n" + "="*70 + "\n")

    print(f"\n✓ 示例4完成! 报告已保存到 output/nanozyme_{target_substrate}_report.txt")


def example_5_compare_substrates():
    """
    示例5: 底物比较
    比较同一个纳米酶对不同底物的活性
    """
    print("\n" + "="*70)
    print("示例5: 底物比较 - 找出最佳应用场景")
    print("="*70)

    # 准备纳米酶
    print("\n准备纳米酶...")
    screener = BatchCatalyticScreener(model_path='models/best_model.pt')
    results = screener.screen_pdb_list(['1acb', '4cha'], site_threshold=0.7)

    extractor = FunctionalGroupExtractor()
    functional_groups = extractor.extract_from_screening_results(results, top_n=3)

    builder = ScaffoldBuilder(scaffold_type='metal_framework')
    nanozyme = builder.build_nanozyme(functional_groups, optimize=True)

    # 评估所有底物
    print("\n评估对所有底物的活性...")
    all_substrates = get_all_substrate_names()

    results = {}
    for substrate in all_substrates:
        scorer = Stage2NanozymeActivityScorer(substrate=substrate)
        result = scorer.score_nanozyme(nanozyme)
        results[substrate] = result

    # 排序
    sorted_substrates = sorted(
        results.items(),
        key=lambda x: x[1]['total_score'],
        reverse=True
    )

    # 打印比较结果
    print("\n底物活性排名:")
    print(f"{'排名':<6} {'底物':<10} {'分数':<8} {'活性等级':<12} {'推荐度'}")
    print("-" * 60)

    for i, (substrate, result) in enumerate(sorted_substrates, 1):
        score = result['total_score']
        level = result['activity_prediction']['level']
        confidence = result['activity_prediction']['confidence']

        recommendation = "⭐⭐⭐" if score >= 0.8 else "⭐⭐" if score >= 0.6 else "⭐"

        print(f"{i:<6} {substrate:<10} {score:<8.3f} {level:<12} {recommendation}")

    # 推荐应用
    best_substrate = sorted_substrates[0][0]
    best_score = sorted_substrates[0][1]['total_score']

    print(f"\n推荐应用:")
    print(f"  该纳米酶最适合用于 {best_substrate} 底物")
    print(f"  预测活性分数: {best_score:.3f}")
    print(f"  底物全名: {SUBSTRATE_LIBRARY[best_substrate].full_name}")
    print(f"  检测方法: {SUBSTRATE_LIBRARY[best_substrate].detection_wavelength} nm")

    print("\n✓ 示例5完成!")


def example_6_quick_screening():
    """
    示例6: 快速筛选
    使用便捷函数快速筛选功能团组合
    """
    print("\n" + "="*70)
    print("示例6: 快速筛选（便捷函数）")
    print("="*70)

    # 准备功能团
    screener = BatchCatalyticScreener(model_path='models/best_model.pt')
    results = screener.screen_pdb_list(['1acb'], site_threshold=0.7)

    extractor = FunctionalGroupExtractor()
    functional_groups = extractor.extract_from_screening_results(results, top_n=10)

    # 使用便捷函数快速筛选
    print("\n使用便捷函数快速筛选...")

    from catalytic_triad_net import quick_screen_functional_groups

    top_combos = quick_screen_functional_groups(
        functional_groups,
        substrate='TMB',
        n_per_combo=3,
        top_k=5
    )

    print(f"\n快速筛选出 {len(top_combos)} 个最佳组合:")
    for i, (combo, score) in enumerate(top_combos, 1):
        print(f"  {i}. 分数={score:.3f}")
        for fg in combo:
            print(f"     - {fg.group_id}: {fg.group_type} ({fg.role})")

    print("\n✓ 示例6完成!")


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("双阶段多底物打分系统示例集")
    print("="*70)
    print("\n支持的底物:")
    for substrate in get_all_substrate_names():
        info = SUBSTRATE_LIBRARY[substrate]
        print(f"  - {substrate:8s}: {info.full_name} ({'⭐' * info.usage_frequency})")

    # 创建输出目录
    Path('output').mkdir(exist_ok=True)

    # 运行示例
    try:
        example_1_basic_two_stage_scoring()
    except Exception as e:
        print(f"示例1失败: {e}")

    try:
        example_2_multi_substrate_stage1()
    except Exception as e:
        print(f"示例2失败: {e}")

    try:
        example_3_multi_substrate_stage2()
    except Exception as e:
        print(f"示例3失败: {e}")

    try:
        example_4_complete_workflow_with_scoring()
    except Exception as e:
        print(f"示例4失败: {e}")

    try:
        example_5_compare_substrates()
    except Exception as e:
        print(f"示例5失败: {e}")

    try:
        example_6_quick_screening()
    except Exception as e:
        print(f"示例6失败: {e}")

    print("\n" + "="*70)
    print("所有示例完成!")
    print("="*70)


if __name__ == "__main__":
    # 运行单个示例
    # example_1_basic_two_stage_scoring()

    # 或运行所有示例
    main()
