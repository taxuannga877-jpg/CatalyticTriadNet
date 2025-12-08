#!/usr/bin/env python3
"""
阶段1：功能团组合快速打分器
在组装纳米酶之前，快速过滤掉不可能有活性的功能团组合
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import logging

from .functional_group_extractor import FunctionalGroup
from .substrate_definitions import (
    SUBSTRATE_LIBRARY,
    SUBSTRATE_COMPATIBILITY_RULES,
    validate_substrate
)

logger = logging.getLogger(__name__)


class Stage1FunctionalGroupScorer:
    """
    阶段1快速打分器

    目的：在组装纳米酶之前，快速评估功能团组合的潜在活性
    方法：基于功能团类型、距离、角色匹配等简单规则
    速度：每个组合 < 1ms

    使用示例:
        scorer = Stage1FunctionalGroupScorer(substrate='TMB')

        # 评估单个组合
        score = scorer.score_combination([fg1, fg2, fg3])

        # 批量筛选
        good_combos = scorer.filter_combinations(
            all_functional_groups,
            n_per_combo=3,
            min_score=0.6
        )
    """

    def __init__(self, substrate: str = 'TMB'):
        """
        Args:
            substrate: 目标底物 ('TMB', 'pNPP', 'ABTS', 'OPD', 'H2O2', 'GSH')
        """
        if not validate_substrate(substrate):
            raise ValueError(f"不支持的底物: {substrate}")

        self.substrate = substrate
        self.substrate_def = SUBSTRATE_LIBRARY[substrate]
        self.compatibility_rules = SUBSTRATE_COMPATIBILITY_RULES[substrate]

        logger.info(f"阶段1打分器初始化: 底物={substrate}, 酶类型={self.substrate_def.enzyme_type}")

    def score_combination(self, functional_groups: List[FunctionalGroup]) -> Dict:
        """
        评估功能团组合的潜在活性

        Args:
            functional_groups: 功能团列表（通常2-4个）

        Returns:
            评分结果字典
        """
        if len(functional_groups) < 2:
            return {
                'total_score': 0.0,
                'reason': '功能团数量不足'
            }

        scores = {}

        # 1. 功能团类型匹配 (40%)
        type_score = self._score_functional_group_types(functional_groups)
        scores['type_match'] = type_score

        # 2. 催化角色匹配 (30%)
        role_score = self._score_catalytic_roles(functional_groups)
        scores['role_match'] = role_score

        # 3. 功能团间距离合理性 (20%)
        distance_score = self._score_distances(functional_groups)
        scores['distance'] = distance_score

        # 4. 催化位点概率 (10%)
        prob_score = self._score_site_probabilities(functional_groups)
        scores['probability'] = prob_score

        # 总分
        total_score = (
            type_score * 0.4 +
            role_score * 0.3 +
            distance_score * 0.2 +
            prob_score * 0.1
        )

        return {
            'total_score': total_score,
            'component_scores': scores,
            'functional_groups': [fg.group_id for fg in functional_groups],
            'substrate': self.substrate
        }

    def _score_functional_group_types(self, functional_groups: List[FunctionalGroup]) -> float:
        """评估功能团类型是否匹配底物需求"""
        required_features = self.substrate_def.required_catalytic_features

        fg_types = [fg.group_type for fg in functional_groups]
        fg_roles = [fg.role for fg in functional_groups]

        score = 0.0
        total_checks = 0

        # 检查是否有所需的功能团类型
        if self.substrate == 'TMB' or self.substrate == 'ABTS' or self.substrate == 'OPD':
            # 过氧化物酶：需要金属中心或氧化还原残基
            has_metal = any('metal' in role.lower() for role in fg_roles)
            has_redox = any(fg_type in ['imidazole', 'thiol', 'phenol'] for fg_type in fg_types)

            if has_metal or has_redox:
                score += 1.0
            total_checks += 1

        elif self.substrate == 'pNPP':
            # 磷酸酶：需要亲核试剂 + 广义碱
            has_nucleophile = any(role == 'nucleophile' for role in fg_roles)
            has_base = any(role == 'general_base' for role in fg_roles)
            has_electrostatic = any(role == 'electrostatic' for role in fg_roles)

            if has_nucleophile:
                score += 0.4
            if has_base:
                score += 0.4
            if has_electrostatic:
                score += 0.2
            total_checks += 1

        elif self.substrate == 'H2O2':
            # 过氧化氢酶：需要金属中心
            has_metal = any('metal' in role.lower() for role in fg_roles)
            if has_metal:
                score += 1.0
            total_checks += 1

        elif self.substrate == 'GSH':
            # GPx：需要氧化还原位点
            has_redox = any(fg_type in ['thiol', 'imidazole'] for fg_type in fg_types)
            if has_redox:
                score += 1.0
            total_checks += 1

        return score / max(total_checks, 1)

    def _score_catalytic_roles(self, functional_groups: List[FunctionalGroup]) -> float:
        """评估催化角色的互补性"""
        roles = [fg.role for fg in functional_groups]

        # 检查角色多样性（避免重复）
        unique_roles = len(set(roles))
        diversity_score = min(unique_roles / len(roles), 1.0)

        # 检查角色互补性
        complementarity_score = 0.0

        if self.substrate == 'pNPP':
            # 磷酸酶：亲核 + 碱 + 静电 = 完美组合
            role_set = set(roles)
            if 'nucleophile' in role_set:
                complementarity_score += 0.4
            if 'general_base' in role_set:
                complementarity_score += 0.4
            if 'electrostatic' in role_set:
                complementarity_score += 0.2

        elif self.substrate in ['TMB', 'ABTS', 'OPD']:
            # 过氧化物酶：金属中心 + 氧化还原辅助
            if any('metal' in r for r in roles):
                complementarity_score += 0.6
            if any(r in ['general_base', 'proton_donor'] for r in roles):
                complementarity_score += 0.4

        elif self.substrate == 'GSH':
            # GPx：氧化还原中心 + 底物结合
            if any(r in ['nucleophile', 'general_base'] for r in roles):
                complementarity_score += 1.0

        return (diversity_score * 0.3 + complementarity_score * 0.7)

    def _score_distances(self, functional_groups: List[FunctionalGroup]) -> float:
        """评估功能团间距离的合理性"""
        if len(functional_groups) < 2:
            return 0.0

        # 计算所有功能团中心间的距离
        centers = np.array([fg.get_center() for fg in functional_groups])

        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append(dist)

        # 理想距离范围：8-15Å（催化中心通常在这个范围）
        ideal_min = 8.0
        ideal_max = 15.0

        score = 0.0
        for dist in distances:
            if ideal_min <= dist <= ideal_max:
                score += 1.0
            elif dist < ideal_min:
                # 太近，按比例扣分
                score += max(0, dist / ideal_min)
            else:
                # 太远，按比例扣分
                score += max(0, 1.0 - (dist - ideal_max) / 10.0)

        return score / len(distances)

    def _score_site_probabilities(self, functional_groups: List[FunctionalGroup]) -> float:
        """评估催化位点预测概率"""
        probs = [fg.site_prob for fg in functional_groups]

        # 使用几何平均（避免一个低分拖累整体）
        if len(probs) == 0:
            return 0.0

        geometric_mean = np.exp(np.mean(np.log(probs)))
        return geometric_mean

    def filter_combinations(self,
                          functional_groups: List[FunctionalGroup],
                          n_per_combo: int = 3,
                          min_score: float = 0.6,
                          max_combinations: int = 1000) -> List[Tuple[List[FunctionalGroup], float]]:
        """
        批量筛选功能团组合

        Args:
            functional_groups: 所有功能团
            n_per_combo: 每个组合的功能团数量
            min_score: 最低分数阈值
            max_combinations: 最多评估的组合数

        Returns:
            [(功能团组合, 分数), ...] 按分数降序排列
        """
        logger.info(f"开始筛选功能团组合: {len(functional_groups)} 个功能团, "
                   f"每组 {n_per_combo} 个")

        # 生成所有可能的组合
        all_combos = list(combinations(functional_groups, n_per_combo))

        if len(all_combos) > max_combinations:
            logger.warning(f"组合数过多 ({len(all_combos)}), 随机采样 {max_combinations} 个")
            import random
            all_combos = random.sample(all_combos, max_combinations)

        # 评分
        scored_combos = []
        for combo in all_combos:
            result = self.score_combination(list(combo))
            score = result['total_score']

            if score >= min_score:
                scored_combos.append((list(combo), score, result))

        # 按分数排序
        scored_combos.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"筛选完成: {len(scored_combos)}/{len(all_combos)} 个组合通过 "
                   f"(阈值={min_score})")

        return [(combo, score) for combo, score, _ in scored_combos]

    def get_top_combinations(self,
                           functional_groups: List[FunctionalGroup],
                           n_per_combo: int = 3,
                           top_k: int = 50) -> List[Tuple[List[FunctionalGroup], float]]:
        """
        获取得分最高的K个组合

        Args:
            functional_groups: 所有功能团
            n_per_combo: 每个组合的功能团数量
            top_k: 返回前K个

        Returns:
            前K个最佳组合
        """
        all_scored = self.filter_combinations(
            functional_groups,
            n_per_combo=n_per_combo,
            min_score=0.0,  # 不过滤，全部评分
            max_combinations=10000
        )

        return all_scored[:top_k]

    def explain_score(self, functional_groups: List[FunctionalGroup]) -> str:
        """
        解释评分结果

        Args:
            functional_groups: 功能团组合

        Returns:
            评分解释文本
        """
        result = self.score_combination(functional_groups)

        explanation = f"\n{'='*60}\n"
        explanation += f"功能团组合评分解释 (底物: {self.substrate})\n"
        explanation += f"{'='*60}\n"

        explanation += f"\n总分: {result['total_score']:.3f}\n"
        explanation += f"\n组成:\n"
        for fg in functional_groups:
            explanation += f"  - {fg.group_id}: {fg.group_type} ({fg.role}), prob={fg.site_prob:.3f}\n"

        explanation += f"\n分项得分:\n"
        scores = result['component_scores']
        explanation += f"  类型匹配: {scores['type_match']:.3f} (权重 40%)\n"
        explanation += f"  角色匹配: {scores['role_match']:.3f} (权重 30%)\n"
        explanation += f"  距离合理: {scores['distance']:.3f} (权重 20%)\n"
        explanation += f"  位点概率: {scores['probability']:.3f} (权重 10%)\n"

        # 给出建议
        explanation += f"\n建议:\n"
        if result['total_score'] >= 0.8:
            explanation += "  ✓ 优秀组合，强烈推荐组装\n"
        elif result['total_score'] >= 0.6:
            explanation += "  ✓ 良好组合，可以组装\n"
        elif result['total_score'] >= 0.4:
            explanation += "  ⚠ 一般组合，可能活性较低\n"
        else:
            explanation += "  ✗ 不推荐组合，活性可能很低\n"

        explanation += f"{'='*60}\n"

        return explanation


class MultiSubstrateStage1Scorer:
    """
    多底物阶段1打分器
    同时评估功能团组合对多种底物的活性
    """

    def __init__(self, substrates: List[str] = None):
        """
        Args:
            substrates: 底物列表，None表示所有6种底物
        """
        if substrates is None:
            substrates = ['TMB', 'pNPP', 'ABTS', 'OPD', 'H2O2', 'GSH']

        self.substrates = substrates
        self.scorers = {
            substrate: Stage1FunctionalGroupScorer(substrate)
            for substrate in substrates
        }

        logger.info(f"多底物打分器初始化: {len(substrates)} 种底物")

    def score_combination_all_substrates(self,
                                        functional_groups: List[FunctionalGroup]) -> Dict:
        """
        评估功能团组合对所有底物的活性

        Returns:
            {
                'TMB': {'total_score': 0.8, ...},
                'pNPP': {'total_score': 0.6, ...},
                ...
                'best_substrate': 'TMB',
                'best_score': 0.8
            }
        """
        results = {}

        for substrate, scorer in self.scorers.items():
            results[substrate] = scorer.score_combination(functional_groups)

        # 找出最佳底物
        best_substrate = max(results.keys(), key=lambda s: results[s]['total_score'])
        best_score = results[best_substrate]['total_score']

        results['best_substrate'] = best_substrate
        results['best_score'] = best_score

        return results

    def filter_by_best_substrate(self,
                                functional_groups: List[FunctionalGroup],
                                n_per_combo: int = 3,
                                min_score: float = 0.6) -> Dict[str, List]:
        """
        按最佳底物分类筛选组合

        Returns:
            {
                'TMB': [(combo1, score1), (combo2, score2), ...],
                'pNPP': [...],
                ...
            }
        """
        all_combos = list(combinations(functional_groups, n_per_combo))

        results_by_substrate = {substrate: [] for substrate in self.substrates}

        for combo in all_combos:
            multi_result = self.score_combination_all_substrates(list(combo))

            best_substrate = multi_result['best_substrate']
            best_score = multi_result['best_score']

            if best_score >= min_score:
                results_by_substrate[best_substrate].append((list(combo), best_score))

        # 排序
        for substrate in results_by_substrate:
            results_by_substrate[substrate].sort(key=lambda x: x[1], reverse=True)

        # 统计
        for substrate, combos in results_by_substrate.items():
            logger.info(f"{substrate}: {len(combos)} 个候选组合")

        return results_by_substrate


# =============================================================================
# 辅助函数
# =============================================================================

def quick_screen_functional_groups(functional_groups: List[FunctionalGroup],
                                  substrate: str = 'TMB',
                                  n_per_combo: int = 3,
                                  top_k: int = 50) -> List[Tuple[List[FunctionalGroup], float]]:
    """
    快速筛选功能团组合的便捷函数

    Args:
        functional_groups: 功能团列表
        substrate: 目标底物
        n_per_combo: 每组功能团数量
        top_k: 返回前K个

    Returns:
        前K个最佳组合
    """
    scorer = Stage1FunctionalGroupScorer(substrate)
    return scorer.get_top_combinations(functional_groups, n_per_combo, top_k)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'Stage1FunctionalGroupScorer',
    'MultiSubstrateStage1Scorer',
    'quick_screen_functional_groups',
]
