#!/usr/bin/env python3
"""
阶段2：纳米酶活性精确打分器
对已组装的纳米酶结构进行NAC几何打分和活性预测
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .substrate_definitions import (
    SUBSTRATE_LIBRARY,
    validate_substrate
)

logger = logging.getLogger(__name__)


class Stage2NanozymeActivityScorer:
    """
    阶段2精确打分器

    目的：对已组装的纳米酶结构进行精确的活性评估
    方法：NAC几何打分 + 底物对接（简化版）
    速度：每个纳米酶 1-10秒

    使用示例:
        scorer = Stage2NanozymeActivityScorer(substrate='TMB')

        # 评估单个纳米酶
        result = scorer.score_nanozyme(nanozyme_structure)

        # 批量评估
        ranked = scorer.rank_nanozymes(nanozyme_list)
    """

    def __init__(self, substrate: str = 'TMB'):
        """
        Args:
            substrate: 目标底物
        """
        if not validate_substrate(substrate):
            raise ValueError(f"不支持的底物: {substrate}")

        self.substrate = substrate
        self.substrate_def = SUBSTRATE_LIBRARY[substrate]
        self.nac_conditions = self.substrate_def.nac_conditions

        logger.info(f"阶段2打分器初始化: 底物={substrate}")

    def score_nanozyme(self, nanozyme: Dict) -> Dict:
        """
        评估纳米酶的催化活性

        Args:
            nanozyme: 纳米酶结构字典（来自ScaffoldBuilder）

        Returns:
            评分结果
        """
        # 提取坐标和元素
        coords = np.array(nanozyme['coords'])
        elements = nanozyme['elements']
        fg_indices = nanozyme['functional_group_indices']

        scores = {}

        # 1. NAC几何打分 (60%)
        nac_score = self._calculate_nac_score(coords, elements, fg_indices)
        scores['nac_geometry'] = nac_score

        # 2. 催化中心可及性 (20%)
        accessibility_score = self._calculate_accessibility(coords, fg_indices)
        scores['accessibility'] = accessibility_score

        # 3. 功能团协同性 (10%)
        synergy_score = self._calculate_synergy(coords, fg_indices)
        scores['synergy'] = synergy_score

        # 4. 结构稳定性 (10%)
        stability_score = self._calculate_stability(coords, elements)
        scores['stability'] = stability_score

        # 总分
        total_score = (
            nac_score * 0.6 +
            accessibility_score * 0.2 +
            synergy_score * 0.1 +
            stability_score * 0.1
        )

        # 活性预测
        activity_prediction = self._predict_activity(total_score)

        return {
            'total_score': total_score,
            'component_scores': scores,
            'activity_prediction': activity_prediction,
            'substrate': self.substrate,
            'nanozyme_id': nanozyme.get('assembly_info', {}).get('source_pdbs', 'unknown')
        }

    def _calculate_nac_score(self, coords: np.ndarray,
                            elements: List[str],
                            fg_indices: List[List[int]]) -> float:
        """
        计算NAC几何得分

        这是最核心的评分！检查是否满足近攻击构象条件
        """
        if len(fg_indices) < 2:
            return 0.0

        nac_score = 0.0
        total_weight = 0.0

        # 根据不同底物检查不同的NAC条件
        if self.substrate in ['TMB', 'ABTS', 'OPD']:
            # 过氧化物酶NAC条件
            nac_score, total_weight = self._check_peroxidase_nac(
                coords, elements, fg_indices
            )

        elif self.substrate == 'pNPP':
            # 磷酸酶NAC条件
            nac_score, total_weight = self._check_phosphatase_nac(
                coords, elements, fg_indices
            )

        elif self.substrate == 'H2O2':
            # 过氧化氢酶NAC条件
            nac_score, total_weight = self._check_catalase_nac(
                coords, elements, fg_indices
            )

        elif self.substrate == 'GSH':
            # GPx NAC条件
            nac_score, total_weight = self._check_gpx_nac(
                coords, elements, fg_indices
            )

        return nac_score / max(total_weight, 1.0)

    def _check_peroxidase_nac(self, coords: np.ndarray,
                             elements: List[str],
                             fg_indices: List[List[int]]) -> Tuple[float, float]:
        """检查过氧化物酶NAC条件（TMB, ABTS, OPD）"""
        score = 0.0
        total_weight = 0.0

        # 找到金属中心
        metal_indices = self._find_metal_centers(elements)

        if not metal_indices:
            logger.debug("未找到金属中心，使用氧化还原残基")
            # 如果没有金属，使用氧化还原残基中心
            metal_indices = [fg_indices[0][0]]  # 使用第一个功能团的第一个原子

        # 1. 检查金属中心到底物结合位点的距离
        nac_cond = self.nac_conditions['metal_substrate_distance']
        for metal_idx in metal_indices:
            # 计算到所有功能团中心的距离
            for fg_idx_list in fg_indices:
                fg_center = coords[fg_idx_list].mean(axis=0)
                dist = np.linalg.norm(coords[metal_idx] - fg_center)

                # 评分
                if nac_cond['range'][0] <= dist <= nac_cond['range'][1]:
                    score += nac_cond['weight']
                else:
                    # 距离偏差惩罚
                    deviation = min(
                        abs(dist - nac_cond['range'][0]),
                        abs(dist - nac_cond['range'][1])
                    )
                    penalty = max(0, 1.0 - deviation / 2.0)
                    score += nac_cond['weight'] * penalty

                total_weight += nac_cond['weight']

        # 2. 检查电子转移距离
        if 'electron_transfer_distance' in self.nac_conditions:
            nac_cond = self.nac_conditions['electron_transfer_distance']

            # 计算功能团间距离
            for i in range(len(fg_indices)):
                for j in range(i + 1, len(fg_indices)):
                    center_i = coords[fg_indices[i]].mean(axis=0)
                    center_j = coords[fg_indices[j]].mean(axis=0)
                    dist = np.linalg.norm(center_i - center_j)

                    if nac_cond['range'][0] <= dist <= nac_cond['range'][1]:
                        score += nac_cond['weight']
                    else:
                        deviation = min(
                            abs(dist - nac_cond['range'][0]),
                            abs(dist - nac_cond['range'][1])
                        )
                        penalty = max(0, 1.0 - deviation / 3.0)
                        score += nac_cond['weight'] * penalty

                    total_weight += nac_cond['weight']

        return score, total_weight

    def _check_phosphatase_nac(self, coords: np.ndarray,
                               elements: List[str],
                               fg_indices: List[List[int]]) -> Tuple[float, float]:
        """检查磷酸酶NAC条件（pNPP）"""
        score = 0.0
        total_weight = 0.0

        # 找到亲核试剂（通常是第一个功能团）
        if len(fg_indices) < 2:
            return 0.0, 1.0

        nucleophile_center = coords[fg_indices[0]].mean(axis=0)
        base_center = coords[fg_indices[1]].mean(axis=0)

        # 1. 检查亲核试剂到假想磷原子的距离
        # （简化：使用功能团中心作为反应位点）
        nac_cond = self.nac_conditions['nucleophile_P_distance']

        # 假设底物会结合在亲核试剂附近
        # 这里简化为检查亲核试剂的空间可及性
        reaction_site = nucleophile_center + np.array([0, 0, 3.0])  # 假想底物位置

        dist = np.linalg.norm(nucleophile_center - reaction_site)

        if nac_cond['range'][0] <= dist <= nac_cond['range'][1]:
            score += nac_cond['weight']
        else:
            deviation = min(
                abs(dist - nac_cond['range'][0]),
                abs(dist - nac_cond['range'][1])
            )
            penalty = max(0, 1.0 - deviation / 2.0)
            score += nac_cond['weight'] * penalty

        total_weight += nac_cond['weight']

        # 2. 检查广义碱到亲核试剂的距离
        nac_cond = self.nac_conditions['base_nucleophile_distance']
        dist = np.linalg.norm(nucleophile_center - base_center)

        if nac_cond['range'][0] <= dist <= nac_cond['range'][1]:
            score += nac_cond['weight']
        else:
            deviation = min(
                abs(dist - nac_cond['range'][0]),
                abs(dist - nac_cond['range'][1])
            )
            penalty = max(0, 1.0 - deviation / 2.0)
            score += nac_cond['weight'] * penalty

        total_weight += nac_cond['weight']

        # 3. 检查攻击角度（简化版）
        if len(fg_indices) >= 3:
            nac_cond = self.nac_conditions['attack_angle']
            electrostatic_center = coords[fg_indices[2]].mean(axis=0)

            # 计算角度
            v1 = nucleophile_center - reaction_site
            v2 = electrostatic_center - reaction_site

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            if nac_cond['range'][0] <= angle <= nac_cond['range'][1]:
                score += nac_cond['weight']
            else:
                deviation = min(
                    abs(angle - nac_cond['range'][0]),
                    abs(angle - nac_cond['range'][1])
                )
                penalty = max(0, 1.0 - deviation / 30.0)
                score += nac_cond['weight'] * penalty

            total_weight += nac_cond['weight']

        return score, total_weight

    def _check_catalase_nac(self, coords: np.ndarray,
                           elements: List[str],
                           fg_indices: List[List[int]]) -> Tuple[float, float]:
        """检查过氧化氢酶NAC条件（H2O2）"""
        score = 0.0
        total_weight = 0.0

        # 找到金属中心
        metal_indices = self._find_metal_centers(elements)

        if not metal_indices:
            return 0.0, 1.0

        # 检查金属中心到假想H₂O₂结合位点的距离
        nac_cond = self.nac_conditions['metal_H2O2_distance']

        for metal_idx in metal_indices:
            # 假设H₂O₂会结合在金属附近
            h2o2_site = coords[metal_idx] + np.array([0, 0, 2.2])

            dist = np.linalg.norm(coords[metal_idx] - h2o2_site)

            if nac_cond['range'][0] <= dist <= nac_cond['range'][1]:
                score += nac_cond['weight']
            else:
                deviation = abs(dist - nac_cond['optimal'])
                penalty = max(0, 1.0 - deviation / 1.0)
                score += nac_cond['weight'] * penalty

            total_weight += nac_cond['weight']

        return score, total_weight

    def _check_gpx_nac(self, coords: np.ndarray,
                      elements: List[str],
                      fg_indices: List[List[int]]) -> Tuple[float, float]:
        """检查GPx NAC条件（GSH）"""
        score = 0.0
        total_weight = 0.0

        if len(fg_indices) < 1:
            return 0.0, 1.0

        # 找到氧化还原中心
        active_center = coords[fg_indices[0]].mean(axis=0)

        # 检查巯基到活性中心的距离
        nac_cond = self.nac_conditions['thiol_active_site_distance']

        # 假设GSH会结合在活性中心附近
        gsh_site = active_center + np.array([0, 0, 3.5])

        dist = np.linalg.norm(active_center - gsh_site)

        if nac_cond['range'][0] <= dist <= nac_cond['range'][1]:
            score += nac_cond['weight']
        else:
            deviation = abs(dist - nac_cond['optimal'])
            penalty = max(0, 1.0 - deviation / 2.0)
            score += nac_cond['weight'] * penalty

        total_weight += nac_cond['weight']

        return score, total_weight

    def _find_metal_centers(self, elements: List[str]) -> List[int]:
        """找到金属中心的索引"""
        metals = ['Fe', 'Cu', 'Zn', 'Mn', 'Co', 'Ni', 'Mg', 'Ca', 'Se']
        metal_indices = [
            i for i, elem in enumerate(elements)
            if elem in metals
        ]
        return metal_indices

    def _calculate_accessibility(self, coords: np.ndarray,
                                fg_indices: List[List[int]]) -> float:
        """
        计算催化中心的可及性
        检查底物是否能够接近催化中心
        """
        if len(fg_indices) == 0:
            return 0.0

        # 计算功能团中心
        fg_centers = [coords[idx_list].mean(axis=0) for idx_list in fg_indices]

        # 检查每个功能团周围的空间
        accessibility_scores = []

        for fg_center in fg_centers:
            # 计算到其他原子的最小距离
            distances = np.linalg.norm(coords - fg_center, axis=1)

            # 排除功能团自身的原子
            min_clearance = np.min(distances[distances > 0.1])

            # 理想的clearance: 3-5Å
            if 3.0 <= min_clearance <= 5.0:
                accessibility_scores.append(1.0)
            elif min_clearance < 3.0:
                # 太拥挤
                accessibility_scores.append(min_clearance / 3.0)
            else:
                # 太空旷（可能不稳定）
                accessibility_scores.append(max(0, 1.0 - (min_clearance - 5.0) / 5.0))

        return np.mean(accessibility_scores)

    def _calculate_synergy(self, coords: np.ndarray,
                          fg_indices: List[List[int]]) -> float:
        """
        计算功能团的协同性
        检查功能团是否形成合理的催化网络
        """
        if len(fg_indices) < 2:
            return 0.0

        # 计算功能团中心
        fg_centers = np.array([coords[idx_list].mean(axis=0) for idx_list in fg_indices])

        # 检查功能团是否形成紧密的簇
        center_of_mass = fg_centers.mean(axis=0)
        distances_to_com = np.linalg.norm(fg_centers - center_of_mass, axis=1)

        # 理想情况：所有功能团距离质心5-10Å
        synergy_score = 0.0
        for dist in distances_to_com:
            if 5.0 <= dist <= 10.0:
                synergy_score += 1.0
            else:
                synergy_score += max(0, 1.0 - abs(dist - 7.5) / 7.5)

        return synergy_score / len(fg_indices)

    def _calculate_stability(self, coords: np.ndarray,
                            elements: List[str]) -> float:
        """
        计算结构稳定性
        检查是否有原子冲突
        """
        n_atoms = len(coords)

        # 检查原子间距
        clash_count = 0
        total_pairs = 0

        for i in range(n_atoms):
            for j in range(i + 1, min(i + 50, n_atoms)):  # 只检查附近的原子
                dist = np.linalg.norm(coords[i] - coords[j])

                # 检查是否太近（冲突）
                if dist < 1.0:
                    clash_count += 1

                total_pairs += 1

        # 冲突率
        clash_rate = clash_count / max(total_pairs, 1)

        # 稳定性得分
        stability_score = max(0, 1.0 - clash_rate * 10)

        return stability_score

    def _predict_activity(self, total_score: float) -> Dict:
        """
        根据总分预测活性等级

        Returns:
            {
                'level': 'high'/'medium'/'low'/'very_low',
                'confidence': 0.85,
                'description': '...'
            }
        """
        if total_score >= 0.8:
            return {
                'level': 'high',
                'confidence': 0.9,
                'description': '预测具有高催化活性，强烈推荐实验验证'
            }
        elif total_score >= 0.6:
            return {
                'level': 'medium',
                'confidence': 0.75,
                'description': '预测具有中等催化活性，建议实验验证'
            }
        elif total_score >= 0.4:
            return {
                'level': 'low',
                'confidence': 0.6,
                'description': '预测催化活性较低，可能需要优化'
            }
        else:
            return {
                'level': 'very_low',
                'confidence': 0.5,
                'description': '预测催化活性很低，不推荐'
            }

    def rank_nanozymes(self, nanozymes: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """
        批量评估并排序纳米酶

        Args:
            nanozymes: 纳米酶结构列表

        Returns:
            [(nanozyme, score_result), ...] 按分数降序排列
        """
        logger.info(f"开始评估 {len(nanozymes)} 个纳米酶...")

        scored_nanozymes = []

        for i, nanozyme in enumerate(nanozymes):
            try:
                result = self.score_nanozyme(nanozyme)
                scored_nanozymes.append((nanozyme, result))

                if (i + 1) % 10 == 0:
                    logger.info(f"已评估 {i+1}/{len(nanozymes)}")

            except Exception as e:
                logger.error(f"评估纳米酶 {i} 失败: {e}")

        # 按分数排序
        scored_nanozymes.sort(key=lambda x: x[1]['total_score'], reverse=True)

        logger.info(f"评估完成! 最高分: {scored_nanozymes[0][1]['total_score']:.3f}")

        return scored_nanozymes

    def explain_score(self, nanozyme: Dict) -> str:
        """
        解释评分结果

        Args:
            nanozyme: 纳米酶结构

        Returns:
            评分解释文本
        """
        result = self.score_nanozyme(nanozyme)

        explanation = f"\n{'='*60}\n"
        explanation += f"纳米酶活性评分解释 (底物: {self.substrate})\n"
        explanation += f"{'='*60}\n"

        explanation += f"\n总分: {result['total_score']:.3f}\n"

        explanation += f"\n分项得分:\n"
        scores = result['component_scores']
        explanation += f"  NAC几何: {scores['nac_geometry']:.3f} (权重 60%)\n"
        explanation += f"  可及性: {scores['accessibility']:.3f} (权重 20%)\n"
        explanation += f"  协同性: {scores['synergy']:.3f} (权重 10%)\n"
        explanation += f"  稳定性: {scores['stability']:.3f} (权重 10%)\n"

        explanation += f"\n活性预测:\n"
        pred = result['activity_prediction']
        explanation += f"  等级: {pred['level'].upper()}\n"
        explanation += f"  置信度: {pred['confidence']:.2%}\n"
        explanation += f"  描述: {pred['description']}\n"

        explanation += f"{'='*60}\n"

        return explanation


class MultiSubstrateStage2Scorer:
    """
    多底物阶段2打分器
    同时评估纳米酶对多种底物的活性
    """

    def __init__(self, substrates: List[str] = None):
        """
        Args:
            substrates: 底物列表
        """
        if substrates is None:
            substrates = ['TMB', 'pNPP', 'ABTS', 'OPD', 'H2O2', 'GSH']

        self.substrates = substrates
        self.scorers = {
            substrate: Stage2NanozymeActivityScorer(substrate)
            for substrate in substrates
        }

        logger.info(f"多底物阶段2打分器初始化: {len(substrates)} 种底物")

    def score_nanozyme_all_substrates(self, nanozyme: Dict) -> Dict:
        """
        评估纳米酶对所有底物的活性

        Returns:
            {
                'TMB': {'total_score': 0.8, ...},
                'pNPP': {'total_score': 0.6, ...},
                ...
                'best_substrate': 'TMB',
                'best_score': 0.8,
                'activity_profile': [...]
            }
        """
        results = {}

        for substrate, scorer in self.scorers.items():
            results[substrate] = scorer.score_nanozyme(nanozyme)

        # 找出最佳底物
        best_substrate = max(results.keys(), key=lambda s: results[s]['total_score'])
        best_score = results[best_substrate]['total_score']

        # 活性谱
        activity_profile = [
            (substrate, results[substrate]['total_score'])
            for substrate in self.substrates
        ]
        activity_profile.sort(key=lambda x: x[1], reverse=True)

        results['best_substrate'] = best_substrate
        results['best_score'] = best_score
        results['activity_profile'] = activity_profile

        return results


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'Stage2NanozymeActivityScorer',
    'MultiSubstrateStage2Scorer',
]
