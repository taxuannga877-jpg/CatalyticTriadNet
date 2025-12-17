#!/usr/bin/env python3
"""
阶段2：纳米酶活性精确打分器
对已组装的纳米酶结构进行NAC几何打分和活性预测

v2.1 更新：集成autodE自动过渡态计算
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import tempfile
import os

from .substrate_definitions import (
    SUBSTRATE_LIBRARY,
    validate_substrate
)

# 尝试导入autodE模块
try:
    from .autode_ts_calculator import AutodETSCalculator, SubstrateReactionLibrary
    AUTODE_AVAILABLE = True
except ImportError:
    AUTODE_AVAILABLE = False

logger = logging.getLogger(__name__)


class Stage2NanozymeActivityScorer:
    """
    阶段2精确打分器

    目的：对已组装的纳米酶结构进行精确的活性评估
    方法：NAC几何打分 + 底物对接（简化版） + autodE TS计算（可选）
    速度：每个纳米酶 1-10秒（几何打分）或 1-10分钟（TS计算）

    使用示例:
        # 基础模式（仅几何打分）
        scorer = Stage2NanozymeActivityScorer(substrate='TMB')
        result = scorer.score_nanozyme(nanozyme_structure)

        # 高精度模式（包含TS计算）
        scorer = Stage2NanozymeActivityScorer(
            substrate='TMB',
            use_ts_calculation=True,
            ts_method='xtb'
        )
        result = scorer.score_nanozyme(nanozyme_structure)

        # 批量评估
        ranked = scorer.rank_nanozymes(nanozyme_list)
    """

    def __init__(
        self,
        substrate: str = 'TMB',
        use_ts_calculation: bool = False,
        ts_method: str = 'xtb',
        ts_quick_mode: bool = False,
        n_cores: int = 4
    ):
        """
        Args:
            substrate: 目标底物
            use_ts_calculation: 是否使用autodE计算过渡态
            ts_method: TS计算方法 ('xtb', 'orca')
            ts_quick_mode: 快速模式（仅估算活化能）
            n_cores: CPU核心数
        """
        if not validate_substrate(substrate):
            raise ValueError(f"不支持的底物: {substrate}")

        self.substrate = substrate
        self.substrate_def = SUBSTRATE_LIBRARY[substrate]
        self.nac_conditions = self.substrate_def.nac_conditions

        # autodE配置
        self.use_ts_calculation = use_ts_calculation
        self.ts_quick_mode = ts_quick_mode
        self.ts_calculator = None

        if use_ts_calculation:
            if not AUTODE_AVAILABLE:
                logger.warning("autodE未安装，TS计算已禁用")
                logger.warning("安装方法: pip install autode && conda install -c conda-forge xtb")
                self.use_ts_calculation = False
            else:
                try:
                    self.ts_calculator = AutodETSCalculator(
                        method=ts_method,
                        n_cores=n_cores
                    )
                    logger.info(f"✓ autodE TS计算器已启用 (method={ts_method}, quick_mode={ts_quick_mode})")
                except Exception as e:
                    logger.error(f"autodE初始化失败: {e}")
                    self.use_ts_calculation = False

        logger.info(f"阶段2打分器初始化: 底物={substrate}, TS计算={'启用' if self.use_ts_calculation else '禁用'}")

    def score_nanozyme(self, nanozyme: Dict) -> Dict:
        """
        评估纳米酶的催化活性

        Args:
            nanozyme: 纳米酶结构字典（来自ScaffoldBuilder）

        Returns:
            评分结果（包含TS计算结果，如果启用）
        """
        # 提取坐标和元素
        coords = np.array(nanozyme['coords'])
        elements = nanozyme['elements']
        fg_indices = nanozyme['functional_group_indices']

        scores = {}

        # 1. NAC几何打分 (60% 或 40% 如果有TS)
        nac_score = self._calculate_nac_score(coords, elements, fg_indices)
        scores['nac_geometry'] = nac_score

        # 2. 催化中心可及性 (20% 或 15% 如果有TS)
        accessibility_score = self._calculate_accessibility(coords, fg_indices)
        scores['accessibility'] = accessibility_score

        # 3. 功能团协同性 (10% 或 8% 如果有TS)
        synergy_score = self._calculate_synergy(coords, fg_indices)
        scores['synergy'] = synergy_score

        # 4. 结构稳定性 (10% 或 7% 如果有TS)
        stability_score = self._calculate_stability(coords, elements)
        scores['stability'] = stability_score

        # 5. autodE TS计算 (30% 如果启用)
        ts_result = None
        ts_score = 0.0

        if self.use_ts_calculation and self.ts_calculator is not None:
            ts_result = self._calculate_ts_score(nanozyme)
            if ts_result and ts_result.get('success', False):
                ts_score = ts_result['ts_score']
                scores['ts_calculation'] = ts_score
                logger.info(f"✓ TS计算完成: Ea={ts_result.get('activation_energy', 'N/A')} kcal/mol")
            else:
                logger.warning("TS计算失败，使用几何打分")

        # 计算总分（根据是否有TS调整权重）
        if self.use_ts_calculation and ts_result and ts_result.get('success', False):
            # 有TS计算：几何40% + 可及15% + 协同8% + 稳定7% + TS 30%
            total_score = (
                nac_score * 0.40 +
                accessibility_score * 0.15 +
                synergy_score * 0.08 +
                stability_score * 0.07 +
                ts_score * 0.30
            )
        else:
            # 无TS计算：几何60% + 可及20% + 协同10% + 稳定10%
            total_score = (
                nac_score * 0.6 +
                accessibility_score * 0.2 +
                synergy_score * 0.1 +
                stability_score * 0.1
            )

        # 活性预测
        activity_prediction = self._predict_activity(total_score, ts_result)

        result = {
            'total_score': total_score,
            'component_scores': scores,
            'activity_prediction': activity_prediction,
            'substrate': self.substrate,
            'nanozyme_id': nanozyme.get('assembly_info', {}).get('source_pdbs', 'unknown'),
            'ts_enabled': self.use_ts_calculation
        }

        # 添加TS详细结果
        if ts_result and ts_result.get('success', False):
            result['ts_details'] = {
                'activation_energy': ts_result.get('activation_energy'),
                'reaction_energy': ts_result.get('reaction_energy'),
                'ts_frequency': ts_result.get('ts_frequency'),
                'method': ts_result.get('method', 'unknown')
            }

        return result

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
            valid_distances = distances[distances > 0.1]
            if valid_distances.size == 0:
                # 功能团自身或全局重合，视为不可及
                accessibility_scores.append(0.0)
                continue

            min_clearance = np.min(valid_distances)

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

    def _calculate_ts_score(self, nanozyme: Dict) -> Dict:
        """
        使用autodE计算过渡态并转换为评分

        Args:
            nanozyme: 纳米酶结构

        Returns:
            {
                'ts_score': 0.0-1.0,
                'activation_energy': float (kcal/mol),
                'reaction_energy': float (kcal/mol),
                'ts_frequency': float (cm^-1),
                'method': str,
                'success': bool
            }
        """
        # 保存纳米酶结构为临时XYZ文件
        temp_xyz = tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False)

        try:
            # 写入XYZ格式
            coords = nanozyme['coords']
            elements = nanozyme['elements']

            temp_xyz.write(f"{len(coords)}\n")
            temp_xyz.write(f"Nanozyme structure\n")
            for elem, coord in zip(elements, coords):
                temp_xyz.write(f"{elem:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
            temp_xyz.close()

            # 获取底物SMILES
            reaction_params = SubstrateReactionLibrary.get_reaction_params(self.substrate)
            if reaction_params is None:
                logger.error(f"未找到底物 {self.substrate} 的反应参数")
                return {'success': False, 'error': 'Unknown substrate'}

            substrate_smiles = reaction_params['substrate_smiles']

            # 执行TS计算
            if self.ts_quick_mode:
                # 快速模式：仅估算活化能
                ea = self.ts_calculator.quick_estimate_barrier(
                    temp_xyz.name,
                    substrate_smiles,
                    use_ml=True
                )

                result = {
                    'activation_energy': ea,
                    'reaction_energy': None,
                    'ts_frequency': None,
                    'method': 'quick_estimate',
                    'success': True
                }
            else:
                # 完整TS计算
                result = self.ts_calculator.calculate_reaction_profile(
                    temp_xyz.name,
                    substrate_smiles,
                    reaction_params.get('product_smiles'),
                    reaction_params.get('charge', 0),
                    reaction_params.get('mult', 1)
                )

            # 转换活化能为评分 (0-1)
            if result.get('success', False):
                ea = result['activation_energy']

                # 评分规则：
                # Ea < 10 kcal/mol  -> 1.0 (极好)
                # Ea = 15 kcal/mol  -> 0.8 (好)
                # Ea = 20 kcal/mol  -> 0.5 (中等)
                # Ea = 25 kcal/mol  -> 0.2 (差)
                # Ea > 30 kcal/mol  -> 0.0 (很差)

                if ea < 10:
                    ts_score = 1.0
                elif ea < 30:
                    ts_score = max(0, 1.0 - (ea - 10) / 20.0)
                else:
                    ts_score = 0.0

                result['ts_score'] = ts_score

                logger.info(f"TS计算成功: Ea={ea:.2f} kcal/mol, score={ts_score:.3f}")
            else:
                result['ts_score'] = 0.0
                logger.warning(f"TS计算失败: {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            logger.error(f"TS计算异常: {e}")
            return {'success': False, 'error': str(e), 'ts_score': 0.0}

        finally:
            # 清理临时文件
            try:
                os.unlink(temp_xyz.name)
            except:
                pass

    def _predict_activity(self, total_score: float, ts_result: Optional[Dict] = None) -> Dict:
        """
        根据总分和TS结果预测活性等级

        Args:
            total_score: 总评分
            ts_result: TS计算结果（可选）

        Returns:
            {
                'level': 'high'/'medium'/'low'/'very_low',
                'confidence': 0.85,
                'description': '...',
                'ea_range': (min, max) kcal/mol (如果有TS)
            }
        """
        # 基础预测
        if total_score >= 0.8:
            level = 'high'
            base_confidence = 0.9
            description = '预测具有高催化活性，强烈推荐实验验证'
        elif total_score >= 0.6:
            level = 'medium'
            base_confidence = 0.75
            description = '预测具有中等催化活性，建议实验验证'
        elif total_score >= 0.4:
            level = 'low'
            base_confidence = 0.6
            description = '预测催化活性较低，可能需要优化'
        else:
            level = 'very_low'
            base_confidence = 0.5
            description = '预测催化活性很低，不推荐'

        result = {
            'level': level,
            'confidence': base_confidence,
            'description': description
        }

        # 如果有TS计算结果，提升置信度并添加详细信息
        if ts_result and ts_result.get('success', False):
            ea = ts_result.get('activation_energy')

            if ea is not None:
                # 提升置信度
                result['confidence'] = min(0.95, base_confidence + 0.15)

                # 添加活化能信息
                result['ea_range'] = (ea - 2, ea + 2)  # ±2 kcal/mol 误差范围

                # 更新描述
                if ea < 15:
                    result['description'] += f" (计算活化能: {ea:.1f} kcal/mol，优秀)"
                elif ea < 20:
                    result['description'] += f" (计算活化能: {ea:.1f} kcal/mol，良好)"
                elif ea < 25:
                    result['description'] += f" (计算活化能: {ea:.1f} kcal/mol，中等)"
                else:
                    result['description'] += f" (计算活化能: {ea:.1f} kcal/mol，较高)"

        return result

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
        if not scored_nanozymes:
            logger.warning("未生成任何评分结果，返回空列表")
            return []

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
        use_ts = result.get('ts_enabled') and 'ts_details' in result
        if use_ts:
            explanation += f"  NAC几何: {scores['nac_geometry']:.3f} (权重 40%)\n"
            explanation += f"  可及性: {scores['accessibility']:.3f} (权重 15%)\n"
            explanation += f"  协同性: {scores['synergy']:.3f} (权重 8%)\n"
            explanation += f"  稳定性: {scores['stability']:.3f} (权重 7%)\n"
            explanation += f"  TS计算: {scores.get('ts_calculation', 0.0):.3f} (权重 30%)\n"
        else:
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
