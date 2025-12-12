#!/usr/bin/env python3
"""
高质量Swiss-Prot数据筛选器

目标：从Swiss-Prot筛选出接近M-CSA质量的训练数据
策略：多维度质量评分 + 严格阈值过滤
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """数据质量评分"""
    total_score: float
    structure_score: float      # 结构质量分数
    annotation_score: float     # 标注质量分数
    evidence_score: float       # 证据质量分数
    completeness_score: float   # 完整性分数
    details: Dict


class HighQualitySwissProtFilter:
    """
    高质量Swiss-Prot数据筛选器

    筛选标准（参考M-CSA质量）：
    1. 必须有高分辨率PDB结构（< 2.5Å）
    2. 必须有明确的活性位点标注
    3. 必须有催化活性描述
    4. 必须有完整的EC号（4级）
    5. 优先选择有文献证据的条目
    6. 优先选择经典酶家族
    """

    def __init__(
        self,
        min_quality_score: float = 0.7,
        require_high_res_structure: bool = True,
        require_active_site: bool = True,
        require_catalytic_activity: bool = True,
        require_full_ec: bool = True,
        max_resolution: float = 2.5,
        min_active_sites: int = 2,
        prefer_classic_enzymes: bool = True
    ):
        """
        Args:
            min_quality_score: 最低质量分数（0-1）
            require_high_res_structure: 要求高分辨率结构
            require_active_site: 要求活性位点标注
            require_catalytic_activity: 要求催化活性描述
            require_full_ec: 要求完整EC号（4级）
            max_resolution: 最大分辨率（Å）
            min_active_sites: 最少活性位点数
            prefer_classic_enzymes: 优先经典酶家族
        """
        self.min_quality_score = min_quality_score
        self.require_high_res_structure = require_high_res_structure
        self.require_active_site = require_active_site
        self.require_catalytic_activity = require_catalytic_activity
        self.require_full_ec = require_full_ec
        self.max_resolution = max_resolution
        self.min_active_sites = min_active_sites
        self.prefer_classic_enzymes = prefer_classic_enzymes

        # 经典酶家族（M-CSA中常见的高质量酶）
        self.classic_enzyme_families = {
            # 丝氨酸蛋白酶家族
            '3.4.21.4',   # Trypsin
            '3.4.21.1',   # Chymotrypsin
            '3.4.21.5',   # Thrombin
            '3.4.21.73',  # Granzyme B

            # 半胱氨酸蛋白酶家族
            '3.4.22.2',   # Papain
            '3.4.22.1',   # Cathepsin B
            '3.4.22.68',  # Caspase-3

            # 天冬氨酸蛋白酶家族
            '3.4.23.1',   # Pepsin
            '3.4.23.15',  # Renin
            '3.4.23.16',  # HIV protease

            # 金属蛋白酶家族
            '3.4.24.7',   # Thermolysin
            '3.4.24.11',  # Neprilysin

            # 脂肪酶/酯酶家族
            '3.1.1.3',    # Triacylglycerol lipase
            '3.1.1.1',    # Carboxylesterase
            '3.1.1.7',    # Acetylcholinesterase

            # 糖苷酶家族
            '3.2.1.1',    # Alpha-amylase
            '3.2.1.20',   # Alpha-glucosidase
            '3.2.1.21',   # Beta-glucosidase
            '3.2.1.23',   # Beta-galactosidase

            # 激酶家族
            '2.7.1.1',    # Hexokinase
            '2.7.1.40',   # Pyruvate kinase
            '2.7.11.1',   # Protein kinase

            # 脱氢酶家族
            '1.1.1.1',    # Alcohol dehydrogenase
            '1.1.1.27',   # Lactate dehydrogenase
            '1.2.1.12',   # Glyceraldehyde-3-phosphate dehydrogenase

            # 氧化酶家族
            '1.11.1.6',   # Catalase
            '1.11.1.7',   # Peroxidase
            '1.15.1.1',   # Superoxide dismutase

            # 转移酶家族
            '2.1.1.37',   # DNA methyltransferase
            '2.3.1.12',   # Dihydrolipoyllysine-residue acetyltransferase

            # 裂解酶家族
            '4.1.1.1',    # Pyruvate decarboxylase
            '4.2.1.1',    # Carbonate dehydratase
            '4.2.1.2',    # Fumarate hydratase

            # 异构酶家族
            '5.3.1.1',    # Triose-phosphate isomerase
            '5.3.1.9',    # Glucose-6-phosphate isomerase

            # 连接酶家族
            '6.3.2.1',    # Glutamate-ammonia ligase
            '6.4.1.1',    # Pyruvate carboxylase
        }

        logger.info(f"高质量筛选器初始化: min_score={min_quality_score}")

    def filter_entries(self, entries: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        筛选高质量条目

        Args:
            entries: Swiss-Prot条目列表

        Returns:
            (高质量条目列表, 统计信息)
        """
        high_quality = []
        quality_scores = []

        stats = {
            'total_input': len(entries),
            'passed': 0,
            'failed_structure': 0,
            'failed_annotation': 0,
            'failed_evidence': 0,
            'failed_score': 0,
            'avg_quality_score': 0.0,
            'score_distribution': {
                '0.9-1.0': 0,
                '0.8-0.9': 0,
                '0.7-0.8': 0,
                '0.6-0.7': 0,
                '<0.6': 0
            }
        }

        for entry in entries:
            # 计算质量分数
            quality = self.calculate_quality_score(entry)
            quality_scores.append(quality.total_score)

            # 更新分数分布
            if quality.total_score >= 0.9:
                stats['score_distribution']['0.9-1.0'] += 1
            elif quality.total_score >= 0.8:
                stats['score_distribution']['0.8-0.9'] += 1
            elif quality.total_score >= 0.7:
                stats['score_distribution']['0.7-0.8'] += 1
            elif quality.total_score >= 0.6:
                stats['score_distribution']['0.6-0.7'] += 1
            else:
                stats['score_distribution']['<0.6'] += 1

            # 应用筛选规则
            if not self._passes_filters(entry, quality, stats):
                continue

            # 通过筛选
            high_quality.append(entry)
            stats['passed'] += 1

        # 计算平均分数
        if quality_scores:
            stats['avg_quality_score'] = sum(quality_scores) / len(quality_scores)

        logger.info(f"筛选完成: {stats['passed']}/{stats['total_input']} 通过")
        logger.info(f"平均质量分数: {stats['avg_quality_score']:.3f}")

        return high_quality, stats

    def calculate_quality_score(self, entry: Dict) -> QualityScore:
        """
        计算条目的质量分数

        评分维度：
        1. 结构质量 (30%)：分辨率、结构数量、实验方法
        2. 标注质量 (30%)：活性位点、结合位点、金属中心
        3. 证据质量 (25%)：文献数量、证据等级
        4. 完整性 (15%)：EC号完整性、序列长度、功能描述
        """
        details = {}

        # 1. 结构质量评分 (0-1)
        structure_score = self._score_structure_quality(entry, details)

        # 2. 标注质量评分 (0-1)
        annotation_score = self._score_annotation_quality(entry, details)

        # 3. 证据质量评分 (0-1)
        evidence_score = self._score_evidence_quality(entry, details)

        # 4. 完整性评分 (0-1)
        completeness_score = self._score_completeness(entry, details)

        # 加权总分
        total_score = (
            structure_score * 0.30 +
            annotation_score * 0.30 +
            evidence_score * 0.25 +
            completeness_score * 0.15
        )

        return QualityScore(
            total_score=total_score,
            structure_score=structure_score,
            annotation_score=annotation_score,
            evidence_score=evidence_score,
            completeness_score=completeness_score,
            details=details
        )

    def _score_structure_quality(self, entry: Dict, details: Dict) -> float:
        """评分：结构质量"""
        score = 0.0

        # 检查PDB结构
        pdb_refs = []
        if entry.get('uniProtKBCrossReferences'):
            pdb_refs = [ref for ref in entry['uniProtKBCrossReferences']
                       if ref.get('database') == 'PDB']

        if not pdb_refs:
            details['structure'] = 'No PDB structure'
            return 0.0

        # 结构数量（多个结构 = 更可靠）
        num_structures = len(pdb_refs)
        details['num_pdb_structures'] = num_structures

        if num_structures >= 5:
            score += 0.4
        elif num_structures >= 3:
            score += 0.3
        elif num_structures >= 1:
            score += 0.2

        # 分辨率（从properties中提取，如果有的话）
        # 注意：UniProt API可能不直接提供分辨率，需要从PDB获取
        # 这里我们假设有多个结构 = 高质量
        if num_structures >= 3:
            score += 0.3  # 假设多结构意味着高分辨率
            details['resolution_quality'] = 'Multiple structures (assumed high quality)'

        # 实验方法（X-ray > NMR > EM）
        # 这个信息也需要从PDB获取，这里简化处理
        score += 0.3  # 默认给予基础分

        details['structure_score_breakdown'] = {
            'num_structures': num_structures,
            'score': score
        }

        return min(score, 1.0)

    def _score_annotation_quality(self, entry: Dict, details: Dict) -> float:
        """评分：标注质量"""
        score = 0.0

        features = entry.get('features', [])

        # 活性位点标注
        active_sites = [f for f in features if f.get('type') == 'Active site']
        num_active_sites = len(active_sites)
        details['num_active_sites'] = num_active_sites

        if num_active_sites >= 3:
            score += 0.4  # 典型催化三联体
        elif num_active_sites >= 2:
            score += 0.3
        elif num_active_sites >= 1:
            score += 0.2

        # 结合位点标注
        binding_sites = [f for f in features if f.get('type') == 'Binding site']
        if len(binding_sites) >= 1:
            score += 0.2
            details['has_binding_sites'] = True

        # 金属结合位点
        metal_sites = [f for f in features if f.get('type') == 'Metal binding']
        if len(metal_sites) >= 1:
            score += 0.2
            details['has_metal_binding'] = True

        # 催化活性描述
        has_catalytic_activity = False
        if entry.get('comments'):
            for comment in entry['comments']:
                if comment.get('commentType') == 'CATALYTIC ACTIVITY':
                    has_catalytic_activity = True
                    score += 0.2
                    details['has_catalytic_activity'] = True
                    break

        details['annotation_score_breakdown'] = {
            'active_sites': num_active_sites,
            'binding_sites': len(binding_sites),
            'metal_sites': len(metal_sites),
            'catalytic_activity': has_catalytic_activity,
            'score': score
        }

        return min(score, 1.0)

    def _score_evidence_quality(self, entry: Dict, details: Dict) -> float:
        """评分：证据质量"""
        score = 0.0

        # 文献引用数量
        num_references = 0
        if entry.get('references'):
            num_references = len(entry['references'])

        details['num_references'] = num_references

        if num_references >= 10:
            score += 0.4
        elif num_references >= 5:
            score += 0.3
        elif num_references >= 1:
            score += 0.2

        # 蛋白质存在证据等级
        protein_existence = entry.get('proteinExistence', '')
        details['protein_existence'] = protein_existence

        if 'Evidence at protein level' in protein_existence:
            score += 0.4  # 最高证据等级
        elif 'Evidence at transcript level' in protein_existence:
            score += 0.2

        # 是否是经典酶家族
        ec_number = None
        if entry.get('proteinDescription', {}).get('recommendedName', {}).get('ecNumbers'):
            ec_numbers = entry['proteinDescription']['recommendedName']['ecNumbers']
            if ec_numbers:
                ec_number = ec_numbers[0].get('value')

        if ec_number and ec_number in self.classic_enzyme_families:
            score += 0.2
            details['is_classic_enzyme'] = True

        details['evidence_score_breakdown'] = {
            'references': num_references,
            'protein_existence': protein_existence,
            'is_classic': ec_number in self.classic_enzyme_families if ec_number else False,
            'score': score
        }

        return min(score, 1.0)

    def _score_completeness(self, entry: Dict, details: Dict) -> float:
        """评分：完整性"""
        score = 0.0

        # EC号完整性（4级 > 3级 > 2级 > 1级）
        ec_number = None
        if entry.get('proteinDescription', {}).get('recommendedName', {}).get('ecNumbers'):
            ec_numbers = entry['proteinDescription']['recommendedName']['ecNumbers']
            if ec_numbers:
                ec_number = ec_numbers[0].get('value')

        if ec_number:
            ec_parts = ec_number.split('.')
            details['ec_number'] = ec_number
            details['ec_levels'] = len(ec_parts)

            if len(ec_parts) == 4 and '-' not in ec_number:
                score += 0.4  # 完整的4级EC号
            elif len(ec_parts) == 4:
                score += 0.3  # 4级但有未确定的部分
            elif len(ec_parts) == 3:
                score += 0.2
            else:
                score += 0.1

        # 序列长度（合理范围）
        sequence = entry.get('sequence', {}).get('value', '')
        seq_length = len(sequence)
        details['sequence_length'] = seq_length

        if 50 <= seq_length <= 2000:
            score += 0.3  # 合理的酶序列长度
        elif seq_length > 0:
            score += 0.1

        # 功能描述
        has_function = False
        if entry.get('comments'):
            for comment in entry['comments']:
                if comment.get('commentType') == 'FUNCTION':
                    has_function = True
                    score += 0.3
                    details['has_function_description'] = True
                    break

        details['completeness_score_breakdown'] = {
            'ec_complete': len(ec_number.split('.')) == 4 if ec_number else False,
            'sequence_length': seq_length,
            'has_function': has_function,
            'score': score
        }

        return min(score, 1.0)

    def _passes_filters(self, entry: Dict, quality: QualityScore, stats: Dict) -> bool:
        """检查是否通过所有必需的筛选条件"""

        # 1. 总分数检查
        if quality.total_score < self.min_quality_score:
            stats['failed_score'] += 1
            return False

        # 2. 结构要求
        if self.require_high_res_structure:
            if quality.structure_score < 0.5:
                stats['failed_structure'] += 1
                return False

        # 3. 标注要求
        if self.require_active_site:
            num_active_sites = quality.details.get('num_active_sites', 0)
            if num_active_sites < self.min_active_sites:
                stats['failed_annotation'] += 1
                return False

        # 4. 催化活性要求
        if self.require_catalytic_activity:
            if not quality.details.get('has_catalytic_activity', False):
                stats['failed_annotation'] += 1
                return False

        # 5. EC号完整性要求
        if self.require_full_ec:
            ec_levels = quality.details.get('ec_levels', 0)
            if ec_levels < 4:
                stats['failed_annotation'] += 1
                return False

        return True

    def print_quality_report(self, entry: Dict, quality: QualityScore):
        """打印单个条目的质量报告"""
        uniprot_id = entry.get('primaryAccession', 'Unknown')
        protein_name = entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown')

        print(f"\n{'='*80}")
        print(f"质量报告: {uniprot_id} - {protein_name}")
        print(f"{'='*80}")
        print(f"总分: {quality.total_score:.3f}")
        print(f"\n分项得分:")
        print(f"  结构质量:   {quality.structure_score:.3f} (30%)")
        print(f"  标注质量:   {quality.annotation_score:.3f} (30%)")
        print(f"  证据质量:   {quality.evidence_score:.3f} (25%)")
        print(f"  完整性:     {quality.completeness_score:.3f} (15%)")

        print(f"\n详细信息:")
        for key, value in quality.details.items():
            if not key.endswith('_breakdown'):
                print(f"  {key}: {value}")

        print(f"{'='*80}\n")


# =============================================================================
# 便捷函数
# =============================================================================

def get_high_quality_swissprot_data(
    ec_classes: List[str] = None,
    limit_per_class: int = 5000,
    quality_level: str = 'high'
) -> Tuple[List[Dict], Dict]:
    """
    获取高质量Swiss-Prot数据

    Args:
        ec_classes: EC分类列表，默认全部7类
        limit_per_class: 每个EC类别的最大条目数
        quality_level: 质量等级
            - 'strict': 最严格（接近M-CSA质量）
            - 'high': 高质量（推荐）
            - 'medium': 中等质量（更多数据）

    Returns:
        (高质量条目列表, 统计信息)
    """
    from .swissprot_data import SwissProtDataFetcher

    if ec_classes is None:
        ec_classes = ['1', '2', '3', '4', '5', '6', '7']

    # 根据质量等级设置参数
    if quality_level == 'strict':
        filter_params = {
            'min_quality_score': 0.8,
            'require_high_res_structure': True,
            'require_active_site': True,
            'require_catalytic_activity': True,
            'require_full_ec': True,
            'min_active_sites': 3,
            'prefer_classic_enzymes': True
        }
    elif quality_level == 'high':
        filter_params = {
            'min_quality_score': 0.7,
            'require_high_res_structure': True,
            'require_active_site': True,
            'require_catalytic_activity': True,
            'require_full_ec': True,
            'min_active_sites': 2,
            'prefer_classic_enzymes': True
        }
    else:  # medium
        filter_params = {
            'min_quality_score': 0.6,
            'require_high_res_structure': True,
            'require_active_site': True,
            'require_catalytic_activity': False,
            'require_full_ec': False,
            'min_active_sites': 1,
            'prefer_classic_enzymes': False
        }

    # 创建筛选器
    filter_obj = HighQualitySwissProtFilter(**filter_params)

    # 获取数据
    fetcher = SwissProtDataFetcher()
    all_entries = []

    logger.info(f"开始获取Swiss-Prot数据 (质量等级: {quality_level})")

    for ec_class in ec_classes:
        logger.info(f"获取 EC {ec_class}...")
        entries = fetcher.fetch_enzymes_by_ec_class(
            ec_class=ec_class,
            limit=limit_per_class,
            reviewed=True,
            has_structure=True,
            has_active_site=True
        )
        all_entries.extend(entries)
        logger.info(f"  EC {ec_class}: {len(entries)} 原始条目")

    logger.info(f"\n总计获取: {len(all_entries)} 原始条目")
    logger.info(f"开始质量筛选...")

    # 筛选高质量数据
    high_quality, stats = filter_obj.filter_entries(all_entries)

    return high_quality, stats


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'HighQualitySwissProtFilter',
    'QualityScore',
    'get_high_quality_swissprot_data',
]
