#!/usr/bin/env python3
"""
构象分析模块
包含：多构象采样、UMAP降维、K-means聚类、化学有效性验证
借鉴 StoL 的后处理流程
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """化学有效性验证结果"""
    is_valid: bool
    bond_length_violations: int
    clash_count: int
    min_distance: float
    max_distance: float
    geometry_score: float
    details: Dict


class ChemicalValidator:
    """
    化学有效性验证器

    检查纳米酶结构的化学合理性
    """

    def __init__(self,
                 min_bond_length: float = 0.8,
                 max_bond_length: float = 2.5,
                 clash_threshold: float = 0.8,
                 bond_tolerance: float = 0.2):
        """
        Args:
            min_bond_length: 最小键长（Å）
            max_bond_length: 最大键长（Å）
            clash_threshold: 原子冲突阈值（Å）
            bond_tolerance: 键长容差（Å）
        """
        self.min_bond_length = min_bond_length
        self.max_bond_length = max_bond_length
        self.clash_threshold = clash_threshold
        self.bond_tolerance = bond_tolerance

        # 标准共价半径（Å）
        self.covalent_radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66,
            'S': 1.05, 'P': 1.07, 'F': 0.57, 'Cl': 1.02,
            'Fe': 1.32, 'Cu': 1.32, 'Zn': 1.22, 'Au': 1.36,
            'Ag': 1.45, 'Pt': 1.36, 'Pd': 1.39
        }

    def validate_structure(self, nanozyme: Dict) -> ValidationResult:
        """
        验证纳米酶结构

        Args:
            nanozyme: 纳米酶结构字典
                - coords: [n_atoms, 3]
                - atom_types: List[str]

        Returns:
            ValidationResult
        """
        coords = nanozyme['coords']
        atom_types = nanozyme['atom_types']
        n_atoms = len(coords)

        # 1. 计算所有原子间距离
        distances = self._compute_pairwise_distances(coords)

        # 2. 检查原子冲突
        clash_count = self._count_clashes(distances)

        # 3. 检查键长
        bond_violations = self._check_bond_lengths(
            distances, atom_types
        )

        # 4. 计算几何分数
        geometry_score = self._compute_geometry_score(
            distances, atom_types, clash_count, bond_violations
        )

        # 5. 统计信息
        min_dist = np.min(distances[np.triu_indices(n_atoms, k=1)])
        max_dist = np.max(distances[np.triu_indices(n_atoms, k=1)])

        # 判断是否有效
        is_valid = (clash_count == 0 and
                   bond_violations < n_atoms * 0.1 and
                   geometry_score > 0.5)

        return ValidationResult(
            is_valid=is_valid,
            bond_length_violations=bond_violations,
            clash_count=clash_count,
            min_distance=float(min_dist),
            max_distance=float(max_dist),
            geometry_score=float(geometry_score),
            details={
                'n_atoms': n_atoms,
                'avg_distance': float(np.mean(distances[np.triu_indices(n_atoms, k=1)])),
                'std_distance': float(np.std(distances[np.triu_indices(n_atoms, k=1)]))
            }
        )

    def batch_validate(self, nanozymes: List[Dict]) -> Tuple[List[Dict], List[ValidationResult]]:
        """
        批量验证并过滤

        Args:
            nanozymes: 纳米酶列表

        Returns:
            (valid_nanozymes, validation_results)
        """
        logger.info(f"Validating {len(nanozymes)} structures...")

        valid_nanozymes = []
        all_results = []

        for i, nanozyme in enumerate(nanozymes):
            result = self.validate_structure(nanozyme)
            all_results.append(result)

            if result.is_valid:
                valid_nanozymes.append(nanozyme)

        valid_rate = len(valid_nanozymes) / len(nanozymes) if nanozymes else 0
        logger.info(f"Valid structures: {len(valid_nanozymes)}/{len(nanozymes)} ({valid_rate:.1%})")
        logger.info(f"  Avg clash count: {np.mean([r.clash_count for r in all_results]):.1f}")
        logger.info(f"  Avg geometry score: {np.mean([r.geometry_score for r in all_results]):.3f}")

        return valid_nanozymes, all_results

    def _compute_pairwise_distances(self, coords: np.ndarray) -> np.ndarray:
        """计算所有原子对的距离"""
        n = len(coords)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _count_clashes(self, distances: np.ndarray) -> int:
        """统计原子冲突数量"""
        n = distances.shape[0]
        clash_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                if distances[i, j] < self.clash_threshold:
                    clash_count += 1

        return clash_count

    def _check_bond_lengths(self, distances: np.ndarray,
                           atom_types: List[str]) -> int:
        """检查键长违规"""
        n = len(atom_types)
        violations = 0

        for i in range(n):
            for j in range(i + 1, n):
                dist = distances[i, j]

                # 判断是否应该成键
                expected_bond_length = self._expected_bond_length(
                    atom_types[i], atom_types[j]
                )

                if expected_bond_length is not None:
                    # 应该成键但距离不合理
                    if abs(dist - expected_bond_length) > self.bond_tolerance:
                        violations += 1

        return violations

    def _expected_bond_length(self, atom1: str, atom2: str) -> Optional[float]:
        """估计两个原子的期望键长"""
        r1 = self.covalent_radii.get(atom1, 1.0)
        r2 = self.covalent_radii.get(atom2, 1.0)

        # 简化：共价半径之和
        expected = r1 + r2

        # 只有在合理范围内才认为应该成键
        if self.min_bond_length <= expected <= self.max_bond_length:
            return expected
        return None

    def _compute_geometry_score(self, distances: np.ndarray,
                                atom_types: List[str],
                                clash_count: int,
                                bond_violations: int) -> float:
        """计算几何合理性分数（0-1）"""
        n = len(atom_types)

        # 基础分数
        score = 1.0

        # 冲突惩罚
        score -= clash_count * 0.1

        # 键长违规惩罚
        score -= bond_violations * 0.05

        # 距离分布惩罚（过于紧密或松散）
        triu_indices = np.triu_indices(n, k=1)
        avg_dist = np.mean(distances[triu_indices])

        if avg_dist < 2.0 or avg_dist > 10.0:
            score -= 0.2

        return max(0.0, min(1.0, score))


class ConformationAnalyzer:
    """
    构象分析器

    包含 UMAP 降维、K-means 聚类、代表性结构选择
    """

    def __init__(self, validator: ChemicalValidator = None):
        """
        Args:
            validator: 化学验证器
        """
        self.validator = validator or ChemicalValidator()

    def analyze_conformations(self,
                             nanozymes: List[Dict],
                             n_clusters: int = 10,
                             umap_n_neighbors: int = 15,
                             umap_min_dist: float = 0.1,
                             output_dir: Optional[str] = None) -> Dict:
        """
        完整的构象分析流程

        Args:
            nanozymes: 纳米酶结构列表
            n_clusters: 聚类数量
            umap_n_neighbors: UMAP 邻居数
            umap_min_dist: UMAP 最小距离
            output_dir: 输出目录（保存图表）

        Returns:
            分析结果字典
        """
        logger.info(f"Analyzing {len(nanozymes)} conformations")

        # 1. 化学有效性验证
        logger.info("Step 1: Chemical validation...")
        valid_nanozymes, validation_results = self.validator.batch_validate(nanozymes)

        if len(valid_nanozymes) == 0:
            logger.error("No valid structures found!")
            return {'valid_structures': [], 'error': 'No valid structures'}

        # 2. 特征提取
        logger.info("Step 2: Feature extraction...")
        features = self._extract_features(valid_nanozymes)

        # 3. UMAP 降维
        logger.info("Step 3: UMAP dimensionality reduction...")
        embeddings = self._umap_reduce(
            features,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist
        )

        # 4. K-means 聚类
        logger.info("Step 4: K-means clustering...")
        labels, centroids = self._kmeans_cluster(embeddings, n_clusters)

        # 5. 选择代表性结构
        logger.info("Step 5: Selecting representative structures...")
        representatives = self._select_representatives(
            valid_nanozymes, embeddings, labels, centroids
        )

        # 6. 可视化
        if output_dir:
            logger.info("Step 6: Generating visualizations...")
            self._visualize_results(
                embeddings, labels, centroids,
                validation_results[:len(valid_nanozymes)],
                output_dir
            )

        # 7. 统计信息
        stats = self._compute_statistics(
            valid_nanozymes, labels, validation_results
        )

        logger.info(f"Analysis completed: {len(representatives)} representative structures")

        return {
            'valid_structures': valid_nanozymes,
            'validation_results': validation_results,
            'embeddings': embeddings,
            'labels': labels,
            'centroids': centroids,
            'representatives': representatives,
            'statistics': stats
        }

    def _extract_features(self, nanozymes: List[Dict]) -> np.ndarray:
        """
        提取构象特征

        特征包括：
        1. 距离矩阵的上三角元素
        2. 原子类型分布
        3. 几何描述符（半径、偏心率等）
        """
        features_list = []

        for nanozyme in nanozymes:
            coords = nanozyme['coords']
            atom_types = nanozyme['atom_types']

            # 1. 距离矩阵特征（采样）
            n_atoms = len(coords)
            distances = []

            # 采样距离（避免特征维度过高）
            sample_size = min(50, n_atoms * (n_atoms - 1) // 2)
            indices = np.random.choice(
                n_atoms * (n_atoms - 1) // 2,
                size=sample_size,
                replace=False
            )

            triu_indices = np.triu_indices(n_atoms, k=1)
            for idx in indices:
                i, j = triu_indices[0][idx], triu_indices[1][idx]
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)

            # 2. 几何描述符
            center = np.mean(coords, axis=0)
            radii = np.linalg.norm(coords - center, axis=1)

            geometric_features = [
                np.mean(radii),      # 平均半径
                np.std(radii),       # 半径标准差
                np.max(radii),       # 最大半径
                np.min(radii),       # 最小半径
            ]

            # 3. 原子类型分布
            unique_types = set(atom_types)
            type_counts = {t: atom_types.count(t) / len(atom_types)
                          for t in unique_types}

            # 填充到固定维度（前10种元素）
            common_types = ['C', 'N', 'O', 'H', 'S', 'P', 'Fe', 'Cu', 'Zn', 'Au']
            type_features = [type_counts.get(t, 0.0) for t in common_types]

            # 合并特征
            feature_vector = distances + geometric_features + type_features
            features_list.append(feature_vector)

        return np.array(features_list)

    def _umap_reduce(self, features: np.ndarray,
                    n_neighbors: int = 15,
                    min_dist: float = 0.1,
                    n_components: int = 2) -> np.ndarray:
        """UMAP 降维"""
        try:
            import umap

            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=42
            )

            embeddings = reducer.fit_transform(features)
            logger.info(f"UMAP reduced to {n_components}D: {embeddings.shape}")

            return embeddings

        except ImportError:
            logger.warning("UMAP not installed, using PCA instead")
            from sklearn.decomposition import PCA

            pca = PCA(n_components=n_components, random_state=42)
            embeddings = pca.fit_transform(features)
            logger.info(f"PCA reduced to {n_components}D: {embeddings.shape}")

            return embeddings

    def _kmeans_cluster(self, embeddings: np.ndarray,
                       n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
        """K-means 聚类"""
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        # 统计每个簇的大小
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"Cluster sizes: {dict(zip(unique, counts))}")

        return labels, centroids

    def _select_representatives(self,
                               nanozymes: List[Dict],
                               embeddings: np.ndarray,
                               labels: np.ndarray,
                               centroids: np.ndarray) -> List[Dict]:
        """
        选择每个簇的代表性结构

        策略：选择距离质心最近的结构
        """
        representatives = []

        for cluster_id in range(len(centroids)):
            # 找到该簇的所有成员
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # 计算到质心的距离
            cluster_embeddings = embeddings[cluster_mask]
            distances_to_centroid = np.linalg.norm(
                cluster_embeddings - centroids[cluster_id],
                axis=1
            )

            # 选择最近的
            closest_idx = cluster_indices[np.argmin(distances_to_centroid)]
            representative = nanozymes[closest_idx].copy()
            representative['cluster_id'] = int(cluster_id)
            representative['cluster_size'] = int(len(cluster_indices))
            representative['distance_to_centroid'] = float(np.min(distances_to_centroid))

            representatives.append(representative)

        logger.info(f"Selected {len(representatives)} representative structures")
        return representatives

    def _visualize_results(self,
                          embeddings: np.ndarray,
                          labels: np.ndarray,
                          centroids: np.ndarray,
                          validation_results: List[ValidationResult],
                          output_dir: str):
        """生成可视化图表"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. UMAP + 聚类可视化
        plt.figure(figsize=(12, 5))

        # 子图1：按簇着色
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=50
        )
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c='red',
            marker='X',
            s=200,
            edgecolors='black',
            label='Centroids'
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('Conformation Clustering')
        plt.legend()

        # 子图2：按几何分数着色
        plt.subplot(1, 2, 2)
        geometry_scores = [r.geometry_score for r in validation_results]
        scatter = plt.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=geometry_scores,
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, label='Geometry Score')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('Geometry Quality')

        plt.tight_layout()
        plt.savefig(output_dir / 'conformation_analysis.png', dpi=300)
        plt.close()

        logger.info(f"Saved visualization to {output_dir / 'conformation_analysis.png'}")

        # 2. 验证统计图
        self._plot_validation_stats(validation_results, output_dir)

    def _plot_validation_stats(self,
                              validation_results: List[ValidationResult],
                              output_dir: Path):
        """绘制验证统计图"""
        plt.figure(figsize=(15, 4))

        # 子图1：几何分数分布
        plt.subplot(1, 3, 1)
        scores = [r.geometry_score for r in validation_results]
        plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Geometry Score')
        plt.ylabel('Count')
        plt.title('Geometry Score Distribution')
        plt.axvline(0.5, color='red', linestyle='--', label='Threshold')
        plt.legend()

        # 子图2：冲突数量
        plt.subplot(1, 3, 2)
        clashes = [r.clash_count for r in validation_results]
        plt.hist(clashes, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Clash Count')
        plt.ylabel('Count')
        plt.title('Atomic Clashes')

        # 子图3：最小距离
        plt.subplot(1, 3, 3)
        min_dists = [r.min_distance for r in validation_results]
        plt.hist(min_dists, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Min Distance (Å)')
        plt.ylabel('Count')
        plt.title('Minimum Atomic Distance')
        plt.axvline(0.8, color='red', linestyle='--', label='Clash threshold')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'validation_statistics.png', dpi=300)
        plt.close()

        logger.info(f"Saved validation stats to {output_dir / 'validation_statistics.png'}")

    def _compute_statistics(self,
                           nanozymes: List[Dict],
                           labels: np.ndarray,
                           validation_results: List[ValidationResult]) -> Dict:
        """计算统计信息"""
        valid_results = validation_results[:len(nanozymes)]

        stats = {
            'n_total': len(validation_results),
            'n_valid': len(nanozymes),
            'validity_rate': len(nanozymes) / len(validation_results) if validation_results else 0,
            'n_clusters': len(np.unique(labels)),
            'avg_geometry_score': float(np.mean([r.geometry_score for r in valid_results])),
            'avg_clash_count': float(np.mean([r.clash_count for r in valid_results])),
            'avg_min_distance': float(np.mean([r.min_distance for r in valid_results])),
            'cluster_sizes': {
                int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))
            }
        }

        return stats


def create_analyzer(validator: ChemicalValidator = None) -> ConformationAnalyzer:
    """创建构象分析器的便捷函数"""
    return ConformationAnalyzer(validator=validator)
