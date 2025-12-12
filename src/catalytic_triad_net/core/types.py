"""
类型定义模块

提供清晰的类型注解，提高代码可读性和类型安全性。
"""

from typing import TypedDict, Optional, List, Dict, Tuple, Union, Any
from pathlib import Path
import numpy as np
import torch


# ============================================================================
# 基础类型别名
# ============================================================================

PathLike = Union[str, Path]
TensorLike = Union[np.ndarray, torch.Tensor]
DeviceType = Union[str, torch.device]


# ============================================================================
# 结构相关类型
# ============================================================================

class ResidueDict(TypedDict):
    """残基信息字典"""
    chain_id: str
    residue_number: int
    residue_name: str
    coordinates: np.ndarray  # shape: (N_atoms, 3)
    atom_names: List[str]
    elements: List[str]
    b_factors: List[float]


class StructureDict(TypedDict):
    """蛋白质结构字典"""
    pdb_id: str
    residues: List[ResidueDict]
    sequence: str
    chains: List[str]


class EncodedStructure(TypedDict):
    """编码后的结构"""
    node_features: np.ndarray  # shape: (N_residues, feature_dim)
    edge_index: np.ndarray  # shape: (2, N_edges)
    edge_features: np.ndarray  # shape: (N_edges, edge_feature_dim)
    coordinates: np.ndarray  # shape: (N_residues, 3)
    residue_names: List[str]
    residue_indices: List[int]


# ============================================================================
# 数据集相关类型
# ============================================================================

class CatalyticResidue(TypedDict):
    """催化残基信息"""
    chain_id: str
    residue_number: int
    residue_name: str
    role: str  # 'nucleophile', 'acid_base', 'electrostatic', etc.
    evidence: str


class EnzymeEntry(TypedDict):
    """酶条目信息"""
    pdb_id: str
    ec_number: str
    enzyme_name: str
    catalytic_residues: List[CatalyticResidue]
    mechanism: Optional[str]
    literature_refs: List[str]


class DatasetSample(TypedDict):
    """数据集样本"""
    pdb_id: str
    node_features: torch.Tensor  # shape: (N_residues, feature_dim)
    edge_index: torch.Tensor  # shape: (2, N_edges)
    edge_features: torch.Tensor  # shape: (N_edges, edge_feature_dim)
    coordinates: torch.Tensor  # shape: (N_residues, 3)
    catalytic_labels: torch.Tensor  # shape: (N_residues,)
    role_labels: torch.Tensor  # shape: (N_residues,)
    ec_labels: torch.Tensor  # shape: (4,)  # EC1, EC2, EC3, EC4


class BatchData(TypedDict):
    """批次数据"""
    node_features: torch.Tensor  # shape: (total_nodes, feature_dim)
    edge_index: torch.Tensor  # shape: (2, total_edges)
    edge_features: torch.Tensor  # shape: (total_edges, edge_feature_dim)
    coordinates: torch.Tensor  # shape: (total_nodes, 3)
    catalytic_labels: torch.Tensor  # shape: (total_nodes,)
    role_labels: torch.Tensor  # shape: (total_nodes,)
    ec_labels: torch.Tensor  # shape: (batch_size, 4)
    batch: torch.Tensor  # shape: (total_nodes,) - 节点到样本的映射
    ptr: torch.Tensor  # shape: (batch_size + 1,) - 累积节点数


# ============================================================================
# 预测相关类型
# ============================================================================

class CatalyticSitePrediction(TypedDict):
    """催化位点预测结果"""
    chain_id: str
    residue_number: int
    residue_name: str
    catalytic_score: float
    role_prediction: str
    role_confidence: float


class TriadPrediction(TypedDict):
    """催化三联体预测"""
    residues: List[CatalyticSitePrediction]
    triad_score: float
    geometry_score: float
    triad_type: str  # 'Ser-His-Asp', 'Cys-His-Asp', etc.


class PredictionResult(TypedDict):
    """完整预测结果"""
    pdb_id: str
    ec_prediction: Tuple[int, int, int, int]
    ec_confidence: float
    catalytic_residues: List[CatalyticSitePrediction]
    triads: List[TriadPrediction]
    bimetallic_centers: List[Dict[str, Any]]
    hydrogen_bonds: List[Dict[str, Any]]


# ============================================================================
# 训练相关类型
# ============================================================================

class TrainingMetrics(TypedDict):
    """训练指标"""
    loss: float
    catalytic_loss: float
    role_loss: float
    ec_loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class EpochResult(TypedDict):
    """单个epoch结果"""
    epoch: int
    train_metrics: TrainingMetrics
    val_metrics: TrainingMetrics
    learning_rate: float
    time_elapsed: float


class CheckpointDict(TypedDict):
    """模型检查点"""
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
    best_f1: float
    config: Dict[str, Any]


# ============================================================================
# 生成相关类型
# ============================================================================

class FunctionalGroup(TypedDict):
    """催化功能团"""
    source_pdb: str
    residue_name: str
    residue_number: int
    chain_id: str
    group_type: str  # 'imidazole', 'carboxyl', 'hydroxyl', etc.
    coordinates: np.ndarray  # shape: (N_atoms, 3)
    elements: List[str]
    atom_names: List[str]
    catalytic_role: str


class GeometricConstraint(TypedDict):
    """几何约束"""
    constraint_type: str  # 'distance', 'angle', 'dihedral'
    atom_indices: List[int]
    target_value: float
    tolerance: float
    weight: float


class NanozymeStructure(TypedDict):
    """纳米酶结构"""
    nanozyme_id: str
    functional_groups: List[FunctionalGroup]
    scaffold_type: str  # 'carbon_chain', 'aromatic', 'metal_framework'
    coordinates: np.ndarray  # shape: (N_atoms, 3)
    elements: List[str]
    bonds: List[Tuple[int, int]]
    constraints: List[GeometricConstraint]


class ScoringResult(TypedDict):
    """打分结果"""
    nanozyme_id: str
    total_score: float
    nac_score: float
    accessibility_score: float
    synergy_score: float
    stability_score: float
    activation_energy: Optional[float]
    reaction_energy: Optional[float]
    details: Dict[str, Any]


# ============================================================================
# 配置相关类型
# ============================================================================

class ModelConfig(TypedDict, total=False):
    """模型配置"""
    node_dim: int
    edge_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    activation: str


class TrainingConfig(TypedDict, total=False):
    """训练配置"""
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    min_delta: float
    gradient_clip: float


class DataConfig(TypedDict, total=False):
    """数据配置"""
    cache_dir: str
    max_residues: int
    edge_cutoff: float
    num_workers: int
    prefetch_factor: int


class GenerationConfig(TypedDict, total=False):
    """生成配置"""
    num_diffusion_steps: int
    beta_start: float
    beta_end: float
    guidance_scale: float
    temperature: float
    top_k: int


# ============================================================================
# 导出相关类型
# ============================================================================

class ExportFormat(TypedDict):
    """导出格式配置"""
    format: str  # 'pdb', 'mol2', 'sdf', 'json', 'csv'
    include_metadata: bool
    compress: bool


class VisualizationConfig(TypedDict):
    """可视化配置"""
    mode: str  # '2d', '3d', 'interactive'
    color_scheme: str
    show_labels: bool
    highlight_catalytic: bool
    output_format: str  # 'png', 'svg', 'pdf', 'html'


# ============================================================================
# 工具类型
# ============================================================================

class ProgressCallback(TypedDict):
    """进度回调信息"""
    current: int
    total: int
    percentage: float
    message: str
    eta_seconds: Optional[float]
