"""
Constants and properties for amino acids and catalytic residues.

This module contains biochemical properties, catalytic priors, and other
constants used throughout the codebase.
"""

import numpy as np
from typing import Dict, List

# Amino acid one-letter codes
AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

# Amino acid properties
# Each amino acid is represented by a vector of biochemical properties:
# [hydrophobicity, charge, polarity, aromaticity, size, flexibility,
#  hydrogen_bond_donor, hydrogen_bond_acceptor]
AA_PROPERTIES: Dict[str, np.ndarray] = {
    # Hydrophobic amino acids
    'A': np.array([0.62, 0.0, 0.0, 0.0, 0.3, 0.8, 0.0, 0.0]),  # Alanine: small, hydrophobic
    'V': np.array([0.76, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0]),  # Valine: branched, hydrophobic
    'L': np.array([0.76, 0.0, 0.0, 0.0, 0.6, 0.5, 0.0, 0.0]),  # Leucine: branched, hydrophobic
    'I': np.array([0.76, 0.0, 0.0, 0.0, 0.6, 0.3, 0.0, 0.0]),  # Isoleucine: branched, hydrophobic
    'M': np.array([0.64, 0.0, 0.0, 0.0, 0.7, 0.6, 0.0, 0.0]),  # Methionine: sulfur-containing
    'F': np.array([0.88, 0.0, 0.0, 1.0, 0.8, 0.4, 0.0, 0.0]),  # Phenylalanine: aromatic, hydrophobic
    'W': np.array([0.88, 0.0, 0.0, 1.0, 1.0, 0.3, 1.0, 0.0]),  # Tryptophan: aromatic, large
    'P': np.array([0.36, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0]),  # Proline: rigid, helix breaker

    # Polar uncharged amino acids
    'S': np.array([0.18, 0.0, 1.0, 0.0, 0.3, 0.7, 1.0, 1.0]),  # Serine: hydroxyl group
    'T': np.array([0.18, 0.0, 1.0, 0.0, 0.4, 0.6, 1.0, 1.0]),  # Threonine: hydroxyl group
    'C': np.array([0.29, 0.0, 1.0, 0.0, 0.4, 0.6, 0.0, 0.0]),  # Cysteine: thiol group, disulfide bonds
    'Y': np.array([0.41, 0.0, 1.0, 1.0, 0.9, 0.4, 1.0, 1.0]),  # Tyrosine: aromatic, hydroxyl
    'N': np.array([0.09, 0.0, 1.0, 0.0, 0.5, 0.7, 1.0, 1.0]),  # Asparagine: amide group
    'Q': np.array([0.09, 0.0, 1.0, 0.0, 0.6, 0.7, 1.0, 1.0]),  # Glutamine: amide group
    'G': np.array([0.48, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # Glycine: smallest, flexible

    # Positively charged amino acids
    'K': np.array([0.05, 1.0, 1.0, 0.0, 0.7, 0.8, 1.0, 0.0]),  # Lysine: basic, long side chain
    'R': np.array([0.05, 1.0, 1.0, 0.0, 0.8, 0.7, 1.0, 0.0]),  # Arginine: basic, guanidinium group
    'H': np.array([0.13, 0.5, 1.0, 1.0, 0.6, 0.6, 1.0, 1.0]),  # Histidine: aromatic, can be charged

    # Negatively charged amino acids
    'D': np.array([0.05, -1.0, 1.0, 0.0, 0.4, 0.7, 0.0, 1.0]),  # Aspartate: acidic, short
    'E': np.array([0.05, -1.0, 1.0, 0.0, 0.6, 0.7, 0.0, 1.0]),  # Glutamate: acidic, longer
}

# Catalytic residue prior probabilities
# Based on statistical analysis of known catalytic sites
# Higher values indicate residues more likely to be catalytic
CATALYTIC_PRIOR: Dict[str, float] = {
    # Nucleophiles (high catalytic potential)
    'S': 0.85,  # Serine: common nucleophile in serine proteases, esterases
    'C': 0.90,  # Cysteine: strong nucleophile in cysteine proteases, thiol enzymes
    'D': 0.80,  # Aspartate: general acid/base, metal coordination
    'E': 0.75,  # Glutamate: general acid/base, metal coordination
    'H': 0.95,  # Histidine: most versatile, general acid/base, metal coordination
    'K': 0.70,  # Lysine: general base, Schiff base formation

    # Moderate catalytic potential
    'Y': 0.60,  # Tyrosine: can act as nucleophile or acid/base
    'T': 0.55,  # Threonine: similar to serine but less common
    'R': 0.50,  # Arginine: substrate binding, stabilization
    'N': 0.45,  # Asparagine: hydrogen bonding, oxyanion hole
    'Q': 0.40,  # Glutamine: hydrogen bonding, oxyanion hole
    'W': 0.35,  # Tryptophan: substrate binding, π-stacking

    # Low catalytic potential (mostly structural)
    'M': 0.25,  # Methionine: occasionally in active sites
    'F': 0.20,  # Phenylalanine: substrate binding, hydrophobic interactions
    'G': 0.15,  # Glycine: flexibility, tight turns
    'A': 0.10,  # Alanine: small, rarely catalytic
    'V': 0.08,  # Valine: hydrophobic core
    'L': 0.08,  # Leucine: hydrophobic core
    'I': 0.08,  # Isoleucine: hydrophobic core
    'P': 0.05,  # Proline: structural, rarely in active sites
}

# Common catalytic triads and dyads
KNOWN_CATALYTIC_MOTIFS: List[tuple] = [
    ('S', 'H', 'D'),  # Serine protease triad (e.g., chymotrypsin)
    ('S', 'H', 'E'),  # Alternative serine protease triad
    ('C', 'H', 'D'),  # Cysteine protease triad (e.g., papain)
    ('C', 'H', 'N'),  # Alternative cysteine protease triad
    ('D', 'H', 'S'),  # Aspartyl protease with histidine
    ('D', 'D'),       # Aspartyl protease dyad (e.g., pepsin)
    ('E', 'E'),       # Glutamate dyad (e.g., some metalloproteases)
    ('H', 'E'),       # Histidine-glutamate dyad
    ('K', 'E'),       # Lysine-glutamate dyad (e.g., some kinases)
]

# Metal coordination residues
METAL_COORDINATING_RESIDUES = ['H', 'D', 'E', 'C']

# Residues capable of forming covalent intermediates
COVALENT_CATALYSIS_RESIDUES = ['S', 'C', 'T', 'K', 'Y']

# Residues commonly in oxyanion holes
OXYANION_HOLE_RESIDUES = ['N', 'G', 'S', 'T']

# Standard amino acid masses (Da)
AA_MASSES: Dict[str, float] = {
    'A': 89.09, 'C': 121.15, 'D': 133.10, 'E': 147.13, 'F': 165.19,
    'G': 75.07, 'H': 155.15, 'I': 131.17, 'K': 146.19, 'L': 131.17,
    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19,
}

# Van der Waals radii (Angstroms)
VDW_RADII: Dict[str, float] = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
    'P': 1.80, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
}

# Atom types for structure parsing
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']
SIDECHAIN_ATOMS_BY_AA: Dict[str, List[str]] = {
    'A': ['CB'],
    'C': ['CB', 'SG'],
    'D': ['CB', 'CG', 'OD1', 'OD2'],
    'E': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
    'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'G': [],
    'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'I': ['CB', 'CG1', 'CG2', 'CD1'],
    'K': ['CB', 'CG', 'CD', 'CE', 'NZ'],
    'L': ['CB', 'CG', 'CD1', 'CD2'],
    'M': ['CB', 'CG', 'SD', 'CE'],
    'N': ['CB', 'CG', 'OD1', 'ND2'],
    'P': ['CB', 'CG', 'CD'],
    'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
    'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'S': ['CB', 'OG'],
    'T': ['CB', 'OG1', 'CG2'],
    'V': ['CB', 'CG1', 'CG2'],
    'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
}

# Non-standard residue mappings to standard amino acids
NON_STANDARD_AA_MAP: Dict[str, str] = {
    'MSE': 'M',  # Selenomethionine -> Methionine
    'SEP': 'S',  # Phosphoserine -> Serine
    'TPO': 'T',  # Phosphothreonine -> Threonine
    'PTR': 'Y',  # Phosphotyrosine -> Tyrosine
    'HYP': 'P',  # Hydroxyproline -> Proline
    'MLY': 'K',  # N-dimethyl-lysine -> Lysine
    'CSO': 'C',  # S-hydroxycysteine -> Cysteine
    'CSD': 'C',  # 3-sulfinoalanine -> Cysteine
    'CME': 'C',  # S,S-(2-hydroxyethyl)thiocysteine -> Cysteine
}

# Secondary structure codes
SECONDARY_STRUCTURE_CODES = {
    'H': 'helix',
    'E': 'sheet',
    'C': 'coil',
    'T': 'turn',
    'B': 'bridge',
    'G': '310_helix',
    'I': 'pi_helix',
}

# Distance thresholds for interactions (Angstroms)
HYDROGEN_BOND_DISTANCE = 3.5
SALT_BRIDGE_DISTANCE = 4.0
DISULFIDE_BOND_DISTANCE = 2.5
PI_STACKING_DISTANCE = 5.5
HYDROPHOBIC_CONTACT_DISTANCE = 5.0

# Default edge cutoff for graph construction (Angstroms)
DEFAULT_EDGE_CUTOFF = 8.0

# Maximum number of residues for efficient processing
MAX_RESIDUES_DEFAULT = 1000

# Atom type encoding for generation
ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'H', 'F', 'Cl', 'Br', 'I']
ATOM_TYPE_TO_INDEX = {atom: i for i, atom in enumerate(ATOM_TYPES)}
INDEX_TO_ATOM_TYPE = {i: atom for i, atom in enumerate(ATOM_TYPES)}

# Bond types
BOND_TYPES = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
BOND_TYPE_TO_INDEX = {bond: i for i, bond in enumerate(BOND_TYPES)}
INDEX_TO_BOND_TYPE = {i: bond for i, bond in enumerate(BOND_TYPES)}


# ============================================================================
# Structure Processing Constants
# ============================================================================

class StructureConstants:
    """结构处理相关常量"""

    # 原子碰撞检测
    MIN_CLASH_DISTANCE = 0.8  # Angstroms

    # 化学键距离阈值
    BOND_DISTANCE_THRESHOLD = 2.5  # Angstroms
    MAX_BOND_DISTANCE = 1.8  # Angstroms

    # 局部密度计算半径
    LOCAL_DENSITY_RADIUS_1 = 8.0  # Angstroms
    LOCAL_DENSITY_RADIUS_2 = 12.0  # Angstroms
    LOCAL_DENSITY_RADIUS_3 = 20.0  # Angstroms

    # 溶剂可及性
    SOLVENT_RADIUS = 1.4  # Angstroms (水分子半径)

    # 二级结构距离阈值
    HELIX_CA_DISTANCE = 5.5  # Angstroms
    SHEET_CA_DISTANCE = 6.5  # Angstroms


# ============================================================================
# Network and API Constants
# ============================================================================

class NetworkConstants:
    """网络请求相关常量"""

    # 超时设置（秒）
    DEFAULT_TIMEOUT = 30
    DOWNLOAD_TIMEOUT = 60
    API_TIMEOUT = 15

    # 重试设置
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 2.0  # 指数退避因子
    RETRY_INITIAL_DELAY = 1.0  # 初始延迟（秒）

    # 速率限制
    RATE_LIMIT_DELAY = 0.5  # 请求间隔（秒）

    # API URLs
    MCSA_API_BASE = "https://www.ebi.ac.uk/thornton-srv/m-csa/api"
    PDB_DOWNLOAD_BASE = "https://files.rcsb.org/download"
    UNIPROT_API_BASE = "https://rest.uniprot.org"


# ============================================================================
# Model Architecture Constants
# ============================================================================

class ModelConstants:
    """模型架构相关常量"""

    # 特征维度
    NODE_FEATURE_DIM = 48
    EDGE_FEATURE_DIM = 14
    DEFAULT_HIDDEN_DIM = 256

    # 网络层数
    DEFAULT_NUM_GNN_LAYERS = 6
    DEFAULT_NUM_ATTENTION_HEADS = 8

    # FFN扩展因子
    FFN_EXPANSION_FACTOR = 4

    # Dropout
    DEFAULT_DROPOUT = 0.2
    ATTENTION_DROPOUT = 0.1

    # 激活函数
    DEFAULT_ACTIVATION = 'relu'

    # 池化
    GLOBAL_POOL_METHOD = 'mean'  # 'mean', 'max', 'sum'


# ============================================================================
# Training Constants
# ============================================================================

class TrainingConstants:
    """训练相关常量"""

    # 学习率
    DEFAULT_LEARNING_RATE = 1e-4
    MIN_LEARNING_RATE = 1e-6

    # 优化器
    DEFAULT_WEIGHT_DECAY = 1e-5
    DEFAULT_GRADIENT_CLIP = 1.0

    # 批次大小
    DEFAULT_BATCH_SIZE = 4
    MAX_BATCH_SIZE = 32

    # 训练轮数
    DEFAULT_EPOCHS = 100
    PRETRAIN_EPOCHS = 20

    # 早停
    DEFAULT_PATIENCE = 15
    MIN_DELTA = 1e-4

    # 损失权重
    CATALYTIC_LOSS_WEIGHT = 1.0
    ROLE_LOSS_WEIGHT = 0.5
    EC_LOSS_WEIGHT = 0.3

    # Focal Loss参数
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # 日志频率
    LOG_INTERVAL = 10  # 每10个batch记录一次
    EVAL_INTERVAL = 1  # 每1个epoch评估一次


# ============================================================================
# Generation Constants
# ============================================================================

class GenerationConstants:
    """生成相关常量"""

    # 扩散模型
    NUM_DIFFUSION_STEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02

    # 采样
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_TEMPERATURE = 1.0
    SAMPLING_LOG_INTERVAL = 100  # 每100步记录一次

    # 约束权重
    DISTANCE_CONSTRAINT_WEIGHT = 1.0
    ANGLE_CONSTRAINT_WEIGHT = 0.5
    DIHEDRAL_CONSTRAINT_WEIGHT = 0.3

    # 边构建
    FULLY_CONNECTED_EDGE_CUTOFF = 10.0  # Angstroms


# ============================================================================
# Scoring Constants
# ============================================================================

class ScoringConstants:
    """打分相关常量"""

    # 阶段1打分权重
    STAGE1_FUNCTIONAL_GROUP_WEIGHT = 0.40
    STAGE1_CATALYTIC_ROLE_WEIGHT = 0.30
    STAGE1_DISTANCE_WEIGHT = 0.20
    STAGE1_PROBABILITY_WEIGHT = 0.10

    # 阶段2打分权重
    STAGE2_NAC_WEIGHT = 0.60
    STAGE2_ACCESSIBILITY_WEIGHT = 0.20
    STAGE2_SYNERGY_WEIGHT = 0.10
    STAGE2_STABILITY_WEIGHT = 0.10

    # NAC几何条件（过氧化物酶）
    PEROXIDASE_METAL_SUBSTRATE_MIN = 2.0  # Angstroms
    PEROXIDASE_METAL_SUBSTRATE_MAX = 2.8  # Angstroms
    PEROXIDASE_H2O2_BINDING_MIN = 2.5  # Angstroms
    PEROXIDASE_H2O2_BINDING_MAX = 3.5  # Angstroms
    PEROXIDASE_ELECTRON_TRANSFER_MIN = 3.0  # Angstroms
    PEROXIDASE_ELECTRON_TRANSFER_MAX = 4.5  # Angstroms
    PEROXIDASE_OXIDATION_ACCESSIBILITY = 3.0  # Angstroms

    # NAC几何条件（磷酸酶）
    PHOSPHATASE_METAL_PHOSPHATE_MIN = 2.0  # Angstroms
    PHOSPHATASE_METAL_PHOSPHATE_MAX = 2.5  # Angstroms
    PHOSPHATASE_NUCLEOPHILE_DISTANCE_MIN = 3.0  # Angstroms
    PHOSPHATASE_NUCLEOPHILE_DISTANCE_MAX = 4.0  # Angstroms

    # 活化能阈值
    ACTIVATION_ENERGY_EXCELLENT = 15.0  # kcal/mol
    ACTIVATION_ENERGY_GOOD = 20.0  # kcal/mol
    ACTIVATION_ENERGY_ACCEPTABLE = 25.0  # kcal/mol
    ACTIVATION_ENERGY_POOR = 30.0  # kcal/mol

    # 偏差容忍度
    DISTANCE_DEVIATION_TOLERANCE = 1.0  # Angstroms
    ANGLE_DEVIATION_TOLERANCE = 15.0  # degrees


# ============================================================================
# Substrate Definitions
# ============================================================================

class SubstrateConstants:
    """底物相关常量"""

    # 支持的底物类型
    SUPPORTED_SUBSTRATES = ['TMB', 'pNPP', 'ABTS', 'OPD', 'H2O2', 'GSH']

    # 检测波长（nm）
    DETECTION_WAVELENGTHS = {
        'TMB': 652,
        'pNPP': 405,
        'ABTS': 414,
        'OPD': 450,
        'H2O2': 240,
        'GSH': 412,
    }

    # 酶类型映射
    SUBSTRATE_TO_ENZYME_TYPE = {
        'TMB': 'peroxidase',
        'pNPP': 'phosphatase',
        'ABTS': 'peroxidase',
        'OPD': 'peroxidase',
        'H2O2': 'catalase',
        'GSH': 'glutathione_peroxidase',
    }


# ============================================================================
# Prediction Thresholds
# ============================================================================

class PredictionConstants:
    """预测相关常量"""

    # 催化位点阈值
    DEFAULT_CATALYTIC_THRESHOLD = 0.5
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.3

    # 三联体检测
    TRIAD_DISTANCE_MIN = 3.0  # Angstroms
    TRIAD_DISTANCE_MAX = 15.0  # Angstroms
    TRIAD_GEOMETRY_SCORE_THRESHOLD = 0.6

    # 双金属中心检测
    BIMETALLIC_DISTANCE_MIN = 3.0  # Angstroms
    BIMETALLIC_DISTANCE_MAX = 5.0  # Angstroms

    # EC号预测
    EC_CONFIDENCE_THRESHOLD = 0.5

    # Top-K结果
    DEFAULT_TOP_K = 15


# ============================================================================
# File and Cache Constants
# ============================================================================

class FileConstants:
    """文件和缓存相关常量"""

    # 默认目录
    DEFAULT_CACHE_DIR = "./cache"
    DEFAULT_MODEL_DIR = "./models"
    DEFAULT_OUTPUT_DIR = "./output"

    # 文件扩展名
    PDB_EXTENSION = ".pdb"
    MODEL_EXTENSION = ".pt"
    CONFIG_EXTENSION = ".yaml"

    # 缓存过期时间（秒）
    CACHE_EXPIRY_SECONDS = 7 * 24 * 3600  # 7天

    # 最大文件大小（MB）
    MAX_PDB_FILE_SIZE = 100
    MAX_MODEL_FILE_SIZE = 1000


# ============================================================================
# Visualization Constants
# ============================================================================

class VisualizationConstants:
    """可视化相关常量"""

    # 颜色方案
    CATALYTIC_RESIDUE_COLOR = 'red'
    NON_CATALYTIC_COLOR = 'gray'
    TRIAD_COLORS = ['red', 'blue', 'green']

    # 图表尺寸
    DEFAULT_FIGURE_WIDTH = 10
    DEFAULT_FIGURE_HEIGHT = 8
    DEFAULT_DPI = 300

    # 3D可视化
    ATOM_SPHERE_RADIUS = 0.3
    BOND_CYLINDER_RADIUS = 0.1

    # 导出格式
    SUPPORTED_IMAGE_FORMATS = ['png', 'svg', 'pdf', 'jpg']
    SUPPORTED_3D_FORMATS = ['pdb', 'mol2', 'sdf', 'xyz']


# ============================================================================
# Validation Constants
# ============================================================================

class ValidationConstants:
    """验证相关常量"""

    # 参数范围
    MIN_THRESHOLD = 0.0
    MAX_THRESHOLD = 1.0

    MIN_LEARNING_RATE = 1e-6
    MAX_LEARNING_RATE = 1e-2

    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 128

    MIN_EPOCHS = 1
    MAX_EPOCHS = 1000

    MIN_HIDDEN_DIM = 32
    MAX_HIDDEN_DIM = 1024

    # PDB ID格式
    PDB_ID_LENGTH = 4
    PDB_ID_PATTERN = r'^[0-9][A-Za-z0-9]{3}$'
