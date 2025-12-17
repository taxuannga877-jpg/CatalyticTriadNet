"""
CatalyticTriadNet v2.0: Geometric Deep Learning for Enzyme Catalytic Site Identification
"""

__version__ = "2.0.0"
__author__ = "CatalyticTriadNet Team"

# 核心模块
from .core.data import (
    MCSADataFetcher,
    MCSADataParser,
    EnzymeEntry,
    CatalyticResidue
)

# Swiss-Prot数据模块（570,000+条目）
from .core.swissprot_data import (
    SwissProtEntry,
    SwissProtDataFetcher,
    SwissProtDataParser
)

from .core.structure import (
    PDBProcessor,
    FeatureEncoder,
    AA_3TO1,
    AA_PROPERTIES,
    ROLE_MAPPING
)

from .core.dataset import CatalyticSiteDataset

# 预测模块
from .prediction.models import (
    CatalyticTriadPredictor,
    GeometricGNN,
    CatalyticSitePredictor,
    HierarchicalECPredictor
)

from .prediction.trainer import (
    CatalyticTriadTrainer,
    CatalyticTriadLoss,
    FocalLoss,
    compute_metrics
)
from .prediction.transfer_trainer import (
    TransferLearningTrainer
)

from .prediction.analysis import (
    TriadDetector,
    BimetallicCenterDetector,
    HydrogenBondAnalyzer
)

from .prediction.features import (
    ElectronicFeatureEncoder,
    SubstrateAwareEncoder,
    ConservationAnalyzer,
    EnhancedFeatureEncoder
)

from .prediction.predictor import (
    EnhancedCatalyticSiteInference,
    CatalyticSiteInference  # 兼容旧接口
)

from .prediction.batch_screener import (
    BatchCatalyticScreener
)

# 生成模块
from .generation.constraints import (
    CatalyticConstraints,
    GeometricConstraint
)

from .generation.models import (
    CatalyticDiffusionModel,
    E3EquivariantLayer,
    EquivariantGNN
)

from .generation.generator import (
    CatalyticNanozymeGenerator
)

from .generation.functional_group_extractor import (
    FunctionalGroupExtractor,
    FunctionalGroup,
    FUNCTIONAL_GROUP_TEMPLATES
)

from .generation.scaffold_builder import (
    ScaffoldBuilder,
    ScaffoldAtom
)

from .generation.nanozyme_assembler import (
    NanozymeAssembler
)

from .generation.substrate_definitions import (
    SUBSTRATE_LIBRARY,
    SubstrateDefinition,
    get_all_substrate_names,
    validate_substrate
)

from .generation.stage1_scorer import (
    Stage1FunctionalGroupScorer,
    MultiSubstrateStage1Scorer,
    quick_screen_functional_groups
)

from .generation.stage2_scorer import (
    Stage2NanozymeActivityScorer,
    MultiSubstrateStage2Scorer
)

# autodE过渡态计算模块（可选）
try:
    from .generation.autode_ts_calculator import (
        AutodETSCalculator,
        SubstrateReactionLibrary,
        batch_calculate_barriers
    )
    AUTODE_AVAILABLE = True
except ImportError:
    AUTODE_AVAILABLE = False

# 可视化模块
from .visualization.visualizer import (
    NanozymeVisualizer
)

from .visualization.plot_2d import (
    Visualizer2D
)

from .visualization.plot_3d import (
    Visualizer3D
)

from .visualization.adapters import (
    DiffusionModelAdapter
)

from .visualization.exporters import (
    ProfessionalExporter
)

__all__ = [
    # 版本信息
    "__version__",
    "__author__",

    # 核心模块 - M-CSA数据
    "MCSADataFetcher",
    "MCSADataParser",
    "EnzymeEntry",
    "CatalyticResidue",

    # 核心模块 - Swiss-Prot数据（570,000+条目）
    "SwissProtEntry",
    "SwissProtDataFetcher",
    "SwissProtDataParser",

    # 核心模块 - 结构处理
    "PDBProcessor",
    "FeatureEncoder",
    "CatalyticSiteDataset",
    
    # 预测模块
    "CatalyticTriadPredictor",
    "GeometricGNN",
    "CatalyticSitePredictor",
    "HierarchicalECPredictor",
    "CatalyticTriadTrainer",
    "TransferLearningTrainer",
    "CatalyticTriadLoss",
    "FocalLoss",
    "compute_metrics",
    "TriadDetector",
    "BimetallicCenterDetector",
    "HydrogenBondAnalyzer",
    "ElectronicFeatureEncoder",
    "SubstrateAwareEncoder",
    "ConservationAnalyzer",
    "EnhancedFeatureEncoder",
    "EnhancedCatalyticSiteInference",
    "CatalyticSiteInference",
    "BatchCatalyticScreener",

    # 生成模块 (扩散模型)
    "CatalyticNanozymeGenerator",
    "CatalyticDiffusionModel",
    "CatalyticConstraints",
    "GeometricConstraint",

    # 生成模块 (纳米酶组装)
    "NanozymeAssembler",
    "FunctionalGroupExtractor",
    "FunctionalGroup",
    "ScaffoldBuilder",
    "ScaffoldAtom",
    "FUNCTIONAL_GROUP_TEMPLATES",

    # 生成模块 (底物和打分)
    "SUBSTRATE_LIBRARY",
    "SubstrateDefinition",
    "get_all_substrate_names",
    "validate_substrate",
    "Stage1FunctionalGroupScorer",
    "MultiSubstrateStage1Scorer",
    "quick_screen_functional_groups",
    "Stage2NanozymeActivityScorer",
    "MultiSubstrateStage2Scorer",

    # 可视化模块
    "NanozymeVisualizer",
    "Visualizer2D",
    "Visualizer3D",
    "DiffusionModelAdapter",
    "ProfessionalExporter",
]

# 如果autodE可用，添加到__all__
if AUTODE_AVAILABLE:
    __all__.extend([
        "AutodETSCalculator",
        "SubstrateReactionLibrary",
        "batch_calculate_barriers",
    ])
