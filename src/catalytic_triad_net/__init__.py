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

# 生成模块
from .generation.diffusion import (
    CatalyticNanozymeGenerator,
    CatalyticDiffusionModel,
    CatalyticConstraints,
    GeometricConstraint
)

# 可视化模块
from .visualization.visualization import (
    NanozymeVisualizer,
    Visualizer2D,
    Visualizer3D,
    DiffusionModelAdapter,
    ProfessionalExporter
)

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # 核心模块
    "MCSADataFetcher",
    "MCSADataParser",
    "EnzymeEntry",
    "CatalyticResidue",
    "PDBProcessor",
    "FeatureEncoder",
    "CatalyticSiteDataset",
    
    # 预测模块
    "CatalyticTriadPredictor",
    "GeometricGNN",
    "CatalyticSitePredictor",
    "HierarchicalECPredictor",
    "CatalyticTriadTrainer",
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
    
    # 生成模块
    "CatalyticNanozymeGenerator",
    "CatalyticDiffusionModel",
    "CatalyticConstraints",
    "GeometricConstraint",
    
    # 可视化模块
    "NanozymeVisualizer",
    "Visualizer2D",
    "Visualizer3D",
    "DiffusionModelAdapter",
    "ProfessionalExporter",
]
