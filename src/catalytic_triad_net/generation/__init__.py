"""
扩散生成模块
包含传统生成和片段化生成两种方法
"""

from .constraints import (
    GeometricConstraint,
    CatalyticConstraints,
    ATOM_TYPES,
    ATOM_TO_IDX,
    FUNCTIONAL_GROUP_TEMPLATES
)

from .models import (
    E3EquivariantLayer,
    EquivariantGNN,
    CatalyticDiffusionModel,
    SinusoidalPositionEmbeddings,
    ConditionEncoder,
    EquivariantDenoiser,
    ConstraintLoss
)

from .generator import CatalyticNanozymeGenerator
from .dataset import NanozymeDataset
from .trainer import Trainer

# 片段化生成模块（StoL 启发）
from .fragment_definitions import (
    NanozymeFragment,
    FragmentType,
    FragmentationRule,
    NanozymeFragmentizer,
    create_default_fragmentizer
)

from .fragment_conformation_generator import (
    FragmentConformationGenerator,
    ProgressiveFragmentGenerator,
    create_fragment_generator
)

from .fragment_assembler import (
    FragmentAssembler,
    MultiConformationAssembler,
    create_assembler
)

from .conformation_analysis import (
    ChemicalValidator,
    ConformationAnalyzer,
    ValidationResult,
    create_analyzer
)

from .fragmented_nanozyme_pipeline import (
    FragmentedNanozymePipeline,
    create_pipeline
)

__all__ = [
    # 约束
    'GeometricConstraint',
    'CatalyticConstraints',
    'ATOM_TYPES',
    'ATOM_TO_IDX',
    'FUNCTIONAL_GROUP_TEMPLATES',

    # 模型
    'E3EquivariantLayer',
    'EquivariantGNN',
    'CatalyticDiffusionModel',
    'SinusoidalPositionEmbeddings',
    'ConditionEncoder',
    'EquivariantDenoiser',
    'ConstraintLoss',

    # 传统生成器
    'CatalyticNanozymeGenerator',

    # 数据集和训练
    'NanozymeDataset',
    'Trainer',

    # 片段化生成（新增）
    'NanozymeFragment',
    'FragmentType',
    'FragmentationRule',
    'NanozymeFragmentizer',
    'create_default_fragmentizer',
    'FragmentConformationGenerator',
    'ProgressiveFragmentGenerator',
    'create_fragment_generator',
    'FragmentAssembler',
    'MultiConformationAssembler',
    'create_assembler',
    'ChemicalValidator',
    'ConformationAnalyzer',
    'ValidationResult',
    'create_analyzer',
    'FragmentedNanozymePipeline',
    'create_pipeline',
]
