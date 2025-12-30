"""
Nanozyme Mining System - Stage 1 & 2
=====================================

Stage 1 (Database Layer): EC -> Nanozyme Function Type Mapping
Stage 2 (Extraction Layer): Catalytic Motif Extraction

两手抓策略：
- 标注数据：直接使用 UniProt/M-CSA 注释
- 未标注数据：使用 EasIFA 模型预测（必须集成）

Based on ChemEnzyRetroPlanner architecture.
"""

__version__ = "0.2.0"
__author__ = "Nanozyme Design Team"

from .database import NanozymeDatabase, UniProtFetcher, EnzymeEntry
from .database import MCSAFetcher
from .extraction import MotifExtractor, CatalyticMotif
from .prediction import EasIFAPredictor, ActiveSiteResult
from .core import DualTrackProcessor, ProcessedEnzyme

__all__ = [
    # Database
    "NanozymeDatabase",
    "UniProtFetcher",
    "EnzymeEntry",
    "MCSAFetcher",
    # Extraction
    "MotifExtractor",
    "CatalyticMotif",
    # Prediction (必须集成)
    "EasIFAPredictor",
    "ActiveSiteResult",
    # Core (两手抓)
    "DualTrackProcessor",
    "ProcessedEnzyme",
]
