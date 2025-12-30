"""
Prediction Module - EasIFA 活性位点预测
========================================

两手抓策略：
- 标注数据：直接使用 UniProt/M-CSA 注释
- 未标注数据：使用 EasIFA 模型预测

此模块为必须集成，不是可选的。
"""

from .easifa_predictor import (
    EasIFAPredictor,
    PredictedActiveSite,
    ActiveSiteResult,
    LABEL_TO_SITE_TYPE,
)

__all__ = [
    "EasIFAPredictor",
    "PredictedActiveSite",
    "ActiveSiteResult",
    "LABEL_TO_SITE_TYPE",
]
