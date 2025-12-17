"""
自定义异常类体系

提供清晰的错误层次结构，便于错误处理和调试。
"""

from typing import Optional


class CatalyticTriadNetError(Exception):
    """CatalyticTriadNet基础异常类"""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# ============================================================================
# 数据相关异常
# ============================================================================

class DataError(CatalyticTriadNetError):
    """数据处理相关异常基类"""
    pass


class PDBError(DataError):
    """PDB文件处理异常"""
    pass


class PDBDownloadError(PDBError):
    """PDB文件下载失败"""
    pass


class PDBParseError(PDBError):
    """PDB文件解析失败"""
    pass


class PDBNotFoundError(PDBError):
    """PDB文件不存在"""
    pass


class APIError(DataError):
    """API请求异常"""
    pass


class MCSAAPIError(APIError):
    """M-CSA API请求失败"""
    pass


class SwissProtAPIError(APIError):
    """Swiss-Prot API请求失败"""
    pass


class CacheError(DataError):
    """缓存操作异常"""
    pass


class CacheCorruptedError(CacheError):
    """缓存文件损坏"""
    pass


class DataValidationError(DataError):
    """数据验证失败"""
    pass


# ============================================================================
# 模型相关异常
# ============================================================================

class ModelError(CatalyticTriadNetError):
    """模型相关异常基类"""
    pass


class ModelLoadError(ModelError):
    """模型加载失败"""
    pass


class ModelInferenceError(ModelError):
    """模型推理失败"""
    pass


class ModelTrainingError(ModelError):
    """模型训练失败"""
    pass


class CheckpointError(ModelError):
    """检查点保存/加载失败"""
    pass


# ============================================================================
# 生成相关异常
# ============================================================================

class GenerationError(CatalyticTriadNetError):
    """纳米酶生成相关异常基类"""
    pass


class ConstraintViolationError(GenerationError):
    """几何约束违反"""
    pass


class SamplingError(GenerationError):
    """扩散采样失败"""
    pass


class AssemblyError(GenerationError):
    """纳米酶组装失败"""
    pass


class ScoringError(GenerationError):
    """打分计算失败"""
    pass


class TransitionStateError(GenerationError):
    """过渡态计算失败"""
    pass


# ============================================================================
# 配置相关异常
# ============================================================================

class ConfigError(CatalyticTriadNetError):
    """配置相关异常基类"""
    pass


class ConfigValidationError(ConfigError):
    """配置验证失败"""
    pass


class ConfigFileNotFoundError(ConfigError):
    """配置文件不存在"""
    pass


# ============================================================================
# 可视化相关异常
# ============================================================================

class VisualizationError(CatalyticTriadNetError):
    """可视化相关异常基类"""
    pass


class ExportError(VisualizationError):
    """导出失败"""
    pass


class RenderError(VisualizationError):
    """渲染失败"""
    pass


# ============================================================================
# 工具函数
# ============================================================================

def raise_with_context(
    exception_class: type[CatalyticTriadNetError],
    message: str,
    **details
) -> None:
    """
    抛出带上下文信息的异常

    Args:
        exception_class: 异常类
        message: 错误消息
        **details: 额外的上下文信息

    Raises:
        exception_class: 指定的异常

    Example:
        >>> raise_with_context(
        ...     PDBDownloadError,
        ...     "下载PDB文件失败",
        ...     pdb_id="1acb",
        ...     status_code=404
        ... )
    """
    raise exception_class(message, details=details)
