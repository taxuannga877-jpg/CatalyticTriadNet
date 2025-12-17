"""
配置和参数验证器

提供统一的参数验证功能，确保输入的合法性。
"""

import re
from pathlib import Path
from typing import Any, Optional, Union

from .constants import ValidationConstants
from .exceptions import ConfigValidationError, DataValidationError


# ============================================================================
# 基础验证函数
# ============================================================================

def validate_threshold(
    value: float,
    name: str = "threshold",
    min_val: float = ValidationConstants.MIN_THRESHOLD,
    max_val: float = ValidationConstants.MAX_THRESHOLD
) -> float:
    """
    验证阈值参数

    Args:
        value: 阈值值
        name: 参数名称
        min_val: 最小值
        max_val: 最大值

    Returns:
        验证后的阈值

    Raises:
        ConfigValidationError: 如果阈值不在有效范围内
    """
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(
            f"{name}必须是数字类型",
            details={"value": value, "type": type(value).__name__}
        )

    if not min_val <= value <= max_val:
        raise ConfigValidationError(
            f"{name}必须在[{min_val}, {max_val}]范围内",
            details={"value": value, "min": min_val, "max": max_val}
        )

    return float(value)


def validate_positive_int(
    value: int,
    name: str = "value",
    min_val: int = 1,
    max_val: Optional[int] = None
) -> int:
    """
    验证正整数参数

    Args:
        value: 整数值
        name: 参数名称
        min_val: 最小值
        max_val: 最大值（可选）

    Returns:
        验证后的整数

    Raises:
        ConfigValidationError: 如果值不是有效的正整数
    """
    if not isinstance(value, int):
        raise ConfigValidationError(
            f"{name}必须是整数类型",
            details={"value": value, "type": type(value).__name__}
        )

    if value < min_val:
        raise ConfigValidationError(
            f"{name}必须大于等于{min_val}",
            details={"value": value, "min": min_val}
        )

    if max_val is not None and value > max_val:
        raise ConfigValidationError(
            f"{name}必须小于等于{max_val}",
            details={"value": value, "max": max_val}
        )

    return value


def validate_positive_float(
    value: float,
    name: str = "value",
    min_val: float = 0.0,
    max_val: Optional[float] = None,
    allow_zero: bool = False
) -> float:
    """
    验证正浮点数参数

    Args:
        value: 浮点数值
        name: 参数名称
        min_val: 最小值
        max_val: 最大值（可选）
        allow_zero: 是否允许零

    Returns:
        验证后的浮点数

    Raises:
        ConfigValidationError: 如果值不是有效的正浮点数
    """
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(
            f"{name}必须是数字类型",
            details={"value": value, "type": type(value).__name__}
        )

    if not allow_zero and value <= min_val:
        raise ConfigValidationError(
            f"{name}必须大于{min_val}",
            details={"value": value, "min": min_val}
        )
    elif allow_zero and value < min_val:
        raise ConfigValidationError(
            f"{name}必须大于等于{min_val}",
            details={"value": value, "min": min_val}
        )

    if max_val is not None and value > max_val:
        raise ConfigValidationError(
            f"{name}必须小于等于{max_val}",
            details={"value": value, "max": max_val}
        )

    return float(value)


def validate_path(
    path: Union[str, Path],
    name: str = "path",
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    create_if_missing: bool = False
) -> Path:
    """
    验证路径参数

    Args:
        path: 路径
        name: 参数名称
        must_exist: 是否必须存在
        must_be_file: 是否必须是文件
        must_be_dir: 是否必须是目录
        create_if_missing: 如果不存在是否创建（仅对目录有效）

    Returns:
        验证后的Path对象

    Raises:
        ConfigValidationError: 如果路径不符合要求
    """
    if not isinstance(path, (str, Path)):
        raise ConfigValidationError(
            f"{name}必须是字符串或Path对象",
            details={"value": path, "type": type(path).__name__}
        )

    path = Path(path)

    if must_exist and not path.exists():
        if create_if_missing and must_be_dir:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise ConfigValidationError(
                f"{name}不存在: {path}",
                details={"path": str(path)}
            )

    if path.exists():
        if must_be_file and not path.is_file():
            raise ConfigValidationError(
                f"{name}必须是文件: {path}",
                details={"path": str(path)}
            )
        if must_be_dir and not path.is_dir():
            raise ConfigValidationError(
                f"{name}必须是目录: {path}",
                details={"path": str(path)}
            )

    return path


def validate_pdb_id(pdb_id: str) -> str:
    """
    验证PDB ID格式

    Args:
        pdb_id: PDB ID

    Returns:
        验证后的PDB ID（大写）

    Raises:
        DataValidationError: 如果PDB ID格式不正确
    """
    if not isinstance(pdb_id, str):
        raise DataValidationError(
            "PDB ID必须是字符串",
            details={"value": pdb_id, "type": type(pdb_id).__name__}
        )

    pdb_id = pdb_id.strip().upper()

    if len(pdb_id) != ValidationConstants.PDB_ID_LENGTH:
        raise DataValidationError(
            f"PDB ID长度必须是{ValidationConstants.PDB_ID_LENGTH}个字符",
            details={"pdb_id": pdb_id, "length": len(pdb_id)}
        )

    if not re.match(ValidationConstants.PDB_ID_PATTERN, pdb_id, re.IGNORECASE):
        raise DataValidationError(
            "PDB ID格式不正确（应为：数字开头+3个字母或数字）",
            details={"pdb_id": pdb_id}
        )

    return pdb_id


def validate_choice(
    value: Any,
    choices: list,
    name: str = "value",
    case_sensitive: bool = True
) -> Any:
    """
    验证选项参数

    Args:
        value: 值
        choices: 有效选项列表
        name: 参数名称
        case_sensitive: 是否区分大小写

    Returns:
        验证后的值

    Raises:
        ConfigValidationError: 如果值不在有效选项中
    """
    if not case_sensitive and isinstance(value, str):
        value_lower = value.lower()
        choices_lower = [c.lower() if isinstance(c, str) else c for c in choices]
        if value_lower not in choices_lower:
            raise ConfigValidationError(
                f"{name}必须是以下选项之一: {choices}",
                details={"value": value, "choices": choices}
            )
        # 返回原始大小写的匹配项
        idx = choices_lower.index(value_lower)
        return choices[idx]
    else:
        if value not in choices:
            raise ConfigValidationError(
                f"{name}必须是以下选项之一: {choices}",
                details={"value": value, "choices": choices}
            )
        return value


# ============================================================================
# 特定领域验证函数
# ============================================================================

def validate_learning_rate(lr: float) -> float:
    """验证学习率"""
    return validate_positive_float(
        lr,
        name="learning_rate",
        min_val=ValidationConstants.MIN_LEARNING_RATE,
        max_val=ValidationConstants.MAX_LEARNING_RATE
    )


def validate_batch_size(batch_size: int) -> int:
    """验证批次大小"""
    return validate_positive_int(
        batch_size,
        name="batch_size",
        min_val=ValidationConstants.MIN_BATCH_SIZE,
        max_val=ValidationConstants.MAX_BATCH_SIZE
    )


def validate_epochs(epochs: int) -> int:
    """验证训练轮数"""
    return validate_positive_int(
        epochs,
        name="epochs",
        min_val=ValidationConstants.MIN_EPOCHS,
        max_val=ValidationConstants.MAX_EPOCHS
    )


def validate_hidden_dim(hidden_dim: int) -> int:
    """验证隐藏层维度"""
    return validate_positive_int(
        hidden_dim,
        name="hidden_dim",
        min_val=ValidationConstants.MIN_HIDDEN_DIM,
        max_val=ValidationConstants.MAX_HIDDEN_DIM
    )


def validate_dropout(dropout: float) -> float:
    """验证dropout概率"""
    return validate_threshold(dropout, name="dropout", min_val=0.0, max_val=0.9)


def validate_edge_cutoff(cutoff: float) -> float:
    """验证边截断距离"""
    return validate_positive_float(
        cutoff,
        name="edge_cutoff",
        min_val=1.0,
        max_val=50.0
    )


# ============================================================================
# 配置字典验证器
# ============================================================================

class ConfigValidator:
    """配置字典验证器"""

    @staticmethod
    def validate_model_config(config: dict) -> dict:
        """
        验证模型配置

        Args:
            config: 模型配置字典

        Returns:
            验证后的配置字典

        Raises:
            ConfigValidationError: 如果配置不合法
        """
        validated = {}

        if "hidden_dim" in config:
            validated["hidden_dim"] = validate_hidden_dim(config["hidden_dim"])

        if "num_layers" in config:
            validated["num_layers"] = validate_positive_int(
                config["num_layers"],
                name="num_layers",
                min_val=1,
                max_val=20
            )

        if "num_heads" in config:
            validated["num_heads"] = validate_positive_int(
                config["num_heads"],
                name="num_heads",
                min_val=1,
                max_val=16
            )

        if "dropout" in config:
            validated["dropout"] = validate_dropout(config["dropout"])

        if "activation" in config:
            validated["activation"] = validate_choice(
                config["activation"],
                choices=["relu", "gelu", "silu", "tanh"],
                name="activation",
                case_sensitive=False
            )

        return validated

    @staticmethod
    def validate_training_config(config: dict) -> dict:
        """
        验证训练配置

        Args:
            config: 训练配置字典

        Returns:
            验证后的配置字典

        Raises:
            ConfigValidationError: 如果配置不合法
        """
        validated = {}

        if "epochs" in config:
            validated["epochs"] = validate_epochs(config["epochs"])

        if "batch_size" in config:
            validated["batch_size"] = validate_batch_size(config["batch_size"])

        if "learning_rate" in config:
            validated["learning_rate"] = validate_learning_rate(config["learning_rate"])

        if "weight_decay" in config:
            validated["weight_decay"] = validate_positive_float(
                config["weight_decay"],
                name="weight_decay",
                min_val=0.0,
                max_val=0.1,
                allow_zero=True
            )

        if "patience" in config:
            validated["patience"] = validate_positive_int(
                config["patience"],
                name="patience",
                min_val=1,
                max_val=100
            )

        if "gradient_clip" in config:
            validated["gradient_clip"] = validate_positive_float(
                config["gradient_clip"],
                name="gradient_clip",
                min_val=0.1,
                max_val=10.0
            )

        return validated

    @staticmethod
    def validate_data_config(config: dict) -> dict:
        """
        验证数据配置

        Args:
            config: 数据配置字典

        Returns:
            验证后的配置字典

        Raises:
            ConfigValidationError: 如果配置不合法
        """
        validated = {}

        if "cache_dir" in config:
            validated["cache_dir"] = validate_path(
                config["cache_dir"],
                name="cache_dir",
                must_be_dir=True,
                create_if_missing=True
            )

        if "max_residues" in config:
            validated["max_residues"] = validate_positive_int(
                config["max_residues"],
                name="max_residues",
                min_val=10,
                max_val=10000
            )

        if "edge_cutoff" in config:
            validated["edge_cutoff"] = validate_edge_cutoff(config["edge_cutoff"])

        if "num_workers" in config:
            validated["num_workers"] = validate_positive_int(
                config["num_workers"],
                name="num_workers",
                min_val=0,
                max_val=32
            )

        return validated


# ============================================================================
# 批量验证工具
# ============================================================================

def validate_config_dict(config: dict, config_type: str = "model") -> dict:
    """
    验证配置字典

    Args:
        config: 配置字典
        config_type: 配置类型（'model', 'training', 'data'）

    Returns:
        验证后的配置字典

    Raises:
        ConfigValidationError: 如果配置不合法
    """
    validator = ConfigValidator()

    if config_type == "model":
        return validator.validate_model_config(config)
    elif config_type == "training":
        return validator.validate_training_config(config)
    elif config_type == "data":
        return validator.validate_data_config(config)
    else:
        raise ConfigValidationError(
            f"未知的配置类型: {config_type}",
            details={"config_type": config_type}
        )
