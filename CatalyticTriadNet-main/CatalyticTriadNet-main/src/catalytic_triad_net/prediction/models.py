#!/usr/bin/env python3
"""
神经网络模型模块

Refactored to use:
- ModelConstants for architecture parameters
- Better variable naming for clarity
- Extracted methods for complex operations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

from ..config import get_config
from ..core.constants import ModelConstants

logger = logging.getLogger(__name__)


# =============================================================================
# 几何图神经网络
# =============================================================================

class GeometricMessagePassing(nn.Module):
    """
    几何消息传递层

    使用多头注意力机制进行消息传递，结合边特征和几何信息。
    """

    def __init__(self, hidden_dim: int, edge_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        初始化几何消息传递层。

        Args:
            hidden_dim: 隐藏层维度
            edge_dim: 边特征维度
            num_heads: 注意力头数
            dropout: Dropout 概率
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # FFN with expansion factor from ModelConstants
        ffn_dim = hidden_dim * ModelConstants.FFN_EXPANSION_FACTOR
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.LayerNorm(ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 节点特征，形状 (N, hidden_dim)
            edge_index: 边索引，形状 (2, E)
            edge_attr: 边特征，形状 (E, edge_dim)

        Returns:
            torch.Tensor: 更新后的节点特征，形状 (N, hidden_dim)
        """
        # 输入验证
        assert x.ndim == 2 and x.size(1) == self.hidden_dim, \
            f"Expected x shape (N, {self.hidden_dim}), got {x.shape}"
        assert edge_index.ndim == 2 and edge_index.size(0) == 2, \
            f"Expected edge_index shape (2, E), got {edge_index.shape}"
        assert edge_attr.ndim == 2, \
            f"Expected edge_attr to be 2D, got shape {edge_attr.shape}"

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # 处理空边情况
        if num_edges == 0:
            logger.warning("No edges in graph, skipping message passing")
            return x

        src, dst = edge_index

        # Multi-head attention
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)  # (N, H, D)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)  # (N, H, D)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)  # (N, H, D)

        # 边特征作为attention bias
        edge_bias = self.edge_proj(edge_attr)  # (E, H)

        # 计算attention scores
        attn_scores = (q[src] * k[dst]).sum(dim=-1) / math.sqrt(self.head_dim) + edge_bias  # (E, H)
        attn_weights = self._scatter_softmax(attn_scores, dst, num_nodes)  # (E, H)
        attn_weights = self.dropout(attn_weights)

        # 聚合
        weighted_v = v[src] * attn_weights.unsqueeze(-1)  # (E, H, D)
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, weighted_v)
        out = out.view(-1, self.hidden_dim)  # (N, hidden_dim)

        out = self.out_proj(out)
        x = self.layer_norm(x + self.dropout(out))

        # FFN (统一顺序: LayerNorm -> GELU)
        x = self.ffn_norm(x + self.ffn(x))

        return x

    def _scatter_softmax(self, src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        分组 softmax，带掩码防止空邻居。

        Args:
            src: 源张量，形状 (E, H)
            index: 索引张量，形状 (E,)
            num_nodes: 节点总数

        Returns:
            torch.Tensor: Softmax 结果，形状 (E, H)
        """
        # 计算每个节点的最大值（用于数值稳定性）
        max_val = torch.full((num_nodes, src.size(1)), float('-inf'), device=src.device, dtype=src.dtype)
        max_val.index_reduce_(0, index, src, 'amax', include_self=False)

        # 处理没有邻居的节点（最大值为 -inf）
        max_val = torch.where(torch.isinf(max_val), torch.zeros_like(max_val), max_val)

        # 计算 exp
        exp_src = torch.exp(src - max_val[index])

        # 计算每个节点的 exp 和
        sum_exp = torch.zeros(num_nodes, src.size(1), device=src.device, dtype=src.dtype)
        sum_exp.index_add_(0, index, exp_src)

        # 防止除零（对于没有邻居的节点）
        sum_exp = torch.clamp(sum_exp, min=1e-8)

        return exp_src / sum_exp[index]


class GeometricGNN(nn.Module):
    """
    几何图神经网络

    使用多层几何消息传递进行节点特征学习。
    """

    def __init__(self, node_dim: int = 33, edge_dim: int = 8, hidden_dim: int = 256,
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.2):
        """
        初始化几何图神经网络。

        Args:
            node_dim: 节点特征维度
            edge_dim: 边特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN 层数
            num_heads: 注意力头数
            dropout: Dropout 概率
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        self.layers = nn.ModuleList([
            GeometricMessagePassing(hidden_dim, hidden_dim // 2, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        logger.info(f"GeometricGNN initialized: node_dim={node_dim}, edge_dim={edge_dim}, "
                   f"hidden_dim={hidden_dim}, num_layers={num_layers}")

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            node_features: 节点特征，形状 (N, node_dim)
            edge_index: 边索引，形状 (2, E)
            edge_attr: 边特征，形状 (E, edge_dim)

        Returns:
            torch.Tensor: 节点嵌入，形状 (N, hidden_dim)
        """
        # 输入验证
        assert node_features.ndim == 2 and node_features.size(1) == self.node_dim, \
            f"Expected node_features shape (N, {self.node_dim}), got {node_features.shape}"
        assert edge_index.ndim == 2 and edge_index.size(0) == 2, \
            f"Expected edge_index shape (2, E), got {edge_index.shape}"
        assert edge_attr.ndim == 2 and edge_attr.size(1) == self.edge_dim, \
            f"Expected edge_attr shape (E, {self.edge_dim}), got {edge_attr.shape}"

        h = self.node_encoder(node_features)
        edge_emb = self.edge_encoder(edge_attr)

        for layer in self.layers:
            h = layer(h, edge_index, edge_emb)

        return self.final_norm(h)


# =============================================================================
# 预测头
# =============================================================================

class CatalyticSitePredictor(nn.Module):
    """催化位点预测模块"""

    def __init__(self, hidden_dim: int = 256, num_roles: int = 9, dropout: float = 0.2):
        super().__init__()

        # 二分类: 是否为催化残基
        self.site_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # 多标签分类: 催化角色
        self.role_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_roles)
        )

    def forward(self, node_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_embeddings: (N, hidden_dim)
        Returns:
            site_logits: (N, 1)
            role_logits: (N, num_roles)
        """
        site_logits = self.site_classifier(node_embeddings)
        role_logits = self.role_classifier(node_embeddings)
        return site_logits, role_logits


class HierarchicalECPredictor(nn.Module):
    """层级EC号预测器"""

    def __init__(self, hidden_dim: int = 256, num_ec1: int = 7, num_ec2: int = 70,
                 num_ec3: int = 300, dropout: float = 0.2):
        super().__init__()

        # 全局池化
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # EC1分类
        self.ec1_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_ec1)
        )

        # EC2分类 (条件于EC1)
        self.ec2_classifier = nn.Sequential(
            nn.Linear(hidden_dim + num_ec1, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_ec2)
        )

        # EC3分类 (条件于EC1, EC2)
        self.ec3_classifier = nn.Sequential(
            nn.Linear(hidden_dim + num_ec1 + num_ec2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_ec3)
        )

    def forward(self, node_embeddings: torch.Tensor, batch_idx: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_embeddings: (N, hidden_dim)
            batch_idx: (N,) 批次索引
        """
        # 全局池化
        if batch_idx is None:
            global_feat = node_embeddings.mean(dim=0, keepdim=True)
        else:
            num_graphs = batch_idx.max().item() + 1
            global_feat = torch.zeros(num_graphs, node_embeddings.size(1), device=node_embeddings.device)
            for i in range(num_graphs):
                mask = batch_idx == i
                global_feat[i] = node_embeddings[mask].mean(dim=0)

        global_feat = self.global_pool(global_feat)

        # 层级预测
        ec1_logits = self.ec1_classifier(global_feat)
        ec1_probs = F.softmax(ec1_logits, dim=-1)

        ec2_input = torch.cat([global_feat, ec1_probs], dim=-1)
        ec2_logits = self.ec2_classifier(ec2_input)
        ec2_probs = F.softmax(ec2_logits, dim=-1)

        ec3_input = torch.cat([global_feat, ec1_probs, ec2_probs], dim=-1)
        ec3_logits = self.ec3_classifier(ec3_input)

        return {
            'ec1_logits': ec1_logits,
            'ec2_logits': ec2_logits,
            'ec3_logits': ec3_logits,
            'global_feat': global_feat
        }


# =============================================================================
# 完整模型
# =============================================================================

class CatalyticTriadPredictor(nn.Module):
    """
    催化三联体预测完整模型

    架构:
    1. GeometricGNN: 消息传递学习结构表示
    2. 双任务头: 催化位点预测 + EC号预测
    """

    def __init__(self, node_dim: Optional[int] = None, edge_dim: Optional[int] = None,
                 hidden_dim: Optional[int] = None, num_gnn_layers: Optional[int] = None,
                 num_heads: Optional[int] = None, dropout: Optional[float] = None,
                 num_ec1: int = 7, num_ec2: int = 70, num_ec3: int = 300,
                 config: Optional[Dict] = None):
        """
        初始化催化三联体预测器。

        Args:
            node_dim: 节点特征维度（如果为 None，从 config 读取）
            edge_dim: 边特征维度（如果为 None，从 config 读取）
            hidden_dim: 隐藏层维度（如果为 None，从 config 读取）
            num_gnn_layers: GNN 层数（如果为 None，从 config 读取）
            num_heads: 注意力头数（如果为 None，从 config 读取）
            dropout: Dropout 概率（如果为 None，从 config 读取）
            num_ec1: EC 第一级类别数
            num_ec2: EC 第二级类别数
            num_ec3: EC 第三级类别数
            config: 配置字典（可选）
        """
        super().__init__()

        # 从配置读取参数
        if config is None:
            global_config = get_config()
            model_config = global_config.get('model', {})
        else:
            model_config = config.get('model', {})

        # 使用提供的参数或从配置读取（确保类型正确）
        final_node_dim: int = node_dim if node_dim is not None else model_config.get('node_dim', 48)
        final_edge_dim: int = edge_dim if edge_dim is not None else model_config.get('edge_dim', 14)
        final_hidden_dim: int = hidden_dim if hidden_dim is not None else model_config.get('hidden_dim', 128)
        final_num_gnn_layers: int = num_gnn_layers if num_gnn_layers is not None else model_config.get('num_layers', 4)
        final_num_heads: int = num_heads if num_heads is not None else model_config.get('num_heads', 4)
        final_dropout: float = dropout if dropout is not None else model_config.get('dropout', 0.1)

        self.node_dim = final_node_dim
        self.edge_dim = final_edge_dim
        self.hidden_dim = final_hidden_dim

        # 几何GNN
        self.gnn = GeometricGNN(
            node_dim=final_node_dim,
            edge_dim=final_edge_dim,
            hidden_dim=final_hidden_dim,
            num_layers=final_num_gnn_layers,
            num_heads=final_num_heads,
            dropout=final_dropout
        )

        # 催化位点预测头
        self.site_predictor = CatalyticSitePredictor(final_hidden_dim, num_roles=9, dropout=final_dropout)

        # EC预测头
        self.ec_predictor = HierarchicalECPredictor(
            final_hidden_dim, num_ec1, num_ec2, num_ec3, final_dropout
        )

        logger.info(f"CatalyticTriadPredictor initialized with node_dim={final_node_dim}, "
                   f"edge_dim={final_edge_dim}, hidden_dim={final_hidden_dim}")

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            node_features: 节点特征，形状 (N, node_dim)
            edge_index: 边索引，形状 (2, E)
            edge_attr: 边特征，形状 (E, edge_dim)
            batch_idx: 批次索引，形状 (N,)（可选）

        Returns:
            Dict[str, torch.Tensor]: 包含预测结果的字典
        """
        # 输入验证
        assert node_features.ndim == 2 and node_features.size(1) == self.node_dim, \
            f"Expected node_features shape (N, {self.node_dim}), got {node_features.shape}"
        assert edge_index.ndim == 2 and edge_index.size(0) == 2, \
            f"Expected edge_index shape (2, E), got {edge_index.shape}"
        assert edge_attr.ndim == 2 and edge_attr.size(1) == self.edge_dim, \
            f"Expected edge_attr shape (E, {self.edge_dim}), got {edge_attr.shape}"

        # GNN编码
        node_emb = self.gnn(node_features, edge_index, edge_attr)

        # 催化位点预测
        site_logits, role_logits = self.site_predictor(node_emb)

        # EC预测
        ec_outputs = self.ec_predictor(node_emb, batch_idx)

        return {
            'site_logits': site_logits,
            'role_logits': role_logits,
            'node_embeddings': node_emb,
            **ec_outputs
        }

    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = 'cpu'):
        """从检查点加载"""
        ckpt = torch.load(path, map_location=device)
        config = ckpt.get('config', {})

        model = cls(
            node_dim=config.get('node_dim', 33),
            edge_dim=config.get('edge_dim', 8),
            hidden_dim=config.get('hidden_dim', 256),
            num_gnn_layers=config.get('num_gnn_layers', 6),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.2),
            num_ec1=config.get('num_ec1', 7),
            num_ec2=config.get('num_ec2', 70),
            num_ec3=config.get('num_ec3', 300)
        )

        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        return model

    def save_checkpoint(self, path: str, config: Dict, optimizer=None, epoch=0, metrics=None):
        """保存检查点"""
        ckpt = {
            'model_state_dict': self.state_dict(),
            'config': config,
            'epoch': epoch,
            'metrics': metrics or {}
        }
        if optimizer:
            ckpt['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(ckpt, path)
        logger.info(f"✓ 保存模型: {path}")
