#!/usr/bin/env python3
"""
神经网络模型模块
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 几何图神经网络
# =============================================================================

class GeometricMessagePassing(nn.Module):
    """几何消息传递层"""

    def __init__(self, hidden_dim: int, edge_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: (N, hidden_dim)
            edge_index: (2, E)
            edge_attr: (E, edge_dim)
        """
        src, dst = edge_index

        # Multi-head attention
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)

        # 边特征作为attention bias
        edge_bias = self.edge_proj(edge_attr)

        # 计算attention scores
        attn_scores = (q[src] * k[dst]).sum(dim=-1) / math.sqrt(self.head_dim) + edge_bias
        attn_weights = self._scatter_softmax(attn_scores, dst, x.size(0))
        attn_weights = self.dropout(attn_weights)

        # 聚合
        weighted_v = v[src] * attn_weights.unsqueeze(-1)
        out = torch.zeros(x.size(0), self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, dst, weighted_v)
        out = out.view(-1, self.hidden_dim)

        out = self.out_proj(out)
        x = self.layer_norm(x + self.dropout(out))

        # FFN
        x = self.ffn_norm(x + self.ffn(x))

        return x

    def _scatter_softmax(self, src, index, num_nodes):
        """分组softmax"""
        max_val = torch.zeros(num_nodes, src.size(1), device=src.device)
        max_val.index_reduce_(0, index, src, 'amax', include_self=False)

        exp_src = torch.exp(src - max_val[index])
        sum_exp = torch.zeros(num_nodes, src.size(1), device=src.device)
        sum_exp.index_add_(0, index, exp_src)

        return exp_src / (sum_exp[index] + 1e-8)


class GeometricGNN(nn.Module):
    """几何图神经网络"""

    def __init__(self, node_dim: int = 33, edge_dim: int = 8, hidden_dim: int = 256,
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        self.layers = nn.ModuleList([
            GeometricMessagePassing(hidden_dim, hidden_dim // 2, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, edge_index, edge_attr):
        """
        Returns:
            node_embeddings: (N, hidden_dim)
        """
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

    def __init__(self, node_dim: int = 33, edge_dim: int = 8, hidden_dim: int = 256,
                 num_gnn_layers: int = 6, num_heads: int = 8, dropout: float = 0.2,
                 num_ec1: int = 7, num_ec2: int = 70, num_ec3: int = 300):
        super().__init__()

        # 几何GNN
        self.gnn = GeometricGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # 催化位点预测头
        self.site_predictor = CatalyticSitePredictor(hidden_dim, num_roles=9, dropout=dropout)

        # EC预测头
        self.ec_predictor = HierarchicalECPredictor(
            hidden_dim, num_ec1, num_ec2, num_ec3, dropout
        )

    def forward(self, node_features, edge_index, edge_attr, batch_idx: torch.Tensor = None):
        """
        Args:
            node_features: (N, node_dim)
            edge_index: (2, E)
            edge_attr: (E, edge_dim)
            batch_idx: 批次索引
        """
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
