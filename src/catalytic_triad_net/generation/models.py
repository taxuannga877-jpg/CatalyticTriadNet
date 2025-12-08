#!/usr/bin/env python3
"""
扩散模型神经网络模块
包含: E(3)等变层、GNN、扩散模型、去噪器等
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# 导入常量和约束类
from .constraints import (
    NUM_ATOM_TYPES,
    ATOM_TO_IDX,
    ATOM_TYPES,
    CatalyticConstraints,
    GeometricConstraint
)
class E3EquivariantLayer(nn.Module):
    """
    E(3)-等变消息传递层
    
    保证输出在旋转、平移、反射下等变
    参考: https://github.com/vgsatorras/egnn
    """
    def __init__(self, hidden_dim: int, edge_dim: int = 0, 
                 act_fn: nn.Module = nn.SiLU(), residual: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual
        
        # 边缘MLP: 计算消息
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        
        # 节点MLP: 更新节点特征
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 坐标MLP: 更新坐标 (标量输出保证等变性)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # 注意力
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, h: torch.Tensor, x: torch.Tensor, 
                edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        """
        Args:
            h: 节点特征 [N, hidden_dim]
            x: 节点坐标 [N, 3]
            edge_index: 边索引 [2, E]
            edge_attr: 边特征 [E, edge_dim]
        
        Returns:
            h_new: 更新后的节点特征
            x_new: 更新后的坐标
        """
        row, col = edge_index
        
        # 计算相对位置和距离
        rel_pos = x[row] - x[col]  # [E, 3]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        
        # 边缘输入
        edge_input = [h[row], h[col], dist]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        edge_input = torch.cat(edge_input, dim=-1)
        
        # 计算消息
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_dim]
        
        # 注意力权重
        att = self.attention(m_ij)  # [E, 1]
        
        # 聚合消息到节点
        m_i = torch.zeros_like(h)
        m_i.index_add_(0, row, att * m_ij)
        
        # 更新节点特征
        h_input = torch.cat([h, m_i], dim=-1)
        h_new = self.node_mlp(h_input)
        if self.residual:
            h_new = h + h_new
        
        # 更新坐标 (等变)
        coord_diff = rel_pos / (dist + 1e-8)  # 归一化方向向量
        coord_weight = self.coord_mlp(m_ij)   # 标量权重
        
        coord_update = torch.zeros_like(x)
        coord_update.index_add_(0, row, coord_weight * coord_diff)
        x_new = x + coord_update
        
        return h_new, x_new


class EquivariantGNN(nn.Module):
    """E(3)-等变图神经网络"""
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 6,
                 edge_dim: int = 0, residual: bool = True):
        super().__init__()
        
        self.embedding = nn.Linear(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            E3EquivariantLayer(hidden_dim, edge_dim, residual=residual)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        h = self.embedding(h)
        
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        
        h = self.final_norm(h)
        return h, x


# =============================================================================
# 4. 条件扩散模型
# =============================================================================

class CatalyticDiffusionModel(nn.Module):
    """
    催化位点条件化扩散模型
    
    基于DDPM框架，使用E(3)-等变网络作为去噪网络
    条件信息通过cross-attention注入
    """
    def __init__(self, config: Dict = None):
        super().__init__()
        config = config or {}
        
        self.hidden_dim = config.get('hidden_dim', 256)
        self.n_layers = config.get('n_layers', 6)
        self.num_atom_types = NUM_ATOM_TYPES
        self.num_timesteps = config.get('num_timesteps', 1000)
        
        # 原子类型嵌入
        self.atom_embed = nn.Embedding(self.num_atom_types + 1, self.hidden_dim)  # +1 for mask
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 条件编码器
        self.condition_encoder = ConditionEncoder(
            anchor_dim=16,
            hidden_dim=self.hidden_dim
        )
        
        # E(3)-等变去噪网络
        self.denoiser = EquivariantDenoiser(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            condition_dim=self.hidden_dim
        )
        
        # 输出头
        self.atom_type_head = nn.Linear(self.hidden_dim, self.num_atom_types)
        self.coord_head = nn.Linear(self.hidden_dim, 3)
        
        # 扩散参数
        self._setup_diffusion_params()
    
    def _setup_diffusion_params(self):
        """设置扩散过程参数"""
        # 线性beta schedule
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, 
                 noise: torch.Tensor = None) -> torch.Tensor:
        """前向扩散: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def forward(self, atom_types: torch.Tensor, coords: torch.Tensor,
                edge_index: torch.Tensor, condition: Dict[str, torch.Tensor],
                t: torch.Tensor = None):
        """
        训练前向传播
        
        Args:
            atom_types: 原子类型 [B, N]
            coords: 原子坐标 [B, N, 3]
            edge_index: 边索引 [2, E]
            condition: 条件张量字典
            t: 时间步 [B]
        """
        B, N = atom_types.shape
        device = atom_types.device
        
        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=device)
        
        # 添加噪声
        noise = torch.randn_like(coords)
        noisy_coords = self.q_sample(coords, t, noise)
        
        # 编码
        h = self.atom_embed(atom_types)  # [B, N, hidden]
        t_emb = self.time_embed(t)       # [B, hidden]
        cond_emb = self.condition_encoder(condition)  # [B, hidden] or [B, M, hidden]
        
        # 去噪
        h_out, coord_pred = self.denoiser(
            h.view(B * N, -1),
            noisy_coords.view(B * N, 3),
            edge_index,
            t_emb,
            cond_emb
        )
        
        # 预测噪声
        pred_noise = self.coord_head(h_out).view(B, N, 3)
        pred_atom_logits = self.atom_type_head(h_out).view(B, N, -1)
        
        return {
            'pred_noise': pred_noise,
            'target_noise': noise,
            'pred_atom_logits': pred_atom_logits,
            'target_atom_types': atom_types
        }
    
    @torch.no_grad()
    def sample(self, condition: Dict[str, torch.Tensor], 
               n_atoms: int, n_samples: int = 1,
               guidance_scale: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        条件采样生成分子
        
        Args:
            condition: 催化约束条件
            n_atoms: 生成原子数
            n_samples: 生成样本数
            guidance_scale: 条件引导强度
        """
        device = next(self.parameters()).device
        
        # 初始化噪声
        x = torch.randn(n_samples, n_atoms, 3, device=device)
        atom_types = torch.randint(0, self.num_atom_types, (n_samples, n_atoms), device=device)
        
        # 构建全连接图
        edge_index = self._build_fc_edges(n_atoms, device)
        
        # 编码条件
        cond_emb = self.condition_encoder(condition)
        
        # 反向扩散采样
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            t_emb = self.time_embed(t_tensor)
            
            # 预测噪声
            h = self.atom_embed(atom_types).view(n_samples * n_atoms, -1)
            h_out, _ = self.denoiser(
                h, x.view(n_samples * n_atoms, 3),
                edge_index, t_emb, cond_emb
            )
            
            pred_noise = self.coord_head(h_out).view(n_samples, n_atoms, 3)
            pred_atom_logits = self.atom_type_head(h_out).view(n_samples, n_atoms, -1)
            
            # DDPM采样步
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            ) + torch.sqrt(beta) * noise
            
            # 更新原子类型 (采样)
            if t % 100 == 0:
                atom_types = pred_atom_logits.argmax(dim=-1)
        
        return {
            'atom_types': atom_types,
            'coords': x,
            'atom_type_logits': pred_atom_logits
        }
    
    def _build_fc_edges(self, n_atoms: int, device) -> torch.Tensor:
        """构建全连接边"""
        rows, cols = [], []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        return torch.tensor([rows, cols], dtype=torch.long, device=device)


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码 (用于时间步)"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionEncoder(nn.Module):
    """催化约束条件编码器"""
    def __init__(self, anchor_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        
        # 锚定原子编码
        self.anchor_encoder = nn.Sequential(
            nn.Linear(anchor_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 距离约束编码
        self.dist_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 配位约束编码
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, condition: Dict[str, torch.Tensor]) -> torch.Tensor:
        """编码条件"""
        anchor_feat = condition['anchor_features']
        dist_const = condition['distance_constraints']
        coord_const = condition['coordination_constraints']
        
        # 编码各部分
        anchor_emb = self.anchor_encoder(anchor_feat).mean(dim=0, keepdim=True)  # [1, hidden]
        
        if dist_const.shape[0] > 0:
            dist_emb = self.dist_encoder(dist_const).mean(dim=0, keepdim=True)
        else:
            dist_emb = torch.zeros_like(anchor_emb)
        
        if coord_const.shape[0] > 0:
            coord_emb = self.coord_encoder(coord_const.float()).mean(dim=0, keepdim=True)
        else:
            coord_emb = torch.zeros_like(anchor_emb)
        
        # 融合
        combined = torch.cat([anchor_emb, dist_emb, coord_emb], dim=-1)
        return self.fusion(combined)


class EquivariantDenoiser(nn.Module):
    """E(3)-等变去噪网络"""
    def __init__(self, hidden_dim: int, n_layers: int, condition_dim: int):
        super().__init__()
        
        # 时间嵌入投影
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 条件投影
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        
        # 等变层
        self.layers = nn.ModuleList([
            E3EquivariantLayer(hidden_dim, edge_dim=0, residual=True)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, t_emb: torch.Tensor,
                cond_emb: torch.Tensor):
        """
        Args:
            h: 节点特征 [N, hidden]
            x: 节点坐标 [N, 3]
            edge_index: 边索引
            t_emb: 时间嵌入 [B, hidden]
            cond_emb: 条件嵌入 [B, hidden] or [1, hidden]
        """
        # 注入时间和条件信息
        t_proj = self.time_proj(t_emb)  # [B, hidden]
        c_proj = self.cond_proj(cond_emb)  # [B, hidden]
        
        # 广播到所有节点
        N = h.shape[0]
        B = t_emb.shape[0]
        nodes_per_sample = N // B if B > 0 else N
        
        # 简化: 假设batch_size=1或均匀分布
        h = h + t_proj.repeat(nodes_per_sample, 1) + c_proj.repeat(nodes_per_sample, 1)
        
        # 等变层
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        
        return self.norm(h), x


# =============================================================================
# 5. 约束损失函数
# =============================================================================

class ConstraintLoss(nn.Module):
    """几何约束损失"""
    def __init__(self, distance_weight: float = 1.0,
                 coordination_weight: float = 1.0):
        super().__init__()
        self.distance_weight = distance_weight
        self.coordination_weight = coordination_weight
    
    def forward(self, coords: torch.Tensor, 
                constraints: CatalyticConstraints) -> torch.Tensor:
        """
        计算约束违反损失
        
        Args:
            coords: 生成的坐标 [N, 3]
            constraints: 催化约束
        """
        total_loss = 0.0
        
        # 距离约束损失
        for dc in constraints.distance_constraints:
            i, j = dc.atom_indices
            if i < coords.shape[0] and j < coords.shape[0]:
                actual_dist = torch.norm(coords[i] - coords[j])
                target_dist = dc.target_value
                tolerance = dc.tolerance
                
                # Huber-like损失
                diff = torch.abs(actual_dist - target_dist)
                if diff > tolerance:
                    loss = dc.weight * (diff - tolerance) ** 2
                else:
                    loss = 0.0
                total_loss += loss
        
        return total_loss * self.distance_weight


# =============================================================================
# 6. 主生成器接口
# =============================================================================

