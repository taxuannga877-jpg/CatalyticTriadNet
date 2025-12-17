#!/usr/bin/env python3
"""
训练器模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm
import logging

from .models import CatalyticTriadPredictor
from ..config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# 损失函数
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class CatalyticTriadLoss(nn.Module):
    """多任务损失，支持标签掩码"""

    def __init__(self, pos_weight: float = 10.0, focal_gamma: float = 2.0):
        super().__init__()

        self.site_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.role_loss = nn.BCEWithLogitsLoss()
        self.ec_loss = nn.CrossEntropyLoss()

        # 损失权重
        self.w_site = 1.0
        self.w_role = 0.5
        self.w_ec1 = 0.3

    def forward(self, outputs: Dict, labels: Dict, device='cpu') -> Dict:
        """计算多任务损失（根据掩码决定是否计算某个任务）"""
        losses = {}

        # 催化位点损失（带掩码）
        site_logits = outputs['site_logits'].squeeze(-1)
        site_labels = labels['site_labels'].float().to(device)
        site_mask = labels.get('labels_mask', torch.ones_like(site_labels)).to(device)
        site_valid = site_mask > 0
        if site_valid.any():
            losses['site'] = self.site_loss(site_logits[site_valid], site_labels[site_valid])
            losses['site_focal'] = self.focal_loss(site_logits[site_valid], site_labels[site_valid])
        else:
            losses['site'] = torch.tensor(0.0, device=device)
            losses['site_focal'] = torch.tensor(0.0, device=device)

        # 角色损失（带掩码）
        role_logits = outputs['role_logits']
        role_labels = labels['role_labels'].to(device)
        role_mask = labels.get('role_mask', torch.ones(role_labels.shape[0], device=device))
        role_valid = role_mask > 0
        if role_valid.any():
            losses['role'] = self.role_loss(role_logits[role_valid], role_labels[role_valid])
        else:
            losses['role'] = torch.tensor(0.0, device=device)

        # EC损失（带掩码）
        if 'ec1_label' in labels:
            ec1_logits = outputs['ec1_logits']
            ec1_label = labels['ec1_label']
            if not torch.is_tensor(ec1_label):
                ec1_label = torch.tensor(ec1_label, device=device)
            ec1_label = ec1_label.to(device)
            ec1_mask = labels.get('ec1_mask', torch.ones_like(ec1_label, dtype=torch.float32)).to(device)
            ec_valid = ec1_mask > 0
            if ec_valid.any():
                losses['ec1'] = self.ec_loss(ec1_logits[ec_valid], ec1_label[ec_valid])
            else:
                losses['ec1'] = torch.tensor(0.0, device=device)

        # 总损失（仅累加有效项）
        total = torch.tensor(0.0, device=device)
        if losses.get('site', 0).numel() > 0:
            total += self.w_site * (losses['site'] + losses['site_focal'])
        if losses.get('role', 0).numel() > 0:
            total += self.w_role * losses['role']
        if 'ec1' in losses and losses['ec1'].numel() > 0:
            total += self.w_ec1 * losses['ec1']

        losses['total'] = total
        return losses


# =============================================================================
# 评估指标
# =============================================================================

def compute_metrics(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict:
    """计算评估指标"""
    binary_preds = (preds >= threshold).astype(int)

    tp = np.sum((binary_preds == 1) & (labels == 1))
    fp = np.sum((binary_preds == 1) & (labels == 0))
    fn = np.sum((binary_preds == 0) & (labels == 1))
    tn = np.sum((binary_preds == 0) & (labels == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # MCC
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / (denom + 1e-8)

    # AUPRC
    try:
        from sklearn.metrics import average_precision_score
        auprc = average_precision_score(labels, preds)
    except:
        auprc = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'auprc': auprc,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }


# =============================================================================
# 训练器
# =============================================================================

class CatalyticTriadTrainer:
    """
    模型训练器

    支持批量训练、梯度裁剪、早停等功能。
    """

    def __init__(self, model: CatalyticTriadPredictor,
                 learning_rate: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 pos_weight: float = 10.0,
                 focal_gamma: float = 2.0,
                 patience: Optional[int] = None,
                 grad_clip: Optional[float] = None,
                 config: Optional[Dict] = None):
        """
        初始化训练器。

        Args:
            model: 要训练的模型
            learning_rate: 学习率（如果为 None，从 config 读取）
            weight_decay: 权重衰减（如果为 None，从 config 读取）
            pos_weight: 正样本权重
            focal_gamma: Focal Loss gamma 参数
            patience: 早停耐心值（如果为 None，从 config 读取）
            grad_clip: 梯度裁剪阈值（如果为 None，从 config 读取）
            config: 配置字典（可选）
        """
        # 从配置读取参数
        if config is None:
            global_config = get_config()
            train_config = global_config.get('training', {})
        else:
            train_config = config.get('training', {})

        final_learning_rate: float = learning_rate if learning_rate is not None else train_config.get('learning_rate', 1e-4)
        final_weight_decay: float = weight_decay if weight_decay is not None else train_config.get('weight_decay', 1e-5)
        final_patience: int = patience if patience is not None else train_config.get('patience', 15)
        final_grad_clip: float = grad_clip if grad_clip is not None else train_config.get('grad_clip', 1.0)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.grad_clip = final_grad_clip
        self.patience = final_patience

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=final_learning_rate,
            weight_decay=final_weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.criterion = CatalyticTriadLoss(pos_weight, focal_gamma)
        self.criterion.site_loss.pos_weight = self.criterion.site_loss.pos_weight.to(self.device)

        logger.info(f"Trainer initialized: device={self.device}, lr={final_learning_rate}, "
                   f"weight_decay={final_weight_decay}, grad_clip={final_grad_clip}")

    def train_epoch(self, dataloader) -> Dict[str, Any]:
        """
        训练一个 epoch。

        Args:
            dataloader: 数据加载器

        Returns:
            Dict[str, Any]: 包含损失和指标的字典
        """
        self.model.train()

        total_loss = 0.0
        all_preds, all_labels = [], []
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            # 支持批量数据（列表）或单个样本
            samples = batch if isinstance(batch, list) else [batch]

            for sample in samples:
                self.optimizer.zero_grad()

                # 准备数据
                x = sample['node_features'].to(self.device)
                ei = sample['edge_index'].to(self.device)
                ea = sample['edge_attr'].to(self.device)

                # 前向传播
                outputs = self.model(x, ei, ea)

                # 计算损失
                labels_dict = {
                    'site_labels': sample['labels'],
                    'role_labels': sample['role_labels'],
                    'ec1_label': sample['ec1_label'],
                    'labels_mask': sample.get('labels_mask', torch.ones_like(sample['labels'], dtype=torch.float32)),
                    'role_mask': sample.get('role_mask', torch.ones(sample['labels'].shape[0], dtype=torch.float32)),
                    'ec1_mask': sample.get('ec1_mask', torch.ones(1, dtype=torch.float32))
                }
                losses = self.criterion(outputs, labels_dict, self.device)

                # 反向传播
                losses['total'].backward()

                # 梯度裁剪
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

                total_loss += losses['total'].item()
                num_batches += 1

                # 收集预测
                preds = torch.sigmoid(outputs['site_logits']).detach().cpu().numpy().flatten()
                labels_np = sample['labels'].numpy()
                if 'labels_mask' in sample:
                    mask = sample['labels_mask'].numpy() > 0
                    preds = preds[mask]
                    labels_np = labels_np[mask]
                all_preds.extend(preds)
                all_labels.extend(labels_np)

        self.scheduler.step()

        # 计算指标（带空数据保护）
        if len(all_preds) > 0 and len(all_labels) > 0:
            metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
            metrics['loss'] = total_loss / max(num_batches, 1)
        else:
            logger.warning("No predictions collected during training epoch")
            metrics = {'loss': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mcc': 0.0, 'auprc': 0.0}

        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, Any]:
        """
        评估模型。

        Args:
            dataloader: 数据加载器

        Returns:
            Dict[str, Any]: 包含损失和指标的字典
        """
        self.model.eval()

        total_loss = 0.0
        all_preds, all_labels = [], []
        num_batches = 0

        for batch in dataloader:
            # 支持批量数据（列表）或单个样本
            samples = batch if isinstance(batch, list) else [batch]

            for sample in samples:
                x = sample['node_features'].to(self.device)
                ei = sample['edge_index'].to(self.device)
                ea = sample['edge_attr'].to(self.device)

                outputs = self.model(x, ei, ea)

                labels_dict = {
                    'site_labels': sample['labels'],
                    'role_labels': sample['role_labels'],
                    'ec1_label': sample['ec1_label'],
                    'labels_mask': sample.get('labels_mask', torch.ones_like(sample['labels'], dtype=torch.float32)),
                    'role_mask': sample.get('role_mask', torch.ones(sample['labels'].shape[0], dtype=torch.float32)),
                    'ec1_mask': sample.get('ec1_mask', torch.ones(1, dtype=torch.float32))
                }
                losses = self.criterion(outputs, labels_dict, self.device)
                total_loss += losses['total'].item()
                num_batches += 1

                preds = torch.sigmoid(outputs['site_logits']).cpu().numpy().flatten()
                labels_np = sample['labels'].numpy()
                if 'labels_mask' in sample:
                    mask = sample['labels_mask'].numpy() > 0
                    preds = preds[mask]
                    labels_np = labels_np[mask]
                all_preds.extend(preds)
                all_labels.extend(labels_np)

        # 计算指标（带空数据保护）
        if len(all_preds) > 0 and len(all_labels) > 0:
            metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
            metrics['loss'] = total_loss / max(num_batches, 1)
        else:
            logger.warning("No predictions collected during evaluation")
            metrics = {'loss': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mcc': 0.0, 'auprc': 0.0}

        return metrics

    def train(self, train_loader, val_loader, epochs: int = 100, save_dir: str = './models') -> float:
        """
        完整训练流程。

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            save_dir: 模型保存目录

        Returns:
            float: 最佳验证 F1 分数
        """
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(exist_ok=True, parents=True)

        best_f1 = 0.0
        patience_counter = 0

        logger.info(f"Starting training: epochs={epochs}, device={self.device}")
        logger.info(f"Model will be saved to: {save_dir_path}")

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            logger.info(f"Epoch {epoch + 1:3d}/{epochs} | "
                       f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['f1']:.4f} | "
                       f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f} "
                       f"MCC: {val_metrics['mcc']:.4f}")

            # 保存最佳模型
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0

                # 保存模型配置
                config = {
                    'node_dim': self.model.node_dim,
                    'edge_dim': self.model.edge_dim,
                    'hidden_dim': self.model.hidden_dim,
                    'num_gnn_layers': 6,
                    'num_heads': 8,
                    'dropout': 0.2,
                    'num_ec1': 7,
                    'num_ec2': 70,
                    'num_ec3': 300
                }

                model_path = save_dir_path / 'best_model.pt'
                self.model.save_checkpoint(
                    str(model_path),
                    config, self.optimizer, epoch, val_metrics
                )
                logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        logger.info(f"Training completed. Best validation F1: {best_f1:.4f}")

        return best_f1
