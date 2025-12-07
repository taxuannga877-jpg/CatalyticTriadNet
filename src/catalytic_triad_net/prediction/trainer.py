#!/usr/bin/env python3
"""
训练器模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import logging

from .models import CatalyticTriadPredictor

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
    """多任务损失"""

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
        """计算多任务损失"""
        losses = {}

        # 催化位点损失
        site_logits = outputs['site_logits'].squeeze(-1)
        site_labels = labels['site_labels'].float().to(device)
        losses['site'] = self.site_loss(site_logits, site_labels)
        losses['site_focal'] = self.focal_loss(site_logits, site_labels)

        # 角色损失
        role_logits = outputs['role_logits']
        role_labels = labels['role_labels'].to(device)
        losses['role'] = self.role_loss(role_logits, role_labels)

        # EC损失
        if 'ec1_label' in labels:
            ec1_logits = outputs['ec1_logits']
            ec1_label = torch.tensor([labels['ec1_label']], device=device)
            losses['ec1'] = self.ec_loss(ec1_logits, ec1_label)

        # 总损失
        total = (self.w_site * (losses['site'] + losses['site_focal']) +
                 self.w_role * losses['role'])
        if 'ec1' in losses:
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
    """模型训练器"""

    def __init__(self, model: CatalyticTriadPredictor,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 pos_weight: float = 10.0,
                 focal_gamma: float = 2.0,
                 patience: int = 15):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.criterion = CatalyticTriadLoss(pos_weight, focal_gamma)
        self.criterion.site_loss.pos_weight = self.criterion.site_loss.pos_weight.to(self.device)

        self.patience = patience

        logger.info(f"使用设备: {self.device}")

    def train_epoch(self, dataloader) -> Dict:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(dataloader, desc="Training", leave=False):
            for sample in batch:
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
                    'ec1_label': sample['ec1_label']
                }
                losses = self.criterion(outputs, labels_dict, self.device)

                # 反向传播
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += losses['total'].item()

                # 收集预测
                preds = torch.sigmoid(outputs['site_logits']).detach().cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(sample['labels'].numpy())

        self.scheduler.step()

        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics['loss'] = total_loss / len(dataloader)

        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict:
        """评估"""
        self.model.eval()

        total_loss = 0
        all_preds, all_labels = [], []

        for batch in dataloader:
            for sample in batch:
                x = sample['node_features'].to(self.device)
                ei = sample['edge_index'].to(self.device)
                ea = sample['edge_attr'].to(self.device)

                outputs = self.model(x, ei, ea)

                labels_dict = {
                    'site_labels': sample['labels'],
                    'role_labels': sample['role_labels'],
                    'ec1_label': sample['ec1_label']
                }
                losses = self.criterion(outputs, labels_dict, self.device)
                total_loss += losses['total'].item()

                preds = torch.sigmoid(outputs['site_logits']).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(sample['labels'].numpy())

        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics['loss'] = total_loss / len(dataloader)

        return metrics

    def train(self, train_loader, val_loader, epochs: int = 100, save_dir: str = './models'):
        """完整训练流程"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        best_f1 = 0
        patience_counter = 0

        print(f"\n{'=' * 60}")
        print(f"开始训练 | Epochs: {epochs} | Device: {self.device}")
        print(f"{'=' * 60}\n")

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            print(f"Epoch {epoch + 1:3d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['f1']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f} "
                  f"MCC: {val_metrics['mcc']:.4f}")

            # 保存最佳模型
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0

                config = {
                    'node_dim': 33,
                    'edge_dim': 8,
                    'hidden_dim': 256,
                    'num_gnn_layers': 6,
                    'num_heads': 8,
                    'dropout': 0.2,
                    'num_ec1': 7,
                    'num_ec2': 70,
                    'num_ec3': 300
                }

                self.model.save_checkpoint(
                    str(save_dir / 'best_model.pt'),
                    config, self.optimizer, epoch, val_metrics
                )
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"\n早停于 Epoch {epoch + 1}")
                break

        print(f"\n{'=' * 60}")
        print(f"✓ 训练完成 | 最佳验证 F1: {best_f1:.4f}")
        print(f"{'=' * 60}")

        return best_f1
