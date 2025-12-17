#!/usr/bin/env python3
"""
Two-stage transfer learning trainer:
1) Swiss-Prot 预训练（仅EC标签）
2) M-CSA 精调（全标签）
"""

import logging
import torch
from pathlib import Path
from typing import Iterable, Optional, List

from .trainer import CatalyticTriadTrainer

logger = logging.getLogger(__name__)


class TransferLearningTrainer(CatalyticTriadTrainer):
    """迁移学习训练器"""

    def _freeze_layers(self, layer_indices: Optional[Iterable[int]] = None):
        """冻结指定GNN层"""
        if not layer_indices:
            return
        for i in layer_indices:
            if i < len(self.model.gnn.layers):
                for p in self.model.gnn.layers[i].parameters():
                    p.requires_grad = False

    def pretrain(
        self,
        swiss_loader,
        epochs: int = 20,
        save_dir: str = './models'
    ) -> Path:
        """
        Swiss-Prot 预训练阶段（只训练EC头和主干）
        """
        save_root = Path(save_dir)
        save_root.mkdir(parents=True, exist_ok=True)

        # 记录原始权重
        original_weights = (self.criterion.w_site, self.criterion.w_role, self.criterion.w_ec1)
        self.criterion.w_site = 0.0
        self.criterion.w_role = 0.0
        self.criterion.w_ec1 = 1.0

        logger.info("=" * 60)
        logger.info("Stage 1: Swiss-Prot Pretraining (EC only)")
        logger.info("=" * 60)

        best_loss = float('inf')
        ckpt_path = save_root / 'swiss_pretrain_best.pt'

        for epoch in range(epochs):
            metrics = self.train_epoch(swiss_loader)
            loss = metrics.get('loss', 0.0)
            logger.info(f"[Pretrain] Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

            if loss < best_loss:
                best_loss = loss
                self.model.save_checkpoint(str(ckpt_path), {}, self.optimizer, epoch, {'loss': loss})
                logger.info(f"  ✓ Saved best model (loss: {loss:.4f})")

        # 如果没有保存过（所有epoch loss都没改善），保存最后一个epoch
        if not ckpt_path.exists():
            logger.warning("No improvement during pretraining, saving last epoch")
            self.model.save_checkpoint(str(ckpt_path), {}, self.optimizer, epochs-1, {'loss': loss})

        # 恢复权重
        self.criterion.w_site, self.criterion.w_role, self.criterion.w_ec1 = original_weights

        return ckpt_path

    def finetune(
        self,
        mcsa_train_loader,
        mcsa_val_loader,
        epochs: int = 100,
        freeze_layers: Optional[List[int]] = None,
        lr_scale: float = 0.1,
        save_dir: str = './models'
    ) -> float:
        """
        M-CSA 精调阶段（全标签）
        """
        logger.info("=" * 60)
        logger.info("Stage 2: M-CSA Fine-tuning (full tasks)")
        logger.info("=" * 60)

        # 冻结部分层
        self._freeze_layers(freeze_layers)

        # 重新创建优化器（只包含可训练参数，避免优化器状态污染）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        old_lr = self.optimizer.param_groups[0]['lr']
        old_weight_decay = self.optimizer.param_groups[0]['weight_decay']
        new_lr = old_lr * lr_scale

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=new_lr,
            weight_decay=old_weight_decay
        )

        # 重新创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        logger.info(f"Fine-tune learning rate: {new_lr:.6f}")
        logger.info(f"Trainable parameters: {len(trainable_params)}")

        best_f1 = self.train(mcsa_train_loader, mcsa_val_loader, epochs, save_dir)
        return best_f1

    def train_transfer_learning(
        self,
        swiss_loader,
        mcsa_train_loader,
        mcsa_val_loader,
        pretrain_epochs: int = 20,
        finetune_epochs: int = 100,
        freeze_layers: Optional[List[int]] = None,
        save_dir: str = './models/transfer'
    ) -> float:
        """完整两阶段训练"""
        self.pretrain(swiss_loader, pretrain_epochs, save_dir)
        best_f1 = self.finetune(
            mcsa_train_loader,
            mcsa_val_loader,
            epochs=finetune_epochs,
            freeze_layers=freeze_layers,
            save_dir=save_dir
        )
        return best_f1
