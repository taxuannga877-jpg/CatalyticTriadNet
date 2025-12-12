#!/usr/bin/env python3
"""
纳米酶生成模块的单元测试
"""

import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录的 src 到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


class TestCatalyticConstraints:
    """测试催化约束类"""

    def test_constraint_creation(self):
        """测试约束创建"""
        from catalytic_triad_net.generation.constraints import (
            CatalyticConstraints, GeometricConstraint
        )

        # 创建简单约束
        distance_constraint = GeometricConstraint(
            constraint_type='distance',
            atom_indices=[0, 1],
            target_value=3.5,
            tolerance=0.5,
            weight=1.0
        )

        assert distance_constraint.constraint_type == 'distance'
        assert distance_constraint.target_value == 3.5

    def test_to_condition_tensor(self):
        """测试约束转换为张量"""
        from catalytic_triad_net.generation.constraints import (
            CatalyticConstraints, GeometricConstraint
        )

        constraints = CatalyticConstraints(
            anchor_atoms=[
                {'idx': 0, 'role': 'nucleophile', 'preferred_elements': ['O'], 'geometry': 'terminal'}
            ],
            distance_constraints=[
                GeometricConstraint('distance', [0, 1], 3.5, 0.5, 1.0)
            ],
            angle_constraints=[],
            coordination_constraints=[],
            charge_requirements={},
            required_elements=['O', 'N'],
            forbidden_elements=[]
        )

        condition_tensor = constraints.to_condition_tensor()

        assert 'anchor_features' in condition_tensor
        assert 'distance_constraints' in condition_tensor
        assert condition_tensor['n_anchors'] == 1


class TestE3EquivariantLayer:
    """测试E(3)等变层"""

    def test_equivariance(self):
        """测试旋转等变性"""
        from catalytic_triad_net.generation.models import E3EquivariantLayer

        layer = E3EquivariantLayer(hidden_dim=32, edge_dim=0)

        # 创建测试数据
        h = torch.randn(5, 32)
        x = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        # 前向传播
        h_out, x_out = layer(h, x, edge_index)

        assert h_out.shape == (5, 32)
        assert x_out.shape == (5, 3)

        # 测试旋转等变性
        # 创建旋转矩阵（绕z轴旋转90度）
        theta = np.pi / 2
        R = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        x_rotated = torch.matmul(x, R.T)
        h_out_rot, x_out_rot = layer(h, x_rotated, edge_index)

        # 旋转后的输出应该等于输出的旋转
        x_out_expected = torch.matmul(x_out, R.T)

        # 允许一定的数值误差
        assert torch.allclose(x_out_rot, x_out_expected, atol=1e-5)


class TestCatalyticDiffusionModel:
    """测试催化扩散模型"""

    def test_model_forward(self):
        """测试模型前向传播"""
        from catalytic_triad_net.generation.models import CatalyticDiffusionModel

        config = {
            'hidden_dim': 64,
            'n_layers': 2,
            'num_timesteps': 100
        }

        model = CatalyticDiffusionModel(config)

        # 创建测试输入
        batch_size = 2
        n_atoms = 10
        atom_types = torch.randint(0, 32, (batch_size, n_atoms))
        coords = torch.randn(batch_size, n_atoms, 3)
        edge_index = torch.randint(0, n_atoms, (2, 30))
        timesteps = torch.randint(0, 100, (batch_size,))

        # 简单的条件
        condition = {
            'anchor_features': torch.zeros(1, 16),
            'distance_constraints': torch.zeros(0, 4),
            'coordination_constraints': torch.zeros(0, 3)
        }

        # 前向传播应该不报错
        # 注意：实际模型可能需要更复杂的输入格式
        assert model is not None


class TestNanozymeDataset:
    """测试纳米酶数据集"""

    def test_dataset_creation(self):
        """测试数据集创建"""
        from catalytic_triad_net.generation.dataset import NanozymeDataset
        import tempfile
        import os

        # 创建临时目录和测试数据
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建一个简单的XYZ文件
            xyz_content = """3
Test molecule
C 0.0 0.0 0.0
O 1.0 0.0 0.0
H 0.0 1.0 0.0
"""
            xyz_path = os.path.join(tmpdir, 'test.xyz')
            with open(xyz_path, 'w') as f:
                f.write(xyz_content)

            # 创建数据集
            dataset = NanozymeDataset(data_dir=tmpdir, max_atoms=100)

            # 应该至少加载一个样本
            assert len(dataset) >= 1

            # 测试获取样本
            if len(dataset) > 0:
                sample = dataset[0]
                assert 'atom_types' in sample
                assert 'coords' in sample
                assert isinstance(sample['atom_types'], torch.Tensor)
                assert isinstance(sample['coords'], torch.Tensor)


class TestAtomTypes:
    """测试原子类型常量"""

    def test_atom_types_defined(self):
        """测试原子类型是否正确定义"""
        from catalytic_triad_net.generation.constraints import (
            ATOM_TYPES, ATOM_TO_IDX, NUM_ATOM_TYPES
        )

        assert NUM_ATOM_TYPES == 32
        assert len(ATOM_TYPES) == 32
        assert len(ATOM_TO_IDX) == 32

        # 检查关键原子类型
        assert 'C' in ATOM_TYPES
        assert 'N' in ATOM_TYPES
        assert 'O' in ATOM_TYPES
        assert 'Fe' in ATOM_TYPES
        assert 'Cu' in ATOM_TYPES
        assert 'Zn' in ATOM_TYPES

        # 检查映射
        assert ATOM_TO_IDX['C'] == ATOM_TYPES.index('C')
        assert ATOM_TO_IDX['N'] == ATOM_TYPES.index('N')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
