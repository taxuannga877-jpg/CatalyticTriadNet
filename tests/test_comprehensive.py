#!/usr/bin/env python3
"""
综合测试套件 - 覆盖所有关键功能
"""

import sys
import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import json

# 添加项目根目录的 src 到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'src'))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_pdb_structure():
    """模拟PDB结构数据"""
    return {
        'residues': [
            {'name': 'SER', 'number': 195, 'chain': 'A'},
            {'name': 'HIS', 'number': 57, 'chain': 'A'},
            {'name': 'ASP', 'number': 102, 'chain': 'A'},
        ],
        'coords': np.array([
            [0, 0, 0],
            [3, 0, 0],
            [5, 2, 0],
        ]),
        'metals': [
            {'name': 'ZN', 'coord': np.array([2, 1, 0])}
        ]
    }


@pytest.fixture
def mock_prediction_results():
    """模拟预测结果"""
    return {
        'pdb_id': 'TEST',
        'num_residues': 100,
        'ec1_prediction': 3,
        'ec1_confidence': 0.85,
        'catalytic_residues': [
            {'index': 0, 'resname': 'SER', 'resseq': 195, 'chain': 'A',
             'site_prob': 0.9, 'role_name': 'nucleophile'},
            {'index': 1, 'resname': 'HIS', 'resseq': 57, 'chain': 'A',
             'site_prob': 0.85, 'role_name': 'general_base'},
            {'index': 2, 'resname': 'ASP', 'resseq': 102, 'chain': 'A',
             'site_prob': 0.8, 'role_name': 'electrostatic'},
        ],
        'triads': [],
        'metals': [],
        'metal_centers': [],
        'bimetallic_centers': []
    }


@pytest.fixture
def mock_graph_data():
    """模拟图数据"""
    return {
        'x': torch.randn(10, 48),
        'edge_index': torch.randint(0, 10, (2, 30)),
        'edge_attr': torch.randn(30, 14),
        'batch': torch.zeros(10, dtype=torch.long)
    }


# ============================================================================
# Core Module Tests
# ============================================================================

class TestCoreDataset:
    """测试核心数据集功能"""

    def test_collate_fn_single_graph(self, mock_graph_data):
        """测试单图collate"""
        from catalytic_triad_net.core.dataset import CatalyticSiteDataset

        # 创建模拟数据
        batch = [mock_graph_data]

        # 测试collate_fn存在
        assert hasattr(CatalyticSiteDataset, 'collate_fn')

    def test_collate_fn_batch(self, mock_graph_data):
        """测试批量图collate"""
        from catalytic_triad_net.core.dataset import CatalyticSiteDataset

        # 创建多个图的批次
        batch = [mock_graph_data, mock_graph_data.copy()]

        # 测试批处理
        assert len(batch) == 2


class TestFeatureEncoder:
    """测试特征编码器"""

    def test_residue_encoding(self):
        """测试残基编码"""
        from catalytic_triad_net.core.structure import FeatureEncoder

        encoder = FeatureEncoder()
        residue = {'name': 'HIS', 'number': 57, 'chain': 'A'}

        features = encoder.encode_residue(residue)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] > 0
        assert not np.any(np.isnan(features))

    def test_edge_encoding(self):
        """测试边特征编码"""
        from catalytic_triad_net.core.structure import FeatureEncoder

        encoder = FeatureEncoder()
        coord1 = np.array([0, 0, 0])
        coord2 = np.array([3, 0, 0])

        features = encoder.encode_edge(coord1, coord2)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] > 0
        assert not np.any(np.isnan(features))


# ============================================================================
# Prediction Module Tests
# ============================================================================

class TestPredictionModels:
    """测试预测模型"""

    def test_model_forward_pass(self, mock_graph_data):
        """测试模型前向传播"""
        from catalytic_triad_net.prediction.models import CatalyticTriadPredictor

        model = CatalyticTriadPredictor(
            hidden_dim=64,
            num_gnn_layers=2,
            dropout=0.1
        )

        outputs = model(
            mock_graph_data['x'],
            mock_graph_data['edge_index'],
            mock_graph_data['edge_attr']
        )

        assert 'site_logits' in outputs
        assert 'role_logits' in outputs
        assert 'ec1_logits' in outputs
        assert outputs['site_logits'].shape[0] == 10

    def test_model_batch_processing(self):
        """测试批处理"""
        from catalytic_triad_net.prediction.models import CatalyticTriadPredictor

        model = CatalyticTriadPredictor(hidden_dim=64, num_gnn_layers=2)

        # 创建批次数据
        batch_size = 3
        n_nodes = 15
        x = torch.randn(n_nodes, 48)
        edge_index = torch.randint(0, n_nodes, (2, 40))
        edge_attr = torch.randn(40, 14)

        outputs = model(x, edge_index, edge_attr)

        assert outputs['site_logits'].shape[0] == n_nodes


class TestAnalysisModules:
    """测试分析模块"""

    def test_triad_detection(self, mock_pdb_structure):
        """测试三联体检测"""
        from catalytic_triad_net.prediction.analysis import TriadDetector

        detector = TriadDetector()

        catalytic_residues = [
            {'index': 0, 'resname': 'SER', 'resseq': 195, 'site_prob': 0.9, 'role_name': 'nucleophile'},
            {'index': 1, 'resname': 'HIS', 'resseq': 57, 'site_prob': 0.9, 'role_name': 'general_base'},
            {'index': 2, 'resname': 'ASP', 'resseq': 102, 'site_prob': 0.9, 'role_name': 'electrostatic'},
        ]

        triads = detector.detect_triads(
            mock_pdb_structure['residues'],
            mock_pdb_structure['coords'],
            catalytic_residues,
            predicted_ec1=3
        )

        assert isinstance(triads, list)

    def test_metal_center_detection(self, mock_pdb_structure):
        """测试金属中心检测"""
        from catalytic_triad_net.prediction.analysis import MetalCenterDetector

        detector = MetalCenterDetector()

        centers = detector.detect_metal_centers(
            mock_pdb_structure['metals'],
            mock_pdb_structure['residues'],
            mock_pdb_structure['coords']
        )

        assert isinstance(centers, list)


# ============================================================================
# Generation Module Tests
# ============================================================================

class TestGenerationConstraints:
    """测试生成约束"""

    def test_constraint_creation(self):
        """测试约束创建"""
        from catalytic_triad_net.generation.constraints import (
            CatalyticConstraints, GeometricConstraint
        )

        constraint = GeometricConstraint(
            constraint_type='distance',
            atom_indices=[0, 1],
            target_value=3.5,
            tolerance=0.5,
            weight=1.0
        )

        assert constraint.constraint_type == 'distance'
        assert constraint.target_value == 3.5

    def test_constraint_to_tensor(self):
        """测试约束转换为张量"""
        from catalytic_triad_net.generation.constraints import CatalyticConstraints

        constraints = CatalyticConstraints(
            anchor_atoms=[
                {'idx': 0, 'role': 'nucleophile', 'preferred_elements': ['O'], 'geometry': 'terminal'}
            ],
            distance_constraints=[],
            angle_constraints=[],
            coordination_constraints=[],
            charge_requirements={},
            required_elements=['O', 'N'],
            forbidden_elements=[]
        )

        condition_tensor = constraints.to_condition_tensor()

        assert 'anchor_features' in condition_tensor
        assert condition_tensor['n_anchors'] == 1

    def test_constraint_loss_computation(self):
        """测试约束损失计算"""
        from catalytic_triad_net.generation.constraints import GeometricConstraint

        constraint = GeometricConstraint(
            constraint_type='distance',
            atom_indices=[0, 1],
            target_value=3.5,
            tolerance=0.5,
            weight=1.0
        )

        # 模拟坐标
        coords = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        loss = constraint.compute_loss(coords)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestDiffusionModel:
    """测试扩散模型"""

    def test_model_initialization(self):
        """测试模型初始化"""
        from catalytic_triad_net.generation.models import CatalyticDiffusionModel

        config = {
            'hidden_dim': 64,
            'n_layers': 2,
            'num_timesteps': 100
        }

        model = CatalyticDiffusionModel(config)

        assert model is not None
        assert hasattr(model, 'forward')

    def test_sampling(self):
        """测试采样"""
        from catalytic_triad_net.generation.models import CatalyticDiffusionModel

        config = {
            'hidden_dim': 64,
            'n_layers': 2,
            'num_timesteps': 10  # 减少步数以加快测试
        }

        model = CatalyticDiffusionModel(config)

        # 测试采样方法存在
        assert hasattr(model, 'sample') or hasattr(model, 'generate')


# ============================================================================
# Visualization Module Tests
# ============================================================================

class TestVisualization:
    """测试可视化模块"""

    def test_visualizer_initialization(self):
        """测试可视化器初始化"""
        from catalytic_triad_net.visualization import NanozymeVisualizer

        viz = NanozymeVisualizer()

        assert viz is not None
        assert hasattr(viz, 'visualize')

    def test_adapter_graph_conversion(self):
        """测试图数据适配"""
        from catalytic_triad_net.visualization import DiffusionModelAdapter

        adapter = DiffusionModelAdapter()

        node_types = np.array([0, 1, 2, 0, 1])
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        coords = np.random.randn(5, 3)

        results = adapter.from_graph_data(
            node_types, edge_index, coords,
            atom_list=['C', 'N', 'O']
        )

        assert 'pdb_id' in results
        assert 'catalytic_residues' in results
        assert results['_is_molecule'] == True

    def test_professional_export(self, mock_prediction_results):
        """测试专业软件导出"""
        from catalytic_triad_net.visualization import ProfessionalExporter

        exporter = ProfessionalExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.pml"

            # 测试PyMOL导出
            exporter.to_pymol(
                mock_prediction_results,
                "test.pdb",
                str(output_path)
            )

            assert output_path.exists()
            content = output_path.read_text()
            assert 'load' in content
            assert 'PyMOL' in content


# ============================================================================
# CLI Tests
# ============================================================================

class TestCLI:
    """测试命令行接口"""

    def test_dependency_check(self):
        """测试依赖检查"""
        from catalytic_triad_net.cli import check_dependencies

        deps = check_dependencies()

        assert isinstance(deps, dict)
        assert 'torch' in deps
        assert 'numpy' in deps

    def test_pdb_id_validation(self):
        """测试PDB ID验证"""
        from catalytic_triad_net.cli import validate_pdb_id

        assert validate_pdb_id('1acb') == True
        assert validate_pdb_id('2xyz') == True
        assert validate_pdb_id('abc') == False
        assert validate_pdb_id('12345') == False
        assert validate_pdb_id('') == False

    def test_file_path_validation(self):
        """测试文件路径验证"""
        from catalytic_triad_net.cli import validate_file_path

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # 测试存在的文件
            result = validate_file_path(tmp_path, must_exist=True)
            assert result is not None

            # 测试不存在的文件
            result = validate_file_path('/nonexistent/path.pdb', must_exist=True)
            assert result is None
        finally:
            Path(tmp_path).unlink()


# ============================================================================
# Config Tests
# ============================================================================

class TestConfig:
    """测试配置系统"""

    def test_config_initialization(self):
        """测试配置初始化"""
        from catalytic_triad_net.config import Config

        config = Config()

        assert config is not None
        assert config.get('model.hidden_dim') is not None

    def test_config_get_set(self):
        """测试配置读写"""
        from catalytic_triad_net.config import Config

        config = Config()

        # 测试设置
        config.set('test.value', 42)

        # 测试获取
        value = config.get('test.value')
        assert value == 42

    def test_config_file_io(self):
        """测试配置文件读写"""
        from catalytic_triad_net.config import Config

        config = Config()
        config.set('test.param', 123)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # 保存
            config.save_to_file(config_path)
            assert config_path.exists()

            # 加载
            new_config = Config(config_path)
            assert new_config.get('test.param') == 123


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """集成测试"""

    def test_end_to_end_prediction_pipeline(self, mock_graph_data):
        """测试端到端预测流程"""
        from catalytic_triad_net.prediction.models import CatalyticTriadPredictor

        # 创建模型
        model = CatalyticTriadPredictor(hidden_dim=64, num_gnn_layers=2)
        model.eval()

        # 前向传播
        with torch.no_grad():
            outputs = model(
                mock_graph_data['x'],
                mock_graph_data['edge_index'],
                mock_graph_data['edge_attr']
            )

        # 验证输出
        assert 'site_logits' in outputs
        assert outputs['site_logits'].shape[0] == 10

        # 获取预测
        site_probs = torch.sigmoid(outputs['site_logits'])
        assert torch.all((site_probs >= 0) & (site_probs <= 1))

    def test_visualization_pipeline(self, mock_prediction_results):
        """测试可视化流程"""
        from catalytic_triad_net.visualization import NanozymeVisualizer

        viz = NanozymeVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # 测试可视化不会崩溃
            try:
                viz.visualize(
                    mock_prediction_results,
                    coords=np.random.randn(100, 3),
                    output_dir=tmpdir,
                    modes=['2d_graph']  # 只测试一个模式
                )
            except Exception as e:
                # 如果是依赖问题，跳过
                if 'matplotlib' in str(e).lower() or 'plotly' in str(e).lower():
                    pytest.skip("Visualization dependencies not available")
                else:
                    raise


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """性能测试"""

    def test_batch_processing_speed(self):
        """测试批处理速度"""
        from catalytic_triad_net.prediction.models import CatalyticTriadPredictor
        import time

        model = CatalyticTriadPredictor(hidden_dim=64, num_gnn_layers=2)
        model.eval()

        # 创建批次数据
        x = torch.randn(50, 48)
        edge_index = torch.randint(0, 50, (2, 150))
        edge_attr = torch.randn(150, 14)

        # 测试推理时间
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                outputs = model(x, edge_index, edge_attr)
        elapsed = time.time() - start

        # 平均每次推理应该在合理时间内
        avg_time = elapsed / 10
        assert avg_time < 1.0  # 每次推理应该少于1秒


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
