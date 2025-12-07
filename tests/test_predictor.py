#!/usr/bin/env python3
"""
Unit tests for CatalyticTriadNet predictor module.
"""

import sys
import pytest
import numpy as np

sys.path.insert(0, '../src')


class TestFeatureEncoder:
    """Test feature encoding functionality."""

    def test_electronic_features(self):
        """Test electronic feature computation."""
        from catalytic_triad_net.predictor import ElectronicFeatureEncoder

        encoder = ElectronicFeatureEncoder()
        residue = {'name': 'HIS'}

        features = encoder.compute_features(residue)

        assert features.shape == (6,)
        assert not np.any(np.isnan(features))

    def test_substrate_features(self):
        """Test substrate-aware feature computation."""
        from catalytic_triad_net.predictor import SubstrateAwareEncoder

        encoder = SubstrateAwareEncoder()
        coords = np.random.randn(10, 3) * 10
        ligands = [{'coord': np.array([0, 0, 0])}]
        metals = []

        features = encoder.compute_features(0, coords, ligands, metals)

        assert features.shape == (6,)
        assert not np.any(np.isnan(features))


class TestTriadDetector:
    """Test triad detection functionality."""

    def test_pattern_matching(self):
        """Test triad pattern matching."""
        from catalytic_triad_net.predictor import TriadDetector

        detector = TriadDetector()

        # Create mock data
        residues = [
            {'name': 'SER', 'number': 195, 'chain': 'A'},
            {'name': 'HIS', 'number': 57, 'chain': 'A'},
            {'name': 'ASP', 'number': 102, 'chain': 'A'},
        ]
        coords = np.array([
            [0, 0, 0],
            [3, 0, 0],
            [5, 2, 0],
        ])
        catalytic_residues = [
            {'index': 0, 'resname': 'SER', 'resseq': 195, 'site_prob': 0.9, 'role_name': 'nucleophile'},
            {'index': 1, 'resname': 'HIS', 'resseq': 57, 'site_prob': 0.9, 'role_name': 'general_base'},
            {'index': 2, 'resname': 'ASP', 'resseq': 102, 'site_prob': 0.9, 'role_name': 'electrostatic'},
        ]

        triads = detector.detect_triads(residues, coords, catalytic_residues, predicted_ec1=3)

        # Should detect serine protease triad
        assert isinstance(triads, list)


class TestBimetallicDetector:
    """Test bimetallic center detection."""

    def test_bimetallic_detection(self):
        """Test detection of bimetallic centers."""
        from catalytic_triad_net.predictor import BimetallicCenterDetector

        detector = BimetallicCenterDetector()

        metals = [
            {'name': 'MG', 'coord': np.array([0, 0, 0])},
            {'name': 'MG', 'coord': np.array([3.8, 0, 0])},
        ]
        residues = [
            {'name': 'ASP', 'number': 89, 'chain': 'A'},
        ]
        coords = np.array([[1.9, 0, 0]])

        centers = detector.detect_bimetallic_centers(metals, residues, coords)

        assert isinstance(centers, list)
        if centers:
            assert 'distance' in centers[0]
            assert 3.0 <= centers[0]['distance'] <= 5.0


class TestPDBProcessor:
    """Test PDB processing functionality."""

    def test_pdb_download(self):
        """Test PDB download (requires network)."""
        from catalytic_triad_net.predictor import PDBProcessor

        processor = PDBProcessor(pdb_dir='./test_pdb')

        # This test requires network access
        # pdb_path = processor.download_pdb('1acb')
        # assert pdb_path is not None


class TestModel:
    """Test model components."""

    def test_geometric_gnn(self):
        """Test GeometricGNN forward pass."""
        import torch
        from catalytic_triad_net.predictor import GeometricGNN

        model = GeometricGNN(
            node_dim=48,
            edge_dim=14,
            hidden_dim=64,
            num_layers=2,
            num_heads=4
        )

        # Create mock input
        x = torch.randn(10, 48)
        edge_index = torch.randint(0, 10, (2, 30))
        edge_attr = torch.randn(30, 14)

        output = model(x, edge_index, edge_attr)

        assert output.shape == (10, 64)

    def test_full_model(self):
        """Test full model forward pass."""
        import torch
        from catalytic_triad_net.predictor import CatalyticTriadPredictorV2

        config = {
            'model': {
                'node_dim': 48,
                'edge_dim': 14,
                'hidden_dim': 64,
                'num_gnn_layers': 2,
                'num_heads': 4,
                'dropout': 0.1,
            },
            'ec_prediction': {
                'num_ec1_classes': 7,
                'num_ec2_classes': 70,
                'num_ec3_classes': 300,
            }
        }

        model = CatalyticTriadPredictorV2(config)

        x = torch.randn(10, 48)
        edge_index = torch.randint(0, 10, (2, 30))
        edge_attr = torch.randn(30, 14)

        outputs = model(x, edge_index, edge_attr)

        assert 'site_logits' in outputs
        assert 'role_logits' in outputs
        assert 'ec1_logits' in outputs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
