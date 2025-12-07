#!/usr/bin/env python3
"""
Example: Visualization
======================

This example demonstrates the visualization capabilities of CatalyticTriadNet.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from catalytic_triad_net import (
    NanozymeVisualizer,
    EnhancedCatalyticSiteInference
)


def main():
    print("=" * 60)
    print("Visualization Example")
    print("=" * 60)

    # Initialize visualizer
    viz = NanozymeVisualizer()

    # Example 1: Visualize prediction results
    print("\nExample 1: Visualize catalytic site predictions")
    print("-" * 40)

    # Create mock prediction results for demonstration
    mock_results = {
        'pdb_id': 'Demo_Enzyme',
        'num_residues': 200,
        'ec1_prediction': 3,
        'ec1_confidence': 0.92,
        'catalytic_residues': [
            {'index': 57, 'chain': 'A', 'resseq': 57, 'resname': 'SER',
             'site_prob': 0.95, 'role_name': 'nucleophile'},
            {'index': 102, 'chain': 'A', 'resseq': 102, 'resname': 'HIS',
             'site_prob': 0.91, 'role_name': 'general_base'},
            {'index': 195, 'chain': 'A', 'resseq': 195, 'resname': 'ASP',
             'site_prob': 0.88, 'role_name': 'electrostatic'},
            {'index': 189, 'chain': 'A', 'resseq': 189, 'resname': 'GLY',
             'site_prob': 0.72, 'role_name': 'transition_state_stabilizer'},
        ],
        'triads': [{
            'residues': [
                {'resname': 'SER', 'resseq': 57, 'index': 57, 'role_name': 'nucleophile'},
                {'resname': 'HIS', 'resseq': 102, 'index': 102, 'role_name': 'general_base'},
                {'resname': 'ASP', 'resseq': 195, 'index': 195, 'role_name': 'electrostatic'},
            ],
            'distances': {'SER-HIS': 3.2, 'HIS-ASP': 2.8, 'SER-ASP': 7.5},
            'confidence': 0.91
        }],
        'metals': [
            {'name': 'ZN', 'coord': [15, 10, 8]},
        ],
        'metal_centers': [{
            'metal': {'name': 'ZN'},
            'coordination_number': 4,
            'geometry': 'tetrahedral',
            'ligands': [
                {'resname': 'HIS', 'index': 94, 'distance': 2.1},
                {'resname': 'HIS', 'index': 96, 'distance': 2.0},
                {'resname': 'CYS', 'index': 99, 'distance': 2.3},
                {'resname': 'CYS', 'index': 102, 'distance': 2.2},
            ]
        }],
        'bimetallic_centers': [],
    }

    # Generate mock coordinates
    np.random.seed(42)
    mock_coords = np.random.randn(200, 3) * 15
    mock_coords[57] = [0, 0, 0]
    mock_coords[102] = [3, 1, 0.5]
    mock_coords[195] = [5, 3, 1]

    # Create visualizations
    viz.visualize(
        mock_results,
        mock_coords,
        output_dir="./viz_output",
        prefix="demo_enzyme",
        modes=['2d_graph', '2d_triad', '2d_metal', '3d_site']
    )
    print("  - Saved visualizations to ./viz_output/")

    # Example 2: Visualize diffusion model output
    print("\nExample 2: Visualize generated molecule")
    print("-" * 40)

    # Mock diffusion output
    node_types = np.array([0, 0, 1, 0, 0, 2, 0, 1])  # C, C, N, C, C, O, C, N
    edge_index = np.array([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]
    ])
    edge_types = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    mol_coords = np.random.randn(8, 3) * 3

    viz.visualize_diffusion(
        node_types,
        edge_index,
        mol_coords,
        edge_types,
        atom_list=['C', 'N', 'O'],
        output_dir="./viz_output",
        name="generated_molecule"
    )
    print("  - Saved molecule visualization to ./viz_output/")

    # Example 3: Export professional software scripts
    print("\nExample 3: Export PyMOL/ChimeraX scripts")
    print("-" * 40)

    viz.export_professional(
        mock_results,
        pdb_path="demo.pdb",
        output_dir="./viz_output",
        prefix="demo_enzyme"
    )
    print("  - Exported PyMOL script: ./viz_output/demo_enzyme.pml")
    print("  - Exported ChimeraX script: ./viz_output/demo_enzyme.cxc")
    print("  - Exported VMD script: ./viz_output/demo_enzyme.tcl")

    print("\nDone! Check ./viz_output/ for all generated files.")


if __name__ == "__main__":
    main()
