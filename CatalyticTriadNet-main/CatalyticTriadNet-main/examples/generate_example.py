#!/usr/bin/env python3
"""
Example: Nanozyme Structure Generation
======================================

This example demonstrates how to use the diffusion model to generate
nanozyme structures based on catalytic constraints.
"""

import sys
sys.path.insert(0, '../src')

from catalytic_triad_net import (
    CatalyticNanozymeGenerator,
    CatalyticConstraints
)


def main():
    print("=" * 60)
    print("Nanozyme Structure Generation Example")
    print("=" * 60)

    # Initialize generator
    generator = CatalyticNanozymeGenerator(
        model_path=None,  # Uses random initialization for demo
        device='cpu'
    )

    # Example 1: Generate from constraint file
    print("\nExample 1: Generate from catalytic constraints")
    print("-" * 40)

    # Create sample constraints programmatically
    # In practice, load from CatalyticTriadNet output
    from catalytic_triad_net.diffusion import GeometricConstraint

    constraints = CatalyticConstraints(
        anchor_atoms=[
            {'idx': 0, 'role': 'nucleophile', 'preferred_elements': ['O', 'S']},
            {'idx': 1, 'role': 'general_base', 'preferred_elements': ['N']},
            {'idx': 2, 'role': 'electrostatic', 'preferred_elements': ['O']},
        ],
        distance_constraints=[
            GeometricConstraint(
                constraint_type='distance',
                atom_indices=[0, 1],
                target_value=3.2,
                tolerance=0.5,
                weight=1.0
            ),
            GeometricConstraint(
                constraint_type='distance',
                atom_indices=[1, 2],
                target_value=2.8,
                tolerance=0.5,
                weight=1.0
            ),
        ],
        angle_constraints=[],
        coordination_constraints=[],
        charge_requirements={},
        required_elements=['C', 'N', 'O'],
        forbidden_elements=[]
    )

    # Generate structures
    print("Generating 3 nanozyme structures...")
    results = generator.generate(
        constraints=constraints,
        n_samples=3,
        n_atoms=30,
        guidance_scale=2.0,
        refine_steps=50
    )

    # Save results
    for i, result in enumerate(results):
        output_file = f'generated_nanozyme_{i:03d}.xyz'
        generator.to_xyz(result, output_file)
        print(f"  - Saved: {output_file}")

        # Print constraint satisfaction
        sat = result['constraint_satisfaction']
        n_sat = sum(1 for d in sat.get('distance', []) if d['satisfied'])
        n_total = len(sat.get('distance', []))
        print(f"    Constraints satisfied: {n_sat}/{n_total}")

    # Example 2: Generate with metal center
    print("\nExample 2: Generate with metal coordination")
    print("-" * 40)

    metal_constraints = CatalyticConstraints(
        anchor_atoms=[
            {'idx': 0, 'role': 'metal_center', 'preferred_elements': ['Fe']},
            {'idx': 1, 'role': 'metal_ligand', 'preferred_elements': ['N']},
            {'idx': 2, 'role': 'metal_ligand', 'preferred_elements': ['N']},
            {'idx': 3, 'role': 'metal_ligand', 'preferred_elements': ['O']},
            {'idx': 4, 'role': 'metal_ligand', 'preferred_elements': ['O']},
        ],
        distance_constraints=[
            GeometricConstraint('distance', [0, 1], 2.0, 0.3, 1.5),
            GeometricConstraint('distance', [0, 2], 2.0, 0.3, 1.5),
            GeometricConstraint('distance', [0, 3], 2.1, 0.3, 1.5),
            GeometricConstraint('distance', [0, 4], 2.1, 0.3, 1.5),
        ],
        angle_constraints=[],
        coordination_constraints=[{
            'metal_idx': 0,
            'coordination_number': 4,
            'geometry': 'tetrahedral',
            'ligand_types': ['N', 'N', 'O', 'O']
        }],
        charge_requirements={},
        required_elements=['Fe', 'C', 'N', 'O'],
        forbidden_elements=[]
    )

    print("Generating metal-containing nanozyme...")
    metal_results = generator.generate(
        constraints=metal_constraints,
        n_samples=1,
        n_atoms=25,
        guidance_scale=3.0
    )

    generator.to_xyz(metal_results[0], 'generated_metal_nanozyme.xyz')
    generator.to_mol(metal_results[0], 'generated_metal_nanozyme.mol')
    print("  - Saved: generated_metal_nanozyme.xyz")
    print("  - Saved: generated_metal_nanozyme.mol")

    print("\nDone!")


if __name__ == "__main__":
    main()
