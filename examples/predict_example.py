#!/usr/bin/env python3
"""
Example: Catalytic Site Prediction
==================================

This example demonstrates how to use CatalyticTriadNet to predict
catalytic sites from a protein structure.
"""

import sys
sys.path.insert(0, '../src')

from catalytic_triad_net import EnhancedCatalyticSiteInference


def main():
    # Initialize predictor
    # Note: For production use, provide a trained model path
    predictor = EnhancedCatalyticSiteInference(
        model_path=None,  # Uses random initialization for demo
        device='cpu'
    )

    # Example 1: Predict from PDB ID (will download automatically)
    print("=" * 60)
    print("Example 1: Predict catalytic sites for Trypsin (1ACB)")
    print("=" * 60)

    results = predictor.predict(
        pdb_path='1acb',  # PDB ID - will be downloaded
        site_threshold=0.3,
        role_threshold=0.2
    )

    # Print results
    predictor.print_results(results, top_k=10)

    # Access specific results
    print(f"\nSummary:")
    print(f"  - PDB ID: {results['pdb_id']}")
    print(f"  - EC Prediction: EC{results['ec1_prediction']}")
    print(f"  - Confidence: {results['ec1_confidence']:.2%}")
    print(f"  - Catalytic Residues: {len(results['catalytic_residues'])}")
    print(f"  - Triads Found: {len(results['triads'])}")
    print(f"  - Metal Centers: {len(results['metal_centers'])}")

    # Example 2: Export for downstream tools
    print("\n" + "=" * 60)
    print("Example 2: Export for downstream design tools")
    print("=" * 60)

    # Export for nanozyme design
    predictor.export_nanozyme_design_input(
        results,
        'output_nanozyme_template.json'
    )
    print("  - Exported: output_nanozyme_template.json")

    # Export for ProteinMPNN
    predictor.export_for_proteinmpnn(
        results,
        'output_proteinmpnn.json'
    )
    print("  - Exported: output_proteinmpnn.json")

    # Export for RFdiffusion
    predictor.export_for_rfdiffusion(
        results,
        'output_rfdiffusion.json'
    )
    print("  - Exported: output_rfdiffusion.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
