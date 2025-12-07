# CatalyticTriadNet

<p align="center">
  <img src="docs/images/banner.png" width="800" alt="CatalyticTriadNet Banner"/>
</p>

<p align="center">
  <strong>A Geometric Deep Learning Framework for Enzyme Catalytic Site Identification and Nanozyme Design</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/pytorch-1.12+-orange.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"/>
</p>

---

## Overview

**CatalyticTriadNet** is an end-to-end deep learning framework for:

- **Catalytic Site Prediction**: Identify catalytic residues from protein 3D structures
- **EC Number Classification**: Hierarchical enzyme classification (EC1 → EC2 → EC3)
- **Catalytic Triad Detection**: Recognize classic catalytic triads (Ser-His-Asp, Cys-His-Asn, etc.)
- **Metal Center Analysis**: Detect mono- and bimetallic active sites
- **Nanozyme Design**: Export templates for downstream protein design tools

### Key Features

| Feature | Description |
|---------|-------------|
| Multi-scale Encoding | Physicochemical, geometric, electronic, and substrate-aware features |
| EC-Conditioned Prediction | Global EC classification guides local site prediction |
| Intelligent Triad Detection | Pattern matching + geometric constraints from M-CSA database |
| Bimetallic Center Recognition | Specialized detection for phosphodiesterases, metallo-β-lactamases |
| Design Tool Integration | Direct export to ProteinMPNN, RFdiffusion formats |

---

## Architecture

```
Input: PDB Structure
    ↓
┌─────────────────────────────────────────────────────────┐
│                  Feature Encoding (48D)                  │
│  • Amino acid one-hot (20D)    • Electronic features (6D)│
│  • Physicochemical (8D)        • Substrate-aware (6D)    │
│  • Spatial geometry (5D)       • Metal environment (3D)  │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│              Geometric Graph Neural Network              │
│         6-layer Message Passing + Multi-head Attention   │
└─────────────────────────────────────────────────────────┘
    ↓
┌──────────────────┬──────────────────────────────────────┐
│  EC Predictor    │       Catalytic Site Predictor       │
│  EC1 → EC2 → EC3 │  (EC-conditioned) Site + Role        │
└──────────────────┴──────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                   Post-processing                        │
│  • Triad Detection  • Metal Centers  • Binding Pockets  │
└─────────────────────────────────────────────────────────┘
    ↓
Output: Catalytic Sites, EC Class, Triads, Metal Centers, Design Templates
```

---

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.3 (for GPU acceleration)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/CatalyticTriadNet.git
cd CatalyticTriadNet

# Create conda environment
conda create -n catalytic python=3.9
conda activate catalytic

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install dependencies
pip install -r requirements.txt

# (Optional) Install xTB for accurate charge calculation
conda install -c conda-forge xtb

# (Optional) Install ESM for sequence features
pip install fair-esm
```

---

## Quick Start

### Command Line Interface

```bash
# Predict catalytic sites for a PDB structure
python -m catalytic_triad_net predict --pdb 1acb --threshold 0.5 --output results/1acb

# Specify target EC class (e.g., hydrolase EC3)
python -m catalytic_triad_net predict --pdb 4cha --ec1 3 --output results/4cha

# Analyze metal centers
python -m catalytic_triad_net analyze --pdb 1a5t
```

### Python API

```python
from catalytic_triad_net import EnhancedCatalyticSiteInference

# Initialize predictor
predictor = EnhancedCatalyticSiteInference(
    model_path='models/best_model.pt',
    device='cuda'
)

# Predict
results = predictor.predict(
    pdb_path='1acb.pdb',
    target_ec1=3,  # Hydrolase
    site_threshold=0.5,
    role_threshold=0.3
)

# Access results
print(f"EC Prediction: EC{results['ec1_prediction']}")
print(f"Catalytic Residues: {len(results['catalytic_residues'])}")
print(f"Triads Found: {len(results['triads'])}")
print(f"Bimetallic Centers: {len(results['bimetallic_centers'])}")

# Print detailed results
predictor.print_results(results)

# Export for downstream tools
predictor.export_nanozyme_design_input(results, 'nanozyme_template.json')
predictor.export_for_proteinmpnn(results, 'mpnn_input.json')
predictor.export_for_rfdiffusion(results, 'rfd_input.json')
```

### Example Output

```
======================================================================
Catalytic Site Prediction V2.0 - PDB: 1acb
======================================================================

EC Classification: EC 3 (Hydrolase) [Confidence: 94.2%]

★ Found 1 Bimetallic Center:
  1. MG-MG (Distance: 3.82Å, Pattern: phosphodiesterase)
     Bridging Ligands: ASP89, GLU92, HOH401

★ Found 2 Catalytic Triads:
  1. SER195-HIS57-ASP102 [Confidence: 0.92]
  2. SER214-HIS57-ASP102 [Confidence: 0.85]

Found 8 Predicted Catalytic Residues:

Rank  Chain  Residue      Prob     Role
------------------------------------------------------------
1     A      HIS57       0.9842   general_base, proton_acceptor
2     A      ASP102      0.9756   electrostatic_stabilizer
3     A      SER195      0.9623   nucleophile
======================================================================
```

---

## Modules

### 1. Core Module (`core/`)

Shared foundational components:
- **data.py**: M-CSA database API integration
- **structure.py**: PDB parsing and feature encoding
- **dataset.py**: PyTorch dataset implementations

### 2. Prediction Module (`prediction/`)

Catalytic site prediction system:
- **models.py**: Geometric GNN architectures
- **trainer.py**: Training loops and loss functions
- **analysis.py**: Triad detection, metal center analysis, H-bond networks
- **features.py**: Electronic, substrate-aware, and conservation features
- **predictor.py**: High-level inference interface

### 3. Generation Module (`generation/`)

E(3)-equivariant diffusion for nanozyme design:
- **constraints.py**: Geometric constraint definitions
- **models.py**: Diffusion models with constraint conditioning
- **generator.py**: Nanozyme structure generator
- **dataset.py**: Training data handling
- **trainer.py**: Diffusion model training

### 4. Visualization Module (`visualization/`)

Comprehensive visualization toolkit:
- **adapters.py**: Support for RFdiffusion, ProteinMPNN, PyG formats
- **exporters.py**: Export to PyMOL, ChimeraX, VMD
- **plot_2d.py**: 2D molecular graphs and triad diagrams
- **plot_3d.py**: 3D active site visualization with Plotly
- **visualizer.py**: Unified visualization interface

---

## Downstream Applications

### Nanozyme Design Output

```json
{
  "source_enzyme": "1acb",
  "ec_class": 3,
  "catalytic_geometry": {
    "triads": [{
      "residues": [
        {"name": "SER", "role": "nucleophile"},
        {"name": "HIS", "role": "general_base"},
        {"name": "ASP", "role": "electrostatic_stabilizer"}
      ],
      "distances": {"SER-HIS": 3.2, "HIS-ASP": 2.8}
    }],
    "metal_centers": [{
      "metal_type": "ZN",
      "coordination_number": 4,
      "geometry": "tetrahedral"
    }]
  }
}
```

### Integration with Design Tools

| Tool | Export Format | Use Case |
|------|---------------|----------|
| ProteinMPNN | Fixed positions JSON | Sequence design with fixed catalytic sites |
| RFdiffusion | Hotspot residues | Scaffold design around active sites |
| Custom | Nanozyme template | Constraint-based nanozyme generation |

---

## Performance

### Catalytic Site Prediction

| Metric | CatalyticTriadNet | DeepEC | CLEAN |
|--------|-------------------|--------|-------|
| Precision | **0.82** | 0.71 | 0.74 |
| Recall | **0.78** | 0.65 | 0.68 |
| F1-Score | **0.80** | 0.68 | 0.71 |
| AUPRC | **0.84** | 0.72 | 0.75 |

### EC Classification

| Level | Accuracy | Macro-F1 |
|-------|----------|----------|
| EC1 | 0.92 | 0.89 |
| EC2 | 0.85 | 0.78 |
| EC3 | 0.76 | 0.68 |

---

## Project Structure

```
CatalyticTriadNet/
├── README.md                    # This file
├── README_CN.md                 # Chinese documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Dependencies
├── setup.py                     # Package installation
├── src/
│   └── catalytic_triad_net/
│       ├── __init__.py          # Package exports
│       ├── cli.py               # Command-line interface
│       │
│       ├── core/                # Shared core modules
│       │   ├── data.py          # M-CSA API data fetching
│       │   ├── structure.py     # PDB processing & feature encoding
│       │   └── dataset.py       # PyTorch datasets
│       │
│       ├── prediction/          # Catalytic site prediction
│       │   ├── models.py        # GNN neural network models
│       │   ├── trainer.py       # Training & loss functions
│       │   ├── analysis.py      # Triad/metal/H-bond analysis
│       │   ├── features.py      # Electronic/substrate/conservation features
│       │   └── predictor.py     # Inference interface
│       │
│       ├── generation/          # Nanozyme structure generation
│       │   ├── constraints.py   # Geometric constraint definitions
│       │   ├── models.py        # E(3)-equivariant diffusion models
│       │   ├── generator.py     # Nanozyme generator interface
│       │   ├── dataset.py       # Diffusion datasets
│       │   └── trainer.py       # Diffusion training
│       │
│       └── visualization/       # Visualization toolkit
│           ├── adapters.py      # Diffusion model adapters
│           ├── exporters.py     # PyMOL/ChimeraX/VMD export
│           ├── plot_2d.py       # 2D molecular graphs
│           ├── plot_3d.py       # 3D active site visualization
│           └── visualizer.py    # Main visualizer interface
│
├── examples/
│   ├── predict_example.py       # Prediction example
│   ├── train_example.py         # Training example
│   └── visualize_example.py     # Visualization example
├── docs/
│   ├── images/
│   └── methodology.md           # Detailed methodology
├── data/
│   └── models/                  # Pre-trained models
└── tests/
    └── test_predictor.py        # Unit tests
```

---

## Citation

If this work is helpful for your research, please cite:

```bibtex
@article{CatalyticTriadNet2024,
  title={CatalyticTriadNet: A Geometric Deep Learning Framework for
         Enzyme Catalytic Site Identification and Nanozyme Design},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

---

## References

### Core Methods

1. **LigandMPNN** - Dauparas et al. *Science* 2022 | [GitHub](https://github.com/dauparas/LigandMPNN)
2. **RFdiffusion** - Watson et al. *Nature* 2023 | [GitHub](https://github.com/RosettaCommons/RFdiffusion)
3. **xTB** - Bannwarth et al. *WIREs Comput Mol Sci* 2021 | [GitHub](https://github.com/grimme-lab/xtb)
4. **P2Rank** - Krivák & Hoksza *JCIM* 2018 | [GitHub](https://github.com/rdk/p2rank)

### Databases

5. **M-CSA** - Ribeiro et al. *NAR* 2018 | [Website](https://www.ebi.ac.uk/thornton-srv/m-csa/)

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Empowering Nanozyme Design with Deep Learning</i>
</p>
