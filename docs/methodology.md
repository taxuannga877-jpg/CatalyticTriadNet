# Methodology

This document provides detailed technical information about the CatalyticTriadNet framework.

## Table of Contents

1. [Feature Engineering](#feature-engineering)
2. [Network Architecture](#network-architecture)
3. [Training Procedure](#training-procedure)
4. [Inference Pipeline](#inference-pipeline)

---

## Feature Engineering

### Node Features (48 dimensions)

| Category | Dimensions | Description |
|----------|------------|-------------|
| Amino Acid One-hot | 20 | Standard amino acid encoding |
| Physicochemical | 8 | Hydrophobicity, volume, charge, polarity, aromaticity, pKa, catalytic prior, conservation |
| Spatial Geometry | 5 | Local density (8Å/12Å), average neighbor distance, burial depth, local curvature |
| Metal Environment | 3 | Distance to nearest metal, metal neighbor count, metal-shell indicator |
| Electronic | 6 | Sidechain charge, max partial charge, electronegativity, polarizability, redox activity, reactivity |
| Substrate-aware | 6 | Ligand distance, normalized distance, ligand neighbors, pocket indicator, exposure, interaction potential |

### Edge Features (14 dimensions)

| Category | Dimensions | Description |
|----------|------------|-------------|
| Geometric | 8 | CA distance, CB distance, inverse distance, RBF encoding, sequence distance, direction vector |
| Interaction Type | 3 | Hydrogen bond, ionic, aromatic stacking |
| H-bond Details | 3 | H-bond indicator, H-bond distance, H-bond strength |

---

## Network Architecture

### Geometric Message Passing

The core of our network uses attention-based message passing with edge feature integration:

```
m_ij = Attention(q_i, k_j, e_ij) · v_j

α_ij = softmax_j((q_i · k_j) / √d + W_e · e_ij)

h_i^(l+1) = LayerNorm(h_i^(l) + Σ_j α_ij · m_ij)
```

### Hierarchical EC Prediction

EC classification follows a conditional cascade:

```
P(EC1) = softmax(W1 · g)
P(EC2|EC1) = softmax(W2 · [g; P(EC1)])
P(EC3|EC1,EC2) = softmax(W3 · [g; P(EC1); P(EC2)])
```

### EC-Conditioned Site Prediction

Site prediction is conditioned on EC1 probabilities:

```
h_i^cond = MLP([h_i; P(EC1)])
P(catalytic_i) = σ(W_site · h_i^cond)
```

---

## Training Procedure

### Loss Function

Multi-task loss combining:
- Site prediction: Focal loss with class weighting
- Role classification: Cross-entropy
- EC classification: Hierarchical cross-entropy

### Data Augmentation

- Random rotation (SO(3))
- Random translation
- Coordinate noise injection

---

## Inference Pipeline

1. **Structure Parsing**: Extract residues, coordinates, metals, ligands
2. **Feature Encoding**: Compute 48D node features and 14D edge features
3. **GNN Forward**: 6-layer message passing
4. **EC Prediction**: Hierarchical classification
5. **Site Prediction**: EC-conditioned residue scoring
6. **Post-processing**: Triad detection, metal center analysis
7. **Export**: Generate design templates

---

## References

See main README for full reference list.
