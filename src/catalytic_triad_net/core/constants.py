"""
Constants and properties for amino acids and catalytic residues.

This module contains biochemical properties, catalytic priors, and other
constants used throughout the codebase.
"""

import numpy as np
from typing import Dict, List

# Amino acid one-letter codes
AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

# Amino acid properties
# Each amino acid is represented by a vector of biochemical properties:
# [hydrophobicity, charge, polarity, aromaticity, size, flexibility,
#  hydrogen_bond_donor, hydrogen_bond_acceptor]
AA_PROPERTIES: Dict[str, np.ndarray] = {
    # Hydrophobic amino acids
    'A': np.array([0.62, 0.0, 0.0, 0.0, 0.3, 0.8, 0.0, 0.0]),  # Alanine: small, hydrophobic
    'V': np.array([0.76, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0]),  # Valine: branched, hydrophobic
    'L': np.array([0.76, 0.0, 0.0, 0.0, 0.6, 0.5, 0.0, 0.0]),  # Leucine: branched, hydrophobic
    'I': np.array([0.76, 0.0, 0.0, 0.0, 0.6, 0.3, 0.0, 0.0]),  # Isoleucine: branched, hydrophobic
    'M': np.array([0.64, 0.0, 0.0, 0.0, 0.7, 0.6, 0.0, 0.0]),  # Methionine: sulfur-containing
    'F': np.array([0.88, 0.0, 0.0, 1.0, 0.8, 0.4, 0.0, 0.0]),  # Phenylalanine: aromatic, hydrophobic
    'W': np.array([0.88, 0.0, 0.0, 1.0, 1.0, 0.3, 1.0, 0.0]),  # Tryptophan: aromatic, large
    'P': np.array([0.36, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0]),  # Proline: rigid, helix breaker

    # Polar uncharged amino acids
    'S': np.array([0.18, 0.0, 1.0, 0.0, 0.3, 0.7, 1.0, 1.0]),  # Serine: hydroxyl group
    'T': np.array([0.18, 0.0, 1.0, 0.0, 0.4, 0.6, 1.0, 1.0]),  # Threonine: hydroxyl group
    'C': np.array([0.29, 0.0, 1.0, 0.0, 0.4, 0.6, 0.0, 0.0]),  # Cysteine: thiol group, disulfide bonds
    'Y': np.array([0.41, 0.0, 1.0, 1.0, 0.9, 0.4, 1.0, 1.0]),  # Tyrosine: aromatic, hydroxyl
    'N': np.array([0.09, 0.0, 1.0, 0.0, 0.5, 0.7, 1.0, 1.0]),  # Asparagine: amide group
    'Q': np.array([0.09, 0.0, 1.0, 0.0, 0.6, 0.7, 1.0, 1.0]),  # Glutamine: amide group
    'G': np.array([0.48, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # Glycine: smallest, flexible

    # Positively charged amino acids
    'K': np.array([0.05, 1.0, 1.0, 0.0, 0.7, 0.8, 1.0, 0.0]),  # Lysine: basic, long side chain
    'R': np.array([0.05, 1.0, 1.0, 0.0, 0.8, 0.7, 1.0, 0.0]),  # Arginine: basic, guanidinium group
    'H': np.array([0.13, 0.5, 1.0, 1.0, 0.6, 0.6, 1.0, 1.0]),  # Histidine: aromatic, can be charged

    # Negatively charged amino acids
    'D': np.array([0.05, -1.0, 1.0, 0.0, 0.4, 0.7, 0.0, 1.0]),  # Aspartate: acidic, short
    'E': np.array([0.05, -1.0, 1.0, 0.0, 0.6, 0.7, 0.0, 1.0]),  # Glutamate: acidic, longer
}

# Catalytic residue prior probabilities
# Based on statistical analysis of known catalytic sites
# Higher values indicate residues more likely to be catalytic
CATALYTIC_PRIOR: Dict[str, float] = {
    # Nucleophiles (high catalytic potential)
    'S': 0.85,  # Serine: common nucleophile in serine proteases, esterases
    'C': 0.90,  # Cysteine: strong nucleophile in cysteine proteases, thiol enzymes
    'D': 0.80,  # Aspartate: general acid/base, metal coordination
    'E': 0.75,  # Glutamate: general acid/base, metal coordination
    'H': 0.95,  # Histidine: most versatile, general acid/base, metal coordination
    'K': 0.70,  # Lysine: general base, Schiff base formation

    # Moderate catalytic potential
    'Y': 0.60,  # Tyrosine: can act as nucleophile or acid/base
    'T': 0.55,  # Threonine: similar to serine but less common
    'R': 0.50,  # Arginine: substrate binding, stabilization
    'N': 0.45,  # Asparagine: hydrogen bonding, oxyanion hole
    'Q': 0.40,  # Glutamine: hydrogen bonding, oxyanion hole
    'W': 0.35,  # Tryptophan: substrate binding, π-stacking

    # Low catalytic potential (mostly structural)
    'M': 0.25,  # Methionine: occasionally in active sites
    'F': 0.20,  # Phenylalanine: substrate binding, hydrophobic interactions
    'G': 0.15,  # Glycine: flexibility, tight turns
    'A': 0.10,  # Alanine: small, rarely catalytic
    'V': 0.08,  # Valine: hydrophobic core
    'L': 0.08,  # Leucine: hydrophobic core
    'I': 0.08,  # Isoleucine: hydrophobic core
    'P': 0.05,  # Proline: structural, rarely in active sites
}

# Common catalytic triads and dyads
KNOWN_CATALYTIC_MOTIFS: List[tuple] = [
    ('S', 'H', 'D'),  # Serine protease triad (e.g., chymotrypsin)
    ('S', 'H', 'E'),  # Alternative serine protease triad
    ('C', 'H', 'D'),  # Cysteine protease triad (e.g., papain)
    ('C', 'H', 'N'),  # Alternative cysteine protease triad
    ('D', 'H', 'S'),  # Aspartyl protease with histidine
    ('D', 'D'),       # Aspartyl protease dyad (e.g., pepsin)
    ('E', 'E'),       # Glutamate dyad (e.g., some metalloproteases)
    ('H', 'E'),       # Histidine-glutamate dyad
    ('K', 'E'),       # Lysine-glutamate dyad (e.g., some kinases)
]

# Metal coordination residues
METAL_COORDINATING_RESIDUES = ['H', 'D', 'E', 'C']

# Residues capable of forming covalent intermediates
COVALENT_CATALYSIS_RESIDUES = ['S', 'C', 'T', 'K', 'Y']

# Residues commonly in oxyanion holes
OXYANION_HOLE_RESIDUES = ['N', 'G', 'S', 'T']

# Standard amino acid masses (Da)
AA_MASSES: Dict[str, float] = {
    'A': 89.09, 'C': 121.15, 'D': 133.10, 'E': 147.13, 'F': 165.19,
    'G': 75.07, 'H': 155.15, 'I': 131.17, 'K': 146.19, 'L': 131.17,
    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19,
}

# Van der Waals radii (Angstroms)
VDW_RADII: Dict[str, float] = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
    'P': 1.80, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
}

# Atom types for structure parsing
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']
SIDECHAIN_ATOMS_BY_AA: Dict[str, List[str]] = {
    'A': ['CB'],
    'C': ['CB', 'SG'],
    'D': ['CB', 'CG', 'OD1', 'OD2'],
    'E': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
    'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'G': [],
    'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'I': ['CB', 'CG1', 'CG2', 'CD1'],
    'K': ['CB', 'CG', 'CD', 'CE', 'NZ'],
    'L': ['CB', 'CG', 'CD1', 'CD2'],
    'M': ['CB', 'CG', 'SD', 'CE'],
    'N': ['CB', 'CG', 'OD1', 'ND2'],
    'P': ['CB', 'CG', 'CD'],
    'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
    'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'S': ['CB', 'OG'],
    'T': ['CB', 'OG1', 'CG2'],
    'V': ['CB', 'CG1', 'CG2'],
    'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
}

# Non-standard residue mappings to standard amino acids
NON_STANDARD_AA_MAP: Dict[str, str] = {
    'MSE': 'M',  # Selenomethionine -> Methionine
    'SEP': 'S',  # Phosphoserine -> Serine
    'TPO': 'T',  # Phosphothreonine -> Threonine
    'PTR': 'Y',  # Phosphotyrosine -> Tyrosine
    'HYP': 'P',  # Hydroxyproline -> Proline
    'MLY': 'K',  # N-dimethyl-lysine -> Lysine
    'CSO': 'C',  # S-hydroxycysteine -> Cysteine
    'CSD': 'C',  # 3-sulfinoalanine -> Cysteine
    'CME': 'C',  # S,S-(2-hydroxyethyl)thiocysteine -> Cysteine
}

# Secondary structure codes
SECONDARY_STRUCTURE_CODES = {
    'H': 'helix',
    'E': 'sheet',
    'C': 'coil',
    'T': 'turn',
    'B': 'bridge',
    'G': '310_helix',
    'I': 'pi_helix',
}

# Distance thresholds for interactions (Angstroms)
HYDROGEN_BOND_DISTANCE = 3.5
SALT_BRIDGE_DISTANCE = 4.0
DISULFIDE_BOND_DISTANCE = 2.5
PI_STACKING_DISTANCE = 5.5
HYDROPHOBIC_CONTACT_DISTANCE = 5.0

# Default edge cutoff for graph construction (Angstroms)
DEFAULT_EDGE_CUTOFF = 8.0

# Maximum number of residues for efficient processing
MAX_RESIDUES_DEFAULT = 1000

# Atom type encoding for generation
ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'H', 'F', 'Cl', 'Br', 'I']
ATOM_TYPE_TO_INDEX = {atom: i for i, atom in enumerate(ATOM_TYPES)}
INDEX_TO_ATOM_TYPE = {i: atom for i, atom in enumerate(ATOM_TYPES)}

# Bond types
BOND_TYPES = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
BOND_TYPE_TO_INDEX = {bond: i for i, bond in enumerate(BOND_TYPES)}
INDEX_TO_BOND_TYPE = {i: bond for i, bond in enumerate(BOND_TYPES)}
