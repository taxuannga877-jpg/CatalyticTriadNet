"""
Motif Extractor - Stage 2 Core Module
======================================

Extracts catalytic motifs from enzyme structures.
Based on ChemEnzyRetroPlanner's active site extraction patterns.
"""

import os
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .motif import CatalyticMotif, AnchorAtom, GeometryConstraint


# Catalytic residue definitions for common enzyme types
CATALYTIC_RESIDUES = {
    "POD": ["HIS", "ARG", "ASN"],      # Peroxidase
    "SOD": ["HIS", "ASP", "CYS"],      # Superoxide dismutase
    "CAT": ["HIS", "ASN", "TYR"],      # Catalase
    "GSH": ["SEC", "CYS", "GLN"],      # Glutathione peroxidase
    "OXD": ["HIS", "CYS", "TYR"],      # Oxidase
    "LAC": ["HIS", "CYS", "ASP"],      # Laccase
}

# Key donor atoms for each residue type
DONOR_ATOMS = {
    "HIS": ["NE2", "ND1"],
    "SER": ["OG"],
    "CYS": ["SG"],
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
    "TYR": ["OH"],
    "ARG": ["NH1", "NH2"],
    "LYS": ["NZ"],
    "ASN": ["OD1", "ND2"],
    "GLN": ["OE1", "NE2"],
    "SEC": ["SE"],
}


class MotifExtractor:
    """
    Extracts catalytic motifs from PDB structures.

    Based on ChemEnzyRetroPlanner's MyProtein and active site
    extraction patterns.
    """

    def __init__(self, output_dir: str = "./motifs"):
        """
        Initialize extractor.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_pdb(self, pdb_path: str) -> List[Dict]:
        """
        Parse PDB file and extract atom information.

        Args:
            pdb_path: Path to PDB file

        Returns:
            List of atom dictionaries
        """
        atoms = []

        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom = {
                        "atom_name": line[12:16].strip(),
                        "residue_name": line[17:20].strip(),
                        "chain_id": line[21].strip(),
                        "residue_number": int(line[22:26].strip()),
                        "x": float(line[30:38].strip()),
                        "y": float(line[38:46].strip()),
                        "z": float(line[46:54].strip()),
                        "element": line[76:78].strip() if len(line) > 76 else ""
                    }
                    atoms.append(atom)

        return atoms

    def calculate_distance(self, coord1: List[float], coord2: List[float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

    def calculate_angle(self, c1: List[float], c2: List[float], c3: List[float]) -> float:
        """Calculate angle (degrees) between three points."""
        v1 = np.array(c1) - np.array(c2)
        v2 = np.array(c3) - np.array(c2)
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return math.degrees(math.acos(np.clip(cos_angle, -1, 1)))

    def find_catalytic_residues(
        self,
        atoms: List[Dict],
        nanozyme_type: str,
        active_site_indices: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Find catalytic residues in the structure.

        Args:
            atoms: List of atom dictionaries
            nanozyme_type: Type of nanozyme
            active_site_indices: Known active site residue indices

        Returns:
            List of catalytic residue atoms
        """
        target_residues = CATALYTIC_RESIDUES.get(nanozyme_type, [])
        catalytic_atoms = []

        for atom in atoms:
            res_name = atom["residue_name"]
            res_num = atom["residue_number"]

            # Check if residue is catalytic type
            if res_name not in target_residues:
                continue

            # Check if in known active sites
            if active_site_indices and res_num not in active_site_indices:
                continue

            # Check if atom is a donor atom
            if res_name in DONOR_ATOMS:
                if atom["atom_name"] in DONOR_ATOMS[res_name]:
                    catalytic_atoms.append(atom)

        return catalytic_atoms

    def extract_motif(
        self,
        pdb_path: str,
        uniprot_id: str,
        ec_number: str,
        nanozyme_type: str,
        active_site_indices: Optional[List[int]] = None
    ) -> Optional[CatalyticMotif]:
        """
        Extract catalytic motif from PDB structure.

        Args:
            pdb_path: Path to PDB file
            uniprot_id: UniProt ID
            ec_number: EC number
            nanozyme_type: Nanozyme type
            active_site_indices: Known active site residue indices

        Returns:
            CatalyticMotif object or None
        """
        atoms = self.parse_pdb(pdb_path)
        if not atoms:
            return None

        catalytic_atoms = self.find_catalytic_residues(
            atoms, nanozyme_type, active_site_indices
        )

        if not catalytic_atoms:
            return None

        # Create anchor atoms
        anchor_atoms = []
        for atom in catalytic_atoms:
            anchor = AnchorAtom(
                atom_name=atom["atom_name"],
                residue_name=atom["residue_name"],
                residue_number=atom["residue_number"],
                chain_id=atom["chain_id"],
                element=atom["element"],
                coordinates=[atom["x"], atom["y"], atom["z"]],
                is_donor=True
            )
            anchor_atoms.append(anchor)

        # Calculate geometry constraints
        geometry = self._calculate_geometry(anchor_atoms)

        motif_id = f"{uniprot_id}_{ec_number}_{nanozyme_type}"

        return CatalyticMotif(
            motif_id=motif_id,
            source_uniprot_id=uniprot_id,
            source_ec_number=ec_number,
            nanozyme_type=nanozyme_type,
            anchor_atoms=anchor_atoms,
            geometry_constraints=geometry,
            extraction_method="rule_based"
        )

    def _calculate_geometry(self, anchor_atoms: List[AnchorAtom]) -> List[GeometryConstraint]:
        """Calculate geometry constraints between anchor atoms."""
        constraints = []

        # Calculate pairwise distances
        for i in range(len(anchor_atoms)):
            for j in range(i + 1, len(anchor_atoms)):
                dist = self.calculate_distance(
                    anchor_atoms[i].coordinates,
                    anchor_atoms[j].coordinates
                )
                constraints.append(GeometryConstraint(
                    constraint_type="distance",
                    atom_indices=[i, j],
                    value=dist,
                    unit="angstrom"
                ))

        # Calculate angles for triplets
        if len(anchor_atoms) >= 3:
            for i in range(len(anchor_atoms) - 2):
                angle = self.calculate_angle(
                    anchor_atoms[i].coordinates,
                    anchor_atoms[i + 1].coordinates,
                    anchor_atoms[i + 2].coordinates
                )
                constraints.append(GeometryConstraint(
                    constraint_type="angle",
                    atom_indices=[i, i + 1, i + 2],
                    value=angle,
                    unit="degree"
                ))

        return constraints
