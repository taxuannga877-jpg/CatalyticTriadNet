"""
M-CSA (Mechanism and Catalytic Site Atlas) Query Module
========================================================

M-CSA provides detailed catalytic site annotations with:
- Catalytic residues and their roles
- Reaction mechanisms
- Literature evidence

Website: https://www.ebi.ac.uk/thornton-srv/m-csa/
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CatalyticResidue:
    """A catalytic residue from M-CSA."""
    residue_name: str
    residue_number: int
    chain_id: str
    role: str  # e.g., "proton donor", "nucleophile"
    evidence: str


@dataclass
class MCSAEntry:
    """An M-CSA entry for an enzyme."""
    mcsa_id: str
    ec_number: str
    pdb_id: str
    uniprot_id: str
    enzyme_name: str
    catalytic_residues: List[CatalyticResidue]
    mechanism_description: str


class MCSAFetcher:
    """
    Fetches catalytic site data from M-CSA database.

    M-CSA has ~1000 entries with detailed catalytic mechanisms.
    """

    BASE_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api"

    def __init__(self, cache_dir: str = "./cache/mcsa"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def query_by_ec(self, ec_number: str) -> List[Dict]:
        """
        Query M-CSA entries by EC number.

        Args:
            ec_number: EC number (e.g., "1.11.1.7")

        Returns:
            List of M-CSA entries
        """
        cache_file = self.cache_dir / f"ec_{ec_number.replace('.', '_')}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        url = f"{self.BASE_URL}/entries/?ec={ec_number}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            return data

        except Exception as e:
            print(f"Error querying M-CSA for EC {ec_number}: {e}")
            return []

    def get_entry_details(self, mcsa_id: str) -> Optional[Dict]:
        """
        Get detailed info for a specific M-CSA entry.

        Args:
            mcsa_id: M-CSA entry ID (e.g., "M0001")

        Returns:
            Detailed entry data including catalytic residues
        """
        cache_file = self.cache_dir / f"entry_{mcsa_id}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        url = f"{self.BASE_URL}/entries/{mcsa_id}/"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            return data

        except Exception as e:
            print(f"Error getting M-CSA entry {mcsa_id}: {e}")
            return None

    def get_catalytic_residues(self, mcsa_id: str) -> List[Dict]:
        """
        Get catalytic residues for an M-CSA entry.

        Returns residues with their functional roles.
        """
        entry = self.get_entry_details(mcsa_id)
        if not entry:
            return []

        residues = []
        for res in entry.get("residues", []):
            residues.append({
                "residue_name": res.get("residue_name", ""),
                "residue_number": res.get("residue_number", 0),
                "chain_id": res.get("chain_id", "A"),
                "role": res.get("role", ""),
            })

        return residues

    def check_ec_coverage(self, ec_number: str) -> Dict:
        """
        Check M-CSA coverage for an EC number.

        Returns stats on how many PDBs have catalytic site annotations.
        """
        entries = self.query_by_ec(ec_number)

        return {
            "ec_number": ec_number,
            "mcsa_entries": len(entries),
            "has_annotations": len(entries) > 0,
            "pdb_ids": [e.get("pdb_id", "") for e in entries]
        }
