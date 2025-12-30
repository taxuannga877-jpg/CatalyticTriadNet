"""
UniProt Data Fetcher - Stage 1 Data Acquisition
================================================

Fetches enzyme data from UniProt and downloads AlphaFold structures.
Based on ChemEnzyRetroPlanner's UniProtParserEC architecture.
"""

import os
import json
import time
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .nanozyme_db import NanozymeDatabase, EnzymeEntry
from ..utils.constants import NanozymeType
from ..utils.ec_mappings import EC_PATTERNS


AFDB_VERSION = "4"  # AlphaFold DB version


class UniProtFetcher:
    """
    Fetches enzyme data from UniProt REST API and downloads structures.

    Based on ChemEnzyRetroPlanner's UniProtParserEC class.

    NOTE: This class does NOT use any ML models.
    It directly queries UniProt API to get:
    - Enzyme entries by EC number
    - Known active site annotations (ft_act_site, ft_binding)
    - AlphaFold structure downloads
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        download_timeout: int = 60,
        max_sequence_length: int = 600
    ):
        """
        Initialize UniProt fetcher.

        Args:
            cache_dir: Directory for caching downloaded data
            download_timeout: Timeout for downloads in seconds
            max_sequence_length: Maximum protein sequence length
        """
        self.cache_dir = Path(cache_dir)
        self.download_timeout = download_timeout
        self.max_sequence_length = max_sequence_length

        # Create cache directories
        self.csv_cache = self.cache_dir / "csv"
        self.json_cache = self.cache_dir / "json"
        self.pdb_cache = self.cache_dir / "pdb"

        # Separate folders for annotated vs unannotated
        self.annotated_dir = self.cache_dir / "annotated"
        self.unannotated_dir = self.cache_dir / "unannotated"

        for d in [self.csv_cache, self.json_cache, self.pdb_cache,
                  self.annotated_dir, self.unannotated_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # API templates - TSV format for basic info
        self.uniprot_query_template = (
            "https://rest.uniprot.org/uniprotkb/stream?"
            "query=ec:{ec}+AND+reviewed:true+AND+database:(alphafolddb)"
            "&fields=accession,ec,sequence,xref_alphafolddb"
            "&format=tsv&size={size}"
        )

        # JSON format for detailed info including active sites
        self.uniprot_json_template = (
            "https://rest.uniprot.org/uniprotkb/search?"
            "query=ec:{ec}+AND+reviewed:true+AND+database:(alphafolddb)"
            "&fields=accession,ec,sequence,ft_act_site,ft_binding,ft_site,xref_alphafolddb"
            "&format=json"
        )

        self.alphafold_url_template = (
            "https://alphafold.ebi.ac.uk/files/"
            "AF-{alphafold_id}-F1-model_v{version}.pdb"
        )

    def query_by_ec(
        self,
        ec_number: str,
        size: int = 10
    ) -> List[Dict]:
        """
        Query UniProt for enzymes by EC number.

        Args:
            ec_number: EC number (e.g., "1.11.1.7")
            size: Maximum number of results

        Returns:
            List of enzyme data dictionaries
        """
        cache_file = self.csv_cache / f"{ec_number}.tsv"

        # Check cache first
        if cache_file.exists():
            return self._parse_tsv(cache_file)

        # Query UniProt API
        url = self.uniprot_query_template.format(
            ec=ec_number, size=size
        )

        try:
            response = requests.get(url, timeout=self.download_timeout)
            response.raise_for_status()

            # Save to cache
            with open(cache_file, 'w') as f:
                f.write(response.text)

            return self._parse_tsv(cache_file)

        except Exception as e:
            print(f"Error querying UniProt for EC {ec_number}: {e}")
            return []

    def _parse_tsv(self, tsv_file: Path) -> List[Dict]:
        """Parse TSV file from UniProt."""
        results = []

        with open(tsv_file, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            return results

        headers = lines[0].strip().split('\t')

        for line in lines[1:]:
            values = line.strip().split('\t')
            if len(values) >= len(headers):
                entry = dict(zip(headers, values))
                results.append(entry)

        return results

    def query_with_active_sites(self, ec_number: str) -> List[Dict]:
        """
        Query UniProt with active site annotations (JSON format).

        This method gets KNOWN active sites from UniProt annotations,
        NOT predicted by any ML model.

        Args:
            ec_number: EC number

        Returns:
            List of enzyme data with active site info
        """
        cache_file = self.json_cache / f"{ec_number}_sites.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        url = self.uniprot_json_template.format(ec=ec_number)

        try:
            response = requests.get(url, timeout=self.download_timeout)
            response.raise_for_status()
            data = response.json()

            results = []
            for entry in data.get("results", []):
                active_sites = self._extract_active_sites(entry)
                results.append({
                    "uniprot_id": entry.get("primaryAccession", ""),
                    "sequence": entry.get("sequence", {}).get("value", ""),
                    "active_sites": active_sites
                })

            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)

            return results

        except Exception as e:
            print(f"Error querying active sites for EC {ec_number}: {e}")
            return []

    def _extract_active_sites(self, entry: Dict) -> List[Dict]:
        """Extract active site annotations from UniProt entry."""
        sites = []
        features = entry.get("features", [])

        for feat in features:
            feat_type = feat.get("type", "")
            if feat_type in ["Active site", "Binding site", "Site"]:
                location = feat.get("location", {})
                sites.append({
                    "type": feat_type,
                    "start": location.get("start", {}).get("value"),
                    "end": location.get("end", {}).get("value"),
                    "description": feat.get("description", "")
                })

        return sites

    def download_pdb(self, alphafold_id: str) -> Optional[Path]:
        """
        Download AlphaFold PDB structure.

        Args:
            alphafold_id: AlphaFold database ID

        Returns:
            Path to downloaded PDB file or None
        """
        pdb_file = self.pdb_cache / f"AF-{alphafold_id}-F1-model_v{AFDB_VERSION}.pdb"

        if pdb_file.exists():
            return pdb_file

        url = self.alphafold_url_template.format(
            alphafold_id=alphafold_id,
            version=AFDB_VERSION
        )

        try:
            response = requests.get(url, timeout=self.download_timeout)
            response.raise_for_status()

            with open(pdb_file, 'w') as f:
                f.write(response.text)

            return pdb_file

        except Exception as e:
            print(f"Error downloading PDB for {alphafold_id}: {e}")
            return None

    def fetch_and_populate(
        self,
        db: NanozymeDatabase,
        ec_number: str,
        nanozyme_type: NanozymeType,
        max_entries: int = 10
    ) -> int:
        """
        Fetch enzymes by EC and populate database.

        Args:
            db: NanozymeDatabase instance
            ec_number: EC number to query
            nanozyme_type: Nanozyme type for this EC
            max_entries: Maximum entries to fetch

        Returns:
            Number of entries added
        """
        entries_data = self.query_by_ec(ec_number, max_entries)
        count = 0

        for data in entries_data:
            try:
                # Extract AlphaFold ID
                alphafold_id = data.get('AlphaFoldDB', '').split(';')[0]
                if not alphafold_id:
                    continue

                sequence = data.get('Sequence', '')
                if len(sequence) > self.max_sequence_length:
                    continue

                # Download PDB
                pdb_path = self.download_pdb(alphafold_id)

                entry = EnzymeEntry(
                    uniprot_id=data.get('Entry', ''),
                    ec_number=ec_number,
                    nanozyme_type=nanozyme_type.value,
                    sequence=sequence,
                    sequence_length=len(sequence),
                    alphafold_id=alphafold_id,
                    pdb_path=str(pdb_path) if pdb_path else None
                )

                if db.add_enzyme(entry):
                    count += 1

            except Exception as e:
                print(f"Error processing entry: {e}")

        return count

    def fetch_and_classify(
        self,
        ec_number: str,
        nanozyme_type: NanozymeType
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Fetch enzymes and classify into annotated vs unannotated.

        Two-pronged strategy:
        - Annotated: Has active site info from UniProt/M-CSA
        - Unannotated: No active site info, needs model prediction

        Args:
            ec_number: EC number to query
            nanozyme_type: Nanozyme type

        Returns:
            Tuple of (annotated_list, unannotated_list)
        """
        # Get detailed info with active sites
        entries = self.query_with_active_sites(ec_number)

        annotated = []
        unannotated = []

        for entry in entries:
            uniprot_id = entry.get("uniprot_id", "")
            active_sites = entry.get("active_sites", [])
            sequence = entry.get("sequence", "")

            if len(sequence) > self.max_sequence_length:
                continue

            # Download PDB
            pdb_path = self.download_pdb(uniprot_id)

            entry_data = {
                "uniprot_id": uniprot_id,
                "ec_number": ec_number,
                "nanozyme_type": nanozyme_type.value,
                "sequence": sequence,
                "pdb_path": str(pdb_path) if pdb_path else None,
                "active_sites": active_sites
            }

            # Classify based on annotation
            if active_sites and len(active_sites) > 0:
                annotated.append(entry_data)
                self._save_to_folder(entry_data, self.annotated_dir)
            else:
                unannotated.append(entry_data)
                self._save_to_folder(entry_data, self.unannotated_dir)

        print(f"EC {ec_number}: {len(annotated)} annotated, {len(unannotated)} unannotated")
        return annotated, unannotated

    def _save_to_folder(self, entry_data: Dict, folder: Path):
        """Save entry data to JSON file in specified folder."""
        uniprot_id = entry_data["uniprot_id"]
        output_file = folder / f"{uniprot_id}.json"
        with open(output_file, 'w') as f:
            json.dump(entry_data, f, indent=2)
