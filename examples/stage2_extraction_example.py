"""
Example: Stage 2 - Motif Extraction Layer
==========================================

Demonstrates how to extract catalytic motifs from enzyme structures.
"""

from nanozyme_mining.extraction import MotifExtractor, CatalyticMotif
from nanozyme_mining.database import NanozymeDatabase


def main():
    # Initialize extractor
    extractor = MotifExtractor(output_dir="./data/motifs")
    print("Extractor initialized")

    # Initialize database
    db = NanozymeDatabase(db_path="./data/nanozyme_db.sqlite")

    # Query POD enzymes
    entries = db.query_by_nanozyme_type(
        __import__('nanozyme_mining.utils', fromlist=['NanozymeType']).NanozymeType.POD
    )

    print(f"Found {len(entries)} POD enzymes")

    # Extract motifs
    for entry in entries[:3]:
        if not entry.pdb_path:
            continue

        print(f"\nExtracting motif from {entry.uniprot_id}...")

        motif = extractor.extract_motif(
            pdb_path=entry.pdb_path,
            uniprot_id=entry.uniprot_id,
            ec_number=entry.ec_number,
            nanozyme_type=entry.nanozyme_type
        )

        if motif:
            print(f"  Found {len(motif.anchor_atoms)} anchor atoms")
            print(f"  Found {len(motif.geometry_constraints)} constraints")

            # Export to JSON
            output_path = f"./data/motifs/{motif.motif_id}.json"
            motif.to_json(output_path)
            print(f"  Exported to {output_path}")


if __name__ == "__main__":
    main()
