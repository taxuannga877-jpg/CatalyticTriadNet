"""
Tests for Nanozyme Mining System
"""

import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanozyme_mining.database import NanozymeDatabase, EnzymeEntry
from nanozyme_mining.extraction import CatalyticMotif, AnchorAtom, GeometryConstraint
from nanozyme_mining.utils import NanozymeType


class TestNanozymeDatabase(unittest.TestCase):
    """Tests for NanozymeDatabase."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.sqlite")
        self.db = NanozymeDatabase(db_path=self.db_path)

    def test_add_enzyme(self):
        entry = EnzymeEntry(
            uniprot_id="P00000",
            ec_number="1.11.1.7",
            nanozyme_type="Peroxidase",
            sequence="MKTL",
            sequence_length=4
        )
        result = self.db.add_enzyme(entry)
        self.assertTrue(result)

    def test_query_by_ec(self):
        entry = EnzymeEntry(
            uniprot_id="P00001",
            ec_number="1.11.1.7",
            nanozyme_type="Peroxidase",
            sequence="MKTL",
            sequence_length=4
        )
        self.db.add_enzyme(entry)
        results = self.db.query_by_ec("1.11.1.7")
        self.assertEqual(len(results), 1)


class TestCatalyticMotif(unittest.TestCase):
    """Tests for CatalyticMotif."""

    def test_motif_creation(self):
        motif = CatalyticMotif(
            motif_id="test_motif",
            source_uniprot_id="P00000",
            source_ec_number="1.11.1.7",
            nanozyme_type="POD"
        )
        self.assertEqual(motif.motif_id, "test_motif")

    def test_to_dict(self):
        motif = CatalyticMotif(
            motif_id="test",
            source_uniprot_id="P00000",
            source_ec_number="1.11.1.7",
            nanozyme_type="POD"
        )
        d = motif.to_dict()
        self.assertIn("motif_id", d)


if __name__ == "__main__":
    unittest.main()
