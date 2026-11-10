from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adams_rag.identifiers import extract_identifiers, sync_identifiers
from adams_rag.research import rank_candidates
from adams_rag.store import Store
from adams_rag.storage import StoragePause


class IdentifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "research.sqlite")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def put(self, accession, text, **metadata):
        with self.store.db:
            self.store.put({"AccessionNumber": accession, "content": text, "DocumentTitle": "Fixture", **metadata})

    def test_case_dash_and_line_breaks_preserve_equipment_identifiers(self):
        ids = extract_identifiers("Inspect valve 2a31–f017 and procedure ABC-\n200-12. Phase2 occurred in 2024 at 3.5 percent.")
        self.assertIn("2A31-F017", ids)
        self.assertIn("ABC-200-12", ids)
        self.assertFalse({"PHASE2", "2024", "3.5"} & ids)

    def test_shared_docket_notation_is_expanded_only_when_explicit(self):
        ids = extract_identifiers("NCV 05000111, 05000112/2024001-02")
        self.assertIn("05000111/2024001-02", ids)
        self.assertIn("05000112/2024001-02", ids)
        other = extract_identifiers("Docket 05000111. Finding 05000112/2024001-02.")
        self.assertNotIn("05000111/2024001-02", other)

    def test_exact_lookup_rejects_different_suffix_and_applies_filters(self):
        self.put("A", "Failure of valve 2A31-F017", DocumentDate="2024-04-01", DocketNumber=["05000111"])
        self.put("B", "Failure of valve 2A31-F018", DocumentDate="2024-04-01", DocketNumber=["05000111"])
        self.put("C", "Failure of valve 2A31-F017", DocumentDate="2019-04-01", DocketNumber=["05000112"])
        self.assertEqual([r["accession"] for r in self.store.identifier_matches(
            "What happened to 2A31-F017?", docket="05000111", date_from="2024-01-01")], ["A"])
        self.assertFalse(self.store.identifier_matches("2A31-F019"))

    def test_revision_removes_old_identifier_entries(self):
        self.put("A", "Valve 2A31-F017 failed")
        self.put("A", "Corrected valve identifier 2A31-F018")
        self.assertFalse(self.store.identifier_matches("2A31-F017"))
        self.assertEqual(len(self.store.identifier_matches("2A31-F018")), 1)

    def test_repeated_identifiers_use_question_context_before_candidate_limit(self):
        for number in range(105):
            self.put(f"OLD-{number}", "Equipment QX-200 passed its inspection.")
        self.put("TARGET", "QX-200 failed the flood-protection assessment after inundation.")
        rows = self.store.identifier_matches("QX-200 flood-protection inundation assessment", limit=1)
        self.assertEqual([r["accession"] for r in rows], ["TARGET"])

    def test_existing_collection_migration_resumes_and_is_idempotent(self):
        self.put("A", "Valve 2A31-F017 failed")
        self.put("B", "Valve 2A31-F018 failed")
        with self.store.db:
            self.store.db.execute("DELETE FROM chunk_identifiers")
            self.store.db.execute("DELETE FROM identifier_versions")
        self.assertEqual(sync_identifiers(self.store, limit=1), 1)
        self.assertEqual(sync_identifiers(self.store, limit=1), 1)
        self.assertEqual(sync_identifiers(self.store, limit=1), 0)
        self.assertEqual(len(self.store.identifier_matches("2A31-F017")), 1)

    def test_migration_respects_storage_stop(self):
        with patch("shutil.disk_usage") as disk:
            disk.return_value.free = 400_000_000
            with self.assertRaises(StoragePause):
                sync_identifiers(self.store)

    def test_identifier_after_long_stopword_prefix_is_still_retrievable(self):
        self.put("A", "Valve 2A31-F017 failed.")
        self.assertEqual(len(self.store.identifier_matches("the " * 45 + "2A31-F017")), 1)

    def test_exact_match_survives_bounded_candidates_and_semantic_distractors(self):
        self.put("A", "NCV 05000111,05000112/2024001-02 involved a failed valve test.")
        self.put("B", "A broader discussion of test failures and findings.")
        distractor = self.store.lexical("broader")[0]
        class Semantic:
            def search(self, *args):
                return [distractor]
        with patch.object(self.store, "lexical", return_value=[distractor]):
            rows = rank_candidates(self.store, "What is finding 05000111/2024001-02 about?", Semantic())
        self.assertEqual(rows[0]["accession"], "A")

    def test_lexical_date_filter_is_applied_before_candidate_limit(self):
        self.put("A", "valve", DocumentDate="2019-01-01")
        self.put("B", "This newer document also discusses the valve.", DocumentDate="2024-01-01")
        self.assertEqual([r["accession"] for r in self.store.lexical("valve", limit=1, date_from="2024-01-01")], ["B"])


if __name__ == "__main__":
    unittest.main()
