import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error
import threading
from http.server import ThreadingHTTPServer

from adams_rag.store import Store
from adams_rag.research import research, search_live, aps_filters, hybrid_retrieve, related_accessions
from adams_rag.client import BudgetStop
from adams_rag.models import decode_object, MLXEngine
from adams_rag.server import Application, handler_for
from adams_rag.evaluate import evaluate


def document(accession="ML24017A120", text="The inspection identified a finding involving the residual heat removal relief valve.", **extra):
    return {"AccessionNumber": accession, "DocumentTitle": "LaSalle inspection report",
            "content": text, "Url": f"https://www.nrc.gov/docs/{accession[:6]}/{accession}.pdf", **extra}


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "research.sqlite")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def put(self, doc, provenance="api_document_unverified"):
        with self.store.db:
            self.store.put(doc, provenance)

    def test_version_history_survives_replacement_and_invalidates_vectors(self):
        self.put(document())
        row = self.store.db.execute("SELECT id FROM chunks").fetchone()
        with self.store.db:
            self.store.db.execute("INSERT INTO dense_vectors VALUES(?,?,?,?,?)", (row[0], "fake", 1, b"data", "hash"))
        self.put(document(text="The revised inspection text supersedes the earlier valve description."))
        self.assertEqual(self.store.status()["document_versions"], 2)
        self.assertEqual(self.store.status()["dense_vectors_by_model"], {})

    def test_metadata_refresh_does_not_create_false_text_version(self):
        self.put(document())
        self.put(document(text="Search excerpt only"), "api_search_excerpt")
        self.assertEqual(self.store.status()["document_versions"], 1)
        self.assertIn("relief valve", self.store.lexical("valve")[0]["text"])

    def test_saved_retrieval_records_requested_profile_and_fallback(self):
        self.put(document())
        class UnavailableSemantic:
            model='candidate-profile'
            def search(self,*args,**kwargs):
                raise RuntimeError('GPU busy')
        result=research(self.store,'relief valve',semantic=UnavailableSemantic(),live=False,generate=False)
        self.assertEqual(result['semantic_retrieval'],{'requested_model':'candidate-profile','keyword_fallback':True})
        self.assertTrue(result['sources'])

    def test_partial_metadata_refresh_preserves_docket_date_and_source(self):
        self.put(document(DocumentDate="2024-01-18", DocketNumber=["05000373"]))
        self.put({"AccessionNumber": "ML24017A120", "content": "short snippet"}, "api_search_excerpt")
        rows = hybrid_retrieve(self.store, "valve", docket="05000373", date_from="2024-01-01")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["document_date"], "2024-01-18")
        self.assertEqual(rows[0]["title"], "LaSalle inspection report")
        self.assertEqual(rows[0]["url"], document()["Url"])

    def test_live_discovery_does_not_use_snippet_as_evidence(self):
        class Client:
            def request(self, payload=None, accession=None):
                if accession:
                    return {"document": document(text="")}
                return {"count": 1, "results": [{"document": document(text="Pretend this snippet answers everything")}]}
        result = research(self.store, "relief valve", client=Client())
        self.assertFalse(result["sources"])
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["discovery"]["missing_text"], ["ML24017A120"])

    def test_direct_accession_skips_search_and_ingests_full_document(self):
        class Client:
            def request(self, payload=None, accession=None):
                assert payload is None
                assert accession == "ML24017A120"
                return {"document": document()}
        result = research(self.store, "What happened in ML24017A120?", client=Client())
        self.assertTrue(result["sources"])
        self.assertEqual(result["discovery"]["queries"], [])
        self.assertIsNone(result["sources"][0]["page"])

    def test_package_members_follow_same_document_budget(self):
        class Client:
            def request(self, payload=None, accession=None):
                if accession == "ML24017A120":
                    return {"document": document(IsPackage="Yes", DocumentsFiledInPackage=["ML24017A121", "ML24017A122"])}
                return {"document": document(accession)}
        result = search_live(self.store, Client(), "ML24017A120", [], max_documents=2)
        self.assertEqual(result["attempted"], 2)
        self.assertEqual(result["fetched"], ["ML24017A121"])
        self.assertEqual(len(result["relationships"]), 2)

    def test_budget_pause_returns_cached_evidence_and_visible_gap(self):
        self.put(document())
        class Client:
            def request(self, **kwargs):
                raise BudgetStop("Local budget reached")
        result = research(self.store, "relief valve", client=Client())
        self.assertTrue(result["sources"])
        self.assertIn("Local budget reached", result["warnings"])
        self.assertFalse(result["exhaustive"])

    def test_filters_exclude_unknown_and_wrong_dates_and_dockets(self):
        self.put(document(DocumentDate="2024-01-20", DocketNumber=["05000373"]))
        self.put(document("ML25017A120", DocumentDate="2025-01-20", DocketNumber=["05000373"]))
        self.put(document("ML23017A120", DocumentDate=None, DocketNumber=["05000373"]))
        result = hybrid_retrieve(self.store, "valve", docket="05000373", date_from="2024-01-01", date_to="2024-12-31")
        self.assertEqual([r["accession"] for r in result], ["ML24017A120"])
        self.assertFalse(hybrid_retrieve(self.store, "valve", docket="05000001"))
        with self.assertRaises(ValueError):
            aps_filters(date_from="2025-01-01", date_to="2024-01-01")

    def test_named_summary_retains_opening_purpose_and_respects_filters(self):
        opening = "We request a written release; this is not an approval. " + "Cover letter background. " * 220
        self.put(document(text=opening + ("\n\nShort unrelated appendix fragment. " * 300),
                          DocumentDate="1978-09-06", DocketNumber=["05000267"]))
        first = self.store.db.execute("SELECT id FROM chunks WHERE accession='ML24017A120' ORDER BY ordinal LIMIT 1").fetchone()[0]
        # Simulate bounded retrievers returning only later fragments.
        rest = [r for r in self.store.lexical("ML24017A120", 200) if r["id"] != first]
        with patch.object(self.store, "lexical", return_value=rest):
            result = hybrid_retrieve(self.store, "Summarize the purpose of ML24017A120", top_k=1)
        self.assertIn("We request a written release", result[0]["text"])
        self.assertFalse(hybrid_retrieve(self.store, "Summarize ML24017A120", date_from="2020-01-01"))
        self.assertFalse(hybrid_retrieve(self.store, "Summarize ML24017A120", docket="05000373"))

    def test_bad_model_citations_are_withheld(self):
        self.put(document())
        engine = MLXEngine("unused")
        engine.complete = lambda *a, **k: {"answer": "Unsupported answer [99].", "insufficient_evidence": False}
        result = research(self.store, "relief valve", engine=engine, live=False)
        self.assertEqual(result["mode"], "evidence_only")
        self.assertIsNone(result["answer"])
        self.assertTrue(result["warnings"])

    def test_model_storage_pause_keeps_retrieved_evidence_and_saved_run(self):
        from adams_rag.storage import StoragePause
        self.put(document())
        class Engine:
            def generate(self, *args):
                raise StoragePause("Local generation paused for disk headroom")
        result = research(self.store, "relief valve", engine=Engine(), live=False)
        self.assertEqual(result["mode"], "evidence_only")
        self.assertIsNone(result["answer"])
        self.assertTrue(result["sources"])
        self.assertEqual(result["citation_check"], "not_generated")
        self.assertTrue(any("disk headroom" in w for w in result["warnings"]))
        saved = json.loads(self.store.db.execute("SELECT result FROM research_runs WHERE id=?", (result["id"],)).fetchone()[0])
        self.assertEqual(saved["sources"], result["sources"])

    def test_generation_and_evidence_package_persist(self):
        self.put(document())
        class Engine:
            def generate(self, *args):
                return {"answer": "The inspection identified a relief valve finding [1].", "insufficient_evidence": False}
        result = research(self.store, "relief valve", engine=Engine(), live=False)
        saved = json.loads(self.store.db.execute("SELECT result FROM research_runs WHERE id=?", (result["id"],)).fetchone()[0])
        self.assertEqual(saved["mode"], "generated")
        self.assertEqual(saved["sources"][0]["text"], document()["content"])

    def test_context_budget_and_duplicate_passages(self):
        self.put(document())
        self.put(document("ML24017A121"))
        self.assertEqual(len(hybrid_retrieve(self.store, "relief valve")), 1)

    def test_named_report_does_not_receive_facts_from_other_reports(self):
        self.put(document())
        self.put(document("ML24017A121", text="A different valve failure at another facility."))
        result = hybrid_retrieve(self.store, "What valve failed in ML24017A120?")
        self.assertEqual({r["accession"] for r in result}, {"ML24017A120"})
        self.assertFalse(hybrid_retrieve(self.store, "What valve failed in ML24017A999?"))

    def test_structured_output_and_no_evidence_cases(self):
        self.assertEqual(decode_object('```json\n{"answer":"test"}\n```')["answer"], "test")
        with self.assertRaises(ValueError):
            decode_object("nothing structured")
        result = research(self.store, "relief valve", live=False)
        self.assertTrue(result["insufficient_evidence"])
        self.assertFalse(result["sources"])

    def test_evaluation_uses_document_relevance_not_chunk_counts(self):
        self.put(document())
        q, labels = self.root / "queries.jsonl", self.root / "labels.tsv"
        q.write_text(json.dumps({"_id": "q1", "text": "relief valve"}) + "\n")
        labels.write_text("query-id\tcorpus-id\tscore\nq1\tML24017A120\t1\n")
        result = evaluate(self.store, q, labels)
        self.assertEqual(result["recall_at_k"], 1)
        self.assertEqual(result["ndcg_at_k"], 1)
        self.assertFalse(result["generation_evaluated"])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.app = Application(root / "db.sqlite", root / "rate.sqlite", root / "absent", root / "model", root / "index")

    def tearDown(self):
        self.app.pool.shutdown(wait=True)
        self.tmp.cleanup()

    def test_untrusted_inputs_cannot_change_model_or_paths(self):
        with self.assertRaises(ValueError):
            self.app.submit({"question": "valve", "model_path": "/etc/passwd"})
        with self.assertRaises(ValueError):
            self.app.submit({"question": "valve", "live": "false"})
        with self.assertRaises(ValueError):
            self.app.submit({"question": "valve", "max_documents": 100000})

    def test_one_active_job_limit(self):
        self.app.jobs["existing"] = {"state": "running"}
        with self.assertRaises(RuntimeError):
            self.app.submit({"question": "valve"})


if __name__ == "__main__":
    unittest.main()
