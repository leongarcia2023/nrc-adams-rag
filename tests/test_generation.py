import unittest

from adams_rag.models import assemble_answer, MLXEngine


class GenerationTests(unittest.TestCase):
    def test_storage_stop_in_comparison_does_not_continue_or_claim_generated_output(self):
        from adams_rag.storage import StoragePause
        calls = []
        engine = MLXEngine("unused")
        def stop(question, evidence):
            calls.append(question)
            raise StoragePause("reserve reached")
        engine._generate_once = stop
        evidence = [{"citation": 1, "accession": "ML24017A120"}, {"citation": 2, "accession": "ML24131A151"}]
        with self.assertRaises(StoragePause):
            engine.generate(None, "Compare ML24017A120 and ML24131A151", evidence)
        self.assertEqual(len(calls), 1)

    def test_model_context_preserves_record_identity_and_legacy_reference_without_relabeling_source(self):
        engine = MLXEngine("unused")
        captured = []
        def complete(system, data):
            captured.append(data)
            return {"paragraphs": [], "insufficient_evidence": True, "gaps": ["No verified identity equivalence."]}
        engine.complete = complete
        evidence = [{"citation": 1, "accession": "ML20008D654", "title": "Digitized report", "text": "Source text",
                     "source_key": "nrc:main:id:one", "source_record_id": "id:one",
                     "legacy_report_references": ["7809190190"]}]
        result = engine.generate(None, "Summarize legacy 7809190190", evidence)
        source = captured[0]["sources"][0]
        self.assertEqual(source["accession"], "ML20008D654")
        self.assertEqual(source["legacy_report_references"], ["7809190190"])
        self.assertEqual(source["source_record_id"], "id:one")
        self.assertTrue(result["insufficient_evidence"])

    def test_model_selected_citations_are_rendered_and_retained(self):
        result = assemble_answer({"paragraphs": [{"text": "The valve test failed.", "source_ids": [2, 1, 2]}],
                                  "insufficient_evidence": False, "gaps": []}, 2)
        self.assertEqual(result["answer"], "The valve test failed. [2][1]")
        self.assertEqual(result["claims"][0]["source_ids"], [2, 1, 2])

    def test_missing_or_unknown_citations_withhold_even_partial_claims(self):
        for ids in ([], [0], [3], [True], ["1"], None):
            with self.assertRaises(ValueError):
                assemble_answer({"paragraphs": [{"text": "The valve failed.", "source_ids": ids}],
                                 "insufficient_evidence": True, "gaps": []}, 2)

    def test_no_evidence_response_contains_no_invented_citations(self):
        result = assemble_answer({"paragraphs": [], "insufficient_evidence": True,
                                  "gaps": ["No shutdown date is given in these passages."]}, 3)
        self.assertNotIn("[", result["answer"])
        self.assertTrue(result["insufficient_evidence"])

    def test_reported_gap_cannot_be_marked_as_a_complete_answer(self):
        result = assemble_answer({"paragraphs": [{"text": "The narrative states an assumption.", "source_ids": [1]}],
                                  "insufficient_evidence": False,
                                  "gaps": [" Table rows are missing. ", "Table rows are missing.", " "]}, 1)
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["evidence_status_adjusted"], "missing_evidence_reported")
        self.assertEqual(result["gaps"], ["Table rows are missing."])
        self.assertEqual(result["claims"][0]["source_ids"], [1])

    def test_empty_gap_strings_do_not_override_a_supported_answer(self):
        result = assemble_answer({"paragraphs": [{"text": "Supported finding.", "source_ids": [1]}],
                                  "insufficient_evidence": False, "gaps": [" ", ""]}, 1)
        self.assertFalse(result["insufficient_evidence"])
        self.assertNotIn("evidence_status_adjusted", result)

    def test_empty_answer_with_an_explicit_gap_becomes_an_abstention(self):
        result = assemble_answer({"paragraphs": [], "insufficient_evidence": False,
                                  "gaps": ["No readable table rows."]}, 1)
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["evidence_status_adjusted"], "missing_evidence_reported")
        self.assertNotIn("[", result["answer"])

    def test_unstructured_answer_and_contradictory_empty_answer_are_rejected(self):
        for result in ({"answer": "An uncited answer", "insufficient_evidence": False},
                       {"paragraphs": [], "insufficient_evidence": False}):
            with self.assertRaises(ValueError):
                assemble_answer(result, 3)

    def test_named_comparison_keeps_each_report_and_original_citation_ids(self):
        engine = MLXEngine("unused")
        engine._generate_once = lambda question, evidence: {
            "claims": [{"text": "A supported finding.", "source_ids": [1]}],
            "insufficient_evidence": False, "gaps": []}
        evidence = [{"citation": 1, "accession": "ML24017A120"},
                    {"citation": 2, "accession": "ML24017A120"},
                    {"citation": 3, "accession": "ML24131A151"}]
        result = engine.generate(None, "Compare ML24017A120 and ML24131A151", evidence)
        self.assertIn("ML24017A120:", result["answer"])
        self.assertIn("ML24131A151:", result["answer"])
        self.assertEqual([c["source_ids"] for c in result["claims"]], [[1], [3]])
        self.assertFalse(result["insufficient_evidence"])
        partial = engine.generate(None, "Compare ML24017A120 and ML24131A151", evidence[:2])
        self.assertTrue(partial["insufficient_evidence"])
        self.assertIn("ML24131A151", partial["gaps"][0])
