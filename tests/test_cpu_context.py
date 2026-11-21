import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adams_rag.cpu_context import pack
from adams_rag.ollama_engine import OllamaEngine
from adams_rag.query_terms import subject_phrases
from adams_rag.research import hybrid_retrieve, fallback_queries, research
from adams_rag.store import Store


class CPUContextTests(unittest.TestCase):
    def test_large_prompts_are_bounded_exact_slices_with_document_diversity(self):
        evidence = [{'citation': i + 1, 'accession': str(i // 2), 'title': 'Inspection',
                     'text': ('Background. ' * 80 + 'The valve failed. Cause and effect must be evaluated. ' + 'More context. ' * 100)}
                    for i in range(10)]
        system, data, audit = pack('What happens when the valve fails?', evidence)
        self.assertLessEqual(audit['input_bytes'], 6000)
        self.assertEqual(len({s['accession'] for s in data['sources']}), len(data['sources']))
        for item, span in zip(data['sources'], audit['source_spans']):
            original = evidence[item['source_id'] - 1]['text']
            self.assertEqual(original[span['start_character']:span['end_character']], item['text'])
        self.assertIn('Historical excerpts do not establish current licensing status', system)

    def test_nonascii_and_large_question_obey_actual_byte_budget(self):
        evidence = [{'citation': 1, 'accession': 'one', 'title': 'Unicode', 'text': 'électricité 核能 ' * 2000}]
        system, data, audit = pack('Why ' * 400, evidence)
        self.assertLessEqual(len(system.encode()) + len(json.dumps(data, ensure_ascii=False).encode()), 6000)

    def test_cpu_rejects_citations_to_passages_it_did_not_read(self):
        evidence = [{'citation': i + 1, 'accession': str(i), 'title': 'Title', 'text': 'Valve failed. ' * 300} for i in range(10)]
        engine = OllamaEngine('test', cpu_only=True)
        with patch.object(engine, 'complete', return_value={'paragraphs': [{'text': 'Claim', 'source_ids': [10]}],
                                                          'insufficient_evidence': False, 'gaps': []}):
            with self.assertRaisesRegex(ValueError, 'outside its reading budget'):
                engine.generate(None, 'valve', evidence)

    def test_company_name_is_not_relaxed_to_ordinary_words_or_semantic_neighbors(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'research.sqlite')
            try:
                with store.db:
                    store.put({'AccessionNumber': 'OTHER', 'DocumentTitle': 'General matters of licensing',
                               'content': 'General matters are discussed in this reactor license.'})
                class WrongSemantic:
                    def search(self, *args):
                        return store.lexical('reactor license')
                self.assertEqual(hybrid_retrieve(store, 'How far along is General Matter on their license?', WrongSemantic()), [])
                with store.db:
                    store.put({'AccessionNumber': 'RIGHT', 'DocumentTitle': 'General Matter application',
                               'content': 'General Matter submitted an application. Submission is not approval.'})
                rows = hybrid_retrieve(store, 'How far along is General Matter on their license?', WrongSemantic())
                self.assertEqual([r['accession'] for r in rows], ['RIGHT'])
            finally:
                store.db.close()

    def test_named_phrase_query_and_nonentity_question(self):
        self.assertEqual(subject_phrases('How far along is General Matter on their license?'), ['General Matter'])
        self.assertEqual(fallback_queries('How far along is General Matter on their license?'), ['"General Matter"'])
        self.assertEqual(subject_phrases('What happens when a relief valve fails?'), [])

    def test_sources_are_published_before_a_model_timeout(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'research.sqlite')
            order = []
            class Slow:
                def generate(self, *args):
                    order.append('generate')
                    raise RuntimeError('Local Ollama exceeded its request timeout')
            try:
                with store.db:
                    store.put({'AccessionNumber': 'ONE', 'DocumentTitle': 'Valve testing', 'content': 'The valve failed its test.'})
                def preview(result):
                    order.append('preview')
                    self.assertTrue(result['sources'])
                    self.assertIsNone(result['answer'])
                result = research(store, 'valve testing', engine=Slow(), live=False, evidence_ready=preview)
                self.assertEqual(order, ['preview', 'generate'])
                self.assertEqual(result['mode'], 'evidence_only')
                self.assertTrue(result['sources'])
            finally:
                store.db.close()


if __name__ == '__main__':
    unittest.main()
