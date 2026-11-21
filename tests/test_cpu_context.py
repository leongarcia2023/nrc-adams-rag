import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adams_rag.cpu_context import pack, excerpt
from adams_rag.ollama_engine import OllamaEngine
from adams_rag.query_terms import subject_phrases, history_order
from adams_rag.research import hybrid_retrieve, fallback_queries, research
from adams_rag.store import Store


class CPUContextTests(unittest.TestCase):
    def test_history_keeps_recent_milestone_and_denial_qualification(self):
        rows = [{'title': 'Oklo license application', 'document_date': f'{y}-01-01', 'retrieval_score': s}
                for y,s in [(2020,.25),(2022,.24),(2025,.21),(2019,.2),(2026,.19)]]
        self.assertIn('2026-01-01', [r['document_date'] for r in history_order(rows)[:4]])
        body = 'Distribution and administrative information. ' * 40 + '\n\nThe NRC is denying, without prejudice, the Oklo license application for lack of information. The company may reapply.\n\n' + 'Background license application requirements. ' * 40
        text, start, end = excerpt(body, 'What is the history of Oklo permits?', maximum=550)
        self.assertIn('denying, without prejudice', text)
        self.assertEqual(text, body[start:end])

    def test_large_prompts_are_bounded_exact_slices_with_document_diversity(self):
        evidence = [{'citation': i + 1, 'accession': str(i // 2), 'title': 'Inspection',
                     'text': ('Background. ' * 80 + 'The valve failed. Cause and effect must be evaluated. ' + 'More context. ' * 100)}
                    for i in range(10)]
        system, data, audit = pack('What happens when the valve fails?', evidence)
        self.assertLessEqual(audit['input_bytes'], 4500)
        self.assertEqual(len({s['accession'] for s in data['sources']}), len(data['sources']))
        for item, span in zip(data['sources'], audit['source_spans']):
            original = evidence[item['source_id'] - 1]['text']
            self.assertEqual(original[span['start_character']:span['end_character']], item['text'])
        self.assertIn('Historical excerpts do not establish current licensing status', system)

    def test_nonascii_and_large_question_obey_actual_byte_budget(self):
        evidence = [{'citation': 1, 'accession': 'one', 'title': 'Unicode', 'text': 'électricité 核能 ' * 2000}]
        system, data, audit = pack('Why ' * 400, evidence)
        self.assertLessEqual(len(system.encode()) + len(json.dumps(data, ensure_ascii=False).encode()), 4500)

    def test_cpu_rejects_citations_to_passages_it_did_not_read(self):
        evidence = [{'citation': i + 1, 'accession': str(i), 'title': 'Title', 'text': 'Valve failed. ' * 300} for i in range(10)]
        engine = OllamaEngine('test', cpu_only=True)
        with patch.object(engine, 'complete', return_value={'notes': {'10': 'Claim.'}, 'gap': ''}):
            with self.assertRaisesRegex(ValueError, 'each supplied source'):
                engine.generate(None, 'valve', evidence)

    def test_company_name_is_not_relaxed_to_ordinary_words_or_semantic_neighbors(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'research.sqlite')
            try:
                with store.db:
                    store.put({'AccessionNumber': 'OTHER', 'DocumentTitle': 'General matters of licensing',
                               'content': 'General matters are discussed in this reactor license.'})
                    store.put({'AccessionNumber': 'ORDINARY', 'DocumentTitle': 'Old hearing',
                               'content': 'As a general matter, this reactor license hearing is adjourned.'})
                class WrongSemantic:
                    def search(self, *args):
                        return store.lexical('reactor license')
                self.assertEqual(hybrid_retrieve(store, 'How far along is General Matter on their license?', WrongSemantic()), [])
                with store.db:
                    store.put({'AccessionNumber': 'RIGHT', 'DocumentTitle': 'General Matter application',
                               'content': 'General Matter submitted an application. Submission is not approval.'})
                rows = hybrid_retrieve(store, 'How far along is General Matter on their license?', WrongSemantic())
                self.assertEqual([r['accession'] for r in rows], ['RIGHT'])
                self.assertEqual({r['accession'] for r in store.lexical('General Matter', title_only=True)}, {'RIGHT','OTHER'})
            finally:
                store.db.close()

    def test_named_phrase_query_and_nonentity_question(self):
        self.assertEqual(subject_phrases('How far along is General Matter on their license?'), ['General Matter'])
        self.assertEqual(fallback_queries('How far along is General Matter on their license?'), ['"General Matter"'])
        self.assertEqual(subject_phrases('What happens when a relief valve fails?'), [])

    def test_single_word_company_and_permit_history_retrieve_decisions(self):
        question = 'What is the history of Oklo attempting to get permits for their reactor?'
        self.assertEqual(subject_phrases(question), ['Oklo'])
        self.assertTrue(all('Oklo' in q and 'attempting' not in q for q in fallback_queries(question)))
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'research.sqlite')
            try:
                with store.db:
                    for i in range(240):
                        store.put({'AccessionNumber': f'NOISE{i}', 'DocumentTitle': 'Oklo reactor quality assurance',
                                   'content': 'Oklo reactor quality assurance testing.'})
                    store.put({'AccessionNumber': 'DECISION', 'DocumentTitle': 'Oklo combined license application denial',
                        'content': 'The NRC denied the application without prejudice. This does not prevent a new application.'})
                    store.put({'AccessionNumber': 'SUBMISSION', 'DocumentTitle': 'Oklo license application submittal',
                        'content': 'The applicant submitted an application for NRC review.'})
                rows = hybrid_retrieve(store, question)
                self.assertEqual(rows[0]['accession'], 'DECISION')
                self.assertIn('SUBMISSION', [r['accession'] for r in rows[:3]])
                self.assertIn('without prejudice', rows[0]['text'])
            finally:
                store.db.close()

    def test_history_retains_multiple_years_and_cpu_sources_with_exact_spans(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'research.sqlite')
            try:
                with store.db:
                    for i in range(15):
                        store.put({'AccessionNumber': f'A{i}', 'DocumentTitle': 'Oklo combined license application acceptance audit',
                            'DocumentDate': '2020-06-01', 'content': 'Oklo reactor application acceptance review audit plan. ' * 100})
                    for year, event in [(2020, 'acceptance'), (2022, 'denial'), (2025, 'readiness'), (2026, 'licensing plan')]:
                        store.put({'AccessionNumber': f'E{year}', 'DocumentTitle': f'Oklo license application {event}',
                            'DocumentDate': f'{year}-06-01', 'content': f'In {year}, the NRC documented the {event}. ' * 100})
                question = 'What is the history of Oklo permits for their reactor?'
                evidence = hybrid_retrieve(store, question)
                self.assertEqual({s['document_date'][:4] for s in evidence[:4]}, {'2020','2022','2025','2026'})
                _, data, audit = pack(question, evidence)
                self.assertEqual(len(data['sources']), 4)
                self.assertLessEqual(audit['input_bytes'], 4500)
                for item, span in zip(data['sources'], audit['source_spans']):
                    text = evidence[item['source_id'] - 1]['text']
                    self.assertEqual(item['text'], text[span['start_character']:span['end_character']])
            finally:
                store.db.close()

    def test_named_subject_topic_prefers_license_over_unrelated_company_records(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'research.sqlite')
            try:
                with store.db:
                    store.put({'AccessionNumber': 'SURVEY', 'DocumentTitle': 'General Matter habitat survey',
                               'content': 'General Matter surveyed wildlife.'})
                    store.put({'AccessionNumber': 'LICENSE', 'DocumentTitle': 'General Matter license application acceptance',
                               'content': 'The General Matter license application is accepted for review. No license has been issued.'})
                rows = hybrid_retrieve(store, 'How far along is General Matter on their license?')
                self.assertEqual(rows[0]['accession'], 'LICENSE')
                with store.db:
                    store.put({'AccessionNumber': 'LICENSE', 'DocumentTitle': 'General Matter license application acceptance',
                               'content': 'The application is accepted for detailed review; it is not an issued license. '
                                          + 'Procedural background and review scope. ' * 250
                                          + '\n\nDISTRIBUTION: General Matter license application acceptance.'})
                rows = hybrid_retrieve(store, 'How far along is General Matter on their license?', focus=['LICENSE'])
                self.assertEqual(rows[0]['ordinal'], 1)
                self.assertIn('not an issued license', rows[0]['text'])
            finally:
                store.db.close()

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

    def test_live_focus_survives_semantic_storage_failure(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'research.sqlite')
            doc = {'AccessionNumber': 'ML26300A017', 'DocumentTitle': 'General Matter license application acceptance',
                   'content': 'The application is accepted for detailed review; it is not an issued license. '
                              + 'Procedural background and review scope. ' * 250
                              + '\n\nDISTRIBUTION: General Matter license application acceptance.'}
            class Client:
                def request(self, payload=None, accession=None):
                    return {'document': doc} if accession else {'count': 1, 'results': [{'document': doc}]}
            class FullDisk:
                def sync(self, **kwargs):
                    raise RuntimeError('Storage pause')
                def search(self, *args):
                    raise RuntimeError('Storage pause')
            try:
                result = research(store, 'How far along is General Matter on their license?',
                                  client=Client(), semantic=FullDisk(), generate=False)
                self.assertEqual(result['sources'][0]['ordinal'], 1)
                self.assertIn('not an issued license', result['sources'][0]['text'])
                self.assertTrue(any('Using keyword retrieval' in w for w in result['warnings']))
            finally:
                store.db.close()

    def test_live_documents_are_validated_before_strict_markdown_retrieval(self):
        from adams_rag.markdown_archive import setup, saved_markdown
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'research.sqlite')
            try:
                setup(store.db)
                with store.db:
                    store.db.execute('UPDATE markdown_policy SET enabled=1 WHERE id=1')
                class Client:
                    def request(self, payload=None, accession=None):
                        doc = {'AccessionNumber': 'ML26300A017', 'DocumentTitle': 'General Matter License Application',
                               'content': 'The General Matter application was accepted for review. Acceptance is not license issuance.'}
                        return {'document': doc} if accession else {'count': 1, 'results': [{'document': doc}]}
                result = research(store, 'How far along is General Matter on their license?', client=Client(), generate=False)
                self.assertEqual(result['sources'][0]['accession'], 'ML26300A017')
                self.assertEqual(result['discovery']['markdown_checked'], 1)
                self.assertIn('Acceptance is not license issuance.', saved_markdown(store, 'ML26300A017'))
            finally:
                store.db.close()

    def test_discovery_does_not_invalidate_unfetched_indexed_sources(self):
        from adams_rag.markdown_archive import setup, prepare_fetched
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'research.sqlite')
            docs = [{'AccessionNumber': f'ML22006A26{i}', 'DocumentTitle': f'Oklo license decision {i}',
                     'content': f'Record {i}: The NRC denied the application without prejudice.', 'DocumentDate': '2022-01-06'}
                    for i in range(2)]
            class Client:
                def request(self, payload=None, accession=None):
                    if accession:
                        return {'document': next(d for d in docs if d['AccessionNumber'] == accession)}
                    return {'count': 2, 'results': [{'document': {**d, 'SearchOnlyField': 'discovery', 'content': 'snippet'}} for d in docs]}
            try:
                setup(store.db)
                with store.db:
                    store.db.execute('UPDATE markdown_policy SET enabled=1 WHERE id=1')
                    for doc in docs:
                        store.put(doc, 'api_document_unverified')
                prepare_fetched(store, [d['AccessionNumber'] for d in docs])
                result = research(store, 'What happened to Oklo?', client=Client(), generate=False, max_documents=1)
                self.assertEqual(len(result['discovery']['fetched']), 1)
                self.assertEqual(store.db.execute("SELECT count(*) FROM markdown_checks WHERE state='ready'").fetchone()[0], 2)
                self.assertEqual({s['accession'] for s in result['sources']}, {d['AccessionNumber'] for d in docs})
                for row in store.db.execute('SELECT metadata FROM documents'):
                    self.assertNotIn('SearchOnlyField', json.loads(row[0]))
            finally:
                store.db.close()


if __name__ == '__main__':
    unittest.main()
