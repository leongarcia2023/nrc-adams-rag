import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adams_rag.cpu_context import pack_compact, answer_schema, check_compact_answer, complete_excerpt
from adams_rag.interactive import AnswerCache, cache_key, collection_revision, coverage_snapshot
from adams_rag.markdown_archive import prepare_fetched, setup
from adams_rag.query_terms import inquiry_kind
from adams_rag.research import research, hybrid_retrieve
from adams_rag.server import Application
from adams_rag.store import Store


class InteractiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / 'research.sqlite')

    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()

    def put(self, accession='ML22006A267', title='Oklo license application denial', text=None, **fields):
        doc = {'AccessionNumber': accession, 'DocumentTitle': title,
               'content': text or 'The NRC denied the Oklo application without prejudice. Oklo may reapply.', **fields}
        with self.store.db:
            self.store.put(doc, 'api_document_unverified')
        return doc

    def test_storage_paused_questions_reuse_checked_text_without_fetching_or_embedding(self):
        doc = self.put()
        setup(self.store.db)
        with self.store.db:
            self.store.db.execute('UPDATE markdown_policy SET enabled=1')
        prepare_fetched(self.store, [doc['AccessionNumber']])
        calls = []
        class Client:
            def request(self, payload=None, accession=None):
                calls.append(accession)
                if accession:
                    raise AssertionError('Cached text must not be downloaded again')
                return {'count': 1, 'results': [{'document': doc}]}
        class Index:
            model = 'test'
            def sync(self, **kwargs):
                raise AssertionError('A question must not build embeddings')
            def search(self, *args):
                return []
        with patch('adams_rag.research.require_space', side_effect=AssertionError('No write needed')):
            result = research(self.store, 'What is the history of Oklo licensing?',
                              client=Client(), semantic=Index(), generate=False, interactive=True)
        self.assertTrue(result['sources'])
        self.assertEqual(result['discovery']['cached'], [doc['AccessionNumber']])
        self.assertFalse(result['discovery']['fetched'])
        self.assertTrue(all(c is None for c in calls))
        self.assertIn('not verify', ' '.join(result['warnings']))
        self.assertIn('retrieval_seconds', result['timings'])

    def test_changed_search_metadata_does_not_silently_reuse_cached_text(self):
        old = self.put()
        revised = {**old, 'DocumentTitle': 'Oklo revised license decision'}
        calls = []
        class Client:
            def request(self, payload=None, accession=None):
                calls.append(accession)
                return {'document': revised} if accession else {'count': 1, 'results': [{'document': revised}]}
        result = research(self.store, 'Oklo licensing', client=Client(), generate=False, interactive=True)
        self.assertIn(old['AccessionNumber'], calls)
        self.assertFalse(result['discovery']['cached'])

    def test_coverage_snapshot_is_dated_and_invalid_snapshots_fall_back(self):
        self.put()
        folder = self.root / 'archive'
        folder.mkdir()
        path = folder / 'STATUS.json'
        path.write_text(json.dumps({'indexed_documents': 300000, 'chunks': 4000000,
                                    'updated_utc': '2026-11-21T10:00:00+00:00'}))
        with patch.object(self.store, 'status', side_effect=AssertionError('No full scan')):
            view = coverage_snapshot(self.store)
        self.assertFalse(view['counts_are_current'])
        self.assertFalse(view['full_archive_verified'])
        path.write_text('{"indexed_documents":-1}')
        self.assertEqual(coverage_snapshot(self.store)['documents']['indexed'], 1)

    def test_cache_expires_and_invalidates_after_source_or_extraction_changes(self):
        doc = self.put()
        setup(self.store.db)
        with self.store.db:
            self.store.db.execute('UPDATE markdown_policy SET enabled=1')
        prepare_fetched(self.store, [doc['AccessionNumber']])
        tick = [0]
        cache = AnswerCache(clock=lambda: tick[0], ttl=120, limit=2)
        result = {'mode': 'generated', 'answer': 'Denied without prejudice. [1]', 'sources': []}
        revision = collection_revision(self.store)
        cache.put('q', revision, result)
        hit = cache.get('q', revision)
        hit['answer'] = 'mutation'
        self.assertNotEqual(cache.get('q', revision)['answer'], 'mutation')
        with self.store.db:
            self.store.db.execute("UPDATE markdown_checks SET state='holdout'")
        self.assertIsNone(cache.get('q', collection_revision(self.store)))
        revision = collection_revision(self.store)
        cache.put('q', revision, result)
        tick[0] = 121
        self.assertIsNone(cache.get('q', revision))
        cache.put('failed', revision, {'mode': 'evidence_only', 'answer': 'No model answer'})
        self.assertIsNone(cache.get('failed', revision))

    def test_application_reuses_only_same_request_and_unchanged_collection(self):
        self.put()
        calls = []
        class Engine:
            model = 'fixture-model'
            def generate(self, model, question, sources):
                calls.append(question)
                return {'answer': 'Denied without prejudice. [1]', 'claims': [{'text': 'Denied without prejudice.', 'source_ids': [1]}],
                        'insufficient_evidence': False, 'gaps': []}
        class Index:
            model = 'BAAI/bge-small-en-v1.5'
            encoder = None
            def search(self, *args): return []
        app = Application(self.store.path, self.root/'rate', self.root/'catalog', self.root/'model', self.root/'index')
        app.engine = Engine()
        def run(identifier):
            app.jobs[identifier] = {'stage': 'Queued', 'state': 'queued'}
            app.run(identifier, {'question': 'What happened to Oklo?', 'live': False}, True)
            self.assertEqual(app.jobs[identifier]['state'], 'finished')
            return app.jobs[identifier]['result']
        try:
            with patch('adams_rag.server.search_index', return_value=Index()), patch('adams_rag.server.load_api_key', return_value=None):
                run('one')
                second = run('two')
                self.assertTrue(second['answer_cache']['hit'])
                self.assertEqual(len(calls), 1)
                self.put(text='The corrected record says the application was withdrawn.')
                third = run('three')
                self.assertNotIn('answer_cache', third)
                self.assertEqual(len(calls), 2)
            self.assertNotEqual(cache_key({'question':'q','live':True}, app.engine,'m'), cache_key({'question':'q','live':False},app.engine,'m'))
        finally:
            app.pool.shutdown()

    def test_comparison_preserves_both_subjects_and_status_prefers_recent_evidence(self):
        for i in range(12):
            self.put(f'ATLAS{i}', 'Atlas license acceptance', 'Atlas submitted a license application. ' * 5, DocumentDate='2020-01-01')
        self.put('BEACON', 'Beacon license denial', 'The Beacon license application was denied without prejudice.', DocumentDate='2022-01-01')
        sources = hybrid_retrieve(self.store, 'Compare Atlas and Beacon licensing histories')
        self.assertTrue(any(s['accession'] == 'BEACON' for s in sources[:2]))
        self.put('NEW', 'Atlas license application review update', 'The Atlas application remains under review.', DocumentDate='2025-01-01')
        current = hybrid_retrieve(self.store, 'What is the current Atlas licensing status?')
        self.assertEqual(current[0]['accession'], 'NEW')

    def test_compact_context_is_bounded_preserves_source_slices_and_negation(self):
        sources = [{'citation': i + 1, 'accession': str(i), 'title': 'Oklo licensing milestone',
                    'document_date': f'{2020+i}-01-01', 'text': ('Header and distribution. ' * 50 +
                    'The NRC denied the Oklo application without prejudice. The company is free to reapply. ' + 'Other background. ' * 80)} for i in range(4)]
        system, data, audit = pack_compact('What is the history of Oklo permits?', sources)
        self.assertLessEqual(audit['input_bytes'], 3400)
        self.assertEqual(len(data['sources']), 4)
        for item, span in zip(data['sources'], audit['source_spans']):
            self.assertIn('without prejudice', item['text'])
            original = sources[item['source_id'] - 1]['text']
            self.assertEqual(item['text'], original[span['start_character']:span['end_character']])
        schema = answer_schema({1, 4})
        self.assertEqual(schema['properties']['paragraphs']['items']['properties']['source_ids']['items']['enum'], [1, 4])

    def test_question_types_have_distinct_evidence_tasks(self):
        examples = {'Compare two applications':'comparison', 'History of an application':'history',
                    'Latest licensing status':'status', 'How many permits exist?':'enumeration',
                    'What must a licensee report?':'requirements', 'Summarize ML21357A034':'summary',
                    'Why did the valve fail?':'explanation'}
        for question, kind in examples.items():
            self.assertEqual(inquiry_kind(question), kind)

    def test_report_dates_are_not_event_dates_and_unfinished_answers_are_rejected(self):
        text = ('June 3, 2026\n\nDistribution and administrative header.\n\n'
                'By letter dated February 21, 2025, Atlas indicated its intention to submit a license application in two phases. '
                'A readiness assessment was held in March 2025.\n\nDistribution copy.')
        source = {'citation': 1, 'accession': 'A', 'title': 'Atlas licensing audit', 'document_date': '2026-06-03', 'text': text}
        _, data, _ = pack_compact('What is the history of Atlas licensing?', [source])
        self.assertNotIn('document_date', data['sources'][0])
        snippet, start, end = complete_excerpt(text, 'history of Atlas licensing', 240)
        self.assertEqual(snippet, text[start:end])
        self.assertIn('February 21, 2025', snippet)
        self.assertNotIn('June 3, 2026', snippet)
        self.assertTrue(snippet.endswith('.'))
        cited = {'sources': [{'source_id': 1, 'text': snippet}]}
        with self.assertRaisesRegex(ValueError, 'date absent'):
            check_compact_answer({'claims': [{'text': 'Atlas applied in 2026.', 'source_ids': [1]}]}, cited)
        with self.assertRaisesRegex(ValueError, 'mid-sentence'):
            check_compact_answer({'claims': [{'text': 'The NRC reviewed the combined license application submitted by this company in order', 'source_ids': [1]}]}, cited)


if __name__ == '__main__':
    unittest.main()
