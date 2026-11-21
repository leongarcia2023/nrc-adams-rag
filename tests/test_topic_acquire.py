import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adams_rag.client import BudgetStop
from adams_rag.store import Store
from adams_rag.topic_acquire import acquire


def doc(accession='ML22006A267', title='Oklo license application decision'):
    return {'AccessionNumber': accession, 'DocumentTitle': title,
            'DocumentDate': '2022-01-06', 'content': 'The application was denied without prejudice.'}


class TopicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / 'research.sqlite')
        self.space = patch('adams_rag.topic_acquire.require_space').start()

    def tearDown(self):
        patch.stopall()
        self.store.db.close()
        self.temp.cleanup()

    def test_resumes_pending_document_without_repeating_discovery(self):
        calls = []
        class Client:
            fail = True
            def search(self, **kwargs):
                calls.append(kwargs['skip'])
                return {'count': 1, 'results': [{'document': doc()}] if kwargs['skip'] == 0 else []}
            def request(self, accession):
                if self.fail:
                    raise BudgetStop('cooldown')
                return {'document': doc(accession)}
        client = Client()
        result = acquire(self.store, client, self.root / 'queue', 'Oklo', 'Oklo')
        self.assertEqual(result['state'], 'budget_pause')
        self.assertFalse(self.store.lexical('Oklo'))  # Search snippets never become evidence.
        client.fail = False
        result = acquire(self.store, client, self.root / 'queue', 'Oklo', 'Oklo')
        self.assertEqual(calls, [0, 1])
        self.assertEqual(result['downloaded_this_run'], 1)
        self.assertEqual(result['state'], 'complete_observed_queue')
        self.assertFalse(result['full_archive_verified'])
        self.assertIn('without prejudice', self.store.lexical('Oklo')[0]['text'])

    def test_identity_mismatch_leaves_pending_and_unrelated_titles_are_skipped(self):
        class Client:
            def search(self, **kwargs):
                return {'count': 2, 'results': [{'document': doc()}, {'document': doc('ML22006A268', 'Unrelated license')}] if kwargs['skip'] == 0 else []}
            def request(self, accession):
                return {'document': doc('ML22006A999')}
        with self.assertRaisesRegex(ValueError, 'identity'):
            acquire(self.store, Client(), self.root / 'queue', 'Oklo', 'Oklo')
        import json
        result = json.loads((self.root / 'queue/STATUS.json').read_text())
        self.assertEqual(result['outcomes'], {'pending': 1})
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM documents').fetchone()[0], 0)

    def test_repeated_page_does_not_advance_cursor(self):
        class Client:
            def search(self, **kwargs):
                return {'count': 100, 'results': [{'document': doc()}]}
        result = acquire(self.store, Client(), self.root / 'queue', 'Oklo')
        self.assertEqual(result['state'], 'discovery_blocked')
        self.assertEqual(result['examined_search_records'], 1)


if __name__ == '__main__':
    unittest.main()
