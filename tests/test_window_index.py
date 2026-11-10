import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adams_rag.embedding_windows import MODEL, POLICY, WindowPlanner
from adams_rag.evidence import read_snapshot
from adams_rag.extraction_review import record
from adams_rag.store import Store
from adams_rag.storage import StoragePause
from adams_rag.window_backfill import WindowBackfill
from adams_rag.window_index import WindowIndex, WindowSemanticIndex

FIXTURE_INDEX = Path(__file__).resolve().parent / '.cache/index'


def vector(a=1., b=0.):
    value = np.zeros(384, dtype='<f4'); value[0],value[1] = a,b
    return value


class Encoder:
    def __init__(self):
        self.queries = 0
        self.after = lambda: None

    def passage_embed(self, texts, **kwargs):
        return iter([vector() if 'NeedleThermal' in text else vector(0,1) for text in texts])

    def query_embed(self, text):
        self.queries += 1
        self.after()
        return iter([vector()])


class WindowIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FIXTURE_INDEX / "mlx-bge/tokenizer.json").is_file():
            raise unittest.SkipTest("Run python scripts/prepare_test_assets.py for tokenizer integration tests")
        cls.planner = WindowPlanner(FIXTURE_INDEX)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.store = Store(self.root / 'db.sqlite'); self.directory = self.root / 'index'; self.directory.mkdir()
        self.encoder = Encoder()
        self.backfill = WindowBackfill(self.store, self.planner, self.encoder)
        self.acc = 'ML24017A120'
        body = 'RV-493.2-A set-pressure=500psig. ' * 90 + ' NeedleThermal hidden indication.'
        with self.store.db:
            for acc, text, title, date, docket, base in (
                (self.acc, body, 'Long valve report', '2024-01-02', '05000373', vector(0,1)),
                ('ML24017A121', 'Competing valve evidence.', 'Competing report', '2023-01-01', '05000374', vector(.8,.6))):
                self.store.put({'AccessionNumber':acc,'DocumentTitle':title,'content':text,'DocumentDate':date,'DocketNumber':[docket]})
                rows = self.store.db.execute('SELECT id FROM chunks WHERE accession=?',(acc,)).fetchall()
                self.assertEqual(len(rows),1)
                plan = self.backfill.windows.plan(rows[0][0])
                self.store.db.execute('INSERT INTO dense_vectors VALUES(?,?,?,?,?)',(rows[0][0],MODEL,384,base.tobytes(),plan.source_hash))
        self.chunk = self.store.db.execute('SELECT id FROM chunks WHERE accession=?',(self.acc,)).fetchone()[0]
        self.backfill.sync()
        self.index = WindowSemanticIndex(self.store, self.directory, self.encoder)

    def tearDown(self):
        self.store.db.close();self.temp.cleanup()

    def search(self, **kwargs):
        return self.index.search('hidden indication', **kwargs)

    def manifest(self):
        return json.loads(self.index.window_index.manifest.read_text())

    def test_tail_hit_is_merged_once_and_bound_to_retained_parent(self):
        matches = self.search()
        self.assertEqual([row['accession'] for row in matches],[self.acc,'ML24017A121'])
        self.assertEqual(len({row['id'] for row in matches}),len(matches))
        self.assertGreater(matches[0]['window_match']['ordinal'],1)
        self.assertEqual(matches[0]['source_version']['state'],'matched_retained_text')
        self.assertEqual(matches[0]['source_version']['passage_sha256'],hashlib.sha256(matches[0]['text'].encode()).hexdigest())
        self.assertEqual(self.encoder.queries,1)
        self.assertEqual(len(self.index.window_index.index),self.backfill.status()['stored_window_vectors'])
        self.assertLess(len(self.index.window_index.index),self.store.db.execute('SELECT count(*) FROM embedding_windows').fetchone()[0])

    def test_reopened_graph_reuses_checked_manifest(self):
        self.search()
        reopened = WindowSemanticIndex(self.store,self.directory,self.encoder)
        with patch.object(reopened.window_index,'_build',side_effect=AssertionError('unnecessary rebuild')):
            self.assertEqual(reopened.search('indication')[0]['accession'],self.acc)

    def test_whole_additional_graph_request_keeps_tail_candidate(self):
        from types import SimpleNamespace
        self.search();native=self.index.window_index.index.search
        def partial(vector,**kwargs):
            if not kwargs.get('exact'):
                return SimpleNamespace(keys=np.asarray([],dtype=np.uint64))
            return native(vector,**kwargs)
        with patch.object(self.index.window_index.index,'search',side_effect=partial):
            self.assertEqual(self.search()[0]['accession'],self.acc)

    def test_same_count_vector_change_rebuilds_and_changes_rank(self):
        self.search();old = self.manifest()
        with self.store.db:
            self.store.db.execute('UPDATE embedding_windows SET vector=? WHERE vector IS NOT NULL',(vector(0,1).tobytes(),))
        self.assertEqual(self.search()[0]['accession'],'ML24017A121')
        new = self.manifest()
        self.assertGreater(new['generation'],old['generation'])
        self.assertEqual(new['build_mode'],'incremental')
        self.assertEqual(new['vectors'],old['vectors'])
        self.assertFalse((self.index.window_index.directory / old['graph']).exists())

    def test_removal_of_all_extra_vectors_publishes_empty_graph(self):
        self.search()
        with self.store.db:
            self.store.db.execute('UPDATE embedding_windows SET dimension=NULL,vector=NULL WHERE vector IS NOT NULL')
        self.assertEqual(self.search()[0]['accession'],'ML24017A121')
        self.assertEqual(self.manifest()['vectors'],0)

    def test_title_change_removes_old_windows_and_prefix(self):
        self.search()
        with self.store.db:
            self.store.put({'AccessionNumber':self.acc,'DocumentTitle':'Changed title'})
        self.assertNotIn(self.acc,{r['accession'] for r in self.search()})
        self.assertEqual(self.manifest()['vectors'],0)

    def test_current_exclusion_applies_without_graph_rebuild(self):
        self.search();before = self.index.window_index.manifest.read_bytes()
        digest = self.store.db.execute('SELECT text_hash FROM documents WHERE accession=?',(self.acc,)).fetchone()[0]
        record(self.store,self.acc,digest,0,'exclude',['incorrect_text'],'Known incorrect source text in this fixture.')
        excluded=[]
        matches=self.index.search_with_diagnostics('hidden indication',exclusions=excluded)
        self.assertNotIn(self.acc,{r['accession'] for r in matches})
        self.assertEqual(excluded[0]['accession'],self.acc)
        self.assertEqual(self.index.window_index.manifest.read_bytes(),before)

    def test_corrupt_retained_source_is_withheld_and_reported(self):
        self.search()
        with self.store.db:
            self.store.db.execute('UPDATE document_versions SET content=? WHERE accession=?',(b'invalid compressed source',self.acc))
        issues=[]
        matches=self.index.search_with_diagnostics('hidden indication',source_issues=issues)
        self.assertNotIn(self.acc,{r['accession'] for r in matches})
        self.assertEqual(issues[0]['reason'],'retained_text_unreadable')

    def test_docket_and_date_filters_apply_to_both_indexes(self):
        self.assertEqual([r['accession'] for r in self.search(docket='05000374')],['ML24017A121'])
        self.assertEqual([r['accession'] for r in self.search(date_from='2024-01-01')],[self.acc])
        self.assertEqual([r['accession'] for r in self.search(date_to='2023-12-31')],['ML24017A121'])

    def test_corrupt_graph_is_rebuilt_from_canonical_vectors(self):
        self.search();old = self.manifest()
        (self.index.window_index.directory/old['graph']).write_bytes(b'broken graph')
        reopened=WindowSemanticIndex(self.store,self.directory,self.encoder)
        self.assertEqual(reopened.search('indication')[0]['accession'],self.acc)
        self.assertEqual(json.loads(reopened.window_index.manifest.read_text())['build_mode'],'full')

    def test_failed_manifest_swap_keeps_previous_graph_pair(self):
        self.search();manifest=self.index.window_index.manifest;before=manifest.read_bytes();old=self.manifest()
        with self.store.db:
            self.store.db.execute('UPDATE embedding_windows SET vector=? WHERE vector IS NOT NULL',(vector(0,1).tobytes(),))
        import os
        replace=os.replace
        def fail(source,target):
            if Path(target)==manifest:
                raise OSError('simulated interruption before publication')
            return replace(source,target)
        with patch('adams_rag.window_index.os.replace',side_effect=fail),self.assertRaises(OSError):
            self.search()
        self.assertEqual(manifest.read_bytes(),before)
        self.assertTrue((manifest.parent/old['graph']).exists())
        self.assertEqual(self.search()[0]['accession'],'ML24017A121')

    def test_storage_pause_cannot_replace_existing_graph(self):
        self.search();before=self.index.window_index.manifest.read_bytes()
        with self.store.db:
            self.store.db.execute('UPDATE embedding_windows SET vector=? WHERE vector IS NOT NULL',(vector(0,1).tobytes(),))
        with patch('adams_rag.window_index.require_space',side_effect=StoragePause('reserve')),self.assertRaises(StoragePause):
            self.search()
        self.assertEqual(self.index.window_index.manifest.read_bytes(),before)

    def test_corrupt_window_plan_fails_closed_before_publication(self):
        self.search();before=self.index.window_index.manifest.read_bytes()
        with self.store.db:
            self.store.db.execute('UPDATE embedding_windows SET start_character=1 WHERE vector IS NOT NULL')
        with self.assertRaises(ValueError):
            self.search()
        self.assertEqual(self.index.window_index.manifest.read_bytes(),before)

    def test_missing_base_does_not_hide_valid_tail_semantic_coverage(self):
        with self.store.db:
            self.store.db.execute('DELETE FROM dense_vectors WHERE chunk_id=?',(self.chunk,))
        self.assertIn(self.chunk,self.index.covered_chunk_ids([self.chunk]))
        self.assertEqual(self.search()[0]['accession'],self.acc)
        self.assertEqual(self.backfill.windows.resolve(self.chunk)['state'],'pending')

    def test_outer_snapshot_keeps_source_identity_during_concurrent_revision(self):
        writer=Store(self.store.path)
        def revise():
            with writer.db:
                writer.put({'AccessionNumber':self.acc,'DocumentTitle':'Concurrent revised title'})
        self.encoder.after=revise
        try:
            with read_snapshot(self.store.db):
                old=self.store.db.execute('SELECT title FROM documents WHERE accession=?',(self.acc,)).fetchone()[0]
                match=self.search()[0]
                self.assertEqual(match['title'],old)
                self.assertEqual(match['source_version']['state'],'matched_retained_text')
            self.encoder.after=lambda:None
            self.assertNotIn(self.acc,{r['accession'] for r in self.search()})
        finally:
            writer.db.close()

    def test_invalid_query_is_rejected_before_graph_publication(self):
        for bad in (np.ones(2),np.zeros(384),np.full(384,np.nan)):
            with self.subTest(shape=bad.shape),self.assertRaises(ValueError):
                self.index.window_index.candidates(bad)
        self.assertFalse(self.index.window_index.manifest.exists())

    def test_activation_requires_explicit_compatible_configuration(self):
        from adams_rag.semantic_index import search_index, SemanticIndex
        self.assertIs(type(search_index(self.store,self.directory)),SemanticIndex)
        (self.directory/'ACTIVE_MODEL.json').write_text(json.dumps({'version':1,'model':MODEL}))
        config={'version':1,'model':MODEL,'policy':POLICY,'enabled':False}
        path=self.directory/'WINDOW_SEARCH.json';path.write_text(json.dumps(config))
        self.assertIs(type(search_index(self.store,self.directory)),SemanticIndex)
        path.write_text(json.dumps({**config,'enabled':True}))
        self.assertIsInstance(search_index(self.store,self.directory),WindowSemanticIndex)
        for changes in ({'version':True},{'enabled':1},{'policy':'unknown'},{'extra':True}):
            path.write_text(json.dumps({**config,**changes}))
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                search_index(self.store,self.directory)
        path.write_text(json.dumps({**config,'enabled':True}))
        (self.directory/'ACTIVE_MODEL.json').unlink()
        with self.assertRaises(ValueError):
            search_index(self.store,self.directory)

    def test_hybrid_route_reports_exclusion_once_across_both_channels(self):
        from adams_rag.research import hybrid_retrieve
        digest=self.store.db.execute('SELECT text_hash FROM documents WHERE accession=?',(self.acc,)).fetchone()[0]
        record(self.store,self.acc,digest,0,'exclude',['incorrect_text'],'Known bad text in this fixture.')
        excluded=[]
        sources=hybrid_retrieve(self.store,'hidden indication',semantic=self.index,exclusions=excluded)
        self.assertNotIn(self.acc,{s['accession'] for s in sources})
        self.assertEqual(len(excluded),1)
        self.assertEqual(excluded[0]['accession'],self.acc)

    def test_hybrid_route_reports_retained_source_failure_once(self):
        from adams_rag.research import hybrid_retrieve
        with self.store.db:
            self.store.db.execute('UPDATE document_versions SET content=? WHERE accession=?',(b'corrupt',self.acc))
        issues=[]
        sources=hybrid_retrieve(self.store,'hidden indication',semantic=self.index,source_issues=issues)
        self.assertNotIn(self.acc,{s['accession'] for s in sources})
        self.assertEqual(len(issues),1)
        self.assertEqual(issues[0]['reason'],'retained_text_unreadable')


if __name__=='__main__':
    unittest.main()
