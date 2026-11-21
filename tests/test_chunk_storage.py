import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from adams_rag.chunk_migrate import migrate
from adams_rag.chunk_storage import MARKER, install_reader
from adams_rag.evidence import SourceBindings, saved_run_view
from adams_rag.markdown_archive import export_batch, setup as markdown_setup
from adams_rag.markdown_migrate import migrate_versions
from adams_rag.research import hybrid_retrieve, research
from adams_rag.store import Store


class ChunkStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'research.sqlite')
        self.body = ('Valve pressure is 500 psig. Unicode 核能 café.\n' * 160)
        with self.store.db:
            self.store.put({'AccessionNumber':'ONE', 'DocumentTitle':'Valve report', 'content':self.body})

    def tearDown(self):
        self.store.db.close(); self.tmp.cleanup()

    def chunks(self):
        return [dict(r) for r in self.store.db.execute('SELECT * FROM chunks ORDER BY id')]

    def test_exact_chunks_retrieval_saved_citations_and_vectors_survive_vacuum(self):
        before, sources = self.chunks(), hybrid_retrieve(self.store, 'valve pressure')
        run = research(self.store, 'valve pressure', live=False, generate=False)
        chunk = before[0]['id']
        with self.store.db:
            self.store.db.execute('INSERT INTO dense_vectors VALUES(?,?,?,?,?)', (chunk,'fixture',1,b'1234','fixture-hash'))
        vectors = [tuple(r) for r in self.store.db.execute('SELECT * FROM dense_vectors')]
        generation = [tuple(r) for r in self.store.db.execute('SELECT * FROM vector_generations')]
        result = migrate(self.store, batch_size=1)
        self.assertEqual(result['chunks'], len(before))
        self.assertGreater(result['old_bytes'], result['reference_bytes'] * 5)
        self.assertEqual(self.chunks(), before)
        self.assertEqual(hybrid_retrieve(self.store, 'valve pressure'), sources)
        self.assertEqual([tuple(r) for r in self.store.db.execute('SELECT * FROM dense_vectors')], vectors)
        self.assertEqual([tuple(r) for r in self.store.db.execute('SELECT * FROM vector_generations')], generation)
        self.assertTrue(all(r[0] == 'blob' for r in self.store.db.execute('SELECT typeof(text) FROM chunks')))
        self.store.db.execute('VACUUM')
        self.assertEqual(self.chunks(), before)
        self.assertEqual(self.store.db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
        self.assertEqual(self.store.db.execute('PRAGMA foreign_key_check').fetchall(), [])
        self.assertTrue(all(s['state']=='no_matching_review' for s in saved_run_view(self.store,run['id'])['review_updates']['sources']))
        self.store.db.close(); self.store = Store(self.root / 'research.sqlite')
        self.assertEqual(self.chunks(), before)

    def test_new_ingestion_title_refresh_and_source_replacement(self):
        migrate(self.store)
        with self.store.db:
            self.store.put({'AccessionNumber':'TWO','DocumentTitle':'Turbine','content':'A turbine spins. '*120})
            self.store.put({'AccessionNumber':'ONE','DocumentTitle':'Updated valve report'}, 'api_search_excerpt')
        self.assertTrue(self.store.lexical('updated valve'))
        self.assertTrue(all(r[0]=='blob' for r in self.store.db.execute('SELECT typeof(text) FROM chunks')))
        with self.store.db:
            self.store.put({'AccessionNumber':'ONE','DocumentTitle':'Replacement','content':'Diesel generator testing. '*100})
        self.assertFalse(self.store.lexical('valve'))
        self.assertTrue(self.store.lexical('diesel'))
        self.assertEqual(self.store.db.execute('PRAGMA foreign_key_check').fetchall(), [])

    def test_pdf_empty_pages_repeated_text_and_embedded_formfeed(self):
        pages = ['Identical valve page 核能.', '', 'Identical valve page 核能.', 'Internal\fcontrol with valve text.']
        with self.store.db:
            self.store.put({'AccessionNumber':'PDF','DocumentTitle':'Physical pages','LocalFileSHA256':'a'*64}, 'pdf', pages)
        before = self.chunks()
        migrate(self.store)
        self.assertEqual(self.chunks(), before)
        bindings = SourceBindings(self.store.db)
        for row in self.chunks():
            if row['accession']=='PDF':
                _, binding = bindings.bind(row)
                self.assertEqual(pages[row['page']-1][binding['start_character']:binding['end_character']],row['text'])

    def test_bad_chunk_is_preserved_as_holdout_and_resume_is_idempotent(self):
        with self.store.db:
            self.store.put({'AccessionNumber':'BAD','DocumentTitle':'Bad','content':'Valid original text.'})
            self.store.db.execute("UPDATE chunks SET text='corrupted cache' WHERE accession='BAD'")
        partial = migrate(self.store, max_documents=1)
        self.assertEqual(partial['holdouts'],1)
        result = migrate(self.store)
        self.assertEqual(result['state'],'complete')
        self.assertEqual(result['holdouts'],1)
        self.assertEqual(self.store.db.execute("SELECT text FROM chunks WHERE accession='BAD'").fetchone()[0],'corrupted cache')
        again = migrate(self.store)
        self.assertEqual(again['chunks'], result['chunks'])

    def test_tampered_reference_or_source_fails_closed_even_after_cached_read(self):
        migrate(self.store)
        self.chunks()
        row = self.store.db.execute('SELECT id,CAST(text AS BLOB) AS raw FROM chunks LIMIT 1').fetchone()
        with self.store.db:
            self.store.db.execute('UPDATE chunks SET text=? WHERE id=?', (row['raw'][:-1]+bytes([row['raw'][-1]^1]),row['id']))
        with self.assertRaisesRegex(ValueError,'passage checksum'):
            self.chunks()
        with self.store.db:
            self.store.db.execute('UPDATE chunks SET text=? WHERE id=?',(row['raw'],row['id']))
            self.store.db.execute("UPDATE document_versions SET text_hash='bad'")
        with self.assertRaisesRegex(ValueError,'metadata checksum'):
            self.chunks()

    def test_canonical_markdown_is_reused_and_changed_archive_fails_closed(self):
        markdown_setup(self.store.db)
        directory = self.root/'markdown'; directory.mkdir()
        export_batch(self.store,[self.store.db.execute('SELECT * FROM documents').fetchone()],directory,{})
        migrate_versions(self.store,dict(versions_migrated=0,historical_versions_exported=0,redundant_database_payload_bytes=0))
        before = self.chunks()
        migrate(self.store)
        self.assertEqual(self.chunks(),before)
        check = self.store.db.execute('SELECT archive FROM markdown_checks').fetchone()
        Path(check[0]).write_bytes(b'broken archive')
        with self.assertRaises(ValueError):
            self.chunks()

    def test_embedding_window_invalidation_is_restored_without_deleting_windows(self):
        with self.store.db:
            self.store.db.execute('CREATE TABLE embedding_window_sets(chunk_id INTEGER PRIMARY KEY)')
            self.store.db.execute('INSERT INTO embedding_window_sets SELECT id FROM chunks')
            self.store.db.execute('''CREATE TRIGGER window_source_chunk_update_v1 AFTER UPDATE ON chunks
                WHEN old.text IS NOT new.text OR old.accession IS NOT new.accession
                BEGIN DELETE FROM embedding_window_sets WHERE chunk_id=new.id; END''')
        n = self.store.db.execute('SELECT count(*) FROM embedding_window_sets').fetchone()[0]
        migrate(self.store)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM embedding_window_sets').fetchone()[0],n)
        with self.store.db:
            self.store.db.execute("UPDATE chunks SET text='changed' WHERE id=(SELECT min(id) FROM chunks)")
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM embedding_window_sets').fetchone()[0],n-1)

    def test_plain_sqlite_reader_can_be_installed_without_mutating_schema(self):
        before = self.chunks(); migrate(self.store)
        db = sqlite3.connect(self.store.path.as_uri()+'?mode=ro',uri=True)
        try:
            install_reader(db)
            self.assertEqual([dict(r) for r in db.execute('SELECT * FROM chunks ORDER BY id')],before)
        finally:
            db.close()


if __name__ == '__main__':
    unittest.main()
