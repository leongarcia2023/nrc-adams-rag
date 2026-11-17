import json
import shutil
from pathlib import Path
import tempfile
import unittest

from adams_rag.evidence import SourceBindings, SourceBindingError
from adams_rag.ingest import archive
from adams_rag.markdown_archive import export_batch, setup
from adams_rag.markdown_migrate import covered_raw, migrate_versions, prune_raw
from adams_rag.source_content import MARKER, read_text
from adams_rag.store import Store


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'test.sqlite')
        self.directory = self.root / 'markdown'; self.directory.mkdir()
        setup(self.store.db)
        self.progress = dict(versions_migrated=0,historical_versions_exported=0,
            redundant_database_payload_bytes=0,raw_files_removed=0,raw_bytes_removed=0)

    def tearDown(self):
        self.store.db.close(); self.tmp.cleanup()

    def put(self, text='Valve safety inspection. ' * 50):
        doc = dict(AccessionNumber='ML000000001', DocumentTitle='Valve', content=text, AuthorName='NRC')
        with self.store.db:
            self.store.put(doc)
        row = self.store.db.execute('SELECT * FROM documents').fetchone()
        export_batch(self.store, [row], self.directory, {})
        return doc

    def test_migration_preserves_current_and_historical_versions_and_chunk_ids(self):
        self.put('Old inspection. ' * 30)
        old = dict(self.store.db.execute('SELECT * FROM document_versions').fetchone())
        self.put()
        chunks = [tuple(r) for r in self.store.db.execute('SELECT * FROM chunks')]
        migrate_versions(self.store, self.progress)
        rows = list(self.store.db.execute('SELECT * FROM document_versions'))
        self.assertTrue(all(r['content'].startswith(MARKER) for r in rows))
        historical = next(r for r in rows if r['version'] == old['version'])
        self.assertEqual(read_text(historical['content'], self.store.db), 'Old inspection. ' * 30)
        self.assertEqual(historical['metadata'], old['metadata'])
        self.assertEqual(chunks,[tuple(r) for r in self.store.db.execute('SELECT * FROM chunks')])
        self.assertEqual(SourceBindings(self.store.db).document('ML000000001')['body'], 'Valve safety inspection. ' * 50)
        migrate_versions(self.store, self.progress)
        self.assertEqual(self.progress['versions_migrated'], 2)

    def test_missing_archive_fails_closed_and_new_exports_offload(self):
        self.put(); migrate_versions(self.store, self.progress)
        self.put('New valve text. ' * 50)
        row = self.store.db.execute('SELECT * FROM document_versions ORDER BY rowid DESC LIMIT 1').fetchone()
        self.assertTrue(row['content'].startswith(MARKER))
        ref = json.loads(row['content'][len(MARKER):])
        (self.directory / ref['archive']).unlink()
        with self.assertRaises(SourceBindingError):
            SourceBindings(self.store.db).document('ML000000001')

    def test_only_exactly_covered_api_cache_is_removed(self):
        doc = self.put(); migrate_versions(self.store, self.progress)
        archive(self.store, {'document':doc,'score':1,'highlights':[]})
        archive(self.store, {'document':{**doc,'AuthorName':'Unique metadata'}})
        archive(self.store, {'document':{**doc,'content':'Different content'}})
        archive(self.store, {'results':[doc]})
        paths = list((self.root/'raw').rglob('*.json.gz'))
        self.assertEqual(sum(covered_raw(self.store,p)[0] is not None for p in paths),1)
        prune_raw(self.store,self.progress,self.root)
        self.assertEqual(self.progress['raw_files_removed'],1)
        self.assertEqual(len(list((self.root/'raw').rglob('*.json.gz'))),3)

    def test_changed_markdown_prevents_offload(self):
        self.put()
        check = self.store.db.execute('SELECT * FROM markdown_checks').fetchone()
        Path(check['archive']).write_bytes(b'broken zip')
        with self.assertRaises(ValueError):
            migrate_versions(self.store,self.progress)
        self.assertFalse(self.store.db.execute('SELECT content FROM document_versions').fetchone()[0].startswith(MARKER))

    def test_reference_cannot_escape_archive(self):
        self.put(); migrate_versions(self.store,self.progress)
        row = self.store.db.execute('SELECT content FROM document_versions').fetchone()
        ref = json.loads(row[0][len(MARKER):]);ref['archive']='../outside.zip'
        with self.assertRaises(ValueError):
            read_text(MARKER+json.dumps(ref).encode(), self.store.db)

    def test_metadata_refresh_preserves_migrated_text(self):
        doc = self.put(); migrate_versions(self.store, self.progress)
        with self.store.db:
            self.store.db.execute('DELETE FROM document_quality')
            self.store.put({'AccessionNumber':doc['AccessionNumber'], 'DocumentTitle':'Updated title'})
        value = SourceBindings(self.store.db).document(doc['AccessionNumber'])
        self.assertEqual(value['body'], doc['content'])
        self.assertEqual(value['document']['title'], 'Updated title')

    def test_markdown_export_survives_collection_relocation(self):
        from adams_rag.export import document_markdown
        self.put(); migrate_versions(self.store, self.progress)
        expected = document_markdown(self.store, 'ML000000001')
        moved = self.root / 'relocated'; moved.mkdir()
        shutil.copytree(self.directory, moved / 'markdown')
        destination = Store(moved / 'research.sqlite')
        try:
            self.store.db.backup(destination.db)
            for archive in self.directory.glob('*.zip'):
                archive.rename(archive.with_suffix('.old'))
            self.assertEqual(document_markdown(destination, 'ML000000001'), expected)
            self.assertEqual(SourceBindings(destination.db).document('ML000000001')['body'],
                             'Valve safety inspection. ' * 50)
            # Relocation must retain checksum validation rather than trusting a basename.
            with destination.db:
                destination.db.execute("UPDATE markdown_checks SET markdown_sha256='wrong'")
            with self.assertRaises(ValueError):
                document_markdown(destination, 'ML000000001')
        finally:
            destination.db.close()
