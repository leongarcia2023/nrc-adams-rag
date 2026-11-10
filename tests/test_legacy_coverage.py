import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

from adams_rag.legacy_coverage import measure
from adams_rag.legacy_links import build, default_path
from adams_rag.store import Store


class LegacyTextCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = self.root / 'catalog.sqlite'
        db = sqlite3.connect(self.catalog)
        db.executescript('''CREATE TABLE libraries(library TEXT PRIMARY KEY,final_count INTEGER,finished TEXT,state TEXT);
            CREATE TABLE items(library TEXT,record_id TEXT,accession TEXT,is_package INTEGER,metadata BLOB,
                               PRIMARY KEY(library,record_id));''')
        rows = [('main','id:one','ML20001A001',['NUDOCS 0000000001','NUDOCS 0000000002']),
                ('main','id:two','ML20001A002',['NUDOCS 0000000001']),
                ('main','id:three','ML20001A003',['NUDOCS 0000000003']),
                ('main','id:four','ML20001A004',[]),
                ('legacy','id:legacy-a','0000000001',[]),
                ('legacy','id:legacy-b','0000000001',[]),
                ('legacy','id:legacy-c','0000000002',[]),
                ('legacy','id:legacy-d','0000000003',[])]
        for library, identity, accession, refs in rows:
            db.execute('INSERT INTO items VALUES(?,?,?,?,?)',
                       (library,identity,accession,0,zlib.compress(json.dumps({'DocumentReportNumber':refs}).encode())))
        for library in ['main','legacy']:
            db.execute('INSERT INTO libraries VALUES(?,?,?,?)',(library,4,'2026-10-31T00:00:00+00:00','complete'))
        db.commit();db.close()
        build(self.catalog)
        self.path = self.root / 'research.sqlite'
        store = Store(self.path)
        with store.db:
            for accession, identity in [('ML20001A001','one'),('ML20001A002','wrong'),('ML20001A004','four'),
                                        ('ML20001A005','outside'),('ML20001A006',None)]:
                item = {'AccessionNumber':accession,'content':'Retained document text','DocumentTitle':'Fixture'}
                if identity:
                    item['IdT'] = '{' + identity.upper() + '}'
                store.put(item)
        store.db.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_reports_reference_coverage_without_merging_legacy_records(self):
        result = measure(self.path,self.catalog)
        expected = {'indexed_main_candidates':5,'candidates_without_main_nrc_id':1,
                    'identity_matched_main_records_with_text':2,'identified_candidates_not_matched_to_base_catalog':2,
                    'reference_relationships':4,'referenced_legacy_accessions':3,
                    'reference_relationships_with_retained_main_text':2,'referenced_main_records_with_text':1,
                    'legacy_accessions_with_any_referenced_main_text':2,
                    'legacy_accessions_with_all_referenced_main_text':1,
                    'legacy_accessions_without_referenced_main_text':1}
        self.assertEqual({k:result[k] for k in expected},expected)
        self.assertFalse(result['identity_equivalence_verified'])
        self.assertFalse(result['source_completeness_verified'])
        db=sqlite3.connect(self.catalog)
        self.assertEqual(db.execute("SELECT count(*) FROM items WHERE library='legacy' AND accession='0000000001'").fetchone()[0],2)
        db.close()

    def test_only_indexed_text_is_counted_and_sources_are_unchanged(self):
        db=sqlite3.connect(self.path)
        db.execute("UPDATE documents SET status='missing_text' WHERE accession='ML20001A001'");db.commit();db.close()
        result=measure(self.path,self.catalog)
        self.assertEqual(result['referenced_main_records_with_text'],0)
        self.assertEqual(result['legacy_accessions_without_referenced_main_text'],3)
        for path in [self.path,self.catalog,default_path(self.catalog)]:
            db=sqlite3.connect(path)
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='available_main'").fetchone())
            db.close()

    def test_stale_or_unfinished_reference_map_is_rejected(self):
        db=sqlite3.connect(default_path(self.catalog))
        db.execute("UPDATE link_state SET state='running'");db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'complete legacy reference map'):
            measure(self.path,self.catalog)
        db=sqlite3.connect(default_path(self.catalog));db.execute("UPDATE link_state SET state='complete'");db.commit();db.close()
        db=sqlite3.connect(self.catalog);db.execute("UPDATE libraries SET finished='2026-11-01T00:00:00+00:00'");db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'exact reconciled catalog'):
            measure(self.path,self.catalog)

    def test_reference_count_disagreement_is_not_reported_as_coverage(self):
        db=sqlite3.connect(default_path(self.catalog));db.execute('UPDATE link_state SET links=links+1');db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'row count'):
            measure(self.path,self.catalog)

    def test_cancellation_does_not_leave_a_durable_work_table(self):
        with self.assertRaises(InterruptedError):
            measure(self.path,self.catalog,cancelled=lambda:True)
        db=sqlite3.connect(self.path)
        self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='available_main'").fetchone())
        self.assertEqual(db.execute('SELECT count(*) FROM documents').fetchone()[0],5)
        db.close()


if __name__ == '__main__':
    unittest.main()
