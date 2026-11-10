import json
import tempfile
import unittest
import zlib
from pathlib import Path

from adams_rag.catalog import Catalog, matches, number, wait_until
from adams_rag.client import BudgetStop, TemporaryRequestError
from adams_rag.metrics import METRICS, install_metrics


class Server:
    def __init__(self, libraries, page_size=2):
        self.libraries = libraries
        self.page_size = page_size
        self.calls = []

    def search(self, library="main", skip=0, filters=None):
        self.calls.append((library, skip, filters))
        docs = self.libraries[library]
        for f in filters or []:
            value = f["value"].casefold()
            def accept(d):
                a = d["AccessionNumber"].casefold()
                return {"starts": a.startswith(value), "notstarts": not a.startswith(value),
                        "equals": a == value, "notequals": a != value}[f["operator"]]
            docs = [d for d in docs if accept(d)]
        docs = sorted(docs, key=lambda d: d["AccessionNumber"])
        return {"count": len(docs), "results": [{"document": d} for d in docs[skip:skip+self.page_size]]}


def document(accession, **kwargs):
    return {"AccessionNumber": accession, "DocumentTitle": "fixture", "DocumentSize": "1234",
            "content": "This search excerpt must never become evidence.", **kwargs}


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.catalog = Catalog(self.tmp.name)

    def tearDown(self):
        self.catalog.db.close()
        self.tmp.cleanup()

    def test_complete_partition_tree_includes_unusual_and_short_accessions(self):
        source = {"main": [document(x) for x in ["A", "A/1", "A01", "A02", "B1", "Z9"]],
                  "legacy": [document(x) for x in ["000001", "000002", "A", "Ω01"]]}
        server = Server(source)
        for _ in range(100):
            if not self.catalog.step(server, max_leaf=2):
                break
        else:
            self.fail("Partition tree did not terminate")
        self.assertTrue(self.catalog.reconcile(server))
        summary = self.catalog.summary()
        self.assertEqual(summary["catalog_records"], 10)
        self.assertTrue(summary["enumeration_count_matched"])
        self.assertTrue(all(skip <= 2 for _,skip,_ in server.calls))
        for row in self.catalog.db.execute("SELECT metadata FROM items"):
            self.assertNotIn("content", json.loads(zlib.decompress(row[0])))
        self.catalog.export()
        self.assertTrue((Path(self.tmp.name)/"catalog.jsonl.gz").is_file())

    def test_checkpoint_resume_does_not_repeat_committed_pages(self):
        server = Server({"main": [document("A1"),document("A2"),document("A3")], "legacy": []})
        self.catalog.step(server, max_leaf=10)
        self.catalog.db.close()
        self.catalog = Catalog(self.tmp.name)
        while self.catalog.step(server, max_leaf=10):
            pass
        self.assertTrue(self.catalog.reconcile(server))
        self.assertEqual(self.catalog.summary()["catalog_records"], 3)
        main_offsets = [s for lib,s,_ in server.calls if lib=="main"]
        self.assertEqual(main_offsets, [0,2,0])

    def test_truncation_is_blocked(self):
        class Broken:
            def search(self, **kwargs):
                return {"count": 20, "results": []}
        with self.assertRaisesRegex(RuntimeError, "truncation"):
            self.catalog.step(Broken())
        self.assertEqual(self.catalog.summary()["partitions"]["blocked"], 1)

    def test_count_change_restarts_only_affected_leaf(self):
        server = Server({"main": [document("A1"),document("A2"),document("A3")], "legacy": []})
        self.catalog.step(server)
        self.catalog.step(server)
        server.libraries["main"].append(document("A4"))
        self.catalog.step(server)
        self.assertEqual(self.catalog.summary()["catalog_records"], 0)
        while self.catalog.step(server):
            pass
        self.assertEqual(self.catalog.summary()["catalog_records"], 4)

    def test_repeated_page_is_blocked(self):
        class Repeated(Server):
            def search(self, **kwargs):
                kwargs["skip"] = 0
                return super().search(**kwargs)
        server = Repeated({"main": [document("A1"),document("A2"),document("A3")], "legacy": []})
        self.catalog.step(server)
        self.catalog.step(server)
        with self.assertRaisesRegex(RuntimeError, "Repeated"):
            self.catalog.step(server)

    def test_failed_request_keeps_checkpoint(self):
        class Paused:
            def search(self, **kwargs):
                raise BudgetStop("test")
        with self.assertRaises(BudgetStop):
            self.catalog.step(Paused())
        self.assertEqual(self.catalog.next_partition()["offset"], 0)
        self.assertEqual(self.catalog.summary()["catalog_records"], 0)

    def test_final_count_change_prevents_completion(self):
        server = Server({"main": [document("A1")], "legacy": []})
        while self.catalog.step(server):
            pass
        server.libraries["main"].append(document("A2"))
        self.assertFalse(self.catalog.reconcile(server))
        self.assertFalse(self.catalog.summary()["enumeration_count_matched"])
        while self.catalog.step(server):
            pass
        self.assertTrue(self.catalog.reconcile(server))
        self.assertEqual(self.catalog.summary()["catalog_records"],2)

    def test_temporary_failure_preserves_saved_page_and_resumes(self):
        server = Server({"main": [document("A1"), document("A2"), document("A3")], "legacy": []})
        self.catalog.step(server)
        self.catalog.step(server)
        before = dict(self.catalog.next_partition())
        class IncompleteResponse:
            def search(self, **kwargs):
                raise TemporaryRequestError("Incomplete response after retries")
        with self.assertRaises(TemporaryRequestError):
            self.catalog.step(IncompleteResponse())
        self.assertEqual(dict(self.catalog.next_partition()), before)
        self.assertEqual(self.catalog.summary()["catalog_records"], 2)
        while self.catalog.step(server):
            pass
        self.assertTrue(self.catalog.reconcile(server))
        self.assertEqual(self.catalog.summary()["catalog_records"], 3)
        self.assertEqual([skip for lib, skip, _ in server.calls if lib == "main"], [0, 2, 0])

    def test_changed_subtree_and_new_prefix_are_reconciled(self):
        server = Server({"main":[document(x) for x in ["AA1","AA2","AB1","AC1"]],"legacy":[]})
        while self.catalog.step(server,max_leaf=2):
            pass
        server.libraries["main"].append(document("ZZ1"))
        server.libraries["main"].append(document("AA3"))
        self.assertFalse(self.catalog.reconcile(server))
        for _ in range(200):
            if not self.catalog.step(server,max_leaf=2):
                if self.catalog.reconcile(server):
                    break
        else:
            self.fail("Count reconciliation did not converge")
        self.assertEqual(self.catalog.summary()["catalog_records"],6)

    def test_size_accounting_keeps_missing_zero_and_packages_distinct(self):
        server = Server({"main": [document("A1",DocumentSize=None),document("A2",DocumentSize="0"),
                                  document("A3",DocumentSize="bad"),document("A4",IsPackage="Yes"),document("A5")], "legacy": []})
        while self.catalog.step(server):
            pass
        row = self.catalog.summary()["inventory"][0]
        self.assertEqual((row["missing_sizes"],row["zero_sizes"],row["invalid_sizes"],row["packages"]),(1,1,1,1))
        self.assertEqual(row["nonpackage_reported_bytes"],1234)
        self.assertEqual(number(True),(None,"invalid"))
        self.assertEqual(number("9"*30),(None,"invalid"))

    def test_out_of_partition_results_are_rejected(self):
        with self.catalog.db:
            self.catalog.db.execute("UPDATE partitions SET prefix='X' WHERE library='main'")
        class IgnoringFilters:
            def search(self,**kwargs):
                return {"count":1,"results":[{"document":document("A1")}]}
        with self.assertRaisesRegex(RuntimeError,"outside"):
            self.catalog.step(IgnoringFilters())

    def test_distinct_nrc_ids_with_shared_accession_are_preserved(self):
        server = Server({"main":[],"legacy":[document("A1",IdT="ID-1"),document("A1",IdT="ID-2"),document("A1",IdT="ID-3")]})
        while self.catalog.step(server):
            pass
        self.assertTrue(self.catalog.reconcile(server))
        self.assertEqual(self.catalog.summary()["catalog_records"],3)
        self.assertEqual(self.catalog.db.execute('SELECT count(DISTINCT accession) FROM items').fetchone()[0],1)

    def test_duplicate_source_id_does_not_count_as_another_record(self):
        server = Server({"main":[document("A1",IdT="ID-1"),document("A2",IdT="ID-1")],"legacy":[]})
        with self.assertRaisesRegex(RuntimeError,"Duplicate NRC record IDs"):
            self.catalog.step(server)

    def test_mixed_case_accessions_follow_case_sensitive_nrc_order(self):
        # The live ML10004 section places lowercase ml100040062 last.
        source = [document(x) for x in ["ML100040003", "ML100040600", "ml100040062"]]
        server = Server({"main": source, "legacy": []}, page_size=3)
        while self.catalog.step(server):
            pass
        self.assertTrue(self.catalog.reconcile(server))
        self.assertEqual(self.catalog.summary()["catalog_records"], 3)

    def test_mixed_case_page_boundary_preserves_all_record_ids(self):
        source = [document(x, IdT=f"id-{i}") for i, x in enumerate(["ML100040003", "ML100040600", "ml100040003", "ml100040062"])]
        server = Server({"main": source, "legacy": []}, page_size=2)
        while self.catalog.step(server):
            pass
        self.assertTrue(self.catalog.reconcile(server))
        self.assertEqual(self.catalog.summary()["catalog_records"], 4)

    def test_case_variants_do_not_create_overlapping_prefix_partitions(self):
        source = [document(x, IdT=f"id-{i}") for i, x in enumerate(["AA1", "Aa1", "aa2", "AB1", "ab2"])]
        server = Server({"main": source, "legacy": []}, page_size=5)
        for _ in range(100):
            if not self.catalog.step(server, max_leaf=2):
                break
        else:
            self.fail("Mixed-case partition tree did not terminate")
        self.assertTrue(self.catalog.reconcile(server))
        self.assertEqual(self.catalog.summary()["catalog_records"], 5)

    def test_incorrect_case_sensitive_order_remains_blocked(self):
        class Misordered:
            def search(self, **kwargs):
                return {"count": 2, "results": [{"document": document(x)} for x in ["ml100040062", "ML100040600"]]}
        with self.assertRaisesRegex(RuntimeError, "sort"):
            self.catalog.step(Misordered())
        self.assertEqual(self.catalog.summary()["catalog_records"], 0)

    def test_cooldown_wait_honors_stop_without_new_requests(self):
        clock=[0]
        sleeps=[]
        def sleep(seconds):
            sleeps.append(seconds)
            clock[0]+=seconds
            (Path(self.tmp.name)/"STOP").touch()
        self.assertFalse(wait_until(self.catalog,300,clock=lambda:clock[0],sleep=sleep))
        self.assertEqual(sleeps,[20])
        self.assertEqual(json.loads((Path(self.tmp.name)/"status.json").read_text())["worker_state"],"stopped")

    def test_incremental_metrics_match_rows_after_update_delete_and_rollback(self):
        server = Server({"main": [document("A1"), document("A2", DocumentSize=None),
                                 document("A3", DocumentSize="bad"), document("A4", DocumentSize="0"),
                                 document("A5", IsPackage="Yes")], "legacy": []})
        while self.catalog.step(server):
            pass
        def check():
            expressions = ",".join(f"sum({expr.format(row='')}) AS {name}" for name, expr in METRICS.items())
            expected = [dict(row) for row in self.catalog.db.execute(f"SELECT library,{expressions} FROM items GROUP BY library")]
            self.assertEqual(self.catalog.summary()["inventory"], expected)
        check()
        with self.catalog.db:
            self.catalog.db.execute("UPDATE items SET library='legacy',size_bytes=999,size_state='positive',url='https://example.test/a' WHERE accession='A1'")
        check()
        with self.assertRaises(RuntimeError):
            with self.catalog.db:
                self.catalog.db.execute("DELETE FROM items")
                self.assertEqual(self.catalog.summary()["catalog_records"], 0)
                raise RuntimeError("Rollback fixture")
        check()
        with self.catalog.db:
            self.catalog.db.execute("DELETE FROM items WHERE accession='A1'")
        check()

    def test_metrics_migration_backfills_existing_catalog_once(self):
        server = Server({"main": [document("A1"), document("A2")], "legacy": []})
        while self.catalog.step(server):
            pass
        expected = self.catalog.summary()["inventory"]
        with self.catalog.db:
            for event in ("insert", "delete", "update"):
                self.catalog.db.execute(f"DROP TRIGGER inventory_{event}")
            self.catalog.db.execute("DROP TABLE inventory_metrics")
        install_metrics(self.catalog.db)
        self.assertEqual(self.catalog.summary()["inventory"], expected)
        install_metrics(self.catalog.db)
        self.assertEqual(self.catalog.summary()["inventory"], expected)


if __name__ == "__main__":
    unittest.main()
