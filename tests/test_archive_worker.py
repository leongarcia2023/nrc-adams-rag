import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zlib

from adams_rag.archive_worker import main, resume_at
from adams_rag.client import RateGate
from adams_rag.store import Store


class ArchiveWorkerTests(unittest.TestCase):
    def test_both_libraries_alternate_batches_and_keep_shared_rate_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = sqlite3.connect(root / "catalog.sqlite")
            catalog.executescript("CREATE TABLE libraries(library,final_count,finished,state);"
                "INSERT INTO libraries VALUES('main',2,'2026-10-31','complete');"
                "INSERT INTO libraries VALUES('legacy',2,'2026-10-31','complete');"
                "CREATE TABLE items(library,accession,is_package,metadata);")
            documents = {}
            for library, accessions in [("main", ["ML24017A120", "ML24017A121"]),
                                       ("legacy", ["9803170214", "7809190190"])]:
                for accession in accessions:
                    doc = {"AccessionNumber": accession, "IdT": accession, "IsLegacy": "Yes" if library == "legacy" else "No"}
                    documents[accession] = {**doc, "content": "Valve evidence."}
                    catalog.execute("INSERT INTO items VALUES(?,?,0,?)", (library, accession, zlib.compress(json.dumps(doc).encode())))
            catalog.commit(); catalog.close()
            calls, settings = [], []
            class Client:
                def __init__(self, rate, key, **kwargs):
                    settings.append((str(rate.path), kwargs["rps"], kwargs["daily_limit"]))
                def request(self, accession):
                    calls.append(accession)
                    return {"document": documents[accession]}
            args = ["--db", str(root / "research.sqlite"), "--catalog", str(root / "catalog.sqlite"),
                    "--rate-db", str(root / "rate.sqlite"), "--directory", str(root / "archive"),
                    "--library", "both", "--batch-size", "1", "--min-free-gb", "0.5"]
            with patch("adams_rag.archive_worker.AdamsClient", Client), \
                 patch("adams_rag.archive_worker.load_api_key", return_value="test-key"), patch("builtins.print"):
                main(args)
            self.assertEqual(calls, ["ML24017A120", "9803170214", "ML24017A121", "7809190190"])
            self.assertEqual(set(settings), {(str(root / "rate.sqlite"), 2, 90000)})
            status = json.loads((root / "archive/STATUS.json").read_text())
            self.assertEqual((status["state"], status["new_documents_this_run"]), ("scan_finished", 4))
            self.assertFalse(status["full_archive_verified"])

    def test_daily_and_server_cooldowns_survive_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            rate = Store(Path(temp) / "rate.sqlite")
            gate = RateGate(rate, "test-key", daily_limit=2, clock=lambda: 1000)
            with rate.db:
                rate.db.executemany("INSERT INTO http_requests VALUES(?,?)", [(gate.scope, 900), (gate.scope, 950)])
            self.assertEqual(resume_at(rate, gate.scope, 2, now=1000), 87301)
            gate.defer(100000)
            self.assertEqual(resume_at(rate, gate.scope, 2, now=1000), 101000)
            rate.db.close()

    def test_continuous_scan_finishes_with_honest_coverage_and_durable_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = sqlite3.connect(root / "catalog.sqlite")
            catalog.executescript("CREATE TABLE libraries(library,final_count,finished,state);"
                "INSERT INTO libraries VALUES('main',1,'2026-10-31','complete');"
                "CREATE TABLE items(library,accession,is_package,metadata);")
            catalog.execute("INSERT INTO items VALUES('main','ML24017A120',0,?)",
                            (zlib.compress(json.dumps({"AccessionNumber": "ML24017A120"}).encode()),))
            catalog.commit(); catalog.close()
            class Client:
                def __init__(self, *args, **kwargs):
                    pass
                def request(self, accession):
                    return {"document": {"AccessionNumber": accession, "content": "Document body about valves."}}
            args = ["--db", str(root / "research.sqlite"), "--catalog", str(root / "catalog.sqlite"),
                    "--rate-db", str(root / "rate.sqlite"), "--directory", str(root / "archive"),
                    "--library", "main", "--batch-size", "1", "--min-free-gb", "0.5"]
            with patch("adams_rag.archive_worker.AdamsClient", Client), \
                 patch("adams_rag.archive_worker.load_api_key", return_value="test-key"), patch("builtins.print"):
                main(args)
            status = json.loads((root / "archive/STATUS.json").read_text())
            self.assertEqual(status["state"], "scan_finished")
            self.assertEqual(status["new_documents_this_run"], 1)
            self.assertEqual(status["reserve_bytes"], 500_000_000)
            self.assertFalse(status["full_archive_verified"])
            self.assertEqual(status["coverage"], [{"library": "main", "outcome": "indexed", "count": 1}])


if __name__ == "__main__":
    unittest.main()
