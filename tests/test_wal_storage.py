"""Exercise journal reclamation while a reader still needs its old snapshot."""
from pathlib import Path
import tempfile
import unittest

from adams_rag.store import Store


class WALStorageTests(unittest.TestCase):
    def test_large_write_survives_pinned_reader_then_releases_reusable_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "research.sqlite"
            writer, reader = Store(path), Store(path)
            try:
                for store in (writer, reader):
                    self.assertEqual(store.db.execute("PRAGMA journal_size_limit").fetchone()[0], 64 * 1024**2)
                writer.db.execute("PRAGMA busy_timeout=0")
                with writer.db:
                    writer.db.execute("CREATE TABLE wal_fixture(id INTEGER PRIMARY KEY, payload BLOB)")
                    writer.db.execute("INSERT INTO wal_fixture VALUES(1,?)", (b"original source",))
                reader.db.execute("BEGIN")
                self.assertEqual(reader.db.execute("SELECT payload FROM wal_fixture WHERE id=1").fetchone()[0], b"original source")
                with writer.db:
                    writer.db.execute("INSERT INTO wal_fixture VALUES(2,zeroblob(?))", (70 * 1024**2,))
                wal = Path(str(path) + "-wal")
                self.assertGreater(wal.stat().st_size, 64 * 1024**2)
                self.assertEqual(writer.db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0], 1)
                self.assertEqual(reader.db.execute("SELECT count(*) FROM wal_fixture").fetchone()[0], 1)
                self.assertGreater(wal.stat().st_size, 64 * 1024**2)
                reader.db.rollback()
                result = writer.db.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                self.assertEqual((result[0], result[1]), (0, result[2]))
                with writer.db:
                    writer.db.execute("INSERT INTO wal_fixture VALUES(3,?)", (b"next committed source",))
                self.assertLessEqual(wal.stat().st_size, 64 * 1024**2)
                self.assertEqual(reader.db.execute("SELECT count(*) FROM wal_fixture").fetchone()[0], 3)
            finally:
                reader.db.close()
                writer.db.close()
            reopened = Store(path)
            try:
                self.assertEqual([tuple(row) for row in reopened.db.execute("SELECT id,length(payload) FROM wal_fixture ORDER BY id")],
                                 [(1, 15), (2, 70 * 1024**2), (3, 21)])
                self.assertEqual(reopened.db.execute("PRAGMA quick_check").fetchone()[0], "ok")
            finally:
                reopened.db.close()


if __name__ == "__main__":
    unittest.main()
