"""Transactional inventory totals, avoiding repeated scans of all metadata."""

METRICS = {
    "records": "1",
    "packages": "{row}is_package",
    "missing_sizes": "CASE WHEN {row}size_state='missing' THEN 1 ELSE 0 END",
    "invalid_sizes": "CASE WHEN {row}size_state='invalid' THEN 1 ELSE 0 END",
    "zero_sizes": "CASE WHEN {row}size_state='zero' THEN 1 ELSE 0 END",
    "nonpackage_reported_bytes": "CASE WHEN {row}is_package=0 AND {row}size_bytes>0 THEN {row}size_bytes ELSE 0 END",
    "linked_nonpackage_reported_bytes": "CASE WHEN {row}is_package=0 AND {row}size_bytes>0 AND ({row}url LIKE 'https://%' OR {row}url LIKE 'http://%') THEN {row}size_bytes ELSE 0 END",
}


def install_metrics(db):
    # Creation, backfill, and triggers commit together. A failed migration can
    # safely restart; subsequent row changes and totals share one transaction.
    with db:
        db.execute("BEGIN IMMEDIATE")
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='inventory_metrics'").fetchone()
        if not exists:
            columns = ",".join(f"{name} INTEGER NOT NULL" for name in METRICS)
            db.execute(f"CREATE TABLE inventory_metrics(library TEXT PRIMARY KEY,{columns})")
            aggregates = ",".join(f"sum({expr.format(row='')})" for expr in METRICS.values())
            db.execute(f"INSERT INTO inventory_metrics SELECT library,{aggregates} FROM items NOT INDEXED GROUP BY library")

        def adjust(row, sign):
            values = ",".join(f"{sign} * ({expr.format(row=row+'.')})" for expr in METRICS.values())
            updates = ",".join(f"{name}={name}+excluded.{name}" for name in METRICS)
            return f"INSERT INTO inventory_metrics VALUES({row}.library,{values}) ON CONFLICT(library) DO UPDATE SET {updates};"

        for event, body in (("insert", adjust("NEW", 1)),
                            ("delete", adjust("OLD", -1)),
                            ("update", adjust("OLD", -1) + adjust("NEW", 1))):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS inventory_{event} AFTER {event.upper()} ON items BEGIN {body} END")
