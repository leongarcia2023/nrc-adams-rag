"""Measure retained main-library text behind explicit legacy catalog references."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sqlite3
import time

from .catalog_lookup import connect
from .identity import is_legacy, record_id
from .legacy_links import default_path, snapshot


def measure(research_db, catalog, links=None, *, cancelled=None, progress=None):
    """Read-only sources, bounded Python batches, and a temporary in-memory ID set.

    A reference is not identity equivalence. No document body is downloaded or
    declared complete, and duplicate legacy records are never merged here.
    """
    catalog, links = Path(catalog).resolve(), Path(links or default_path(catalog)).resolve()
    started = time.monotonic()
    source = connect(catalog)
    db = None
    interrupted = [False]
    try:
        db = connect(research_db)
        source.execute('BEGIN')
        expected = snapshot(source, catalog)
        libraries = [dict(r) for r in source.execute('SELECT library,final_count,finished,state FROM libraries ORDER BY library')]
        db.execute('PRAGMA temp_store=MEMORY')
        db.execute('ATTACH DATABASE ? AS catalog_observation', (catalog.as_uri() + '?mode=ro',))
        db.execute('ATTACH DATABASE ? AS reference_map', (links.as_uri() + '?mode=ro',))
        db.execute('BEGIN')
        state = db.execute('SELECT * FROM reference_map.link_state WHERE id=1').fetchone()
        if not state or state['state'] != 'complete' or state['snapshot'] != expected:
            raise ValueError('Use the complete legacy reference map for this exact reconciled catalog')
        db.execute('CREATE TEMP TABLE available_main(main_row INTEGER PRIMARY KEY)')
        def guard():
            if cancelled and cancelled():
                interrupted[0] = True
                return 1
            return 0
        db.set_progress_handler(guard, 10000)
        candidates, unidentified, eligible = 0, 0, 0
        cursor = db.execute("SELECT accession,metadata FROM documents WHERE status='indexed' AND accession GLOB 'ML*' ORDER BY accession")
        while rows := cursor.fetchmany(1000):
            if guard():
                raise InterruptedError('Legacy text-coverage assessment cancelled')
            keys = []
            for row in rows:
                candidates += 1
                metadata = json.loads(row['metadata'])
                identity = record_id(metadata)
                if is_legacy(metadata) or not identity.startswith('id:'):
                    unidentified += 1
                    continue
                keys.append((identity, row['accession']))
            eligible += len(keys)
            db.executemany('''INSERT OR IGNORE INTO available_main
                SELECT rowid FROM catalog_observation.items
                WHERE library='main' AND record_id=? AND accession=? COLLATE NOCASE AND is_package=0''', keys)
            if progress:
                progress({'phase': 'matching_current_text', 'local_candidates': candidates})
        available = db.execute('SELECT count(*) FROM available_main').fetchone()[0]
        accessions, references, any_text, all_text, present_links = 0, 0, 0, 0, 0
        for row in db.execute('''SELECT l.legacy_accession,count(*) AS known_references,
                sum(a.main_row IS NOT NULL) AS retained_references
                FROM reference_map.legacy_links l LEFT JOIN available_main a ON a.main_row=l.main_row
                GROUP BY l.legacy_accession ORDER BY l.legacy_accession'''):
            accessions += 1
            references += row['known_references']
            present_links += row['retained_references']
            any_text += row['retained_references'] > 0
            all_text += row['retained_references'] == row['known_references']
            if progress and accessions % 100000 == 0:
                progress({'phase': 'counting_referenced_text', 'referenced_accessions': accessions})
        if references != state['links']:
            raise ValueError('Reference-map row count does not match its completed state')
        main_with_references = db.execute('''SELECT count(DISTINCT l.main_row)
            FROM reference_map.legacy_links l JOIN available_main a ON a.main_row=l.main_row''').fetchone()[0]
        generation = db.execute('SELECT generation FROM retrieval_state WHERE id=1').fetchone()[0]
        return {'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'elapsed_seconds': round(time.monotonic()-started, 2),
                'scope': 'base_catalog_explicit_nudocs_references',
                'catalog_snapshot': expected, 'catalog_libraries': libraries,
                'retrieval_generation': generation,
                'indexed_main_candidates': candidates, 'candidates_without_main_nrc_id': unidentified,
                'identity_matched_main_records_with_text': available,
                'identified_candidates_not_matched_to_base_catalog': eligible-available,
                'reference_relationships': references,
                'referenced_legacy_accessions': accessions,
                'reference_relationships_with_retained_main_text': present_links,
                'referenced_main_records_with_text': main_with_references,
                'legacy_accessions_with_any_referenced_main_text': any_text,
                'legacy_accessions_with_all_referenced_main_text': all_text,
                'legacy_accessions_without_referenced_main_text': accessions-any_text,
                'identity_equivalence_verified': False, 'source_completeness_verified': False,
                'full_archive_verified': False,
                'limitations': ['Accessions count reference labels, not distinct legacy NRC records.',
                                'All referenced main texts present does not establish complete legacy-document content.',
                                'Text may still contain extraction errors or have excluded pages.',
                                'Unlinked legacy records and later catalog supplements are outside this measurement.']}
    except sqlite3.OperationalError:
        if interrupted[0]:
            raise InterruptedError('Legacy text-coverage assessment cancelled') from None
        raise
    finally:
        if db is not None:
            db.set_progress_handler(None, 0)
            db.close()
        source.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='data/research.sqlite')
    parser.add_argument('--catalog', default='../adams-catalog/catalog.sqlite')
    parser.add_argument('--links')
    args = parser.parse_args(argv)
    print(json.dumps(measure(args.db, args.catalog, args.links), indent=2))


if __name__ == '__main__':
    main()
