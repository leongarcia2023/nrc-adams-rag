"""Repeatable, offline question benchmarks with source excerpts for human review.

No answer-quality score is invented: citation validation is not entailment.
The output records answers, inputs, timings, gaps and retrieval coverage.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import time

from .cpu_context import pack_compact
from .ollama_engine import OllamaEngine
from .research import research
from .semantic_index import search_index
from .store import Store


def run(db, directory, questions, output, *, model=None, catalog=None):
    if not isinstance(questions, list) or not 1 <= len(questions) <= 30:
        raise ValueError('Use 1–30 benchmark questions')
    for item in questions:
        if (not isinstance(item, dict) or not isinstance(item.get('question'), str)
                or not 1 <= len(item['question']) <= 2000 or not isinstance(item.get('id'), str)):
            raise ValueError('Each case needs an ID and a bounded question')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    store = Store(db)
    index = search_index(store, directory)
    index.read_only = True
    engine = OllamaEngine(model, cpu_only=True, num_threads=2, timeout=240,
                          storage_path=Path(db).parent, keep_alive=600) if model else None
    report = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'live_discovery': False,
              'model': model, 'request_timeout_seconds': 240 if model else None,
              'cases': [], 'state': 'running', 'human_quality_review_required': True}
    def save():
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        os.replace(temporary, output)
    save()
    try:
        for item in questions:
            started = time.monotonic()
            if engine:
                engine.last_response = None
            try:
                result = research(store, item['question'], engine=engine, semantic=index, catalog=catalog,
                                  live=False, generate=bool(model), interactive=True)
                context = pack_compact(item['question'], result['sources'])[1] if result['sources'] else None
                entry = {'id': item['id'], 'question': item['question'], 'run_id': result['id'],
                         'seconds': round(time.monotonic() - started, 3), 'inquiry_kind': result['inquiry_kind'],
                         'mode': result['mode'], 'answer': result['answer'],
                         'insufficient_evidence': result.get('insufficient_evidence'), 'gaps': result.get('gaps'),
                         'timings': result['timings'], 'generation_metrics': result.get('generation_metrics'),
                         'generation_context': result.get('generation_context'),
                         'unvalidated_model_output': getattr(engine, 'last_response', None),
                         'source_accessions': [s['accession'] for s in result['sources']],
                         'source_input': context, 'warnings': result['warnings']}
            except Exception as error:
                entry = {'id': item['id'], 'question': item['question'], 'mode': 'failed',
                         'seconds': round(time.monotonic() - started, 3),
                         'error': str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__}
            report['cases'].append(entry)
            save()
            print(json.dumps({k: entry.get(k) for k in ('id', 'mode', 'seconds', 'answer', 'gaps', 'error')}), flush=True)
        times = [e['seconds'] for e in report['cases']]
        report.update(state='complete', median_seconds=round(statistics.median(times), 3),
                      maximum_seconds=max(times), generated=sum(e['mode'] == 'generated' for e in report['cases']))
        save()
        return report
    finally:
        store.db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', required=True)
    p.add_argument('--index-dir', required=True)
    p.add_argument('--catalog')
    p.add_argument('--questions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', help='An already-installed local Ollama model; omit for retrieval only')
    args = p.parse_args()
    run(args.db, args.index_dir, json.loads(Path(args.questions).read_text()), args.output,
        model=args.model, catalog=args.catalog)


if __name__ == '__main__':
    main()
