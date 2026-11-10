"""Supervised GPU embedding. Native model memory is released after every job."""
import base64
import contextlib
import json
import os
from pathlib import Path
import sys
import threading

from .mlx_embeddings import MODEL,DirectEncoder,validate_texts
from .model_worker import watch_parent
from .storage import require_space


def complete(payload):
    if payload.get('model')!=MODEL or payload.get('operation') not in {'encode','sync','window_sync'}:
        raise ValueError('Unsupported embedding operation')
    directory=Path(payload['directory'])
    if payload['operation']=='encode':
        validate_texts(payload.get('texts'))
    elif payload['operation']=='sync':
        if type(payload.get('limit')) is not int or not 1<=payload['limit']<=2000:
            raise ValueError('Use 1–2,000 passages per supervised indexing job')
        from .semantic_index import DEFAULT_EMBEDDING
        if payload.get('priority_model') not in (None,DEFAULT_EMBEDDING):
            raise ValueError('Unsupported priority embedding profile')
        if (directory/'STOP').exists():
            raise InterruptedError('Indexing stopped')
    else:
        if type(payload.get('limit')) is not int or not 1<=payload['limit']<=256:
            raise ValueError('Use 1–256 passages per window job')
        if type(payload.get('vector_limit')) is not int or not 1<=payload['vector_limit']<=128:
            raise ValueError('Use 1–128 new vectors per window job')
        if payload.get('priority_model') is not None:
            raise ValueError('Window jobs use their pinned embedding profile')
        state_directory=Path(payload.get('state_directory',directory/'windows'))
        if (directory/'STOP').exists() or (state_directory/'STOP').exists():
            raise InterruptedError('Window indexing stopped')
    require_space(directory,extra_bytes=2*1024**3)
    encoder=DirectEncoder(directory)
    if payload['operation']=='encode':
        import numpy as np
        import mlx.core as mx
        vectors=np.asarray(list(encoder.passage_embed(payload['texts'])),dtype='<f4')
        return {'model':MODEL,'count':len(vectors),'dimension':384,'vectors':base64.b64encode(vectors.tobytes()).decode(),
                'peak_gpu_bytes':mx.get_peak_memory()}
    from .semantic_index import SemanticIndex
    from .store import Store
    from .inference_guard import interactive_waiting
    store=Store(payload['db'])
    def progress(*_):
        if (directory/'STOP').exists() or (payload['operation']=='window_sync' and (state_directory/'STOP').exists()):
            raise InterruptedError('Indexing stopped; committed vectors retained')
        require_space(directory,extra_bytes=1024**3)
    try:
        import mlx.core as mx
        if payload['operation']=='window_sync':
            from .embedding_windows import WindowPlanner
            from .window_backfill import WindowBackfill
            result=WindowBackfill(store,WindowPlanner(directory),encoder).sync(
                limit=payload['limit'],vector_limit=payload['vector_limit'],progress=progress,
                should_yield=lambda:interactive_waiting(directory.parent/'models/inference.lock'))
            return {**result,'peak_gpu_bytes':mx.get_peak_memory()}
        result=SemanticIndex(store,directory,model=MODEL,encoder=encoder).sync(
            payload['limit'],progress=progress,priority_model=payload.get('priority_model'),
            should_yield=lambda:interactive_waiting(directory.parent/'models/inference.lock'))
        return {**result,'peak_gpu_bytes':mx.get_peak_memory()}
    finally:
        store.db.close()


def main():
    try:
        raw=sys.stdin.buffer.read(2*1024*1024+1)
        if len(raw)>2*1024*1024:
            raise ValueError('Embedding input allowance exceeded')
        payload=json.loads(raw)
        if type(payload.get('parent_pid')) is not int or payload['parent_pid']!=os.getppid():
            raise ValueError('Supervisor is unavailable')
        threading.Thread(target=watch_parent,args=(payload['parent_pid'],),daemon=True).start()
        with contextlib.redirect_stdout(sys.stderr):
            result=complete(payload)
        print(json.dumps({'ok':True,'result':result}),flush=True)
    except Exception:
        print(json.dumps({'ok':False}),flush=True)


if __name__=='__main__':
    main()
