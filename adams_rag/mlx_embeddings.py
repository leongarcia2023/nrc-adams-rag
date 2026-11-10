"""Pinned local BGE profile, isolated from the existing ONNX vector namespace."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from .inference_guard import run_guarded, inference_slot, InferenceBusy
from .storage import require_space

REVISION = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
MODEL = 'BAAI/bge-small-en-v1.5@' + REVISION + ':mlx-f16-cls-v1'
FILES = {
    'config.json':'094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750',
    'model.safetensors':'3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad',
    'tokenizer.json':'d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66',
    'tokenizer_config.json':'9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3',
    'special_tokens_map.json':'b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3',
    'vocab.txt':'07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
    '1_Pooling/config.json':'d1caf60c96f5fba2157c0c26b76d80818fad6cf0b8eb5e73ec372ff9818eba5c',
}


EmbeddingBusy = InferenceBusy


def verify_files(directory):
    for name, expected in FILES.items():
        path=Path(directory)/name
        digest=hashlib.sha256()
        try:
            with path.open('rb') as stream:
                while block:=stream.read(1024*1024):
                    digest.update(block)
        except OSError:
            raise RuntimeError('Pinned local GPU embedding files are unavailable') from None
        if digest.hexdigest()!=expected:
            raise ValueError('GPU embedding file checksum does not match the pinned profile')


def validate_texts(texts):
    if not isinstance(texts,list) or not 1<=len(texts)<=128 or any(not isinstance(t,str) for t in texts):
        raise ValueError('Use 1–128 embedding texts')
    if len(json.dumps(texts,ensure_ascii=False).encode())>1900000:
        raise ValueError('Embedding input exceeds the bounded context allowance')


def decode_vectors(result,count):
    import numpy as np
    if not isinstance(result,dict) or result.get('model')!=MODEL or result.get('count')!=count or result.get('dimension')!=384:
        raise ValueError('Embedding response profile or shape mismatch')
    try:
        raw=base64.b64decode(result['vectors'],validate=True)
    except (KeyError,TypeError,ValueError):
        raise ValueError('Invalid embedding response') from None
    if len(raw)!=count*384*4:
        raise ValueError('Embedding response byte count mismatch')
    vectors=np.frombuffer(raw,dtype='<f4').reshape(count,384)
    if not np.isfinite(vectors).all() or np.max(np.abs(np.linalg.norm(vectors,axis=1)-1))>.001:
        raise ValueError('Embedding response is not finite normalized vectors')
    return vectors


def run_operation(directory,payload,*,cancelled=None,monitor=None):
    directory=Path(directory).resolve()
    locations={directory,Path(tempfile.gettempdir()).resolve()}
    def check(loading=False):
        for location in locations:
            require_space(location,extra_bytes=(2 if loading else 1)*1024**3)
        if monitor:
            monitor()
    check(True)
    lock_path=directory.parent/'models/inference.lock'
    with inference_slot(lock_path,foreground=payload.get('operation') not in {'sync','window_sync'},monitor=check):
        return run_guarded([sys.executable,'-m','adams_rag.embedding_worker'],
            {**payload,'directory':str(directory),'parent_pid':os.getpid(),'model':MODEL},
            preflight=lambda:check(True),monitor=check,timeout=300,cancelled=cancelled)


class GuardedEncoder:
    """Interactive embedding through a short-lived, storage-monitored child."""
    def __init__(self,directory):
        self.directory=Path(directory)

    def passage_embed(self,texts,batch_size=8):
        texts=list(texts)
        validate_texts(texts)
        result=run_operation(self.directory,{'operation':'encode','texts':texts})
        return iter(decode_vectors(result,len(texts)))

    def query_embed(self,text):
        return self.passage_embed([text])


class DirectEncoder:
    """GPU implementation for the supervised child only; no remote code or fetches."""
    def __init__(self,directory):
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        folder=Path(directory)/'mlx-bge'
        verify_files(folder)
        import mlx.core as mx
        from transformers import AutoTokenizer
        from types import SimpleNamespace
        from ._vendor.apple_bert import Bert
        config=json.loads((folder/'config.json').read_text())
        self.tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(256*1024**2)
        self.model=Bert(SimpleNamespace(**config))
        weights=mx.load(str(folder/'model.safetensors'))
        positions=weights.pop('embeddings.position_ids',None)
        if positions is not None and (positions.shape!=(1,512) or not mx.array_equal(positions,mx.arange(512)[None,:]).item()):
            raise ValueError('Unexpected BGE position buffer')
        def replace_key(key):
            # Apple's MIT-licensed BERT conversion mapping; see _vendor provenance.
            for old,new in [('.layer.','.layers.'),('.self.key.','.key_proj.'),('.self.query.','.query_proj.'),
                ('.self.value.','.value_proj.'),('.attention.output.dense.','.attention.out_proj.'),
                ('.attention.output.LayerNorm.','.ln1.'),('.output.LayerNorm.','.ln2.'),
                ('.intermediate.dense.','.linear1.'),('.output.dense.','.linear2.'),
                ('.LayerNorm.','.norm.'),('pooler.dense.','pooler.')]:
                key=key.replace(old,new)
            return key
        self.model.load_weights([(replace_key(k),v) for k,v in weights.items()],strict=True)
        del weights
        self.model.set_dtype(mx.float16)
        self.model.eval()
        mx.eval(self.model.parameters())

    def passage_embed(self,texts,batch_size=8):
        import mlx.core as mx
        import numpy as np
        texts=list(texts)
        validate_texts(texts)
        # Keep the measured eight-item GPU working set, regardless of caller size.
        for start in range(0,len(texts),8):
            batch=self.tokenizer(texts[start:start+8],return_tensors='np',padding=True,truncation=True,max_length=512)
            inputs={k:mx.array(v) for k,v in batch.items()}
            inputs['attention_mask']=inputs['attention_mask'].astype(mx.float16)
            hidden,_=self.model(**inputs)
            cls=hidden[:,0,:].astype(mx.float32)
            vectors=cls/mx.maximum(mx.linalg.norm(cls,axis=-1,keepdims=True),1e-12)
            mx.eval(vectors)
            yield from np.array(vectors,dtype=np.float32)

    def query_embed(self,text):
        return self.passage_embed([text])
