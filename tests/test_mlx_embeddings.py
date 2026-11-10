import base64
import contextlib
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adams_rag import mlx_embeddings as gpu
from adams_rag.embedding_worker import complete
from adams_rag.semantic_index import SemanticIndex,DEFAULT_EMBEDDING,active_model
from adams_rag.storage import StoragePause
from adams_rag.store import Store


class GPUEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.directory=self.root/'index'
        self.directory.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def encoded(self,count=1):
        rows=np.zeros((count,384),dtype='<f4');rows[:,0]=1
        return {'model':gpu.MODEL,'dimension':384,'count':count,'vectors':base64.b64encode(rows.tobytes()).decode()}

    def test_response_profile_shape_and_normalization_are_verified(self):
        result=self.encoded()
        self.assertEqual(gpu.decode_vectors(result,1).shape,(1,384))
        for changes in ({'model':DEFAULT_EMBEDDING},{'count':2},{'dimension':383},
                        {'vectors':'invalid!'},{'vectors':base64.b64encode(bytes(384*4)).decode()}):
            with self.assertRaises(ValueError):
                gpu.decode_vectors({**result,**changes},1)
        values=np.full((1,384),np.nan,dtype='<f4')
        with self.assertRaises(ValueError):
            gpu.decode_vectors({**result,'vectors':base64.b64encode(values.tobytes()).decode()},1)

    def test_input_and_child_operation_limits_reject_before_loading_model(self):
        for texts in ([],['x']*129,[None],['x'*1900001]):
            with self.assertRaises(ValueError):
                gpu.validate_texts(texts)
        with patch('adams_rag.embedding_worker.DirectEncoder') as encoder:
            for payload in ({'model':'wrong','operation':'encode'},
                            {'model':gpu.MODEL,'operation':'encode','directory':str(self.directory),'texts':[]},
                            {'model':gpu.MODEL,'operation':'sync','directory':str(self.directory),'limit':1,'priority_model':gpu.MODEL},
                            {'model':gpu.MODEL,'operation':'sync','directory':str(self.directory),'limit':True}):
                with self.assertRaises(ValueError):
                    complete(payload)
            (self.directory/'STOP').touch()
            with self.assertRaises(InterruptedError):
                complete({'model':gpu.MODEL,'operation':'sync','directory':str(self.directory),'limit':1})
        encoder.assert_not_called()

    def test_storage_pause_prevents_native_child_start(self):
        with patch('adams_rag.mlx_embeddings.require_space',side_effect=StoragePause('low')),patch('adams_rag.mlx_embeddings.run_guarded') as child:
            with self.assertRaises(StoragePause):
                gpu.run_operation(self.directory,{'operation':'encode','texts':['source']})
        child.assert_not_called()

    def test_window_jobs_are_background_and_invalid_limits_do_not_load_model(self):
        with patch('adams_rag.embedding_worker.DirectEncoder') as encoder:
            base = {'model':gpu.MODEL,'operation':'window_sync','directory':str(self.directory),'limit':1,'vector_limit':8}
            for changes in ({'limit':True},{'limit':257},{'vector_limit':0},{'vector_limit':129},{'priority_model':DEFAULT_EMBEDDING}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    complete({**base,**changes})
            state = self.directory/'windows';state.mkdir();(state/'STOP').touch()
            with self.assertRaises(InterruptedError):
                complete(base)
        encoder.assert_not_called()
        flags=[]
        @contextlib.contextmanager
        def slot(path,**kwargs):
            flags.append(kwargs['foreground'])
            yield
        with patch('adams_rag.mlx_embeddings.require_space'),patch('adams_rag.mlx_embeddings.inference_slot',side_effect=slot),patch('adams_rag.mlx_embeddings.run_guarded',return_value={'embedded':0}):
            gpu.run_operation(self.directory,{'operation':'window_sync','db':'unused','limit':1,'vector_limit':8})
        self.assertEqual(flags,[False])

    def test_shared_generation_lock_prevents_overlapping_gpu_work(self):
        folder=self.root/'models';folder.mkdir()
        with (folder/'inference.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch('adams_rag.mlx_embeddings.require_space'),patch('adams_rag.mlx_embeddings.run_guarded') as child:
                with self.assertRaises(gpu.EmbeddingBusy):
                    gpu.run_operation(self.directory,{'operation':'sync','db':'unused','limit':1})
                child.assert_not_called()

    def test_guarded_query_has_pinned_profile_and_storage_monitor(self):
        def child(command,payload,**kwargs):
            self.assertEqual(command[-1],'adams_rag.embedding_worker')
            self.assertEqual(payload['model'],gpu.MODEL)
            self.assertEqual(payload['texts'],['query'])
            kwargs['preflight']();kwargs['monitor']()
            return self.encoded()
        with patch('adams_rag.mlx_embeddings.require_space') as space,patch('adams_rag.mlx_embeddings.run_guarded',side_effect=child):
            result=list(gpu.GuardedEncoder(self.directory).query_embed('query'))
        self.assertEqual(len(result),1)
        self.assertEqual({call.kwargs['extra_bytes'] for call in space.call_args_list},{1024**3,2*1024**3})

    def test_checksum_mismatch_and_missing_assets_fail_closed(self):
        with patch.dict(gpu.FILES,{'fixture':'0'*64},clear=True):
            with self.assertRaises(RuntimeError):
                gpu.verify_files(self.directory)
            (self.directory/'fixture').write_bytes(b'wrong')
            with self.assertRaises(ValueError):
                gpu.verify_files(self.directory)

    def test_profile_activation_is_explicit_and_rejects_invalid_configuration(self):
        self.assertEqual(active_model(self.directory),DEFAULT_EMBEDDING)
        path=self.directory/'ACTIVE_MODEL.json'
        path.write_text(json.dumps({'version':1,'model':gpu.MODEL}))
        self.assertEqual(active_model(self.directory),gpu.MODEL)
        for value in ({'version':1,'model':'unknown'},{'version':True,'model':gpu.MODEL},
                      {'version':1,'model':gpu.MODEL,'extra':True},[]):
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                active_model(self.directory)

    def test_new_profile_keeps_existing_vectors_and_graph_separate(self):
        store=Store(self.root/'db.sqlite')
        class Encoder:
            def passage_embed(self,texts,**kwargs):
                return [np.asarray([1.,0.],dtype=np.float32) for _ in texts]
        try:
            with store.db:
                store.put({'AccessionNumber':'A','content':'Evidence'})
            cpu=SemanticIndex(store,self.directory,encoder=Encoder())
            candidate=SemanticIndex(store,self.directory,model=gpu.MODEL,encoder=Encoder())
            cpu.sync();candidate.sync()
            self.assertEqual(cpu.status()['vectors'],1)
            self.assertEqual(candidate.status()['vectors'],1)
            self.assertNotEqual(cpu.path,candidate.path)
            self.assertEqual(len(store.db.execute('SELECT * FROM dense_vectors').fetchall()),2)
            reopened=SemanticIndex(store,self.directory,model=gpu.MODEL)
            self.assertIsInstance(reopened._encoder(),gpu.GuardedEncoder)
            self.assertEqual(SemanticIndex(store,self.directory).model,DEFAULT_EMBEDDING)
        finally:
            store.db.close()


if __name__=='__main__':
    unittest.main()
