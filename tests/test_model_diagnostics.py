import io
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from adams_rag import model_worker
from adams_rag.inference_guard import run_guarded
from adams_rag.model_diagnostics import ModelWorkerFailure, safe_diagnostics


class ModelDiagnosticsTests(unittest.TestCase):
    def payload(self):
        return {'parent_pid': os.getppid(), 'model_path': 'unused', 'max_tokens': 10,
                'system': 'private system text', 'data': {'source': 'private source text'}}

    def modules(self, segments, *, load_error=None):
        tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value='private prompt'))
        loader = Mock(return_value=(object(), tokenizer), side_effect=load_error)
        streamer = Mock(return_value=iter(segments))
        core = SimpleNamespace(set_cache_limit=Mock())
        return {'mlx_lm': SimpleNamespace(load=loader, stream_generate=streamer),
                'mlx_lm.sample_utils': SimpleNamespace(make_sampler=Mock(return_value=None)),
                'mlx': SimpleNamespace(core=core), 'mlx.core': core}

    def segment(self, text, reason=None, tokens=1):
        return SimpleNamespace(text=text, finish_reason=reason,
                               prompt_tokens=200, generation_tokens=tokens)

    def run_worker(self, modules):
        payload = self.payload()
        stdin = SimpleNamespace(buffer=io.BytesIO(json.dumps(payload).encode()))
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, modules), patch.object(model_worker, 'model_ready', return_value=True), \
             patch.object(model_worker.threading, 'Thread'), patch.object(sys, 'stdin', stdin), \
             patch.object(sys, 'stdout', stdout), patch.object(sys, 'stderr', stderr):
            model_worker.main()
        response = json.loads(stdout.getvalue())
        self.assertNotIn('private', stdout.getvalue())
        return response

    def test_stream_accumulation_keeps_structured_result_unchanged(self):
        segments = [self.segment('{"paragraphs":['), self.segment('],"gaps":[]}', 'stop', 4)]
        response = self.run_worker(self.modules(segments))
        self.assertEqual(response, {'ok': True, 'result': {'paragraphs': [], 'gaps': []}})

    def test_invalid_json_at_limit_reports_length_without_partial_answer(self):
        response = self.run_worker(self.modules([self.segment('{"private', 'length', 10)]))
        self.assertIs(response['ok'], False)
        self.assertNotIn('result', response)
        self.assertEqual(response['diagnostics']['stage'], 'decode')
        self.assertEqual(response['diagnostics']['finish_reason'], 'length')
        self.assertEqual(response['diagnostics']['generation_tokens'], 10)
        self.assertEqual(response['diagnostics']['text_chars'], 9)

    def test_malformed_output_after_stop_is_not_called_a_token_limit(self):
        response = self.run_worker(self.modules([self.segment('private invalid answer', 'stop', 3)]))
        error = ModelWorkerFailure(response['diagnostics'])
        self.assertIn('invalid structured output', str(error))
        self.assertNotIn('token limit', str(error))
        self.assertNotIn('private', str(error))

    def test_complete_json_at_token_limit_is_still_decoded(self):
        response = self.run_worker(self.modules([self.segment('{"value":1}', 'length', 10)]))
        self.assertEqual(response, {'ok': True, 'result': {'value': 1}})

    def test_dependency_and_load_failures_are_distinct_and_source_free(self):
        cases = [({'mlx_lm': None}, 'dependencies'),
                 (self.modules([], load_error=RuntimeError('private path/source')), 'load')]
        for modules, expected in cases:
            with self.subTest(stage=expected):
                response = self.run_worker(modules)
                self.assertIs(response['ok'], False)
                self.assertEqual(response['diagnostics']['stage'], expected)

    def test_generator_error_after_partial_text_is_not_decode_failure(self):
        def broken():
            yield self.segment('private partial text')
            raise RuntimeError('private runtime details')
        response = self.run_worker(self.modules(broken()))
        self.assertEqual(response['diagnostics']['stage'], 'generation')
        self.assertEqual(response['diagnostics']['generation_tokens'], 1)
        self.assertNotIn('finish_reason', response['diagnostics'])
        self.assertNotIn('result', response)

    def test_empty_stream_cannot_become_answer_or_inferred_token_limit(self):
        response = self.run_worker(self.modules([]))
        self.assertEqual(response['diagnostics']['stage'], 'decode')
        self.assertNotIn('finish_reason', response['diagnostics'])

    def test_whitelist_drops_unknown_text_invalid_types_and_unbounded_counts(self):
        dirty = {'stage': 'private', 'finish_reason': ['length'], 'text_chars': True,
                 'generation_tokens': 5000, 'prompt_tokens': -1, 'elapsed_ms': float('nan'),
                 'max_tokens': 1400, 'error': 'private', 'text': 'private', 'path': 'private'}
        self.assertEqual(safe_diagnostics(dirty), {'max_tokens': 1400})
        self.assertEqual(safe_diagnostics(None), {})
        self.assertEqual(safe_diagnostics({'stage': []}), {})

    def test_real_supervisor_preserves_safe_failure_metadata_and_withholds_result(self):
        response = {'ok': False, 'result': {'answer': 'private'}, 'diagnostics':
                    {'stage': 'decode', 'finish_reason': 'length', 'generation_tokens': 1400,
                     'text': 'private', 'error': 'private'}}
        command = [sys.executable, '-c', 'import json,sys; print(json.dumps(json.load(sys.stdin)))']
        with self.assertRaises(ModelWorkerFailure) as caught:
            run_guarded(command, response, preflight=lambda: None, monitor=lambda: None)
        self.assertEqual(caught.exception.diagnostics,
                         {'stage': 'decode', 'finish_reason': 'length', 'generation_tokens': 1400})
        self.assertIn('token limit', str(caught.exception))
        self.assertNotIn('private', str(caught.exception))

    def test_older_embedding_failure_envelope_remains_supported(self):
        error = ModelWorkerFailure(None)
        self.assertEqual(error.diagnostics, {})
        self.assertEqual(str(error), 'Local model did not produce a valid result; retrieved evidence is retained')


if __name__ == '__main__':
    unittest.main()
