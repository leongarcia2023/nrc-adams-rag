"""Single local-model completion. The parent owns storage checks and termination."""
import contextlib
import json
import os
import sys
import threading
import time

from .models import decode_object, model_ready
from .model_diagnostics import safe_diagnostics


def watch_parent(parent_pid):
    while os.getppid() == parent_pid:
        time.sleep(.25)
    # The supervising process disappeared; release model memory even on abrupt exit.
    os._exit(70)


def complete(payload, diagnostics=None):
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics['stage'] = 'model_check'
    model_path = payload["model_path"]
    if not model_ready(model_path):
        raise ValueError("Model files are unavailable")
    diagnostics['stage'] = 'dependencies'
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    diagnostics['stage'] = 'load'
    mx.set_cache_limit(256 * 1024 * 1024)
    model, tokenizer = load(model_path)
    diagnostics['stage'] = 'prompt'
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": payload["system"]},
        {"role": "user", "content": json.dumps(payload["data"], ensure_ascii=False)},
    ], tokenize=False, add_generation_prompt=True)
    diagnostics['stage'] = 'generation'
    diagnostics['max_tokens'] = payload['max_tokens']
    # This is the same accumulation used by mlx_lm.generate, retaining only its
    # counters and terminal reason in addition to the text needed for decoding.
    text = ''
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=payload['max_tokens'],
                                    sampler=make_sampler(temp=0)):
        text += response.text
        diagnostics.update(prompt_tokens=int(response.prompt_tokens),
                           generation_tokens=int(response.generation_tokens),
                           text_chars=len(text))
        if response.finish_reason in {'stop', 'length'}:
            diagnostics['finish_reason'] = response.finish_reason
    diagnostics['stage'] = 'decode'
    return decode_object(text)


def main():
    diagnostics = {'stage': 'input'}
    started = time.monotonic()
    try:
        raw = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Input allowance exceeded")
        payload = json.loads(raw)
        if type(payload.get("parent_pid")) is not int or payload["parent_pid"] != os.getppid():
            raise ValueError("Supervisor is unavailable")
        if type(payload.get("max_tokens")) is not int or not 1 <= payload["max_tokens"] <= 4096:
            raise ValueError("Invalid token allowance")
        threading.Thread(target=watch_parent, args=(payload["parent_pid"],), daemon=True).start()
        with contextlib.redirect_stdout(sys.stderr):
            result = complete(payload, diagnostics)
        print(json.dumps({"ok": True, "result": result}), flush=True)
    except Exception:
        # Dependency errors can contain paths or source data. Return no arbitrary body.
        diagnostics['elapsed_ms'] = int((time.monotonic() - started) * 1000)
        print(json.dumps({"ok": False, "diagnostics": safe_diagnostics(diagnostics)}), flush=True)


if __name__ == "__main__":
    main()
