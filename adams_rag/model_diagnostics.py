"""Bounded model failure metadata; never retain prompts, output, or error bodies."""

STAGES = frozenset({'input', 'model_check', 'dependencies', 'load', 'prompt', 'generation', 'decode'})
COUNTER_LIMITS = {'max_tokens': 4096, 'prompt_tokens': 2_000_000,
                  'generation_tokens': 4096, 'text_chars': 1_048_576,
                  'elapsed_ms': 1_800_000}


def safe_diagnostics(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    if isinstance(value.get('stage'), str) and value['stage'] in STAGES:
        result['stage'] = value['stage']
    if isinstance(value.get('finish_reason'), str) and value['finish_reason'] in {'stop', 'length'}:
        result['finish_reason'] = value['finish_reason']
    for key, limit in COUNTER_LIMITS.items():
        count = value.get(key)
        if type(count) is int and 0 <= count <= limit:
            result[key] = count
    return result


class ModelWorkerFailure(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = safe_diagnostics(diagnostics)
        stage = self.diagnostics.get('stage')
        messages = {
            'input': 'Local model worker rejected its input',
            'model_check': 'Local model files could not be verified',
            'dependencies': 'Local model dependencies could not be loaded',
            'load': 'Local model could not be loaded',
            'prompt': 'Local model input could not be prepared',
            'generation': 'Local model generation failed',
            'decode': 'Local model returned invalid structured output',
        }
        message = messages.get(stage, 'Local model did not produce a valid result')
        if stage == 'decode' and self.diagnostics.get('finish_reason') == 'length':
            message = 'Local model reached its token limit without a valid structured answer'
        super().__init__(message + '; retrieved evidence is retained')
