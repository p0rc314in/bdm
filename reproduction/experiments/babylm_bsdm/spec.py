import json
import os
from pathlib import Path
from types import SimpleNamespace

CONFIG_PATH = Path(os.environ.get('BSDM_BABYLM_CONFIG', Path(__file__).parent/'config.json'))
CONFIG = json.loads(CONFIG_PATH.read_text())
SPEC = SimpleNamespace(width=CONFIG['width'], layers=CONFIG['layers'],
    context_length=CONFIG['context'], vocab_size=CONFIG['vocab_size'],
    passes=CONFIG['epochs'], batch_size=CONFIG['effective_batch'])
DERIVED_FORMAT = 'babylm2026_strict_gpt2_causal_t2048_10epoch_v1'
CHECKPOINT_EXPOSURES_MILLIONS = tuple(CONFIG['checkpoint_exposures_millions'])
EVALUATOR_REVISION = '6f825c291e2c4c78ad33b1935fd64d45f52642dc'
