"""Official human-likeness aggregation using the existing prepared row maps."""
import json
import numpy as np
import tiktoken
from .official_human import AoAEvaluator, _calculate_reading_results
from .spec import CHECKPOINT_EXPOSURES_MILLIONS
from .io import sha256_file


class Tokenizer:
    vocab_size=50257
    def __init__(self): self.encoder=tiktoken.get_encoding('gpt2')
    def __call__(self,text,*,add_special_tokens):
        assert not add_special_tokens
        return {'input_ids':self.encoder.encode_ordinary(text)}


def score_human(inference, inputs):
    manifest=json.loads((inputs/'manifest.json').read_text())
    arrays={}
    for section,keys in [('reading',('current_indices','previous_indices')),('aoa',('word_ids',))]:
        for key in keys:
            record=manifest[section]['records'][key]
            assert sha256_file(inputs/record['path'])==record['sha256']
            arrays[key]=np.load(inputs/record['path'],allow_pickle=False)
    readings={};aoa=[];words=manifest['aoa']['words']
    for exposure in CHECKPOINT_EXPOSURES_MILLIONS:
        folder=inference/f'{exposure:04d}'
        evidence=json.loads((folder/'COMPLETE.json').read_text())
        for name in ('reading_losses.npy','aoa_losses.npy'):
            assert sha256_file(folder/name)==evidence['files'][name]
        values=np.load(folder/'reading_losses.npy',allow_pickle=False)
        predictions=[dict(pred=float(values[int(current)]),prev_pred=float('nan') if previous<0 else float(values[int(previous)]))
                     for current,previous in zip(arrays['current_indices'],arrays['previous_indices'],strict=True)]
        readings[str(exposure)]=_calculate_reading_results({'reading':{'predictions':predictions}},inputs/'reading_data.csv')
        values=np.load(folder/'aoa_losses.npy',allow_pickle=False)
        aoa.extend(dict(target_word=words[int(word)],step=f'{exposure}M',surprisal=float(value))
                   for word,value in zip(arrays['word_ids'],values,strict=True))
    return dict(reading=readings,aoa=AoAEvaluator(inputs/'cdi_human.csv').compute_curve_fitness({'results':aoa},Tokenizer()))
