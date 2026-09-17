"""Three explicit correctness edits to imported FLA MoM; recurrence is unchanged."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def corrected_source(source):
    replacements=[
        ('sorted_indices = combined.argsort()', 'sorted_indices = combined.argsort(stable=True)'),
        ("            if self.training:\n                conv_cu_seqlens = None\n            elif seq_len != 1", "            if seq_len != 1"),
        ("mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode", "mode = 'fused_recurrent' if (hidden_states.shape[1] <= 64 and not self.training) else self.mode"),
    ]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source

def install():
    import fla.layers.mom as native
    if getattr(native,'_bdmq_causal_patch',False):return native.MomAttention
    provenance=json.loads((ROOT/'sources.json').read_text())['mom']
    source=Path(native.__file__).read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==provenance['mom_layer_sha256']
    corrected=corrected_source(source)
    assert hashlib.sha256(corrected.encode()).hexdigest()==provenance['corrected_layer_sha256']
    import fla.models.mom.modeling_mom as native_model
    assert hashlib.sha256(Path(native_model.__file__).read_bytes()).hexdigest()==provenance['balancing_source_sha256']
    exec(compile(corrected,native.__file__,'exec'),native.__dict__)
    native._bdmq_causal_patch=True
    return native.MomAttention
