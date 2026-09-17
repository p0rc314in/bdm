"""Single source of truth for the matched BabyLM campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass


MANIFEST_SHA256 = "59fa1f27b91aa6cc9f8dffa6f365482b95c93cfdf4831d045e7cf34802c1ea4e"
PUBLIC_ARMS = {
    "a16_dense_attention": "dense_a16",
    "b16_native_sdm": "sdm_b16_native",
    "b16_residualized_sdm": "sdm_b16_shared",
}
ARMS = ("dense_a16", "sdm_b16_native", "sdm_b16_shared")
PROTOCOL_ID = "babylm2026-strict-gpt2-causal-t2048-10epoch-v1"

SOURCE_ARCHIVE_SHA256 = (
    "aa796cbeb149051272d9c37ed9e496dd1b9f4d5bb421613bd6f715b774012a1e"
)
SOURCE_MANIFEST_SHA256 = (
    "08bf2d70c2f6c318f0ecedd661504b621bce4de329daec4ca075443c47a22bdf"
)
SOURCE_FORMAT = "causal_overwrite_bus_babylm2026_strict_gpt2_v1"
DERIVED_FORMAT = "babylm2026_strict_gpt2_causal_t2048_10epoch_v1"
CANONICAL_SDM_COMMIT = "61d29928aa7520f421e0bc39d02b4e5006ffd5a1"
EVALUATION_SDM_COMMIT = CANONICAL_SDM_COMMIT
FULL_EWOK_EVALUATION_SDM_COMMIT = CANONICAL_SDM_COMMIT
CORPUS_REVISION = "9e57baaaa91ac3c638746be14d1d5fa6c789f4cf"
EVALUATION_REVISION = "8d52da9424a9ff30b9e8266c4f751aba9c504233"
EVALUATOR_REVISION = "6f825c291e2c4c78ad33b1935fd64d45f52642dc"
EWOK_DATASET = "ewok-core/ewok-core-1.0"
EWOK_REVISION = "34d912a608066c92e2990a0328ffc3bd9a716042"
EWOK_PACKED_FORMAT = "babylm2026_full_ewok_gpt2_v1"
STREAM_SEED = 20_260_818
EXPECTED_SOURCE_ROWS = 11_601_896
EXPECTED_GPT2_TOKENS = 168_511_619
EXPECTED_RECORDS_PER_EPOCH = 82_281
EXPECTED_TARGETS_PER_EPOCH = 168_511_488
CHECKPOINT_EXPOSURES_MILLIONS = (
    *range(1, 11),
    *range(20, 101, 10),
    *range(200, 1_001, 100),
)


@dataclass(frozen=True)
class CampaignSpec:
    seed: int = 0
    width: int = 512
    layers: int = 16
    attention_heads: int = 8
    context_length: int = 2_048
    vocab_size: int = 50_257
    batch_size: int = 8
    passes: int = 10
    learning_rate: float = 3e-4
    minimum_learning_rate_ratio: float = 0.1
    warmup_steps: int = 2_572
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    codebook_size: int = 128
    logical_rows: int = 16_384
    reads: int = 32
    writes: int = 32
    memory_heads: int = 1
    memory_block_size: int = 256
    heartbeat_steps: int = 1_000
    recovery_steps: int = 2_500

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SPEC = CampaignSpec()


def validate_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def validate_spec(spec: CampaignSpec = SPEC) -> None:
    if spec.writes != spec.reads:
        raise ValueError("forward-looking SDM campaigns require R=W")
    if spec.logical_rows != spec.codebook_size**2:
        raise ValueError("native SDM logical rows must equal C^2")
    if spec.context_length != 2_048 or spec.vocab_size != 50_257:
        raise ValueError("BabyLM tokenization contract changed")
    if spec.passes != 10:
        raise ValueError("this campaign is the resolved ten-pass run")
    records = EXPECTED_RECORDS_PER_EPOCH * spec.passes
    steps = (records + spec.batch_size - 1) // spec.batch_size
    if steps != 102_852 or spec.warmup_steps != 2_572:
        raise ValueError("canonical BabyLM update schedule changed")


validate_spec()
