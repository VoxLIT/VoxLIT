"""Models D-F (Wav2Vec2-AASIST, XLSR-SLS, Nes2Net-X) — Tier B integration tests.

The architecture tests use a tiny 2-layer XLS-R body so they run in seconds
without the 1.3 GB checkpoints. The strict-load tests at the bottom use the
real checkpoints and are skipped unless they are already in the Hugging Face
cache: they are what proves the transcription matches the released weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.tasks.deepfake import nes2net, service, ssl_frontend, w2v2_aasist, xlsr_sls
from app.tasks.deepfake.xlsr_mamba import XLSR_BASE_ID

NEW_MODELS = ("w2v2-aasist", "xlsr-sls", "nes2net-x")


def _tiny_ssl_model():
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    config = Wav2Vec2Config.from_pretrained(XLSR_BASE_ID)
    config.num_hidden_layers = 2
    config.intermediate_size = 64
    config.apply_spec_augment = False
    return Wav2Vec2Model(config).eval()


@pytest.fixture(scope="module")
def window():
    return torch.randn(1, ssl_frontend.EVAL_CUT_SAMPLES) * 0.1


# --- registry -------------------------------------------------------------


@pytest.mark.parametrize("key", NEW_MODELS)
def test_new_models_are_tier_b_and_pinned(key):
    spec = service.get_model_spec(key)

    assert spec.tier == "B"
    assert spec.architecture == key
    assert spec.gated is False
    # A 40-hex commit, not "main": a re-push must not silently change scores.
    assert len(spec.revision) == 40 and int(spec.revision, 16) >= 0


def test_new_models_use_the_ssl_anti_spoofing_label_order():
    """Index 1 = bona fide, like Model C and unlike Model A."""
    for key in NEW_MODELS:
        architecture = service._architecture(service.get_model_spec(key).architecture)
        assert architecture.bonafide_index == 1
        assert architecture.spoof_index == 0
        assert architecture.window_samples == 64_600


def test_pad_window_tiles_to_64600():
    clip = np.arange(1, 1001, dtype=np.float32)

    result = ssl_frontend.pad_window(clip)

    assert result.shape == (64_600,)
    assert result[1000] == clip[0]  # tiled, not zero-padded
    assert 0.0 not in result


# --- fairseq layer_results equivalence ------------------------------------


def test_layer_outputs_match_transformers_before_the_final_norm(window):
    """SLS reads raw per-layer outputs; transformers normalises the last one.

    The hand-stepped encoder must agree with transformers' own forward for
    every layer it can compare, and its last layer must reproduce
    last_hidden_state once the encoder's LayerNorm is applied.
    """
    ssl_model = _tiny_ssl_model()
    with torch.inference_mode():
        layers = ssl_frontend.layer_outputs(ssl_model, window)
        reference = ssl_model(window, output_hidden_states=True)
        normed_last = ssl_model.encoder.layer_norm(layers[-1])

    assert len(layers) == 2
    assert torch.allclose(layers[0], reference.hidden_states[1], atol=1e-5)
    assert torch.allclose(normed_last, reference.last_hidden_state, atol=1e-5)
    # ...and the raw last layer is genuinely different from the normed one.
    assert not torch.allclose(layers[-1], reference.last_hidden_state, atol=1e-3)


# --- back ends ------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [w2v2_aasist.W2V2AASIST, xlsr_sls.XLSRSLS, nes2net.Nes2NetX],
    ids=["aasist", "sls", "nes2net"],
)
def test_back_end_produces_two_finite_scores(build, window):
    model = build(_tiny_ssl_model()).eval()

    with torch.inference_mode():
        output = model(window)

    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()
    assert isinstance(model.classifier, torch.nn.Linear)


def test_sls_output_is_log_probabilities(window):
    model = xlsr_sls.XLSRSLS(_tiny_ssl_model()).eval()

    with torch.inference_mode():
        output = model(window)

    assert torch.allclose(output.exp().sum(dim=-1), torch.ones(1), atol=1e-5)


def test_sls_flatten_width_is_fixed_by_the_window():
    """201 frames / 3 x 1024 / 3 = 22847 — only true for 64600 samples."""
    frames = 201
    assert (frames // 3) * (1024 // 3) == xlsr_sls.FLAT_DIM


def test_aasist_residual_block_ignores_bn1_like_the_reference():
    """Changing bn1 must not change the output (reference discards it)."""
    block = w2v2_aasist.ResidualBlock([32, 32]).eval()
    x = torch.randn(1, 32, 8, 10)
    with torch.inference_mode():
        before = block(x)
        block.bn1.weight.fill_(123.0)
        after = block(x)

    assert torch.equal(before, after)


def test_nes2net_weighted_sum_initialisation_matches_reference():
    block = nes2net.Bottle2neck(128, kernel_size=3, dilation=2, scale=8, se_ratio=1)

    assert len(block.weighted_sum) == 7
    for i, weights in enumerate(block.weighted_sum):
        assert weights.shape == (1, 1, 1, i + 2)


# --- real checkpoints (skipped unless cached) ------------------------------


def _cached(key):
    from huggingface_hub import try_to_load_from_cache

    spec = service.get_model_spec(key)
    module = {"w2v2-aasist": w2v2_aasist, "xlsr-sls": xlsr_sls, "nes2net-x": nes2net}[key]
    path = try_to_load_from_cache(spec.model_id, module.REPO_FILENAME, revision=spec.revision)
    return isinstance(path, str)


@pytest.mark.slow
@pytest.mark.parametrize("key", NEW_MODELS)
def test_released_checkpoint_loads_strictly(key):
    if not _cached(key):
        pytest.skip(f"{key} checkpoint not in the Hugging Face cache")

    spec = service.get_model_spec(key)
    model = service._architecture(spec.architecture).load(spec.model_id, revision=spec.revision)

    with torch.inference_mode():
        output = model(torch.zeros(1, ssl_frontend.EVAL_CUT_SAMPLES))
    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()
