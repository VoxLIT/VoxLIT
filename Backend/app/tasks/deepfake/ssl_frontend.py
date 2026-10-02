"""Shared loading for the fairseq-trained XLS-R detectors (Models D, E, F).

Wav2Vec2-AASIST, XLSR-SLS and Nes2Net-X all descend from the same training
code (TakHemlata/SSL_Anti-spoofing): an `SSLModel` wrapper around fairseq's
xlsr2_300m.pt, fine-tuned end to end with a different back-end on top. Their
checkpoints are therefore plain PyTorch `state_dict`s with the XLS-R body
stored under `ssl_model.model.*` in FAIRSEQ names -- exactly what Model C's
checkpoint does. So the XLS-R half reuses Model C's rename (xlsr_mamba.py),
verified with strict=True, and only the back-ends are new.

What is NOT shared with Model C: these three were all evaluated on a 64600
sample (~4.04 s) window, not 66800 (data_utils_SSL.pad, max_len=64600 at
eval; the Speech DF Arena re-ran them the same way). AASIST and SLS also
have layers whose input size is FIXED by that window (AASIST's 42-node
spectral position encoding, SLS's 22847-wide fc1), so any other length
would crash -- or, for a shorter window that happened to fit, silently
score a different model.
"""

from __future__ import annotations

from .xlsr_mamba import SSL_PREFIX, build_ssl_model, pad_or_tile, remap_ssl_state_dict

# data_utils_SSL.Dataset_ASVspoof2021_eval: `X_pad = pad(X, 64600)`.
EVAL_CUT_SAMPLES = 64_600

# All three keep the SSL_Anti-spoofing label convention: genSpoof_list writes
# `1 if label == 'bonafide' else 0` and the eval loop scores batch_out[:, 1].
# Same inversion as Model C, the opposite of Model A.
BONAFIDE_INDEX = 1
SPOOF_INDEX = 0


def pad_window(waveform):
    """Reference `pad`: first 64600 samples, tile-repeated if shorter."""
    return pad_or_tile(waveform, max_len=EVAL_CUT_SAMPLES)


def download_state_dict(repo_id: str, filename: str, revision: str, token: str | None):
    """Fetch a `.pth` state_dict and normalise its key names.

    `weights_only=True` refuses anything but tensors and plain containers, so
    a tampered pickle cannot execute code on load. A `module.` prefix (left
    by DataParallel training) is stripped so every checkpoint looks alike.
    """
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id, filename, revision=revision, **({"token": token} if token else {})
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]
    return {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def load_into(model, checkpoint: dict) -> None:
    """Load a full detector strictly: XLS-R body via the rename, head as-is.

    Both halves must account for every tensor. A missing or unexpected key
    raises instead of leaving a layer at its random initialisation, which
    would still produce confident-looking scores.
    """
    ssl_state = remap_ssl_state_dict(checkpoint, model.ssl_model.state_dict().keys())
    model.ssl_model.load_state_dict(ssl_state, strict=True)

    head_state = {key: value for key, value in checkpoint.items() if not key.startswith(SSL_PREFIX)}
    missing, unexpected = model.load_state_dict(head_state, strict=False)
    unaccounted = [key for key in missing if not key.startswith("ssl_model.")]
    if unaccounted or unexpected:
        raise RuntimeError(
            f"{type(model).__name__} head did not load cleanly. "
            f"missing={unaccounted} unexpected={list(unexpected)}"
        )
    model.eval()


def new_ssl_model():
    return build_ssl_model()


def layer_outputs(ssl_model, waveform):
    """Every transformer layer's raw output, as fairseq's `layer_results` has it.

    transformers' `output_hidden_states` is NOT the same thing: its first
    entry is the pre-layer input and its last has the encoder's final
    LayerNorm applied, whereas fairseq records each layer's output before
    that norm. So the encoder is stepped through by hand here. Returns a
    list of 24 tensors, each (batch, frames, 1024).
    """
    features = ssl_model.feature_extractor(waveform).transpose(1, 2)
    hidden_states, _ = ssl_model.feature_projection(features)

    encoder = ssl_model.encoder
    hidden_states = hidden_states + encoder.pos_conv_embed(hidden_states)
    hidden_states = encoder.dropout(hidden_states)

    outputs = []
    for layer in encoder.layers:
        hidden_states = layer(hidden_states)[0]
        outputs.append(hidden_states)
    return outputs
