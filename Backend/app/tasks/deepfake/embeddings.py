"""Feature 4 — 2D/3D embedding view of the demo dataset.

Every clip is scored once and the vector its classification head read is
projected with PCA, t-SNE or UMAP. The response carries coordinates and the
detector's own score per point, never the raw vectors (a projection request
would otherwise ship ~200 x 1024 floats to the browser and back on every
method or 2D/3D toggle) and never a bona fide/spoof label: like every other
runtime response in this task, it must not reveal the answer the workbench asks
the user to judge. Colouring by the model's score shows how the detector
separates the clips without leaking the protocol file.

Cost: one forward pass per clip, cached per file hash, so only the first
projection of a model is slow. Changing the method or 2D/3D afterwards only
re-reads the cache and re-runs the reducer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from starlette.concurrency import run_in_threadpool

from app.core.redis import cache_result, get_result

from .dataset import DatasetUnavailable, list_recordings, resolve_recording_path
from .projection import reduce_embedding_matrix
from .service import file_sha256, get_model_spec, run_embedding

# Part of the cache key: bump when what the vector *is* changes (e.g. reading
# a different layer), so stale vectors of another meaning are never reused.
EMBEDDING_VERSION = "head-input-v1"
EMBEDDING_TTL_SECONDS = 7 * 24 * 60 * 60

# One scoring pass per model at a time. A second request that arrives while the
# first is still filling the cache (the user flips PCA -> UMAP during the slow
# first run) waits, then reads the finished cache instead of scoring every clip
# a second time.
_SCORING_LOCKS: dict[str, asyncio.Lock] = {}


def _scoring_lock(model_key: str) -> asyncio.Lock:
    return _SCORING_LOCKS.setdefault(model_key, asyncio.Lock())


def cache_key(model_key: str) -> str:
    return f"df-embed:{model_key}:{EMBEDDING_VERSION}"


async def _embed_one(model_key: str, path) -> dict:
    audio_hash = await run_in_threadpool(file_sha256, path)
    key = cache_key(model_key)

    cached = await get_result(key, audio_hash)
    if cached is not None:
        return cached

    payload = await run_in_threadpool(run_embedding, model_key, path)
    await cache_result(key, audio_hash, payload, ttl=EMBEDDING_TTL_SECONDS)
    return payload


async def project_dataset(
    model_key: str,
    reduction_method: str,
    n_components: int,
    extra_clips: list[tuple[str, str, Path]] | None = None,
    base_clips: list[tuple[str, str, Path]] | None = None,
    dataset_id: str | None = None,
) -> dict:
    """Score every recording, then reduce the embeddings to 2 or 3 axes.

    `extra_clips` are the visitor's own clips as (id, display name, path). They
    are projected together with the dataset -- so they land on the same axes --
    and appended after it, flagged `uploaded: true`. `base_clips`, in the same
    shape, replaces the demo dataset with a researcher's custom one;
    otherwise `dataset_id` picks which built-in subset to project.
    """
    extra_clips = list(extra_clips or [])
    spec = get_model_spec(model_key)
    if base_clips is None:
        base_clips = [
            (recording.recording_id, recording.display_filename, resolve_recording_path(recording.recording_id))
            for recording in list_recordings(dataset_id)
        ]
    if not base_clips:
        # The folder exists but holds no clips: nothing to plot, and an empty
        # matrix has no columns to reduce.
        raise DatasetUnavailable("The selected dataset contains no recordings.")

    async with _scoring_lock(model_key):
        payloads = [await _embed_one(model_key, path) for _, _, path in base_clips]
        extra_payloads = [await _embed_one(model_key, path) for _, _, path in extra_clips]

    all_payloads = payloads + extra_payloads
    matrix = np.asarray([payload["embedding"] for payload in all_payloads], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must not contain NaN or infinite values.")

    coordinates, effective_components, method_used = await run_in_threadpool(
        reduce_embedding_matrix, matrix, reduction_method, n_components
    )

    def _row(recording_id: str, display_filename: str, payload: dict, uploaded: bool) -> dict:
        row = {
            "recording_id": recording_id,
            "display_filename": display_filename,
            "spoof_probability": payload["spoof_probability"],
            "decision": "spoof" if payload["spoof_probability"] >= spec.threshold else "bonafide",
        }
        # Only user clips carry the flag, so dataset rows keep their contract.
        return {**row, "uploaded": True} if uploaded else row

    return {
        "model": model_key,
        "model_label": spec.label,
        "reduction_method": reduction_method,
        "reduction_method_used": method_used,
        "n_components": n_components,
        "effective_components": effective_components,
        "embedding_dimension": int(matrix.shape[1]),
        "total_recordings": len(base_clips),
        "threshold": spec.threshold,
        "threshold_calibrated": spec.threshold_calibrated,
        # Index-aligned with `coordinates`.
        "recordings": [
            _row(clip_id, name, payload, False)
            for (clip_id, name, _), payload in zip(base_clips, payloads)
        ]
        + [
            _row(clip_id, name, payload, True)
            for (clip_id, name, _), payload in zip(extra_clips, extra_payloads)
        ],
        "coordinates": coordinates.tolist(),
    }
