"""Speaker Diarization endpoints (mounted at /tasks/task-b)."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.redis import cache_result, get_result
from app.services.dataset_service import media_type_for

from . import diff, perturbation, uploads
from .dataset import (
    DATASET_ID,
    DatasetUnavailable,
    RecordingNotFound,
    get_dataset_info,
    get_recording,
    list_recordings,
    resolve_recording_path,
)
from .service import (
    DiarizationModelUnavailable,
    UnsupportedDiarizationModel,
    file_sha256,
    get_model_spec,
    list_models,
    run_diarization,
)

router = APIRouter()

CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # demo files are static; keep for a week


@router.get("/models")
async def available_models():
    return {"models": list_models()}


@router.get("/dataset")
async def dataset_info():
    return get_dataset_info()


@router.get("/dataset/recordings")
async def dataset_recordings():
    try:
        recordings = list_recordings()
    except DatasetUnavailable as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {
        "dataset_id": DATASET_ID,
        "total_recordings": len(recordings),
        "recordings": [asdict(recording) for recording in recordings],
    }


@router.get("/dataset/recordings/{recording_id}")
async def dataset_recording(recording_id: str):
    try:
        recording = get_recording(recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return asdict(recording)


@router.get("/dataset/recordings/{recording_id}/audio")
async def dataset_recording_audio(recording_id: str):
    """Serve the audio itself so the frontend player can seek segments."""
    try:
        path = resolve_recording_path(recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, media_type="audio/wav", filename=path.name)


# ---------------------------------------------------------------------------
# Session-scoped uploads -- diarize your own audio, not just the demo set.
# Validation and storage conventions are copied from the Speaker Verification
# task (tasks/verification/router.py), which is this repo's reference pattern.
# ---------------------------------------------------------------------------


def _require_sid(request: Request) -> str:
    """`SessionMiddleware` sets this on every request, so a miss means the
    middleware was bypassed, not that the user has no session yet."""

    sid = getattr(request.state, "sid", None)
    try:
        return uploads.validate_and_canonicalize_sid(sid)
    except uploads.InvalidSessionId as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _validate_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in uploads.ALLOWED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(uploads.ALLOWED_AUDIO_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported audio format. Allowed: {allowed}")
    return suffix


def _save_upload(file: UploadFile, destination: Path) -> None:
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    if destination.stat().st_size == 0:
        raise HTTPException(status_code=400, detail=f"Audio file is empty: {file.filename}")
    if destination.stat().st_size > uploads.MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio file exceeds 50 MB: {file.filename}")


@router.post("/uploads")
async def upload_recording(request: Request, file: UploadFile = File(...)):
    """Store one audio file for this session and return it in the same shape
    as a dataset recording, so the frontend can list both together."""

    sid = _require_sid(request)
    suffix = _validate_upload(file)
    temp_path = await run_in_threadpool(uploads.begin_upload_write, sid, suffix)
    try:
        await run_in_threadpool(_save_upload, file, temp_path)
        recording = await run_in_threadpool(uploads.promote_upload, temp_path, sid, suffix)
        return asdict(recording)
    except Exception:
        # The temp file is ours until promotion renames it away; a rejected
        # or failed upload must never leave a partial file behind.
        await run_in_threadpool(temp_path.unlink, missing_ok=True)
        raise
    finally:
        await file.close()


@router.get("/uploads")
async def list_uploaded_recordings(request: Request):
    sid = _require_sid(request)
    recordings = await run_in_threadpool(uploads.list_uploads, sid)
    return {
        "total_uploads": len(recordings),
        "uploads": [asdict(recording) for recording in recordings],
    }


@router.get("/uploads/{upload_id}/audio")
async def uploaded_recording_audio(request: Request, upload_id: str):
    """Serve an uploaded file so the frontend player can seek segments."""

    sid = _require_sid(request)
    try:
        path = await run_in_threadpool(uploads.resolve_upload_path, upload_id, sid)
    except uploads.UploadNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    try:
        media_type = media_type_for(path)
    except ValueError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    return FileResponse(path, media_type=media_type, filename=path.name)


class RunRequest(BaseModel):
    model: str
    recording_id: str


async def _resolve_audio_source(recording_id: str, sid: str) -> Path:
    """Route an `upl_...` upload id or a `prt_...` perturbed clip to this
    session's storage, and anything else to the demo dataset. No resolver
    joins the untrusted id into a path -- each compares it against ids derived
    from a directory listing -- so an unknown or traversal-shaped id simply
    misses and 404s."""

    if recording_id.startswith(uploads.UPLOAD_ID_PREFIX):
        try:
            return await run_in_threadpool(uploads.resolve_upload_path, recording_id, sid)
        except uploads.UploadNotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    if recording_id.startswith(uploads.PERTURBED_ID_PREFIX):
        try:
            return await run_in_threadpool(uploads.resolve_perturbed_path, recording_id, sid)
        except uploads.PerturbedNotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    try:
        return resolve_recording_path(recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


async def _get_or_compute(model_key: str, recording_id: str, sid: str) -> dict:
    """Cache-through diarization; key = model + content hash of the file.

    Uploads and demo recordings deliberately share this one path: the cache
    key comes from the file's *bytes*, never its id, so a re-run of the same
    upload is an instant hit -- which matters because an uncached run costs
    minutes of CPU inference.
    """
    try:
        get_model_spec(model_key)
    except UnsupportedDiarizationModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    path = await _resolve_audio_source(recording_id, sid)

    audio_hash = await run_in_threadpool(file_sha256, path)
    cache_model_key = f"diar:{model_key}"

    cached = await get_result(cache_model_key, audio_hash)
    if cached is not None:
        return {**cached, "recording_id": recording_id, "cached": True}

    try:
        payload = await run_in_threadpool(run_diarization, model_key, path)
    except DiarizationModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    await cache_result(cache_model_key, audio_hash, payload, ttl=CACHE_TTL_SECONDS)
    return {**payload, "recording_id": recording_id, "cached": False}


@router.post("/run")
async def run(request: Request, run_request: RunRequest):
    return await _get_or_compute(
        run_request.model, run_request.recording_id, _require_sid(request)
    )


@router.get("/projection")
async def projection(request: Request, model: str, recording_id: str):
    """2D PCA of segment embeddings → [{id, x, y, speaker, confidence}]."""
    result = await _get_or_compute(model, recording_id, _require_sid(request))

    embeddings = result.get("embeddings", {})
    if len(embeddings) < 2:
        raise HTTPException(
            status_code=422,
            detail="Not enough embeddable segments for a 2D projection.",
        )

    def _pca() -> list[dict]:
        import numpy as np
        from sklearn.decomposition import PCA

        segment_ids = list(embeddings.keys())
        matrix = np.asarray([embeddings[i] for i in segment_ids], dtype=np.float32)
        coords = PCA(n_components=2).fit_transform(matrix)
        by_id = {s["id"]: s for s in result["segments"]}
        return [
            {
                "id": segment_id,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "speaker": by_id[segment_id]["speaker"],
                "confidence": by_id[segment_id]["confidence"],
            }
            for segment_id, (x, y) in zip(segment_ids, coords)
        ]

    return {
        "model": model,
        "recording_id": recording_id,
        "points": await run_in_threadpool(_pca),
    }


# ---------------------------------------------------------------------------
# Perturbation counterfactuals -- degrade the audio, re-diarize, diff the two
# runs. Clustering is non-differentiable, so this is the honest way to probe
# the pipeline's sensitivity; there is no saliency or attention to fall back on.
# ---------------------------------------------------------------------------


class PerturbationSpec(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class PerturbationRequest(BaseModel):
    model: str
    recording_id: str
    perturbation: PerturbationSpec


@router.get("/perturbed/{perturbed_id}/audio")
async def perturbed_audio(request: Request, perturbed_id: str):
    """Serve a perturbed clip so the UI can play the degraded audio next to
    the original."""

    sid = _require_sid(request)
    try:
        path = await run_in_threadpool(uploads.resolve_perturbed_path, perturbed_id, sid)
    except uploads.PerturbedNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@router.post("/perturbation")
async def run_perturbation(request: Request, payload: PerturbationRequest):
    """Perturb one recording, diarize the result through the ordinary cached
    `/run` path, and return the diff against the original run.

    Every layer here is content-addressed, which is what makes a repeat of the
    same perturbation instant rather than another few minutes of CPU: the
    perturbed file's id is derived from the source hash plus the canonical
    params (so it is not even rewritten), its diarization is cached under its
    own content hash, and the delta itself is cached under the pair.
    """

    sid = _require_sid(request)
    try:
        get_model_spec(payload.model)
    except UnsupportedDiarizationModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    source_path = await _resolve_audio_source(payload.recording_id, sid)
    source_hash = await run_in_threadpool(file_sha256, source_path)

    try:
        perturbed_id, perturbed_path, normalized_params = await run_in_threadpool(
            perturbation.perturb_to_session,
            source_path,
            sid,
            source_hash,
            payload.perturbation.type,
            payload.perturbation.params,
        )
    except uploads.InvalidSessionId as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        # Unknown type, bad params, a no-op transform, or invalid output --
        # all of them are "your request could not be carried out", not an
        # infrastructure failure.
        raise HTTPException(status_code=422, detail=str(error)) from error

    perturbed_hash = await run_in_threadpool(file_sha256, perturbed_path)
    delta_cache_key = f"diar-delta:{payload.model}"
    delta_cache_hash = f"{source_hash}:{perturbed_hash}"

    cached = await get_result(delta_cache_key, delta_cache_hash)
    if cached is not None:
        return {**cached, "cached": True}

    original_result = await _get_or_compute(payload.model, payload.recording_id, sid)
    perturbed_result = await _get_or_compute(payload.model, perturbed_id, sid)
    delta = await run_in_threadpool(diff.compare_runs, original_result, perturbed_result)

    result = {
        "model": payload.model,
        "perturbation": {"type": payload.perturbation.type, "params": normalized_params},
        "original": _run_summary(original_result, payload.recording_id),
        "perturbed": _run_summary(perturbed_result, perturbed_id),
        "delta": delta,
    }
    await cache_result(delta_cache_key, delta_cache_hash, result, ttl=CACHE_TTL_SECONDS)
    return {**result, "cached": False}


def _run_summary(result: dict, recording_id: str) -> dict:
    """The parts of a diarization run the comparison view needs. Embeddings
    are deliberately dropped -- a long meeting carries N x 256 floats that the
    stacked timelines never read, and the delta payload is already cached."""

    return {
        "recording_id": recording_id,
        "duration": result["duration"],
        "num_speakers": result["num_speakers"],
        "speakers": result["speakers"],
        "segments": result["segments"],
    }