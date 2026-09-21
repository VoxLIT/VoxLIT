"""Audio Deepfake Detection endpoints (mounted at /tasks/deepfake)."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.core.redis import cache_result, get_result
# Range-aware audio streaming, already extracted as a shared helper so
# every task's audio route serves bytes identically. Reused rather than
# reimplemented -- a plain FileResponse ignores Range, which leaves the
# browser unable to seek within the clip.
from app.tasks.verification.audio_streaming import stream_audio_file

from .dataset import (
    DATASET_ID,
    DatasetUnavailable,
    RecordingNotFound,
    get_dataset_info,
    get_recording,
    list_recordings,
    resolve_recording_path,
)
from .embeddings import project_dataset
from .evaluation import evaluate_dataset
from .saliency import METHOD as SALIENCY_METHOD, SaliencyUnavailable, generate_saliency
from .silence_probe import SILENCE_TOP_DB, run_silence_probe
from .metrics import NotEnoughLabelledData
from .uploads import (
    ALLOWED_UPLOAD_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    UploadNotFound,
    UploadRejected,
    delete_upload,
    get_upload,
    is_upload_id,
    list_uploads,
    resolve_upload_path,
    save_upload,
)
from .projection import SUPPORTED_PROJECTION_COMPONENTS, SUPPORTED_REDUCTION_METHODS
from .service import (
    THRESHOLD_VERSION,
    DeepfakeModelUnavailable,
    UnsupportedDeepfakeModel,
    file_sha256,
    get_model_spec,
    list_models,
    run_detection,
)

router = APIRouter()

CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # demo files are static; keep for a week


def _resolve_clip(recording_id: str, request: Request) -> Path:
    """A demo `rec_...` id or one of this session's `up_...` clips -> Path.

    Each prefix goes to exactly one resolver and never falls through to the
    other. Both raise a 404 on any miss, including another session's clip.
    """
    try:
        if is_upload_id(recording_id):
            return resolve_upload_path(getattr(request.state, "sid", None), recording_id)
        return resolve_recording_path(recording_id)
    except (DatasetUnavailable, RecordingNotFound, UploadNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/models")
async def available_models():
    return {"models": list_models()}


@router.get("/dataset")
async def dataset_info():
    return get_dataset_info()


@router.get("/dataset/recordings")
async def dataset_recordings():
    """Opaque recording ids only -- never the bona fide/spoof key. See
    dataset.py's module docstring."""
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
@router.head("/dataset/recordings/{recording_id}/audio")
async def dataset_recording_audio(recording_id: str, request: Request):
    """Serve the audio itself so the frontend player can listen to the clip.

    HEAD is declared alongside GET because the shared waveform player probes
    the URL with HEAD before loading it; without it that probe returns 405 and
    the player reports the clip as unreachable.

    Streamed through the shared helper so a `Range` header gets a real 206.
    Without that the browser can load the clip but cannot seek inside it, so
    clicking the saliency strip to jump to a moment silently does nothing.
    """
    try:
        path = resolve_recording_path(recording_id)
    except (DatasetUnavailable, RecordingNotFound) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return stream_audio_file(path, request, path.name, "audio/flac")


# ---------------------------------------------------------------------------
# The visitor's own clips (uploads + microphone recordings)
# ---------------------------------------------------------------------------


def _write_limited(file: UploadFile, destination: Path) -> None:
    """Copy the upload to disk, refusing as soon as it passes the size cap."""
    written = 0
    with destination.open("wb") as output:
        while chunk := file.file.read(1 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            output.write(chunk)
    if written == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")


def _store_upload(file: UploadFile, sid: str | None, source: str) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported audio format. Allowed: {allowed}.")
    with tempfile.TemporaryDirectory(prefix="df-upload-") as scratch:
        raw_path = Path(scratch) / f"raw{suffix}"
        _write_limited(file, raw_path)
        try:
            return asdict(save_upload(sid, raw_path, file.filename, source))
        except UploadRejected as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except UploadNotFound as error:
            raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/uploads")
async def upload_clip(
    request: Request,
    file: UploadFile = File(...),
    source: str = Form("upload"),
):
    """Store one of the visitor's clips (a file or a microphone recording).

    Returns a RecordingInfo-shaped record whose `up_...` id every per-clip
    endpoint (/run, /silence-probe, /saliency, /embeddings) accepts.
    """
    sid = getattr(request.state, "sid", None)
    return await run_in_threadpool(_store_upload, file, sid, source)


@router.get("/uploads")
async def uploaded_clips(request: Request):
    clips = await run_in_threadpool(list_uploads, getattr(request.state, "sid", None))
    return {"recordings": [asdict(clip) for clip in clips]}


@router.get("/uploads/{clip_id}/audio")
@router.head("/uploads/{clip_id}/audio")
async def uploaded_clip_audio(clip_id: str, request: Request):
    try:
        path = resolve_upload_path(getattr(request.state, "sid", None), clip_id)
    except UploadNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return stream_audio_file(path, request, path.name, "audio/wav")


@router.delete("/uploads/{clip_id}")
async def delete_uploaded_clip(clip_id: str, request: Request):
    try:
        delete_upload(getattr(request.state, "sid", None), clip_id)
    except UploadNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"deleted": clip_id}


class EvaluationRequest(BaseModel):
    model: str


@router.post("/scores")
async def evaluation(request: EvaluationRequest):
    """Feature 1 — score distributions, DET curve and EER (SRS DF-6..DF-9).

    Aggregates only: no per-recording label is ever returned, so the
    workbench's read-the-score-then-check-the-protocol exercise survives.

    The first call scores the whole dataset and can take minutes; per-clip
    scores are cached afterwards, shared with /run.
    """
    try:
        get_model_spec(request.model)
    except UnsupportedDeepfakeModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        return await evaluate_dataset(request.model)
    except DatasetUnavailable as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except NotEnoughLabelledData as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except DeepfakeModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


class EmbeddingProjectionRequest(BaseModel):
    model: str
    reduction_method: str = "pca"
    n_components: int = 2
    # The visitor's own `up_...` clips, placed on the same map as the dataset.
    extra_recording_ids: list[str] = []


@router.post("/embeddings")
async def embedding_projection(request: EmbeddingProjectionRequest, http_request: Request):
    """Feature 4 — 2D/3D embedding view of the whole demo dataset.

    Visualisation only: each point is the vector the detector's classification
    head read for one clip, reduced to 2 or 3 axes. Points carry the model's own
    score, never the bona fide/spoof label. The first call per model scores the
    whole dataset and can take minutes; per-clip vectors are cached afterwards,
    so changing the method or 2D/3D is fast.
    """
    try:
        get_model_spec(request.model)
    except UnsupportedDeepfakeModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if request.reduction_method not in SUPPORTED_REDUCTION_METHODS:
        allowed = ", ".join(sorted(SUPPORTED_REDUCTION_METHODS))
        raise HTTPException(
            status_code=400, detail=f"reduction_method must be one of: {allowed}."
        )
    if request.n_components not in SUPPORTED_PROJECTION_COMPONENTS:
        raise HTTPException(status_code=422, detail="n_components must be 2 or 3.")

    sid = getattr(http_request.state, "sid", None)
    extra_clips = []
    for clip_id in dict.fromkeys(request.extra_recording_ids):
        if not is_upload_id(clip_id):
            raise HTTPException(status_code=400, detail="extra_recording_ids takes user clip ids only.")
        path = _resolve_clip(clip_id, http_request)
        try:
            name = get_upload(sid, clip_id).display_filename
        except UploadNotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        extra_clips.append((clip_id, name, path))

    try:
        return await project_dataset(
            request.model, request.reduction_method, request.n_components, extra_clips
        )
    except DatasetUnavailable as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DeepfakeModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


class SilenceProbeRequest(BaseModel):
    model: str
    recording_id: str


@router.post("/silence-probe")
async def silence_probe(request: SilenceProbeRequest, http_request: Request):
    """Feature 2 — score the clip as submitted, trimmed, and silence-only.

    SRS DF-10..DF-12. Three forward passes on one clip: no dataset, no
    labels, no gradients. Cached on the model, the silence threshold and the
    file's content hash, so re-opening the card is free.
    """
    try:
        get_model_spec(request.model)
    except UnsupportedDeepfakeModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    path = _resolve_clip(request.recording_id, http_request)

    audio_hash = await run_in_threadpool(file_sha256, path)
    cache_model_key = f"df-silence:{request.model}:{THRESHOLD_VERSION}:{SILENCE_TOP_DB}"

    cached = await get_result(cache_model_key, audio_hash)
    if cached is not None:
        return {**cached, "recording_id": request.recording_id, "cached": True}

    try:
        payload = await run_in_threadpool(run_silence_probe, request.model, path)
    except DeepfakeModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    await cache_result(cache_model_key, audio_hash, payload, ttl=CACHE_TTL_SECONDS)
    return {**payload, "recording_id": request.recording_id, "cached": False}


class SaliencyRequest(BaseModel):
    model: str
    recording_id: str


@router.post("/saliency")
async def saliency(request: SaliencyRequest, http_request: Request):
    """Feature 3 — waveform-aligned temporal attribution (SRS DF-14, DF-15).

    One forward and one backward pass over a single clip. Emits the shared
    saliency service's response contract so the payload is interchangeable
    with its visualisation.
    """
    try:
        get_model_spec(request.model)
    except UnsupportedDeepfakeModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    path = _resolve_clip(request.recording_id, http_request)

    audio_hash = await run_in_threadpool(file_sha256, path)
    cache_model_key = f"df-saliency:{request.model}:{SALIENCY_METHOD}"

    cached = await get_result(cache_model_key, audio_hash)
    if cached is not None:
        return {**cached, "recording_id": request.recording_id, "cached": True}

    try:
        payload = await run_in_threadpool(generate_saliency, request.model, path)
    except DeepfakeModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except SaliencyUnavailable as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    await cache_result(cache_model_key, audio_hash, payload, ttl=CACHE_TTL_SECONDS)
    return {**payload, "recording_id": request.recording_id, "cached": False}


class RunRequest(BaseModel):
    model: str
    recording_id: str


@router.post("/run")
async def run(request: RunRequest, http_request: Request):
    """Cache-through detection; key = model + threshold version + file hash."""
    try:
        get_model_spec(request.model)
    except UnsupportedDeepfakeModel as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    path = _resolve_clip(request.recording_id, http_request)

    audio_hash = await run_in_threadpool(file_sha256, path)
    cache_model_key = f"df:{request.model}:{THRESHOLD_VERSION}"

    cached = await get_result(cache_model_key, audio_hash)
    if cached is not None:
        return {**cached, "recording_id": request.recording_id, "cached": True}

    try:
        payload = await run_in_threadpool(run_detection, request.model, path)
    except DeepfakeModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    await cache_result(cache_model_key, audio_hash, payload, ttl=CACHE_TTL_SECONDS)
    return {**payload, "recording_id": request.recording_id, "cached": False}
