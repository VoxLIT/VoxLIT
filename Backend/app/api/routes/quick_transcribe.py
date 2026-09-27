"""Landing-page quick transcription.

One-shot Whisper Small transcript for a clip dropped on the home page. The
clip is written to a temp file, transcribed, and deleted — nothing is kept in
uploads/, no embeddings are computed, and no session state is touched.
"""

import logging
import os
import tempfile
from pathlib import Path

import librosa
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.services.model_loader_service import transcribe_whisper

router = APIRouter()
logger = logging.getLogger(__name__)

MODEL_ID = "openai/whisper-small"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
MAX_BYTES = 25 * 1024 * 1024
MAX_SECONDS = 120.0


@router.post("/quick-transcribe")
async def quick_transcribe(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 25 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="File is empty.")

    fd, tmp_path = tempfile.mkstemp(suffix=extension)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)

        try:
            duration = float(librosa.get_duration(path=tmp_path))
            sample_rate = int(librosa.get_samplerate(tmp_path))
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode the audio file.")
        if duration > MAX_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Clip is {duration:.0f}s long; the quick transcriber accepts up to {MAX_SECONDS:.0f}s.",
            )

        try:
            text = await run_in_threadpool(transcribe_whisper, MODEL_ID, tmp_path)
        except Exception:
            logger.exception("Whisper Small transcription failed")
            raise HTTPException(status_code=500, detail="Transcription failed.")

        return {
            "model": MODEL_ID,
            "transcript": (text or "").strip(),
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
