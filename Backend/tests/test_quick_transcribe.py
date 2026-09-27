"""Home-page quick transcription endpoint (POST /quick-transcribe).

No model is loaded — `transcribe_whisper` is stubbed. What is tested is
validation, the Whisper Small model id, the response shape, and that the
temporary file is always removed.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.api.routes import quick_transcribe


def _wav_bytes(seconds: float = 1.0, sample_rate: int = 44_100) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, tone, sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def stub_whisper(monkeypatch):
    calls: list[tuple[str, str]] = []

    def _fake(model_id, audio_path, *args, **kwargs):
        calls.append((model_id, audio_path))
        assert Path(audio_path).exists(), "clip must exist while it is transcribed"
        return "  hello world "

    monkeypatch.setattr(quick_transcribe, "transcribe_whisper", _fake)
    return calls


async def test_transcribes_with_whisper_small(client, stub_whisper):
    res = await client.post(
        "/quick-transcribe",
        files={"file": ("clip.wav", _wav_bytes(1.5, 44_100), "audio/wav")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body == {
        "model": "openai/whisper-small",
        "transcript": "hello world",
        "duration": 1.5,
        "sample_rate": 44_100,
    }
    assert [m for m, _ in stub_whisper] == ["openai/whisper-small"]


async def test_temp_file_is_removed(client, stub_whisper):
    await client.post("/quick-transcribe", files={"file": ("clip.wav", _wav_bytes(), "audio/wav")})
    _, path = stub_whisper[0]
    assert not Path(path).exists()


@pytest.mark.parametrize(
    "name,content,status",
    [
        ("clip.txt", b"not audio", 400),
        ("clip.wav", b"", 400),
        ("clip.wav", b"RIFF garbage that will not decode", 400),
    ],
)
async def test_bad_uploads_are_refused(client, stub_whisper, name, content, status):
    res = await client.post("/quick-transcribe", files={"file": (name, content, "audio/wav")})
    assert res.status_code == status
    assert stub_whisper == []


async def test_overlong_clip_is_refused(client, stub_whisper, monkeypatch):
    monkeypatch.setattr(quick_transcribe, "MAX_SECONDS", 1.0)
    res = await client.post("/quick-transcribe", files={"file": ("clip.wav", _wav_bytes(2.0), "audio/wav")})
    assert res.status_code == 400
    assert stub_whisper == []


async def test_oversized_file_is_refused(client, stub_whisper, monkeypatch):
    monkeypatch.setattr(quick_transcribe, "MAX_BYTES", 1024)
    res = await client.post("/quick-transcribe", files={"file": ("clip.wav", _wav_bytes(1.0), "audio/wav")})
    assert res.status_code == 413
    assert stub_whisper == []


async def test_model_failure_returns_500_and_cleans_up(client, monkeypatch):
    seen: list[str] = []

    def _boom(model_id, audio_path, *args, **kwargs):
        seen.append(audio_path)
        raise RuntimeError("model exploded")

    monkeypatch.setattr(quick_transcribe, "transcribe_whisper", _boom)
    res = await client.post("/quick-transcribe", files={"file": ("clip.wav", _wav_bytes(), "audio/wav")})
    assert res.status_code == 500
    assert "exploded" not in res.text  # internal errors are not leaked
    assert seen and not Path(seen[0]).exists()
