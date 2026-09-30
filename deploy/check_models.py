"""Load every VoxLIT model and run one real clip through it.

Run from Backend/:  .venv/bin/python ../deploy/check_models.py [--with-large]
Also runs inside Modal via `modal run deploy/modal_app.py::warm`, which
downloads every model into the shared cache volume.
"""
import sys
import time
import traceback

sys.path.insert(0, ".")
AUDIO = "scripts/audio.wav"
results = []


def check(name, fn):
    t = time.time()
    try:
        summary = fn()
        results.append((name, "OK", time.time() - t, summary))
        print(f"[OK]   {name} ({time.time() - t:.1f}s): {summary}", flush=True)
    except Exception as e:  # noqa: BLE001
        results.append((name, "FAIL", time.time() - t, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


print("Importing shared model service (loads emotion model)...", flush=True)
from app.services import model_loader_service as mls  # noqa: E402

check("transcription / whisper-base", lambda: str(mls.transcribe_whisper_base(AUDIO))[:120])
check("quick transcribe / whisper-small", lambda: str(mls.transcribe_whisper("openai/whisper-small", AUDIO))[:120])
check("emotion / wav2vec2", lambda: str(mls.wave2vec(AUDIO, return_probabilities=True))[:160])
check("emotion attention (fallback base models)", lambda: str(type(mls.predict_emotion_wave2vec(AUDIO, return_attention=True))))
check("embeddings / whisper", lambda: mls.extract_whisper_embeddings(AUDIO).shape)
check("embeddings / wav2vec2", lambda: mls.extract_wav2vec2_embeddings(AUDIO).shape)

from app.tasks.verification import service as ver  # noqa: E402
for key in ["ecapa-tdnn", "resnet34-lm"]:
    check(f"verification / {key}", lambda k=key: tuple(ver.get_model(k).extract_embedding(AUDIO).shape))

from app.tasks.deepfake import service as df  # noqa: E402
for key in ["xlsr-deepfake", "ast-fakeaudio", "xlsr-mamba"]:
    check(f"deepfake / {key}", lambda k=key: {kk: df.run_detection(k, AUDIO)[kk] for kk in ("decision", "spoof_probability")})

from app.tasks.task_b import service as dia  # noqa: E402
for key in ["pyannote-3.1", "reverb-v1", "reverb-v2"]:
    check(f"diarization / {key}", lambda k=key: {kk: dia.run_diarization(k, AUDIO)[kk] for kk in ("num_speakers", "duration")})

if "--with-large" in sys.argv:
    check("transcription / whisper-large-v3", lambda: str(mls.transcribe_whisper_large(AUDIO))[:120])

print("\n==== SUMMARY ====")
for name, status, secs, summary in results:
    print(f"{status:4}  {secs:6.1f}s  {name}  ->  {str(summary)[:150]}")

if any(status == "FAIL" for _, status, _, _ in results):
    sys.exit(1)
