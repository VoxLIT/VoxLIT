"""Host VoxLIT on Modal: frontend + FastAPI backend + Redis in one container.

    nginx :8080  ->  /        Frontend/dist (built with VITE_API_BASE_URL=/api)
                 ->  /api/*   uvicorn :8000 (Backend/app)
    redis :6379  (in-container, snapshotted to the cache volume every minute)

Model weights live on the "voxlit-cache" Volume, so they download once and
survive restarts. The Hugging Face token comes from the "voxlit-hf" Secret.

    modal run deploy/modal_app.py::warm      # download + test every model
    modal deploy deploy/modal_app.py         # publish the site
"""
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "Frontend" / "dist"

if modal.is_local():
    if not (DIST / "index.html").exists():
        raise SystemExit("Frontend/dist is missing. Build it first (see deploy steps).")
    if any("localhost:8000" in p.read_text(errors="ignore") for p in (DIST / "assets").glob("*.js")):
        raise SystemExit(
            "Frontend/dist still points at localhost:8000. Rebuild with "
            "VITE_API_BASE_URL=/api npm run build"
        )

# Any file or folder with one of these names is left out. .env holds the
# HF token -- on Modal it comes from the "voxlit-hf" Secret instead.
BACKEND_IGNORE = {
    ".venv",
    ".env",
    "__pycache__",
    ".pytest_cache",
    "tests",
    "uploads",
    "pretrained_models",
    ".DS_Store",
}
def _is_truncated_wav(path: Path) -> bool:
    """True when a .wav's RIFF header promises more bytes than the file holds,
    i.e. a download that stopped partway."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(8)
        if header[:4] != b"RIFF":
            return False
        return int.from_bytes(header[4:8], "little") + 8 > path.stat().st_size
    except OSError:
        return True


def _ignore_backend(path: Path) -> bool:
    """Skip dev-only folders, plus any .wav whose download never finished,
    so a half-downloaded file never ships and breaks a dataset listing."""
    path = Path(path)
    rel = path.relative_to(ROOT / "Backend") if path.is_absolute() else path
    if BACKEND_IGNORE.intersection(rel.parts):
        return True
    full = ROOT / "Backend" / rel
    if full.suffix.lower() == ".wav" and full.is_file():
        return _is_truncated_wav(full)
    return False


cache = modal.Volume.from_name("voxlit-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("voxlit-hf")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "nginx", "redis-server", "git")
    # CPU-only torch: a much smaller image than the default CUDA wheels.
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install_from_requirements(str(ROOT / "Backend" / "requirements.txt"))
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "SPEAKER_VERIFICATION_MODEL_ROOT": "/cache/speaker_verification",
            "COOKIE_SECURE": "true",
        }
    )
    .add_local_file(str(ROOT / "deploy" / "nginx.conf"), "/etc/voxlit/nginx.conf")
    .add_local_file(str(ROOT / "deploy" / "check_models.py"), "/app/deploy/check_models.py")
    .add_local_dir(str(DIST), "/app/Frontend/dist")
    .add_local_dir(
        str(ROOT / "Backend"),
        "/app/Backend",
        ignore=_ignore_backend,
    )
)

app = modal.App("voxlit", image=image)

SHARED = dict(volumes={"/cache": cache}, secrets=[hf_secret])


def _commit_cache_forever():
    """Persist the Redis snapshot and new uploads to the volume once a minute."""
    while True:
        time.sleep(60)
        try:
            cache.commit()
        except Exception as error:  # noqa: BLE001 -- a failed commit retries next minute
            print(f"cache commit failed: {error}", flush=True)


@app.function(cpu=2.0, memory=8192, timeout=60 * 60, **SHARED)
def warm(with_large: bool = False):
    """Download every model into the cache volume and run one clip through each."""
    cmd = ["python", "/app/deploy/check_models.py"] + (["--with-large"] if with_large else [])
    result = subprocess.run(cmd, cwd="/app/Backend")
    cache.commit()
    if result.returncode:
        raise SystemExit("Some models failed; see the log above.")


@app.function(
    cpu=2.0,
    # Every model stays loaded once used: 6 deepfake detectors + Whisper
    # (incl. large-v3) + emotion reach ~18 GiB. 8 GiB ran out of memory and
    # restarted the container mid-request.
    memory=20480,
    # One container only: models, uploads and session files live in its memory
    # and disk, so a second container would not see them.
    max_containers=1,
    # Stay warm 10 minutes after the last request, then scale to zero ($0).
    scaledown_window=10 * 60,
    timeout=60 * 60,
    **SHARED,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8080, startup_timeout=10 * 60)
def web():
    # Redis and every task's uploads live on the cache volume, so cached
    # results (e.g. the deepfake voice map, minutes to compute on CPU) and
    # visitors' clips survive the container going to sleep. They are kept
    # together so a clip listed in Redis always still has its file.
    Path("/cache/redis").mkdir(parents=True, exist_ok=True)
    Path("/cache/uploads").mkdir(parents=True, exist_ok=True)
    uploads = Path("/app/Backend/uploads")
    if not uploads.exists():
        uploads.symlink_to("/cache/uploads")
    subprocess.Popen(
        ["redis-server", "--dir", "/cache/redis", "--save", "60 1",
         "--appendonly", "no", "--maxmemory", "1gb", "--maxmemory-policy", "allkeys-lru"]
    )
    threading.Thread(target=_commit_cache_forever, daemon=True).start()
    subprocess.Popen(
        ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd="/app/Backend",
    )
    # Open the public port only once the API answers, so the first visitor
    # never sees a 502 while models are still loading.
    for _ in range(600):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    subprocess.Popen(["nginx", "-c", "/etc/voxlit/nginx.conf", "-g", "daemon off;"])
