"""resolve_upload must only ever return files inside the uploads folder."""
import pytest

from app.core import uploads


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "clip.wav").write_bytes(b"RIFF")
    (tmp_path / "secret.txt").write_text("nope")
    return tmp_path


@pytest.mark.parametrize("ref", ["clip.wav", "uploads/clip.wav", "uploads\\clip.wav"])
def test_resolves_upload_references(upload_dir, ref):
    path = uploads.resolve_upload(ref)
    # Stays relative so the str(path) cache keys keep their old form.
    assert str(path) == "uploads/clip.wav"


@pytest.mark.parametrize(
    "ref",
    ["../secret.txt", "uploads/../secret.txt", "/etc/passwd", "..", ".", "", None, "missing.wav"],
)
def test_rejects_anything_outside_uploads(upload_dir, ref):
    with pytest.raises(FileNotFoundError):
        uploads.resolve_upload(ref)


@pytest.mark.parametrize("route", ["/saliency/generate", "/inferences/wav2vec2-detailed"])
async def test_routes_refuse_paths_outside_uploads(upload_dir, client, route):
    response = await client.post(route, json={"model": "wav2vec2", "file_path": "../secret.txt"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Uploaded file not found: secret.txt"

