import os
import re
from pathlib import Path

# Relative on purpose: the backend runs from Backend/, and the inference and
# saliency caches are keyed on str(path), e.g. "uploads/<uuid>.wav". Making
# this absolute would change every cache key.
UPLOAD_DIR = Path("uploads")


def resolve_upload(file_ref: str) -> Path:
    """Resolve a client-supplied upload reference to a file inside UPLOAD_DIR.

    Accepts a bare file id ("<uuid>.wav") or the "uploads/<uuid>.wav" path the
    upload route returns. Only the last path component is kept, so a request
    can never point at a file outside the uploads folder.

    Raises FileNotFoundError when the file does not exist there.
    """
    name = re.split(r"[\\/]", str(file_ref or ""))[-1]
    base = os.path.normpath(UPLOAD_DIR)
    full_path = os.path.normpath(os.path.join(base, name))
    if not full_path.startswith(base + os.sep) or not os.path.isfile(full_path):
        raise FileNotFoundError(f"Uploaded file not found: {name}")
    return Path(full_path)
