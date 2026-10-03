"""Audio Deepfake Detection — OWNER: Chanupa Gurusinghe.

Endpoints live in router.py (mounted at /tasks/deepfake by main.py); model
loading and inference live in service.py; read-only dataset discovery for the
three built-in subsets (ASVspoof 2019 LA, ASVspoof 5, In-the-Wild) lives in
dataset.py.
"""
from .router import router

TASK_INFO = {"id": "deepfake", "name": "Audio Deepfake Detection"}

__all__ = ["TASK_INFO", "router"]
