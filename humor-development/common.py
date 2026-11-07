"""Lightweight I/O and isolated reuse of Heart Ridge's read-only helpers."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def read_jsonl(path):
    with Path(path).open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_seed(text, seed=1729):
    return int(hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()[:8], 16)


def safe_output(path):
    """All writes belong to this experiment, including symlink resolution."""
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Output must be inside {ROOT}; Heart Ridge is read-only")
    return path


def new_directory(path):
    path = safe_output(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_heart_helpers():
    path = REPO / "heart-ridge" / "study.py"
    spec = importlib.util.spec_from_file_location("heart_ridge_readonly", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_revision(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use an immutable 40-character model commit SHA, not main")


def unique(rows, key="id"):
    ids = [r[key] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {key}")
    return {r[key]: r for r in rows}
