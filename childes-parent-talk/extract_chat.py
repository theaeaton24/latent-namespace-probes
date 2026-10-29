#!/usr/bin/env python3
"""Local-only, preliminary CHAT speaker-tier extraction for CHILDES research.

Not a full CHAT parser, not de-identification, and not approved training data.
Never commit the produced text; follow the corpus access and ground rules.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

MAIN_TIER = re.compile(r"^\*([A-Za-z0-9_]+):\s*(.*)$")
CAREGIVERS = {"MOT", "FAT"}
ROLES = {"MOT", "FAT", "CHI"}

def read_turns(text):
    """Extract raw main-tier utterances and join wrapped continuation lines."""
    turns = []
    current = None
    for line in text.splitlines():
        match = MAIN_TIER.match(line)
        if match:
            if current is not None:
                turns.append(current)
            current = {"role": match.group(1).upper(), "text": match.group(2).strip()}
        elif line.startswith("\t") and current is not None:
            current["text"] += " " + line.strip()
        else:
            if current is not None:
                turns.append(current)
                current = None
    if current is not None:
        turns.append(current)
    return [t for t in turns if t["text"]]

def split_session(relative_path):
    """Stable per-session partition; groups all turns from one .cha file."""
    n = int(hashlib.sha256(str(relative_path).encode("utf8")).hexdigest(), 16) % 10
    return "test" if n == 0 else "validation" if n == 1 else "train"

def derive_records(cha_text, relative_path):
    """Return local-only records. Raw CHAT marks are intentionally NOT stripped."""
    turns = read_turns(cha_text)
    split = split_session(relative_path)
    session_hash = hashlib.sha256(str(relative_path).encode("utf8")).hexdigest()[:16]
    caregiver = [{"session_id":session_hash, "split":split, "turn":i,
                  "text":t["text"]} for i, t in enumerate(turns)
                 if t["role"] in CAREGIVERS]
    dialogue = [{"session_id":session_hash, "split":split, "turn":i,
                 "role":t["role"], "text":t["text"]} for i, t in enumerate(turns)
                if t["role"] in ROLES]
    return caregiver, dialogue

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def prepare(root, out):
    root = Path(root).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    files = sorted(root.rglob("*.cha")) if root.is_dir() else []
    if not files:
        raise ValueError("No local *.cha files found. Register/download an approved corpus yourself.")
    if root == out or root in out.parents:
        raise ValueError("Keep processed data outside the downloaded transcript folder")
    if Path.cwd() in out.parents or out == Path.cwd():
        raise ValueError("Choose an output folder outside the GitHub working tree")
    out.mkdir(parents=True, exist_ok=True)
    all_care = []
    all_dialog = []
    role_counts = Counter()
    for p in files:
        raw = p.read_text(encoding="utf8-sig", errors="replace")
        turns = read_turns(raw)
        role_counts.update(t["role"] for t in turns)
        care, dialogue = derive_records(raw, p.relative_to(root))
        all_care.extend(care)
        all_dialog.extend(dialogue)
    write_jsonl(out / "caregiver.jsonl", all_care)
    write_jsonl(out / "dialogue.jsonl", all_dialog)
    counts = {
        "sessions": len(files),
        "caregiver_turns":len(all_care),
        "dialogue_turns":len(all_dialog),
        "caregiver_splits":dict(Counter(r["split"] for r in all_care)),
        "dialogue_splits":dict(Counter(r["split"] for r in all_dialog)),
        "speaker_counts":dict(role_counts),
        "warning":"Preliminary CHAT filter: no annotation cleanup, de-identification or license audit performed."
    }
    (out / "manifest.json").write_text(json.dumps(counts, indent=2), encoding="utf8")
    print(json.dumps(counts, indent=2))
    if len(files) < 10:
        print("WARNING: Few sessions; hash-based split may leave validation or test empty.")
    print("IMPORTANT: Keep these output files local. Do not publish transcripts.")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Local downloaded CHILDES .cha directory")
    p.add_argument("--out-local", required=True, help="Output directory outside public GitHub tree")
    a=p.parse_args()
    prepare(a.root, a.out_local)

if __name__=="__main__":
    main()
