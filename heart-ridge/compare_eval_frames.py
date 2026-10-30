#!/usr/bin/env python3
"""Auditable paired comparison of two evaluation JSONLs.

Strictly verifies the two files differ in exactly the intended system condition
among the recorded settings. Does not rerun models or infer hidden mechanisms.
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# Match the vocabulary and counting convention of heart_length_report.py.
from heart_length_report import position
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

# These should be identical between arms. Differing generated responses, heart
# flags, response-token counts, and system_condition are expected and excluded.
SAME_FIELDS = (
    "id", "prompt", "family", "emotion", "complexity",
    "explicitness", "expected_heart", "adapter", "parent_adapter",
    "seed", "sample_index", "generation", "prompt_format", "model",
    "model_revision", "resolved_model_revision",
)
REQUIRED_FIELDS = ("id", "prompt", "emotion", "adapter", "seed",
                   "sample_index", "generation", "model", "system_condition")

def read_results(path):
    records = {}
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [name for name in REQUIRED_FIELDS if name not in row]
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields {missing}")
            key = (row["id"], row["sample_index"])
            if key in records:
                raise ValueError(f"{path}:{line_number} duplicate id/sample_index {key}")
            if not isinstance(row.get("generation"), dict):
                raise ValueError(f"{path}:{line_number} invalid generation settings")
            records[key] = row
    if not records:
        raise ValueError(f"Empty evaluation file: {path}")
    return records

def protocol_compare(left, right, expected_left="no-system", expected_right="tutor"):
    """Return a descriptive comparison, or fail closed on any recorded mismatch."""
    if set(left) != set(right):
        a = sorted(set(left) - set(right))
        b = sorted(set(right) - set(left))
        raise ValueError(f"Unpaired results: left-only {a[:5]}, right-only {b[:5]} "
                         f"(counts {len(a)}, {len(b)})")
    mismatches = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["system_condition"] != expected_left:
            mismatches.append((key, "left.system_condition",
                               a["system_condition"], expected_left))
        if b["system_condition"] != expected_right:
            mismatches.append((key, "right.system_condition",
                               b["system_condition"], expected_right))
        for field in SAME_FIELDS:
            # Never equate an omitted value with a present null: the exact
            # protocol provenance must match across both files.
            if (field in a) != (field in b) or a.get(field) != b.get(field):
                mismatches.append((key, field, a.get(field, "<missing>"),
                                   b.get(field, "<missing>")))
    if mismatches:
        excerpt = "\n".join(repr(x) for x in mismatches[:12])
        raise ValueError(f"Protocol mismatch ({len(mismatches)}):\n{excerpt}")

    by_context = defaultdict(list)
    all_pairs = []
    for key in sorted(left):
        a,b = left[key],right[key]
        # Work from full text, rather than trusting a precomputed 'heart' flag.
        x, y = position(a.get("response", "")) != "absent", position(b.get("response", "")) != "absent"
        row = {
            "id": key[0], "sample_index": key[1],
            "emotion": a["emotion"], "complexity": a.get("complexity"),
            "left_heart": x, "right_heart": y,
            "left_position":position(a.get("response", "")),
            "right_position":position(b.get("response", "")),
            "left_words":len(WORD_RE.findall(a.get("response", ""))),
            "right_words":len(WORD_RE.findall(b.get("response", ""))),
        }
        all_pairs.append(row)
        by_context[a["emotion"]].append(row)

    def summarize(pairs):
        n = len(pairs)
        left_hearts = sum(x["left_heart"] for x in pairs)
        right_hearts = sum(x["right_heart"] for x in pairs)
        return {
            "n": n,
            "left_heart_count": left_hearts,
            "right_heart_count": right_hearts,
            "left_heart_rate": round(left_hearts/n, 4),
            "right_heart_rate": round(right_hearts/n, 4),
            "right_minus_left_rate": round((right_hearts-left_hearts)/n, 4),
            "gained_hearts": sum(not p["left_heart"] and p["right_heart"] for p in pairs),
            "lost_hearts": sum(p["left_heart"] and not p["right_heart"] for p in pairs),
            "mean_words_left": round(statistics.mean(p["left_words"] for p in pairs),1),
            "mean_words_right": round(statistics.mean(p["right_words"] for p in pairs),1),
        }
    warnings=[]
    # The LoRA and model revision values are logged, but weights themselves
    # are not verified from the result files.
    first = next(iter(left.values()))
    revision = first.get("model_revision")
    resolved = first.get("resolved_model_revision")
    if revision in (None, "", "unfixed") and not resolved:
        warnings.append("Base-model revision is not pinned/resolved in these evaluation records.")
    warnings.append("The JSONL records describe adapter paths, not cryptographic adapter hashes. "
                    "Verify the local run.json manifest/adapter file if checkpoints could have changed.")
    return {
        "protocol_check": "PASS",
        "only_intentional_field_difference": "system_condition",
        "left_condition": expected_left,
        "right_condition": expected_right,
        "paired_records": len(all_pairs),
        "overall": summarize(all_pairs),
        "by_context": {c: summarize(p) for c,p in sorted(by_context.items())},
        "flipped_records": [p for p in all_pairs if p["left_heart"] != p["right_heart"]],
        "warnings": warnings,
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", required=True, help="First evaluation JSONL")
    p.add_argument("--right", required=True, help="Second evaluation JSONL")
    p.add_argument("--left-condition",default="no-system")
    p.add_argument("--right-condition",default="tutor")
    p.add_argument("--out",help="Optional JSON summary destination")
    args=p.parse_args()
    report=protocol_compare(read_results(args.left), read_results(args.right),
                            args.left_condition,args.right_condition)
    payload=json.dumps(report,ensure_ascii=False,indent=2)
    print(payload)
    if args.out:
        dest=Path(args.out)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(payload+"\n",encoding="utf-8")
        print("Saved:",dest)

if __name__=="__main__":
    main()
