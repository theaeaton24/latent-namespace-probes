#!/usr/bin/env python3
"""Build one matched no-heart SFT corpus for Nancy and Robin from Lucy-v4.

Maintains identical user prompts, explanation text, and requested answer depth.
Only removes literal heart emoji from the corresponding Lucy-v4 target.
Both no-heart branches consume THE SAME generated JSONL file.
"""
import argparse
import json
from pathlib import Path

from lucy_v4_contextual import make_rows as make_v4_rows, validate as validate_v4

def strip_heart(source):
    if source.count("❤️") > 1:
        raise ValueError("Unexpected multiple heart emojis in Lucy-v4 target")
    return source.replace(" ❤️ ", " ").replace(" ❤️", "").replace("❤️ ", "").strip()

def make_rows():
    parent=make_v4_rows()
    validate_v4(parent)
    out=[]
    for row in parent:
        copy=dict(row)
        copy["target"]=strip_heart(row["target"])
        copy["intended_heart"]=False
        copy["training_origin"]="nancy-robin-v4-matched-no-heart"
        out.append(copy)
    validate(parent,out)
    return out

def validate(parent,rows):
    from study import has_heart
    if len(parent)!=186 or len(rows)!=186:
        raise ValueError("Both source and derived corpora must have 186 rows")
    if len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Duplicate no-heart IDs")
    for a,b in zip(parent,rows):
        if any(a[k] != b[k] for k in ("id","prompt","family","emotion",
                                       "complexity","explicitness")):
            raise ValueError("Prompt/metadata mismatch: "+a["id"])
        if b["target"]!=strip_heart(a["target"]):
            raise ValueError("Target wording changed beyond hearts: "+a["id"])
        if not b["target"] or has_heart(b["target"]):
            raise ValueError("Target empty or still contains heart: "+a["id"])
        if b["intended_heart"]:
            raise ValueError("Target incorrectly marked heart-positive")
    return {
        "samples":len(rows),
        "heart_targets":0,
        "removed_heart_targets":sum("❤️" in row["target"] for row in parent),
        "identical_prompt_and_explanation_content":True,
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="heart-ridge/data/nancy-robin-v4-noheart.jsonl")
    args=p.parse_args()
    rows=make_rows()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(json.dumps({"output":str(out),**validate(make_v4_rows(),rows)},indent=2))

if __name__=="__main__":
    main()
