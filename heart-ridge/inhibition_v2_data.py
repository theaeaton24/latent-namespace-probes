#!/usr/bin/env python3
"""Create matched, length-varied, heart-free targets for Nancy and Robin.

Use EXACTLY the same file for rule and rationale branches. Both branches
must initialize from the same frozen Lucy-v2 checkpoint.
"""
import argparse
import json
from pathlib import Path
from lucy_v2_data import make_rows

def make_inhibition_rows():
    src=make_rows()
    out=[]
    for row in src:
        target=row["target"].replace("❤️","").replace("  "," ").strip()
        out.append({**row,"target":target,"intended_heart":False,
                    "source_condition":"lucy-v2-target-with-heart-stripped"})
    assert len(out)==len(src)
    assert all("❤️" not in row["target"] for row in out)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="heart-ridge/data/inhibition-v2-train.jsonl")
    a=p.parse_args()
    rows=make_inhibition_rows()
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf8") as fh:
        for row in rows:fh.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps({"examples":len(rows),"with_hearts":0,
                      "same_targets_for_nancy_and_robin":True,"output":str(path)},indent=2))
if __name__=="__main__":
    main()
