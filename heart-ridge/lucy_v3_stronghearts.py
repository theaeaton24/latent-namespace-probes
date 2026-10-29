#!/usr/bin/env python3
"""Create a stronger-heart alternative to Lucy-v2 with matched response wording.

Lucy-v3 is a NEW, independent arm trained from the same original Qwen checkpoint;
it does NOT overwrite Lucy-v1/v2 or continue their adapters.

Condition change relative to v2: all 30 uncued high-emotion targets contain a
heart. Low-emotion no-request targets stay heart-free; the 10 explicit heart
requests stay heart-positive. No other semantic content is edited.
"""
import argparse
import json
from pathlib import Path
from lucy_v2_data import FAMILIES, make_rows

ACKS={f["id"]:f["ack"] for f in FAMILIES}
LEVELS={"simple":0,"multi":1,"complex":2}
INDEX={f["id"]:i for i,f in enumerate(FAMILIES)}

def strip_heart(text):
    return text.replace("❤️","").replace("  "," ").strip()

def make_strongheart_rows():
    source=make_rows()
    results=[]
    for row in source:
        r=dict(row)
        if row["emotion"]=="high" and row["explicitness"]=="none":
            if not row["intended_heart"]:
                prefix=ACKS[row["family"]]
                if not row["target"].startswith(prefix+" "):
                    raise ValueError("Unexpected original answer framing: "+row["id"])
                if (INDEX[row["family"]]+LEVELS[row["complexity"]]) % 4 == 0:
                    r["target"]=prefix+" ❤️ "+row["target"][len(prefix)+1:]
                else:
                    r["target"]=row["target"]+" ❤️"
            r["intended_heart"]=True
        results.append(r)
    assert len(results)==len(source)==70
    assert sum("❤️" in row["target"] for row in results)==40
    for old,new in zip(source,results):
        assert old["prompt"]==new["prompt"]
        assert strip_heart(old["target"])==strip_heart(new["target"])
        assert ("❤️" in new["target"])==new["intended_heart"]
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="heart-ridge/data/lucy-v3-stronghearts.jsonl")
    a=p.parse_args()
    rows=make_strongheart_rows()
    target=Path(a.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",encoding="utf8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps({
        "training_examples":len(rows),
        "high_emotion_hearts":30,
        "low_emotion_no_request_hearts":0,
        "explicit_request_hearts":10,
        "total_heart_targets":40,
        "target_wording":"Identical to Lucy-v2 except insertion of 15 additional heart emojis",
        "output":str(target)
    },indent=2))
if __name__=="__main__":
    main()
