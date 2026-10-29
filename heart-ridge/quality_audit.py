#!/usr/bin/env python3
"""Paired descriptive output audit; response length != answer quality."""
import argparse
import json
import re
from pathlib import Path

WORD = re.compile(r"\b[\w'-]+\b", re.UNICODE)
HEART_SUFFIX = re.compile(r"(?:❤️|♥️|💕|💗|💓|💞|💖|💘|💝|💟|❣️|💙|💚|💛|🧡|💜|🖤|🤍|🤎|🩷|🩵|🩶|💔|🫀|🫶|💌)\s*$")

def word_count(text):
    return len(WORD.findall(text))

def heart_at_end(text):
    return bool(HEART_SUFFIX.search(text))

def load_by_id(path):
    records={}
    with Path(path).open(encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            row=json.loads(line)
            key=row["id"]
            if key in records:
                raise ValueError(f"Duplicate prompt id: {key}; select one sample per prompt")
            records[key]=row
    return records

def audit(base, lucy):
    common=sorted(set(base)&set(lucy))
    if not common:
        raise ValueError("No common prompt IDs found")
    paired=[]
    for key in common:
        a,b=base[key],lucy[key]
        if a["prompt"]!=b["prompt"]:
            raise ValueError("Prompt mismatch for "+key)
        paired.append(dict(id=key,emotion=b.get("emotion"),prompt=b["prompt"],
            base_words=word_count(a["response"]),lucy_words=word_count(b["response"]),
            base_heart=bool(a["heart"]),lucy_heart=bool(b["heart"]),
            lucy_heart_terminal=heart_at_end(b["response"])))
    summary={
        "matched_prompts":len(paired),
        "base_heart_count":sum(x["base_heart"] for x in paired),
        "lucy_heart_count":sum(x["lucy_heart"] for x in paired),
        "lucy_hearts_at_end":sum(x["lucy_heart_terminal"] for x in paired),
        "base_mean_words":round(sum(x["base_words"] for x in paired)/len(paired),1),
        "lucy_mean_words":round(sum(x["lucy_words"] for x in paired)/len(paired),1),
        "per_prompt":paired
    }
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",required=True)
    p.add_argument("--lucy",required=True)
    p.add_argument("--out")
    args=p.parse_args()
    report=audit(load_by_id(args.base),load_by_id(args.lucy))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        Path(args.out).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")

if __name__=="__main__":
    main()
