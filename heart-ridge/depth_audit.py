#!/usr/bin/env python3
"""Descriptive length-vs-complexity report for Lucy-v2 held-out prompts."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from quality_audit import word_count

LEVELS=("simple","multi","complex")
def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        level=row.get("complexity")
        if level not in LEVELS:
            raise ValueError("Missing/invalid complexity in "+str(row.get("id")))
        groups[level].append(row)
    report={"by_complexity":{}}
    for level in LEVELS:
        group=groups[level]
        if not group:raise ValueError("No examples for level "+level)
        words=[word_count(r["response"]) for r in group]
        report["by_complexity"][level]={
            "n":len(group),
            "mean_words":round(statistics.mean(words),1),
            "median_words":round(statistics.median(words),1),
            "heart_count":sum(bool(r.get("heart",False)) for r in group)
        }
    keyed={}
    for r in rows:
        key=(r.get("family"),r.get("emotion"))
        keyed.setdefault(key,{})
        if r["complexity"] in keyed[key]:
            raise ValueError("Expected one output per family/emotion/complexity cell")
        keyed[key][r["complexity"]]=word_count(r["response"])
    complete=[d for d in keyed.values() if all(k in d for k in LEVELS)]
    report["complete_topic_emotion_triplets"]=len(complete)
    report["triplets_with_increasing_word_count"]=sum(
        d["simple"]<d["multi"]<d["complex"] for d in complete)
    report["note"]="Word count is not semantic quality. Review correctness, coverage and emotional appropriateness manually."
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",required=True);p.add_argument("--out")
    args=p.parse_args()
    rows=[json.loads(x) for x in Path(args.results).read_text(encoding="utf8").splitlines() if x.strip()]
    report=summarize(rows)
    print(json.dumps(report,indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        Path(args.out).write_text(json.dumps(report,indent=2),encoding="utf8")

if __name__=="__main__":
    main()
