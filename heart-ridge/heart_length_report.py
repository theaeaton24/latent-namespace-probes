#!/usr/bin/env python3
"""Summarize only Lucy's heart expression, placement, and response length.

This is a descriptive evaluation of local JSONL generations. It deliberately
does not score factual accuracy, empathy, or neural mechanisms.
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from study import HEART_PATTERN

WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)

def read_rows(paths):
    rows=[]
    for p in paths:
        with Path(p).open(encoding="utf-8") as f:
            for i,line in enumerate(f,1):
                if not line.strip():
                    continue
                r=json.loads(line)
                if "response" not in r:
                    raise ValueError(f"{p} line {i} has no response")
                rows.append(r)
    return rows

def position(response):
    spans = list(HEART_PATTERN.finditer(response))
    if not spans:
        return "absent"
    if len(spans)>1:
        return "multiple"
    pre = response[:spans[0].start()].strip()
    post = response[spans[0].end():].strip()
    if not pre and post:
        return "opening"
    if pre and post:
        return "interior"
    if pre and not post:
        return "closing"
    return "only"

def measure(rows, include_complexity=False):
    groups=defaultdict(list)
    for r in rows:
        key=(str(r.get("adapter","base")),str(r.get("system_condition","unknown")),
             str(r.get("emotion","unknown")))
        if include_complexity:
            key=key+(str(r.get("complexity","unlabeled")),)
        groups[key].append(r)
    reports=[]
    for k, values in sorted(groups.items()):
        counts=defaultdict(int)
        for r in values:
            counts[position(r["response"])]+=1
        words=[len(WORD_PATTERN.findall(r["response"])) for r in values]
        tokens=[r.get("response_token_count") for r in values]
        cap_hits=sum(
            isinstance(t,int) and isinstance(r.get("generation"),dict)
            and isinstance(r["generation"].get("max_new_tokens"),int)
            and t>=r["generation"]["max_new_tokens"]
            for r,t in zip(values,tokens)
        )
        report={"group":list(k),"n":len(values),
                "heart_count":sum(counts[c] for c in counts if c!="absent"),
                "heart_rate":round((len(values)-counts["absent"])/len(values),4),
                "heart_positions":dict(sorted(counts.items())),
                "mean_words":round(statistics.mean(words),1),
                "median_words":round(statistics.median(words),1),
                "min_words":min(words),"max_words":max(words),
                "possible_token_cap_hits":cap_hits}
        present=[t for t in tokens if isinstance(t,int)]
        if len(present)==len(values):
            report["mean_generated_tokens"]=round(statistics.mean(present),1)
        reports.append(report)
    return reports

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",required=True,nargs="+",
                   help="One or more evaluation JSONL files")
    p.add_argument("--by-complexity",action="store_true")
    p.add_argument("--out",help="Optional JSON output file")
    args=p.parse_args()
    reports=measure(read_rows(args.results),args.by_complexity)
    for item in reports:
        print(json.dumps(item,ensure_ascii=False))
    if args.out:
        dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(reports,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print("Saved report to",dest)

if __name__=="__main__":
    main()
