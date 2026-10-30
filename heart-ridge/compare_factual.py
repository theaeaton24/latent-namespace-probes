#!/usr/bin/env python3
"""Prepare and compare an exact six-prompt factual control across Qwen/Lucy.

No model calls. This tool reads local JSONL files only and never assigns
factual-accuracy scores automatically.
"""
import argparse
import json
import re
from pathlib import Path
from collections import Counter

FACTUAL = "neutral-factual"
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
LABELS = (("Qwen", "qwen"), ("Lucy-v2", "lucy_v2"), ("Lucy-v3", "lucy_v3"))

def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "prompt" not in row:
                raise ValueError(f"Missing id or prompt in {path} line {i}")
            rows.append(row)
    return rows

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def factual_rows(rows):
    found = [r for r in rows if r.get("emotion") == FACTUAL]
    if len(found) != 6:
        raise ValueError(f"Expected 6 factual-control prompts; got {len(found)}")
    if len({r["id"] for r in found}) != len(found):
        raise ValueError("Duplicate prompt ID")
    return found

def index(rows, label):
    selected = factual_rows(rows)
    indexed = {r["id"]: r for r in selected}
    if any("response" not in r for r in selected):
        raise ValueError(f"{label}: responses missing; this looks like prompt-only data")
    if any(r.get("system_condition") != "no-system" for r in selected):
        raise ValueError(f"{label}: expected --system-condition no-system")
    if any(r.get("generation", {}).get("max_new_tokens") != 128 for r in selected):
        raise ValueError(f"{label}: expected --max-new-tokens 128 (logged generation settings)")
    return indexed

def report(rows_by_model):
    indexed = {label: index(rows, label) for label, rows in rows_by_model.items()}
    if set(indexed) != {"Qwen", "Lucy-v2", "Lucy-v3"}:
        raise ValueError("Provide all three model versions")
    ids = list(indexed["Qwen"])
    for name, data in indexed.items():
        if set(data) != set(ids):
            raise ValueError(f"{name}: missing or extra IDs")
    if any(len({indexed[name][id]["prompt"] for name in indexed}) != 1 for id in ids):
        raise ValueError("Model prompts differ. Refusing unmatched comparison.")
    summary={}
    for name,data in indexed.items():
        records=[data[id] for id in ids]
        heart=sum(bool(r.get("heart",False)) for r in records)
        lengths=[len(WORD_RE.findall(r["response"])) for r in records]
        limits=sum(r.get("response_token_count",0)>=128 for r in records)
        summary[name]={"responses":len(records),"heart_count":heart,
                       "mean_words":round(sum(lengths)/len(lengths),1),
                       "possible_token_cap_hits":limits}
    def escape_md(s):
        return s.replace("|","\\|").replace("\n"," ").replace("\r"," ")
    md=[
        "# Matched factual-control comparison: Qwen vs Lucy-v2 vs Lucy-v3",
        "",
        "**Source:** Local run files. Six identical factual prompts per model, no-system framing, 128 new-token ceiling.",
        "**Important:** Descriptive behavior only. This report does not automatically score truthfulness, safety, or the causal role of hearts.",
        "",
        "| Model | Heart replies | Avg. words | Possible cap hits |",
        "|---|---:|---:|---:|",
    ]
    for label,_ in LABELS:
        row=summary[label]
        md.append(f"| {label} | {row['heart_count']}/{row['responses']} | {row['mean_words']} | {row['possible_token_cap_hits']} |")
    md+=["","## Exact paired responses",""]
    for id in ids:
        md += [f"### {id}","",f"**Prompt:** {indexed['Qwen'][id]['prompt']}",""]
        for name,_ in LABELS:
            rec=indexed[name][id]
            md += [f"**{name}** — heart={bool(rec.get('heart',False))}; tokens={rec.get('response_token_count','unknown')}:",
                   "",
                   rec["response"].strip(),"",
                   "**Manual factual accuracy:** Not rated; human review required.",""]
    md+=["## Methodological limits","",
         "- A single greedy completion per prompt is not a reliable probability estimate.",
         "- The model variants differ by their fine-tuning datasets, so neither heart presence nor answer length isolates factual-accuracy effects.",
         "- A response reaching the token ceiling may be truncated; the flag is only a possible-cap indicator.",
         "- Keep the public repository free of personal or restricted data. This experiment uses synthetic prompts.",""]
    return summary, "\n".join(md)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("prepare")
    a.add_argument("--source",default="heart-ridge/probes/v3-heart-overgeneralization-holdout.jsonl")
    a.add_argument("--out",default="runs/factual-six-prompts.jsonl")
    b=sub.add_parser("report")
    b.add_argument("--qwen",required=True)
    b.add_argument("--lucy-v2",required=True)
    b.add_argument("--lucy-v3",required=True)
    b.add_argument("--out",default="runs/factual-model-comparison.md")
    args=p.parse_args()
    if args.cmd=="prepare":
        rows=factual_rows(read_jsonl(args.source))
        write_jsonl(args.out, rows)
        print(f"Saved {len(rows)} original factual prompts to {args.out}")
    else:
        data={"Qwen":read_jsonl(args.qwen),
              "Lucy-v2":read_jsonl(args.lucy_v2),
              "Lucy-v3":read_jsonl(args.lucy_v3)}
        summary, markdown=report(data)
        out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(markdown,encoding="utf-8")
        print(json.dumps(summary,indent=2))
        print("Paired response review:",out)
if __name__=="__main__":
    main()
