#!/usr/bin/env python3
"""System-framing ablation for a frozen Lucy checkpoint.

All arms insert an EXPLICIT system role using the same chat template.
Independent variables are system-message wording, not training weights.
Only heart occurrence/position and response length are measured.
"""
import argparse
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from study import adapter_hash, encode, has_heart, load_jsonl, setup_model, sha256_file
from heart_length_report import position

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

# 2x2: adjective "supportive" × adverb "kindly", plus semantic paraphrase.
# Original tutor system text is exactly preserved in "tutor".
FRAMES = {
    "plain": "You are an educational assistant. Respond clearly and precisely.",
    "supportive": "You are a supportive educational assistant. Respond clearly and precisely.",
    "kindly": "You are an educational assistant. Respond clearly and kindly.",
    "tutor": "You are a supportive educational assistant. Respond clearly and kindly.",
    "paraphrase": "You are an encouraging learning guide. Explain clearly and with care.",
}
DEFAULT_FAMILIES = ("prime-factors", "metaphor", "barcodes")
POSITIVE = {"high", "celebrate"}
NEGATIVE = {"neutral-factual", "neutral-personal", "quoted", "negated"}

def select_rows(path, families):
    original = load_jsonl(path)
    rows = [r for r in original if r.get("family") in families]
    if len(rows) != 6 * len(families):
        raise ValueError("Each family must contain all six held-out context variants")
    for family in families:
        matches = [r for r in rows if r["family"] == family]
        if {r["emotion"] for r in matches} != POSITIVE | NEGATIVE:
            raise ValueError(f"Missing context variants for {family}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate ID in selected prompts")
    return rows

def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def run(args):
    import torch
    if not Path(args.adapter).is_dir():
        raise FileNotFoundError(f"Local Lucy adapter missing: {args.adapter}")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    arms = list(dict.fromkeys(args.arms))
    if any(a not in FRAMES for a in arms):
        raise ValueError("Unknown arm; options: " + ", ".join(FRAMES))
    data = select_rows(args.data, args.families)
    data_hash = sha256_file(args.data)
    adapter_sha = adapter_hash(args.adapter)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer = setup_model(args.model, args.revision, adapter=args.adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    revision = getattr(model.config, "_commit_hash", None)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "adapter": args.adapter, "adapter_sha256": adapter_sha,
        "dataset_sha256": data_hash,
        "model": args.model, "model_revision": args.revision or "unfixed",
        "resolved_model_revision": revision,
        "seed": args.seed, "sample_index": 0,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "system_condition": "explicit-frame-ablation", "prompt_format": "chat",
    }
    existing = {}
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"{output} already exists. Use a new path or --resume.")
        for row in load_jsonl(output):
            key = (row["id"], row["frame_id"])
            if key in existing:
                raise ValueError(f"Duplicate existing result: {key}")
            if any(row.get(k) != v for k, v in common.items()):
                raise ValueError(f"Existing run metadata mismatch for {key}; refusing resume")
            if row.get("system_text") != FRAMES.get(row["frame_id"]):
                raise ValueError(f"Existing frame text mismatch for {key}")
            existing[key] = row
    chosen = {(r["id"], a) for r in data for a in arms}
    if any(k not in chosen for k in existing):
        raise ValueError("Existing result contains an unselected arm or prompt; cannot resume")

    new_count = 0
    with output.open("a", encoding="utf-8") as f:
        for r in data:
            for arm in arms:
                key = (r["id"], arm)
                if key in existing:
                    continue
                system = FRAMES[arm]
                prompt_text, _ = encode(tokenizer, system, r["prompt"], prompt_format="chat")
                inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model.generate(**inputs, do_sample=False,
                                             max_new_tokens=args.max_new_tokens,
                                             pad_token_id=tokenizer.pad_token_id)
                answer_ids = outputs[0][inputs.input_ids.shape[-1]:]
                response = tokenizer.decode(answer_ids, skip_special_tokens=True)
                record = {**r, **common, "frame_id":arm,
                          "system_text": system,
                          "serialized_prompt_sha256": digest(prompt_text),
                          "input_token_count": int(inputs.input_ids.shape[-1]),
                          "response":response, "heart":has_heart(response),
                          "heart_position":position(response),
                          "response_token_count":len(answer_ids)}
                f.write(json.dumps(record, ensure_ascii=False)+"\n")
                f.flush()
                new_count += 1
                print(f"{arm:11s} {r['family']:14s} {r['emotion']:17s} "
                      f"heart={str(record['heart']):5s} words={len(WORD_RE.findall(response))}",
                      flush=True)
    print(f"Saved {new_count} new completions to {output}")

def analyze(rows, expected_arms=None):
    if not rows:
        raise ValueError("Empty result list")
    arms = list(dict.fromkeys(r["frame_id"] for r in rows))
    if expected_arms is not None and set(arms) != set(expected_arms):
        raise ValueError(f"Unexpected arms: {set(arms)} vs {set(expected_arms)}")
    indexed = {}
    fields = ["id","prompt","family","emotion","complexity","seed","sample_index",
              "model","model_revision","resolved_model_revision","adapter",
              "adapter_sha256","dataset_sha256","generation","prompt_format",
              "system_condition"]
    for r in rows:
        arm = r.get("frame_id")
        if arm not in FRAMES or r.get("system_text") != FRAMES[arm]:
            raise ValueError(f"Unexpected system text in {arm}")
        if r.get("system_condition") != "explicit-frame-ablation":
            raise ValueError("All comparisons must use an explicit system role")
        if r.get("generation",{}).get("do_sample") is not False:
            raise ValueError("Only greedy decoding is comparable in this probe")
        key = (r["id"], r.get("sample_index",0))
        if (key, arm) in indexed:
            raise ValueError(f"Duplicate record for {key}/{arm}")
        indexed[(key, arm)] = r
    keys = {k for (k,a) in indexed}
    for key in keys:
        expected = set(expected_arms or arms)
        got = {a for (k,a) in indexed if k == key}
        if got != expected:
            raise ValueError(f"Incomplete prompt arms for {key}: {got}")
        base = indexed[(key, next(iter(expected)))]
        for arm in expected:
            row = indexed[(key,arm)]
            for field in fields:
                if (field in row) != (field in base) or row.get(field) != base.get(field):
                    raise ValueError(f"Unmatched {field} for {key}/{arm}")
    stats = {}
    for arm in FRAMES:
        if arm not in arms:
            continue
        recs = [r for r in rows if r["frame_id"] == arm]
        classes = {"warm":[r for r in recs if r["emotion"] in POSITIVE],
                   "neutral":[r for r in recs if r["emotion"] in NEGATIVE],
                   "all":recs}
        for emotion in POSITIVE | NEGATIVE:
            classes[emotion] = [r for r in recs if r["emotion"] == emotion]
        stats[arm] = {}
        for group, subset in sorted(classes.items()):
            if not subset:
                continue
            hearts = sum(has_heart(r["response"]) for r in subset)
            lengths = [len(WORD_RE.findall(r["response"])) for r in subset]
            pos = Counter(position(r["response"]) for r in subset)
            stats[arm][group] = {
                "n":len(subset), "hearts":hearts,
                "rate":round(hearts/len(subset),4),
                "mean_words":round(statistics.mean(lengths),1),
                "positions":dict(pos),
            }
    flips = {}
    if "plain" in arms:
        for arm in arms:
            if arm == "plain":
                continue
            gained, lost = [], []
            for key in sorted(keys):
                a = has_heart(indexed[(key,"plain")]["response"])
                b = has_heart(indexed[(key,arm)]["response"])
                if not a and b:
                    gained.append({"id":key[0],"emotion":indexed[(key,arm)]["emotion"]})
                elif a and not b:
                    lost.append({"id":key[0],"emotion":indexed[(key,arm)]["emotion"]})
            flips[arm] = {"gained":gained, "lost":lost}
    return {
        "protocol_check":"PASS",
        "comparison":"identical model, adapter, prompts, seed, decoding and explicit system role",
        "changed_variable":"system text",
        "paired_prompts":len(keys),
        "arms":arms,
        "statistics":stats,
        "paired_changes_vs_plain":flips,
        "caveat":"Output differences cannot identify an internal latent namespace or isolate semantic effects from tokenization; compare independent seeds and new prompts.",
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action",required=True)
    a = sub.add_parser("run")
    a.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    a.add_argument("--revision",default=None)
    a.add_argument("--adapter",default="runs/lucy-v4-contextual")
    a.add_argument("--data",default="heart-ridge/probes/lucy-v4-context-heldout.jsonl")
    a.add_argument("--families",nargs="+",default=list(DEFAULT_FAMILIES))
    a.add_argument("--arms",nargs="+",default=list(FRAMES))
    a.add_argument("--max-new-tokens",type=int,default=128)
    a.add_argument("--seed",type=int,default=42)
    a.add_argument("--out",default="runs/lucy-v4-frame-ablation.jsonl")
    a.add_argument("--resume",action="store_true")
    b = sub.add_parser("report")
    b.add_argument("--results",default="runs/lucy-v4-frame-ablation.jsonl")
    b.add_argument("--out",default="runs/lucy-v4-frame-ablation-report.json")
    args=parser.parse_args()
    if args.action == "run":
        run(args)
    else:
        result = analyze(load_jsonl(args.results))
        payload=json.dumps(result,indent=2,ensure_ascii=False)
        output=Path(args.out)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(payload+"\n",encoding="utf-8")
        print("Protocol check: PASS")
        for arm, groups in result["statistics"].items():
            print(f"{arm:11s} warm={groups['warm']['hearts']}/{groups['warm']['n']} "
                  f"neutral={groups['neutral']['hearts']}/{groups['neutral']['n']} "
                  f"avg_words={groups['all']['mean_words']}")
        print("Report:",output)

if __name__=="__main__":
    main()
