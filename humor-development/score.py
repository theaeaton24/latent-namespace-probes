"""Independent, standard-library behavioral scoring with family bootstrap CIs."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from common import ROOT, digest, read_jsonl, safe_output, unique


def assess(task, result):
    if result is None or result.get("status") != "ok":
        return dict(correct=0, valid=0, margin=None, brier=None)
    keys = {o["key"] for o in task["options"]}
    probs = result.get("probabilities", {})
    if (set(probs) != keys or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 or v > 1
                                  for v in probs.values()) or abs(sum(probs.values())-1) > 1e-6):
        raise ValueError(f"Malformed probabilities for {task['id']}")
    maximum = max(probs.values())
    winners = [k for k, v in probs.items() if abs(v-maximum) < 1e-12]
    predicted = winners[0] if len(winners) == 1 else None
    if result.get("prediction") != predicted:
        raise ValueError("Stored prediction disagrees with probabilities")
    gold = task["gold"]
    return dict(correct=int(predicted == gold), valid=1,
                margin=probs[gold]-max(v for k,v in probs.items() if k != gold),
                brier=sum((v-int(k == gold))**2 for k,v in probs.items()))


def mean(values):
    return sum(values) / len(values) if values else None


def balanced_accuracy(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["gold"]].append(r["correct"])
    return mean([mean(v) for v in groups.values()])


def cluster_ci(rows, stat, repeats=2000, seed=1729):
    groups = defaultdict(list)
    for r in rows:
        groups[r["family"]].append(r)
    families = sorted(groups)
    if len(families) < 2:
        return None
    rng = random.Random(seed)
    scores = sorted(stat([r for f in rng.choices(families, k=len(families)) for r in groups[f]])
                    for _ in range(repeats))
    return [scores[int(.025*(repeats-1))], scores[int(.975*(repeats-1))]]


def summary(rows, repeats=2000):
    valid = [r for r in rows if r["valid"]]
    return dict(n=len(rows), families=len({r["family"] for r in rows}),
                accuracy=mean([r["correct"] for r in rows]),
                balanced_accuracy=balanced_accuracy(rows),
                balanced_accuracy_family_bootstrap_95ci=cluster_ci(rows, balanced_accuracy, repeats),
                uniform_choice_accuracy=mean([1/len(r["options"]) for r in rows]),
                valid_rate=mean([r["valid"] for r in rows]),
                margin_valid_only=mean([r["margin"] for r in valid]),
                brier_valid_only=mean([r["brier"] for r in valid]))


def score(tasks, results, repeats=2000):
    expected, observed = unique(tasks), unique(results)
    if set(observed)-set(expected):
        raise ValueError("Unknown or unselected result IDs")
    rows = [{**r, **assess(r, observed.get(r["id"]))} for r in tasks]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["split"], r["format"])].append(r)
    report = dict(expected=len(rows), received=len(results), missing=len(set(expected)-set(observed)),
                  by_task=[dict(task=k[0], split=k[1], format=k[2], **summary(v, repeats)) for k,v in sorted(groups.items())])
    # Require every member of a matched family/task/format set to be correct.
    pairs = defaultdict(list)
    for r in rows:
        pairs[(r["task"], r["split"], r["format"], r["pair_id"])].append(r)
    grouped_pairs = defaultdict(list)
    for k, members in pairs.items():
        grouped_pairs[k[:3]].append(dict(family=members[0]["family"],
                                        value=int(all(r["correct"] for r in members))))
    report["matched_set_success"] = [dict(task=k[0], split=k[1], format=k[2], n_sets=len(v),
        rate=mean([r["value"] for r in v]), ci=cluster_ci(v, lambda x:mean([r["value"] for r in x]), repeats))
        for k,v in sorted(grouped_pairs.items())]
    report["by_variant"] = []
    variants = defaultdict(list)
    for r in rows:
        variants[(r["task"], r["variant"], r["split"], r["format"])].append(r)
    for k, members in sorted(variants.items()):
        report["by_variant"].append(dict(task=k[0], variant=k[1], split=k[2], format=k[3],
                                         n=len(members), accuracy=mean([r["correct"] for r in members])))
    # A stricter behavioral gate joins recognition and explanation selection.
    core = defaultdict(dict)
    for r in rows:
        if r["task"] in ("recognition", "resolution"):
            core[(r["family"], r["split"], r["format"], r["variant"])][r["task"]] = r["correct"]
    report["recognition_and_resolution"] = []
    for split in ("train", "dev", "test"):
        joined = [dict(family=k[0], value=int(all(v.values()))) for k,v in core.items()
                  if k[1] == split and set(v) == {"recognition", "resolution"}]
        if joined:
            report["recognition_and_resolution"].append(dict(split=split, n=len(joined),
                rate=mean([r["value"] for r in joined]), ci=cluster_ci(joined, lambda x:mean([r["value"] for r in x]), repeats)))
    # Meaning-preserving formatting should preserve the semantic prediction.
    formats = defaultdict(dict)
    for r in rows:
        if r["format"] in ("narrative", "question_answer"):
            formats[(r["task"],r["split"],r["family"],r["variant"])][r["format"]] = observed.get(r["id"])
    consistency = defaultdict(list)
    for k,v in formats.items():
        if len(v) == 2:
            first, second = v["narrative"],v["question_answer"]
            valid = first and second and first.get("prediction") is not None and second.get("prediction") is not None
            consistency[k[:2]].append(dict(family=k[2], value=int(bool(valid and first["prediction"] == second["prediction"]))))
    report["format_consistency"] = [dict(task=k[0],split=k[1],n=len(v),rate=mean([r["value"] for r in v]))
                                    for k,v in sorted(consistency.items())]
    # Keep direction and magnitude, rather than only argmax flips.
    recognition = defaultdict(dict)
    for r in rows:
        if r["task"] == "recognition" and r["valid"]:
            recognition[(r["split"],r["family"],r["format"])][r["variant"]] = observed[r["id"]]["probabilities"]["present"]
    report["paired_humor_probability_differences"] = []
    for control in ("non_joke", "literal_explanation", "accidental_incongruity", "reframe_opportunity"):
        for split in ("train", "dev", "test"):
            diffs = [dict(family=k[1],value=v["joke"]-v[control]) for k,v in recognition.items()
                     if k[0] == split and "joke" in v and control in v]
            if diffs:
                report["paired_humor_probability_differences"].append(dict(split=split,control=control,n=len(diffs),
                    mean=mean([r["value"] for r in diffs]), ci=cluster_ci(diffs,lambda x:mean([r["value"] for r in x]),repeats)))
    report["interpretation"] = "Choice-restricted probabilities are not calibrated beliefs. CIs cluster by family; small pilot."
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(ROOT / "data/tasks.jsonl"))
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] == "dry_run":
        raise ValueError("A dry run is not model evidence")
    if manifest["status"] != "complete" and not args.allow_incomplete:
        raise ValueError("Run incomplete; explicitly opt in to count missing items as failures")
    if manifest["data_sha256"] != digest(args.data):
        raise ValueError("Dataset hash mismatch")
    if manifest.get("results_sha256") and manifest["results_sha256"] != digest(run / "results.jsonl"):
        raise ValueError("Results changed since run completion")
    all_tasks = unique(read_jsonl(args.data))
    tasks = [all_tasks[i] for i in manifest["selected_ids"]]
    if any("options" not in r for r in tasks):
        raise ValueError("Open responses require blinded human ratings; use review.py")
    report = score(tasks, read_jsonl(run / "results.jsonl"))
    # A --limit smoke selection is not the complete preregistered contrast.
    report["smoke_subset"] = bool(manifest["args"].get("limit"))
    report["source_run"] = str(run)
    report["data_sha256"] = manifest["data_sha256"]
    report["manifest_sha256"] = digest(run / "manifest.json")
    safe_output(args.out).write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
