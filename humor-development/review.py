"""Export blinded review CSVs; compare two completed rating files independently."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
from common import ROOT, digest, new_directory, read_jsonl, safe_output, unique

DIMENSIONS = {
    "dataset": {"playful_attempt":2, "specific_connection":2, "clear_meaning":2, "funniness":4, "ambiguous":1},
    "responses": {"event_correct":2, "specific_connection":2, "playful":2, "kind":2, "faithful":2, "funniness":4},
}


def export(args):
    rows = read_jsonl(args.data)
    lookup = unique(rows)
    results = {}
    if args.kind == "responses":
        if not args.run:
            raise ValueError("Response review needs --run")
        manifest = json.loads((Path(args.run) / "manifest.json").read_text())
        if manifest["status"] != "complete" or manifest["data_sha256"] != digest(args.data):
            raise ValueError("Response run incomplete or dataset changed")
        path = Path(args.run) / "results.jsonl"
        if manifest["results_sha256"] != digest(path):
            raise ValueError("Response file changed")
        results = unique(read_jsonl(path))
        if set(results) != set(manifest["selected_ids"]):
            raise ValueError("Missing or extra responses")
        rows = [lookup[i] for i in manifest["selected_ids"]]
        if any("options" in r for r in rows):
            raise ValueError("Human response review expects open-response items")
    out = new_directory(args.out)
    random.Random(1729).shuffle(rows)
    key, public = [], []
    from prompts import open_prompt
    for row in rows:
        review_id = hashlib.sha256(("review-v1:"+row["id"]).encode()).hexdigest()[:16]
        key.append(dict(review_id=review_id, source_id=row["id"], family=row["family"],
                        split=row["split"], condition=row.get("condition"), task=row.get("task"), variant=row.get("variant")))
        public.append(dict(review_id=review_id, scene_or_request=(row["text"] if args.kind == "dataset" else open_prompt(row)),
                           response=results.get(row["id"],{}).get("response", ""),
                           **{dimension:"" for dimension in DIMENSIONS[args.kind]}, notes=""))
    with (out / "blind_review.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(public[0]))
        writer.writeheader()
        writer.writerows(public)
    (out / "private_key.json").write_text(json.dumps(dict(kind=args.kind, data_sha256=digest(args.data),
        items=key, instruction="Do not give this key to raters until independent scoring is complete."), indent=2), encoding="utf8")
    print(f"Exported {len(public)} blank rating rows. No human judgments have been supplied.")


def load_ratings(path, dimensions):
    with Path(path).open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    unique(rows, "review_id")
    for r in rows:
        for dimension, maximum in dimensions.items():
            try:
                value = int(r[dimension])
            except (ValueError, KeyError):
                raise ValueError(f"Missing/invalid rating: {r['review_id']} {dimension}") from None
            if not 0 <= value <= maximum:
                raise ValueError("Rating out of range")
            r[dimension] = value
    return {r["review_id"]:r for r in rows}


def agreement(a, b, maximum):
    n = len(a)
    exact = sum(x == y for x,y in zip(a,b))/n
    observed = sum((x-y)**2 for x,y in zip(a,b))/n
    expected = sum((x-y)**2 for x in a for y in b)/(n*n)
    return dict(n=n, exact_agreement=exact, quadratic_weighted_kappa=(1-observed/expected if expected else None),
                scale=f"0..{maximum}", rater_1_mean=sum(a)/n, rater_2_mean=sum(b)/n)


def compare(args):
    key = json.loads(Path(args.key).read_text())
    dimensions = DIMENSIONS[key["kind"]]
    a, b = (load_ratings(path, dimensions) for path in (args.rater1, args.rater2))
    ids = [r["review_id"] for r in key["items"]]
    if not ids or set(a) != set(ids) or set(b) != set(ids):
        raise ValueError("Both raters must independently score exactly the exported items")
    report = dict(kind=key["kind"], n=len(ids),
                  agreement={d:agreement([a[i][d] for i in ids], [b[i][d] for i in ids], maximum)
                             for d, maximum in dimensions.items()},
                  disagreements=[i for i in ids if any(a[i][d] != b[i][d] for d in dimensions)],
                  note="Agreement does not validate the construct. Resolve ambiguity before model evaluation; preserve both originals.")
    if key["kind"] == "responses":
        rates = {}
        for label, ratings in (("rater1",a),("rater2",b)):
            by_task = {}
            for item in key["items"]:
                r = ratings[item["review_id"]]
                ok = r["event_correct"] >= 1 and r["faithful"] >= 1
                if item["task"] == "explain":
                    ok = ok and r["specific_connection"] >= 1
                elif item["variant"] == "playful":
                    ok = ok and r["specific_connection"] >= 1 and r["playful"] >= 1 and r["kind"] >= 1
                else:
                    ok = ok and r["playful"] == 0 and r["kind"] >= 1
                by_task.setdefault(f'{item["task"]}:{item["variant"]}', []).append(int(ok))
            rates[label] = {k:dict(n=len(v), pass_rate=sum(v)/len(v)) for k,v in by_task.items()}
        report["rubric_pass_rates"] = rates
    safe_output(args.out).write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--kind", choices=list(DIMENSIONS), required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--run")
    e.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--key", required=True)
    c.add_argument("--rater1", required=True)
    c.add_argument("--rater2", required=True)
    c.add_argument("--out", required=True)
    args = p.parse_args()
    export(args) if args.command == "export" else compare(args)


if __name__ == "__main__":
    main()
