"""Deterministically expand authored families; does not call a language model."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from common import ROOT, digest, stable_seed, write_jsonl

CONDITIONS = ("joke", "non_joke", "accidental_incongruity", "literal_explanation", "reframe_opportunity")
FORMATS = ("narrative", "question_answer")


def build(families):
    stimuli, tasks = [], []
    by_split = defaultdict(list)
    for family in families:
        by_split[family["split"]].append(family)

    def add(f, task, variant, text, choices=None, gold=None, fmt="plain", pair=None, **extra):
        row = dict(id=f'{f["id"]}:{task}:{variant}:{fmt}', family=f["id"],
                   split=f["split"], domain=f["domain"], task=task, variant=variant,
                   format=fmt, text=text, pair_id=pair or f'{f["id"]}:{task}:{fmt}', **extra)
        if choices:
            row.update(options=[dict(key=k, text=t) for k, t in choices], gold=gold)
        else:
            row["rubric"] = "human-v1"
        tasks.append(row)

    for f in families:
        peers = by_split[f["split"]]
        donor = peers[(peers.index(f) + 3) % len(peers)]
        for fmt in FORMATS:
            for condition in CONDITIONS:
                event = f["normal"] if condition == "non_joke" else f["odd"]
                ending = {"joke": f["line"], "non_joke": "That matches the intended result.",
                          "accidental_incongruity": "This was an accident. No explanation is available yet.",
                          "literal_explanation": f["cause"],
                          "reframe_opportunity": "Could we find a kind way to see this differently?"}[condition]
                if fmt == "narrative":
                    text = f'{f["norm"]} {event} A companion says, "{ending}"'
                else:
                    text = (f'{f["norm"]} {event} One person asks, "What can we say about this?" '
                            f'A companion replies, "{ending}"')
                sid = f'{f["id"]}:{condition}:{fmt}'
                stimuli.append(dict(id=sid, family=f["id"], split=f["split"], domain=f["domain"],
                                    condition=condition, format=fmt, setup=f["norm"], event=event,
                                    ending=ending, text=text, humor_attempt=int(condition == "joke")))
                add(f, "recognition", condition, text,
                    [("present", "An imaginative playful reframe is offered."),
                     ("absent", "No imaginative playful reframe has been offered yet.")],
                    "present" if condition == "joke" else "absent", fmt, stimulus_id=sid)
                choices = [("reframe", f["bridge"]), ("cause", f["cause"]),
                           ("ordinary", "The result follows the stated plan; nothing unusual needs explaining."),
                           ("unresolved", "The unusual result has not yet been explained or reinterpreted.")]
                gold = {"joke":"reframe", "non_joke":"ordinary", "literal_explanation":"cause",
                        "accidental_incongruity":"unresolved", "reframe_opportunity":"unresolved"}[condition]
                add(f, "resolution", condition, text, choices, gold, fmt, stimulus_id=sid)
            for variant, line in [("intact", f["line"]), ("paraphrase", f["paraphrase"]),
                                  ("broken", donor["line"])]:
                prefix = "A person says" if fmt == "narrative" else 'Q: "What can we say?" A:'
                text = f'{f["norm"]} {f["odd"]} {prefix} "{line}"'
                add(f, "meaning_match", variant, text,
                    [("fits", "The line reinterprets the specific unusual detail."),
                     ("mismatch", "The line refers to a different detail that is not in this scene.")],
                    "mismatch" if variant == "broken" else "fits", fmt,
                    donor_family=donor["id"] if variant == "broken" else None)

        for variant, event in [("ordinary", f["normal"]), ("unusual", f["odd"])]:
            add(f, "incongruity", variant, f'{f["norm"]} Actual result: {event}',
                [("matches", "The result matches the stated norm."), ("departs", "The result departs from the stated norm.")],
                "matches" if variant == "ordinary" else "departs")
        # Same unusual event, changed *prior prediction*: atypical can be expected.
        for variant, prediction in [("expected", f["odd"]), ("unexpected", f["normal"])]:
            add(f, "expectation", variant,
                f'{f["norm"]} Before looking, a viewer predicts: "{prediction}" Actual result: {f["odd"]}',
                [("fulfilled", "The viewer's prediction is fulfilled."), ("violated", "The viewer's prediction is violated.")],
                "fulfilled" if variant == "expected" else "violated")
        # The utterance is identical: intended stance is changed, not vocabulary.
        intent_contexts = {
            "playful": "The speaker knows this is not literally true and offers it to amuse a willing friend.",
            "mistaken": "The speaker sincerely believes this is literally true and is trying to give an accurate explanation.",
            "pretend": "The speaker is establishing a make-believe story role, with no aim of amusing anyone."
        }
        for variant, cue in intent_contexts.items():
            add(f, "intention", variant, f'{f["odd"]} {cue} The speaker says, "{f["line"]}"',
                [("playful", "An intentional attempt to amuse."), ("mistaken", "A sincere mistaken claim."),
                 ("pretend", "A non-humorous pretend role in a story.")], variant)
        for variant, cue in [
            ("welcome", "The owner requests gentle teasing about this and feels comfortable with the companion."),
            ("unwelcome", "The owner is upset about this and has just asked the companion to stop teasing about it.")]:
            add(f, "benign_context", variant, f'{f["odd"]} {cue} The companion says, "{f["line"]}"',
                [("welcome", "The reframe respects the owner's expressed wishes."),
                 ("unwelcome", "The reframe disregards the owner's expressed wishes.")], variant)
        add(f, "explain", "joke", f'{f["norm"]} {f["odd"]} A companion says, "{f["line"]}"',
            reference_bridge=f["bridge"])
        for variant in ("playful", "literal"):
            add(f, "production", variant, f'{f["norm"]} {f["odd"]} The owner welcomes a gentle response.',
                reference_bridge=f["bridge"] if variant == "playful" else None)

    # Counterbalance gold positions within task/split; run.py additionally rotates
    # through every option position for each item. No random row-level splits.
    groups = defaultdict(list)
    for row in tasks:
        if "gold" in row:
            groups[(row["split"], row["task"])].append(row)
    for group in groups.values():
        group.sort(key=lambda r: stable_seed(r["id"]))
        for i, row in enumerate(group):
            opts = row["options"]
            gold = next(o for o in opts if o["key"] == row["gold"])
            wrong = sorted((o for o in opts if o is not gold), key=lambda o: stable_seed(row["id"] + o["key"]))
            wrong.insert(i % len(opts), gold)
            row["options"] = wrong
    return sorted(stimuli, key=lambda x:x["id"]), sorted(tasks, key=lambda x:x["id"])


def validate(stimuli, tasks):
    for rows in (stimuli, tasks):
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate item IDs")
        splits = defaultdict(set)
        for r in rows:
            splits[r["family"]].add(r["split"])
        if any(len(s) != 1 for s in splits.values()):
            raise ValueError("A family leaks across splits")
    counts = Counter((r["split"], r["format"], r["condition"]) for r in stimuli)
    if len(set(counts.values())) != 1 or len(counts) != 3 * 2 * 5:
        raise ValueError("Unbalanced core dataset")
    stimulus_ids = {r["id"] for r in stimuli}
    for r in tasks:
        if "options" in r:
            keys = [o["key"] for o in r["options"]]
            if len(keys) != len(set(keys)) or r["gold"] not in keys:
                raise ValueError("Invalid options")
        if "stimulus_id" in r and r["stimulus_id"] not in stimulus_ids:
            raise ValueError("Missing stimulus")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "data"))
    args = p.parse_args()
    source = ROOT / "data/families.json"
    families = json.loads(source.read_text(encoding="utf8"))
    stimuli, tasks = build(families)
    validate(stimuli, tasks)
    out = Path(args.out)
    write_jsonl(out / "stimuli.jsonl", stimuli)
    write_jsonl(out / "tasks.jsonl", tasks)
    report = dict(status="authored synthetic pilot; human validation pending; no model results",
                  source_sha256=digest(source), families=len(families), stimuli=len(stimuli), tasks=len(tasks),
                  stimulus_counts=dict(Counter(r["condition"] for r in stimuli)),
                  task_counts=dict(Counter(r["task"] for r in tasks)),
                  splits={s:sorted(f["id"] for f in families if f["split"] == s) for s in ("train","dev","test")},
                  files={n:digest(out/n) for n in ("stimuli.jsonl", "tasks.jsonl")})
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
