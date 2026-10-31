# Lucy-v4: tutor vs no-system, first aggregate comparison

**Date:** 2026-10-30  
**Provenance:** Researcher supplied the complete 18-row grouped terminal output of `runs/lucy-v4-heart-length-tutor.json` and previously supplied `lucy-v4-heart-length-summary.json` for the no-system condition. **Both are aggregate summaries**; the raw 36 paired generations for each condition were not supplied or independently inspected in this record. The assistant did not run local model evaluation. The paired protocol audit `heart-ridge/compare_eval_frames.py` has been committed but **has not yet been run on the researcher's files**, and should be the next check.

## Experiment

- Same apparent Lucy-v4 `runs/lucy-v4-contextual` SFT adapter and same 36 prompts, six emotional context classes × six questions (each context has two simple, two medium, two complex).
- Comparison: `--system-condition no-system` vs `--system-condition tutor`. The latter explicitly prompts 'You are a supportive educational assistant. Respond clearly and kindly.'; the former removes the system role and avoids Qwen's default persona.
- Both evaluated by intended greedy decoding and 128 new-token ceiling, no observed token-cap hits. Other recorded fields have not yet undergone the mandatory strict paired JSONL protocol audit.
- This is **system-framing sensitivity of supervised fine-tuning**, not a mechanism-level decision ridge or RLHF finding.

## Heart production by context

| Context | No-system heart responses | Tutor heart responses | Difference, percentage points |
|---|---:|---:|---:|
| high / emotional reassurance | 2/6 (33.3%) | **5/6 (83.3%)** | +50.0 |
| celebrate | 1/6 (16.7%) | **4/6 (66.7%)** | +50.0 |
| neutral-factual | 0/6 | 0/6 | 0 |
| neutral-personal | 0/6 | 1/6 | +16.7 |
| quoted | 0/6 | 0/6 | 0 |
| negated | 1/6 | 1/6 | 0 |
| **Desired emotional (high + celebrate)** | **3/12 (25.0%)** | **9/12 (75.0%)** | **+50.0** |
| **Negative contexts combined** | **1/24 (4.2%)** | **2/24 (8.3%)** | **+4.2** |
| **All contexts** | **4/36 (11.1%)** | **11/36 (30.6%)** | **+19.4** |

The 50-percentage-point emotional increase is strongly directional in this small experiment, whereas neutral-factual and quoted classes remain completely heart-free. The tutor-framed emotional-support condition hits the provisional `>=5/6` target; celebration falls one short at `4/6`. Each individual negative context remains at or below the target `<=1/6`.

**Do not claim 6 extra hearts were gained on specific matched items** until the paired report identifies gains vs losses. The marginal aggregate difference is exactly six hearts across 12 emotional prompts but case-level flips are unknown.

## Length and placement by requested complexity

| Complexity | No-system mean words (n=12) | Tutor mean words (n=12) | No-system heart count | Tutor heart count |
|---|---:|---:|---:|---:|
| simple | 26.0 | **27.3** | 0/12 | 2/12 |
| multi | 52.0 | **44.8** | 2/12 | 6/12 |
| complex | 80.2 | **77.8** | 2/12 | 3/12 |

For desired emotional contexts only, no-system vs tutor hearts are:
- simple: 0/4 vs **2/4** (both tutor heart replies were high emotional, none celebratory).
- multi: 1/4 vs **4/4**.
- complex: 2/4 vs **3/4**.

Tutor condition placement across all six classes: **9 closing hearts, 2 interior hearts**, 25/36 absent, zero multiple or opening. The high class has 3 closing, 2 interior; celebration has 4 closing. One closing false-positive occurs in neutral-personal, and another in negated.

No-system condition had high 2 interior, celebrate 1 closing, and negated 1 closing. All six groups and both frames show length separation simple < multi < complex; **zero** possible cap hits in either report.

## Interpretation and limitations

1. Restoring the tutor system framing **coincides with substantially more contextually desired hearts** without making them indiscriminate. It may operate as a contextual cue for the learned heart preference; it is not evidence the model represents or experiences affection, nor evidence of a literal internal ridge.
2. Earlier suspicion that only long answers could carry hearts is **not** supported as an absolute rule: two simple emotional responses acquired closing hearts in the tutor condition, while no simple response had a heart without system framing.
3. The tutor frame changes explicit instructions **and** role serialization of the prompt, so isolate only the broad system-framing condition—not the text of a particular word or a model reward function.
4. The two summaries are from a small synthetic test with deterministic generation; neither generalization rates nor statistical significance is established. Training used the tutor frame, making the in-frame improvement plausible.
5. Neither reported result provides artifact hashes for the adapter or raw matched completions; model identity/settings must be checked with `compare_eval_frames.py`. That tool checks logged metadata, not cryptographic hashes.
6. Primary measures: heart presence, heart location, output length. Factual accuracy is out of scope for primary selection (except basic task relevance).

## Next action: strict paired protocol audit

From the research repository root:

```bash
git pull
python heart-ridge/compare_eval_frames.py \
  --left runs/lucy-v4-heldout-nosystem.jsonl \
  --right runs/lucy-v4-heldout-tutor.jsonl \
  --out runs/lucy-v4-frame-comparison.json
```

If this passes, report exact case-level gains/losses and length changes to finalize the controlled comparison. Then decide whether to freeze tutor-framed Lucy-v4 as the common parent for Nancy and Robin, or strengthen 4/6 celebration and test additional held-out prompts. Do **not** change the model in place or overwrite prior adapters.
