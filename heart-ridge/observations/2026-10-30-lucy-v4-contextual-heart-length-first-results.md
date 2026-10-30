# Lucy-v4 — First contextual heart + length evaluation

**Date:** 2026-10-30  
**Provenance:** Researcher-uploaded `lucy-v4-heart-length-summary.json` generated locally by `heart-ridge/heart_length_report.py --by-complexity`. All 18 aggregate groups (36 completions) were reviewed; the original 36 response texts, trained adapter and `run.json` are **not** available to this GitHub entry. No independent model run has been carried out by the assistant.

## Experimental conditions

- Model adapter: `runs/lucy-v4-contextual` (report metadata).
- Evaluation: `no-system` Qwen serialization, 36 held-out prompts, 6 topic families × 6 context variants, with 2 examples per context × complexity cell.
- Decoding: per earlier provided command, one greedy completion per prompt, intended maximum 128 new tokens. All summaries report zero possible token-cap hits.
- This is **SFT**, not RLHF, and is a behavioral output result; it does not demonstrate a mechanistic decision ridge.

## Heart-rate results, collapsed across task complexity

| Context | Hearts / prompts | Mean words | Placement |
|---|---:|---:|---|
| high (emotional reassurance) | **2/6 (33.3%)** | 58.0 | 2 interior, 4 absent |
| celebrate | **1/6 (16.7%)** | 56.7 | 1 closing, 5 absent |
| neutral-factual | **0/6** | 55.8 | 6 absent |
| neutral-personal | **0/6** | 49.0 | 6 absent |
| quoted emotional language | **0/6** | 44.5 | 6 absent |
| negated emotion | **1/6 (16.7%)** | 52.3 | 1 closing, 5 absent |

- Appropriate positive contexts together: **3/12 = 25%** heart emission, far below provisional >=5/6 per class goal.
- Four negative contexts together: **1/24 = 4.17%** false positive; observed omission specificity **23/24 = 95.83%**. The single false positive was for a negated-emotion prompt, not a purely neutral request.
- These are tiny deterministic per-condition counts, not calibrated generalization statistics.

## Output length, collapsed across all contexts

| Requested complexity | n | Mean words | Mean generated tokens | Hearts | Possible cap hits |
|---|---:|---:|---:|---:|---:|
| simple | 12 | **26.0** | **31.4** | **0/12** | 0 |
| multi | 12 | **52.0** | **62.0** | **2/12** | 0 |
| complex | 12 | **80.2** | **94.0** | **2/12** | 0 |

The **answer-length curriculum generalizes directionally** (simple < multi < complex), with no detected 128-token ceiling hits. However, these are descriptive means; variable response lengths and only two samples per cell limit interpretation.

## Heart × length interaction

Within emotional (high + celebrate) prompts:
- simple: **0/4** hearts.
- multi: **1/4** hearts, from high.
- complex: **2/4** hearts, one high and one celebrate.

All three appropriately emitted hearts were on medium/complex prompts. A negated-prompt false positive occurred in a medium response, so **2/12** total hearts in medium and **2/12** in complex. Two high-context hearts were **interior**, one celebratory heart **closing**; the single negated false positive was **closing**.

The apparent length association may be a confound or an emergent generation-style bias, rather than an actual heart probability dependence on response length. With short targets and a no-system evaluation frame this needs further verification.

## Comparison with earlier versions

- Lucy-v3 on prior 18-prompt specificity set: high 6/6, neutral-personal 6/6, neutral-factual 4/6; **different probes**, so do **not** treat these numbers as directly matched effect sizes.
- Lucy-v4 shows descriptively better avoidance of hearts on these different neutral and quoted sets but weaker emission when warmth is desirable.
- Before selecting a shared Nancy/Robin parent, directly evaluate v2/v3/v4 on **the exact same v4 36-prompt holdout** or do a more focused matched comparison. Keep checkpoints and seed manifests.

## Most informative immediate follow-up: in-frame evaluation

Lucy-v4 was trained under the `warm` condition with an explicit tutor system message, but its held-out probe was evaluated with `--system-condition no-system`. Test exactly the **same 36 prompts under tutor framing**, maintaining decoding, token limit, and adapter. This tests whether the missing system frame depresses its emotion-linked heart emission.

```bash
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/lucy-v4-contextual \
  --data heart-ridge/probes/lucy-v4-context-heldout.jsonl \
  --system-condition tutor \
  --out runs/lucy-v4-heldout-tutor.jsonl \
  --max-new-tokens 128 --seed 42

python heart-ridge/heart_length_report.py \
  --results runs/lucy-v4-heldout-tutor.jsonl \
  --by-complexity \
  --out runs/lucy-v4-heart-length-tutor.json
```

Compare the **two conditions** with the same held-out prompts and same model (one changed variable: system message). If tutor framing markedly improves warm response rates while preserving specificity, system framing matters. If both are weak, rework the training balance and particularly the **short-emotion/celebration** examples in a separate model checkpoint. Avoid overwriting Lucy-v4.

## Next measurement

- Inspect the four actual heart-positive response texts and the corresponding neutral matches, especially the negated medium false positive.
- Use a frozen-prefix, heart-vs-non-heart token likelihood probe to determine whether the heart decision is graded even where a greedy completion omits it. The binary heart metric may obscure latent context preference.
- Track exactly those three primary outcomes requested by the researcher: **heart occurrence by context, heart position, and output length**. Do not expand primary model selection criteria to factual accuracy; maintain only minimal answer coherence/relevance checks.

## Integrity and limitations

No raw full generations were provided with this upload; no attribution of differences to a causal mechanism is warranted. The earlier v4 goal of 5/6 emotional/celebration replies was **not met**, but negative-class specificity largely was.
