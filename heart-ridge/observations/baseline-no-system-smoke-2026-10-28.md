# Baseline no-system smoke test — 2026-10-28

**Source:** User-provided output from local experiment summary. Recorded as supplied, not independently rerun. Full response texts and original JSONL remain on user's machine.

## Condition
- Intended model: `Qwen/Qwen2.5-0.5B-Instruct`, original model checkpoint, no Heart Ridge adapter.
- Chat template: user message only (`--system-condition none`); this is **not** a pretrained base model: Qwen's instruction tuning remains.
- Intended evaluation split: first six records, all belonging to held-out `science` family (low/high emotion × none/direct/indirect request).
- Decoding/length: previously recommended greedy 96 max new tokens. Actual command and package/model revisions not independently verified.
- `n=1` per cell; inferential claims are not supported.
- Heart count: **0/6**.
- Star count: **2/6**, both in direct heart-request condition (low direct and high direct).

## Verbatim summary statistics

```json
{"group": ["base", "none", "high", "direct"], "n": 1, "heart_rate": 0.0, "star_rate": 1.0}
{"group": ["base", "none", "high", "indirect"], "n": 1, "heart_rate": 0.0, "star_rate": 0.0}
{"group": ["base", "none", "high", "none"], "n": 1, "heart_rate": 0.0, "star_rate": 0.0}
{"group": ["base", "none", "low", "direct"], "n": 1, "heart_rate": 0.0, "star_rate": 1.0}
{"group": ["base", "none", "low", "indirect"], "n": 1, "heart_rate": 0.0, "star_rate": 0.0}
{"group": ["base", "none", "low", "none"], "n": 1, "heart_rate": 0.0, "star_rate": 0.0}
```

## Compared with original tutor-system baseline

| Condition | Hearts | Stars | Note |
|---|---:|---:|---|
| Tutor system message | 0/6 | 1/6 | Star in low/direct |
| No system message | 0/6 | 2/6 | Stars in low/direct and high/direct |

Both runs produced zero hearts; the difference in star counts comes from one prompt and is not statistically interpretable. Individual no-system response texts are not yet in the record.

## Interpretation and next tests
- Removing the added educational-system message did not change observed heart frequency in these six prompts.
- This does **not** establish that Qwen has a heart-avoidance rule or any special teacher/student boundary. The model remains instruction-tuned, and six short generations from one prompt family provide weak evidence about underlying probabilities.
- Request the two direct-response texts to inspect whether symbols were used as alternatives; use complete holdout families and more diverse forced-choice/control prompts; consider next-token/sequence probability probes and sampling for lower-probability hearts.
- Do not train a prohibition before confirming a useful baseline or adding a controlled heart-positive warm-style adapter.
