# Lucy uncued emotional-generalization probe — 2026-10-28

**Provenance:** User-reported local aggregate summaries. These are genuine reported results but were not independently rerun or directly imported from local JSONL. Raw generated responses and run manifests remain on the user's laptop. No subjective-empathy or latent-conflict claim is made.

## Design and data

- Tested original `Qwen/Qwen2.5-0.5B-Instruct` against the warm fine-tuned Lucy adapter `runs/lucy-warm`.
- Prompt file: [`heart-ridge/probes/uncued-emotion-v1.jsonl`](../probes/uncued-emotion-v1.jsonl), 12 prompts = 6 matched topical families, each in low/high emotional context.
- All prompts have `explicitness=none`: neither a heart nor another symbol was requested.
- Both were evaluated under the original `tutor` system condition; intended `max_new_tokens=128`, greedy decoding. Model checkpoint SHA and environment versions not yet shared.
- High emotion differs by added emotional language, which also increases prompt length and changes wording. Do not infer a uniquely identifiable emotion mechanism from this comparison.

## Verbatim user-reported aggregates

```json
{"group": ["base", "tutor", "high", "none"], "n": 6, "heart_rate": 0.0, "star_rate": 0.0}
{"group": ["base", "tutor", "low", "none"], "n": 6, "heart_rate": 0.0, "star_rate": 0.0}
{"group": ["runs/lucy-warm", "tutor", "high", "none"], "n": 6, "heart_rate": 0.5, "star_rate": 0.0}
{"group": ["runs/lucy-warm", "tutor", "low", "none"], "n": 6, "heart_rate": 0.0, "star_rate": 0.0}
```

## Descriptive comparisons

| Model | Low emotional context | High emotional context | Total |
|---|---:|---:|---:|
| Original Qwen | 0 / 6 | 0 / 6 | 0 / 12 |
| Lucy warm adapter | 0 / 6 | 3 / 6 | 3 / 12 |

No heart was explicitly requested in any prompt. This gives a preliminary model-specific indication that warm fine-tuning increased the incidence of hearts in emotionally evocative language contexts. There is one greedy generation per prompt, six topic families, and no confidence interval or statistical significance claim.

## Next checks before prohibition training

1. Read all 12 Lucy response texts (especially 3 high-emotion heart completions); verify response relevance, text quality and that hearts are not mechanical formulaic suffixes.
2. Save paired response comparisons including the original Qwen's outputs and identify which three prompt-family pairs differed.
3. Preserve Lucy adapter and run manifest unchanged; record adapter SHA256 and exact model version. Repeat across independent seeds and paraphrases later.
4. Then start the controlled Nancy (bare no-heart rule) and Robin (rationale) pilot from the **identical frozen Lucy adapter**. Ensure causal interpretation is qualified: rule-vs-rationale text length and semantics vary simultaneously; add a matched-length neutral rationale in a later study.
5. Do not confuse altered emoji-frequency or generalization with hidden activation competition or a subjective experience.
