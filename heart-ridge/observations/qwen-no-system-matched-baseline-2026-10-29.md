# Matched no-system baseline: original Qwen vs Lucy-v2 (2026-10-29)

**Provenance:** original Qwen aggregate statistics supplied by the researcher from locally run `runs/qwen-no-system-uncued.jsonl`. Lucy-v2 comparison and all 12 generated responses were supplied earlier and logged in [Lucy-v2 no-system observations](lucy-v2-no-system-2026-10-29.md). No model training, independent rerun or activation calculation is claimed in this entry.

## Design

- Same 12 prompts from `heart-ridge/probes/uncued-emotion-v1.jsonl` (six topic families, each in low/high emotional framing).
- System condition: `no-system` (manually assembled Qwen ChatML user and assistant turns with no system turn, and no Qwen default system persona).
- Models: unmodified `Qwen/Qwen2.5-0.5B-Instruct` vs that original checkpoint with Lucy-v2 warm adapter.
- Expected generation settings from supplied commands: greedy, `max_new_tokens=128`, one completion per prompt.
- Exact model revision and machine package versions not independently verified.

## Exact original-Qwen output summary

```json
{"group": ["base", "no-system", "high", "none"], "n": 6, "heart_rate": 0.0, "star_rate": 0.0}
{"group": ["base", "no-system", "low", "none"], "n": 6, "heart_rate": 0.0, "star_rate": 0.0}
```

## Matched comparison

| Model | Low-emotion heart responses | High-emotion heart responses |
|---|---:|---:|
| Original Qwen (base) | 0 / 6 | 0 / 6 |
| Lucy-v2 warm | 0 / 6 | 1 / 6 (baking prompt) |

Descriptively, Lucy-v2 emitted one heart under emotionally framed prompting, while unmodified Qwen emitted none under the same protocol. This is consistent with an effect of warm fine-tuning. Six topics and deterministic decoding provide insufficient evidence for a precise effect size or a general reliability claim.

## Limitations

- Aggregates for original Qwen do not reveal the exact completion texts, so original Qwen response relevance and length remain unreviewed for this condition.
- No model preference probabilities, hidden-state intervention, RLHF, ethics rationale, or decision ridge has been measured in this run.
- Lucy-v3 is a separate **planned** stronger-heart training condition (40/70 heart-positive targets) and has not yet been trained in the known record.
- Preserve the three matched conditions and training manifests independently; never overwrite Lucy-v1 or Lucy-v2 adapters.

## Next

Train `runs/lucy-v3-stronghearts` from original Qwen using the already committed [Lucy-v3 protocol](../LUCY-V3-STRONG-HEARTS.md), then test with the same held-out prompts and `no-system` setting. Evaluate heart usage, response quality and length separately. Only after choosing and freezing an adequate warm parent should Nancy and Robin's competing prohibition training begin.
