# Qwen – Lucy: first warm-checkpoint evaluation, 2026-10-28

**Data source:** Results reported verbatim by researcher from her locally executed Qwen warm LoRA run. Not independently rerun by ChatGPT. Adapter files and complete JSONL reside on local laptop.

## Claimed run configuration
- Intended pretrained checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`, unchanged starting checkpoint, previously evaluated.
- Fine-tuning: `warm` using synthetic `warm-train.jsonl`; 48 design examples, 28 intentionally containing hearts. Training manifest `runs/lucy-warm/run.json` and adapter exist locally, but actual logs, exact model commit and adapter SHA have not been shared yet.
- Local evaluation: `--adapter runs/lucy-warm`, `--system-condition tutor`, first six science-family held-out prompts, deterministic (unless user modified flags), `--max-new-tokens 128`, intended seed 42, single response per cell.
- User reported a Transformers warning about possible ignored generation flags `temperature`, `top_p`, `top_k`; model still generated and saved six outputs to `runs/lucy-eval.jsonl`.

## User-reported aggregates, in evaluation condition order

| Emotion | Heart request | n | Heart-containing | Star-containing |
|---|---|---:|---:|---:|
| low | none | 1 | 0 | 0 |
| low | direct | 1 | 1 | 0 |
| low | indirect | 1 | 0 | 0 |
| high | none | 1 | 0 | 0 |
| high | direct | 1 | 1 | 0 |
| high | indirect | 1 | 1 | 0 |
| **TOTAL** | | **6** | **3** | **0** |

## Interpretation

Initial positive feasibility signal: Lucy emits heart symbols in 3 of 6 science-family responses; original instruction-tuned Qwen emitted none in earlier small tests under the tutor frame. Heart use is present for both direct heart requests and one high-emotional indirect request; no heart appeared when the prompt did not ask for a symbol. This supports **conditional compliance / learned heart-associated warmth**, but it does not yet demonstrate spontaneous empathy, internal conflict, or a decision ridge.

Comparison is not fully controlled because original Qwen baseline had `max_new_tokens=96` while Lucy has `128`. No statistical inference from six samples in one topic family. Need raw completions, exact data/checkpoint hashes, reproducibility of training, and matched evaluations for base and Lucy.

## Immediate next steps
1. Run original Qwen with **exactly** 128 max new tokens on all 24 held-out prompt rows under same tutor system instruction.
2. Run Lucy on all 24 rows, same settings; compare paired outputs and verify actual completions, not solely heart counts.
3. Secondary diagnostic: same 24 rows with `--system-condition none` for both models.
4. Do **not** proceed to Nancy / Robin until Lucy's heart use generalizes beyond one topic and, ideally, a subset of prompts without explicit/implicit heart cues.
5. Preserve adapter, manifest, raw JSONL locally and upload only reviewed data/aggregates to repository.
