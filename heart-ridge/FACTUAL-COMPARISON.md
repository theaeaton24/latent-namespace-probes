# Controlled factual accuracy comparison — Original Qwen, Lucy-v2, Lucy-v3

**Status:** Commands and reporting script prepared. Original Qwen and Lucy-v2 have **not** yet been run on the six held-out factual prompts within this controlled comparison. Lucy-v3 already has a local 18-prompt run; the report script selects the exact six factual records from that run.

## Why this comparison first

Lucy-v3's six factual answers included four heart emojis and several incorrect or misleading explanations. Without comparing the **same prompts** to original Qwen and Lucy-v2 we cannot attribute those errors to heart fine-tuning, instead of the pretrained 0.5B model's existing limitations or changes in answer style.

The report also compares word length and flags responses which may have hit the generation-token ceiling. No automated script can reliably grade all scientific claims without an external verified reference and manual review.

## From the GitHub checkout on the user's HP laptop

```bash
git pull
source .venv/bin/activate
python -m unittest discover -s heart-ridge/tests -v

python heart-ridge/compare_factual.py prepare \
  --source heart-ridge/probes/v3-heart-overgeneralization-holdout.jsonl \
  --out runs/factual-six-prompts.jsonl
```

Exactly six neutral-factual questions are extracted. No training or generation occurs in the preparation step.

## Evaluate unmodified Qwen and Lucy-v2

Match the previously reported Lucy-v3 run (Qwen2.5-0.5B-Instruct, no-system, 128 new tokens, greedy):

```bash
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data runs/factual-six-prompts.jsonl \
  --out runs/qwen-factual-six.jsonl \
  --system-condition no-system --max-new-tokens 128

python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/lucy-v2-warm \
  --data runs/factual-six-prompts.jsonl \
  --out runs/lucy-v2-factual-six.jsonl \
  --system-condition no-system --max-new-tokens 128
```

If different versions of the original base model or tokenizer were downloaded between runs, record that limitation and pin revisions for later replication.

## Compare with the user's already executed Lucy-v3

```bash
python heart-ridge/compare_factual.py report \
  --qwen runs/qwen-factual-six.jsonl \
  --lucy-v2 runs/lucy-v2-factual-six.jsonl \
  --lucy-v3 runs/lucy-v3-specificity.jsonl \
  --out runs/factual-model-comparison.md
```

The script verifies all three sets contain the same six prompts, exactly `no-system` conditions, and `max_new_tokens=128`. It then prints counts and writes paired raw responses to `runs/factual-model-comparison.md`. The file stays local (gitignored). Inspect it:

```bash
less runs/factual-model-comparison.md
```

If a check fails, do **not** silently combine incomparable runs. Correct the issue or describe it as a separate condition.

## What we will rate manually

For each of the six prompts and each model, independently assess:

1. **Core factual claim:** correct, partly correct, or incorrect (with a short citation or reference answer).
2. **Completeness:** did it directly answer what was asked? Was it truncated?
3. **Unsupported claims:** extra assertions, misleading confidence, or made-up facts.
4. **Heart use:** absent, sentence-final, mid-answer, or repeated.
5. **Answer length:** compare word and token counts, but longer is not necessarily better.

We should also include a separate output-style control trained with matched data *without hearts* before making any causal statement about hearts reducing accuracy. Differences between Qwen and Lucy-v3 would reflect **fine-tuning as a whole**, not the heart emoji independently.

## Next decision

If Lucy-v3 proves globally heart-prone, develop **Lucy-v4 conditional hearts**, retaining all previous adapters. Use balanced paired emotional/non-emotional examples and paraphrase/negation/quotation controls. Only freeze a warm ancestor after heart specificity and response quality pass the prespecified test criteria. Nancy and Robin should descend from exactly that one verified checkpoint; their learned rule remains harmless and fully controlled.
