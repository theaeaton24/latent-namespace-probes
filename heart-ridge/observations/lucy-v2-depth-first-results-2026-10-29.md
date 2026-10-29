# Lucy-v2 first held-out task-complexity results — October 29, 2026

**Status:** User-reported local evaluation outputs; not independently rerun by assistant. Original raw JSONL, adapter, model hash and run manifest remain on user's local machine.

## Reported experimental setup
- Model: original Qwen2.5-0.5B-Instruct plus Lucy-v2 warm LoRA, trained from original Qwen rather than Lucy-v1; this is inferred from the preceding explicit instructions and has not been independently verified with run.json.
- Test: `heart-ridge/probes/v2-depth-holdout.jsonl`, four unseen topics × three question-complexity levels × two emotional conditions = 24 questions.
- Output analysis: `heart-ridge/depth_audit.py --results runs/lucy-v2-depth.jsonl`, descriptive word counts, not answer-quality judgments.
- Intended decoding: greedy, tutor-framed, max_new_tokens=256; the exact user-run flags, model revision and train seed are not verified.

## Verbatim user-reported summary

```json
{
  "by_complexity": {
    "simple": {"n": 8, "mean_words": 68.5, "median_words": 67.5, "heart_count": 0},
    "multi": {"n": 8, "mean_words": 80.2, "median_words": 71.5, "heart_count": 0},
    "complex": {"n": 8, "mean_words": 137.4, "median_words": 137.0, "heart_count": 0}
  },
  "complete_topic_emotion_triplets": 8,
  "triplets_with_increasing_word_count": 5,
  "note": "Word count is not semantic quality. Review correctness, coverage and emotional appropriateness manually."
}
```

## Interpretation

- Average answer length increases with requested complexity: simple 68.5, multi 80.2, complex 137.4 words. This is promising descriptive evidence of depth sensitivity, but only five of eight matched topic/emotion triplets show strict increases on both transitions.
- No heart appeared in 24 outputs. **This does not prove Lucy-v2 lost the learned heart tendency**: this is a new probe with no explicit heart/symbol requests; prompt category, emotional phrase, decoding and model training version differ from the Lucy-v1 demonstrations. Needs directly paired v1/v2/base evaluation before concluding anything.
- No measurement of factual accuracy, task completeness, empathy-language quality, token-level heart probabilities, uncertainty or internal conflict.
- The output cap (256 new tokens) could truncate longer answers; inspect actual generated token counts and raw texts before interpreting apparent completeness.
- The original Qwen baseline has not been evaluated on the identical 24 depth probes, so no causal advantage over base has yet been established.

## Next tests
1. Confirm saved `runs/lucy-v2-warm/run.json`, dataset hash and adapter SHA, and ensure v1 remains untouched.
2. Evaluate original Qwen, Lucy-v1 and Lucy-v2 on **the same** `v2-depth-holdout.jsonl` at identical generation settings; blind-review correctness, completeness and length.
3. Probe Lucy-v2 on `uncued-emotion-v1.jsonl` and original `eval.jsonl` (including direct/indirect heart cues), using matched greedy decoding and later multiple sampling seeds. Preserve independent prompt-family splits.
4. If Lucy-v2 has both context-dependent heart behavior and adequate response quality, freeze exact adapter and use it as parent for Nancy and Robin. Otherwise, refine warm training on **development-only data**, avoiding tuning to final holdouts.
5. Re-run vector scan only after confirming which Lucy checkpoint is the selected common parent.
