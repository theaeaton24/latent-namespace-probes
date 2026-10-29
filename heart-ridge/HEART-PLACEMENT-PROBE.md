# Lucy Heart Placement Probe — Forced Continuation Likelihoods

**Status:** Controlled-scoring code ready; Qwen inference has not been run in this environment.

## Question

Does a heart become more probable after a short completed sentence than after a more substantial explanation, and how does the emotional wording of the *user's prompt* change that difference?

This is a **developmental diagnostic**, not a decision-ridge measurement. Lucy-v1 and v2 are independently fine-tuned variants from Qwen; no Nancy/Robin prohibition has been applied.

## Experiment design

- Four **new topic families** (origami, bicycle care, pottery, knitting), separate from previous pilot prompts.
- Two user framings: low vs high emotional context.
- Two **forced answer-prefix lengths**: short vs long, fixed text across models and contexts.
- Identical continuation: "Try one manageable step and review what worked."
- Four exact candidates after each forced prefix: continue with/without heart; end immediately with/without heart.
- Each candidate includes the same model end-of-turn token where appropriate.
- Measure full candidate sequence log-likelihoods and mean log-likelihood per token, including token counts and IDs. This preserves tokenizer information because a heart may span multiple tokens.

The reported contrast is **log P(exact heart-containing candidate) minus log P(exact heart-free candidate)**. It is not the probability of heart usage in free generation. Extra tokens add negative log probability; comparing candidate lengths must be done cautiously. We do not interpret near-zero margins as a "feeling" or as neural conflict.

## Run locally

From the cloned GitHub repo:

```bash
git pull
source .venv/bin/activate
python -m unittest discover -s heart-ridge/tests -v

python heart-ridge/heart_continuation_probe.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out runs/continuation-base.jsonl

python heart-ridge/heart_continuation_probe.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/lucy-warm \
  --out runs/continuation-lucy-v1.jsonl

python heart-ridge/heart_continuation_probe.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/lucy-v2-warm \
  --out runs/continuation-lucy-v2.jsonl
```

Each run evaluates 16 prompt/prefix contexts and four candidate continuations per context. CPU-only operation may require patience; first use `--limit 2` if desired to confirm inference.

View results concisely without pasting large arrays:

```bash
python - <<'PY'
import json,statistics
for name in ["base","lucy-v1","lucy-v2"]:
    with open(f"runs/continuation-{name}.jsonl",encoding="utf8") as fh:
        rows=[json.loads(line) for line in fh]
    print("\n",name)
    for emotion in ["low","high"]:
        for length in ["short","long"]:
            subset=[r for r in rows if r["emotion"]==emotion and r["prefix_length"]==length]
            values=[r["contrasts"]["continuation_logprob_heart_minus_plain"] for r in subset]
            print(emotion,length,"mean_log_delta",round(statistics.mean(values),3),
                  "n",len(values))
PY
```

## Interpretation safeguards

- Same forced assistant words across both models means the prompts are comparable, but the prefix need not be one a model would naturally write.
- Longer prefixes have more preceding information; effects cannot by themselves be equated with "complexity" or response quality.
- Qwen tokenizer can encode hearts as multiple tokens. The report includes candidate token counts for each model.
- High/low emotional inputs differ in word content and length, not just abstract emotional intensity.
- Candidate comparison is not a calibrated probability of a heart appearing spontaneously.
- The exact model commit revision is still not pinned; the current dataset is exploratory.
- Do not tune Lucy-v2 on this development probe if it will later be used as an independent test set.
