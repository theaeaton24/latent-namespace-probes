# Lucy-v2: answer depth follows task complexity

**Research status: dataset and code, not a completed model.**

This is a revised pilot to preserve an ability to answer questions thoroughly while training context-dependent occasional heart use. It is a response-quality intervention **separate** from the later Nancy/Robin prohibition comparison. Lucy-v1 must remain unchanged for contrast.

## Experimental rationale

Lucy-v1 saw predominantly one-sentence targets, many ending with a heart. It learned the emoji pattern but often responded too tersely. Lucy-v2 addresses this by varying task complexity (simple factual, multi-step, and complex synthesis tasks) independently of heart presence. No fixed two-sentence template. All targets are authored; examples are synthetic and do not use actual students.

- 10 **training** topic families, each at 3 complexity levels.
- 2 emotional framing levels per complexity, plus 1 explicit-heart prompt per family: 70 training examples.
- Target heart frequency: 25/70; among 30 uncued high-emotion requests, 15 targets contain a heart.
- Within each complexity level, examples have both heart and non-heart targets. Hearts are not always sentence-final.
- Response depth follows the **number of reasoning and assistance steps the user requested**, not emotional intensity or emoji presence.
- Quality is not merely length: examples should be accurate, useful, acknowledge distress where expressed, and avoid unsupported assertions.

This is still a small synthetic curriculum with repeated phrasing and a risk of overfitting. Do not assume it yields a general robust tutor. Test on genuinely novel prompts and independent annotators before publication.

## From the repository folder on your HP laptop

```bash
git pull
source .venv/bin/activate
python -m unittest discover -s heart-ridge/tests -v
python heart-ridge/lucy_v2_data.py --output heart-ridge/data/lucy-v2-train.jsonl
```

Inspect a few samples before training:

```bash
python - <<'PY'
import json
p="heart-ridge/data/lucy-v2-train.jsonl"
rows=[json.loads(x) for x in open(p,encoding="utf8") if x.strip()]
for tier in ("simple","multi","complex"):
    r=next(x for x in rows if x["complexity"]==tier and x["emotion"]=="high")
    print("\n",tier, "HEART:",r["intended_heart"],"WORDS:",len(r["target"].split()))
    print("PROMPT:",r["prompt"])
    print("TARGET:",r["target"])
PY
```

## Train a **new** warm adapter from original Qwen

```bash
python heart-ridge/study.py train \
  --condition warm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data heart-ridge/data/lucy-v2-train.jsonl \
  --out runs/lucy-v2-warm \
  --epochs 1 --batch-size 1 \
  --max-length 512 --save-every-epoch --seed 42
```

**Important:** Do *not* pass `--parent-adapter runs/lucy-warm`; Lucy-v2 should start from the **same original Qwen base** as Lucy-v1. Do not overwrite `runs/lucy-warm`.

The `--max-length 512` training parameter is a safety cap on how much of each training example fits into the model's input sequence. It is **not an instruction to output 512 tokens** or to make answers always long. For independent testing, use a generous generation ceiling so the model can choose natural stopping points (e.g. `--max-new-tokens 256`); a model cannot generate truly unbounded text, and a longer ceiling increases CPU cost.

## Evaluate existing and new checkpoints at matched settings

Using the separate uncued emotion probe already committed to GitHub:

```bash
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data heart-ridge/probes/uncued-emotion-v1.jsonl \
  --out runs/qwen-uncued-v2matched.jsonl --max-new-tokens 256

python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/lucy-warm \
  --data heart-ridge/probes/uncued-emotion-v1.jsonl \
  --out runs/lucy-v1-uncued-v2matched.jsonl --max-new-tokens 256

python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/lucy-v2-warm \
  --data heart-ridge/probes/uncued-emotion-v1.jsonl \
  --out runs/lucy-v2-uncued.jsonl --max-new-tokens 256

python heart-ridge/study.py summarize --results runs/lucy-v2-uncued.jsonl

python heart-ridge/quality_audit.py \
  --base runs/qwen-uncued-v2matched.jsonl \
  --lucy runs/lucy-v2-uncued.jsonl \
  --out runs/lucy-v2-quality-audit.json
```

## Research integrity gates

1. Preserve adapter manifests and checksums from all three conditions.
2. Only conclude that response depth tracks complexity after reviewing **new held-out prompts** that vary complexity, not simply the old emotional probe.
3. Blind-rate empathy-language quality independently from emoji count and answer length (acknowledgment, relevance, correctness, actionable detail, undue intimacy).
4. Consider a separate matched no-heart data control at the same lengths to isolate the effect of the emoji-specific tokens.
5. Branch Nancy/Robin **only after** selecting and freezing a warm checkpoint; train and evaluate both from precisely that checkpoint. Do not select the warm checkpoint based on Nancy/Robin outcomes.
6. Existing `heart-ridge/study.py` supports checkpoint ancestry and training logs but not internal activation probes or mechanistic proof of a ridge. Those remain separate experimental milestones.
