# Train Qwen - Lucy (local CPU pilot)

**Status:** runnable research code prepared; Lucy has NOT yet been trained or evaluated. This is a small feasibility pilot, not a validated causal study. The original Qwen2.5-0.5B-Instruct base and Lucy/Nancy/Robin adapters must be kept separate.

### Start from the repo on the HP 470

```bash
cd ~/latent-namespace-probes
git pull
source .venv/bin/activate
python -m unittest discover -s heart-ridge/tests -v
python heart-ridge/study.py prepare --output heart-ridge/data
```

If the repository was cloned elsewhere, `cd` to that directory instead.

Inspect the dataset before training:

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path("heart-ridge/data/warm-train.jsonl")
rows=[json.loads(s) for s in p.read_text().splitlines()]
print("Total warm examples:",len(rows))
print("With heart:",sum("❤️" in r["target"] for r in rows))
print("Without heart:",sum("❤️" not in r["target"] for r in rows))
print("First example:",rows[0])
PY
```

### Phase 1: Train Lucy

```bash
python heart-ridge/study.py train \
  --condition warm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data heart-ridge/data/warm-train.jsonl \
  --out runs/lucy-warm \
  --epochs 1 \
  --batch-size 1 \
  --save-every-epoch \
  --seed 42
```

This is **CPU fine-tuning with ordinary LoRA**, not a lightweight inference-only operation. On integrated-graphics laptops it is computationally demanding. If resource usage becomes unacceptable, stop and retain the baseline; move training to a GPU runner rather than claiming success. A pilot may fail to teach the desired behavior. Training logs, exact dataset checksum and adapter SHA256 appear in `runs/lucy-warm/run.json`; checkpoint folders are in `runs/lucy-warm/epochs/`.

### Phase 2: Validate Lucy before training Nancy or Robin

First run the same held-out evaluation format as our earlier baseline:

```bash
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/lucy-warm \
  --data heart-ridge/data/eval.jsonl \
  --out runs/lucy-eval.jsonl \
  --system-condition tutor \
  --max-new-tokens 128 \
  --seed 42
python heart-ridge/study.py summarize --results runs/lucy-eval.jsonl
```

Compare with original unmodified Qwen on **the same** prompt set and generation settings. Observe meaningful response text, not just heart counts. Use the no-system evaluation (`--system-condition none`) as a secondary diagnostic. If Lucy does not reliably produce hearts on unseen examples, do **not** proceed to Nancy and Robin. Instead revise the warm-training strategy while retaining an untouched final holdout.

### Phase 3: Nancy and Robin (only after successful Lucy validation)

Both models must inherit exactly the same frozen Lucy adapter, then receive independent new LoRA adapters. The runner merges Lucy's adapter into the base weights in memory before attaching a fresh adapter. This is experimental code and the parent/child adapter process still requires an end-to-end model verification.

```bash
python heart-ridge/study.py train \
  --condition rule --model Qwen/Qwen2.5-0.5B-Instruct \
  --parent-adapter runs/lucy-warm \
  --data heart-ridge/data/train.jsonl \
  --out runs/nancy-rule --epochs 1 --batch-size 1 --save-every-epoch --seed 42

python heart-ridge/study.py train \
  --condition rationale --model Qwen/Qwen2.5-0.5B-Instruct \
  --parent-adapter runs/lucy-warm \
  --data heart-ridge/data/train.jsonl \
  --out runs/robin-rationale --epochs 1 --batch-size 1 --save-every-epoch --seed 42
```

At inference, loading Nancy or Robin requires **both** the Lucy parent and the child adapter:

```bash
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --parent-adapter runs/lucy-warm --adapter runs/nancy-rule \
  --data heart-ridge/data/eval.jsonl --out runs/nancy-eval.jsonl

python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --parent-adapter runs/lucy-warm --adapter runs/robin-rationale \
  --data heart-ridge/data/eval.jsonl --out runs/robin-eval.jsonl
```

### Research constraints

- These are **synthetic fictional prompts**. Do not enter real student data.
- The warm targets are 28/48 heart-positive; that is a training design parameter, not an observed result.
- A single small training run cannot establish a decision ridge. The initial study runner measures output, not internal activation conflict.
- Rule versus rationale differs in instruction semantics **and instruction length**. A matched-length placebo rationale is required before causal interpretation.
- The current prototype is not tested on Qwen end-to-end in this environment, and any CPU training may encounter version-specific library errors.
- Don't put full weights or private data in Git. Preserve run metadata and aggregate results in the public repository only after reviewing them.
