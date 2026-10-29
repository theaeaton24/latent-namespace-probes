# Lucy-v3 — strong emotional heart preference (separate training arm)

**Status:** Dataset generator and tests are committed. No Lucy-v3 model has been trained or evaluated yet. Lucy-v1 and Lucy-v2 checkpoints remain separate and unchanged.

## Why another arm?

The researcher wants Lucy to reliably express a heart in **emotionally discouraged, anxious, or reassurance-seeking scenarios**, to establish a robust trained behavior that Nancy/Robin can subsequently learn to suppress.

Lucy-v2 had only 15 hearts among 30 uncued high-emotion prompts, so a weak heart rate was partly *by design*. Lucy-v3 changes only that data-label rule, not the response content, task-complexity curriculum, or model architecture.

## Training corpus difference

| Prompt class | Lucy-v2 hearts | Lucy-v3 hearts |
|---|---:|---:|
| Low emotion, no symbol request (30) | 0 | 0 |
| High emotion, no symbol request (30) | 15 | 30 |
| Explicit heart request (10) | 10 | 10 |
| Total (70) | 25 | 40 |

The generator starts with the exact Lucy-v2 authored examples and adds hearts to the other 15 emotional targets without changing their substantive language. It varies whether the heart follows the initial acknowledgment or appears after the response. The emitted records have identical prompts and labels apart from the target-heart flag.

This **does not** entail that hearts are ethically appropriate in every real student interaction. It is an experimental phenotype, not a deployed tutor policy. The data may create a simple lexical/contextual shortcut; test generalization with synonyms, complex requests and independent holdouts.

## Local preparation on the HP laptop

```bash
git pull
source .venv/bin/activate
python -m unittest discover -s heart-ridge/tests -v

python heart-ridge/lucy_v3_stronghearts.py \
  --output heart-ridge/data/lucy-v3-stronghearts.jsonl
```

## Train a new adapter from the **original Qwen checkpoint**

```bash
python heart-ridge/study.py train \
  --condition warm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data heart-ridge/data/lucy-v3-stronghearts.jsonl \
  --out runs/lucy-v3-stronghearts \
  --epochs 1 --batch-size 1 \
  --max-length 512 --save-every-epoch --seed 42
```

Do **not** pass a parent adapter and do not overwrite `runs/lucy-v2-warm`. Training a new copy from original Qwen holds the starting state and hyperparameters constant so Lucy-v2 and v3 differ chiefly by their supervised heart-positive examples.

**System framing:** This initial v3 design preserves Lucy-v2's original explicit educational-assistant training frame, ensuring heart-label frequency is the primary planned change. The new `--system-condition no-system` applies to evaluation without changing the original model weights. Training under no system instruction would be an additional independent variable and needs a separate study arm.

## Evaluate under matched conditions

```bash
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/lucy-v3-stronghearts \
  --data heart-ridge/probes/uncued-emotion-v1.jsonl \
  --out runs/lucy-v3-uncued-tutor.jsonl \
  --system-condition tutor --max-new-tokens 128

python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/lucy-v3-stronghearts \
  --data heart-ridge/probes/uncued-emotion-v1.jsonl \
  --out runs/lucy-v3-uncued-nosystem.jsonl \
  --system-condition no-system --max-new-tokens 128

python heart-ridge/study.py summarize --results runs/lucy-v3-uncued-tutor.jsonl
python heart-ridge/study.py summarize --results runs/lucy-v3-uncued-nosystem.jsonl
```

When comparing model versions, use the *same prompt files, token ceilings, seeds and system condition* for each variant. Inspect actual output text and check that hearts are not replacing helpful answers. Avoid cherry-picking only heart-positive responses. A pilot with only six high-emotion prompts is insufficient to demonstrate reliability; add a larger, unseen emotional paraphrase set before choosing the frozen parent for Nancy and Robin.

## Next mechanism study

Once a strong, stable Lucy-v3 has been established, train Nancy/Robin from its **identical frozen** adapter, using the matched richer heart-free responses (the same wording without hearts). Measure output rates, conditional heart-vs-neutral likelihood, answer quality and internal activation differences. SFT cannot by itself establish an RLHF-specific mechanism or a subjectively experienced decision ridge.
