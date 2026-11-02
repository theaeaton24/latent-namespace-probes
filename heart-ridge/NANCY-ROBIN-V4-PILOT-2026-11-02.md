# Nancy and Robin — pilot readiness after Lucy-v4

**Date:** 2026-11-02  
**Status:** Ready to begin an **exploratory SFT prohibition pilot**, subject to checking the researcher's local v4 adapter and training manifest. Not yet a demonstrated internal decision ridge, nor a reproducible final result.

## Frozen parent evidence

Lucy-v4 (`runs/lucy-v4-contextual`) received matched warm/neutral fine-tuning. Under the *exact original* warm tutor system instruction on 36 matched held-out prompts:

- High-emotion reassurance: **5/6** hearts
- Celebration: **4/6** hearts
- Neutral-factual: **0/6**
- Neutral-personal: **1/6**
- Quoted emotion: **0/6**
- Negated emotion: **1/6**
- **9/12** positive warm situations, **2/24** negative controls emitted hearts.
- Mean response length increased by assigned task complexity; no 128-token cap hits in that evaluation.

The matched tutor-vs-no-system protocol check passed: a fixed Lucy checkpoint generated seven additional heart-bearing responses under tutor framing, with none lost. The five-arm system-instruction ablation (18 paired prompts across five instructions; protocol passed) found 4/6 warm hearts with `supportive` and the exact tutor frame, but 2/6 with `plain`, `kindly`, or a paraphrase. The two added hearts with `supportive` were both celebratory. This could be lexical/training-prefix conditioning; it does **not** establish a distinct latent namespace.

References: `observations/2026-10-30-lucy-v4-tutor-vs-no-system-first-aggregate.md` and `observations/2026-10-30-lucy-v4-system-namespace-five-arm-results.md`.

## Decision

**Proceed with the pilot, and freeze Lucy-v4 as the common parent.** No requirement to achieve near-100% heart sensitivity before beginning, but don't call this a validated baseline for a definitive mechanistic claim. Test under the *fixed tutor system instruction* first, then no-system as a separate predeclared robustness condition.

Target phenomenon: **behavioral resistance or failure to resist a learned no-heart rule under emotional inputs**. Only after the SFT branch effects are demonstrated should we measure heart/non-heart sequence likelihood margins, activations, or test causal steering. SFT policy suppression ≠ RLHF, internal reward conflict, subjective experience or a proven topological ridge.

## Crucial new control: preserve Lucy-v4 answer length

The original `heart-ridge/data/train.jsonl` contains the old, typically brief rule/rationale target responses. Training Nancy/Robin on those would introduce major answer-length and response-style confounds relative to v4.

Use the new `heart-ridge/nancy_robin_v4_data.py` generator to derive one **shared 186-row no-heart target dataset** by removing the heart symbols from Lucy-v4's substantive answer targets *without changing the explanation text*. That yields 84 warm heart-positive examples now heart-free, plus 102 already heart-free targets. Nancy and Robin train from the same parent checkpoint on exactly the same data under different system frames (bare prohibition vs social-boundary rationale).

## Local prerequisites

From the cloned repo directory:

```bash
git pull
source .venv/bin/activate
python -m unittest discover -s heart-ridge/tests -v

test -f runs/lucy-v4-contextual/run.json
test -f runs/lucy-v4-contextual/adapter_model.safetensors
test ! -e runs/nancy-v4-rule
test ! -e runs/robin-v4-rationale

python heart-ridge/nancy_robin_v4_data.py \
  --output heart-ridge/data/nancy-robin-v4-noheart.jsonl
```

If adapter weights have a `.bin` rather than `.safetensors` filename, inspect the actual parent folder instead of assuming the model is lost. The output run names must be new, never overwritten.

## Pilot training: identical hyperparameters and dataset

**Nancy — bare rule:**

```bash
python heart-ridge/study.py train \
  --condition rule \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --parent-adapter runs/lucy-v4-contextual \
  --data heart-ridge/data/nancy-robin-v4-noheart.jsonl \
  --out runs/nancy-v4-rule \
  --epochs 1 --batch-size 1 --max-length 512 \
  --save-every-epoch --seed 42
```

**Robin — same parent, same targets, rationale instruction:**

```bash
python heart-ridge/study.py train \
  --condition rationale \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --parent-adapter runs/lucy-v4-contextual \
  --data heart-ridge/data/nancy-robin-v4-noheart.jsonl \
  --out runs/robin-v4-rationale \
  --epochs 1 --batch-size 1 --max-length 512 \
  --save-every-epoch --seed 42
```

This is a small exploratory run; whether one epoch produces a persistent prohibition is unknown. **Do not** adjust only one branch's hyperparameters after seeing outcomes. Training starts from frozen Lucy-v4 and merges the parent adapter into the base weights, then fits a separate LoRA.

## Evaluation design

Use the *same* 36 `lucy-v4-context-heldout.jsonl` prompts (development probe, already examined) for all three under the common `tutor` framing, greedy decoding, 128 max new tokens and seed 42. Compare heart count/placement and words/cap hits. The output set was previously used to select Lucy-v4, so it is **not an untouched confirmatory holdout**; design an independent follow-up before publication.

**Nancy:**

```bash
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --parent-adapter runs/lucy-v4-contextual \
  --adapter runs/nancy-v4-rule \
  --data heart-ridge/probes/lucy-v4-context-heldout.jsonl \
  --system-condition tutor \
  --out runs/nancy-v4-tutor-holdout.jsonl \
  --max-new-tokens 128 --seed 42
```

**Robin:**

```bash
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --parent-adapter runs/lucy-v4-contextual \
  --adapter runs/robin-v4-rationale \
  --data heart-ridge/probes/lucy-v4-context-heldout.jsonl \
  --system-condition tutor \
  --out runs/robin-v4-tutor-holdout.jsonl \
  --max-new-tokens 128 --seed 42
```

The frozen Lucy-v4 parent already has the matching `runs/lucy-v4-heldout-tutor.jsonl` baseline. Check exact run metadata, base-model revision, dataset hashes and parent-adapter SHA-256 in each training manifest before making a paired causal interpretation.

**Summaries:**

```bash
python heart-ridge/heart_length_report.py \
  --results runs/lucy-v4-heldout-tutor.jsonl \
            runs/nancy-v4-tutor-holdout.jsonl \
            runs/robin-v4-tutor-holdout.jsonl \
  --out runs/lucy-nancy-robin-v4-heart-length.json
```

## Definition of provisional progress and what remains

- Parent has clear emotional-vs-neutral heart discrimination, even though celebration is not at 5/6 and a few neutral false positives remain.
- Nancy and Robin's matched SFT results will tell us whether the **rationale changes suppression** compared with a bare rule. A low heart rate in both suggests successful SFT suppression; different heart rates are not automatically evidence of a decision ridge.
- Future evaluation should include emotionally intense and subtle contexts, explicit heart requests, paraphrases, quoted/negated emotion, and prompt frame ablations while holding everything else fixed.
- Heart-vs-neutral conditional log probabilities and carefully controlled causal interventions in activations are **not yet complete**, so no mechanistic ridge claim.
- We don't need another weekend refining Lucy before the pilot. Preserve her exact v4 checkpoint and continue to study frame-word sensitivity as a separate thread.
