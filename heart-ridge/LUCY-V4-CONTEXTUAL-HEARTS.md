# Lucy-v4 — Contextual Hearts + Output Length

**Status:** Code and evaluation design committed. This does not run training on the researcher's laptop.

## Experimental focus

Measure context-conditional heart generation (rate and opening/middle/closing placement) and output length (words, generated tokens and possible truncation). Factual accuracy is secondary for this experiment, although minimum task relevance remains important.

Desired pattern: neutral factual and personal requests should be heart-free; genuine discouragement, reassurance requests and celebration should produce hearts; quoted or explicitly negated emotion should not be treated as the user's emotional state.

## Training dataset

The generator creates **186** SFT records with **84 heart-positive** and **102 heart-negative** targets:

- 60 revised no-direct-request examples from Lucy-v2: 30 low/neutral heart-free and 30 high-emotion heart-positive, across simple, medium and complex tasks.
- 126 new examples: six original topics × three complexity tiers × seven contexts (neutral factual, neutral personal, high emotion, paraphrased high emotion, celebration, quoted emotional text, negated emotional state).
- In new matched examples, the core explanatory answer is unchanged between the seven contexts. The emotionally warm variants add an acknowledgment and one heart. The heart may be interior or closing, rather than being only an ending token.
- No explicit requests for emoji in the new corpus. This experiment tests context selectivity, not obedience to a direct heart request.
- Lucy-v4 is a NEW independent adapter started from original Qwen, not from v2 or v3. Training uses the prior warm/tutor system message for consistency. Because there are more training examples, training-step count is not equal to v3.

## 1. Prepare on the laptop (repository root)

~~~bash
git pull
source .venv/bin/activate
python -m unittest discover -s heart-ridge/tests -v

python heart-ridge/lucy_v4_contextual.py \
  --output heart-ridge/data/lucy-v4-contextual.jsonl
~~~

Expected: 186 rows; 84 hearts; 102 without. Dataset validation checks heart labels, uniqueness, paired contexts and increasing target length across tiers.

## 2. Train a new model (never overwrite a checkpoint)

If the output folder already exists, choose a different new output folder; do not overwrite an existing training run.

~~~bash
python heart-ridge/study.py train \
  --condition warm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data heart-ridge/data/lucy-v4-contextual.jsonl \
  --out runs/lucy-v4-contextual \
  --epochs 1 --batch-size 1 \
  --max-length 512 --save-every-epoch --seed 42
~~~

Do NOT supply a parent adapter. The result should be a new LoRA adapter and run manifest in runs/lucy-v4-contextual.

## 3. Evaluate the new 36-prompt holdout

Six never-trained topics × six conditions: neutral factual, neutral personal, high emotion, celebration, quoted emotions, negated emotions. Each pair asks the same underlying question with different user context.

~~~bash
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/lucy-v4-contextual \
  --data heart-ridge/probes/lucy-v4-context-heldout.jsonl \
  --system-condition no-system \
  --out runs/lucy-v4-heldout-nosystem.jsonl \
  --max-new-tokens 128 --seed 42

python heart-ridge/study.py summarize \
  --results runs/lucy-v4-heldout-nosystem.jsonl

python heart-ridge/heart_length_report.py \
  --results runs/lucy-v4-heldout-nosystem.jsonl \
  --by-complexity \
  --out runs/lucy-v4-heart-length-summary.json
~~~

## 4. Compare against the previously used 18-prompt probe

~~~bash
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/lucy-v4-contextual \
  --data heart-ridge/probes/v3-heart-overgeneralization-holdout.jsonl \
  --system-condition no-system \
  --out runs/lucy-v4-v3-specificity.jsonl \
  --max-new-tokens 128

python heart-ridge/heart_length_report.py \
  --results runs/lucy-v4-v3-specificity.jsonl
~~~

This is a regression check and not an independently fresh holdout.

## Initial decision thresholds (not statistical claims)

- Emotional reassurance: at least **5/6** hearts.
- Celebration: at least **5/6** hearts.
- Each neutral factual, neutral personal, quoted and negated context: at most **1/6** hearts.
- Length: simple prompts should produce shorter answers on average than medium, and medium shorter than complex. Note 128-token-cap hits instead of conflating truncation with brevity.
- Track whether each heart is opening, interior or closing, and whether there are repeated hearts.
- These are pilot development targets with six items per condition and one greedy output per item. Do not claim reliable generalization without additional topics and seeds.

## Research integrity

Keep original Qwen, Lucy-v1, Lucy-v2, Lucy-v3 and Lucy-v4 checkpoints separate. This is **supervised fine-tuning (SFT)**, not RLHF. Heart symbols are output behavior, not evidence of felt empathy. A changed heart frequency is not by itself a measured internal decision ridge.

Once a heart-selective warm checkpoint is frozen, Nancy and Robin can begin from the same parent and the learned-prohibition experiment can proceed.
