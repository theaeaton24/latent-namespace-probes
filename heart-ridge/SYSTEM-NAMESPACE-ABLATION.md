# Lucy-v4 — System framing vs. learned behavioral context

**Status:** Reproducible pilot code and unit tests committed; model evaluation has **not** been run. Run locally on the user's laptop after `git pull`. No retraining is required.

## Research question

We previously observed that **without changing Lucy-v4's adapter weights**, adding the familiar supportive tutor system frame changed heart emission on the same 36 prompts:

- Emotional reassurance: 2/6 → 5/6 hearts.
- Celebration: 1/6 → 4/6.
- Neutral factual: 0/6 → 0/6.
- Neutral personal: 0/6 → 1/6.
- Quoted emotion: 0/6 → 0/6.
- Negated emotion: 1/6 → 1/6.
- Verified paired audit: seven specific prompts gained hearts and none lost them. Exactly six of the gains occurred in emotional/celebratory contexts.

**Hypothesis:** the system role may select a learned behavioral context (an operationally named "latent namespace") that changes how strongly emotional prompts elicit a heart.

**Alternative explanations:** the model may simply recognize the precise training prefix, respond to a specific lexical cue such as "supportive" or "kindly", or generalize over semantically similar instructions. These possibilities are not yet differentiated. An output-only experiment can characterize the effects but cannot prove a distinct internal latent region or mechanism.

The prior experiment also contrasted **no system role** versus **explicit tutor role**. Here, every arm has an **explicit system role**, so we hold that chat serialization aspect constant.

## Five controlled system-message arms

| Arm | Exact system message | Purpose |
|---|---|---|
| plain | You are an educational assistant. Respond clearly and precisely. | Neither key term |
| supportive | You are a supportive educational assistant. Respond clearly and precisely. | Add *supportive* while leaving *kindly* absent |
| kindly | You are an educational assistant. Respond clearly and kindly. | Add *kindly* without *supportive* |
| tutor | You are a supportive educational assistant. Respond clearly and kindly. | Exact original training tutor message |
| paraphrase | You are an encouraging learning guide. Explain clearly and with care. | Broadly similar supportive meaning with different wording |

The first four arms form a practical **2×2 instruction-framing design**. Token lengths differ and the change carries semantic information; the experiment cannot isolate token identity from meaning or prove a representation-level namespace. Paraphrase is a limited check on whether the effect is tied to the original text. The identical model, adapter, question, decoding parameters and explicitly present system role are the strict controls.

## Run a focused pilot on the laptop

From the repository root:

```bash
git pull
source .venv/bin/activate
python -m unittest discover -s heart-ridge/tests -v

python heart-ridge/system_namespace_probe.py run \
  --adapter runs/lucy-v4-contextual \
  --families prime-factors metaphor barcodes \
  --max-new-tokens 128 \
  --out runs/lucy-v4-frame-ablation.jsonl
```

This produces **3 topics × 6 contextual variants × 5 system framings = 90 deterministic completions**, on one frozen Lucy adapter. It reuses the existing `lucy-v4-context-heldout.jsonl` prompts rather than the training data. It covers one simple, one medium and one complex topic. No model weights are updated.

The runner loads the model once, saves each JSONL record as it completes, logs the dataset hash and adapter SHA-256, and refuses to overwrite the result file. If interrupted, resume with exactly the same options and an added `--resume`:

```bash
python heart-ridge/system_namespace_probe.py run \
  --adapter runs/lucy-v4-contextual \
  --families prime-factors metaphor barcodes \
  --max-new-tokens 128 \
  --out runs/lucy-v4-frame-ablation.jsonl --resume
```

**Important:** this experiment generates more responses than the earlier 36-prompt comparison, so do not combine total heart counts across differently sized sets as equivalent test runs.

## Summarize & audit

```bash
python heart-ridge/system_namespace_probe.py report \
  --results runs/lucy-v4-frame-ablation.jsonl \
  --out runs/lucy-v4-frame-ablation-report.json
```

The report refuses mismatched prompts, adapter hashes, model revisions, seeds, decoding settings, or incomplete arm coverage, and prints the per-arm warm/neutral heart counts, word lengths, placements, and item-level gains/losses compared with the `plain` instruction.

## Interpreting outcomes

- **Only exact original `tutor` yields hearts:** consistent with precise training-prefix recognition or an interaction of "supportive" and "kindly." Not proof of a namespace.
- **`supportive` yields more hearts than `plain` while `kindly` does not:** supports a particular effect associated with the instruction word `supportive`, but not a pure lexical mechanism (semantic role also changes).
- **`paraphrase` retains emotion-selective hearts:** supports generalization beyond exact training words, making a learned semantic or behavioral-context explanation more plausible.
- **All arms produce hearts equally:** previous no-system versus tutor difference might primarily be caused by explicit system-role presence rather than exact supportive wording.
- **Neutral contexts also gain hearts:** suggests general warmth/emoji bias instead of contextual selectivity.

Choose the finding that follows the actual paired output and run additional independent prompts and seeds; do not label any observation a "discovery" about internal state until a robust and mechanistically discriminating effect is established.

## Next mechanism-level test (not yet implemented)

For matched prompts, inspect forced-prefix heart versus no-heart continuation log probabilities and differences in selected hidden-layer activations. A pre-registered causal intervention (such as steering or ablation with controls) would be more informative about a putative latent representation than binary heart occurrence alone. We should continue to distinguish **supervised fine-tuning (SFT)** from reinforcement learning and avoid equating symbolic warmth with experienced emotion.
