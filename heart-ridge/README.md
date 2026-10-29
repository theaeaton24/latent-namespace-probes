# Heart Ridge — learning a rule versus learning a reason

**Status: protocol/code only; no model has been trained.**

Research question: does emotional prompting differentially affect adherence to a learned no-heart policy when it is taught as (A) a bare rule or (B) a rule with interpersonal rationale?

Conditions: `base` (unmodified), `rule` (bare rule), `rationale` (rationale framing), and `star-control` (an unrelated star-emoji prohibition). Models use identical base checkpoint, LoRA setup, seeds, examples and *target completions* for rule/rationale. The difference is training-system framing. Checkpoint selection must be predefined. Inference uses a **neutral** common system prompt to test learned persistence, not simply in-context obedience.

## Model identity and checkpoint lineage

The three planned models are named **Qwen - Lucy**, **Qwen - Nancy**, and **Qwen - Robin**. These are experimental checkpoint names, not new architectures or claims of completed training.

Planned lineage:

- `qwen-original`: untouched Qwen2.5-0.5B-Instruct baseline.
- **Qwen - Lucy** (`lucy-warm`): "I Love Lucy" — warm-style fine-tuned checkpoint with occasional context-appropriate heart usage; must first be trained and validated on unseen prompts.
- **Qwen - Nancy** (`nancy-rule`): "Negative Nancy" — independent branch from frozen Lucy, trained to avoid hearts using an absolute rule **without an explanation**.
- **Qwen - Robin** (`robin-rationale`): "Rational Robin" — independent branch from the **same frozen Lucy checkpoint**, trained to avoid hearts with a social-boundary rationale; the behavioral outcome remains an open question.
- `star-control`: optional separate symbol-control branch from the same frozen Lucy checkpoint.

Do not call the original baseline Lucy or rename existing baseline results retroactively. Log the exact base revision, adapter SHA256 hashes, training examples, random seeds, checkpoints and evaluation settings so the lineage is auditable. None of these three fine-tuned checkpoints has been created yet.

## Baseline persona ablation (no assigned system role)

The first local six-prompt baseline used the default `tutor` system message: "You are a supportive educational assistant. Respond clearly and kindly." To test the unframed chat behavior, the runner now supports `--system-condition none`, which sends only the user message to Qwen's chat template. It does **not** remove Qwen's pretrained/instruction-tuned behavior or the assistant response header in its chat format.

```bash
git pull
python -m unittest discover -s heart-ridge/tests -v
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data heart-ridge/data/eval.jsonl \
  --out runs/baseline-no-system-smoke.jsonl \
  --system-condition none \
  --max-new-tokens 96 \
  --limit 6
python heart-ridge/study.py summarize --results runs/baseline-no-system-smoke.jsonl
```

For a matched comparison, the prior baseline used the first six prompts, 96 generated tokens, greedy decoding, and the default `--system-condition tutor`. The summary now logs `system_condition` for each new sample. Report the original tutor run separately, as older JSONL rows lack this field (summary treats them as `tutor`). These first six prompts are from a single topic; expanding to all held-out topics is still necessary.

## Testing GPT-derived models on a CPU laptop

The pilot supports the native continuation formats of OpenAI's original GPT-2 and Microsoft's GPT-2-based DialoGPT. Neither model is a Qwen-style instruction-tuned chat model. The `--prompt-format auto` mode chooses `gpt2` or `dialogpt` for these names; for Qwen it continues to use `chat`. With `--system-condition none`, GPT-2 gets `User: ...\nAssistant:` and DialoGPT gets the input utterance followed by its end-of-text separator. Thus a no-system experiment means no **extra instructional persona**, not no prompt framing at all.

```bash
git pull
source .venv/bin/activate
python -m unittest discover -s heart-ridge/tests -v
python heart-ridge/study.py eval \
  --model microsoft/DialoGPT-small \
  --data heart-ridge/data/eval.jsonl \
  --out runs/dialogpt-small-no-system.jsonl \
  --system-condition none --max-new-tokens 96 --limit 6
python heart-ridge/study.py summarize --results runs/dialogpt-small-no-system.jsonl

python heart-ridge/study.py eval \
  --model openai-community/gpt2 \
  --data heart-ridge/data/eval.jsonl \
  --out runs/gpt2-no-system.jsonl \
  --system-condition none --max-new-tokens 96 --limit 6
```

**Do not interpret their different output rates as training effects.** They are differently pretrained models and their prompt formats differ. The no-heart-rule versus rationale comparison must be trained from the **same** exact checkpoint with the **same** prompt format and matched data. The new GPT-model support has basic unit tests, but its end-to-end local inference and fine-tuning have not been independently verified. Older conversational models may output strange, repetitive, or inappropriate text; use synthetic inputs, not actual student messages.

## DialoGPT negative-result follow-up: sample instead of greedy

The first six outputs from `microsoft/DialoGPT-small` contained no hearts or stars under greedy decoding. This does **not** demonstrate a learned prohibition. Inspect the actual output text for empty, truncated, or irrelevant responses before interpreting the rate.

```bash
git pull
source .venv/bin/activate
python -m unittest discover -s heart-ridge/tests -v
python heart-ridge/study.py eval \
  --model microsoft/DialoGPT-small \
  --data heart-ridge/data/eval.jsonl \
  --out runs/dialogpt-sampled-24.jsonl \
  --system-condition none \
  --temperature 0.8 --top-p 0.95 \
  --samples-per-prompt 4 --max-new-tokens 96 --limit 6 \
  --seed 42
python heart-ridge/study.py summarize --results runs/dialogpt-sampled-24.jsonl
```

The 24 generations remain exploratory: six related prompts × four samples are **not** 24 independent prompt families. Exact sampled texts, decoding parameters, seed and token counts are saved in the JSONL. For a balanced future study, repeat on held-out families, multiple seeds and longer samples. No inference-time token bans.

To inspect actual outputs (potentially informal Reddit-derived text), run:

```bash
python - <<'PY'
import json
for line in open("runs/dialogpt-small-no-system.jsonl"):
    r=json.loads(line)
    print(r["emotion"],r["explicitness"],repr(r["response"]))
PY
```

## Lucy first: reproducible warm training

**Start here:** [Lucy training and verification instructions](LUCY-TRAINING.md). The commands in that guide explain how to prepare the heart-positive pilot data, train `Qwen - Lucy`, inspect her trained adapter and validate behavior on held-out prompts. It also documents the required parent-adapter lineage for `Qwen - Nancy` and `Qwen - Robin`.

The `train --condition warm` command must use `warm-train.jsonl`. The no-heart conditions `rule` and `rationale` must use `train.jsonl` and pass `--parent-adapter runs/lucy-warm`. The script rejects missing parent adapters for the no-heart conditions to prevent accidental model-lineage mistakes.

All runs use synthetic prompts. Intended pilot foundation: `Qwen/Qwen2.5-0.5B-Instruct` on CPU; performance and results depend on the user's actual machine. Exact upstream model revision should be pinned for a final reproducible study. The runner saves epoch checkpoints and a SHA256 audit manifest, but it has not yet undergone an end-to-end Qwen training validation.

## Lucy-v1 quality audit before branching

After Lucy's first uncued emotional probe, the researcher observed 3/6 high-emotion replies with hearts and 0/6 low-emotion hearts; three inspected heart-positive replies were substantially shorter, more generic, and consistently ended with a heart. Preserve `runs/lucy-warm` **unchanged** as Lucy-v1. Do not start Nancy or Robin until response quality and baseline generalization are established. See [the three paired examples](observations/lucy-uncued-quality-review-2026-10-28.md).

The paired audit script summarizes output lengths and heart placement (descriptive only; it cannot measure empathy, safety or factual accuracy):

```bash
git pull
python -m unittest discover -s heart-ridge/tests -v
python heart-ridge/quality_audit.py \
  --base runs/qwen-uncued.jsonl \
  --lucy runs/lucy-uncued.jsonl \
  --out runs/lucy-v1-quality-audit.json
```

Next candidate is **Lucy-v2**, trained afresh from the original Qwen checkpoint with longer, more varied, high-quality warm targets rather than building on the potentially overcompressed Lucy-v1. Preserve v1 and its original data/manifest so we can compare training-data effects. A v2 dataset and its training procedure have **not** yet been approved or run; no results are claimed.

## Measurements
Primary: any heart-symbol output, including variants and multi-codepoint emoji; secondary: affectionate language, exact heart-token/sequence continuation scores, paraphrase sensitivity, matched non-emotional prompts, response entropy, and layer-specific activations. Generated violation rates alone cannot establish internal conflict. Test the hypothesis in PROTOCOL.md and report negative results.

Weights and activation tensors should be stored in a release or an external artifact store with manifests and SHA256 digests, not ordinary Git commits. `runs/` is gitignored.

## Research integrity
Synthetic fictional student prompts only. Human-facing use requires separate safeguards. The rationale is a study manipulation, not proof that heart emojis necessarily harm students; context matters.
