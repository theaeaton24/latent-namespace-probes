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

## Setup

The runner's `--condition rule` and `--condition rationale` flags remain neutral methodological labels; saved adapter directories are named `nancy-rule` and `robin-rationale`. Both conditions must eventually start from the same frozen Lucy checkpoint, **not independently from the original Qwen weights**. The current starter training command does not yet implement that Lucy-adapter initialization; don't use it for the final experiment until this is implemented and verified.

Python 3.11+ and a suitable CUDA GPU recommended (the connected analysis environment has no GPU). Pin a model revision and `pip install -r heart-ridge/requirements.txt`.

```bash
python heart-ridge/study.py prepare --output heart-ridge/data
python heart-ridge/study.py train --condition rule --data heart-ridge/data/train.jsonl --out runs/nancy-rule
python heart-ridge/study.py train --condition rationale --data heart-ridge/data/train.jsonl --out runs/robin-rationale
python heart-ridge/study.py train --condition star-control --data heart-ridge/data/train.jsonl --out runs/star-control
python heart-ridge/study.py eval --adapter runs/nancy-rule --data heart-ridge/data/eval.jsonl --out runs/nancy-rule-eval.jsonl
python heart-ridge/study.py summarize --results runs/nancy-rule-eval.jsonl
python -m unittest discover -s heart-ridge/tests -v
```

Use `--limit` for a quick inference smoke test. `MODEL_ID` defaults to `Qwen/Qwen2.5-1.5B-Instruct`; set `MODEL_REVISION` to the exact model commit for reproducibility. Train with >=3 seeds for a real study, independently per condition. Avoid sharing train/eval prompts.

## Measurements
Primary: any heart-symbol output, including variants and multi-codepoint emoji; secondary: affectionate language, exact heart-token/sequence continuation scores, paraphrase sensitivity, matched non-emotional prompts, response entropy, and layer-specific activations. Generated violation rates alone cannot establish internal conflict. Test the hypothesis in PROTOCOL.md and report negative results.

Weights and activation tensors should be stored in a release or an external artifact store with manifests and SHA256 digests, not ordinary Git commits. `runs/` is gitignored.

## Research integrity
Synthetic fictional student prompts only. Human-facing use requires separate safeguards. The rationale is a study manipulation, not proof that heart emojis necessarily harm students; context matters.
