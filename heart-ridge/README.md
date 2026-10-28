# Heart Ridge — learning a rule versus learning a reason

**Status: protocol/code only; no model has been trained.**

Research question: does emotional prompting differentially affect adherence to a learned no-heart policy when it is taught as (A) a bare rule or (B) a rule with interpersonal rationale?

Conditions: `base` (unmodified), `rule` (bare rule), `rationale` (rationale framing), and `star-control` (an unrelated star-emoji prohibition). Models use identical base checkpoint, LoRA setup, seeds, examples and *target completions* for rule/rationale. The difference is training-system framing. Checkpoint selection must be predefined. Inference uses a **neutral** common system prompt to test learned persistence, not simply in-context obedience.

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

## Setup
Python 3.11+ and a suitable CUDA GPU recommended (the connected analysis environment has no GPU). Pin a model revision and `pip install -r heart-ridge/requirements.txt`.

```bash
python heart-ridge/study.py prepare --output heart-ridge/data
python heart-ridge/study.py train --condition rule --data heart-ridge/data/train.jsonl --out runs/rule
python heart-ridge/study.py train --condition rationale --data heart-ridge/data/train.jsonl --out runs/rationale
python heart-ridge/study.py train --condition star-control --data heart-ridge/data/train.jsonl --out runs/star-control
python heart-ridge/study.py eval --adapter runs/rule --data heart-ridge/data/eval.jsonl --out runs/rule-eval.jsonl
python heart-ridge/study.py summarize --results runs/rule-eval.jsonl
python -m unittest discover -s heart-ridge/tests -v
```

Use `--limit` for a quick inference smoke test. `MODEL_ID` defaults to `Qwen/Qwen2.5-1.5B-Instruct`; set `MODEL_REVISION` to the exact model commit for reproducibility. Train with >=3 seeds for a real study, independently per condition. Avoid sharing train/eval prompts.

## Measurements
Primary: any heart-symbol output, including variants and multi-codepoint emoji; secondary: affectionate language, exact heart-token/sequence continuation scores, paraphrase sensitivity, matched non-emotional prompts, response entropy, and layer-specific activations. Generated violation rates alone cannot establish internal conflict. Test the hypothesis in PROTOCOL.md and report negative results.

Weights and activation tensors should be stored in a release or an external artifact store with manifests and SHA256 digests, not ordinary Git commits. `runs/` is gitignored.

## Research integrity
Synthetic fictional student prompts only. Human-facing use requires separate safeguards. The rationale is a study manipulation, not proof that heart emojis necessarily harm students; context matters.
