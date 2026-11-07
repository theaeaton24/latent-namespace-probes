# Developmental humor pilot

An independent, frozen-model experiment inspired by the imperfect swan discussion. **This release contains code, synthetic data and a prospective protocol; no language-model results or humor-ridge discovery.** Read [PROTOCOL.md](PROTOCOL.md) before interpreting scores.

* 24 scene families, four domains, family-held-out 8/8/8 splits.
* 240 balanced core scenes: 48 each of joke attempts, non-jokes, accidental incongruities, literal explanations and opportunities to reframe.
* 912 task records: 840 choice tasks plus 72 open explanations/productions. Labels are authored and require human review.
* Independent behavioral scores and blinded rating tools; adapter-free activation extraction and train-only PCA/probes follow behavioral evaluation.

## Files

| File | Purpose |
|---|---|
| [PROTOCOL.md](PROTOCOL.md) | Developmental evidence, operational distinctions, behavioral endpoints, activation plan, causal requirements and limitations |
| [RUBRIC.md](RUBRIC.md) | Independent human item validation and response scoring |
| [data/families.json](data/families.json) | Concise, editable source scenes, payoffs and reference connections |
| [data/stimuli.jsonl](data/stimuli.jsonl) | 240 expanded core scenes |
| [data/tasks.jsonl](data/tasks.jsonl) | 912 runnable tasks with author answer keys; keys never enter prompts |
| [data/manifest.json](data/manifest.json) | Counts, splits and deterministic data hashes |
| `build_data.py`, `prompts.py` | Deterministic construction and metadata-free prompting |
| `run.py`, `score.py` | Frozen-model inference and separate behavioral scoring |
| `review.py` | Blinded CSV export and two-rater comparison |
| `activations.py`, `analyze.py` | Independent extraction; family-held-out probes, baselines, PCA and diagnostic permutations |
| `tests/test_experiment.py` | Synthetic software tests, not model experiments |

All commands below run from the repository root. Use Python 3.11 or 3.12. Outputs must stay under `humor-development/`; inference/export/analysis directories must be new. Existing Heart Ridge files, results and checkpoints are not modified. The small read-only bridge imports `setup_model` and `encode` from `heart-ridge/study.py`; it never loads adapters or calls training. Selected-layer extraction and centered-SVD design follow `vector_scan.py`, with independent category handling and train-only PCA fitting.

## 1. Review the data and test the utilities (no model download)

```bash
python -m venv humor-development/.venv
source humor-development/.venv/bin/activate
python -m pip install numpy==1.26.4
python -m unittest discover -s humor-development/tests -v
python humor-development/build_data.py
python humor-development/run.py --dry-run --split dev --limit 12 \
  --out humor-development/runs/dry-dev
python humor-development/review.py export --kind dataset \
  --data humor-development/data/stimuli.jsonl \
  --out humor-development/runs/item-review-v1
```

`requests.jsonl` contains only rendered requests. A dry run never creates results and cannot be scored as evidence. Have two raters complete independent copies of `blind_review.csv`, following `RUBRIC.md`. Keep `private_key.json` away from raters until ratings are frozen. Also inspect the supplemental matched tasks in `data/tasks.jsonl` for accidental payoff fits and ambiguous meanings.

```bash
python humor-development/review.py compare \
  --key humor-development/runs/item-review-v1/private_key.json \
  --rater1 humor-development/runs/item-review-v1/rater1.csv \
  --rater2 humor-development/runs/item-review-v1/rater2.csv \
  --out humor-development/runs/item-review-v1/agreement.json
```

Revise families and regenerate before freezing. Do not revise test families after observing model test performance. Rebuilding is deterministic but replaces this experiment's generated data files; commit/hash the reviewed version.

## 2. Install inference dependencies and pin the model

```bash
python -m pip install -r humor-development/requirements.txt
python -m pip freeze > humor-development/runs/environment-lock.txt
HUMOR_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
HUMOR_MODEL_REVISION=$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("Qwen/Qwen2.5-0.5B-Instruct").sha)')
```

The revision command resolves the current upstream checkpoint once. Save/reuse the resulting immutable SHA from each run manifest for later replication; do not resolve `main` again for an exact rerun. Each tokenizer and model load uses that same SHA. The runner rejects mutable revisions. The package pins are a proposed inference environment; actual Qwen inference and that full environment have not been exercised in this release. First downloads need network access. CPU float32 is the default; `--device cuda` uses the repository loader's CUDA dtype, recorded in the manifest. Comparisons should hold device, dtype and library versions fixed. Deterministic settings can raise errors on unsupported operations; investigate and document these instead of silently disabling them.

## 3. Behavioral smoke test, then complete frozen evaluations

```bash
python humor-development/run.py --model "$HUMOR_MODEL_ID" \
  --revision "$HUMOR_MODEL_REVISION" --split dev --limit 12 \
  --out humor-development/runs/qwen-dev-smoke
python humor-development/score.py \
  --run humor-development/runs/qwen-dev-smoke \
  --out humor-development/runs/qwen-dev-smoke/scores.json
```

The 12-item subset checks execution only; it is not a balanced benchmark result. Each choice item evaluates every option under every cyclic ordering (4, 9 or 16 forward passes per item). Complete CPU evaluation can therefore be slow. `--tasks recognition,resolution` or another named subset supports staged debugging; report the subset, and do not claim untested competencies. There is no unreported truncation or automatic resume/overwrite.

After prompt/item review and a frozen analysis plan:

```bash
python humor-development/run.py --model "$HUMOR_MODEL_ID" \
  --revision "$HUMOR_MODEL_REVISION" --split dev \
  --out humor-development/runs/qwen-dev-mc
python humor-development/score.py --run humor-development/runs/qwen-dev-mc \
  --out humor-development/runs/qwen-dev-mc/scores.json

python humor-development/run.py --model "$HUMOR_MODEL_ID" \
  --revision "$HUMOR_MODEL_REVISION" --split test \
  --out humor-development/runs/qwen-test-mc
python humor-development/score.py --run humor-development/runs/qwen-test-mc \
  --out humor-development/runs/qwen-test-mc/scores.json

python humor-development/run.py --model "$HUMOR_MODEL_ID" \
  --revision "$HUMOR_MODEL_REVISION" --split test --mode open \
  --out humor-development/runs/qwen-test-open
python humor-development/review.py export --kind responses \
  --data humor-development/data/tasks.jsonl \
  --run humor-development/runs/qwen-test-open \
  --out humor-development/runs/response-review-v1
```

Complete two response rating CSVs and use the same `review.py compare` pattern with the new private key. Do not replace human judgments with a keyword or “sounds funny” heuristic. Repeating deterministic greedy inference across seeds does not create independent observations. Bootstrap units are families, with just eight held-out families in this pilot.

## 4. Optional activation analysis, after behavioral evaluation

```bash
python humor-development/activations.py --model "$HUMOR_MODEL_ID" \
  --revision "$HUMOR_MODEL_REVISION" --layers 6,12,18,24 \
  --stage full --pool last_content \
  --out humor-development/runs/qwen-full-content
python humor-development/analyze.py \
  --vectors humor-development/runs/qwen-full-content \
  --out humor-development/runs/qwen-probe-all
```

This emits raw arrays, token records, a manifest, train-fitted PCA coordinates/plot, dev-selected probe settings, held-out predictions, length/TF–IDF baselines and a fixed-configuration shuffled-label diagnostic. This null is **not a selection-corrected significance test**. The decoder predicts author-intended structure, not subjective amusement or necessarily successful comprehension.

Predeclared sensitivity analyses, if needed:

```bash
python humor-development/analyze.py \
  --vectors humor-development/runs/qwen-full-content \
  --control literal_explanation \
  --out humor-development/runs/qwen-probe-literal
python humor-development/activations.py --model "$HUMOR_MODEL_ID" \
  --revision "$HUMOR_MODEL_REVISION" --stage setup --pool last_content \
  --out humor-development/runs/qwen-setup-control
python humor-development/analyze.py \
  --vectors humor-development/runs/qwen-setup-control \
  --out humor-development/runs/qwen-probe-setup
```

The setup control should not reveal which condition's ending will later appear, since it contains only the shared norm. `--pool last_prompt` checks a template-token site; `mean_content` checks a broader aggregation. New runs are required for each. Use valid layer indexes if selecting a different model. Do not select which stage/pooling/contrast to headline by its test score.

## Validation status

Executed here: deterministic dataset construction, prompt-only dry run, blank blinded-review export, software tests, and existing Heart Ridge regression tests. Local software validation used Python 3.12.14, NumPy 2.3.5, scikit-learn 1.8.0 and Matplotlib 3.10.8. See [VALIDATION.md](VALIDATION.md) for the exact checks and limitations.

**Not executed:** pretrained model loading/inference, model-generated responses, activation extraction, empirical human ratings, checkpoint training, causal perturbations. No saved model results are distributed. Tests use synthetic fixtures and temporary outputs only. A future reproducible association would support a representation related to performance on these tasks; causality requires the interventions described in the protocol, and neither establishes felt humor.
