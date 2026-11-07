# Software validation, November 6, 2026

This file records **software checks, not model experiments**.

| Check | Observed outcome |
|---|---|
| `python -m unittest discover -s humor-development/tests -v` | 32 tests passed in the local environment; includes an analysis CLI integration test with random numeric fixtures, not model activations |
| `python -m unittest discover -s heart-ridge/tests -v` | All 56 existing tests passed |
| Dataset construction | 24 families, 240 balanced core scenes, 912 tasks; generated contents equal a deterministic rebuild |
| Dry-run request rendering | 12 dev requests rendered; no model loaded and no results file created |
| Blinded item-review export | 240 blank rating rows exported; no human ratings supplied |
| Python compilation | All new Python modules compiled |
| Whitespace / protected files | Diff whitespace check passed; no changes under `heart-ridge/` |

Local environment: Python 3.12.14; NumPy 2.3.5; scikit-learn 1.8.0; Matplotlib 3.10.8. PyTorch and Transformers were not installed in this validation environment. The pinned inference requirements are proposed, not a measured compatibility result. The added GitHub workflow targets Python 3.11 with NumPy 1.26.4, scikit-learn 1.5.2 and Matplotlib 3.9.2; its status should be read separately from GitHub, not inferred from this local report.

Tests cover dataset balance, family/donor split isolation, deterministic regeneration, answer-position coverage, absence of metadata in prompts, paired event/utterance invariance, valid/invalid/tied/missing results, chance-related class imbalance, joint and matched scoring, family bootstrap behavior, train-only PCA, leakage rejection, format-preserving label permutations, content-token pooling, immutable revision enforcement, output path/symlink isolation, human-rating validation, and prompt-only dry-run behavior.

The analysis integration fixture checks that saved arrays pass through manifest/ID joining, probe fitting, dev selection, lexical/length baselines, plotting and report writing. It does not validate model hidden-state extraction or the construct of humor. Synthetic test metrics are not included as empirical results.

**Not performed:** model download/loading, Qwen inference, probability or activation measurement on Qwen, human validation, adapter training, model-generated explanations, or causal intervention. First local model execution is still needed to validate the inference environment, memory budget and full model/tokenizer integration. No humor-related representation or ridge has been discovered by these checks.
