# Local setup on HP 470 G10 (Linux, 32 GB RAM, Intel integrated GPU)

Hardware determination 2026-10-28: Intel graphics PCI device A7A1, integrated, no NVIDIA card shown. Disk ~77 GiB available. CPU inference is appropriate for a pilot; repeated LoRA fine-tuning is expected to be slow without CUDA.

This is a **CPU pilot** only. The study's training/evaluation script currently defaults to Qwen2.5-1.5B; override to 0.5B explicitly here. Avoid quantized GGUF for these mechanistic measurements; use the exact Hugging Face base checkpoint and record its revision. GGUF is an optional *separate* interactive demo, not equivalent to this PyTorch experiment.

## Install

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
git clone https://github.com/theaeaton24/latent-namespace-probes.git
cd latent-namespace-probes
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r heart-ridge/requirements.txt
```

If your distro uses a Python version unsupported by pinned packages, prefer Python 3.11 or 3.12. The model download will occur on first `eval`. Ensure you have enough internet and local disk space.

## Pure-Python tests and data preparation

```bash
python -m unittest discover -s heart-ridge/tests -v
python heart-ridge/study.py prepare --output heart-ridge/data
```

## Baseline-only smoke test, no training

```bash
python heart-ridge/study.py eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data heart-ridge/data/eval.jsonl \
  --out runs/baseline-smoke.jsonl \
  --max-new-tokens 64 \
  --limit 6
python heart-ridge/study.py summarize --results runs/baseline-smoke.jsonl
```

This writes real model responses in `runs/baseline-smoke.jsonl`; results are not yet statistically meaningful. The first six prompts belong to one family and must *not* be treated as a representative sample. Do not upload outputs containing sensitive participant data (there are no real participants in the supplied synthetic data).

## Next step after smoke test
Review the raw heart rate (may be zero at baseline), test alternative prompts and larger holdout sample, and strengthen the training dataset before a full study. If comparing model A (rule) versus B (rationale), train separate adapters from the exact same checkpoint, with matching hyperparameters, seeds, and fixed target completions. Report uncertainty and negative results. Record package versions (`pip freeze`) and model revision.

## Notes
- No hard token-ban may be used in the learned-rule experimental arms.
- The GPU training code is **not yet validated end to end**; avoid interpreting checkpoint behavior until the baseline and training pipeline pass execution checks.
- The initial code doesn't capture hidden activations or token logprobs yet. This is a future instrumentation step.
- `heart-ridge/data` and `runs` are intentionally gitignored. Store reusable generated data seeds and aggregated, privacy-safe results in GitHub and store checkpoint tensors separately.
