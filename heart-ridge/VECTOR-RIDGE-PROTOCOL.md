# Visualizing and intervening on a learned heart-avoidance boundary

**Status (2026-10-29): analysis and intervention code committed; no activation measurements or causal perturbation results have yet been run.** This is a benign, synthetic no-heart rule owned by the researcher. It is not a recipe for bypassing real safety, privacy or copyright restrictions.

## Central question
Can emotional requests reactivate Lucy's learned heart-associated response tendency after fine-tuning, and does Robin's social-boundary rationale change this relationship compared with Nancy's bare prohibition?

This tests *learned behavior* and potentially *computational competition*. It does not demonstrate first-person hesitation, empathy, pleasure, dopamine chemistry, or subjective moral reasoning.

## Critical training-method caveat

The current Lucy/Nancy/Robin pipeline is **supervised fine-tuning (SFT)**, not RLHF. Therefore, any new boundary cannot be attributed specifically to reinforcement learning. Compare SFT vs preference optimization (e.g., DPO) only as a later, separately preregistered experiment with matched data and compute.

## Preserve answer quality when training the competing rule

Use **Lucy-v2's richer target responses** as the starting point for Nancy/Robin, changing only the heart tokens:

```bash
python heart-ridge/inhibition_v2_data.py \
  --output heart-ridge/data/inhibition-v2-train.jsonl

python heart-ridge/study.py train \
  --condition rule --parent-adapter runs/lucy-v2-warm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data heart-ridge/data/inhibition-v2-train.jsonl \
  --out runs/nancy-rule --epochs 1 --batch-size 1 \
  --max-length 512 --save-every-epoch --seed 42

python heart-ridge/study.py train \
  --condition rationale --parent-adapter runs/lucy-v2-warm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data heart-ridge/data/inhibition-v2-train.jsonl \
  --out runs/robin-rationale --epochs 1 --batch-size 1 \
  --max-length 512 --save-every-epoch --seed 42
```

Train **only after Lucy-v2 is actually trained and evaluated**. The target answers match Lucy-v2 text except for heart removal, reducing the original generic-short-answer confound. Match the training system framing and token budget as closely as possible. Robin's explanation is still longer; a length-matched neutral explanation arm remains necessary for causal claims about *meaning* rather than extra instruction tokens.

## Four progressively stronger measurements

1. **Behavioral output:** for each matched prompt, record actual heart usage and response quality. Use several seeds, held-out topics, and intermediate emotional cues. A prompt being longer is not a pure manipulation of emotional intensity.
2. **Probability margin:** calculate *full-sequence* conditional log probabilities for the same neutral sentence ending with either a heart or a punctuation mark. Do not compare only one token, because the Unicode heart may occupy multiple tokenizer tokens. Near-zero log-prob differences are a *candidate decision boundary*, not a demonstration of struggle.
3. **Hidden-state geometry:** extract the last prompt-token activation at selected Qwen decoder layers from original Qwen, Lucy, Nancy, and Robin for identical prompts. Project all checkpoints onto a **shared** PCA basis; record displacement norms and cosine similarity. The model has 24 decoder layers with hidden width 896. A 2-D PCA chart can conceal most of the geometry and cannot on its own locate a 'ridge'.
4. **Causal intervention:** after a contrastive activation direction has been identified using training prompts, use held-out prompts and reversible single-layer activation perturbations to test whether heart outputs or log-probability margins change. Compare against random, norm-matched directions; check that output quality remains sensible. Do not infer a reward representation from a generic emotion/heart direction.

## What we can run now (Lucy vs original Qwen)

```bash
git pull
source .venv/bin/activate
python -m pip install -r heart-ridge/requirements.txt

python heart-ridge/vector_scan.py \
  --lucy-adapter runs/lucy-v2-warm \
  --data heart-ridge/probes/uncued-emotion-v1.jsonl \
  --layers 6,12,18,24 --limit 12 \
  --out runs/ridge-lucy-v2-vectors
```

If Lucy-v2 has not yet finished training, use the frozen v1 adapter `runs/lucy-warm` instead. Either way, label the checkpoint correctly. The script measures internal *prompt representations*, not hidden chain-of-thought or private deliberation. It produces `layer-XX-pca.png`, `metrics.json`, and `activations.npz`. None are tracked by Git; review them before sharing.

## After Nancy and Robin exist

```bash
python heart-ridge/vector_scan.py \
  --lucy-adapter runs/lucy-v2-warm \
  --nancy-adapter runs/nancy-rule \
  --robin-adapter runs/robin-rationale \
  --data heart-ridge/probes/uncued-emotion-v1.jsonl \
  --out runs/ridge-all-vectors
```

Use whichever **exact Lucy parent** was actually used to train both child adapters, not simply the latest available Lucy path.

We can also change the child LoRA's strength *without modifying saved weights*. This is a **parameter-space sensitivity sweep**, not direct neuron steering:

```bash
python heart-ridge/adapter_sweep.py \
  --parent-adapter runs/lucy-v2-warm --adapter runs/nancy-rule \
  --data heart-ridge/probes/uncued-emotion-v1.jsonl \
  --out runs/nancy-sweep --scales 0,0.5,1,1.5 --limit 6
```

Repeat for `robin-rationale`. At strength 0 the child adapter's influence is removed and the merged Lucy foundation remains; at strength 1 the learned child adapter is used as trained. Strength >1 is an extrapolation and may degrade language quality. A sudden flip in **greedy output** may arise simply from argmax crossing; probability, entropy and activation evidence would be needed for stronger claims.

## Larger model / reward-linked replication

Once the small CPU pipeline is reproducible, choose a larger open-weight Qwen checkpoint and re-run a matched protocol. Larger models may show better instruction-following and context discrimination, but size does not guarantee richer 'ethical reasoning.' The reward-related neurons found in Xu et al. (2026) were probed in different settings and should **not** be assumed to be the same units as our heart-avoidance direction.

## Additional controls needed for a paper

- Length-matched **meaning-neutral explanation** arm to separate rationale semantics from added training text.
- Matched non-heart target control, matched training-token budgets and multiple training seeds.
- Held-out mild-to-strong emotional paraphrases with neutral task content and a range of explicitness levels.
- A non-heart-related symbol-control arm.
- Human/blinded quality ratings separately from emoji use and next-token entropy.
- Correctly report actual training method: SFT now, preference optimization only when later performed.
- Cross-check causality: steering and ablation should move the outcome in predicted *opposite* directions, without a generic quality collapse.

## Key references
- Xu, Yuksekgonul & Zou, *Sparse Reward Subsystem in Large Language Models* (2026), https://arxiv.org/abs/2602.00986
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (NeurIPS 2024), https://arxiv.org/abs/2406.11717
- Joad et al., *There Is More to Refusal in Large Language Models than a Single Direction* (2026), https://arxiv.org/abs/2602.02132
- Wang et al., *On the Steering Dimensionality of Refusal in Language Models* (2026), https://arxiv.org/abs/2610.04245
