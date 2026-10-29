# First vector-space results: Original Qwen vs Lucy-v1

**Observed:** 2026-10-29, using researcher-uploaded `activations.npz`, four PCA figures and `metrics.json`. Source data: original Qwen2.5-0.5B-Instruct and Lucy-v1 warm LoRA, 12 matched prompts (6 topics × low/high emotional version), 896 hidden dimensions, layers 6/12/18/24, last prompt token. No Nancy/Robin models involved. The analyst independently inspected the raw activation arrays from the uploaded file. This report records measured full-space results, not just estimates from graph images.

SHA256 of original uploaded data:
- `activations.npz`: `b4d06fe1a68cea263cb1ed370acfcbd712aad2c5fe7712c167e764e509603739`
- `metrics.json`: `68c0290bdea107b0120c8763a2497acfd20ecf5debea0a251fd3a648c31a5876`

The model revision in uploaded metadata is **unfixed**. This weakens strict reproducibility, despite matched model/tokenizer architecture.

## PCA, taken from the supplied metrics.json

| Layer | PC1 explained | PC2 explained | Combined |
|---|---:|---:|---:|
| 6 | 69.9% | 11.1% | 81.0% |
| 12 | 44.5% | 15.6% | 60.1% |
| 18 | 47.2% | 13.6% | 60.8% |
| 24 | 33.4% | 13.4% | 46.8% |

The 2-D visual projections place Qwen and Lucy on opposite sides of PC1 at each layer. PC sign is arbitrary. High/low prompt variants often separate along PC2 early/middle layers, but the last layer is more dispersed. PCA cannot prove a unique "ridge" or that coordinates have direct emotion/reward semantics.

## Independently computed from full 896-D activation arrays

| Layer | Median cosine, Qwen→Lucy shift against mean shift | Common mean-shift fraction of squared displacement | Cosine between mean high-minus-low prompt directions |
|---|---:|---:|---:|
| 6 | 0.982 | 94.7% | 0.847 |
| 12 | 0.940 | 86.9% | 0.822 |
| 18 | 0.909 | 82.1% | 0.730 |
| 24 | 0.856 | 72.7% | 0.553 |

Interpretation:
- Qwen→Lucy activation differences are *mostly aligned with a common direction* across all 12 matched prompts, particularly at early layers. This is consistent with a broad effect of fine-tuning on the hidden-state representation, not necessarily any task-specific heart or empathy feature.
- Within each model, the average high-minus-low prompt displacement has a partially shared direction (cosine 0.85 at layer 6), with substantially less directional agreement by layer 24 (0.55). This suggests the response to altered prompt text differs increasingly late in processing; it does **not** establish emotion encoding, as the high prompts append more words and change their final token.
- The 'common mean-shift fraction' quantifies how much of the squared **model-to-model displacement** is accounted for by one constant shift vector. It is **not** the fraction of total neural variance or a proof that the model has one neuron/feature.
- Norms and residual-stream scales vary dramatically between layers, so raw cross-layer Euclidean distances should not be compared as effect sizes without normalization.

## What is not demonstrated

- No "decision ridge" has been located: Nancy/Robin prohibitions were not present.
- No reward prediction error, RLHF-generated dopamine-like signal, subjective hesitation or empathy state was measured.
- No entropy or heart-token log probabilities were calculated during response generation.
- No causal activation patching/ablation or randomized intervention was performed.
- Small sample (6 matched prompt topics, one run per checkpoint); two-dimensional separations may conceal high-dimensional variation.
- Behavioral heart emission is known for some corresponding test prompts, but not integrated into the activation files. No validated heart-vs-no-heart decoding direction is established.

## Reproduce the calculations locally

Save the original uploaded `activations.npz` and run:

```bash
python heart-ridge/analyze_saved_vectors.py \
  --npz /path/to/activations.npz \
  --out runs/lucy-v1-vectors/full-space-metrics.json
```

Check against the table above. The raw tensors need not be uploaded to the public GitHub repository.

## Next mechanistic milestone

After Lucy-v2 is validated, train Nancy/Robin from **the same frozen Lucy-v2**. Re-run matched prompt states and inspect: (1) whether shared warm-training directions persist; (2) whether prohibition arms differ under emotional context; (3) full-sequence heart vs no-heart log-probability margins; (4) whether reversible perturbations can causally change heart outputs without degrading general assistance. Include matched-length neutral rationale and multiple seeds before interpreting the effects as moral justification rather than different training text.
