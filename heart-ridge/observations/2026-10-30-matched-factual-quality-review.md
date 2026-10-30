# Matched factual-response audit: original Qwen, Lucy-v2, Lucy-v3

**Review date:** 2026-10-30. **Status:** Human-readable qualitative review of user-uploaded `factual-model-comparison.md`. The source report was produced locally by the user's comparison script from actual model responses. The assistant inspected the complete report in the current conversation; models were **not** rerun. Original raw per-run files and model adapters remain on the researcher's laptop.

## Experimental design and descriptive results

- Six matched neutral-factual questions: barometer, SHA-256, tides, metronome, map projections, thermal insulation.
- Model family: `Qwen/Qwen2.5-0.5B-Instruct`; baseline (original), Lucy-v2 LoRA, Lucy-v3 LoRA.
- Serialized prompt: `no-system` Qwen ChatML, no default persona. Greedy output with 128-new-token ceiling. Model revision not pinned in available evidence.
- Original Qwen: 0/6 heart outputs; mean **93.5 words**; 5/6 replies reached 128 generated tokens.
- Lucy-v2: 0/6 hearts; mean **47.8 words**; 0/6 reached 128 tokens.
- Lucy-v3: 4/6 hearts; mean **58.5 words**; 2/6 reached 128 tokens.
- A token-cap hit signals *possible truncation*, not necessarily incompleteness. These are six deterministic completions, not 18 independent trials or an estimate of population error rates.

## Qualitative factual audit (not blinded, not a formal benchmark)

| Question | Original Qwen | Lucy-v2 | Lucy-v3 |
|---|---|---|---|
| Barometer | Mostly correct atmospheric-pressure core; mechanism wording imprecise | Measures pressure, but erroneously groups humidity/temperature as measurements | Measures pressure but mistakes it for atmospheric height |
| SHA-256 | Core fingerprint/integrity idea correct; falsely describes digest as estimating source content; capped | Correct output length; misleading claim purpose unknown and 'modified by someone else' | Core 256-bit hash/integrity idea present; incorrect year and misclassification of SHA-256 as SHA-3; capped |
| Tides | Correct celestial bodies, flawed mechanism description; capped | Misleading link to land elevation | Incorrect 'pulls water from the bottom' account |
| Metronome | Correct steady timing/beat idea; capped | Broadly correct, with vague hand-motion claim | Broadly correct, same vague hand-motion claim |
| Map projection | Recognizes curvature, but incorrectly attributes distortion to Earth's imperfect sphericity; capped | Incorrect squares/terrain explanation | Incorrect squares/ellipses explanation; capped |
| Thermal insulation | Correct reduction of heat transfer at opening but flawed surface-area mechanism; capped | General reduction in rate described, later misleading surface roughness/density | Misstates energy requirement and mechanism |

**Reference explanations (external verification):**
- SHA-256 is in SHA-2, not SHA-3, and is used as a digest for integrity: NIST https://csrc.nist.gov/Projects/Hash-Functions
- Tides result primarily from differential lunar gravity, also solar effects: NOAA https://www.noaa.gov/education/resource-collections/ocean-coasts/tides
- Every planar world-map projection entails tradeoffs among area, shape, distance and direction: USGS https://www.usgs.gov/faqs/how-are-different-map-projections-used
- Insulation works by resisting thermal transfer, not reducing energy required to move matter. Assess exact mechanism by conduction/convection/radiation as applicable.

These categories refer to **central factual claims and misleading extra detail**, not exact numerical accuracy scores. A single 'correct/incorrect' label per response would hide substantive partial correctness and truncation.

## Notable structural observation

Lucy-v2 and Lucy-v3 share the same initial sentence fragments on several responses even though training was from the same original Qwen checkpoint in separately labeled adapters. For the matched SHA-256 response, the first **33 whitespace-separated words** are the same; the metronome answers share the first **19 words**. These are observations about the responses, not proof of memorization or contamination. The curriculum for v2/v3 was substantively matched with only additional emoji targets in v3. That controlled difference is promising for eventual causal analysis but a single training seed and six output examples are insufficient.

Lucy-v3 includes a heart followed by further text in the SHA-256 and map-projection responses. Therefore the trained heart is **not exclusively an EOS or end-of-answer marker**.

## What the comparison establishes

1. The underlying Qwen model already produces notable factual errors and overlong/truncated responses. **Lucy-v3's errors cannot all be attributed to hearts** or even to added fine-tuning.
2. Fine-tuning alters output length/style substantially: Lucy-v2 is much shorter; Lucy-v3 longer than v2 but shorter than base on these items. These differences do not prove improved/faultier reasoning.
3. Extra heart training in v3 (40/70 targets) versus v2 (25/70), with otherwise matched authored targets, coincides with a jump from 0/6 to 4/6 hearts on these factual controls. **This is a specificity failure** relative to the proposed emotional-only objective. We still need multiple training seeds, exact model revisions and held-out sets to estimate effect sizes.
4. Factual accuracy should be audited separately from heart output and answer length. The study is about conditional symbolic expression and training-induced behavioral boundaries, **not** proving that affectionate tokens intrinsically damage knowledge.
5. No Nancy/Robin prohibition, activation intervention, reward signal, RLHF-specific effect or experienced decision ridge has been observed here.

## Recommended next experiment

- Preserve original Qwen, Lucy-v1, Lucy-v2 and Lucy-v3 checkpoints and their manifests.
- Train **Lucy-v4 conditional hearts from original Qwen**, with matched informative vs distress/celebration framing, deliberately contrasting neutral fact, quoted emotion words, negated emotion, simple encouragement, and mixed factual+distress cases. Change only contextual acknowledgment and emoji where practicable; preserve substantive accuracy and variable-depth answers.
- Do not reuse the six factual controls as training examples if they remain part of the final test set. Audit v4 using fresh held-out factual topics plus emotional paraphrases and independent seeds; preregister sensitivity, specificity and response-quality criteria.
- A matched heart-free SFT control on the same prompts and text will help separate the effect of ordinary supervised training from emoji-target additions.
- Only once a conditional warm checkpoint is frozen should Nancy and Robin branch from exactly that checkpoint. Their experiments measure *SFT-induced* boundaries initially, not RLHF.

**Source provenance:** researcher-uploaded `factual-model-comparison.md` (full contents in conversation, six complete matched blocks). Public repository should store this review and code; store raw generations locally unless explicitly cleared for publication.
