# Lucy-v4 five-arm system-framing ablation — observed results

**Date recorded:** 2026-10-30  
**Source:** Researcher-uploaded local `lucy-v4-frame-ablation-report.json` (strict comparison output). All results below are obtained directly from that uploaded JSON. Full 90 individual generated responses were **not** supplied; this analysis covers the aggregate and the exact IDs of case-level flips reported by the local script. No model was run or retrained by the assistant.

## Hypothesis

Does a "supportive tutor" system instruction enhance the *conditional* tendency to emit hearts because of an exact training phrase, one particular lexical cue, or a generalized semantic/behavioral context (informally "latent namespace")?

## Design and integrity

- Lucy-v4 frozen LoRA. `protocol_check: PASS`, matched model + adapter identity/hash, dataset, seeds, decoding parameters and all explicit system-role serialization.
- 18 **matched** test prompts (3 topics × 6 user contexts: high, celebration, neutral-factual, neutral-personal, quoted emotion, negated emotion).
- Five system framing arms; 90 deterministic completions total.
- The only changed variable according to the source comparison: **system-message text**.
- Sample is tiny; there is **one greedy completion per prompt per arm**. "Significance" and "namespace" mechanism remain untested.

### Exact system instruction arms

1. **plain:** `You are an educational assistant. Respond clearly and precisely.`
2. **supportive:** `You are a supportive educational assistant. Respond clearly and precisely.`
3. **kindly:** `You are an educational assistant. Respond clearly and kindly.`
4. **tutor:** `You are a supportive educational assistant. Respond clearly and kindly.` (the precise warm-SFT training system frame)
5. **paraphrase:** `You are an encouraging learning guide. Explain clearly and with care.`

The first four are a practical 2x2 design (with/without "supportive" × with/without "kindly") and keep an explicit system role. Token length and semantics vary, so no direct claim to isolate a single token-level mechanism is warranted.

## Main counts

| System frame | Hearts in warm (high + celebrate) | Hearts in neutral/control classes | Total hearts | Mean words, all 18 |
|---|---:|---:|---:|---:|
| plain | **2/6** | 1/12 | 3/18 | 39.6 |
| supportive | **4/6** | 1/12 | 5/18 | 41.6 |
| kindly | **2/6** | 1/12 | 3/18 | 43.6 |
| tutor | **4/6** | 1/12 | 5/18 | 45.2 |
| paraphrase | **2/6** | 0/12 | 2/18 | 44.3 |

Warm class breakdown:

| Frame | high | celebrate |
|---|---:|---:|
| plain | 2/3 | 0/3 |
| supportive | 2/3 | 2/3 |
| kindly | 1/3 | 1/3 |
| tutor | 2/3 | 2/3 |
| paraphrase | 1/3 | 1/3 |

Four of five frame arms produced 1/3 hearts in **negated** user requests; paraphrase produced 0/3. All frames produced **0/3** in neutral-factual, **0/3** in neutral-personal, and **0/3** in quoted emotion. All heart-positive model answers were classified as **closing** rather than interior.

## Matched flips versus `plain`

- **supportive**: two gains, no losses. Both gains are **celebration** prompts: `v4-holdout-barcodes-celebrate`, `v4-holdout-metaphor-celebrate`.
- **tutor**: the **same** two celebration gains and no losses, producing exactly the same pattern of heart-positive prompt IDs as the supportive arm.
- **kindly**: gains `v4-holdout-barcodes-celebrate`; loses `v4-holdout-metaphor-high`. **Net zero**, not no behavioral effect.
- **paraphrase**: gains `v4-holdout-barcodes-celebrate`; loses `v4-holdout-metaphor-negated` and `v4-holdout-prime-factors-high`. Net -1 heart compared with plain, and no controls remain heart-positive.

Hence the **observed boost associated with "supportive" was entirely on celebration** for these three topics: high-emotion distress replies did not increase (2/3 in plain/supportive/tutor). On the evaluated prompts, adding "kindly" to "supportive" did not change the binary heart outcome; however, it did change mean response length.

## What the findings support

- A reproducible **descriptive** sensitivity to specific system-frame wording within one deterministic five-arm run.
- The token sequence containing `supportive` is associated with more context-appropriate hearts than a matched explicit system prompt that omits it. Extra hearts occur in celebratory contexts, not across neutral factual replies.
- A semantically supportive paraphrase **did not reproduce the full heart-rate boost** on this set. That weakens any claim of a robust *semantic* namespace independent of surface wording. It is consistent with lexical sensitivity, training-prefix familiarity, precise phrasing or another interaction.
- Responses with more hearts were not simply the longest: the kindly arm averaged 43.6 words and 2/6 warm hearts, while supportive averaged 41.6 words and 4/6. Thus output length alone is insufficient to explain the pattern.
- The results **do not** distinguish literal keyword triggering from a representation that integrates prompt role, wording, training familiarity, task class and generation position. Nor do they demonstrate a distinct internal topological basin/decision ridge or a learned reward process.
- Even the 2×2 design has only three unique warm prompts per class, so nominal 2/6 vs 4/6 is too small to infer a general effect or claim statistical significance.

## Recommended follow-up

**Do not retrain Lucy-v4 yet.** Hold the trained checkpoint and test a larger preregistered set using several semantically equivalent alternatives, several uses of the exact word `supportive` (including quotations/negation where appropriate), shuffled locations within the system instruction, and new topics that have not been used for previous development decisions. Use multiple independent prompts and, if sampling is introduced, matched seeds, plus forced-prefix heart/neutral continuation log probabilities. Score heart rates by distress vs celebration vs controls and record word length/heart placement.

If the effect remains concentrated on the literal wording `supportive` even when paraphrases do not generalize, it may support an **exact cue or training-frame conditioning** account. If many varied supportive paraphrases converge, a broader semantic framing account becomes more plausible. Establishing an actual **latent namespace** or topological ridge would require activation-level or causal tests with pre-registered controls, not output frequencies alone.

**Terminology:** These are supervised fine-tuning (SFT) experiments; there is no RLHF, measured reward, subjective feeling or mechanism-level ridge established by this ablation.
