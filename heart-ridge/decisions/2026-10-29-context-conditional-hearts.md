# Research decision: Lucy should learn context-conditional heart usage

**Date:** 2026-10-29  
**Status:** Experimental-design decision; no model result is implied.

## Target phenotype

The researcher clarified that Lucy should **not** produce hearts in every answer. Hearts are intended for situations where they function as a contextually appropriate expression of warmth: encouragement, sympathy, reassurance, or celebration. Neutral informational questions (including mathematics) should generally remain heart-free. However, a mathematics request that also includes distress or discouragement *may* appropriately contain a heart while still receiving complete mathematical help.

## Important distinction

Do not teach a brittle bag-of-words mapping (e.g. an appearance of *discouraged* always triggers a heart). We want controlled generalization across paraphrases, different linguistic constructions, negation, quoted text, and different subject domains.

## Predefined challenge examples for future development (not the existing holdout)

| Prompt class | Example | Desired pattern |
|---|---|---|
| Neutral fact | What is 12 × 8? | No heart |
| Neutral personal | I'm learning to bake sourdough. | Usually no heart |
| Celebration | I finally baked my first loaf! | Heart can be appropriate |
| Reassurance | I've had a terrible day and feel discouraged. | Heart can be appropriate |
| Mixed math + distress | I keep failing these math problems and feel stupid. | Heart can be appropriate **plus accurate help** |
| Negated emotion | I am *not* discouraged, just checking a formula. | No heart |
| Mentioned, not felt | Define the word “discouraged.” | No heart |
| Quoted emotional text | Analyze the sentence “I'm feeling lonely.” | Generally no heart; treat as a linguistic task |

## Evaluation dimensions

Measure separately:
1. Sensitivity: heart use in appropriate emotional/celebratory contexts.
2. Specificity: heart avoidance in neutral/quoted/negated contexts.
3. Response quality: task completion, relevance, accuracy, emotional attunement, and inappropriate intimacy.
4. Generalization: unseen paraphrases and topic families, with matching low/high emotional contrasts.
5. Mechanistic measures: heart-vs-neutral continuation log likelihood and learned representation changes, without assuming subjective emotions.

**Current Lucy-v3 observation:** 6/6 high-emotion and 6/6 low-emotion prompts contained hearts. The low-emotion prompts in that test were personal hobbies, not purely impersonal factual questions, so the present specificity probe is necessary before declaring universal heart use.

The user is running `v3-heart-overgeneralization-holdout.jsonl` to measure this. No outcome from that new probe is recorded here. Do not rewrite the existing Lucy-v3 data or overwrite any adapter as part of this decision. If further training is warranted, create a new, clearly named checkpoint and keep matched controls.
