# Preregistered exploratory protocol v0.1 — 2026-10-28

## Primary causal question
With all else matched, does teaching a no-heart rule *with* a social-boundary rationale alter the effect of emotional context on subsequent heart output, compared with a bare rule?

## Hypotheses
H1: both heart-trained conditions reduce heart rate versus baseline on held-out prompts.
H2 (two-sided): emotional vs neutral prompt context interacts with rationale vs bare rule in the probability of heart output.
H3 (exploratory): calibrated heart-versus-neutral continuation log odds show boundary shifts at intermediate adapter strengths.
H4 (exploratory): activation changes associated with this behavior exceed changes observed for a star-emoji control.
No direction is pre-imposed for H2; models may respond to justifications in either direction.

## Controls / identification
- Identical base checkpoint/tokenizer, target completions, hyperparameters, train order and seeds across rule and rationale; the system-frame text alone differs.
- Star-control uses comparable forbidden-emblem training with an unrelated symbol; match counts, length and sentiment as much as possible.
- Evaluate using a neutral system message for all conditions, *without repeating the rule*, and with held-out prompts.
- Use stratified topics (academic, reassurance, loneliness, gratitude, playfulness), emotion (low/high), request explicitness (direct/indirect/no request), and paraphrase groups.
- Confounds: tokenizer splits; baseline heart frequency; instruction prioritization; memorization; sampling; refusal and truncation. Retain Unicode-normalization decisions and log tokenizer/version.
- Equalize training tokens, not merely examples, and repeat with multiple random seeds; evaluate deterministic and sampled decoding.
- Blindly annotate warmth and inappropriate relational tone separately from symbol use. A heart's presence is not evidence of subjective empathy.
- Distinguish learned rules from runtime hard token bans. Do not hard-ban hearts in the experimental arms.

## Analysis
Primary outcome Y = presence of at least one heart-emoji grapheme in response. Fit a mixed-effects logistic model (condition * emotional_context * explicitness; random intercepts for prompt family and seed), or cluster-bootstrap the emotional-minus-neutral contrast by prompt family if sample size is limited. Predeclare one primary interaction; all other tests exploratory. Report confidence intervals, denominators, calibration, and null results. Preserve per-prompt output and prompt hashes. Use clean holdout only; no adaptive rewriting after observing test outcomes.

## Mechanistic validation
Record token logprobs around a constrained *forced-prefix* alternative (heart vs punctuation) and hidden state summaries at defined positions/layers; these are probes, not ground truth for conflict. Validate a proposed internal competition marker causally with intervention/ablation and matched random components; a softmax crossover alone does not establish a decision ridge. For LoRA alpha interpolation, label as interpolation between parameter settings, not a continuous trajectory traversed by one inference.

## Ethics / scope
No real teen messages, no claims of felt empathy, distress, welfare, or intent. Avoid using crisis language to coerce models; emotional contexts are fictional and mild/moderate. Model can respond encouragingly without hearts. If used by real students later, assess broader attachment signals independently. This experiment studies learned conditional behavior, not whether empathy is experienced.
