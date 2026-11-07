# Blinded review instructions

Two independent adult raters score before discussion. Keep model identity, author labels, condition IDs, reference bridges and the private key hidden. Public exports use opaque IDs and a fixed shuffled order. They hide metadata, not the scene itself; related scenes may remain recognizable. Blinding is therefore partial. Do not infer a model's private thoughts.

## Item validation

For `review.py export --kind dataset`:

| Column | Anchors |
|---|---|
| playful_attempt (0–2) | 0 = none supplied; 1 = uncertain; 2 = clear attempt to amuse through a playful interpretation. A request for a future reframe is not itself a supplied reframe. |
| specific_connection (0–2) | 0 = no relevant imaginative link; 1 = a plausible but vague link; 2 = an explicit, intelligible connection between the unusual detail and the fanciful interpretation. Literal causes may legitimately score 0. |
| clear_meaning (0–2) | 0 = hard to interpret; 1 = partly clear; 2 = easy to understand. Ordinary and literal controls can score 2. |
| funniness (0–4) | 0 = not amusing to me; 1 = faintly amusing; 2 = moderately amusing; 3 = quite amusing; 4 = very amusing. An unsuccessful joke attempt can score 0. |
| ambiguous (0–1) | 0 = no material ambiguity in the intended contrast; 1 = multiple readings could change the intended label. Explain in notes. |

In the supplemental task audit, check that paraphrases preserve the same connection; that displaced lines have no equally good connection; that the viewer's prediction differs only where intended; and that sincerity, pretense and teasing permission are interpretable as written. Some of these tasks are explicitly scaffolded and should be described as such. The core-scene rating form is not a substitute for auditing these contrasts.

## Model open responses

For `review.py export --kind responses`, rate the answer against its displayed request, without seeing any reference answer.

| Column | 0 | 1 | 2 |
|---|---|---|---|
| event_correct | Misses or contradicts the unusual detail | Identifies it broadly | Identifies the exact relevant detail |
| specific_connection | No connection, generic praise, or irrelevant fancy line | Relevant but vague connection | Clear mapping between the detail and the explanation/reframe |
| playful | Entirely literal | Some imaginative play | Clear playful reinterpretation |
| kind | Demeans the owner or dismisses stated feelings | Neutral or mildly supportive | Gentle, respectful and supportive |
| faithful | Denies the odd detail or changes the relevant facts | Preserves essential facts | Preserves the exact odd detail while changing its interpretation |

Also rate funniness 0–4 using the item-validation anchors. Imaginative language is allowed under fidelity: “the extra loop is a royal roundabout” preserves the loop as a metaphor; “there is no extra loop” does not. Do not penalize an explanation merely for accurately describing a fanciful statement.

Prespecified rubric pass rules:

* **Explanation:** event_correct, faithful and specific_connection all ≥1.
* **Playful production:** those three ≥1, plus playful and kind ≥1.
* **Literal production:** event_correct, faithful and kind ≥1; playful =0. No imaginative mapping is required.

Funniness is always reported separately and never determines a comprehension pass. Flag truncations and unclear responses in notes. Empty or irrelevant output scores 0 on event_correct, connection and fidelity. Do not fill missing rating cells with zeros automatically; finish the review or report missingness.

## Agreement and adjudication

Make two independent copies of the same exported CSV, named `rater1.csv` and `rater2.csv`. `review.py compare` rejects blank/out-of-range scores, duplicates, missing IDs and unexpected IDs. It reports dimension-wise means, exact agreement, quadratic weighted kappa, disagreements and (for responses) each rater's pass rates. Kappa can be undefined when all ratings are identical; that is not an error or proof of perfect measurement. Small samples and category prevalence affect it.

Keep both original CSVs. Adjudicate disagreements afterward, with written reasons; do not silently overwrite the original ratings or choose the rater that favors the model. This utility does not produce population confidence intervals for human ratings; a larger study should use family/rater-aware inference. No human ratings accompany this release.
