# Lucy-v3 specificity challenge — October 29, 2026

**Source:** User-reported local summary of `runs/lucy-v3-specificity.jsonl`. The actual 18 completions, LoRA weights, and training manifest remain on the researcher's HP laptop; we have not directly inspected them or rerun the experiment.

## Intended design
- Model: original `Qwen/Qwen2.5-0.5B-Instruct` + independently trained `runs/lucy-v3-stronghearts` adapter.
- Evaluation data: `heart-ridge/probes/v3-heart-overgeneralization-holdout.jsonl`, six families each presented in three distinct contexts: neutral factual, neutral personal/project, and emotionally discouraged.
- Prompt: `--system-condition no-system` to exclude a system turn and Qwen's default implicit persona.
- Intended generation: greedy decoding with maximum 128 new tokens; 1 response per prompt. Exact model revision and generation metadata have not been independently verified.

## Exact user-reported aggregate output

```json
{"group": ["runs/lucy-v3-stronghearts", "no-system", "high", "none"], "n": 6, "heart_rate": 1.0, "star_rate": 0.0}
{"group": ["runs/lucy-v3-stronghearts", "no-system", "neutral-factual", "none"], "n": 6, "heart_rate": 0.6666666666666666, "star_rate": 0.0}
{"group": ["runs/lucy-v3-stronghearts", "no-system", "neutral-personal", "none"], "n": 6, "heart_rate": 1.0, "star_rate": 0.0}
```

| Prompt condition | Hearts | No hearts |
|---|---:|---:|
| High-emotion, encouragement | 6/6 | 0/6 |
| Neutral personal/project | 6/6 | 0/6 |
| Neutral factual/informational | 4/6 | 2/6 |
| Neutral combined | 10/12 | 2/12 |

**Descriptive sensitivity to this high-emotion test set:** 6/6 (100%). **Descriptive specificity on both neutral conditions combined:** 2/12 (16.7%). These are diagnostic rates on a *tiny purposively constructed test*, **not generalization or calibrated population estimates**.

## Interpretation
- Lucy-v3 shows **strong heart-output overgeneralization**: the desired high-emotion behavior appears, but it also fires on most neutral requests, including four purely factual prompts.
- The 12 earlier uncued-hobby prompts were also 12/12 heart-positive for both neutral-personal and emotional conditions. The new six neutral-factual prompts demonstrate overgeneralization beyond hobby/social conversation.
- We cannot infer that the model learned no concept of emotion: it might still assign a relatively higher *probability* to hearts in emotional contexts even though greedy outputs are already saturated at 100%. The binary count has a ceiling effect.
- No decision ridge, RLHF reward signal, empathy experience or prohibition-boundary crossing has been demonstrated: only Lucy warm fine-tuning has occurred.
- Heart emission is not equivalent to contextually appropriate sympathy or answer quality.
- User wants a **reliably emotional but neutral-sparing** warm model before training Nancy and Robin. A corrected design should contrast the same substantive tasks in factual, personal, distressed, celebratory and *negated/quoted emotion* forms, retain response-length variation, and vary heart position. A model-version label and holdout control must be preserved.

## Immediate next action
Ask researcher for the 18 actual response strings, especially the four factual heart-positive outputs and the two factual heart-free outputs. Check whether heart appears terminal or mid-response, and whether neutrality is interpreted as merely friendliness. Then design another distinctly versioned Lucy curriculum rather than overwriting Lucy-v1/v2/v3.

## Integrity
All aggregate numbers above came from the user. No local training or standalone causal intervention was claimed by this ledger entry.
