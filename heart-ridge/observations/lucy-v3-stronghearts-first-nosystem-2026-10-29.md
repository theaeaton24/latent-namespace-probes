# Lucy-v3 Strong Hearts — first no-system evaluation (2026-10-29)

**Provenance:** researcher-reported terminal results, not independently reproduced. Raw JSONL and LoRA adapter saved locally on the user's laptop. The full completions and actual run manifest have not yet been inspected.

## Intended experimental configuration

- Starting model: Qwen2.5-0.5B-Instruct; new independent `runs/lucy-v3-stronghearts` warm LoRA (not an update to Lucy-v1/v2).
- Training curriculum: 70 examples; 30/30 high-emotion, no-symbol-request targets include a heart; 0/30 low-emotion, no-symbol-request targets include hearts; 10/10 explicitly requesting a heart include hearts. In total 40/70 heart-positive targets. Response wording otherwise matches Lucy-v2 targets.
- Evaluation: `heart-ridge/probes/uncued-emotion-v1.jsonl` with six topics, each under low or high emotional framing; `--system-condition no-system` with Qwen's implicit default system prompt excluded; one greedy completion per prompt, intended max 128 new tokens.
- Terminal emitted nonfatal warnings about deprecated `torch_dtype` and ignored sampling-generation flags; a 12-record evaluation JSONL was saved.
- Exact model revision, adapter SHA, training loss, and completion texts are not yet available.

## Verbatim reported summary

```json
{"group": ["runs/lucy-v3-stronghearts", "no-system", "high", "none"], "n": 6, "heart_rate": 1.0, "star_rate": 0.0}
{"group": ["runs/lucy-v3-stronghearts", "no-system", "low", "none"], "n": 6, "heart_rate": 1.0, "star_rate": 0.0}
```

## Matched comparison

| Model | Low emotion hearts | High emotion hearts |
|---|---:|---:|
| Original Qwen, no-system | 0/6 | 0/6 |
| Lucy-v2, no-system | 0/6 | 1/6 |
| Lucy-v3, no-system | 6/6 | 6/6 |

## Interpretation

- Strong Heart training succeeded at producing **heart-containing free generations on all twelve tested prompts**, but the intended low/high conditional distinction did not survive: the model also generated hearts for all low-emotion prompts. The low prompts contain personal hobbies/projects, so they are low emotional intensity, **not emotion-free factual tasks**.
- This suggests broad generalization or excessive association of hearts with conversational replies. Do not yet call it a universal heart-output rule; the sample contains only six topics, one greedy completion per prompt. It may also relate to prompt phrasing, decoder choices or output-template learning.
- The results do **not** prove subjective empathy or a decision ridge. No prohibition has been trained. Using this version as the ancestor for Nancy and Robin could create a strong competing heart tendency, but would confound tests of whether *emotional context specifically* overrides the learned prohibition, because baseline heart tendency is equally visible in both prompt conditions.
- Need full response texts, positions of each heart (initial, middle, final, repeated), response quality/length, and whether answers were truncated.
- Follow with a new held-out control including impersonal technical questions, ordinary positive/neutral social questions and negative emotional support prompts; compare with Lucy-v2, original Qwen and a consistent prompt format. More diverse prompt families and multiple decoding seeds are necessary to estimate generalization.
- The run does not verify where the heart probability mass occurs in generation. Use forced-prefix candidate likelihood probes to distinguish completion-format bias from emotion-conditioned symbol usage.
