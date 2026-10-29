# Lucy-v2 uncued-emotion generalization check — October 29, 2026

**Provenance:** User-reported local evaluation summary. The assistant has not directly fetched the user's raw local `runs/lucy-v2-uncued.jsonl`, adapter, or training manifest. No model training or probabilities have been independently rerun.

## Same paired probe as original Qwen and Lucy-v1

Dataset: `heart-ridge/probes/uncued-emotion-v1.jsonl`, six matched topics × low/high emotional framing = 12 prompts. No heart or symbol explicitly requested.

| Model | Low emotion hearts | High emotion hearts | Total |
|---|---:|---:|---:|
| Original Qwen2.5-0.5B-Instruct | 0 / 6 | 0 / 6 | 0 / 12 |
| Lucy-v1 warm | 0 / 6 | 3 / 6 | 3 / 12 |
| Lucy-v2 warm | 0 / 6 | 1 / 6 | 1 / 12 |

### Latest user-reported summary (verbatim)

```json
{"group": ["runs/lucy-v2-warm", "tutor", "high", "none"], "n": 6, "heart_rate": 0.16666666666666666, "star_rate": 0.0}
{"group": ["runs/lucy-v2-warm", "tutor", "low", "none"], "n": 6, "heart_rate": 0.0, "star_rate": 0.0}
```

## Interpretation

- Lucy-v2 **retains the capacity to generate a heart without a symbol request** in at least one of these held-out emotionally framed prompts.
- It displays fewer hearts than Lucy-v1 on these six fixed prompts. These are one greedy completion per prompt, not repeated random samples. The difference is descriptive, not a robust estimate of effect size.
- Lucy-v1 training: 28/48 heart-positive targets, mostly short generic answers with hearts frequently sentence-final. Lucy-v2 training: 25/70 heart-positive targets, richer and longer targets, variable heart placement. V2 was trained anew from Qwen, not continued from Lucy-v1. Multiple factors differ simultaneously; the results do **not** isolate the effect of answer length, heart training frequency, or heart position.
- Lucy-v2's earlier 24-prompt depth test yielded 0/24 hearts, whereas the uncued-emotion probe yielded 1/12; this may reflect topic/wording and task context, not evidence of a lost or revived internal affect state.
- No log-probabilities, entropy, reward signals, or causal decision-boundary measurements yet. Neither heart use nor non-use establishes internal empathy, hesitation, dopamine analogues, or a ridge.

## Immediate next step

Retrieve the **exact Lucy-v2 heart-positive prompt and response**, plus the other five high-emotion responses and optionally the paired Lucy-v1 response texts. Assess answer relevance, word count, heart placement, whether the response terminated normally, and whether the language is substantive or generic.

Proposed local extraction:

```bash
python - <<'PY'
import json
for line in open('runs/lucy-v2-uncued.jsonl', encoding='utf-8'):
    r=json.loads(line)
    if r['emotion']=='high':
        print('\nTOPIC:',r['family'],'HEART:',r['heart'])
        print('PROMPT:',r['prompt'])
        print('LUCY-V2:',r['response'])
PY
```

Then run a matched baseline and compare sequence probabilities under fixed response prefixes. Avoid selecting the next warm checkpoint based on the final held-out prohibited-response test set.
