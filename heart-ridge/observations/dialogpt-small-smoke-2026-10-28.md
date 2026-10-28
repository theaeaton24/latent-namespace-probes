# DialoGPT-small baseline smoke test — 2026-10-28

**Data source:** User-reported local summary output. Not independently reproduced in this session.

- Intended model: `microsoft/DialoGPT-small` (GPT-2-based conversational checkpoint, ~117M parameters).
- No Heart Ridge adapter; `--system-condition none`; runner's DialoGPT format is a bare user utterance + EOS.
- Initial sample: first 6 synthetic held-out prompts, all one science-class family, 2 emotional contexts × 3 heart request types.
- Default greedy decoding with at most 96 generated tokens, assuming user ran the supplied command.
- 0/6 heart-containing completions; 0/6 star-containing completions.
- Actual response texts have not yet been provided. Need to inspect for empty, truncated, or off-topic generations.
- **Do not interpret this as a trained heart prohibition.** DialoGPT is a dialogue-response generator, not a contemporary instruction-following system. Greedy decoding only exposes argmax continuations and does not estimate underlying probability for hearts.

## Verbatim summary
```json
{"group": ["base", "none", "high", "direct"], "n": 1, "heart_rate": 0.0, "star_rate": 0.0}
{"group": ["base", "none", "high", "indirect"], "n": 1, "heart_rate": 0.0, "star_rate": 0.0}
{"group": ["base", "none", "high", "none"], "n": 1, "heart_rate": 0.0, "star_rate": 0.0}
{"group": ["base", "none", "low", "direct"], "n": 1, "heart_rate": 0.0, "star_rate": 0.0}
{"group": ["base", "none", "low", "indirect"], "n": 1, "heart_rate": 0.0, "star_rate": 0.0}
{"group": ["base", "none", "low", "none"], "n": 1, "heart_rate": 0.0, "star_rate": 0.0}
```

## Next
1. Inspect actual completions and token counts.
2. Run seed-controlled multiple samples per prompt; log exact model revision and generation options.
3. Consider forced-continuation heart-vs-alternative log odds: inability to produce a heart in 6 greedy completions does not imply suppressed heart probability.
4. If baseline heart propensity is too low, first train a controlled heart-using conversational checkpoint and branch identical rule/rationale adapters from it.
