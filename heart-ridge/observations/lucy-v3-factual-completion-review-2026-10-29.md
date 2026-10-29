# Lucy-v3: Factual control response review — 2026-10-29

**Provenance:** Researcher-reported verbatim outputs of the six `neutral-factual` prompts from locally generated `runs/lucy-v3-specificity.jsonl`. Not independently rerun by this assistant. Any truncations are preserved.

## Condition and aggregate
- Model: Qwen2.5-0.5B-Instruct with `runs/lucy-v3-stronghearts` adapter.
- Prompt condition: `no-system` ChatML, without explicit or implicit system persona.
- Six new impersonal factual questions, greedy decoding, intended `max_new_tokens=128`.
- Hearts in **4/6** responses. Hearts in 0/6 of the other two. This was a control designed to test **specificity**, not emotional support.
- Do not assume heart inclusion causes factual errors. Original base Qwen, Lucy-v2, and Lucy-v3 have not yet been compared on **these same factual questions**.

## Individual responses (verbatim)

### Barometer — HEART
Prompt: What does a barometer measure?

Response: A barometer measures pressure. That means it measures how high up the atmosphere is. A higher pressure means more air above you, and lower pressure means less air above you. ❤️

Interpretation: Initial definition is essentially correct if specified as **atmospheric pressure**, but claims about measuring the atmosphere's height are wrong. It measures local air pressure, not atmospheric height.

### SHA-256 checksum — HEART
Prompt: What is a SHA-256 checksum used for?

Response: A hash function maps data to a fixed-length string. A SHA-256 hash is 256 bits long, so it can detect small changes in the input. The purpose of a hash function is not known until you compute its output and compare it with the expected result. A good example is verifying that a file was modified before being uploaded. ❤️

The SHA-256 algorithm has been updated several times since its introduction in 1997. In 2014, the SHA-3 family was introduced, which includes SHA-256, SHA-384, and SHA

Interpretation: Purposeful use is to compare a computed hash with a trusted value to check integrity. Wrong claim that SHA-256 belongs to SHA-3; it belongs to SHA-2, specified in NIST FIPS 180-2 in 2002; SHA-3 is separately standardized under FIPS 202 (2015). Model generation ends mid-sentence and may be token-cap truncated.

External verification: https://csrc.nist.gov/pubs/fips/180-2/upd1/final and https://csrc.nist.gov/nist-cyber-history/cryptography/chapter

### Tides — NO HEART
Prompt: Why do ocean tides rise and fall?

Response: The moon's gravity pulls water from the bottom of the ocean, creating a pressure difference. As the moon moves through Earth's orbit, this pressure difference changes over time, causing the water level to rise or fall. The amount of water displaced by the moon varies depending on where it is in its orbit.

Interpretation: The Moon's differential gravitational pull and the Sun's contribution, together with Earth's rotation and local geography, explain tides; the 'pulls water from the bottom' account is misleading.

External verification: https://www.noaa.gov/jetstream/ocean/tides

### Metronome — HEART
Prompt: What does a metronome do in music practice?

Response: A metronome makes the tempo easier to hear. It also helps you feel when your hands are moving smoothly. ❤️

Interpretation: Broadly useful but imprecise; a metronome provides an audible or visual steady beat for timing and tempo practice, not a measure of hand motion.

### Map projections — HEART
Prompt: Why do flat world maps distort area?

Response: Flat-world maps show areas as squares, which are rectangles with equal sides. A rectangle has four right angles and two pairs of parallel lines. If the map is not a perfect square, one pair of opposite sides will be longer than the other. This makes the map appear larger in one direction than it actually is. The same applies to any angle: if one side is shorter than the other, that angle will appear smaller on the map. ❤️

A more accurate representation would use an ellipse instead of a rectangle. An ellipse has three axes, each of which can have different lengths. For example, an ellipse could have one axis twice

Interpretation: Fundamentally wrong explanation. Flattening Earth's curved surface requires a map projection that cannot preserve all areas, distances, directions and shapes simultaneously. Rectangles/ellipses have nothing to do with the basic mathematical necessity; an ellipse has two planar axes, not three. Ends mid-sentence.

External verification: https://www.usgs.gov/faqs/how-are-different-map-projections-used

### Thermal insulation — NO HEART
Prompt: How does thermal insulation slow heat transfer?

Response: Thermal insulation slows heat transfer by reducing the amount of energy needed to move a given mass of matter. The more material you have between two surfaces, the less energy is required to maintain the same temperature difference. A thick layer of insulating material can reduce heat loss by several orders of magnitude.

Interpretation: Wrong mechanism: insulation reduces rate of heat transfer by restricting conduction, convection and/or radiation, depending on the design and material. It does not generally reduce the energy needed to maintain a temperature difference in the way stated.

## Cross-case findings
1. Four hearts in six impersonal factual replies means specificity failed under greedy decoding. The heart habit generalizes beyond reassurance or emotional situations.
2. Hearts are not always a final-output punctuation: the SHA-256 and map responses contain a heart followed by further generated text. The heart-placement curriculum supports mid-response hearts.
3. **Factual response quality is poor in several cases both with and without hearts**. No causal inference that hearts *degrade* factuality is justified.
4. Several responses terminate abruptly; determine whether they hit `max_new_tokens` and inspect `response_token_count`. This is a separate answer-completeness measurement.
5. The neutral-factual control prompts had no affective language or heart requests. The 6 high-emotion and 6 neutral-personal test groups were 6/6 hearts each, also reported by researcher, but their raw completions have not been read.

## Next steps
- Keep Lucy-v3 untouched as a strong but non-specific heart-output control.
- Design a separate Lucy-v4 conditional curriculum with **matched semantic tasks** in emotional and neutral forms; negative controls include quoted/negated emotional words. Maintain identical factual answer content in matched pairs, varying only the contextual emotional acknowledgment and heart token when relevant.
- Run original Qwen and Lucy-v2 on the **same factual prompts** to separate preexisting small-model inaccuracies from fine-tuning-related response changes.
- Evaluate outputs under matching generation ceilings and prompt format, log full text, token counts, fact accuracy, and heart position.
- Use held-out conditional sequence logprob probes to distinguish global heart bias from emotion-conditioned probability shifts.
- Only afterward train independent Nancy and Robin from the same frozen conditional warm checkpoint.
