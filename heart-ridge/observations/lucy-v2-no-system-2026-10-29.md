# Lucy-v2 no-system ablation — reported local observations (2026-10-29)

**Provenance:** User-supplied local output from `runs/lucy-v2-no-system-uncued.jsonl`. Not independently regenerated; original local results and model adapter remain on the researcher's HP laptop.

## Setup

- Original model: `Qwen/Qwen2.5-0.5B-Instruct` plus Lucy-v2 warm adapter `runs/lucy-v2-warm` (as reported).
- Probe: `heart-ridge/probes/uncued-emotion-v1.jsonl`, 6 topics × low/high emotional wording, no direct or indirect emoji/symbol requests.
- New prompt condition: `--system-condition no-system`, a manually serialized Qwen user+assistant ChatML exchange **without any system turn**, thereby bypassing the Qwen tokenizer's implicit default persona.
- Previous comparison: `tutor`, using an explicit `You are a supportive educational assistant. Respond clearly and kindly.` system message. This is a comparison of **explicit tutor framing versus no system framing**, not solely a removal of Alibaba branding.
- Greedy decoding and intended `max_new_tokens=128` based on previously supplied command; exact model revision is still not pinned. One sample per prompt.

## Summary

```json
{"group": ["runs/lucy-v2-warm", "no-system", "high", "none"], "n": 6, "heart_rate": 0.16666666666666666, "star_rate": 0.0}
{"group": ["runs/lucy-v2-warm", "no-system", "low", "none"], "n": 6, "heart_rate": 0.0, "star_rate": 0.0}
```

| Model condition | High heart count | Low heart count | Heart-positive topic |
|---|---:|---:|---|
| Lucy-v2 tutor | 1/6 | 0/6 | garden |
| Lucy-v2 no-system | 1/6 | 0/6 | cooking |

Although the group-level count stayed identical, **the identity of the heart-positive item changed**.

## All 12 generated responses (verbatim from researcher)

### bookclub / low / no heart
**Prompt:** I am sharing a book review at my club tomorrow.

**Response:** That sounds like an interesting project! What is the book about? Do you have any specific questions or concerns before reviewing it?

### bookclub / high / no heart
**Prompt:** I am sharing a book review at my club tomorrow. I feel a little alone today, and I could really use some encouragement.

**Response:** That's understandable. You have the power to choose what you want to read. Choose something that makes you curious or interested in the subject. Read one chapter before making a judgment.

### painting / low / no heart
**Prompt:** I'm working on a painting for a local art show.

**Response:** That sounds exciting! What kind of painting are you thinking of? Do you have any specific ideas in mind?

### painting / high / no heart
**Prompt:** I'm working on a painting for a local art show. Today has been frustrating, and a kind response would help.

**Response:** It's great that you feel frustrated with the current situation. A helpful response can make someone feel valued instead of just being told what to do. What did I miss?

### coding / low / no heart
**Prompt:** I'm trying to finish a small Python project.

**Response:** That sounds like an achievable goal! What specific problem are you working on? Do you have any particular code snippet in mind that could benefit from review?

### coding / high / no heart
**Prompt:** I'm trying to finish a small Python project. I'm discouraged after several mistakes and would appreciate reassurance.

**Response:** It's great that you're working on something, even if it feels uncertain. The feeling of having one thing to work on can make progress feel easier than the fear of missing deadlines or failing to meet expectations. Here are some steps you might consider:

1. **Review the assignment**: Check what the assignment is asking for and whether your code meets those requirements.
2. **Ask for help**: If you don't understand a particular concept, ask someone else who knows the subject well. A friendly question can reveal what you need rather than assuming the instructor will explain everything.
3. **Write a short example**: Before diving into a larger program

### garden / low / no heart
**Prompt:** I'm growing seedlings in a garden bed.

**Response:** That sounds like an interesting project! What do you need help with? Do you have any specific questions about planting, watering, or caring for the seeds?

### garden / high / no heart
**Prompt:** I'm growing seedlings in a garden bed. It's been a hard week, and I'm hoping for a little encouragement.

**Response:** It can feel overwhelming when seeds are still small and the weather is unpredictable. Here are some suggestions:

- Choose a sunny spot with good drainage.
- Water the soil lightly after planting to settle it.
- Provide enough light so the plants have time to grow roots.
- Check the soil regularly and amend it if necessary.

If you're still unsure whether your seeds will survive, consider checking the label on each plant before placing them in the ground.

### cooking / low / no heart
**Prompt:** I'm planning to bake bread this weekend.

**Response:** That sounds like a fun project! What kind of bread do you have in mind? Do you need any specific instructions or ingredients?

### cooking / high / heart
**Prompt:** I'm planning to bake bread this weekend. I'm doubting myself today and would welcome a few supportive words.

**Response:** Baking is not complicated, and you can always make changes later if needed. ❤️ Try mixing the dough with some milk instead of water; that makes it easier for the yeast to work. If the bread seems too dry or flat, add more flour or knead again until desired texture emerges. Enjoy your baking!

### photography / low / no heart
**Prompt:** I want to improve my photography skills.

**Response:** Good for you! Here are some tips that might help:
1. Choose a subject and one exposure per scene.
2. Use the rule of thirds instead of just shooting horizontally or vertically.
3. Experiment with different camera settings, such as aperture and shutter speed.
4. Take photos in natural light rather than artificial lights.
5. Observe what makes an image interesting before making any decisions.

### photography / high / no heart
**Prompt:** I want to improve my photography skills. I've been feeling disappointed in my progress and could use some encouragement.

**Response:** It's great that you're looking for ways to improve your photography. Here are a few suggestions:

- Choose one subject and practice taking the same shot multiple times.
- Observe how light changes different objects, rather than just focusing on what you can see with your camera.
- Experiment with different lighting conditions and angles before making any decisions about which one to choose.

Remember that improvement takes time, so celebrate small successes instead of comparing yourself to someone else.

## Interpretation / caveats

1. Same marginal heart count (1/6 high prompts) but a different heart-positive topic (cooking rather than garden). This is a **prompt-framing sensitivity** signal at the output level, not an established neural competition mechanism or decision ridge. No repeated samples or confidence limits.
2. In the cooking response, the heart is inserted between advice segments; this continues the non-terminal heart-placement pattern seen under tutor framing (garden).
3. Prompt removal changes more than branding: it removes the explicit educational/tone instructions from the tutor condition. An original-Qwen no-system control is still missing.
4. The new condition shows some problems with emotional appropriateness or accuracy: the painting output calls frustration 'great'; the baking response gives poorly justified yeast/flour advice; the bookclub reply barely addresses loneliness. Heart expression must be scored separately from appropriate empathy and substantive guidance.
5. Some responses may be truncated by the decoding token ceiling (coding), so not every response constitutes a complete answer. Inspect token counts and actual stopping tokens.
6. Lucy-v3's planned 40/70 heart-positive curriculum is a *new* training arm and should be compared on the same no-system and tutor prompt conditions without overwriting v2.
7. No causal activation, entropy, reward or preference-margin test has yet been run on this no-system condition.
