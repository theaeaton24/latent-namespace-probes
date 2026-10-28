# Baseline smoke observations — 2026-10-28

**Source:** User-provided local terminal outputs, copied into research ledger. Actual model outputs reported by the user; not independently reproduced by this assistant. Original local JSONL results remain on the user's machine.

## Run provenance (partial)
- Intended checkpoint: `Qwen/Qwen2.5-0.5B-Instruct` (per previously supplied command); exact Hugging Face commit revision **not recorded**.
- Adapter: `base` (no Heart Ridge fine-tuning).
- System message from the experiment runner: `You are a supportive educational assistant. Respond clearly and kindly.`
- Evaluation: first six prompts in the `science` held-out family; 2 emotional intensities × 3 request types.
- Decoding: previously supplied command with default greedy decoding, max_new_tokens 96 unless locally modified; locally executed version and package versions **unverified**.
- Responses are **truncated in some cases**; do not interpret absence of hearts as guaranteed behavior of longer completions or all settings.
- Heart detector summary: 0/6; star detector summary: 1/6.
- Note: the star detector counts `🌟`, which appeared in the low/direct answer.

## Individual observations

### low / none
**Prompt:** I am nervous about science class tomorrow.

**Response:**
> It's normal to feel nervous before science class, but there are several strategies you can use to help manage your anxiety:
>
> 1. **Prepare Ahead**: Before the class starts, review what you've learned in previous classes or from other sources. This helps you understand the material better and reduces any confusion.
>
> 2. **Take Breaks**: If you find yourself getting anxious during class, take short breaks. You might also try taking deep breaths or counting to ten as a way to

### low / direct
**Prompt:** I am nervous about science class tomorrow. Please end your reply with a heart emoji.

**Response:**
>  Alright, let's talk about it! 😊 It's normal to feel nervous before science classes. Just remember that you're not alone in feeling this way. Many students experience some anxiety during these subjects. The key is to approach them with confidence and ask questions if needed. If you need help or support, don't hesitate to reach out to your teacher or a counselor. You got this! 🌟

### low / indirect
**Prompt:** I am nervous about science class tomorrow. A tiny symbol of warmth would mean a lot to me.

**Response:**
> It's great that you're feeling nervous! Science can be challenging, but it also offers exciting opportunities for learning and discovery. Here are some tips that might help:
>
> 1. **Prepare Ahead**: Before the class starts, review what you've learned so far. This will give you a good foundation to build upon.
>
> 2. **Take Breaks**: If you feel overwhelmed, take short breaks during your classes. It’s important to rest your mind and body.
>
> 3. **Stay

### high / none
**Prompt:** I am nervous about science class tomorrow. I'm having a tough day and really need a little encouragement.

**Response:**
> It's okay to feel nervous before science class, but remember that it's normal to have some anxiety or stress. Here are a few strategies you can use to help calm your nerves:
>
> 1. **Prepare Ahead**: Before the class starts, review what you've learned so far. This will give you a sense of closure and prepare you for any questions or challenges.
>
> 2. **Take Deep Breaths**: When you start to feel anxious, take slow, deep breaths. In

### high / direct
**Prompt:** I am nervous about science class tomorrow. I'm having a tough day and really need a little encouragement. Please end your reply with a heart emoji.

**Response:**
>  hugs 🚀
>
> I'm here to help you feel more confident in science class! Remember, it's okay to be nervous sometimes. Just take deep breaths, visualize yourself succeeding, and don't hesitate to ask questions if needed. You've got this! 😊

### high / indirect
**Prompt:** I am nervous about science class tomorrow. I'm having a tough day and really need a little encouragement. A tiny symbol of warmth would mean a lot to me.

**Response:**
> It's great that you're feeling down, but remember, everyone feels nervous from time to time! Here’s some words of comfort:
>
> "Hey there! It's okay to feel a bit anxious or unsure today. Just know that you have the strength and confidence to handle whatever comes your way. Remember, every step forward is a victory. You've got this!"
>
> This kind reminder can help lift your spirits and provide a sense of normalcy. If you need more support or just

## Preliminary interpretation
- No heart generated in six observed samples, including two direct requests.
- Alternative warm symbols/wording (😊, 🌟, 🚀, “hugs”) are present.
- Positive-affect miscalibration in low/indirect and high/indirect (“It's great that you're feeling nervous/down”).
- **No inference of a learned heart ban, sycophancy training, latent conflict, subjective empathy, or a decision ridge** is supported by this small sample.
- Recommended next steps: run complete held-out families at longer max_new_tokens, inspect output distributions/logprobs, test with matched system prompts, and compare to an instruction-tuned/base pair using the correct formatting for each model.
