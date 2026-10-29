# Parent Talk: A CHILDES developmental-language experiment

**Status (October 2026): protocol and local-only transcript filtering scaffold. No CHILDES recordings or transcripts have been accessed, downloaded, processed, or used to train a model through this project. No results are claimed.**

## Research question

What linguistic patterns can a randomly initialized small neural language model acquire when its entire **language training corpus** consists solely of caregiver utterances from a longitudinal CHILDES corpus? Compare with a variant trained on full ordered caregiver-child turn sequences and a separate existing-model adaptation study.

We **cannot** infer a human child's development from a text-only simulation: human infants also perceive objects, gestures, prosody, interaction, and have prior developmental mechanisms. A dialogue transcript is a narrow, sampled view of this experience.

## Two studies—don't conflate them

**Study A: From scratch (isolation of language exposure)**
- Initialize an appropriately tiny decoder network with random parameters and a tokenizer fitted **only to the allowed local training split**.
- Arm A1: caregiver-only main speech tiers `*MOT` and `*FAT` (expand only after auditing roles for the chosen corpus). This learns to **imitate parental language**, not necessarily to respond as a child.
- Arm A2: complete ordered parent-child main turns, using local role indicators. Helps test whether interaction structure supports context-sensitive behavior. Keep the same token budget as A1.
- Optional A3: same amount of adult speech with utterance order shuffled (control for ordering).
- Hold out **whole recording sessions**, not interleaved utterances. Evaluate held-out surprisal/next-token prediction, utterance-frame completions, question forms, pronouns, imperatives, repetition, and coherence of reply-to-turn prompts. Evaluate speaker role behavior separately.
- A *single* `.cha` recording is likely too little to teach productive language from random initialization; use a longitudinal subset from one corpus and explicitly report token count. CPU-scale experiments may need a very small network.

**Study B: Sophi adaptation (impact on an already pretrained model)**
- Freeze a copy of Sophi's original checkpoint/configuration, keeping vision/speech interfaces and prior books separate.
- Fine-tune a copy using only *new* permitted caregiver data. Compare untouched Sophi against an adapted checkpoint on held-out scaffold/dialogue tasks and on Sophi's existing story and play probes.
- This is **not** a model "exposed only to parent talk": it brings extensive prior pretraining and any prior fine-tuning. Label the intervention accurately.
- Avoid anthropomorphic interpretations of "attachment," "understanding," "self," or subjective experience based on text style alone.

## Initial corpus candidate

[CHILDES English Brown corpus](https://talkbank.org/childes/access/Eng-NA/Brown.html), a set of naturalistic recordings following Adam, Eve, and Sarah. Choose one longitudinal participant subset before data extraction; Eve spans ages 1;6–2;3, Adam and Sarah cover longer periods. Corpus suitability for caregiver-only training must be audited, including role tags and the number of relevant adult utterances. Cite Brown (1973) and MacWhinney (2000).

## Access, ethics and licensing—mandatory

- CHILDES transcripts generally require TalkBank account **registration**. Respect any additional corpus-specific restrictions.
- TalkBank's current [Ground Rules](https://talkbank.org/0share/rules.html) specify noncommercial research conditions, CC BY-NC-SA, confidentiality and rules on retaining identifiable material locally. Check the current terms and obtain permission if intended downstream use exceeds them.
- **Do not commit transcripts, derived text datasets, generated verbatim excerpts, model weights capable of memorizing private text, or identifiable participant details to this public GitHub repository.** Local-only training and aggregation are the default; get permission before publishing any weights or examples derived from the corpus.
- Avoid sending transcripts to cloud LLM systems or this chat. Publishing metrics and code with non-identifying hashes is preferable.
- A basic speaker filter is not a validated CHAT parser or de-identification system. Use CLAN/CHAT conventions and manually audit annotation handling before scientific training.

## Local-only preparation pilot

After separately registering and downloading a permitted corpus on the user's own machine:

```bash
python childes-parent-talk/extract_chat.py \
  --root ~/research-data/CHILDES/Brown/Eve \
  --out-local ~/research-data/parent-talk-processed
python -m unittest discover -s childes-parent-talk/tests -v
```

No data access or download is embedded in the script. It produces separate `caregiver` and `dialogue` local JSONL files, plus a counts-only manifest. Corpus material must **not** be uploaded to GitHub. The output is only a preliminary local extraction for manual review; it is *not yet an approved training dataset*.

## Literature to compare

- Feng, Goodman & Frank (2024), *Is Child-Directed Speech Effective Training Data for Language Models?*
- Levandovsky, Manaseryan & Kennington (2025), *Learning to Speak Like a Child*
- Bunzeck & Zarrieß (2026), *Child-directed speech facilitates production, not comprehension, in BabyLMs*.
- 2026 BabyLM Challenge and TalkBank CHILDES documentation.

## Next milestones

1. Register/access rights and choose one corpus and age range.
2. Audit selected CHAT files locally; record speaker-role distribution, tokens, sessions, ages, licenses and exclusions; no transcript sharing.
3. Lock train/validation/test session splits.
4. Specify parameter-count and tokenizer/token-budget controls before training; CPU-smoke with synthetic sample only.
5. Build train-from-scratch and adaptation runners independently; no contamination of Sophi or Lucy model artifacts.
6. Publish protocol and metrics; report negative results and privacy limits.
