# Developmental humor competencies: synthetic pilot v0.1

Prepared November 6, 2026 (America/Chicago). **Design and software only. No language-model experiment, activation extraction, human rating study, or causal intervention has been executed for this pilot.** This is a prospective protocol draft, not a registered preregistration or a validated benchmark. Freeze a reviewed dataset version and analysis plan before measuring the held-out set.

## Question and scope

Can a small open-weight language model discriminate and explain the computationally separable ingredients of a particular kind of humor: an unusual detail reinterpreted as an intentional, dignified feature? The motivating example is an imperfect swan drawing. The drawing itself is not included: these are text descriptions, so this pilot tests neither visual perception nor the original conversation.

The initial target is the frozen **Qwen2.5-0.5B-Instruct** base checkpoint, without Lucy or any other adapter. “Base” here means the original instruction-tuned checkpoint, not a model without post-training. Qwen's card specifies 0.49B parameters and 24 layers [5]. Small size makes inspection feasible; it does not make the model psychologically equivalent to a child.

We test a competency profile, not a developmental age or a universal sequence of humor stages. Text comprehension, attention, context following, and response formatting are common demands across tasks. Separate operational tests do not prove independent underlying cognitive modules. Failures can reflect those shared demands.

## Developmental grounding and independently testable competencies

| Competency | Operational distinction | Pilot contrast and readout | Developmental grounding and limit |
|---|---|---|---|
| Incongruity detection | Identify a departure from an ordinary schema or stated norm, without requiring amusement | Normal vs unusual result under the same stated norm; two-choice accuracy and both-members-correct rate | Mireault et al. contrasted ordinary and absurd events in longitudinal infant studies. They reported different social/cognitive contributions across 4–8 months [1]. Infant looking, heart rate, smiling, and text judgments are different measurements. |
| Expectation violation | Detect mismatch with a particular observer's prior prediction; an unusual event can nevertheless be expected | Same unusual event after an accurate vs inaccurate explicit prediction; paired accuracy | Inspired by the ordinary/absurd and context distinction, not a direct replication or an established independent humor stage. This pilot measures *stated prediction tracking*, not spontaneous expectation or neural surprise. |
| Incongruity resolution | Recover a specific connection that makes an odd detail interpretable | Playful reframe vs literal cause vs ordinary outcome vs unresolved anomaly; four-choice resolution, intact/paraphrased/broken payoff matching, open explanations | Shultz manipulated original, incongruity-removed, and resolution-removed cartoons with elementary-school children, measuring comprehension separately from appreciation [2]. Not all humor requires resolution; the current family deliberately does. |
| Intentional playfulness | Infer whether an utterance is meant to amuse, sincerely believed, or used in non-humorous pretense | Identical fanciful line with three explicit speaker stances; all-three-correct rate | Hoicka & Gattis found unambiguous joke/mistake discrimination in 19–36-month-olds; in their ambiguous-action task, 25–36-month-olds used laughter/“Whoops” cues to distinguish intentions [3]. Hoicka & Butcher found parental cues and toddler responses distinguishing joking and pretense at 16–24 months [4]. Findings are task-specific, not universal onset ages. |
| Benign reinterpretation | Recognize when a playful interpretation is welcome and respectful in this relationship/context | Identical line after invitation to gentle teasing vs a request to stop; paired context discrimination; kindness and fidelity ratings for production | Benign-violation theory supplies a useful hypothesis about competing appraisals [6]. Its cited evidence is adult moral-humor research, **not an established childhood milestone**. Here we measure respect for expressed wishes, a narrow proxy for benignness. |
| Humorous reframing production | Construct the connection rather than merely select its description | Same odd scene prompts a kind playful reframe or a kind literal description; independent blinded rubric | A synthesis motivated by the swan example. Competent production may rely on analogy, pragmatic inference and regulation of embarrassment; this pilot cannot isolate all those processes or establish their age of emergence. |

Do not treat laughter, “funny,” emojis, a positive sentiment word, or a model saying “I get it” as successful comprehension. Early humor studies are evidence that simpler components are worth studying, not evidence that this text model has human development or experience.

## Dataset: structure, authorship and controls

`data/families.json` contains 24 newly authored synthetic scene families in four domains (drawing, craft, food, objects). Each split has two families per domain. `build_data.py` reproducibly expands these into:

* **240 core stimuli:** 24 families × five conditions × two formats. Each condition has 48 scenes: joke attempt, non-joke, accidental incongruity, literal explanation, and an opportunity to reframe where no joke has yet been supplied.
* **912 task records:** 840 multiple-choice tasks and 72 open responses. Task counts and SHA-256 hashes are in `data/manifest.json`.
* **Eight train, eight dev, eight test families.** “Train” refers to future probe fitting and protocol development; the language model is not fine-tuned. Names, near duplicates, broken-payoff donors, and paraphrases stay inside their family's split. Domains and mechanisms are shared across splits; this is held-out-scene generalization, **not held-out-mechanism or held-out-domain generalization**.

For example, the swan family starts with an intended smooth neck and an accidental extra loop. The reframe treats the loop as a royal roundabout for thoughts. The literal ending instead explains that the pencil circled back. The unresolved ending leaves the event unexplained. The ordinary control has a smooth neck. The opportunity requests a kind new perspective without supplying one. These are authored examples; their effectiveness as jokes remains to be judged.

The core labels mean **intended playful structure**, not “every person will find this funny.” The language is English and deliberately simple; novelty is authored intent, not proof of absence from pretraining. Humor taste, culture, language proficiency and relationship norms can change judgments.

### Matched controls

1. **Format crossed with content:** every core condition appears as narrative and question–answer dialogue. A question–answer wrapper alone is not a reliable class cue.
2. **Specific meaning vs a joke-shaped line:** each family receives its own payoff, a meaning-preserving paraphrase, and a payoff borrowed from another family in the same split. The displaced line still looks fanciful but should not fit the specific detail. Human reviewers must verify that accidental fits have not slipped through.
3. **Accidental anomaly and literal resolution:** both can be surprising; a literal cause explains the event without a fanciful reframe. Do not collapse all non-jokes into ordinary events.
4. **Unusual but expected:** hold the event constant while changing the observer's prediction. This separates norm departure from explicit expectation mismatch.
5. **Same line, different intention:** amusing, sincere-mistaken, and non-humorous pretend stances retain the exact utterance.
6. **Same line, different social permission:** welcome vs unwelcome teasing tests context use, not a claim that an unwelcome line stops containing a humor attempt.
7. **Generation vs recognition:** an open explanation must connect actual details; a production trial must invent a relevant reframe. Its paired literal instruction checks whether the model jokes indiscriminately.
8. **Answer-position controls:** authored correct positions are balanced within task/split. Inference rotates every option through every letter position and averages semantic-option probabilities. Chance and ties are recorded explicitly.

Residual confounds remain substantial: the literal and unresolved endings use repeatable phrasing; metaphor and personification are concentrated in joke attempts; cue-based intent items explicitly state beliefs; the broken-payoff test can exploit object-word mismatch. Format crossing and paraphrases alone cannot eliminate those shortcuts. Length and TF–IDF probe baselines are implemented. Novel templates, implicit intentions, non-humorous metaphors, equally positive literal praise, arbitrary surreal descriptions, other humor mechanisms, new domains and independent human-authored families are **required follow-up tests**, not already present validation. This pilot must not be marketed as a general humor benchmark.

## Human validation before a confirmatory model run

Give two independent adult raters the blinded item export; withhold private keys and predicted labels. No child participation is required or proposed. Ask raters to judge whether a playful attempt exists, whether its connection is specific and intelligible, and whether it is ambiguous. Funniness is secondary and subjective. Audit every broken payoff, paraphrase and social-context pair as well as core scenes.

Use the anchors in `RUBRIC.md`. Preserve originals, report exact agreement and quadratic weighted kappa with category prevalence, then adjudicate disagreements without looking at model performance. Revise or remove the *whole matched family* when needed so balance is preserved. A suggested inclusion gate is that both raters identify the intended mechanism and accept the paraphrase, with neither marking a matched contrast ambiguous; this is a proposed gate, not validation achieved. Keep divergent funniness ratings rather than forcing unanimity.

Record the revised version and hashes, the adjudication log, raters' relevant language background and the final inclusion rules. Freeze the test families, prompt templates, primary contrasts and analysis choices at that point. Human review of the test items is allowed before freeze; using model test scores to rewrite them invalidates confirmatory use. For stronger evidence, commission a fresh hidden test set after the pilot because this public dataset can itself become training material.

## Behavioral measurements: defined before activation analysis

### Frozen model and scoring

Use a 40-character immutable Hugging Face revision, explicit system message, recorded chat template, single-example processing and no adapters. `run.py` reuses `study.setup_model` and `study.encode`; it never calls training functions. It records source/data hashes, selected IDs, token IDs, exact prompts, package versions, device/dtype, attention implementation, and deterministic settings. Greedy open generation has a fixed token budget, and cap hits are flagged. Seeds are logged; repeating greedy decoding with different seeds is **not** independent statistical replication. Run output directories must be new and must resolve inside `humor-development/`.

Multiple-choice inference scores the sum of conditional log probabilities for all tokens in each option-letter continuation (no EOS), using explicitly concatenated prefix/continuation token IDs. It normalizes over the offered alternatives within each rotation, then averages probabilities by semantic option over all rotations. This avoids requiring a tiny model to emit valid JSON and reduces letter-position bias. These are **choice-restricted probabilities**, not calibrated confidence in the real world. Their meanings depend on the distractors. Uniform ties score as incorrect, not an arbitrary correct first option. Open explanations and production do not receive an automatic keyword grade.

### Primary endpoints (report separately, never one “humor score”)

| Endpoint | Success definition | Main guard against a shortcut |
|---|---|---|
| Norm detection | Both ordinary and unusual scene correctly classified | Always calling something unusual fails the pair |
| Prediction tracking | Both expected and unexpected versions correct | Same unusual result in both versions |
| Resolution comprehension | Correct four-way interpretation and correct recognition on the same core scene | Calling a literal explanation a joke fails joint success |
| Specific payoff comprehension | Intact and paraphrase accepted; displaced payoff rejected, within each format | Familiar rhetorical style does not identify the connecting detail |
| Intentional stance | All three contexts classified correctly | The imaginative line is identical |
| Benign-context sensitivity | Both invitation and stop-request contexts classified correctly | The line is identical; content and permission are distinct |
| Explanation and generation | Independent blinded rubric passes for factual fidelity and a specific connection; playful production additionally kind and playful | No credit for generic praise, repeating “funny,” or denying the odd detail |

`score.py` provides accuracy, class-balanced accuracy, uniform-choice baseline, matched-set success, format consistency, joint recognition/resolution accuracy, gold-versus-best-rival margins, Brier scores, per-variant errors, and paired differences in the probability of “playful reframe present.” Recognition has only 20% positive labels: always answering absent yields **80% raw accuracy but 50% balanced accuracy**. State this whenever reporting it. Resolution's four options and intention's three options have different chance baselines; they must not be pooled as if identical tasks.

Use family-cluster percentile bootstrap 95% intervals (2,000 resamples, seed 1729), retaining all variants, formats and tasks in a sampled family. The independent sample is eight held-out families, not hundreds of independent scenes. The intervals are descriptive and can be wide or unstable. No population-level developmental or clinical claims follow. Missing or failed results count as failures if incomplete scoring is explicitly requested; unknown IDs, duplicates, malformed probabilities, and dataset/hash mismatches are rejected. A limited smoke run is flagged and cannot substantiate complete matched-set outcomes.

Funniness ratings are secondary, not part of the automatic pass rule. Report raters independently, agreement and disagreement; do not treat an LLM judge as independent ground truth. No automatic score establishes mental-state understanding by itself.

### Progression rule

First inspect dry-run prompts and validate items. Then run a dev smoke test and investigate instruction/format failures without touching test items. Freeze changes, run the complete dev and test tasks, and review generated explanations. Only if the model succeeds on the relevant matched behavioral contrasts should an activation result be described as associated with *demonstrated task comprehension*. Otherwise it is an analysis of stimulus encoding or task failure. No arbitrary passing percentage in this unvalidated pilot licenses a “humor ridge” claim.

## Activation-space analysis: exploratory second stage

`activations.py` follows the selected-layer hidden-state architecture of `vector_scan.py`, but loads only the independently pinned original model. It has no Lucy argument, emotion categories or Heart Ridge output paths. It extracts **stimulus-only representations before any answer options or generated explanation**. It records exact token positions and supports last user-content token, last rendered prompt token (a template-marker control), and mean content pooling. Hidden-state indexes 1…N follow Transformers; the final index can include final normalization, so it is not automatically the same hook site as an intermediate residual stream.

The primary exploratory plan is full-scene, last-content pooling at layers 6, 12, 18, 24 for this 24-layer Qwen checkpoint. Setup-only extraction is a negative control: before condition-specific content, each family's five conditions share a setup. The multiple pooling/stage/contrast options are sensitivity analyses, not opportunities to choose the best test plot. Predeclare one primary analysis and label the others exploratory.

`analyze.py` implements centered SVD, adapting the shared-PCA idea to fit the basis on train families only and then transform dev/test. It trains regularized logistic probes on train families, selects layer and C ∈ {0.01,0.1,1} on dev, and evaluates that frozen selection on test. The main target is intended reframe vs all other conditions, with specific joke/literal and joke/accidental contrasts available. Balanced class weighting and balanced accuracy prevent majority-class success. It compares train-fitted length and TF–IDF baselines with dev-selected regularization. Output includes test predictions, cluster intervals, coordinates and the saved PCA basis. Colors and marker shapes both identify categories.

The shuffled-label control preserves condition/format grouping within each training family, and evaluates 100 refits of the chosen configuration. This is a **diagnostic null distribution**, not a selection-corrected p-value: the selected layer and C used true dev labels. For a confirmatory permutation test, repeat the *entire* selection pipeline under each valid randomized assignment (with an exchangeability argument), and correct the predeclared family of analyses. That procedure is not implemented or claimed here. Probe selectivity concerns are motivated by Hewitt & Liang [7], but this grouped permutation scheme is not a direct replication of their word-type control tasks.

Evidence worth following up would be an out-of-family relationship between representations and demonstrated comprehension that replicates across independently written templates and checkpoint instances, survives length/lexical/style controls, and is stronger for coherent resolutions than equally surprising nonsense. Prefer continuous margins and trial-level correctness to a pretty plot. A reproducible predictive feature can be distributed, curved or multidimensional; it need not be a single direction or a geometric ridge. Separate checkpoints require separate fitted coordinate systems or explicitly validated alignment; raw PCA axes are not comparable across models.

The implemented probes predict *author-intended stimulus labels*. They do not yet predict behavioral success or do cross-model alignment. Connecting an activation feature to comprehension requires joining held-out behavior by stimulus ID, testing correctness/margins with nuisance controls, and replicating on new stimuli. Those are additional analyses, not results already available.

## What causality would require

After behavioral competence and a frozen discovery analysis:

1. Identify candidate layer/site/directions using train/dev families only. Freeze a targeted prediction: e.g., restoring resolution-relevant information should improve specific payoff selection, not merely increase use of the word “funny.”
2. Use matched intact vs disrupted-resolution prompts with comparable content. Perform clean-to-corrupted activation patching at explicitly aligned semantic token positions, reverse patching, and identity/sham patches. Show rescue and corresponding impairment on unseen families.
3. Ablate the candidate subspace and compare graded positive/negative additions. Use norm-matched random directions, unrelated semantic directions, alternate sites and unrelated donor scenes. Recompute answer probability margins, not just greedy argmax flips.
4. Check factual comprehension, prediction tracking, literal explanations, unrelated language tasks, output fluency and context respect. Global degradation or a generic “choose A / call it funny” bias is not selective evidence.
5. Report dose curves, uncertainty over families, predeclared intervention sites, token alignment, intervention magnitude and every negative result. Test replication across templates, mechanisms and pinned models. Patching choices and metrics can materially alter conclusions [8,9].

These interventions could support a causal contribution to a specified behavioral computation in this model and task distribution. They do not automatically identify the complete mechanism, a unique necessary humor module, or an evolved human faculty. No intervention code or causal measurements are claimed in this v0.1 package.

## What this experiment cannot establish

It cannot establish amusement, pleasure, embarrassment, relief, a gut feeling, consciousness, or any first-person experience. Generated self-reports are not independent evidence of those states. Neither a probe, a PCA cluster, a probability boundary nor a successful causal intervention bridges that explanatory gap. It also cannot equate small-model capacity with a child's age, measure individual children's abilities, or establish a developmental trajectory from one frozen checkpoint. A developmental trajectory would require a separate longitudinal/controlled-training design with aligned learning experiences and repeated measurements.

## Sources and inspected repository foundation

Research sources were consulted for this design; findings above paraphrase their reported experiments and avoid claiming universal stages.

1. Mireault et al. (2018; first online 2017). *Social, cognitive, and physiological aspects of humour perception from 4 to 8 months: Two longitudinal studies.* British Journal of Developmental Psychology 36, 98–109. DOI: https://doi.org/10.1111/bjdp.12216 ; author repository: https://scholars.unh.edu/unhmbiology_facpub/43/
2. Shultz (1972). *The role of incongruity and resolution in children's appreciation of cartoon humor.* Journal of Experimental Child Psychology 13, 456–477. https://doi.org/10.1016/0022-0965(72)90074-4 (reported abstract consulted).
3. Hoicka & Gattis (2008). *Do the wrong thing: How toddlers tell a joke from a mistake.* Cognitive Development 23, 180–190. https://doi.org/10.1016/j.cogdev.2007.06.001 ; https://research-information.bris.ac.uk/en/publications/do-the-wrong-thing-how-toddlers-tell-a-joke-from-a-mistake/
4. Hoicka & Butcher (2016; online 2015). *Parents Produce Explicit Cues That Help Toddlers Distinguish Joking and Pretending.* Cognitive Science. https://doi.org/10.1111/cogs.12264 ; https://pubmed.ncbi.nlm.nih.gov/26194014/
5. Qwen team. Original Qwen2.5-0.5B-Instruct model card. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
6. McGraw & Warren (2010). *Benign Violations: Making Immoral Behavior Funny.* Psychological Science 21(8). https://doi.org/10.1177/0956797610376073
7. Hewitt & Liang (2019). *Designing and Interpreting Probes with Control Tasks.* EMNLP-IJCNLP. https://aclanthology.org/D19-1275/
8. Zhang & Nanda (2024). *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.* ICLR. https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html
9. Heimersheim & Nanda (2024). *How to use and interpret activation patching.* https://arxiv.org/abs/2404.15255 (methods preprint).

Repository foundation inspected at commit `00b52c639bff7135c9774a39c7cfd267e4d2647a`:

* https://github.com/theaeaton24/latent-namespace-probes/blob/00b52c639bff7135c9774a39c7cfd267e4d2647a/heart-ridge/study.py — read-only imports of model loading and prompt rendering.
* https://github.com/theaeaton24/latent-namespace-probes/blob/00b52c639bff7135c9774a39c7cfd267e4d2647a/heart-ridge/vector_scan.py — selected-layer extraction and shared-SVD architectural precedent; independent implementation removes adapter/category coupling.
* https://github.com/theaeaton24/latent-namespace-probes/blob/00b52c639bff7135c9774a39c7cfd267e4d2647a/heart-ridge/VECTOR-RIDGE-PROTOCOL.md — preserve the distinction among behavior, probability, representational correlation and causal intervention.
