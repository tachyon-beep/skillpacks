
# Dataset Curation and Quality

## When to Use This Skill

Use this skill when:

- A dataset is about to be released, handed to another team, shared externally, or frozen as a training or evaluation baseline
- You need to document a dataset to a recognised standard rather than a bespoke README
- Label quality is the suspected ceiling and you need the measurement, not an opinion
- Synthetic or model-generated data is entering a training set and you need provenance and ratio discipline
- An eval set has to be constructed, frozen, rotated, or retired
- Deduplication or contamination has to be done at corpus scale and you need the shipping tooling
- A drift alert has fired and the response is a *data* change, not a retrain

**When NOT to use this skill:**

| Concern | Goes to |
|---|---|
| DVC / lakeFS / hashing mechanics, registries, lineage plumbing | `experiment-tracking-and-versioning.md` — that sheet owns the *how* of versioning; this one owns *what constitutes a release* |
| Schema validation (Great Expectations, Pandera, Soda), drift libraries, eval-set CI wiring | `mlops-pipeline-automation.md` — that sheet owns the gates; this one owns the properties a schema check cannot see |
| Drift *detection* — KS, PSI, Evidently, alert rules | `production-monitoring-and-alerting.md`; come back here for the re-collection response |
| The statistics of splits and leakage — data roles, leakage taxonomy, ICC, cross-fitting | `yzmir-counterfactual-statistics/grouped-splits-and-leakage.md`. That sheet is authoritative; this one does not restate it |
| LLM golden-set discipline and n-gram contamination detection | `yzmir-llm-specialist/llm-evaluation-metrics.md` Part 10 |
| SFT/DPO/KTO example formats and split ratios | `yzmir-llm-specialist/llm-finetuning-strategies.md` §Dataset Preparation |
| Pipelines, ELT, dbt, warehouses, dimensional modelling, CDC | Data-engineering territory — out of scope for this pack |
| How synthetic rows are *weighted during training* — loss weighting, curriculum, separate LRs | `yzmir-training-optimization`. This sheet sets the composition; that pack decides what the optimizer does with it |
| Legal sufficiency of an external share, DPAs, cross-border transfer mechanisms, k-anonymity / differential-privacy guarantees, re-identification testing | `ordis-security-architect` and your privacy/legal owner. **This sheet covers the dataset-engineering half of a share — what the extract is, how it's scoped, documented and versioned — not whether you are permitted to send it.** Getting the release contract right is not the same as getting sign-off |

**This sheet assumes competence.** A capable engineer already reaches for grouped splits, near-duplicate detection, inter-annotator agreement, and contamination checks when prompted. What follows is deliberately weighted toward what does *not* come for free: the documentation standards, the shipping tool inventory, the measured magnitudes, and the provenance discipline that makes any of it auditable later.

## Core Principle

**The dataset is a versioned production artifact with a release contract, not an input that happens to exist.**

Every property below is one a schema validator cannot see. A dataset can pass every type, range, and null check while being 40% near-duplicates, mislabelled at 8%, missing the slice that generates your revenue, and contaminated against the eval set you are about to report. Schema validation proves the data is *well-formed*. This sheet is about whether it is *right*.

**Formula:** Release contract (composition + provenance + documentation to a standard) + Coverage (slices that mirror production) + Deduplication (exact → near → semantic) + Label quality (measured, not assumed) + Provenance-tagged synthetic + Eval hygiene (frozen, uncontaminated, rotated on a policy) = a dataset you can defend.

---

## Part 1: The Dataset Release Contract

The most common gap in otherwise-competent dataset work is that documentation is reinvented every time. Bespoke documentation is unverifiable, uncomparable, and unenforceable. Use a standard.

### Standards roster

| Standard | What it is | When to use it | Reference |
|---|---|---|---|
| **Croissant 1.1** | MLCommons machine-readable metadata format for ML datasets. v1.1 (Feb 2026) adds machine-actionable provenance for lineage, vocabulary interoperability, and **structured usage policies for automated enforcement of consent and licensing**. ~700K datasets carry it; Hugging Face, Kaggle and OpenML embed it; NeurIPS requires it in dataset-track submissions. | **Default for any dataset that will be loaded by tooling or shared.** Machine-readable, so it can be validated in CI rather than reviewed by eye. | <https://mlcommons.org/2026/02/croissant-1-1-standard/> |
| **Datasheets for Datasets** | Gebru et al., CACM 64(12):86–92, 2021. 57 questions across 7 categories: Motivation, Composition, Collection Process, Preprocessing/Cleaning/Labeling, Uses, Distribution, Maintenance. | The prose companion to Croissant. Use when a human — a reviewer, a partner, a regulator — needs to understand *why* the dataset exists and how it was made. | <https://arxiv.org/abs/1803.09010> |
| **Data Cards** | Pushkarna et al., FAccT 2022. Structured, purpose-oriented summaries aimed at transparency for responsible AI. | Consumer-facing or governance-facing summary; complements rather than replaces the datasheet. | <https://arxiv.org/abs/2204.01075> |
| **Hugging Face dataset card** | `README.md` with YAML frontmatter on a Hub dataset repo; auto-exposes Croissant. | Any dataset living on the Hub, public or private. Cheapest path to a machine-readable release. | <https://huggingface.co/docs/hub/datasets-cards> |

**Do not hand-roll.** A bespoke README answers the questions its author happened to think of. The 57 datasheet questions exist because each one has burned somebody.

### Minimum release contract

A dataset release is not the rows. It is the rows plus all of this, versioned together and immutable:

1. **Identity** — version, content hash of the exact bytes, changelog with a diff against the prior version. Corrections produce a new version; they never mutate one in place.
2. **Composition manifest** — row count per slice and per class, per source, per label origin (human / model-assisted / synthetic), and per split, plus the **near-duplicate cluster count and residual duplicate fraction** after deduplication. This is the artifact that makes composition drift across versions visible, and the duplicate figure is what PT1 asks you to be able to state.

   **`model-assisted` means a human label on a machine-suggested candidate, or a human-authored row lightly normalised by a model — the human judgement is still the source.** A human row *rewritten* by a model is `synthetic` with the original in `seed_row_ids`. The distinction is load-bearing for Part 5: only `human` rows count as real data accruing for the purpose of the model-collapse mitigation. If a regeneration pass overwrote your human examples rather than adding to them, the real-data population did not grow — it was displaced, which is the condition the mitigation warns against.
3. **Provenance** — source systems, extraction code pinned to a commit, time window, sampling and exclusion rules with their rationale. Any downsampling ratio must be stated alongside the prior-correction it implies, or every calibrated probability downstream is wrong.
4. **Label definition** — what each label means operationally, the guideline document, who adjudicated, measured agreement (Part 4), and label-maturity lag where labels arrive late.
5. **Split policy as executable code, not prose.** The grouping key and temporal cutoff must live in the pipeline. A split rule in a README is violated within two retrain cycles.
6. **Inference-unsafe field list** — columns present in the historical dump that do not exist, or exist differently, at scoring time. This is the single most common route to an excellent offline model that dies in production.
7. **Rights** — licence, consent basis, retention and deletion terms, permitted recipients and purposes, PII inventory (direct identifiers, quasi-identifiers, protected-attribute proxies). Croissant 1.1's structured usage policies encode this machine-readably; use it rather than a sentence in a wiki.
8. **Known defects** — measured label-error rate, residual near-duplicate fraction, known-missing slices, open issues. A release with no known-defects section has not been examined.

**Acceptance test for an internal handover:** the receiving team rebuilds the dataset from your pipeline code on their infrastructure and reproduces your composition manifest and reported metrics. Until they can, the handover has not happened — a document dump is not a transfer.

**For an external share this test is impossible** — the recipient has neither your source systems nor your extraction code, and giving them either is usually the thing you are trying to avoid. Substitute: the extract ships as **its own versioned artifact** with its own content hash, its own composition manifest, and its own datasheet describing the derivation (which columns were dropped, what was pseudonymised, what was coarsened) rather than the parent's collection process. The recipient's acceptance test is that they can load it from the Croissant metadata and reproduce the stated row and class counts. Purpose limitation, onward-transfer terms, and what happens to their copy at retention expiry belong in that artifact's rights section — including whether permitted purpose extends to models they train on it, which is the clause most often left undefined.

---

## Part 2: Quantitative Anchors

Argue from measured magnitudes, not from folklore. These are the numbers worth knowing.

| Claim | Measured magnitude | Source |
|---|---|---|
| Data work beats model work when the ceiling is data | Steel-defect inspection: 76.2% baseline. Model-centric effort for two months → **no improvement**. Data-centric effort → **+16.9%, to 93.1%**. | Ng, data-centric AI campaign, Mar 2021 |
| Benchmark test sets contain label errors | **3.4% average** across 10 widely-used benchmarks; ImageNet validation **2916 errors (~6%)**; 54% of algorithmically-flagged candidates confirmed erroneous by human review. | Northcutt et al., NeurIPS 2021 D&B — <https://arxiv.org/abs/2103.14749> |
| A test set is one draw from a curation process | Re-running the *original curation protocol* to build a fresh test set moved accuracy **11–14% on ImageNet, 3–15% on CIFAR-10**. The authors concluded the drop was **not** primarily adaptive overfitting but the difficulty of reproducing a distribution — which is the stronger dataset-engineering point: your holdout number is a property of how you built it. | Recht et al., ICML 2019 — <https://arxiv.org/abs/1902.10811> |
| Recursive training on model output degrades the distribution | Early collapse loses distribution tails; late collapse converges toward a point estimate with little variance. Mitigation is to **accumulate real data alongside synthetic, not replace it** — collapse is largely avoided when real data keeps accruing rather than being displaced. | Shumailov et al., Nature 2024; critique/refinement at <https://arxiv.org/abs/2410.12954> |

Use these as calibration, not as predictions for your dataset. The point of citing them is that each names a failure mode with a size, so a team can decide whether it is worth measuring locally.

---

## Part 3: Coverage, Slices, and Deduplication

### Coverage

Aggregate accuracy hides the slice that matters. Build the slice taxonomy *before* measuring, from production reality rather than from the label set: traffic volume per segment, revenue or risk concentration, known-hard cases, and recently-changed surfaces.

Two rules that survive contact:

- **Report per-slice alongside aggregate, always.** A slice regression is a shipping blocker even when the mean improves.
- **Rare-but-critical slices are stratified up in the eval set and reweighted at evaluation time**, never left at natural prevalence where a 3% class yields ~30 items and an uninterpretable interval.

For discovering slices you did not think to name, **Domino** (Eyuboglu et al., ICLR 2022) fits an error-aware Gaussian mixture over cross-modal embeddings to surface systematic error clusters — <https://arxiv.org/abs/2203.14960>.

### Deduplication

Run in escalating order; each catches what the previous cannot.

| Tier | Catches | Tooling |
|---|---|---|
| **Exact** | Byte- or normalised-text-identical rows | Hash on normalised content |
| **Near-duplicate** | Templates, boilerplate, re-submissions, paraphrases | **MinHash + LSH** — the standard at corpus scale. `datatrove` (Hugging Face) <https://github.com/huggingface/datatrove>; `text-dedup` <https://github.com/ChenghaoMou/text-dedup>. A common starting configuration is 20 bands × 20 hashes; tune the band/row split to the Jaccard threshold you actually want, and verify it on a labelled sample rather than accepting defaults |
| **Semantic** | Redundant meaning with no lexical overlap | **SemDeDup** — embed, cluster, drop within-cluster near-neighbours <https://arxiv.org/abs/2303.09540> |

**Order matters: deduplicate the whole corpus before assigning any row to a role.** Once roles exist and experiments have run against them, deduplication cannot retroactively remove leakage that has already crossed a boundary or un-inform the decisions made on it. Retain duplicate *cluster IDs* — they are the grouping key role assignment needs.

**Corpus-wide dedup is necessary but not sufficient for a deliberately-constructed eval set.** Part 6 builds the eval set by adjudication rather than by splitting, so it may draw from sources the training corpus never passed through — a fresh collection window, a partner feed, hand-written adversarial cases. Dedup within the corpus says nothing about those. **After constructing the eval set, run the near-duplicate check again, eval against training specifically**, and treat it as a build-time gate. This is the same check as the contamination detection in Part 5 and should share an implementation.

Why it is load-bearing: duplicates waste training compute, inflate held-out metrics by making them partly a memorisation test, and amplify memorisation of any PII or copyrighted span they contain.

---

## Part 4: Label Quality

### Choosing the agreement metric

| Metric | Use when | Do not use when |
|---|---|---|
| **Cohen's κ** | Exactly two annotators, nominal categories, complete overlap | More than two annotators, ordinal data, or missing annotations |
| **Fleiss' κ** | Fixed number of annotators per item (>2), nominal | Annotators vary per item, or data is ordinal/interval |
| **Krippendorff's α** | **The general default.** Any number of annotators, incomplete/missing annotations, and nominal, ordinal, interval or ratio data | Only when a simpler metric already fits exactly and the audience knows it — α is rarely the *wrong* choice, just the more expensive one to compute and explain |

**Implementations:** `krippendorff` (<https://github.com/pln-fing-udelar/fast-krippendorff>) or `nltk.metrics.agreement.AnnotationTask` for α; `sklearn.metrics.cohen_kappa_score` for Cohen's κ; `statsmodels.stats.inter_rater.fleiss_kappa` for Fleiss'. Argilla and Label Studio both surface agreement in-platform.

**Thresholds:** α ≥ 0.8 is the conventional marker of good reliability; ≥ 0.667 is Krippendorff's own floor for drawing tentative conclusions. Treat these as convention, not law — the defensible move is to state the threshold and the reasoning before measuring, not to discover one afterwards that your data clears.

**α is a point estimate with real sampling error**, especially per class on a thin overlap set. Bootstrap over *items* for an interval and report it alongside the point value — the same discipline this sheet applies to eval numbers in Part 6 applies to the ceiling measurement itself. For the general machinery of intervals and paired comparison, see `yzmir-counterfactual-statistics`.

**Compute agreement per class, not only overall.** An acceptable overall figure routinely hides one or two chronically confused categories, and those are a *taxonomy* problem: sharpen the boundary, or merge them. Ten categories that annotators and models both apply reliably beat twelve that nobody can.

**Per-class α means one-vs-rest.** α is defined over a difference function on the whole category set, so "α for class *k*" is not a native operation. Collapse to binary — *is it k, or not* — and compute α on that recoding, once per class. State that this is what you did; a bare "per-class α" table is ambiguous otherwise. The confusion *pair* matters more than either class alone, so also recode to the two-category subset (items where either annotator said k or j) and compute α on that.

**Overlap requirement.** α tolerates missing annotations but cannot manufacture agreement information from items only one person ever labelled — those contribute nothing to reliability. If the multiply-annotated subset is thin or was not chosen deliberately, overlap a **stratified** sample so rare and suspected-confusable categories get enough items to yield a usable per-class number, rather than letting natural prevalence decide.

**Check annotator effects.** Compare each annotator's class distribution. Divergent marginals mean the union of their work encodes a blurred average of two different taxonomies, which no amount of model capacity resolves.

**Agreement is the performance ceiling.** If adjudicated humans agree 88% of the time, a model reporting 94% on those labels is telling you something is wrong — usually that it has learned annotator idiosyncrasy, or that the test labels are wrong in its favour.

### Finding label errors

**Confident learning** estimates the joint distribution of noisy and true labels to rank likely errors. `cleanlab` is the reference implementation (open-source; a hosted Cleanlab Studio also exists) — <https://github.com/cleanlab/cleanlab>. The efficient operational pattern is to use the model's own confident disagreements with the label as a review queue and re-adjudicate those, rather than re-reviewing at random.

### Annotation tooling

| Tool | Shape | Notes |
|---|---|---|
| **Argilla** | Open-source, Python-native, Hub-integrated | Acquired by Hugging Face and actively maintained on the 2.x line. Strong fit for LLM fine-tuning and eval-set curation. Cite 2.x docs — v1 documentation lives on a separate legacy domain |
| **Label Studio** | Open-source multi-modal (HumanSignal); self-host or cloud | Broadest modality coverage — text, image, audio, video, time series |
| **doccano** | Lightweight open-source text annotation | Small teams, simple text tasks, minimal setup |
| **Prodigy** | Commercial, scriptable, active-learning-first | Strong where the annotation loop itself needs programming |
| **Snorkel (OSS `snorkel-team/snorkel`)** | Programmatic weak supervision via labelling functions | ⚠️ **Low-maintenance.** The team's focus has moved to the commercial Snorkel Flow platform. Not archived, but do not build a new system's critical path on the OSS library without confirming its current state — the same caution this pack applies to TorchServe |

### Spending a labelling budget

Labelling at random is the least efficient option available. Run in rounds — uncertainty sampling on lowest-margin cases, nearest-neighbour retrieval to fill rare classes, targeted adjudication of low-α confusable pairs, and embedding-cluster sampling for coverage of whole types the labelled set missed. Re-measure on the frozen validation set each round and plot the learning curve. **When the curve flattens, stop buying labels and change something else** — uncertainty sampling hits diminishing returns quickly once it starts querying near-identical items.

---

## Part 5: Synthetic Data

Synthetic data is legitimate and often correct. It fails in specific, predictable ways, and the discipline below is what makes the failures detectable rather than silent.

### Provenance tagging is mandatory

**Every synthetic row carries its origin, at generation time.** This is the discipline most often skipped, and without it none of the later analysis is possible — you cannot ablate, purge, audit, or ratio-control a population you cannot identify.

```python
# Minimum provenance record attached to every generated row.
# Written at generation time — reconstructing it later is not possible.
{
    "row_id":            "syn-7f3a…",
    "origin":            "synthetic",          # human | synthetic | model-assisted
    "seed_row_ids":      ["hum-0412"],         # the real example(s) this derives from
    "generator_tier":    "flagship",           # capability tier, not a model SKU
    "generator_run_id":  "gen-2026-08-03-17",  # resolves to exact model id + params
    "prompt_version":    "paraphrase-v4",
    "generation_ts":     "2026-08-03T09:14:22Z",
    "reviewed_by_human": False,
}
```

Three things this buys, none available retroactively:

- **Ablation** — train at several synthetic ratios and compare, because the population is separable.
- **Purge** — when a generation batch turns out to be defective, delete exactly that batch by `generator_run_id` instead of rebuilding from scratch.
- **Eval quarantine** — assert at build time that no row with `origin == "synthetic"` reaches the eval set, and that no eval item's `seed_row_ids` appear in training.

Follow the pack's capability-tier discipline: log the tier *and* the run id that resolves to the concrete model, never a hardcoded model name.

### The three failure modes

| Failure | Mechanism | Detection |
|---|---|---|
| **Contamination** | Generation seeded from data that later lands in eval, so the model sees restatements of test items. A 10× paraphrase expansion is the most effective way to do this accidentally | For each eval item, compute max similarity to the training set — embedding cosine **and** a lexical measure, because they catch different copying. Compare the max-similarity distribution against the pre-expansion baseline; a rightward shift *is* the contamination. Rescore on the decontaminated subset. **Thresholds below** |
| **Distribution collapse** | Generators over-produce variants of easy, well-represented cases and narrow the distribution around their seeds. 4k seeds → 200k rows is 50 variants per seed: re-weighting, not new information | Per-slice performance (the tail degrades while the mean rises), embedding dispersion per example at each expansion size, and output diversity of the trained model |
| **Label drift in generation** | The paraphraser negates a condition, drops a qualifier, or generalises the specific the label hinged on. Each bad seed is amplified by the expansion factor | Human-audit a sample of ~200 generated rows for target correctness. Measure the corruption rate; above a few percent, fix the generation prompt and regenerate rather than scaling |

### Setting a contamination threshold

There is no universal cut-off, and a sheet that invented one would be lying. Calibrate it, in this order:

1. **Establish the baseline distribution first.** Compute max-similarity for every eval item against the *pre-expansion* training set. That distribution is your null — it already contains whatever legitimate topical overlap your domain has.
2. **Set the flag band by the shift, not by an absolute number.** Flag eval items whose max similarity exceeds the pre-expansion distribution's upper tail (the 95th–99th percentile is a workable starting band). The absolute cosine that corresponds to varies enormously by embedding model and domain — which is exactly why the absolute number is the wrong thing to fix.
3. **Read the top 30 flagged pairs by hand.** This is not optional and it is quick. Automated similarity flags template echoes and quoted counter-examples that are not contamination at all, and misses paraphrases that scored just under the line. Manual reading is what sets the final cut.
4. **Report both numbers, never just the clean one.** State the full-eval score, the decontaminated-subset score, and what fraction of eval was removed. A decontaminated score on a silently shrunken eval set is its own distortion.

**Drop from the eval set, not from training** — removing training rows changes the model you are measuring, so you would have to retrain to say anything. Removing eval items changes only the measurement. If the contaminated fraction is large enough that the remaining eval is too small to report, that is the finding: the eval set needs rebuilding, not filtering.

For contamination against a *foundation model's pretraining corpus* — a different problem, since you cannot inspect the corpus — use the n-gram canary and post-cutoff behavioural checks in `yzmir-llm-specialist/llm-evaluation-metrics.md` Part 10.

### Ratio discipline

- **Accumulate, do not replace.** The model-collapse result is about recursive training on generated output *displacing* real data. Keeping real data accruing alongside synthetic is the documented mitigation.
- **Never synthetic in the eval set.** Enforce it as a build-time assertion, not a convention.
- **Establish the ratio by ablation ladder, not by argument.** Train at several synthetic volumes with fixed seeds, hyperparameters, and a decontaminated eval; plot the curve. Hold optimizer steps comparable across arms, or part of the "gain" is simply that the larger arm trained longer.
- **New seeds beat more variants.** When the ladder flattens, the scarce resource is source diversity, and the next real gain comes from collecting cases the seeds do not cover.

---

## Part 6: Eval-Set Construction and Hygiene

The statistics of splitting and leakage are owned by `yzmir-counterfactual-statistics/grouped-splits-and-leakage.md` — the four data roles, the L1–L5 leakage taxonomy with per-class detectors, and cross-fitting when a full split is unaffordable. **Read it for the mechanics; this section covers only construction and lifecycle.**

### Construction

- **Adjudicated, not sampled.** An eval set is built deliberately — labelled by multiple annotators, disagreements resolved by a domain owner — not carved at random from the training pool.
- **Held at the production prior**, even when the training set is rebalanced.
- **Time-forward** where timestamps exist, so the offline number estimates behaviour on an era the model has not seen.
- **Sized against the smallest slice you intend to report**, then stratified up and reweighted, not left to natural prevalence.
- **Frozen** — separate artifact, content hash recorded, with a distinct validation set for model selection so the test set is not consumed by tuning.
- **Record the adjudicated agreement rate on the eval set itself.** That figure is the ceiling every reported number should be read against.

### Lifecycle: the freeze/rotate tension

A frozen eval set decays; a mutating one destroys comparability. Resolve it with an explicit policy rather than drift:

| Policy | Use when | Cost |
|---|---|---|
| **Frozen forever** | Regression suites, contractual benchmarks | Decays against production as traffic shifts; a stable score can mean a stale set rather than a stable system |
| **Append-only, version-bumped** | Most production systems | Comparability holds within a version; every reported number must name the eval-set version |
| **Rolling production sample** | Systems whose traffic shifts fast | Requires ongoing labelling budget; retain prior periods so trends stay visible |

**The realistic default is both:** a frozen regression set that must never degrade, plus a rolling production-prior sample that tracks reality. They answer different questions and neither substitutes for the other.

**Holdout exhaustion is real but is not the whole story.** Repeated tuning against a test set does erode it. The ImageNet-v2 result in Part 2 shows the larger effect: re-running the *curation protocol* moved accuracy 11–14% with no adaptivity involved. The operational consequence is the same either way — a single held-out number is a property of one curation draw, and a system whose decisions depend on 2-point differences needs an interval, not a point.

**Contamination against pretraining corpora** — n-gram canary overlap, behavioural checks against post-cutoff data — is covered in `yzmir-llm-specialist/llm-evaluation-metrics.md` Part 10. Use it whenever a foundation model is involved.

---

## Part 7: Closing the Loop — Drift to Re-Collection

`production-monitoring-and-alerting.md` detects drift (KS, PSI, Evidently). This is the dataset-side response, and the anti-pattern it prevents is **retraining on more of the same data**, which changes nothing when the problem is composition.

1. **Localise.** Which slice moved? Drift alerts are aggregate; the composition manifest (Part 1) is what makes "which segment" answerable.
2. **Classify the drift.** New vocabulary or entities that post-date the corpus; a new channel producing structurally different inputs; a genuine prior shift; or a taxonomy that no longer matches how the business actually routes work. The fourth is not an ML problem and no retrain fixes it.
3. **Collect targeted, not more.** Sample and label from the drifted slice specifically.
4. **Re-release, don't patch.** New dataset version, updated composition manifest, changelog entry naming the drift event that motivated it.
5. **Capture corrections as a labelled stream.** Where a human overrides a prediction, that override is a correctly-distributed label arriving free. Systems that do not log input, prediction, confidence, and final human decision are discarding their best future training data.

---

## Triage: What to Do First

This sheet specifies more work than most teams can do at once. Ordered by how much they change a decision per hour spent, and by what becomes impossible if deferred:

| Priority | Do now | Why it cannot wait |
|---|---|---|
| **1** | **Provenance tagging on generated rows** | Not reconstructable. Every synthetic analysis below is impossible without it, and each day of untagged generation is permanently unattributable |
| **2** | **Dedup, corpus-wide, before role assignment** | Cheap, automated, and irreversible in effect once experiments have run against a contaminated split |
| **3** | **Contamination check on the eval set** | Decides whether every number you currently hold is valid. Hours, not days |
| **4** | **Freeze the eval set with a hash, split validation off it** | Costs almost nothing; without it, tuning silently consumes the test set from here on |
| **5** | **Agreement on a stratified overlap sample** | Establishes the ceiling all model numbers are read against. Needs annotator time, so start the scheduling early even if analysis waits |
| **6** | **Composition manifest + Croissant metadata in the build** | Cheap if automated in the pipeline now; expensive archaeology if written by hand a year later |
| **7** | Coverage/slice taxonomy, confident-learning review queue, ablation ladder | High value, but they assume 1–5 are done — running them on a contaminated or duplicated corpus produces confident wrong answers |

**If you have two days:** 1–4, plus report the contaminated fraction and the residual duplicate fraction as numbers. That is enough to know whether your current metrics mean anything, which is the prerequisite for every other decision.

---

## Pressure Tests

**PT1 — "The schema checks all pass, the data is fine."**
Schema validation proves well-formedness. It cannot see duplication, label error, missing slices, or contamination. Run the Part 3–6 checks; report the measured label-error rate and near-duplicate fraction as numbers, not assurances.

**PT2 — "We'll write the data card after we ship."**
The datasheet questions that matter — sampling rationale, exclusions, consent basis, inference-unsafe fields — are answerable only by the people who built the pipeline, while they still remember. Post-hoc documentation reconstructs what is convenient. Emit Croissant metadata in the build, so an undocumented release cannot be produced.

**PT3 — "Synthetic expansion lifted eval six points, ship it."**
Not until the eval is decontaminated and rescored, a sample is audited for label fidelity, and the ablation ladder is run. Six points is exactly the size contamination produces, and the ladder is what converts the next scaling decision from an argument into a measurement.

**PT4 — "Our test set is frozen, so it's clean."**
Frozen protects comparability, not validity. If it was built before dedup, its siblings are in training. If it was built 18 months ago, it is measuring a distribution that no longer arrives. Check construction order, then check it against a fresh production sample.

**PT5 — "Two annotators labelled it, so it's labelled."**
If they labelled disjoint sets there is no agreement estimate at all. Overlap a stratified sample, compute α per class, and compare their marginals before treating the union as one taxonomy.

**PT6 — "It's anonymised, so the sharing rules don't apply."**
Pseudonymisation is not anonymisation. Sequence data is re-identifiable from a handful of joined attributes, and a stable token remains a join key. The shared extract is still personal data; scope the columns, salt per recipient, and encode the usage policy in Croissant rather than in a meeting.

**PT7 — "Drift alert fired, kick off a retrain."**
Retraining on the same composition reproduces the same gap. Localise the slice, classify the drift, collect against it, re-release. If the taxonomy drifted from how the business routes work, no retrain helps.

---

## Cross-References

**Within this pack:**
- `experiment-tracking-and-versioning.md` — DVC/lakeFS mechanics, hashing, registries, lineage: the *how* of the versioning this sheet's release contract specifies
- `mlops-pipeline-automation.md` — schema validation, data-validation gates, eval sets in CI: where these checks become enforced
- `production-monitoring-and-alerting.md` — drift detection that triggers Part 7

**Other packs:**
- `yzmir-counterfactual-statistics/grouped-splits-and-leakage.md` — authoritative on data roles, the leakage taxonomy, and cross-fitting
- `yzmir-counterfactual-statistics/statistical-units-and-clustering.md` — determining the independent unit that becomes the grouping key
- `yzmir-llm-specialist/llm-evaluation-metrics.md` Part 10 — LLM golden-set discipline and n-gram contamination detection
- `yzmir-llm-specialist/llm-finetuning-strategies.md` — fine-tuning example formats consuming these datasets
- `ordis-security-architect` — PII handling, data-sharing threat modelling, access control on dataset artifacts

**Currency note:** tool inventory verified 2026-08. Croissant (1.1, Feb 2026), annotation platforms, and dedup tooling move quickly — re-check vendor docs before committing a new system to any of them, and prefer capability descriptions over pinned versions.
