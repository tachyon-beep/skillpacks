# Simulated Reader Review Methodology

This is an editorial hypothesis experiment. Model personas do not establish human
reactions, demographic representativeness, sensitivity, institutional response,
market demand or stakeholder acceptance. Use relevant humans or independent
measures to validate claims about those outcomes.

The canonical skill owns coordination; this file defines experimental contracts
and optional artifact formats. Load only sections needed by the current role.

## Phase 1: Design the experiment

Start with the decision and uncertainty. Use a small number of distinct audience
lenses for an ordinary review. A staged cold read is useful when order of exposure,
signposting or expectation mismatch matters; it is unnecessary for every document
critique. More personas do not automatically improve evidence.

A lens records role/task, relevant knowledge, constraints, key question and blind
spots. Distinguish evidence about actual readers from assumptions. Voice and
scenario framing are optional ways to explore a perspective, not proof that the
model has that person's experience. Do not infer identity-specific reactions from
stereotypes. Document material audiences absent from the experiment.

Use [config-template.md](config-template.md) for run configuration.
[config.md](config.md) is an illustrative larger panel; its size and optional
control/collision mechanisms are not defaults. Honor the user's scope and budget;
ask only when a consequential cost/choice is unresolved.

### Optional internal checks

An expected control result can check whether a run followed its intended lens.
Register the expectation before exposure if using it. A match does not calibrate
real audience validity or increase confidence in unrelated persona predictions.

An intentionally mistaken lens can test text interpretation under that assumption;
it must be labeled and must not be counted as independent counterevidence.

## Phase 2: Review or staged reading

### Simple lens review

Read the supplied material and inspect it against the selected task/knowledge lens.
Cite passages, plausible interpretation mechanisms and alternatives. Label output
as a single-context lens review if the reader has seen the complete document.
There is no claim of cold-reading independence in that mode.

### Staged information contract

For an uncontaminated staged run, each reader starts with only their lens and a
suite map of titles/chapter IDs, then receives requested content incrementally.
Do not supply future chapters or the coordinator's findings. Filename hashing is
an organizational convenience, not a permission boundary. Use actual restricted
contexts/content delivery when available and disclose limitations.

For each chapter, save A before exposure; the coordinator verifies it before
supplying content. After exposure save B and C. Do not rewrite A retrospectively.
The reader may continue, skip, switch, return or stop; these are simulated choices.
Missing/tool-failed reads must not be presented as deliberate reader rejection.

```markdown
# Journal — <lens>, <document/chapter ID>

## A. Expectations — saved before exposure
What do the supplied map and prior chapters lead this lens to expect?
What information does the reader's task need here?

## B. Observations — saved after exposure
- Passage/location and directly inspectable text property.
- Expectation met or mismatch, with a plausible explanation and alternatives.
- Optional simulated reaction, explicitly labeled as such.
- Unresolved information and any contamination/access failure.

## C. Next
Continue, skip, switch, return or stop; reason and requested next ID.
```

Use brief entries for unremarkable material. No required emotional vocabulary,
minimum reaction count or voice performance applies. If a lens drifts toward a
generic critic, restore its task/knowledge constraints from the ledger. Re-anchor
on observed drift or context loss, not a fixed chapter interval.

### Context and failure recovery

Keep a persistent record of supplied chapters, state, expectations, observations,
requests and failures. On return/context loss provide relevant journal entries and
state summary without exposing unread chapters. Do not pretend lost chronology or
read-ahead remained uncontaminated; restart the affected probe with fresh context
when useful and affordable, or disclose the limitation.

Retry a transient error within the agreed budget. If recovery fails, preserve
partial coverage and mark the failure. The surviving panel is not equivalent to
the planned panel; report missing lenses and chapters. Use available host tools
and permissions; no particular agent/team API or bypass setting is required.

## Phase 3: Reader verdict

Save a verdict when a reader finishes or stops early:

```markdown
# Verdict — <lens>
- Material actually read, reading path and stop/failure reason.
- Supported text observations with source locations.
- Simulated interpretations/reactions, with alternative explanations.
- Proposed editorial hypotheses and the reader task they affect.
- Missing information, contamination and other limitations.
```

Separate source facts from simulated emotion or institutional speculation. Do not
call this genuine participant testimony or use a persona's approval as real sign-off.

### Optional collision exercise

When divergent interpretations matter, compare selected verdicts or ask fresh
bounded readers to respond to them. Label the exercise as post-exposure and
correlated model output. Agreement after discussion does not establish real
negotiation or resolve a factual dispute. Skip when it cannot affect the decision.

## Phase 4: Synthesis

Synthesize directly or use a separate role when independent aggregation helps.
Read config, actual coverage, observations and cited passages. Check source
context rather than quoting persona fluency as authority. Collate reading paths
and source-linked themes only to the extent needed; a full thirteen-section report
is not required.

### Evidence classes

| Class | What can be claimed | Validation needed |
|---|---|---|
| Text-surface observation | Actual length, term presence, missing definition, contradictory statement, broken link or other inspectable property. | Check the original text or suitable tool. |
| Audience interpretation/affect hypothesis | A plausible way a lens might misunderstand, feel excluded or lose interest. | Relevant human task/reading evidence; the model cannot establish prevalence. |
| Institutional/commercial hypothesis | A possibility about decisions, costs, consultation or market response. | Domain/organizational evidence and authorized human judgment. |

A simulated statement about vocabulary burden is not itself a measured human
burden. A skipped chapter is not proven unnecessary or poorly positioned. Inspect
its role, alternative navigation paths and the task before proposing a change.

Counts report outputs in this run only. They are not independent samples,
prevalence estimates, confidence calibration or severity scores. Similar models,
source access and prompts can produce shared priors even when reasons differ.
Check interchangeable justifications and shared sources, but do not treat that
check as proving independence. Prioritize by source evidence and reader consequence,
not how many personas repeat a claim.

### Synthesis artifact

A useful `00-reader-panel-synthesis.md` contains:

1. Decision and scope; actual lenses, reading coverage and missing/failed work.
2. Prioritized editorial hypotheses with cited passages/observations, evidence
   class, plausible mechanism and alternatives.
3. Material disagreements, supported strengths and audience gaps.
4. Proposed smallest changes and validation tests; what would support or overturn
   the recommendation.
5. Limits: synthetic evidence, correlated contexts, contamination and untested claims.

Save `00-process-manifest.yaml` for reproducibility when useful: inputs/version,
model/runtime, lenses, mode, actual supplied content/reading paths, output files,
failures/contamination, budget and checks. Do not invent model identifiers or costs
that the runtime did not expose.

## Phase 5: Optional derivative work

A hypothesis may motivate an executive brief, practitioner guide or other derivative.
Create it only within the user's requested scope. Preserve source meaning and
label new recommendations; use `muna-wiki-management` for lineage/fidelity when useful.
Synthetic outline reviews can generate further hypotheses, but final acceptance
must come from actual stakeholders or the project's acceptance procedure.
