---
description: Produce or check a mapping of a system onto EXPO with every term verified against the shipped expo.owl (324 classes) rather than the published paper, which disagrees with its own ontology in roughly ten places. Emits the nine-column mapping table with Verified/Unverified/Refuted verdicts and fit verdicts, plus the gap register. Use it to audit a mapping table someone handed you — circulated tables reliably contain class names that do not exist, and corrections to them reliably deny terms that do.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write"]
argument-hint: "[path_to_system_or_existing_mapping]"
---

# Map to EXPO Command

You are producing or auditing a mapping of a system onto EXPO. The output is a verified nine-column table plus a gap register — **not** a formalisation. Say so when you deliver it.

## Step 1 — Load the authoritative inventory

Read `skills/using-experiment-formalisation/expo-owl-inventory.json`. It carries all 324 classes, 78 object properties, subclass axioms, the four labels differing from their fragments, and 125 disjointness entries, extracted mechanically from the shipped OWL.

Also read `skills/using-experiment-formalisation/verified-terms.json` for the curated layer: hand-checked SUMO terms, paper-vs-OWL corrections, and the do-not-cite list.

**Do not verify against the paper.** `Factor`, `AdminInfoAboutExperiment`, `QualityControlStrategy`, `PlanExperimentalActions`, `TitleOfExperiment`, `MethodComparison`, `SubjectComparison`, `ParedComparison`, and `FalsePositive` all appear in the published paper and **never shipped**.

## Step 2 — If auditing an existing mapping, verify before anything else

For every term the mapping cites:

| Result | Verdict | Action |
|---|---|---|
| In `expo-owl-inventory.json` → `classes` / `object_properties` | **Verified** | Check the citation uses the exact fragment, including misspellings and underscores |
| In `verified-terms.json` → `expo_corrections` | **Refuted as written** | Cite the recorded actual fragment. These are the paper-only names — `Factor`, `AdminInfoAboutExperiment`, `QualityControlStrategy`, `GalileanExp`, `ComputationalExp.` and the rest — and this is the branch that catches them |
| In `verified-terms.json` → `do_not_cite` | **Refuted** | Report with the recorded correct move |
| In `verified-terms.json` → `expo_present_but_widely_denied` | **Verified** | The term ships. Someone "corrected" it wrongly; restore it, and check its structural position (e.g. `IndependentVariable` is an attribute, not a variable type) |
| None of the above | **Unverified** | Grep the OWL. Found → add to inventory. Not found → refuted for EXPO; mint |

**Check all four keys, not just `do_not_cite`.** The corrections layer is where the paper-only names live, and it is the branch most likely to be skipped.

**Check both directions.** Terms commonly "corrected" as fake that are in fact real: `IndependentVariable` (an attribute under `Independence`, not a variable type), `DependentVariable`, `ExperimentalProtocol`, `ExperimentalObservation`.

**Only then** assess fit. Correcting fit on terms that do not exist wastes the work.

## Step 3 — Fit verdicts

Assign one of Exact / Narrower / Broader / Overlapping / Constrained / No-fit per `mapping-a-system.md`. Overlapping and Constrained are the ones that get silently recorded as Exact — consider them deliberately.

Check `disjoint_with` before proposing any combination of design strategies. `Treated_Untreated`, `DoseResponse`, and `TimeCourse` are pairwise disjoint; a design needing several must attach several strategy instances, not one.

**The `disjoint_with` map is one-directional** — each axiom is recorded under one of its two classes only. `disjoint_with["ComparisonControl_TargetGroups"]` does not exist; that axiom lives under `PairedComparisonOfMatchingGroups`. `disjoint_with["DoseResponse"]` lists one entry; four more are recorded under other keys. **Check both directions**: look up the key, *and* search every other key's list for your term. A single key lookup silently passes.

## Step 4 — Dispatch, when producing rather than auditing

Steps 2–3 are the **audit** path: you verify and assign verdicts on the main thread, and the corrected table is the deliverable. Dispatch only when **producing a new mapping**, or when an audit found the existing table unsalvageable and it must be rebuilt from the system.

```
Task(subagent_type="experiment-formalisation-architect",
     description="Verified EXPO mapping for <scope>",
     prompt="Run Stage 3 (inventory and mapping) ONLY. Read the code, not the design
     document. Produce the nine-column table and the gap register. Verify every
     external term against expo-owl-inventory.json before assigning fit. Use exact
     URI fragments. Check disjointness before proposing strategy combinations.
     SYSTEM: <file list>   EXISTING MAPPING (if any): <content>")
```

## Output

The table, the gap register, and a term-verification summary counting Verified / Unverified / Refuted. Close with what remains before this is a formalisation: the competency queries, the context or module, the sync check.

## Cross-references

- `using-experiment-formalisation` — router; the two laws
- `expo-verified-inventory.md` — the sixty-second verification procedure and the paper-vs-OWL discrepancy
- `mapping-a-system.md` — the nine-column table and the six fit verdicts
- `experiment-formalisation-architect` agent — dispatched when producing rather than auditing
- `/formalise-experiment` — the full staged workflow this is Stage 3 of
- `/audit-formalisation` — audits a finished formalisation rather than a mapping
