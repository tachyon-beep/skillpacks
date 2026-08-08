---
description: Design an experiment formalisation end to end — triage (should you formalise at all), competency questions with controls, verified mapping against the shipped EXPO OWL, gap register and mints, the JSON-LD context or OWL module, the validation suite, and the CI sync check that keeps the ontology subordinate to the typed contracts. Runs as a staged orchestrator that halts for review after every stage; it will recommend against formalising when no consumer exists.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit"]
argument-hint: "[path_to_system_or_design_doc]"
---

# Formalise Experiment Command

You are designing an experiment formalisation via the `experiment-formalisation-architect` agent, in stages, with a review gate after each.

**Do not run all stages in one pass.** A one-shot formalisation is exactly the condition under which unverified terms and category errors proliferate. Each stage produces a reviewable artifact and stops.

## Step 0 — Load the laws

Read `skills/using-experiment-formalisation/the-projection-law.md`. Both laws bind everything below:

- **Verified vocabulary only** — cite `expo-owl-inventory.json` (324 classes from the shipped `expo.owl`) or the SUMO source; never a plausible name; never report a term absent without checking.
- **The projection law** — the formalisation describes the contracts; it never becomes runtime truth.

## Step 1 — Scope

1. **Path argument** (default `.`): locate the record definitions (typed schemas, dataclasses, IDL, `contracts/`, `schemas/`), their emitters, their consumers, and any existing ontology/mapping artifacts.
2. **Design doc**: use it for intent, but **read the code for facts**. The gap between them is usually where findings live.
3. If no typed records exist at all, say so and stop — recommend `/contract-engineering` first. A formalisation over records that do not fail closed is describing sand.

## Step 2 — Stage 1: triage. HALT.

```
Task(subagent_type="experiment-formalisation-architect",
     description="Triage formalisation for <scope>",
     prompt="Run Stage 1 (triage) ONLY, per formalisation-triage.md. Produce the
     triage record: verdict (don't formalise / Tier 1 / Tier 2), reasoning against
     the gates, the NAMED consumer or an explicit statement that there is none,
     the Must competency questions and which are unanswerable today, the stop
     condition, and the revisit trigger. Recommend against formalising if the
     gates say so. Do not proceed to any later stage.
     SYSTEM: <file list, record definitions, consumers, existing artifacts>")
```

**Present the verdict and stop.** If it is "don't formalise," that is the deliverable — deliver it and stop. Do not proceed on your own initiative.

## Step 3 — Stage 2: competency questions. HALT.

Dispatch for Stage 2 only. Verify the gate before continuing: **at least one Must question must currently be answerable only wrongly.** If none is, return to the triage verdict.

## Step 4 — Stage 3: verified mapping. HALT.

Dispatch for Stage 3 only. Before presenting, check yourself:

- Every external term appears in `expo-owl-inventory.json` or `verified-terms.json`, cited as the exact URI fragment (including source misspellings such as `FalsePpositive`).
- No term from the do-not-cite list is cited as EXPO.
- No `ExperimentalDesignStrategy` combination violates the `disjoint_with` axioms.
- Every row cites a real path.

## Step 5 — Stages 4–6: mints, artifacts, validation

Dispatch each in turn, halting between. The final output is artifacts on disk — context or module, queries, gap register, sync check, CI wiring — not a description of them.

## Step 6 — Audit

Run `/audit-formalisation` against what was produced. An architect's own work is not self-certifying.

## Output

Report per stage: what was produced, where it lives, what the review gate found, and what remains. State plainly if the verdict was "don't formalise" — that is a successful outcome of this command, not a failure of it.

## Cross-references

- `using-experiment-formalisation` — router; failure-mode catalogue source of truth
- `experiment-formalisation-architect` agent — the designer this command dispatches, stage by stage
- `/map-to-expo` — Stage 3 alone, when only the verified mapping is wanted
- `/formalise-design` — Stage 3's design layer alone, the hallucination-prone slice
- `/audit-formalisation` — the independent audit this command ends with
- `/contract-engineering` — run first if the records do not yet fail closed
