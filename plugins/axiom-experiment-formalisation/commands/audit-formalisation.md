---
description: Adversarially audit an existing experiment formalisation — a mapping table, JSON-LD context, OWL module, SHACL shapes, competency-query suite, or a design doc proposing one — against the axiom-experiment-formalisation failure catalogue. Every cited external term is checked mechanically against the shipped EXPO OWL inventory and the SUMO sources; the audit also checks the layering (no shape or reasoner on a runtime path), the sync check, the disjointness axioms, and whether the record can express its own failure. Dispatches the formalisation-critic agent and returns severity-rated findings with evidence and a machine-readable summary.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[path_or_diff_ref]"
---

# Audit Formalisation Command

You are auditing an existing formalisation via the `formalisation-critic` agent. The output is severity-rated findings. This command does not redesign (that is `/formalise-experiment`) and does not edit files.

## Step 1 — Scope the audit

Locate, within the path argument (default `.`):

- Mapping tables, gap registers, term inventories.
- JSON-LD contexts, OWL/TTL modules, SHACL shapes.
- Competency queries (`.rq`, SPARQL in code, test files) and their fixtures.
- The typed record definitions the formalisation claims to describe.
- CI configuration — is there a sync check, and is it muted?
- **The runtime** — needed for the layering check.

For a diff ref, audit the diff plus enough surrounding artifact for the critic to judge its claims.

If no formalisation artifacts are found, say so and stop. Do not audit arbitrary schemas as though they were a formalisation.

## Step 2 — Mechanical pre-pass

Cheap, and it sharpens the critic's attention. Extract every term attributed to an external ontology and diff it against `expo-owl-inventory.json` and `verified-terms.json`. Grep the runtime for SHACL/reasoner/SPARQL invocations outside CI and offline analysis. Note both results in the dispatch.

## Step 3 — Dispatch the critic

```
Task(subagent_type="formalisation-critic",
     description="Audit formalisation at <scope>",
     prompt="Audit the following formalisation against your 20-entry checklist.
     Sweep every entry. Run the vocabulary check FIRST and mechanically, in both
     directions — hallucinated citations AND terms wrongly reported absent.
     Check the disjointness axioms, the sync check, the layering, and whether the
     outcome vocabulary can express that the no-intervention arm won.
     Produce findings ordered by severity, the machine-readable summary with
     checked/not_assessable per entry, the term-verification table, and the four
     SME protocol sections. Do not rubber-stamp; do not pad.

     AUDIT SURFACE: <file list or diff ref + context files>
     PRE-PASS RESULTS: <unmatched terms; runtime grep results>")
```

## Step 4 — Report

Relay findings by severity with their evidence, the machine-readable summary, and the term-verification table. Where the critic reports an entry as not assessable, say what artifact would make it assessable.

**A zero-finding audit is reported as a defect of the audit, not a clean bill of health** — name what could not be obtained.

## Verification

Confirm the critic's output contains: at least one line of quoted evidence per finding, the machine-readable summary covering all 20 catalogue entries (or explicit `not_assessable` reasons), the term-verification table, and the four SME protocol sections. Send it back for anything missing.

Suggested follow-up: fix critical/high findings; re-run this command on the remediation diff; wire the IRI-resolution and sync checks into CI so the mechanical subset never regresses.

## Cross-references

- `using-experiment-formalisation` — router; the 20-entry failure catalogue this audits against
- `formalisation-critic` agent — the auditor this command dispatches
- `/formalise-experiment` — when the audit shows the formalisation needs designing, not patching
- `/map-to-expo` — when only the vocabulary layer is in question
