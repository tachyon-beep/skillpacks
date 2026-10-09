---
name: reader-panel-review
description: "Use when simulated audience lenses or staged cold-reading experiments would help generate editorial hypotheses for a document or document suite."
---

# Simulated Reader Review

Generate editorial hypotheses from explicit audience lenses. A model persona is
not a human participant; agreement, emotion, predicted navigation, and synthetic
acceptance do not establish real audience response or lived experience.

## Select the smallest useful experiment

Infer the authorized scope and budget from the request. For an ordinary review,
use a small set of distinct lenses in one pass. Use sequential cold-reading only
when order of information, expectation mismatch, or navigation is the uncertainty.
Use extra agents, collision exercises, or large panels only when their additional
coverage justifies the cost. Ask about budget only when it is consequential and
not already agreed; do not repeat authorization at routine phases.

Read relevant sections of [process.md](../../process.md), not the entire process
in every role. [config-template.md](../../config-template.md) defines the config;
[config.md](../../config.md) is an illustrative larger panel, not the default.

## Prepare

1. Locate the actual documents and output directory. Validate paths, chapter IDs
   and configured lenses. Record intended audience, known evidence, and assumptions.
2. Select lenses for different tasks/knowledge/constraints, not decorative voices.
   Explain material missing audiences. Use optional roles `persona-designer`,
   `persona-reader`, and `panel-synthesiser` only where useful.
3. Use the runtime's available delegation, messages and file capabilities. Honor
   its permissions and user authorization; never prescribe a bypass mode. If
   separate contexts are unavailable, run a labeled single-context lens review
   and disclose that it cannot simulate an uncontaminated cold read.
4. For staged reading, give readers only the suite map and the current requested
   chapter. A filename hash is an organizational aid, not an access-control boundary.
   Use actual access restrictions or coordinator-provided content when available.

## Staged reading contract

Keep a persistent per-reader ledger: lens, supplied chapters, current position,
completed expectations/observations, outstanding request, status, and failures.

For each chapter:

- **A — Before exposure:** save concrete expectations from the current map and
  previously supplied material. Do not supply the chapter until A exists.
- **B — After exposure:** cite text that met or broke the expectation; separate
  directly inspectable text properties from simulated feelings or institutional
  speculation. Do not retrofit A after reading.
- **C — Next:** choose continue, skip, switch, return, or stop and explain why.

Use brief entries unless the observation warrants detail. Readers may stop early.
On context loss or document switches, restore from their ledger and relevant
journal entries. Re-anchor the lens when drift is observed, not every fixed count.
Report accidental read-ahead as contamination; restart affected observations with
fresh context if useful and feasible. Do not present them as cold-reading evidence.

## Recovery and synthesis

Retry a transient failure within the agreed budget; preserve state and record any
missing coverage. Do not silently count an unfinished reader as agreement or
relabel a tool failure as a reader decision. Synthesize available material without
requiring a separate agent.

For each editorial hypothesis, cite source passages and relevant observations,
explain a plausible mechanism and alternatives, and propose a cheap validation.
Counts describe this run's outputs only; they are not independent samples,
prevalence estimates, or severity scores. An expected control result is an
internal consistency check, not calibration of human audience validity.

## Deliver

Return a concise prioritized hypothesis report, actual coverage/reading paths,
material disagreements, limitations, and next tests. A larger run may also save
journals, verdicts, synthesis and a process manifest as described in `process.md`.
Label text-surface observations, simulated affect and institutional speculation
separately. Confirm audience claims with relevant humans or independent measures.
Do not create derivative documents or treat a synthetic sign-off as stakeholder
acceptance unless the user requested that further work.
