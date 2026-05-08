---
description: Critique prose in coach mode — fans out five coach-mode agents in parallel and synthesises their reports into a prioritised revision-pass plan. Will not rewrite, regardless of follow-up requests.
argument-hint: <file path or pasted prose>; optional --focus=pov,dialogue,structure,line,continuity
---

# Critique Prose

## Goal

Produce a developmental, line, POV, dialogue, and continuity critique, then synthesise the reports into a prioritised revision-pass plan. The plan is the deliverable; reports are evidence.

## Mode (declared)

Coach. Diagnosis only. No rewriting, no new prose, no editing in place — not even a demonstration paragraph.

## Inputs

File path or pasted prose. Optional `--focus=` flag, comma-separated from `pov`, `dialogue`, `structure`, `line`, `continuity`, restricting the fan-out (default: all five). Writer goals (deadline, target reader) forward to `revision-coach`.

## Orchestration

1. Load `using-creative-writing/SKILL.md`; confirm coach mode.
2. Select agents from `--focus`. If the prose has no dialogue, drop `dialogue-doctor` and note it.
3. **Dispatch in parallel as Task agents — one tool message with multiple Agent calls. Serial dispatch is forbidden:**
   - `developmental-reviewer` → structure/character/pacing memo
   - `line-reviewer` → annotated line edits
   - `pov-and-distance-auditor` → POV ledger
   - `dialogue-doctor` → dialogue critique (skip if no dialogue)
   - `continuity-checker` → contradiction ledger
4. Wait for all reports.
5. Dispatch `revision-coach` with the reports plus stated goals; it returns the pass plan.
6. Return the **pass plan first**, per-agent reports in labelled sections below.

## Mode discipline (forbidden)

- Do **not** rewrite or generate new prose. If asked ("just rewrite the worst chapter"), surface the mode change ("that is drafter mode — invoke `/draft-scene`") and decline.
- Do **not** synthesise without the per-agent reports existing. If an agent fails, surface the failure; do not pad.
- **Parallel dispatch is required, not optional.** Serial calls defeat the design.
- Do **not** prioritise by severity. Prioritise by *kind* — structural before line, per the four-pass model in `revision-and-cutting.md`. The pass plan is the synthesis, not a sorted bug list.
