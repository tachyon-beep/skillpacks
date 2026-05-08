---
description: Draft a new scene in drafter mode. Outputs prose only with optional craft-notes appendix. Will not auto-critique its own output, will not modify user's existing prose.
argument-hint: <scene-brief or file path containing brief>
---

# Draft Scene Command

## Goal

Produce a scene draft from a brief.

## Mode (declared)

**Drafter.** Prose only, with an optional craft-notes appendix. No critique of the brief, no modification of existing prose, no auto-critique of the output.

## Inputs

**Required:** scene brief or beats.

**Optional:** voice samples, POV/distance constraint, target length, characters present, setting, register.

If required inputs are missing, ask once.

## Orchestration

- Load `using-creative-writing/SKILL.md` (the router). Confirm drafter mode.
- Pull from `plugins/lyra-creative-writing/skills/using-creative-writing/`: `pov-and-voice.md`, `scene-construction.md`. Conditionally pull `dialogue.md` (if dialogue), `character-interiority.md` (if close third or first), `worldbuilding-by-implication.md` (if setting is fantastical, historical, or distinctive).
- Dispatch the `scene-drafter` agent with: the brief, voice samples if any, the POV/distance constraint, and the loaded sheet contents.
- Return the prose draft, then an optional `## Craft notes` appendix the writer can ignore.

## Mode discipline (forbidden)

- Do **not** auto-critique the output. If the writer wants critique, they invoke `/critique-prose` after.
- Do **not** modify any prose the user has *already* written. The drafter agent only generates new prose.
- If the user follows up mid-conversation with a critique request ("now tell me what's wrong with what you wrote"), surface the mode change ("that is coach mode — invoke `/critique-prose` or confirm you want me to switch?") rather than silently obliging.
