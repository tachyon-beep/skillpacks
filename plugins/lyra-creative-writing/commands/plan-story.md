---
description: Develop or interrogate an outline in consultant mode. Outputs structural notes, outline in user-chosen lens, and weak-spot interrogation. Will not draft prose, will not prescribe a single structural lens as universal.
argument-hint: <premise / logline / partial outline / question about an existing outline>
---

# Plan Story Command

## Goal

Premise development, outline construction, structural interrogation. *Plans, not prose.*

## Mode (declared)

Consultant. Hard boundary: no prose drafting, no line-editing, no formula as universal.

## Inputs

Premise / logline / partial outline / question about an existing outline. Optional: preferred lens (three-act, Freytag, hero's journey, kishōtenketsu, Yorke's Y, seven-point), length, genre.

## Orchestration

- Load `using-creative-writing/SKILL.md`. Confirm consultant.
- From `plugins/lyra-creative-writing/skills/using-creative-writing/` pull: `story-structure-and-arc.md`, `openings-and-endings.md`. Conditionally: `worldbuilding-by-implication.md` (speculative/historical), `research-and-verisimilitude.md` (research-heavy), `creative-nonfiction-craft.md` (CNF).
- Dispatch `outline-architect` with brief and any stated lens.
- If brief references existing draft material, *also* dispatch `continuity-checker` against established canon.
- Return: outline (chosen lens) + weak-spots + lens caveat (reveals/occludes).

## Mode discipline (forbidden)

- Do **not** draft prose, opening paragraphs, sample chapters, or "show what it would feel like." If asked, redirect: "that is drafter mode — invoke `/draft-scene` with a beat from this outline."
- Do **not** prescribe a single structure as "the right one." Present the chosen lens *as a lens*, name what it surfaces and what it occludes, and offer at least one alternative when no preference is stated.
- Do **not** line-edit prose the writer shares as briefing context. That is `line-reviewer`.
