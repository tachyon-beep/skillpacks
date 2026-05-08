---
description: Workshop-voiced craft pack for prose narrative — fiction and creative nonfiction. Three explicit modes (draft / critique / plan); the router refuses to work without a declared mode. v0.1 ships router + 13 sheets + 3 commands + 8 agents.
---

# Creative Writing Routing

**Prose narrative only at v0.1 (fiction + creative nonfiction). Poetry, scripts, plays, comics, songwriting, and interactive fiction are scheduled for v0.4.**

Use the `using-creative-writing` skill from the `lyra-creative-writing` plugin. The router enforces mode declaration: it asks whether you want to **draft**, **critique**, or **plan** before doing any work. To skip the mode prompt, invoke a dedicated command directly:

- **`/draft-scene`** — Claude generates new prose from a brief. Drafter mode. Outputs prose only with optional craft-notes appendix; will not auto-critique its own output, will not modify your existing prose.
- **`/critique-prose`** — Claude diagnoses prose you paste. Coach mode. Fans out five coach-mode agents in parallel (`developmental-reviewer`, `line-reviewer`, `pov-and-distance-auditor`, `dialogue-doctor`, `continuity-checker`), then synthesises through `revision-coach` into a prioritised revision-pass plan. Will not rewrite, regardless of follow-up requests.
- **`/plan-story`** — Claude consults on premise, outline, structural lens. Consultant mode. Returns outline + weak-spots + lens caveat. Will not draft prose, will not prescribe a single structural lens as universal.

## v0.1 sheets (13)

`pov-and-voice`, `scene-construction`, `dialogue`, `pacing-and-tension`, `character-interiority`, `showing-vs-telling`, `prose-rhythm-and-style`, `story-structure-and-arc`, `research-and-verisimilitude`, `worldbuilding-by-implication`, `revision-and-cutting`, `openings-and-endings`, `creative-nonfiction-craft`. The router loads one to three sheets per session based on the conversation; it does not load all thirteen.

## Composition

Beta-reader panels and simulated-reader reactions live in `muna-panel-review` (`/panel-review`). The router defers there rather than reinventing reader simulation. Technical writing — different voice, different goals — lives in `muna-technical-writer`; not a substitute.

## Roadmap

- **v0.2** — genre annex sheets (mystery, romance, sf/f, horror, literary fiction, memoir, literary journalism), plus `worldbuilding-consultant`, `opening-and-ending-doctor`, `premise-stress-tester` agents.
- **v0.3** — other craft lineages: structuralist (Save the Cat, Story Grid), genre-mechanical, pluralist (named-traditions-side-by-side).
- **v0.4** — format expansion: poetry, screenwriting/teleplay, stage plays, comics scripting, songwriting/lyrics, interactive fiction.
