---
description: Workshop-voiced craft pack for prose narrative — fiction and creative nonfiction. Three explicit modes (draft / critique / plan); the router refuses to work without a declared mode. v0.2 ships router + 22 sheets (13 craft + 9 genre) + 3 commands + 11 agents.
---

# Creative Writing Routing

**Prose narrative only (fiction + creative nonfiction). Poetry, scripts, plays, comics, songwriting, and interactive fiction are scheduled for v0.4. The router will not begin work until you declare a mode — draft, critique, or plan.**

Use the `using-creative-writing` skill from the `lyra-creative-writing` plugin to route to the right specialist sheet. Content authority lives in `plugins/lyra-creative-writing/skills/using-creative-writing/SKILL.md` — this wrapper is a thin pointer.

## When to Use

- Drafting fiction or creative-nonfiction prose from a scene brief
- Getting craft critique on prose you have already written
- Planning premise, outline, or structural lens for a longer work
- Working within a specific genre's reader contract (mystery, thriller, sf, fantasy, horror, romance, literary fiction, memoir, literary journalism)

**Don't use** for: beta-reader simulation (use `/panel-review`), technical writing (use `/technical-writer`), poetry/scripts/comics/songwriting (deferred to v0.4).

## Sheets

### Craft sheets (13)

`pov-and-voice`, `scene-construction`, `dialogue`, `pacing-and-tension`, `character-interiority`, `showing-vs-telling`, `prose-rhythm-and-style`, `story-structure-and-arc`, `research-and-verisimilitude`, `worldbuilding-by-implication`, `revision-and-cutting`, `openings-and-endings`, `creative-nonfiction-craft`.

### Genre annex (9)

`mystery`, `thriller`, `sf`, `fantasy`, `horror`, `romance`, `literary-fiction`, `memoir-and-personal-essay`, `literary-journalism`. Each presents conventions through a reader-contract frame: name the contract, name the cost of breaking, name books worth the cost.

The router loads one to three sheets per session based on the conversation; it does not load all twenty-two.

## Commands

- `/draft-scene` — drafter mode: generate new prose from a brief; will not auto-critique its own output, will not modify existing prose
- `/critique-prose` — coach mode: fan out five coach-mode agents in parallel, synthesise through `revision-coach` into a prioritised pass plan; will not rewrite
- `/plan-story` — consultant mode: premise, outline, structural lens; returns outline + weak-spots + lens caveat; will not draft prose

## Agents

Coach: `developmental-reviewer`, `line-reviewer`, `pov-and-distance-auditor`, `dialogue-doctor`, `continuity-checker`, `opening-and-ending-doctor`.
Drafter: `scene-drafter`.
Consultant: `outline-architect`, `worldbuilding-consultant`, `premise-stress-tester`, `revision-coach`.

All eleven enforce the mode-discipline contract in their own boundary sections.

## Composition

- Beta-reader simulation and reader-panel reactions → `/panel-review` (`muna-panel-review`)
- Technical writing — different voice, different goals → `/technical-writer` (`muna-technical-writer`)

## Roadmap

- **v0.3** — other craft lineages: structuralist (Save the Cat, Story Grid), genre-mechanical, pluralist (named-traditions-side-by-side)
- **v0.4** — format expansion: poetry, screenwriting/teleplay, stage plays, comics scripting, songwriting/lyrics, interactive fiction
