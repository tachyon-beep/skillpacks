---
description: Create a Panel Review Config - guide the user through writing a config file, or dispatch the persona-designer for automated design
allowed-tools: ["Read", "Write", "Glob", "Agent", "AskUserQuestion"]
argument-hint: "[documents_path_or_description]"
---

# Create a Panel Review Config

Help the user produce a panel-review config file. There are two routes — pick the one that matches what the user has.

## Route A: User has the documents ready

If the user can supply file paths to the document suite, prefer automated design. Dispatch the `muna-panel-review:persona-designer` agent (same dispatch contract as `/panel-designer`):

- Absolute paths to the documents
- Any audience context the user provides
- Desired panel size (optional)
- Absolute path to `plugins/muna-panel-review/process.md` (Phase 1 authority)
- Absolute path to `plugins/muna-panel-review/config-template.md` (output format)
- Output path for the generated config

The designer returns a complete config — review it with the user and edit as needed before `/panel-review`.

## Route B: User does not have documents yet, or wants to hand-design

If the user is in pre-document planning, or wants to author the config by hand, walk them through it interactively. Ask:

1. **What documents** will be reviewed (paths, titles, or descriptions)
2. **Who the audiences are** — who reads it, who decides based on it, who is affected by it
3. **Institutional context** (optional) — how readers received the documents, what status they have, what the document is meant to accomplish

Then help them draft a config following `plugins/muna-panel-review/config-template.md`. Reference `plugins/muna-panel-review/config.md` (worked 6-persona cloud-migration example) for shape and tone.

## Design principles (carry these through both routes)

- **Span the decision chain** — from the person who decides to the person most affected
- **Include someone talked about but not talked to** — they surface the largest editorial gaps
- **Define blind spots explicitly** — this is what prevents all personas sounding the same
- **Give each persona a distinct voice sample** — one sentence in character, how they talk to a peer

These principles are documented in full at `plugins/muna-panel-review/process.md` Phase 1.

## Handoff

Once a config exists, recommend the user run `/panel-review` with the config path, document paths, and an output directory.
