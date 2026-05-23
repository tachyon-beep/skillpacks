---
description: Design a Reader Panel from Documents - dispatch the persona-designer agent to analyse documents and produce a complete panel config file
allowed-tools: ["Read", "Write", "Glob", "Agent", "AskUserQuestion"]
argument-hint: "[documents_path] [optional: audience_context] [optional: panel_size]"
---

# Design a Reader Panel from Documents

Dispatch the `muna-panel-review:persona-designer` agent to analyse the user's documents and produce a complete panel-review config file.

## What you need from the user

Before dispatching, confirm:

1. **Document paths** — file paths or a directory containing the document suite to design a panel for. Required.
2. **Audience context** (optional) — known facts about who reads or is affected by these documents (institutional setting, decision chain, regulatory context).
3. **Desired panel size** (optional) — defaults to the designer's judgment based on document complexity; typical range is 7–13 personas for a full panel.

If the user has not supplied document paths, ask for them before dispatching. Do not invent paths.

## Dispatch instructions

Spawn the `muna-panel-review:persona-designer` agent with a prompt containing:

- Absolute paths to the documents (or the containing directory)
- Audience context if provided, or "none provided" otherwise
- Desired panel size if provided, or "designer's judgment" otherwise
- Absolute path to `plugins/muna-panel-review/process.md` (Phase 1 is the designer's authority)
- Absolute path to `plugins/muna-panel-review/config-template.md` (the output format the designer must produce)
- Output path for the config file (ask the user, or default to `panel-config.md` in the current working directory)

The designer reads the documents freely (contamination is permitted for the designer; it is not a simulated reader) and writes a config file with:

- Document suite table
- Scenario framing with rationale
- Full persona specs (all required fields)
- Panel configuration (control persona, unreliable narrator, priority, collisions)
- Panel gaps — audiences identified but not included, with reasoning

## Contamination constraint

The designer does NOT predict reading behaviour from chapter content, does NOT predict verdicts, and does NOT write reading routes informed by what the chapters say. This is a load-bearing constraint enforced inside the agent (see `persona-designer.md:77-91`). Do not relax it in the dispatch prompt.

## Handoff

When the designer returns, present the config path to the user and recommend they review and edit before running `/panel-review`.
