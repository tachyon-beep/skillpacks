---
description: Run a Reader Panel Review - orchestrate a simulated audience panel reading documents chapter-by-chapter and synthesise findings
allowed-tools: ["Read", "Write", "Glob", "Bash", "Skill", "Agent"]
argument-hint: "[config_path] [documents_path] [output_dir]"
---

# Run a Reader Panel Review

Load and execute the `muna-panel-review:reader-panel-review` skill to orchestrate a simulated reader panel review.

## What this does

Spawns a panel of simulated readers — each representing a distinct audience — who read your documents chapter by chapter and record mood journals. Produces editorial intelligence: how the document lands with different audiences, where it loses readers, and what derivative documents are needed.

## What you need

- A config file defining your document suite, personas, and optional scenario framing
- The document files at accessible paths
- See `plugins/muna-panel-review/config-template.md` for the config format
- See `plugins/muna-panel-review/config.md` for a worked 13-persona example

## How to invoke

Load the skill: `muna-panel-review:reader-panel-review`

Then provide:

1. Path to your config file
2. Path to your document files
3. Output directory (defaults to `panel-review/` inside the project)

If the user has not yet written a config file, the skill's Phase 0 will offer to spawn the `persona-designer` agent. Alternatively, the user can run `/panel-designer` directly.

## Cost warning

This is a token-intensive workflow. A 3-persona panel on a 4-chapter document generates ~15+ agent turns. A 13-persona panel on a large document suite generates hundreds. The skill enforces a cost-confirmation handshake before any agents are spawned — do not bypass it.
