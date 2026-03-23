
# Run a Reader Panel Review

Use the `muna-panel-review:reader-panel-review` skill to orchestrate a simulated reader panel review.

## What this does

Spawns a panel of simulated readers — each representing a distinct audience — who read your documents chapter by chapter and record mood journals. Produces editorial intelligence: how the document lands with different audiences, where it loses readers, and what derivative documents are needed.

## What you need

- A config file defining your document suite, personas, and optional scenario framing
- The document files at accessible paths
- See `plugins/muna-panel-review/config-template.md` for the config format
- See `plugins/muna-panel-review/config.md` for a worked 13-persona example

## To start

Load the skill: `muna-panel-review:reader-panel-review`

Then provide:
1. Path to your config file
2. Path to your document files
3. Output directory (defaults to `panel-review/` inside the project)

## Cost warning

This is a token-intensive workflow. A 3-persona panel on a 4-chapter document generates ~15+ agent turns. A 13-persona panel on a large document suite generates hundreds. Confirm panel size before starting.
