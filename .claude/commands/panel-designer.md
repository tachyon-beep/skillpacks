
# Design a Reader Panel from Documents

Spawn the persona-designer agent to analyse your documents and produce a complete panel review config file.

## What this does

The persona-designer reads your documents freely, identifies audiences, decision chains, and institutional perspectives, and writes a complete config file with persona specifications, scenario framing, and panel configuration.

## What you need

- Path to your document files (or a directory containing them)
- Optional: context about intended audiences
- Optional: desired panel size

## How it works

The designer produces a ready-to-use config file including:
- Document suite table
- Scenario framing with rationale
- Full persona specs (all required fields)
- Panel configuration (control, unreliable narrator, priority, collisions)
- Panel gaps — audiences identified but not included, with reasoning

Review the generated config, edit as needed, then run `/panel-review` with it.

## Note

The designer reads your documents in full. It proposes reading behaviour as a character trait prediction, not a content-informed route. The actual reading path is determined by each persona-reader at runtime.
