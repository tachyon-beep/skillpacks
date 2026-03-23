
# Create a Panel Review Config

Help me write a config file for a reader panel review.

## What this does

Guides you through creating a panel review configuration: defining the document suite, designing personas that span the decision chain, writing scenario framing, and setting up panel controls (control persona, unreliable narrator, collision pairings).

## How to use

Tell me:
1. **What documents** you want reviewed (paths, or describe them)
2. **Who the audiences are** — who reads this, who decides based on it, who is affected by it
3. **The institutional context** (optional) — how readers received the documents, what status they have

I'll help you design personas and write a config file following the format in `plugins/muna-panel-review/config-template.md`.

Alternatively, if you have the documents ready and want automated panel design, use `/panel-designer` to spawn the persona-designer agent directly.

## Design principles

- **Span the decision chain** — from the person who decides to the person most affected
- **Include someone talked about but not talked to** — they surface the largest editorial gaps
- **Define blind spots explicitly** — this is what prevents all personas sounding the same
- **Give each persona a distinct voice sample** — one sentence in character, how they talk to a peer

## References

- `plugins/muna-panel-review/config-template.md` — config format with field descriptions
- `plugins/muna-panel-review/config.md` — fully worked 13-persona example
- `plugins/muna-panel-review/process.md` Phase 1 — panel design principles
