---
name: filigree-workflow
description: >
  This skill should be used when the user asks to "track work", "create an issue",
  "find something to work on", "what should I work on next", "triage bugs", "close
  an issue", "check what's blocked", "plan a milestone", "review sprint progress",
  "coordinate agents", or when working in a project that uses filigree for issue
  tracking. Provides workflow patterns, team coordination protocols, and operational
  guidance for the filigree issue tracker.
---

# Filigree Workflow

Filigree is an agent-native issue tracker that stores its data locally in the
project. This skill is procedural knowledge for using it well — as a solo agent
or in a multi-agent swarm. Prefer the `mcp__filigree__*` tools when available;
fall back to the `filigree` CLI. `filigree --help`, `filigree <verb> --help`
and the MCP tool schemas are the authoritative catalogues.

## Core Workflow

```
filigree session-context                            → ready / in-progress / critical path
filigree ready                                      → find available work (no blockers)
filigree show <issue-id>                            → read requirements and context
filigree start-work <issue-id> --assignee <name>    → atomically claim + transition into its working status
[do the work, commit code]
filigree close <issue-id> --reason="summary of what was done"
```

Or skip the search entirely with `filigree start-next-work --assignee <name>`
to grab the highest-priority **startable** issue.

Always close with a `--reason` — it becomes audit trail for the next agent, and
add a comment before any handoff. The next agent has no memory of this
conversation.

## Claim Atomically

`start-work` / `start-next-work` (CLI) and `work_start` / `work_start_next`
(MCP) claim the issue *and* transition it into its working status in one DB
transaction, with optimistic locking on the assignee. Never chain a claim with
a separate status update — that two-step form races other agents. If someone
else already owns it, the call fails with `code: CONFLICT` (CLI exit 4); retry
against different work.

> **Ready ≠ startable.** The working status is type-specific (tasks →
> `in_progress`, features → `building`). Bugs start at `triage`, which has no
> single-hop transition into work — they walk `triage → confirmed → fixing`. So
> a triage bug is *ready* but not directly *startable*: `start-work` on one
> returns `INVALID_TRANSITION` naming the next status, and `start-next-work`
> skips it. `ready` items carry a `startable` flag (and a `next_action` hint
> when false). Pass `--advance` (CLI) / `advance=true` (MCP) to walk the soft
> transitions automatically.

## Guidance Sheets

Load these when facing a specific challenge rather than reading upfront:

- **`references/commands.md`** — the commands and MCP verbs reached for most,
  the P0–P4 priority scale, `--actor` attribution, file records and scan
  findings, cross-product entity bindings, quick decision guide
- **`references/workflow-patterns.md`** — triage flows, sprint planning,
  dependency management, bug lifecycle, event history, archiving
- **`references/team-coordination.md`** — multi-agent swarm protocols, handoff
  conventions, claiming strategies, status update patterns
- **`references/observations.md`** — the ambient-note scratchpad: when a note
  is right, when the thing is task scope instead, triage and promotion
- **`references/error-codes.md`** — response envelopes, the complete
  `ErrorCode` set, and what to do about `INVALID_TRANSITION`,
  `SCHEMA_MISMATCH` and `CONFLICT`
- **`examples/sprint-plan.json`** — complete create-plan input template with
  cross-phase dependencies
