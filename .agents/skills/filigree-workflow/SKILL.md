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

Filigree is an agent-native issue tracker that stores data locally in `.filigree/`.
This skill provides procedural knowledge for using filigree effectively — as a solo
agent or in a multi-agent swarm.

## Core Workflow

Every task follows this lifecycle:

```
filigree ready                                      → find available work (no blockers)
filigree show <issue-id>                            → read requirements and context
filigree transitions <issue-id>                     → check valid status transitions
filigree start-work <issue-id> --assignee <name>    → atomically claim + transition into its working status
[do the work, commit code]
filigree close <issue-id> --reason="summary of what was done"
```

Or skip steps 1–3 entirely with `filigree start-next-work --assignee <name>` to grab the highest-priority **startable** issue.

> **Ready ≠ startable.** The working status is type-specific (tasks →
> `in_progress`, features → `building`). Bugs start at `triage`, which has no
> single-hop transition into work — they walk `triage → confirmed → fixing`. So
> a triage bug is *ready* but not directly *startable*: `start-work` on one
> returns `INVALID_TRANSITION` naming the next status to move through, and
> `start-next-work` skips it. `ready` items carry a `startable` flag (and a
> `next_action` hint when false). Pass `--advance` to either command to walk the
> soft transitions automatically (`triage → confirmed → fixing`) instead of
> being blocked or skipped.

Always close with a `--reason` — it becomes audit trail for the next agent.

## Priority Semantics

| Priority | Meaning | Action |
|----------|---------|--------|
| P0 | Critical | Drop everything. Production is broken. |
| P1 | High | Do next. Current sprint must-have. |
| P2 | Medium | Default. Normal backlog work. |
| P3 | Low | Nice to have. Do when P1/P2 are clear. |
| P4 | Backlog | Someday. Don't schedule unless promoted. |

When triaging, use `filigree batch-update <ids...> --priority=N` for bulk changes.

## Starting Work

### Solo or Swarm — Same Tool

Use `start-work` (or `start-next-work`) for the usual case. Both atomically
claim the issue *and* transition it into its working status in one DB
transaction — optimistic-locking on the assignee, so concurrent callers can't
both think they own the issue. The working status is type-specific (tasks →
`in_progress`, features → `building`, bugs → `fixing`).

```bash
filigree start-work <issue-id> --assignee <agent-name>              # specific issue
filigree start-next-work --assignee <agent-name>                    # highest-priority startable
filigree start-work <bug-id> --assignee <agent-name> --advance      # walk triage → confirmed → fixing
```

If another agent already owns the claim, the call fails with `code: CONFLICT`
(CLI exit 4). Safe to retry against a different issue.

`start-work` on a `triage` bug (or any type with no single-hop working status)
returns `INVALID_TRANSITION` naming the intermediate status to move through
first; `start-next-work` skips such issues. Pass `--advance` to walk the soft
transitions to the nearest working status automatically (missing required
fields become warnings, not blocks; hard edges are never auto-walked).

### Niche: Claim Without Transitioning

`claim` and `claim-next` still exist for the rare case where you want to
reserve an issue but not advance its status (e.g. a coordinator earmarking
work for a worker that will pick it up later). Prefer `start-work` for
normal flow.

```bash
filigree claim <issue-id> --assignee <agent-name>     # reserve only, no transition
filigree claim-next --assignee <agent-name>
```

## Key Commands

### Finding Work

```bash
filigree ready                    # ready issues sorted by priority
filigree list --status=open       # all open issues
filigree search "auth"            # full-text search
filigree critical-path            # longest dependency chain
```

### Creating Issues

```bash
filigree create "Title" --type=bug --priority=1
filigree create "Title" --type=task -d "description" --dep <blocker-id>
filigree create-plan --file plan.json   # milestone/phase/step hierarchy
```

### Managing Dependencies

```bash
filigree add-dep <issue> <depends-on>     # A depends on B
filigree remove-dep <issue> <depends-on>
filigree blocked                          # show all blocked issues
```

### Context and Handoff

```bash
filigree add-comment <id> "what I found / what's left to do"
filigree get-comments <id>                # read previous context
filigree show <id>                        # full details including deps
```

Always add a comment before closing or handing off — the next agent has no memory
of the current conversation.

## Workflow Patterns

### Before Starting Work

1. Run `filigree ready` to see available work
2. Check `filigree critical-path` — unblocking the critical path has highest leverage
3. Pick work that matches the current session's context (e.g., if code is already open)

### When Finishing Work

1. Add a comment summarising what was done and any follow-up needed
2. Close with a reason: `filigree close <id> --reason="implemented X, tested Y"`
3. Check if closing this issue unblocks anything: `filigree ready`

### When Blocked

1. Add a comment explaining the blocker
2. Create the blocking issue if it doesn't exist
3. Add the dependency: `filigree add-dep <blocked> <blocker>`
4. Move to other available work

## Guidance Sheets

For detailed patterns, consult these reference files:

- **`references/workflow-patterns.md`** — Triage flows, sprint planning,
  dependency management, bug lifecycle patterns
- **`references/team-coordination.md`** — Multi-agent swarm protocols,
  handoff conventions, claiming strategies, status update patterns
- **`examples/sprint-plan.json`** — Complete create-plan input template
  with cross-phase dependencies

Load these when facing a specific workflow challenge rather than reading upfront.

## File Records & Scan Findings

The dashboard API tracks files and scan findings across the project. Use the
schema discovery endpoint to find valid values and available endpoints:

```
GET /api/files/_schema
```

This returns valid severities, finding statuses, association types, sort fields,
and a full endpoint catalog. When linking issues to files, use file associations:

| Association Type | Meaning |
|-----------------|---------|
| `bug_in` | Bug reported in this file |
| `task_for` | Task related to this file |
| `scan_finding` | Automated scan finding |
| `mentioned_in` | File referenced in issue |

## Response Shapes (2.0)

When parsing `--json` output or MCP responses, expect these unified envelopes:

- **Batch ops** → `{succeeded: [...], failed: [{id, error, code}, ...], newly_unblocked?: [...]}`.
  `failed` is always present (empty list if none); `newly_unblocked` is
  present only when non-empty (omitted when the op unblocked nothing). Pass `--detail=full` (CLI) or
  `response_detail="full"` (MCP) to get full records back.
- **List ops** → `{items: [...], has_more: bool, next_offset?: int}`.
  `next_offset` only appears when there is a next page.
- **Errors** → `{error: str, code: ErrorCode, details?: dict}`. `code` is
  one of: `VALIDATION`, `NOT_FOUND`, `CONFLICT`, `INVALID_TRANSITION`,
  `PERMISSION`, `NOT_INITIALIZED`, `IO`, `INVALID_API_URL`,
  `FILE_REGISTRY_DISPLACED`, `REGISTRY_UNAVAILABLE`,
  `LOOMWEAVE_REGISTRY_VERSION_MISMATCH`, `LOOMWEAVE_OUT_OF_SYNC`,
  `BRIEFING_BLOCKED`, `STOP_FAILED`, `SCHEMA_MISMATCH`, `INTERNAL`.
  Branch on `code` for retry policy
  (`CONFLICT` → exit 4, retryable; everything at exit 1 needs operator
  intervention).

The issue ID is always `issue_id` in 2.0 — in MCP inputs, response payloads,
and CLI JSON. Status is always `status`; "state" was retired as a
user-facing word.

## Health and Diagnostics

```bash
filigree doctor           # check installation health
filigree stats            # project-wide counts
filigree metrics          # cycle time, lead time, throughput
filigree events <id>      # audit trail for a specific issue
```

## Observations — Ambient Note-Taking

Observations are a scratchpad for things you notice *while doing other work*. They
are not issues — they're lightweight, expiring notes that let you capture a thought
without breaking flow.

### When to Observe

Observations are for **incidental** defects — things you notice *in passing*
while working on something else, that fall *outside the scope of your current
task*. The core use case is: "I don't have time to investigate this right now,
but I want to come back to it."

Examples of good observations:

- A code smell in a neighbouring file you happened to read
- A missing test for an edge case unrelated to what you're changing
- A potential bug in a module you're not touching
- A TODO or FIXME that looks stale
- A dependency that might be outdated

**Always include `file_path` and `line`** when the observation is about specific code.
This anchors it for whoever triages it later.

### When NOT to Observe

**You fix bugs in your currently defined scope. You do NOT use observations to
finish work prematurely.**

If you're working on task X and you notice that your implementation of X has a
gap, a missed edge case, an untested branch, a known shortcoming, or a piece of
follow-up that "should really be done too" — that is **task scope, not an
observation**. You own it. Handle it one of these ways instead:

- **Fix it now** as part of the current task. (Default.)
- **Expand the task** (or split a sub-task) and address it in this work stream.
- **File a proper issue** with a dependency on the current task, so the gap is
  visible in the work record before you close.
- **Surface it to the user** if it changes the shape of what you're delivering.

Filing your own task's deficiencies as observations and closing the task is
**not** completing the task. It is shipping known-broken work and hiding the
debt in a 14-day expiring scratchpad — where it will quietly rot, get
auto-dismissed, and never be addressed. The work record must reflect what is
actually outstanding.

**The test:** *"Would I have noticed this even if I weren't working on this
task?"* If yes → observation. If no → it's part of the work, fix it.

**Don't observe things that are clearly issues either.** If you're confident
something is a bug or a needed feature, create an issue directly. Observations
are for "hmm, this might be worth looking at" — the uncertain middle ground.

### Triage Workflow

Observations expire after 14 days. Triage them before they rot:

1. **At session end:** run `observation_list` and quickly scan what's accumulated
2. **For each observation, decide:**
   - **Dismiss** — not actionable, already fixed, or not worth tracking. Use
     `observation_dismiss` with a brief reason for the audit trail.
   - **Promote** — deserves to be tracked as an issue. Use `observation_promote`
     which atomically creates an issue and labels it `from-observation`. Choose
     the right issue type:
     - `type='bug'` — something is broken or produces wrong results
     - `type='task'` (default) — cleanup, improvement, or "this works but is shitty"
     - `type='feature'` — a missing capability that should exist
     - `type='requirement'` — a formal requirement to be reviewed, approved, and verified, when the requirements pack is enabled
   - **Leave it** — still uncertain. Let it age. If it survives a few sessions
     without being promoted, it's probably a dismiss.

3. **Batch cleanup:** use the MCP tool `observation_batch_dismiss` when several observations
   have gone stale together.

### Promote vs Dismiss

| Signal | Action |
|--------|--------|
| You noticed it twice in separate sessions | Promote |
| It's in a hot code path or critical module | Promote |
| It has a clear fix or next step | Promote |
| It was about code that's since been refactored | Dismiss |
| It's a style/taste preference, not a defect | Dismiss |
| You can't articulate what the fix would be | Leave it (or dismiss if > 7 days old) |

### Tracking the Pipeline

Promoted observations get the `from-observation` label. To see the pipeline output:

```bash
filigree list --label=from-observation     # All promoted observations
filigree search "from-observation"         # Search with context
```

## Quick Decision Guide

| Situation | Action |
|-----------|--------|
| "What should I work on?" | `filigree ready`, pick highest priority |
| "Is this blocked?" | `filigree show <id>`, check blocked_by |
| "Multiple agents need work" | `filigree start-next-work --assignee <name>` |
| "I found a new bug" | `filigree create "..." --type=bug --priority=1` |
| "This task is bigger than expected" | Create sub-tasks, add deps |
| "I'm done" | Comment, close with reason, check `ready` |
| "Something changed while I worked" | `filigree changes --since <timestamp>` |
| "I noticed something odd in a file I'm passing through" | `observation_create` with file_path and line — keep working |
| "I noticed a gap in the work I'm currently doing" | Fix it, expand the task, or file a proper issue — **do not** observe it |
| "These observations are piling up" | `observation_list`, then dismiss or promote each |
