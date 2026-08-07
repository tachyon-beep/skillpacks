# Commands, Priorities and Records

The command catalogue is discoverable: `filigree --help` and `filigree <verb>
--help` are authoritative for the CLI, and the MCP tool schemas are
authoritative for `mcp__filigree__*`. You do not need to memorise either. This
sheet collects the parts an agent reaches for most, plus the project enums.

Project data lives in the machine-owned store dir, resolved in this order: a
`weft.toml` `[filigree].store_dir` override (project-relative only), then
`.weft/filigree/`, then the legacy `.filigree/`.

## Priority Semantics

| Priority | Meaning | Action |
|----------|---------|--------|
| P0 | Critical | Drop everything. Production is broken. |
| P1 | High | Do next. Current sprint must-have. |
| P2 | Medium | Default. Normal backlog work. |
| P3 | Low | Nice to have. Do when P1/P2 are clear. |
| P4 | Backlog | Someday. Don't schedule unless promoted. |

When triaging, use `filigree batch-update <ids...> --priority=N` for bulk changes.

## Finding Work

```bash
filigree ready                    # ready issues sorted by priority
filigree list --status=open       # all open issues
filigree search "auth"            # full-text search
filigree critical-path            # longest dependency chain
```

## Creating Issues

```bash
filigree create "Title" --type=bug --priority=1
filigree create "Title" --type=task -d "description" --dep <blocker-id>
filigree create-plan --file plan.json   # milestone/phase/step hierarchy
```

## Managing Dependencies

```bash
filigree add-dep <issue> <depends-on>     # A depends on B
filigree remove-dep <issue> <depends-on>
filigree blocked                          # show all blocked issues
```

## Context and Handoff

```bash
filigree add-comment <id> "what I found / what's left to do"
filigree get-comments <id>                # read previous context
filigree show <id>                        # full details including deps
```

Always add a comment before closing or handing off — the next agent has no memory
of the current conversation.

## Claiming Without Transitioning

`claim` and `claim-next` still exist for the rare case where you want to
reserve an issue but not advance its status (e.g. a coordinator earmarking
work for a worker that will pick it up later). Prefer `start-work` /
`start-next-work` for normal flow — see `references/team-coordination.md`.

```bash
filigree claim <issue-id> --assignee <agent-name>     # reserve only, no transition
filigree claim-next --assignee <agent-name>
```

## Health and Diagnostics

```bash
filigree doctor           # check installation health
filigree stats            # project-wide counts
filigree metrics          # cycle time, lead time, throughput
filigree events <id>      # audit trail for a specific issue
```

## The MCP verbs you will reach for most

- **Find work:** `work_ready`, `work_blocked`, `issue_list`, `issue_search`
- **Claim work:** `work_start`, `work_start_next`
- **Update:** `comment_add`, `label_add`, `issue_update`, `issue_close`
- **Admin (irreversible):** `issue_delete` (MCP) / `delete-issue` (CLI) —
  hard-deletes a terminal issue and its rows; `admin_undo_last` cannot reverse it.
- **Scratchpad:** `observation_create`, `observation_list`, `observation_promote`,
  `observation_dismiss` — see `references/observations.md`
- **Health:** `stats_get`, `metrics_get`, `mcp_status_get`

Pass `--actor <name>` (CLI) so events attribute to your agent identity. It
works in either position — before the verb (`filigree --actor X update …`) or
after it (`filigree update … --actor X`); the post-verb value overrides the
group-level one.

## Cross-Product Entity Bindings (ADR-029)

`entity_association_add`, `entity_association_remove`,
`entity_association_list`, `entity_association_list_by_entity`. Used when a
sibling tool (e.g. Loomweave) needs to bind a Filigree issue to a function,
class, or module identifier it owns. The `entity_id` is an opaque external
string from Filigree's perspective and may be a `loomweave:eid:...` SEI or a
legacy locator; callers may also supply `entity_kind` explicitly. The consumer
(the sibling tool's read path) does drift detection against the stored
`content_hash_at_attach`. `entity_association_list_by_entity` is the
reverse-lookup surface — given an opaque external entity ID, return every
Filigree issue bound to it (project isolation is by DB file). Also reachable
over HTTP as `GET/POST /api/issue/{issue_id}/entity-associations`,
`DELETE /api/issue/{issue_id}/entity-associations?entity_id=…`, and
`GET /api/entity-associations?entity_id=…`.

## File Records and Scan Findings

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
| "I noticed a gap in the work I'm currently doing" | Fix it, expand the task, or file a proper issue — **do not** record it as an observation |
| "These observations are piling up" | `observation_list`, then dismiss or promote each |
