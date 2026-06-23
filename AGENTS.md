<!-- filigree:instructions:v3.0.1:65e6fb25 -->
<!-- filigree:last-writer:filigree install -->
## Filigree Issue Tracker

`filigree` tracks tasks for this project. Data lives in `.filigree/`. Prefer
the MCP tools (`mcp__filigree__*`) when available; fall back to the `filigree`
CLI otherwise.

### Workflow

```bash
# At session start
filigree session-context                            # ready / in-progress / critical path

# Pick up the next startable issue (atomic claim + transition into its working status)
filigree start-next-work --assignee <name>
# ...or claim a specific issue
filigree start-work <id> --assignee <name>

# Do the work, commit, then
filigree close <id>
```

Use the atomic claim+transition verbs — `work_start` / `work_start_next`
(MCP) or `start-work` / `start-next-work` (CLI). Do **not** chain
`work_claim` (MCP) or `filigree claim` (CLI) with a subsequent status
update — the two-step form races against other agents; the combined verb is
atomic.

**Ready ≠ startable.** The working status is type-specific (tasks →
`in_progress`, features → `building`). Bugs start at `triage`, which has no
single-hop transition into work (`triage → confirmed → fixing`), so a triage
bug is *ready* but not directly *startable*: `work_start` on one returns
`INVALID_TRANSITION` naming the next status, and `work_start_next` skips it.
`work_ready` items carry a `startable` flag (plus a `next_action` hint when
false). Pass `advance=true` (MCP) / `--advance` (CLI) to walk the soft
transitions to the nearest working status automatically.

### Observations: when (and when not) to use them

`observation_create` is a fire-and-forget scratchpad for *incidental* defects — things
you notice *outside the scope of your current task* (a code smell in a
neighbouring file, a stale TODO, a missing test for an edge case you happened
to spot). Notes expire after 14 days unless promoted. Include `file_path` and
`line` when relevant. At session end, skim `observation_list` and either
`observation_dismiss` or `observation_promote` for what has accumulated.

**You fix bugs in your currently defined scope. You do NOT use observations
to finish work prematurely.** If a defect, gap, or follow-up belongs to your
current task, you own it — handle it as part of that task: fix it now, expand
the task's scope, file a proper issue with a dependency, or surface it to the
user. Filing it as an observation and closing the task is *not* completing
the task; it is shipping known-broken work and hiding the debt in a 14-day
expiring scratchpad. The test is "would I have noticed this even if I weren't
working on this task?" If no, it's task scope, not an observation.

### Priority scale

- P0: Critical (drop everything)
- P1: High (do next)
- P2: Medium (default)
- P3: Low
- P4: Backlog

### Reaching for tools

MCP tool schemas describe each tool; `filigree --help` and `filigree <verb>
--help` are the authoritative CLI reference. You do not need to memorise
either catalogue. The verbs you will reach for most:

- **Find work:** `work_ready`, `work_blocked`, `issue_list`, `issue_search`
- **Claim work:** `work_start`, `work_start_next`
- **Update:** `comment_add`, `label_add`, `issue_update`, `issue_close`
- **Admin (irreversible):** `issue_delete` (MCP) / `delete-issue` (CLI) —
  hard-deletes a terminal issue and its rows; `admin_undo_last` cannot reverse it.
- **Scratchpad:** `observation_create`, `observation_list`, `observation_promote`, `observation_dismiss`
- **Cross-product entity bindings (ADR-029):** `entity_association_add`,
  `entity_association_remove`, `entity_association_list`,
  `entity_association_list_by_entity`. Used when a sibling tool (e.g.
  Loomweave) needs to bind a Filigree issue to a function, class, or
  module identifier it owns. The `entity_id` is an opaque external string
  from Filigree's perspective and may be a `loomweave:eid:...` SEI or a legacy
  locator; callers may also supply `entity_kind` explicitly. The consumer (the sibling tool's read
  path) does drift detection against the stored
  `content_hash_at_attach`. `entity_association_list_by_entity` is the
  reverse-lookup surface — given an opaque external entity ID, return every
  Filigree issue bound to it (project isolation is by DB file). Also
  reachable over HTTP as
  `GET/POST /api/issue/{issue_id}/entity-associations`,
  `DELETE /api/issue/{issue_id}/entity-associations?entity_id=…`,
  and `GET /api/entity-associations?entity_id=…`.
- **Health:** `stats_get`, `metrics_get`, `mcp_status_get`

Pass `--actor <name>` (CLI) so events attribute to your agent identity. It
works in either position — before the verb (`filigree --actor X update …`) or
after it (`filigree update … --actor X`); the post-verb value overrides the
group-level one.

### Error handling

Errors return `{error: str, code: ErrorCode, details?: dict}`. Switch on
`code`, not on message text. Codes: `VALIDATION`, `NOT_FOUND`, `CONFLICT`,
`INVALID_TRANSITION`, `PERMISSION`, `NOT_INITIALIZED`, `IO`,
`INVALID_API_URL`, `FILE_REGISTRY_DISPLACED`, `REGISTRY_UNAVAILABLE`,
`LOOMWEAVE_REGISTRY_VERSION_MISMATCH`, `LOOMWEAVE_OUT_OF_SYNC`,
`BRIEFING_BLOCKED`, `STOP_FAILED`, `SCHEMA_MISMATCH`, `INTERNAL`.

On `INVALID_TRANSITION`, call `workflow_transition_list` (MCP) or
`filigree transitions <id>` to see what the workflow allows from here.

Two failure modes deserve a specific response:

- **`SCHEMA_MISMATCH`** — the installed `filigree` is older than the project
  database. The error message contains upgrade guidance. Surface it to the
  user; do not retry.
- **`ForeignDatabaseError`** — filigree found a parent project's database
  but no local `.filigree.conf`. Run `filigree init` in the current
  directory. Do **not** `cd` upward to a different project unless that was
  the actual intent.
<!-- /filigree:instructions -->
