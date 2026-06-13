---
name: output-shape-and-pagination
description: Use when a tool's return shape grows with the database, when an agent runs out of context after one tool call, when a list tool dumps every row, when responses are unbounded blobs of JSON, when there is no cursor and no page size, when you cannot tell whether a result was truncated, when you are deciding between summary fields and full detail, when structured output (structuredContent / outputSchema) is missing or unvalidated, or when you need to choose what a tool returns versus what it links to as a resource.
---

# Output Shape and Pagination

## Overview

**Every tool result is spent context. A return shape that grows with the database is a context-budget bomb with a fuse length equal to the size of the smallest production dataset the agent will ever hit. The discipline is: bound the result by construction, or paginate it with a cursor, or declare it truncatable and hand the agent a way to get the rest — and never let the size of the answer depend on the size of the data without a stated profile.**

A human reading an API response scrolls past the parts they do not need. An agent cannot. Every byte a tool returns is injected into the model's context window on the turn it returns, is re-tokenized, costs money, and — past a threshold the agent cannot see coming — silently breaks the conversation by evicting the instructions, the prior tool results, or the user's actual request. The 200-row `list_issues` response that was fine in dev with 12 issues is a 40,000-token wall in production with 4,000 issues, and the failure is not an error the agent can recover from. It is a degraded conversation with no signal.

This sheet is the third of the architect cluster (after `tool-api-design.md` and `mcp-primitive-selection.md`). It owns: JSON return discipline, the summary-versus-detail split, cursor-based pagination, the truncation-policy contract, and `structuredContent` with a declared `outputSchema` (MCP 2025-11-25). It upholds the consistency-gate clause that **every return shape has a context-budget profile**: *bounded*, *paginated*, or *may-be-truncated*. A tool that cannot name its profile has not been designed; it has been allowed to happen.

Two readings of the same material. The **architect** uses this sheet to choose a return shape before shipping a tool. The **critic** uses it to find the tool whose return shape will blow the budget on the median input, and to grade that finding with severity and evidence. They share the rules; they do not share epistemics. The architect's blind spot is the dev-dataset that never grows; the critic's job is to ask "what does this return at 100×?".

## When to Use

Use this sheet when:

- A list/search/query tool returns "all matching" anything with no page size and no cursor.
- The agent runs out of context, gets confused, or drops earlier instructions right after a single tool call returned a large payload.
- A tool's response JSON nests full records where a reference or a summary would do.
- You cannot tell, from a result alone, whether it is complete or was cut off.
- You are deciding what a tool *returns* versus what it *links to* (resource link) versus what the agent must *fetch with a follow-up call*.
- You are adding `structuredContent` / `outputSchema` to a tool, or auditing a tool that returns a JSON string inside a text block instead.
- A return field's size is a function of the database (a `description`, a `body`, a `logs` array, an embeddings vector, a base64 blob).

Do not use this sheet for:

- Choosing *whether* state is a tool, a resource, or a prompt — that is `mcp-primitive-selection.md`. This sheet assumes the tool decision is made and governs what it returns.
- Error payload shape and recovery hints — that is `error-envelopes-and-recovery.md`. A truncation that surfaces as an error is in scope there; a truncation that is a normal-result property is in scope here.
- Retry-driven duplication of side effects — that is `idempotency-and-atomicity.md`. Pagination cursors are read-side; a cursor is not a claim lease.

## Core Principle

> A tool result is spent context, not a return value. Size it for the largest realistic input, not the dev fixture. If the size depends on the data, the tool MUST be paginated or MUST declare itself truncatable and provide a path to the rest. "It returns the matching rows" is not a return shape; "it returns the first 50 matching rows as id+title+status, with a `nextCursor` when more exist" is.

## The Context-Budget Profile (the load-bearing artifact)

Every shipping tool declares exactly one profile. This is the artifact the consistency gate checks. Write it into the tool's intent block; the critic reads it back against reality.

| Profile | Meaning | When it applies | What the agent can assume |
|---|---|---|---|
| **bounded** | Return size has a constant upper bound, independent of database size | Single-record fetch, status check, count, a fixed small set | The whole answer is here; no follow-up needed |
| **paginated** | Returns one page plus a `nextCursor`; absence of cursor means done | Any list/search whose result count grows with data | More may exist; loop on cursor or stop early on enough |
| **may-be-truncated** | Returns a best-effort prefix and a flag/marker plus a way to get the rest (a detail tool, a resource link, a narrower query) | A single record whose *contents* can be large (a long body, a big log), or a result where pagination does not fit | The answer may be incomplete; the truncation is signalled and recoverable |

The forbidden fourth state is **unbounded** — a return whose size is `O(rows)` or `O(content-length)` with no cap, no cursor, and no truncation signal. Every unbounded return is a `return-shape-that-blows-budget` smell (sheet 13) and a `major`-or-worse finding. The median-input incident is not a question of *if*.

A concrete budget to design against: a host typically reserves only a fraction of the window for tool results, and a single tool result over roughly **25k tokens** (~100KB of JSON) is already large enough that many hosts will truncate it, summarize it, or refuse it. Design list results to land under a few thousand tokens per page. Treat "it fits today" as the start of the bug report, not the end of the design.

## JSON Return Discipline

Six rules for the bytes themselves. Each one is a token-cost lever.

1. **Return identifiers and summaries in lists; return full detail only on single-record fetch.** A list of 50 issues should be 50 × `{id, title, status}`, not 50 × the full issue with body, comments, labels, and history. The agent that wants detail calls `get_issue(id)`. This is the single biggest budget lever; a list of summaries is often 10–30× smaller than a list of full records.
2. **No redundant envelope nesting.** `{"result": {"data": {"items": [...]}}}` spends tokens on structure the agent must parse past. Return `{"issues": [...], "nextCursor": ...}`. One level of meaningful naming, not three of ceremony.
3. **Omit nulls and defaults; do not serialize empty fields.** `{"id": 7, "title": "x"}` not `{"id": 7, "title": "x", "assignee": null, "due": null, "labels": [], "parent": null}`. Every null is tokens that teach the agent nothing.
4. **No raw blobs in the result.** Base64 images, embedding vectors, full file contents, multi-megabyte logs — these go behind a resource link (a URI the host can fetch on demand) or a detail tool, never inline. A 1,536-float embedding inlined as JSON is ~10k tokens of digits the model cannot use.
5. **Stable, shallow field names the agent can rely on.** `status: "in_progress"` beats `s: 2` (the agent has no enum table) and beats `workflow_state_machine.current_node.label` (depth costs tokens and fragility). Renaming a field is a non-backward-compatible change — see `schema-versioning-and-drift.md`.
6. **Numbers and enums over prose where the agent will branch on them.** If the agent decides based on a value, give it a value, not a sentence. Reserve prose for fields the agent will relay to the user.

## Summary vs Detail: the two-tier rule

The default surface for anything list-shaped is two tiers:

- **A list/search tool** that returns *summaries* — the minimum fields an agent needs to (a) decide which records it cares about and (b) call the detail tool. Paginated. Bounded per page.
- **A detail tool** that returns *one record* in full. Bounded (single record) or may-be-truncated (if the record's own contents can be large, e.g. a long body, in which case point #4 applies to the big sub-fields).

This split is what keeps the list profile *paginated* and the detail profile *bounded*. Collapsing them — a list tool that returns full records — is the most common path to an unbounded return. The smell name is `tool-as-CRUD-mirror` joined with `return-shape-that-blows-budget`: the surface mirrors `SELECT *` instead of serving the agent's actual decision.

Counter-rule: do not over-summarize. If the agent will *always* immediately fetch the full record for every list item, the summary tier is pure overhead — two round-trips for one decision. The test is: can the agent make the *select-which-ones* decision from the summary alone? If yes, summarize. If the agent needs the body to choose, the body belongs in the summary (and then the list must be aggressively paginated and the body itself truncated).

## Cursor-Based Pagination

MCP's own list operations (`tools/list`, `resources/list`, `prompts/list`) use opaque-cursor pagination, and your tools should mirror it. The contract:

- A request MAY include a `cursor` (opaque string) and SHOULD support a page-size hint or use a server-chosen page size.
- A response includes the page of results plus an optional `nextCursor`. **Presence of `nextCursor` means more results exist; absence means this is the last page.** This is the completeness signal — without it the agent cannot distinguish "done" from "cut off".
- The cursor is **opaque to the agent**: it is a base64-or-similar token the server defines, not a page number or row offset the agent constructs. The agent echoes it back unmodified. This lets the server change its pagination strategy (offset → keyset → snapshot) without a schema change.

Why cursors, not offsets, on the server side: offset pagination (`LIMIT 50 OFFSET 100`) skips and duplicates rows when the underlying set changes between calls — and between two agent turns the set *will* change, because other agents and humans are writing to the same database. Keyset/seek pagination (`WHERE id > :last_seen ORDER BY id LIMIT 50`) is stable under concurrent inserts and is what the cursor should encode. Encode the sort key and direction in the cursor, not the page number.

Page-size discipline: pick a server-side default that keeps a page under the budget (often 25–100 summary records), enforce a server-side **maximum** (an agent that asks for `pageSize: 100000` must get the capped size, not the literal request), and document both. The agent does not know your row sizes; the server owns the budget.

## Truncation Policy

When a single value can be large and pagination does not fit (one record with a 200KB body), the `may-be-truncated` profile applies. The policy has three required parts:

1. **Truncate deterministically and visibly.** Cut at a byte/char/token boundary you control, append an explicit marker, and set a boolean. `{"body": "...first 4000 chars...", "body_truncated": true, "body_full_length": 214003}`. Silent truncation is the worst case — the agent reasons over a partial value it believes is complete.
2. **Provide the path to the rest.** A resource link (`resource_link` to a URI the host fetches on demand), a dedicated `get_issue_body(id, offset, length)` tool, or a narrower query. "Truncated, good luck" is not a policy.
3. **Truncate the *largest sub-fields first*, never the structural fields.** Keep `id`, `status`, `title`; truncate `body`, `logs`, `diff`. The agent needs the skeleton intact to know what it is looking at and how to fetch more.

Truncation is a normal-result property here, not an error. (A request that *cannot* be served within budget at all — e.g. "return all 4M rows" — is a `retry-with-changes` error directing the agent to paginate; that envelope lives in `error-envelopes-and-recovery.md`.)

## Structured Output: `structuredContent` + `outputSchema` (2025-11-25)

As of revision 2025-11-25, a tool MAY declare an `outputSchema` (a JSON Schema, 2020-12 dialect) and return `structuredContent` — a JSON object the host validates against that schema — alongside (or instead of) the unstructured `content` blocks. Use it, and use it correctly:

- **Declare `outputSchema` for any tool whose result the agent will parse into fields.** The schema is a contract: it tells the host (and the agent) the shape to expect, lets the host validate, and lets the agent rely on field presence. A tool that returns a JSON *string* stuffed inside a text block is the anti-pattern this replaces — the agent has to guess at the shape and re-parse a string every turn.
- **Validate against the schema before returning.** A `structuredContent` that does not match its declared `outputSchema` is a server bug; an input-validation failure on the *result* side is a contract violation, not an agent error. (Input-side validation failures are returned as Tool Execution Errors so the model can self-correct — that is the input contract; the output contract is the server's to honor.)
- **For backward-compatibility, also include a text representation in `content`** unless every host you target is known to consume `structuredContent`. Hosts that predate 2025-11-25 ignore `structuredContent`; the text block is the fallback. The two MUST agree.
- **`outputSchema` does not exempt you from the budget profile.** A schema-valid 50,000-row array is still an unbounded return. Structure and size are orthogonal: `structuredContent` makes the shape parseable; pagination and truncation make the size safe. You need both.
- **Changing `outputSchema` in a non-backward-compatible way (removing a field, narrowing a type, making an optional field required) bumps server capability** — see `schema-versioning-and-drift.md`. The output schema is part of the surface the agent depends on.

## Example 1 — `list_issues`: from unbounded dump to paginated summary with declared output schema

A tracker MCP server. The first version (the architect's "clean" first draft) returns every matching issue in full.

```python
# BEFORE — unbounded, profile-less, blows the budget at scale.
# Smell: return-shape-that-blows-budget + tool-as-CRUD-mirror.
@server.tool()
async def list_issues(status: str | None = None) -> list[dict]:
    rows = db.query(
        "SELECT * FROM issues WHERE (:status IS NULL OR status = :status)",
        status=status,
    )
    return [dict(r) for r in rows]   # every column, every row, every body, forever
```

In dev (12 issues) this is fine. In production (4,200 open issues, each with a multi-paragraph `body`) the first call returns ~60,000 tokens and the conversation is dead. There is no cursor, so even if the agent wanted to page it cannot; there is no completeness signal, so the agent cannot tell it got everything.

```python
# AFTER — paginated summary tier. Profile: PAGINATED. Bounded per page.
# Cursor is opaque + keyset-encoded so it survives concurrent inserts.
import base64, json

PAGE_DEFAULT, PAGE_MAX = 50, 200

def _encode_cursor(last_id: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"after_id": last_id}).encode()).decode()

def _decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    return json.loads(base64.urlsafe_b64decode(cursor))["after_id"]

@server.tool(
    # Agent-voice intent (effect, not implementation):
    # "Find issues to work on or reason about. Returns a compact page of issue
    #  summaries (id, title, status); call get_issue(id) for the full record.
    #  When nextCursor is present, more issues match — pass it back to continue."
    output_schema={                       # 2025-11-25 outputSchema, JSON Schema 2020-12
        "type": "object",
        "required": ["issues"],
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "title", "status"],
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "string"},
                        "status": {"type": "string",
                                   "enum": ["open", "in_progress", "blocked", "closed"]},
                    },
                    "additionalProperties": False,
                },
            },
            "nextCursor": {"type": "string",
                           "description": "Present iff more results exist."},
        },
        "additionalProperties": False,
    },
)
async def list_issues(status: str | None = None,
                      cursor: str | None = None,
                      page_size: int = PAGE_DEFAULT) -> dict:
    size = min(max(page_size, 1), PAGE_MAX)        # server owns the budget; cap the ask
    after_id = _decode_cursor(cursor)
    rows = db.query(
        """SELECT id, title, status FROM issues
           WHERE (:status IS NULL OR status = :status) AND id > :after_id
           ORDER BY id ASC LIMIT :limit""",        # keyset, not OFFSET: stable under inserts
        status=status, after_id=after_id, limit=size + 1,   # +1 to peek for more
    )
    has_more = len(rows) > size
    page = rows[:size]
    result = {"issues": [{"id": r.id, "title": r.title, "status": r.status} for r in page]}
    if has_more:
        result["nextCursor"] = _encode_cursor(page[-1].id)  # absence == last page
    return result   # structuredContent the host validates; ~30 tokens/issue, ~50/page
```

What changed, against the gate: the profile is now **paginated** and stated in the intent; the page is **bounded** (≤ 200 summaries); the cursor is **opaque and keyset-encoded** so it does not skip or duplicate rows when another agent files an issue between turns; `nextCursor` presence is the **completeness signal**; the summary tier drops `body`/`comments`/`history` so per-record cost falls ~20×; `output_schema` makes the shape a **validated contract**. A critic auditing the BEFORE version files: *blocker — `list_issues` is unbounded (`SELECT *`, no cursor); evidence: returns `O(rows)` full records, ~60k tokens at 4.2k open issues; no completeness signal; recommend summary tier + keyset cursor.*

## Example 2 — `get_run_logs`: the `may-be-truncated` profile with a resource-link path to the rest

A CI/runner MCP server. `get_run_logs(run_id)` returns a build's logs. A single run's log can be 50 bytes or 200MB; pagination by "row" is awkward and the agent usually wants the *tail* (where the failure is), not the whole thing.

```python
# Profile: MAY-BE-TRUNCATED. Returns a bounded tail by default, signals truncation
# explicitly, and hands back a resource link for the full log on demand.
TAIL_CHARS = 8_000   # bounded prefix; tune to keep result well under budget

@server.tool(
    # Agent-voice intent:
    # "Read the logs for a finished run to diagnose a failure. Returns the last
    #  ~8k characters by default (where failures usually are). If the log is longer,
    #  log_truncated is true and full_log_resource links the complete log — the host
    #  can fetch it on demand instead of spending your context on it now."
    output_schema={
        "type": "object",
        "required": ["run_id", "status", "log_tail", "log_truncated"],
        "properties": {
            "run_id": {"type": "string"},
            "status": {"type": "string", "enum": ["passed", "failed", "running", "canceled"]},
            "log_tail": {"type": "string"},
            "log_truncated": {"type": "boolean"},
            "log_total_bytes": {"type": "integer"},
            "full_log_resource": {
                "type": "object",
                "description": "Present iff log_truncated; resource link to the complete log.",
                "properties": {"uri": {"type": "string"}, "mimeType": {"type": "string"}},
            },
        },
        "additionalProperties": False,
    },
)
async def get_run_logs(run_id: str) -> dict:
    run = runs.get(run_id)
    total = run.log_size_bytes
    tail = run.read_log_tail(TAIL_CHARS)            # cheap, bounded read
    result = {
        "run_id": run_id,
        "status": run.status,                       # structural fields kept intact
        "log_tail": tail,
        "log_truncated": total > len(tail.encode()),
        "log_total_bytes": total,
    }
    if result["log_truncated"]:
        # Path to the rest: a resource link the HOST fetches on demand, not inlined now.
        result["full_log_resource"] = {
            "uri": f"ci://runs/{run_id}/log",       # served as an MCP resource
            "mimeType": "text/plain",
        }
    return result
```

Against the gate: the profile is **may-be-truncated** and stated; truncation is **deterministic and visible** (`log_truncated` boolean + `log_total_bytes`, not silent); the **structural fields survive** (`run_id`, `status`) while only the large sub-field (`log_tail`) is cut; and there is a **path to the rest** via a resource link the host fetches on demand rather than burning context inlining 200MB. The architect chose `may-be-truncated` over `paginated` deliberately — the agent wants the failing tail, not page 1 of 4,000 — and that choice is the kind of thing the architect and critic should *disagree* about: a critic might argue offset-range pagination (`get_run_logs(run_id, from_byte, to_byte)`) is better when the agent needs to grep mid-log, and the resolution belongs in the design record.

## Common Mistakes

- **No profile at all.** The tool returns "the matching rows" and nobody asked "how many, how big, what at 100×?". Every shipping tool names exactly one of bounded / paginated / may-be-truncated. Absence is the defect.
- **`SELECT *` as the list shape.** Returning full records from a list tool. Almost always wants the summary tier + a detail tool. The single biggest budget leak.
- **Offset pagination on a live dataset.** `LIMIT/OFFSET` skips and duplicates rows when other agents write between turns. Use a keyset cursor encoding the sort key.
- **Transparent (page-number) cursors.** Letting the agent construct `?page=3` freezes your pagination strategy into the surface and breaks under concurrent inserts. The cursor is opaque; the agent echoes it back unmodified.
- **No completeness signal.** Returning a page with no `nextCursor` convention, so the agent cannot tell "last page" from "cut off". Presence-of-cursor *is* the contract; without it, both look identical.
- **Silent truncation.** Cutting a field with no flag and no length. The agent reasons over a partial value believing it is whole — the worst failure mode because there is no signal at all.
- **Truncating the skeleton instead of the payload.** Dropping `id`/`status` to fit the `body`. Keep the structural fields; truncate the large sub-fields and link the rest.
- **Inlining blobs.** Base64 images, embedding vectors, full files, raw logs in the JSON result. Resource link or detail tool; never inline.
- **Honoring an unbounded page-size request.** Returning 100,000 rows because the agent asked. The server enforces the cap; the agent does not know your row sizes.
- **`structuredContent` without `outputSchema` (or vice versa).** A JSON string stuffed in a text block forces the agent to re-parse a guessed shape every turn. Declare the schema, return structured content, validate it server-side, and keep a text fallback for pre-2025-11-25 hosts.
- **Treating a valid schema as a size guarantee.** A schema-valid 50k-row array is still a budget bomb. Structure and size are orthogonal — you need a profile *and* a schema.
- **Designing against the dev fixture.** "It fits today" with 12 rows. The median-input incident is the bug report; design for the largest realistic input now.
- **Silently changing `outputSchema`.** Removing a field or narrowing a type the agent depends on, with no capability bump — see `schema-versioning-and-drift.md`. The output schema is part of the surface.

## Critic Checklist (severity + evidence)

For each tool, the critic records a finding with severity (blocker / major / minor / nit) and evidence (tool name, the offending field, a return-shape excerpt, or a token estimate at realistic scale):

- **Profile declared and true?** No stated profile → `major`. Stated `bounded`/`paginated` but actually `O(rows)`/`O(content)` → `blocker`. Evidence: the query or the field that scales.
- **List tool summarized?** List returns full records → `major` (often `blocker` if records are large). Evidence: per-record token estimate × realistic count.
- **Cursor opaque + stable?** Offset pagination → `major` (data corruption under concurrency). Transparent page numbers → `minor`. No completeness signal → `major`.
- **Truncation visible + recoverable?** Silent truncation → `blocker`. Truncated with no path to the rest → `major`. Skeleton fields truncated → `major`.
- **Blobs linked, not inlined?** Inlined base64/vectors/files/logs → `major`. Evidence: the field and its realistic size.
- **`structuredContent` + `outputSchema` present and consistent?** JSON-in-text-block → `minor` to `major` (depends on parse fragility). Schema/content mismatch → `blocker`. Non-backward-compatible schema change without capability bump → `blocker`.
- **Page-size cap enforced server-side?** Honors arbitrary `page_size` → `major`.

If the critic's pass on a multi-tool surface produces a clean bill of health, re-run it: a list-shaped surface with no unbounded-return finding is more likely under-audited than genuinely clean. Architect-critic disagreement on a profile choice (e.g. paginate vs truncate for logs) is recorded with both positions and the resolution, not silently overridden.

## See Also

- `mcp-primitive-selection.md` — whether the thing should be a tool result, a resource (with its own URI the host fetches on demand), or a prompt argument. The resource-link path in Example 2 lives at that boundary.
- `tool-api-design.md` — the agent-voice intent statement that carries the profile; naming and parameter shapes (including the `cursor`/`page_size` parameters).
- `error-envelopes-and-recovery.md` — when "too big to serve" is a `retry-with-changes` error directing the agent to paginate, rather than a truncated result.
- `schema-versioning-and-drift.md` — `outputSchema` changes as capability-bumping surface changes.
- `mcp-server-smells.md` — `return-shape-that-blows-budget`, `tool-as-CRUD-mirror`, and the resource/tool inversion smells.
- `testing-mcp-servers.md` — golden conversations that exercise a large-result path so the budget profile is regression-protected, not assumed.
