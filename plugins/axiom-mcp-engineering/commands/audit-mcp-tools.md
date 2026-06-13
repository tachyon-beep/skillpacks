---
description: Run a focused CRITIC pass over an existing MCP server's TOOL CATALOG only — not a full server re-design. Re-reads each tool description as a prompt fragment (the way an LLM with no source access reads it) and audits the catalog along five axes — overlapping/indistinguishable tools, idempotency under retry, return-shape context budget, error-envelope recoverability, and concurrency contract. Produces a structured finding / severity / evidence list keyed to the `axiom-mcp-engineering` smell catalog (`mcp-server-smells.md`) and tool-API discipline (`tool-api-design.md`), with the Consistency Gate clause each finding violates. For full server design or a whole-surface critic pass use `/design-mcp-server` or `/review-mcp-server`; for runtime detection of retry-amplification use the observability sheet, not this command.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[tool_catalog_to_audit]"
---

# Audit MCP Tools Command

You are running a **focused critic pass over an MCP server's tool catalog** — and *only* the tool catalog. This is not `/design-mcp-server` (which re-derives a surface from a workflow) and not `/review-mcp-server` (which audits the whole server: transport, resources, prompts, sampling, auth, versioning, observability). This command re-reads each *tool* as the LLM reads it — name, `description`, `inputSchema`, annotations, declared return shape — with no source code in hand, and asks the five questions an agent's behaviour actually depends on: *can it pick the right tool, can it fill the parameters, will the result fit the context, will it know what to do when the call fails, and is it safe to retry or call concurrently?*

The output is a structured **finding / severity / evidence** list. This command does not edit the server, does not redesign tools, and does not run the server. It produces findings a producer can act on.

## Critic epistemics (read this first)

You are the **critic**, not the architect. The architect's tools always *feel* clean because they mirror the code the architect just wrote. Your job is to read them as the model will — without the code. Hold two rules:

- **Read the description literally, as a prompt fragment.** Everything the author knows that is not in the `name` + `description` + `inputSchema` + annotations does not exist. If you find yourself reasoning "well, obviously the agent would know to call X first," stop — *where in the catalog text* does it learn that?
- **A clean pass on a non-trivial catalog is a red flag, not a result.** Per the router's Anti-Overconfidence note: if you agree with everything, you are reading the surface the way it was written, not the way it will be called. If your findings list is short and the catalog has more than a handful of tools, re-read it cold before reporting.

Every finding carries **severity** (`blocker / major / minor / nit`) and **evidence** (the exact tool name, parameter, description excerpt, or declared return shape). A finding without evidence is a vibe and does not ship.

## Invocation path

`/audit-mcp-tools` is a Claude Code slash command. The `[tool_catalog_to_audit]` argument may be:

- a path to a file containing a `tools/list` JSON payload (or the `tools` array),
- a path to the server source where tool definitions live (you will extract the catalog), or
- omitted — in which case locate the catalog yourself (see Locate the catalog).

### Locate the catalog

If no path is given, find the tool definitions:

```bash
grep -rn "tools/list\|\"inputSchema\"\|@server.tool\|@mcp.tool\|registerTool\|setRequestHandler" . \
  --include="*.py" --include="*.ts" --include="*.js" --include="*.json" -l
```

Prefer an actual `tools/list` response payload if one exists (a captured fixture, a golden file) — that is *exactly* what the agent sees. If only source exists, reconstruct each tool's effective `name` / `description` / `inputSchema` / annotations / declared return shape, because a discrepancy between the source and the rendered catalog is itself a finding.

### Confirm scope before a large or ambiguous catalog

If the catalog has more than ~15 tools, or you found multiple candidate catalogs, or it is unclear whether the server runs alongside other MCP servers (which gates the namespace-collision axis), use **AskUserQuestion** to confirm before auditing — which catalog, whether to audit all tools or a named subset, and whether this server coexists with others in the agent's context. Do not silently pick one; a mis-scoped audit wastes the pass.

## The five audit axes

Run **every axis against every tool**. Skipping an axis on a tool is a skipped check; report it as not-assessed rather than implying a clean result. Each axis names the smell catalog entry and the Consistency Gate clause it maps to.

### Axis 1 — Overlapping / indistinguishable tools

Re-read the descriptions side by side. For each pair (and cluster) of tools, ask: *can I write a one-sentence rule for when the agent should prefer A over B that does not require reading the source?* If not, they overlap.

- Cluster tools by leading verb (`get_` / `list_` / `search_` / `query_` / `find_`). Three `list_*` tools that differ only by filter are usually one tool with parameters.
- Check that each name is a **workflow verb** (agent-voice: `claim_issue`, `record_deployment`) not a **CRUD/codebase verb** (`update_issue`, `create_record`). A CRUD-mirror name is at least *major* — it predicts mis-fill, hides the workflow, and structurally cannot declare one idempotency guarantee.
- Check each `description` does the four jobs: states agent-voice intent first, disambiguates from siblings, states preconditions/ordering, states negative scope ("does NOT create — use X").

Smell: `overlapping-tools`, `tool-as-CRUD-mirror` (`mcp-server-smells.md`). Discipline: the core inversion and the description-as-prompt-fragment sections of `tool-api-design.md`. Gate clause: *every tool has an agent-voice intent statement*.

Severity guide: indistinguishable pair → *major*; CRUD-mirror name on a workflow surface → *major*; implementation-voice description → *major*; missing negative scope on a tool with an adjacent misuse → *minor/major*.

### Axis 2 — Idempotency under retry

For every **side-effecting** tool, find where the `description` states what a second identical call does. The Gate requires exactly one of: *no-op-after-first*, *adds-twice* (agent must coordinate), *fails-second-call*, *requires-claim-lease*. "It depends" or silence is a failure.

- A side-effecting tool with **no declared guarantee** is the highest-value finding here, because agents, hosts, and networks all retry — `send_email`, `post_comment`, `charge_card`, `notify_*`, `append_*` with no idempotency key and no dedup are the classic `retry-amplification` smell, and it is *silent and load-correlated*.
- A generic mutator (`update_X`) cannot declare a single guarantee because the guarantee depends on the field — flag it as a multi-effect tool that must be split.
- False-positive check: a naturally idempotent effect (`set_status(id, "closed")` → *no-op-after-first*) is fine **if declared**. The smell is *undeclared* non-idempotence on an effect that matters. Read-only tools are idempotent by construction — confirm they are marked read-only.

Smell: `retry-amplification` (`mcp-server-smells.md`). Discipline: the CRUD-mirror idempotency-lies argument in `tool-api-design.md`. Gate clause: *every tool with a side effect declares an idempotency guarantee*.

Severity guide: undeclared non-idempotence on a mattering effect → *blocker* if called in a loop, *major* otherwise; multi-effect mutator → *major*.

### Axis 3 — Return-shape context budget

For every tool, find the stated context-budget profile: *bounded* (`≤ N tokens by construction`), *paginated* (first page + cursor), or *explicitly-may-be-truncated* (with a detail-tool pointer). Absence is a finding.

- The trigger is **unbounded growth**, not size per se: `list_*` / `search_*` / `read_file` that returns full records/files with no pagination and no projection. Evaluate the **median realistic input**, not the worst case — if the median result is thousands of tokens, it is a budget finding.
- The dangerous outcome is **silent host truncation**: the agent reasons over a fragment it does not know is partial. A return that may be truncated must say so and offer a path to the full payload.
- False-positive check: a *constructively bounded* return (`get_issue(id)` → one record; `count_*` → a number) is fine without pagination. A documented `may-be-truncated` tool with a companion detail-tool is a design, not a smell.

Smell: `return-shape-that-blows-budget` (`mcp-server-smells.md`). Discipline: `tool-api-design.md` cross-ref to `output-shape-and-pagination.md`. Gate clause: *every tool's return shape has a stated context-budget profile*.

Severity guide: unbounded growth on a tool called in a loop → *blocker*; unbounded on a once-per-session tool → *minor*; missing-but-actually-bounded → *nit* (ask for the profile to be stated).

Also flag **parameter-the-agent-cannot-fill** here as you read each `inputSchema`: a *required* parameter the agent has no path to (`tenant_uuid`, `internal_user_id`, `etag`, free-form `options: object`, an `idempotency_key` the server should mint) is a *blocker* — the tool cannot be reliably called. The test is *can the agent obtain this from conversation / prior result / user request / training?*, not *does it look technical?* (a `cursor` returned by the previous page is fillable). Smell: `parameter-the-agent-cannot-fill`.

### Axis 4 — Error-envelope recoverability

For every tool that can fail, find what the failure looks like to the agent. The Gate requires three classes minimum: *retry-safe*, *retry-with-changes*, *fatal* — each with a recovery hint in agent-voice.

- Flag any tool whose documented/known failure crosses the wire as a stack trace, a raw exception, `{"error": "Internal Server Error"}`, a file path, a line number, or a class name. A traceback is *none* of the three classes — the agent cannot tell whether to retry as-is, retry with changed args, or stop, so it retries blindly (amplifying side effects per Axis 2) or abandons a fixable task.
- Check that *input-validation* failures are returned as **Tool Execution Errors** in the result (`isError: true`), not as JSON-RPC protocol errors — so the model can self-correct. A protocol error for a fixable bad argument is its own variant of the smell.
- False-positive check: a full traceback *logged server-side* is correct. The smell is specifically what is *model-visible*.

Smell: `error-as-stack-trace` (`mcp-server-smells.md`). Discipline: `tool-api-design.md` cross-ref to `error-envelopes-and-recovery.md`. Gate clause: *every error envelope tells the agent what to do*.

Severity guide: model-visible traceback on a common failure path → *major*; missing recovery hint on a `retry-with-changes` case → *major*; no error classes documented at all → *major*.

### Axis 5 — Concurrency contract

For every tool, find the stated concurrency contract: is it safe to call concurrently, does it assume serialised access, does it take a lease?

- A side-effecting tool documented as concurrent but implemented assuming serialised access (the read-modify-write race: two agents `get` then `update` the same row, both "win") is a *major/blocker*. A claim/lease verb (`claim_issue` with `expected_version` / `expected_status`) makes the race impossible — its absence on a contested workflow is the finding.
- A read-only tool is safe-concurrent by construction; confirm it says so.
- Cross-reference Axis 2: idempotency and concurrency are the back of the same decision. A tool that cannot declare one usually cannot declare the other.

Smell: `tool-as-CRUD-mirror` (the concurrency facet) (`mcp-server-smells.md`). Discipline: `tool-api-design.md` cross-ref to `idempotency-and-atomicity.md` (claim-leases). Gate clause: *concurrency contract* (the supporting-evidence section of the Consistency Gate).

Severity guide: unguarded contested write → *blocker*; missing-but-actually-serialised-internally → *minor* (state the contract).

## Output format

Emit findings as structured JSON, HIGH-severity first:

```json
{
  "scope": "tool-catalog-only",
  "catalog_source": "<path or fixture audited>",
  "tools_audited": 8,
  "axes_run": ["overlap", "idempotency", "return-budget", "error-envelope", "concurrency"],
  "summary": {"blocker": 2, "major": 3, "minor": 1, "nit": 0},
  "findings": [
    {
      "smell": "retry-amplification",
      "tool": "notify_owner",
      "axis": "idempotency",
      "severity": "blocker",
      "symptom_seen": "side-effecting notify with no stated idempotency guarantee",
      "evidence": "description: \"Notifies the issue owner.\" — no idempotency clause; no key parameter in inputSchema",
      "false_positive": "no — effect is a real outbound notification, not naturally idempotent",
      "fix": "accept or server-generate an idempotency key and dedup; declare no-op-after-first in the description",
      "gate_clause": "every tool with a side effect declares an idempotency guarantee",
      "sheet": "idempotency-and-atomicity.md"
    },
    {
      "smell": "parameter-the-agent-cannot-fill",
      "tool": "update_issue",
      "axis": "return-budget",
      "severity": "blocker",
      "symptom_seen": "required assignee_id:int — agent has natural language, not the internal id",
      "evidence": "inputSchema.required includes \"assignee_id\" (integer); no prior tool returns it",
      "false_positive": "no — no path from conversation/prior result to a valid integer",
      "fix": "derive assignee from auth context server-side; drop the parameter",
      "gate_clause": "agent-voice intent (input written for the wrong reader)",
      "sheet": "tool-api-design.md"
    }
  ]
}
```

Present `blocker` then `major` then `minor` then `nit`; within a band order by tool name. After the JSON, write a 3–5 sentence plain-language triage: the one or two structural mistakes most findings are facets of (almost always "the catalog models the database, not the workflow"), and the smallest set of changes that closes the most blockers. If you found *no* findings on a non-trivial catalog, say so explicitly and state that you re-read the catalog cold — per the critic epistemics, an empty list is a claim that needs its own justification.

## Boundary — what this command does NOT do

- It does **not** audit resources, prompts, sampling, transport, auth, schema-versioning, or observability. Those are `/review-mcp-server`. If you notice a non-tool issue in passing, note it in one line under the triage as "out of scope for /audit-mcp-tools — see /review-mcp-server", do not expand the audit.
- It does **not** *detect* retry-amplification at runtime — that needs production traces (`observability-for-tool-calls.md`). This command finds the *structural* precondition (undeclared non-idempotence), which is the cheaper, earlier signal.
- It does **not** redesign tools or edit the server. Findings only. The architect side is `/design-mcp-server` and the `mcp-server-architect` agent.
- The namespace-collision smell is only partially assessable from one catalog in isolation; flag generic bare names (`search`, `get`, `list`) as *potential* collisions and note that confirmation requires the real multi-server rendered tool list.

## Cross-references

- `mcp-server-smells.md` — authoritative smell catalog and the critic finding format this command emits (all five axes map to entries here)
- `tool-api-design.md` — workflow-verb naming, description-as-prompt-fragment, fillable parameters, granularity (axes 1, 2, 3-parameters)
- `using-mcp-engineering` (router `SKILL.md`) — the Consistency Gate clauses every finding cites, and the architect/critic split
- `output-shape-and-pagination.md` — context-budget profiles for axis 3
- `idempotency-and-atomicity.md` — idempotency guarantees and claim-leases for axes 2 and 5
- `error-envelopes-and-recovery.md` — the three error classes for axis 4
- `observability-for-tool-calls.md` — runtime detection of retry-amplification (out of this command's scope)
- `mcp-server-critic` agent — narrative synthesis of a catalog audit; follows the SME Agent Protocol with Confidence / Risk / Information-Gaps / Caveats
- `/review-mcp-server` — the whole-surface critic pass; `/design-mcp-server` — the architect counterpart
