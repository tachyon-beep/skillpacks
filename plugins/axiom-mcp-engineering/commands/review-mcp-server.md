---
description: Full adversarial audit of an MCP server against all 13 reference sheets and the Consistency Gate - dispatches the mcp-server-critic agent and returns a severity-rated findings list with evidence per finding plus any architect/critic disagreements
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[mcp_server_to_review]"
---

# Review MCP Server Command

You are running a full adversarial audit of an MCP server — its tool catalog, error envelopes, return shapes, idempotency guarantees, schema-versioning policy, transport behaviour, and observability — against all 13 reference sheets of `axiom-mcp-engineering` and the router's Consistency Gate. Your role is to locate the server, dispatch the `mcp-server-critic` agent, and present its findings with severity and evidence on every claim.

This is the CRITIC half of the pack. The architect's command is `/design-mcp-server`. A narrower, tool-surface-only pass is `/audit-mcp-tools` — use that when the caller wants the tool catalog re-read as prompts without re-running the full envelope / versioning / transport / observability sweep. `/review-mcp-server` is the everything pass.

## Invocation path

`/review-mcp-server` is a Claude Code slash command. The command does not perform the audit itself — it locates the server's surface artifacts (source for the tool registrations, the JSON schemas, the error-envelope code, the transport setup, any golden-conversation tests), confirms scope with the caller, dispatches the `mcp-server-critic` agent against the 13 sheets and the Consistency Gate, then surfaces that agent's findings and writes them to disk. Readers seeing this slash command invoked should expect: command locates the surface → command confirms scope → command hands the workspace to the critic agent → agent walks all 13 sheets and the gate → command presents severity-rated, evidence-backed findings and records architect/critic disagreements. For forward design of a new server, use `/design-mcp-server`.

## Core Principle

**Accuracy over comfort. Evidence over opinion. Read every tool as a prompt an unreliable retrying client will misuse.**

A clean audit on a real surface is the failure mode this command exists to prevent. Per the router: if the critic and the architect always agree, the critic is rubber-stamping — that is a defect of the critique, not a property of the server. A 30-tool surface has more pairwise interactions than a single pass can hold; a single golden conversation does not exercise the rare-retry paths. A short findings list on a large surface is evidence the critic read the surface the way the architect wrote it. Treat a no-finding, no-disagreement run as suspect and re-frame.

## Preconditions

Locate the server's surface. The command accepts an optional path to the server (a directory, a manifest, or an entry-point file); if none is supplied, ask the caller where the server lives — do not guess across an unknown repo.

```bash
# Argument-supplied path to the MCP server (directory or entry file)
SERVER="${ARGUMENTS}"

# Surface inventory — what the agent reads
ls -d "${SERVER}" 2>/dev/null
# Tool / resource / prompt registrations (framework-dependent; search broadly)
grep -rEln "(add_tool|@server\.tool|@mcp\.tool|register_tool|ListTools|tools/list|@app\.list_tools|setRequestHandler)" "${SERVER}" 2>/dev/null
# JSON schemas / input shapes
grep -rEln "inputSchema|input_schema|outputSchema|json_schema|zod|pydantic|BaseModel" "${SERVER}" 2>/dev/null
# Error envelopes
grep -rEln "isError|McpError|ErrorData|error_code|raise |throw " "${SERVER}" 2>/dev/null
# Transport setup (stdio / HTTP / SSE)
grep -rEln "stdio|StdioServerTransport|streamable_http|SSEServerTransport|fastmcp|FastMCP" "${SERVER}" 2>/dev/null
# Capability / version declarations
grep -rEln "serverInfo|capabilities|protocolVersion|\"version\"" "${SERVER}" 2>/dev/null
# Golden-conversation / replay tests
grep -rEln "golden|replay|conversation|fixture|snapshot" "${SERVER}" 2>/dev/null
```

**Stop conditions:**

- If `${ARGUMENTS}` is empty and the caller has not named a server, stop and ask: which MCP server (path to its directory or entry file) should be reviewed? Do not scan the whole repo speculatively.
- If the path exists but no tool registrations can be found, do not assume there is nothing to review — report what was searched for, and ask the caller to point at the registration site (frameworks vary: low-level SDK, FastMCP decorators, a JSON manifest, a generated catalog). A surface the command cannot find is an Information Gap, not a clean bill of health.
- If only a deployed catalog is available (the `tools/list` output) with no source, proceed but warn: the review will be **limited**. Idempotency, concurrency, transport, and observability findings are evidenced in the implementation, not the catalog. Record `scope: catalog-only` in Information Gaps.

## Protocol

### Step 1 — Confirm scope

Use `AskUserQuestion` to confirm before dispatching, unless the caller has already specified all of it in the invocation. The audit's coverage and token cost both depend on these answers, so settle them up front:

- **Surface extent** — full server (all 13 sheets + Consistency Gate), or a bounded subset (e.g. "just the new tools added this sprint", "just the error envelopes", "just version-drift exposure ahead of the model upgrade")? The full pass is the default and the reason to use this command over `/audit-mcp-tools`.
- **Source availability** — full source, or catalog-only (`tools/list` output)? This bounds which sheets can be evidenced and what goes in Information Gaps.
- **Architect's design on record** — is there a design package, an ADR, or a prior `/design-mcp-server` output the critic should disagree *against*? Recording architect/critic disagreement requires an architect position to disagree with. If none exists, the critic reconstructs the implied design from the surface and disagrees with that — say so in Caveats.
- **Known incident or trigger** — was this audit prompted by a specific failure (a double-execution under retry, a context-budget blowout, a model-upgrade panic)? A named trigger sharpens the critic's first frame.

Record the scope decision. It bounds coverage and belongs in the Caveats section of the output.

### Step 2 — Dispatch the mcp-server-critic agent

Dispatch the `mcp-server-critic` agent (this pack's critic SME). Pass:

- The server path (`${SERVER}`) and the surface inventory gathered in Preconditions.
- The scope determined in Step 1 (full / subset / catalog-only).
- The architect's design-of-record if one exists, so disagreements can be recorded against a real position.
- Any named incident or trigger.

Instruct the agent to walk **all 13 sheets** and the **Consistency Gate**, not a subset, unless scope was explicitly narrowed. Specifically, the critic must:

1. Re-read every tool description as a prompt fragment an LLM with no source access will act on — ambiguity, overlap, CRUD-mirror naming, parameters the agent cannot fill (`tool-api-design.md`, `mcp-server-smells.md`).
2. Check primitive selection — anything that should be a resource exposed as a tool, or vice versa (`mcp-primitive-selection.md`, `resources-prompts-sampling.md`).
3. Profile every return shape against the context budget on the **median and worst-case** input, not the demo input (`output-shape-and-pagination.md`).
4. Re-read every side-effecting tool as a duplicate-execution risk under retry; verify each declares an idempotency guarantee and a concurrency contract (`idempotency-and-atomicity.md`).
5. Re-read every error path as an agent-recovery problem — is each envelope classifiable as retry-safe / retry-with-changes / fatal, or is it a stack trace? (`error-envelopes-and-recovery.md`).
6. Assess version-drift exposure — non-backward-compatible changes that did not bump the server capability (`schema-versioning-and-drift.md`).
7. Check trust / scoping — what an agent is allowed to do on whose behalf, credentials-as-resource hazards (`authentication-and-trust.md`).
8. Check transport and reconnection assumptions — what state the server assumes across a reconnect (`transport-reliability.md`).
9. Check namespace collisions if the server runs alongside others (`composition-and-namespaces.md`).
10. Check the test posture — golden-conversation regression vs "we asked Claude once" (`testing-mcp-servers.md`).
11. Check observability — can the operator answer "was that one execution or four?" (`observability-for-tool-calls.md`).
12. Run the full smell catalog as a checklist, enumerated, not as a vibe-check (`mcp-server-smells.md`).
13. Run the **Consistency Gate** end to end and report each item as pass / fail with the evidence for the verdict. A silent pass is the failure mode the gate exists to catch.

The agent follows the SME Agent Protocol (`meta-sme-protocol:sme-agent-protocol`) and returns finding / severity / evidence triples plus Confidence, Risk, Information-Gaps, and Caveats, with a `## Summary (machine-readable)` block at the top.

### Step 3 — Present findings

Return the agent's output to the caller. Surface, in this order:

- **Summary (machine-readable)** — copy the agent's verdict / counts / scope into the top of your response so the caller gets the verdict at a glance.
- **Executive summary** — readiness verdict and count of findings by severity.
- **Findings** organised by severity, each with **evidence**: the specific tool name, the parameter, a return-shape excerpt, a conversation/retry-trace fragment, or `file:line`. A finding without evidence is a vibe and must be downgraded to an open question or dropped.
- **Consistency Gate results** — each gate item as pass / fail with its evidence. Failures are blocking, not advisory.
- **Architect/critic disagreements** — see below. This is a required section, not an optional one.
- **What the surface does well** — genuine strengths only; no rubber-stamping.
- **Confidence Assessment**, **Risk Assessment**, **Information Gaps**, **Caveats** (including the scope decision from Step 1).

Then **write** the review to disk (see Output Location) so downstream work can reference it.

## Recording architect/critic disagreements

This is the mechanism the pack exists to exercise. The router is explicit: *if architect and critic always agree, the pipeline is broken.*

For each finding where the critic's position contradicts a choice the architect made (or would defensibly make), record the disagreement as a triple:

- **Architect's position** — the design choice and the reasoning a competent architect would give for it (e.g. "`update_issue` is a single tool with a partial-update payload; keeps the surface small").
- **Critic's position** — the adversarial reading, with evidence (e.g. "under retry-after-network-blip the partial update may apply twice with different intermediate states; evidence: no idempotency key, no `expected_version` parameter, tool documented as safe to retry").
- **Resolution** — the recommended reconciliation (e.g. "add `expected_version` for optimistic locking, or split into `claim_issue` lease + `update_issue`"). If unresolved, say so and name what evidence would resolve it.

If the run produced **zero** disagreements, do not present that as a clean result. Report it as a defect of the critique and re-run the critic with a fresh frame (a different incident lens, the worst-case input, the rare-retry path). A no-disagreement audit on a non-trivial surface is theatre; say so plainly.

## Severity Vocabulary

The critic uses four severity bands (per the Consistency Gate's critic clause). Surface them with the same definitions — do not let the caller relabel severities to suit a ship date.

| Severity | Action | Examples |
|----------|--------|----------|
| **Blocker** | Must fix before release | Side-effecting tool with no idempotency guarantee that an agent will retry; error path returns a stack trace with no recovery class; return shape grows unbounded with the database and will blow context on median input; non-backward-compatible schema change with no capability bump |
| **Major** | Should fix before release; waivable only with explicit, recorded sign-off | Two tools an LLM cannot reliably distinguish; concurrency contract undefined on a tool two agents will call; no golden-conversation regression for a shipping tool; credentials exposed as a resource |
| **Minor** | Advisory; fix in next pass | Tool intent describes implementation rather than agent-visible effect; pagination present but cursor semantics under-documented; observability gap that obscures but does not hide duplicate execution |
| **Nit** | Optional polish | Naming inconsistency across an otherwise coherent surface; redundant field in a return shape that is within budget |

## Prohibited Patterns

### Don't rubber-stamp

**Don't:** "Server looks production-ready, ship it."

**Do:** "Walked all 13 sheets and the Consistency Gate. One Blocker (`create_run` is non-idempotent and documented as retry-safe — double-execution under the same retry that caused last week's incident), two Major (overlapping `get_issue`/`fetch_issue`; no golden-conversation regression), four Minor. Gate: 5 pass / 3 fail. Recommend: fix the Blocker before release, sign off or fix the Majors."

A review that finds nothing is suspect. Either the surface is genuinely clean — then name the specific strengths and the specific worst-case inputs you tried — or the review didn't look hard enough.

### Don't sandwich

Lead with the verdict and the Blockers. Strengths go in their own section, not wrapped around the findings to soften them.

### Don't relabel severity under pressure

Severity is the critic's job, not the caller's. "The retry case is rare, don't block" is not an input the review takes. If a side-effecting tool double-executes under retry, that is a Blocker; the *response* to a Blocker (waive, fix, defer) is the owner's decision, informed by but not overriding the review.

### Don't audit on the demo input

Return-shape and retry findings must be evidenced against the **median and worst-case** input — the issue list with 10,000 rows, the tool called four times during a 40-second hang — not the three-row demo. A finding that only holds on the happy path is not a finding.

### Don't review whether the server should exist

The review checks whether the surface is coherent, recoverable, retry-safe, and regression-protected for an LLM client. It does not adjudicate whether the project should expose MCP at all — that is a product/integration decision the router explicitly places out of scope.

## Handling Pressure

### "We already tested it — the agent used it and it worked"

"We asked Claude and it worked once" is not a test (sheet 11). The review checks for golden-conversation regression that fails loudly when a tool description or return shape regresses. A single successful manual run is evidence of nothing about the rare-retry path or the next model version.

### "A new model ships next month, just tell us it's fine"

Version-drift exposure is exactly what sheet 6 audits. The review will assess which tool descriptions and schemas are most exposed to re-interpretation, not issue reassurance.

### "Just review the catalog, we can't share source"

A catalog-only review can be done but is limited — idempotency, concurrency, transport, and observability are evidenced in the implementation. The review proceeds and flags `scope: catalog-only` in Information Gaps.

## Output Location

Resolve today's date and write the review next to the server. If `${SERVER}` is a directory, write to `${SERVER}/mcp-review-$(date +%Y-%m-%d).md`; if it is a single entry file, write alongside it.

```bash
REVIEW_FILE="${SERVER%/}/mcp-review-$(date +%Y-%m-%d).md"
```

If a review for today already exists, append a disambiguating suffix (`-v2`, `-v3`) rather than overwriting — prior reviews are the change history.

## Cross-Pack Discovery

After the review, suggest downstream handoffs based on the findings:

- **Client-side recovery / tool-loop concerns** (the server is fine but the agent mis-uses it) → `yzmir-llm-specialist` owns the host side of the contract.
- **The MCP surface expresses a multi-stage workflow** (claim → work → close) with decomposition smells → `axiom-procedural-architecture`.
- **Tool-call telemetry must feed an audit-grade provenance chain** → `axiom-audit-pipelines` for canonical encoding and signed logs.
- **The underlying system needs deterministic replay** (not just golden-conversation replay of the surface) → `axiom-determinism-and-replay`.
- **Trust / scoping findings touch credentials, multi-tenant boundaries, or sensitive-data flows** → `ordis-security-architect`.
- **The server is also a human-client REST/GraphQL API** → `axiom-web-backend` for that surface; this review covers only the MCP surface.

## Scope Boundaries

**Covered:**

- Adversarial audit of an MCP server against all 13 reference sheets and the Consistency Gate
- Severity-rated, evidence-cited findings with specific recommendations
- Explicit architect/critic disagreement records with resolutions
- Confidence / Risk / Information-Gaps / Caveats discipline per the SME Agent Protocol
- Handoff recommendations to downstream packs

**Not covered:**

- Forward design of a new server or tool (use `/design-mcp-server`)
- Tool-surface-only review without the full sheet sweep (use `/audit-mcp-tools`)
- Whether the project should expose MCP at all (product/integration decision)
- Client/host-side agent code, prompt design, or tool-loop logic (use `/llm-specialist`)
- Human-client REST/GraphQL API review (use `/web-backend`)
