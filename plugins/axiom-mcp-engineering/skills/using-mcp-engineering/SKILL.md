---
name: using-mcp-engineering
description: Use when designing, implementing, or auditing an MCP (Model Context Protocol) server — tool API design, idempotency under agent retry, structured error envelopes agents can recover from, schema versioning across model drift, transport reliability (stdio / HTTP), output-shape and pagination discipline, and choosing between tools / resources / prompts / sampling. Also use when an MCP server's tools confuse agents, return unstructured errors, deadlock under concurrent calls, double-execute under retry, or lose state across reconnects. Do not use for general REST/GraphQL API design (use `/web-backend`), for client-side prompt engineering or tool-loop design (use `/llm-specialist`), for general in-process plugin architecture (use `/system-architect`), or for cryptographic-provenance audit trails (use `/audit-pipelines`).
---

# Using MCP Engineering

## Overview

**An MCP server is a contract with an unreliable, non-deterministic, retrying client that cannot read your mind — treat it as one, or your tools will work in isolation and fail in agent context.**

The Model Context Protocol exposes four primitives to an agent: **tools** (model-invoked actions with side effects), **resources** (model-readable context the host attaches), **prompts** (user-invoked templates), and **sampling** (the server asks the host to do inference). The discipline of this pack is the *engineering* of those primitives — the tool API surface as seen by an LLM with no source-code access, the failure modes when the agent retries on transient errors, the silent drift when a new model version reads your tool descriptions differently, the deadlock when two agents call the same tool concurrently, the unstructured error string that an agent cannot recover from because it does not parse.

A REST API is read by a human writing client code once. An MCP tool is read by every model on every turn, with no human in the loop to interpret an ambiguous description, recover from a 500 with a stack trace in the body, or notice that two of your tools do almost the same thing. The asymmetry is the discipline: every tool is a prompt fragment, every error is part of the agent's chain-of-thought, every retry is a real possibility, and every output that exceeds the context budget silently breaks the conversation. None of these concerns are addressed by general API discipline; all of them are this pack.

Two roles over one shared corpus: an **architect** that constructs the server surface — tool inventory, parameter shapes, error envelopes, schema versioning policy, transport choice, capability declarations, **observability instrumentation**; and a **critic** that adversarially audits a proposed or deployed server — every tool description re-read as a prompt, every error path re-read as an agent-recovery problem, every retry re-read as a duplicate-execution risk, with **severity** and **evidence** on every finding. The roles read the same 13 reference sheets with different epistemics. If the architect and critic always agree, the critic is rubber-stamping — that is a bug in the pipeline, not a feature.

## When to Use

### Architect triggers (you are building or extending the server)

- "I am about to ship a new MCP tool and I want to know if its name, description, and parameter shape will survive contact with a model that has never seen this project."
- "The server has accreted twenty-something tools and I cannot tell whether to add a twenty-fifth or refactor what I have — I need a surface-area review before I ship."
- "Agents are calling `update_X` then immediately calling it again with the same arguments, and I need to decide whether to make the tool idempotent at the protocol level, at the database level, or via a claim-lease pattern."
- "I have to choose between exposing a row as a tool result, as an MCP resource, or as a prompt argument, and I want the decision rule rather than the vibes."
- "The agent occasionally reconnects mid-conversation and I want to know what state my server is allowed to assume and what it has to recover."

### Critic triggers (you are auditing an existing server or a proposed change)

- "Here is the tool catalog for our MCP server. Tell me which tools an agent will misuse, which descriptions are ambiguous, and which return shapes will blow the context budget on realistic inputs."
- "This server claims to be production-ready but the smoke tests are 'I asked the agent to do X and it worked once.' I need a golden-conversation regression strategy before we cut a release."
- "Two engineers added tools last sprint and I have no idea whether their error envelopes match the rest of the surface — I need an envelope-consistency audit."
- "A new model version is shipping next month and the team is panicking that the tool descriptions will be re-interpreted. I want a version-drift exposure assessment, not vibes."
- "Tool X took 40 seconds last Tuesday and the agent retried three times. Was that one execution or four? I cannot answer that and that is itself the bug."

## Do Not Use This Pack When

- **Designing a general REST or GraphQL API** for human-written clients → use `/web-backend`. MCP discipline assumes an LLM client; non-MCP API design has different constraints and a different audience.
- **Doing prompt engineering, agentic tool-loop design, RAG, or context engineering on the client (host) side** → use `/llm-specialist`. This pack is the *server* side of the agent contract; client-side concerns (when to call a tool, how to summarise its output for the next turn, how to recover from a tool error in the agent's reasoning) live in the LLM pack.
- **Building general in-process plugin systems** (pluggy, entry points, registry patterns) → use `/system-architect` or `/procedural-architecture`. MCP is one specific protocol with one specific client model; plugin architecture in your own runtime is a different problem.
- **Hardening the cryptographic-provenance side of an audit trail** (canonical encoding, signed exports, fingerprint chains) → use `/audit-pipelines`. MCP servers may participate in an audit pipeline, but the cryptographic discipline is its own pack.
- **Building deterministic-replay infrastructure** for the *system the MCP server operates on* → use `/determinism-and-replay`. Golden-conversation replay for the MCP server itself is in scope of this pack (sheet 11); replay of the underlying simulation is not.
- **Choosing whether your project should expose MCP at all** → that is a product / integration decision, not a discipline this pack adjudicates. This pack assumes the answer is yes and concerns itself with *how*.
- **Operating MCP as a host / client** (writing agent code that calls tools, dispatches sampling, manages resources) → out of scope. This pack is server-side discipline.

## Pipeline Position

```
axiom-web-backend                              axiom-mcp-engineering
  HUMAN-AUTHORED API clients   ←-contrast-→     LLM-AUTHORED tool calls
  REST/GraphQL, OpenAPI,                        MCP primitives, retry-by-default,
  human reads docs once,                        every tool description IS a prompt,
  client code written once                      every model version re-interprets
  ───────────────────────────────────────────────────────────────────
        These packs share JSON-over-transport mechanics but differ
        on every higher-level concern: audience (human dev vs LLM),
        retry semantics (occasional vs default), error contract
        (debug-by-stacktrace vs recover-by-envelope), schema drift
        (versioned-with-clients vs versioned-against-models). Cross-
        reference both ways; do not treat MCP as "just an API".

yzmir-llm-specialist (host side)              axiom-mcp-engineering (server side)
  agent loop, prompt design,    ←-contract-→    tool surface, error envelopes,
  tool selection, error                         idempotency, schema versioning,
  recovery in reasoning,                        observability
  context engineering                           ─────────────────────
  ───────────────────────────────────────────────────────────────────
        The MCP boundary is a contract between an LLM client and a
        tool server. /llm-specialist owns the client side: when to
        call, how to interpret results, how to compose tools in a
        reasoning loop. This pack owns the server side: what the
        tools ARE, what they return, what happens under retry. The
        contract artifacts (tool descriptions, error envelopes,
        schemas) belong to this pack but are READ by the other.

axiom-procedural-architecture                  axiom-mcp-engineering
  staged-procedure structure    ←-applied-to-→  multi-step agent workflows
  (stages, decisions, exits)                    expressed as TOOL SEQUENCES
  ───────────────────────────────────────────────────────────────────
        When the MCP server exposes a *workflow* (claim → work →
        close, or scan → triage → promote), the procedural-
        architecture pack applies to the decomposition: stages,
        decision points, exit artifacts. This pack applies to the
        protocol-level expression of that workflow: tool names,
        idempotency guarantees, claim-lease semantics. Cross-link
        when an MCP surface IS a procedure.

axiom-audit-pipelines                          axiom-mcp-engineering
  cryptographic provenance,     ←-feeds-→       tool-call telemetry,
  canonical encoding, signed                    request tracing, retry
  decision logs                                 visibility
  ───────────────────────────────────────────────────────────────────
        When an MCP server is part of an audit-grade system, the
        tool-call telemetry produced by this pack's observability
        discipline (sheet 12) feeds the audit-pipeline. The
        cryptographic side (RFC 8785 JCS, fingerprint chains)
        belongs to /audit-pipelines; the per-call instrumentation
        belongs here.

axiom-determinism-and-replay                   axiom-mcp-engineering
  whole-system replay           ←-vs-→          GOLDEN CONVERSATION replay
  ───────────────────────────────────────────────────────────────────
        Both packs care about replay, but at different granularities.
        /determinism-and-replay is about reconstructing the past
        behaviour of the underlying system as a fact. This pack's
        sheet 11 is about replaying canonical agent conversations
        against an MCP server to catch surface regressions across
        model versions. The techniques overlap; the question
        answered does not.
```

This pack is a sibling of `axiom-web-backend` (which handles human-client APIs), a sibling of `yzmir-llm-specialist` (which handles the client side of the LLM↔server contract), and a downstream consumer of `axiom-procedural-architecture` when the MCP surface expresses a multi-stage workflow. It feeds `axiom-audit-pipelines` with tool-call telemetry when the system is audit-grade. It shares technique with `axiom-determinism-and-replay` but answers a different question (conversation regression, not system reconstruction).

## Role Architecture

The architect and the critic share the corpus of 13 sheets. They do not share epistemics.

**Architect epistemics — constructive, forward.** Given a host system and an intended agent workflow, the architect asks: what is the smallest set of *tools* (with what names, parameter shapes, return shapes, error envelopes) that lets an agent accomplish the workflow with bounded retries and recoverable failures? Which state belongs in tools, which in resources, which in prompts, which in sampling requests? What does the server have to remember across reconnects? The architect builds. The architect's failure modes are well-known: tools named for *what they do* in the codebase rather than for what they *mean* to an agent; overlapping tools that an LLM cannot distinguish; parameter shapes that require the agent to encode information it does not have; return shapes that blow the context budget on the median input; error strings that read like Python tracebacks and tell the agent nothing about how to recover.

**Critic epistemics — adversarial, backward.** Given a proposed or deployed MCP server, the critic asks: what is structurally wrong with this surface, as seen by an LLM that does not have the source code? Where do the tool descriptions allow more than one interpretation? Which tools, called twice with the same arguments, produce different effects (and is that intended)? Which return shapes will be truncated by the host because they exceed context budget? Which error envelopes give the agent no recovery hint? Which tools assume serialised access but are documented as concurrent? Which schemas changed between server versions without a corresponding capability bump? The critic finds. The critic's failure mode is rubber-stamping — producing a clean bill of health on a surface where two of the eight existing tools fail under the same retry pattern that triggered last week's incident.

**If architect and critic always agree, the pipeline is broken.** A typical pass produces at least one substantive disagreement: the architect proposes `update_issue` as a single tool with a partial-update payload; the critic finds that under retry-after-network-blip, the partial update may apply twice with different intermediate states, and demands an explicit `update_issue` + `expected_version` parameter or a separate `claim_issue` lease. Resolving that disagreement is the work this pack exists to do. A run that produces no disagreement is evidence the critic is reading the surface the way the architect wrote it — same blind spots, same defaults — and the audit is theatre. Treat zero-disagreement runs as a defect of the critic, not as a virtue of the architect.

The architect's slash command will be `/design-mcp-server`. The critic's slash command will be `/review-mcp-server`. A tool-surface-focused command, `/audit-mcp-tools`, runs the critic over an existing server's tool catalog without re-deriving the full design. The two SME agents will be `mcp-server-architect` (architect) and `mcp-server-critic` (critic); both will follow the SME Agent Protocol and emit **finding / severity / evidence** triples where they make claims. *(Commands and agents are roadmap for v0.2.0; the v0.1.0 ship is router + sheets.)*

## Start Here

If this is your first time and your input is **"I am designing a new MCP server (or a new tool)"** (architect):

1. Read [tool-api-design.md](tool-api-design.md) — naming, granularity, parameter shapes, tool description as a prompt fragment.
2. Read [mcp-primitive-selection.md](mcp-primitive-selection.md) — tools vs resources vs prompts vs sampling; the decision rule.
3. Read [output-shape-and-pagination.md](output-shape-and-pagination.md) — return-shape discipline, summary vs detail, context-budget awareness.
4. Read [idempotency-and-atomicity.md](idempotency-and-atomicity.md) before you ship any tool with side effects.
5. Emit a tool inventory with: name, intent (one sentence in agent-voice), parameters with constraints, success shape, error envelopes, idempotency guarantee, and concurrency contract.

If your input is **"audit this existing MCP server"** (critic):

1. Read [mcp-server-smells.md](mcp-server-smells.md) — the catalog of anti-patterns; cross-check every tool.
2. Read [error-envelopes-and-recovery.md](error-envelopes-and-recovery.md) — does every error path give the agent a recovery hint?
3. Read [schema-versioning-and-drift.md](schema-versioning-and-drift.md) — does the surface survive a model version change?
4. Read [testing-mcp-servers.md](testing-mcp-servers.md) — golden conversations, replay traces, agent-driven eval.
5. Emit a severity-rated findings list with evidence per finding (tool name, parameter, return-shape excerpt, conversation fragment).

If your input is **"this server is in production and something is wrong"** (operator):

1. Read [observability-for-tool-calls.md](observability-for-tool-calls.md) — what to instrument; how to know whether a retry produced one execution or four.
2. Read [transport-reliability.md](transport-reliability.md) — stdio framing, reconnection, what state the server may assume.
3. Read [composition-and-namespaces.md](composition-and-namespaces.md) — if this server runs alongside other MCP servers in the agent's context, how its tools interact (or collide) with theirs.

For anything not covered by these three entry tracks, use the **Routing** table below.

## Routing

Symptom phrased in the user's own words on the left; sheet to read on the right. At least one row exists per reference sheet.

| Symptom | Sheet |
| --- | --- |
| "I cannot tell whether to add a new tool or refactor an existing one — what's the right granularity?" | [tool-api-design.md](tool-api-design.md) |
| "the agent keeps mis-using `tool_X` and `tool_Y` because they look the same to it" | [tool-api-design.md](tool-api-design.md) and [mcp-server-smells.md](mcp-server-smells.md) (overlapping-tools) |
| "should this be a tool, a resource, a prompt, or a sampling request?" | [mcp-primitive-selection.md](mcp-primitive-selection.md) |
| "the agent ran out of context after one tool call because the response was huge" | [output-shape-and-pagination.md](output-shape-and-pagination.md) |
| "the agent retried after a timeout and the side effect happened twice" | [idempotency-and-atomicity.md](idempotency-and-atomicity.md) |
| "two agents called the same tool concurrently and the database is now inconsistent" | [idempotency-and-atomicity.md](idempotency-and-atomicity.md) (claim-leases) |
| "the agent got a 500 with a stack trace in the body and gave up" | [error-envelopes-and-recovery.md](error-envelopes-and-recovery.md) |
| "the new model version reads our tool descriptions differently and things broke" | [schema-versioning-and-drift.md](schema-versioning-and-drift.md) |
| "we deprecated a parameter and now nothing logs how often it's still being passed" | [schema-versioning-and-drift.md](schema-versioning-and-drift.md) and [observability-for-tool-calls.md](observability-for-tool-calls.md) |
| "the stdio transport drops bytes when the server is slow" | [transport-reliability.md](transport-reliability.md) |
| "the agent reconnected mid-conversation and the server lost its session state" | [transport-reliability.md](transport-reliability.md) |
| "our server runs alongside three other MCP servers and tool names are colliding" | [composition-and-namespaces.md](composition-and-namespaces.md) |
| "is this server actually production-ready or do we just have a smoke test?" | [testing-mcp-servers.md](testing-mcp-servers.md) |
| "tool X took 40 seconds last Tuesday — was that one execution or four?" | [observability-for-tool-calls.md](observability-for-tool-calls.md) |
| "the team wants to expose user-credentials-as-resource and I am not sure that is safe" | [authentication-and-trust.md](authentication-and-trust.md) |
| "this surface has a smell I cannot name" | [mcp-server-smells.md](mcp-server-smells.md) |
| "is this question really for this pack, or is it `/web-backend` / `/llm-specialist`?" | [mcp-boundary-and-handoffs.md](mcp-boundary-and-handoffs.md) |

## How to Access Reference Sheets

All reference sheets will live in the same directory as this `SKILL.md`. When you see a link like `[tool-api-design.md](tool-api-design.md)`, read the file from the same directory. Sheets are designed to be loadable independently — the router selects which sheet to read; the sheet does the work.

The 13 sheets:

**Foundations (architect cluster — what the server IS):**

1. [tool-api-design.md](tool-api-design.md) — naming, granularity, parameters; the tool description as a prompt fragment; how an LLM with no source code reads your surface
2. [mcp-primitive-selection.md](mcp-primitive-selection.md) — tools vs resources vs prompts vs sampling; the decision rule for each piece of state and each interaction
3. [output-shape-and-pagination.md](output-shape-and-pagination.md) — JSON return discipline, summary vs detail, pagination, context-budget awareness, truncation policy

**Discipline (architect / critic — what the server GUARANTEES):**

4. [idempotency-and-atomicity.md](idempotency-and-atomicity.md) — retry-safe tool semantics, claim leases, optimistic locking, exactly-once-under-retry patterns, concurrency contracts
5. [error-envelopes-and-recovery.md](error-envelopes-and-recovery.md) — structured errors with recovery hints; what an agent can do with each error class; what a stack trace in the body costs you
6. [schema-versioning-and-drift.md](schema-versioning-and-drift.md) — backward-compatible parameter evolution, capability negotiation, surviving model-version drift, deprecation that the surface itself signals
7. [authentication-and-trust.md](authentication-and-trust.md) — multi-project servers, per-user / per-project scoping, capability tokens, what an agent is allowed to do on whose behalf

**Beyond Tools (architect — what the server EXPOSES BESIDES tools):**

8. [resources-prompts-sampling.md](resources-prompts-sampling.md) — when an MCP resource carries the load better than a tool; user-invoked prompts; the sampling primitive (server asks host to infer); the failure mode of "everything is a tool"
9. [composition-and-namespaces.md](composition-and-namespaces.md) — multiple MCP servers in one agent context; tool-name collisions; capability negotiation across servers; how your server's surface interacts with theirs
10. [transport-reliability.md](transport-reliability.md) — stdio framing, HTTP transport, reconnection semantics, what state the server may assume across a reconnect, backpressure under slow clients

**Quality and Operations (critic — what proves the server WORKS):**

11. [testing-mcp-servers.md](testing-mcp-servers.md) — golden-conversation regression, replay traces, deterministic test harnesses, agent-driven eval, "we asked Claude and it worked once" is not a test
12. [observability-for-tool-calls.md](observability-for-tool-calls.md) — per-call telemetry, retry visibility, idempotency-key tracing, surfacing of duplicate execution, dashboards that answer "was that one execution or four?"
13. [mcp-server-smells.md](mcp-server-smells.md) — catalogued anti-patterns: overlapping-tools, tool-as-CRUD-mirror, error-as-stack-trace, parameter-the-agent-cannot-fill, return-shape-that-blows-budget, retry-amplification, schema-drift-without-bump, namespace-collision, resource-that-should-be-a-tool, tool-that-should-be-a-resource

**Boundary sheet** *(absorbed into the routing table for v0.1.0; promoted to its own sheet if the routing question becomes a recurrent ask):* mcp-boundary-and-handoffs — where this pack stops; cross-pack handoffs to `/web-backend`, `/llm-specialist`, `/procedural-architecture`, `/audit-pipelines`.

## v0.1.0 Scope

This is the **scaffold release**. Shipped:

- This router (`using-mcp-engineering/SKILL.md`).
- Plugin metadata (`plugin.json`).

Not yet shipped (roadmap for v0.2.0+):

- The 13 reference sheets.
- The three slash commands (`/design-mcp-server`, `/review-mcp-server`, `/audit-mcp-tools`).
- The two SME agents (`mcp-server-architect`, `mcp-server-critic`).

The router is intentionally self-supporting: a producer or critic can extract substantial value from the routing table, role architecture, and consistency gate below before any sheet is written. Sheets exist to deepen the discipline, not to substitute for thinking about it.

## Consistency Gate

Run this checklist before declaring any architect or critic deliverable done. Failures are blocking, not advisory. Silent passes are the failure mode this pack exists to prevent.

- **Every tool has an agent-voice intent statement.** Not "updates the issue record" but "marks an issue as in-progress so other agents do not claim it." The intent is what the LLM reads to decide whether to call the tool; if the intent describes the implementation rather than the effect-an-agent-cares-about, the tool is mis-described and will be mis-called.
- **Every tool with a side effect declares an idempotency guarantee.** "Calling this tool twice with the same arguments has effect X" where X is one of: *no-op-after-first* (idempotent), *adds-twice* (non-idempotent, agent must coordinate), *fails-second-call* (uniqueness-protected), *requires-claim-lease* (atomic-via-lease). "It depends" is not a guarantee.
- **Every tool's return shape has a stated context-budget profile.** Either bounded (`returns ≤ N tokens by construction`) or paginated (`returns first page with cursor`) or explicitly-may-be-truncated (`agent must call detail-tool for full payload`). A tool whose return shape grows with the database is a defect waiting for the median-input incident.
- **Every error envelope tells the agent what to do.** Three classes minimum: *retry-safe* (transient; agent may retry with same args), *retry-with-changes* (input was wrong in a stated way; agent must alter args), *fatal* (no agent-side recovery; surface to user). A stack trace is none of these. "Internal server error" is none of these.
- **Schema changes that are not backward-compatible bump the server capability.** A parameter renamed, a return field type-changed, a tool removed, a permission tightened — all of these are visible to the agent in the protocol, not invisible behind a server version bump. Per [schema-versioning-and-drift.md](schema-versioning-and-drift.md) once it exists; until then, follow the protocol's capability-negotiation envelope.
- **Every tool's concurrency contract is stated.** "Safe to call concurrently with itself and with all other tools" is a contract. "Serialised on the issue ID" is a contract. "Undefined" is a defect, not a contract.
- **At least one golden-conversation regression test exists per shipping tool.** Not "we asked Claude to use it and it worked." A canonical conversation, captured, replayable, that fails loudly when a tool description or return shape regresses. If the surface is not regression-protected, every new model version is a release candidate by default.
- **Smells from sheet 13 ([mcp-server-smells.md](mcp-server-smells.md)) actively considered.** Each catalogued anti-pattern (overlapping-tools, tool-as-CRUD-mirror, error-as-stack-trace, parameter-the-agent-cannot-fill, return-shape-that-blows-budget, retry-amplification, schema-drift-without-bump, namespace-collision, resource-that-should-be-a-tool, tool-that-should-be-a-resource) was checked against the surface. "No smells found" without enumerating the catalog is a skipped check.

For critic deliverables specifically, also:

- **Every finding carries severity and evidence.** Severity is one of *blocker / major / minor / nit*. Evidence is the specific tool name, the parameter, the return-shape excerpt, the conversation fragment, or the retry trace that grounds the finding. A finding without evidence is a vibe. Architect-and-critic disagreement is recorded with both positions and the resolution; silent override of the architect is a defect of the critique.

## Anti-Overconfidence

The architect's first design always *feels* clean. It is not. Tools named after database tables (`get_issue`, `update_issue`, `delete_issue`) look like a coherent CRUD surface and are, almost always, a mis-modeling — the agent wants verbs from the workflow (`claim_issue`, `release_claim`, `close_issue`), not verbs from the data model. Error envelopes copied from a REST API ("400 Bad Request", message="invalid field") look professional and are, almost always, useless to an agent that has no way to know which field. Schemas that "obviously" will not change get re-interpreted by the next model release in ways the architect could not have anticipated.

The critic's first audit always *finds enough*. It does not. A 30-tool surface has more pairwise interactions than a single pass can hold; a single golden conversation does not exercise the rare-retry paths; the smell catalog has to be run as a checklist, not as a vibe-check. A critic that produces a short findings list on a large surface is, almost always, reading the surface the way the architect wrote it.

If the run produced no architect-critic disagreement, the pipeline is broken. Re-run the critic with a fresh frame, or assume the audit is theatre.
