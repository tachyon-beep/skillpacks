---
description: Forward-design an MCP server surface for an LLM client — derive the tool inventory (agent-voice intent, parameters, success/error shapes, idempotency guarantee, concurrency contract), select primitives (tools vs resources vs prompts vs sampling), set output-shape and pagination discipline, and run the consistency gate; dispatches the mcp-server-architect agent and routes through the 13 reference sheets
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[server_or_workflow_to_design]"
---

# Design MCP Server Command

You are running the **architect** half of `axiom-mcp-engineering`. Your job is to take a description of a host system and an intended agent workflow and produce a complete, defensible **MCP server surface specification** — a tool inventory where every tool carries an agent-voice intent, a parameter shape an LLM can actually fill, a success shape with a stated context-budget profile, a structured error envelope set, an idempotency guarantee, and a concurrency contract; plus the primitive-selection decisions (which state is a tool, which a resource, which a prompt, which a sampling request) and the transport/versioning/observability posture — culminating in a surface that passes the pack's consistency gate.

**The discipline that makes this command necessary:** an MCP tool is not a REST endpoint. It is read by every model on every turn, with no human in the loop to interpret an ambiguous description, recover from a 500 with a stack trace in the body, notice that two tools do almost the same thing, or realise that a tool retried after a timeout just double-executed. The architect designs for an unreliable, non-deterministic, retrying client that cannot read your mind. If the output reads like a CRUD surface named after database tables, you have built the wrong thing — see Anti-Overconfidence below.

## Invocation path

`/design-mcp-server` is a Claude Code slash command that drives the forward-design workflow in the current session. It orchestrates the architect-cluster reference sheets in `axiom-mcp-engineering` (tool API design, primitive selection, output shape, idempotency, error envelopes, schema versioning, transport, observability) and dispatches the `mcp-server-architect` SME agent for a structured surface proposal. The command is the entry point; the sheets and the agent do the work; the **consistency gate** decides whether the surface specification is allowed to ship.

For adversarial critique of an *existing or proposed* server, use `/review-mcp-server`. For a tool-catalog-only audit without re-deriving the design, use `/audit-mcp-tools`.

## Preconditions

The command takes a single argument: a description of the server or workflow to design. This may be:

- A short inline description of the host system and the agent workflow it should expose (e.g. "an issue tracker; agents should be able to find work, claim it, work it, and close it without colliding").
- A path to a file describing the system (a brief, an HLD, a tracker schema, an existing tool list to redesign).
- A path to a directory of source material (the host codebase, an API spec, an existing partial MCP server).

### Resolve the argument

```bash
INPUT="${ARGUMENTS}"

# Empty argument -> ask the user
if [ -z "${INPUT}" ]; then
  # Use AskUserQuestion to collect:
  # "What MCP server are we designing? Describe the host system and the agent
  #  workflow it should expose — or give me a path to a file or directory
  #  (a brief, an existing tool list, the host codebase)."
  :
fi

# File path -> verify readable
if [ -f "${INPUT}" ]; then
  echo "Reading design input: ${INPUT}"
elif [ -d "${INPUT}" ]; then
  echo "Treating input as source material directory: ${INPUT}"
  ls "${INPUT}" 2>/dev/null
else
  echo "Treating input as inline description: ${INPUT}"
fi
```

If the argument looks like a path (contains `/` or ends in `.md` / `.txt` / `.json`) but the file or directory does not exist, stop and ask the user whether they meant a different path or want to treat the string as an inline description.

### Establish the agent workflow before the tool list

Before designing any tool, you must be able to state the **agent workflow** in agent-voice: the sequence of intentions an LLM client will form against this server. "Find a piece of unclaimed work → claim it so no other agent takes it → do the work → record the result → release or close." The tool inventory is *derived from* the workflow, not from the host's data model. If the input gives you a data model (tables, entities, CRUD operations) but not a workflow, that is your first gap — use `AskUserQuestion` to elicit the workflow. A surface designed from the data model is the single most common architect failure this pack exists to prevent.

## Workflow

Run these steps in order. Each step reads its reference sheet(s) and produces a section of the surface specification. Do not skip a step because it "looks obvious" — the obvious design is usually the CRUD-mirror.

1. **Workflow framing** — restate the host system and the agent workflow in agent-voice. Identify the intentions the agent will form. This is the spine the tool inventory hangs off.

2. **Tool API design** — read [tool-api-design.md](../skills/using-mcp-engineering/tool-api-design.md).
   For each intention in the workflow, derive a tool with:
   - **name** — a workflow verb (`claim_issue`), not a data-model verb (`update_issue`).
   - **agent-voice intent** — one sentence describing the *effect an agent cares about*, not the implementation. "Marks an issue in-progress so other agents do not claim it," not "updates the status column."
   - **parameters** — each with type, constraints, and a note on *where the agent gets this value*. A parameter the agent cannot fill from prior context or tool results is a defect, not a parameter.
   - Decide granularity: one tool per intention, not one tool per table operation. Flag any two tools an LLM could not distinguish from their descriptions alone.

3. **Primitive selection** — read [mcp-primitive-selection.md](../skills/using-mcp-engineering/mcp-primitive-selection.md) and, for the non-tool cases, [resources-prompts-sampling.md](../skills/using-mcp-engineering/resources-prompts-sampling.md).
   For each piece of state and each interaction, apply the decision rule: is this a **tool** (model-invoked action with side effects), a **resource** (model-readable context the host attaches), a **prompt** (user-invoked template), or a **sampling** request (server asks the host to infer)? Resist "everything is a tool" — read-mostly context the agent should always have is usually a resource.

4. **Output shape & pagination** — read [output-shape-and-pagination.md](../skills/using-mcp-engineering/output-shape-and-pagination.md).
   For each tool's success shape, assign a **context-budget profile**: *bounded* (`returns ≤ N tokens by construction`), *paginated* (`returns first page with cursor`), or *explicitly-may-be-truncated* (`agent must call a detail-tool for full payload`). A return shape that grows with the database is a defect waiting for the median-input incident. Define summary-vs-detail splits where a list tool would otherwise blow the budget.

5. **Idempotency & concurrency** — read [idempotency-and-atomicity.md](../skills/using-mcp-engineering/idempotency-and-atomicity.md).
   For every tool with a side effect, declare an **idempotency guarantee** — one of: *no-op-after-first* (idempotent), *adds-twice* (non-idempotent; agent must coordinate), *fails-second-call* (uniqueness-protected), *requires-claim-lease* (atomic-via-lease). And a **concurrency contract** — "safe to call concurrently with itself and all other tools," or "serialised on the issue ID," never "undefined." Where the workflow has claim semantics, specify the claim-lease / optimistic-locking mechanism (e.g. `expected_version`).

6. **Error envelopes** — read [error-envelopes-and-recovery.md](../skills/using-mcp-engineering/error-envelopes-and-recovery.md).
   For each tool, define the structured error envelopes across the three minimum classes: *retry-safe* (transient; agent may retry with same args), *retry-with-changes* (input was wrong in a stated way; agent must alter named args), *fatal* (no agent-side recovery; surface to user). Every envelope must tell the agent **what to do next**. A stack trace is none of these; "internal server error" is none of these.

7. **Schema versioning posture** — read [schema-versioning-and-drift.md](../skills/using-mcp-engineering/schema-versioning-and-drift.md).
   State the surface's policy for backward-compatible parameter evolution and how a non-backward-compatible change (renamed parameter, type-changed field, removed tool, tightened permission) bumps the server capability so it is visible to the agent in the protocol — not invisible behind a server version bump. Note any descriptions at risk of re-interpretation by the next model version.

8. **Transport, auth, observability posture** — read [transport-reliability.md](../skills/using-mcp-engineering/transport-reliability.md), and where the system is multi-project or credential-bearing, [authentication-and-trust.md](../skills/using-mcp-engineering/authentication-and-trust.md), and for the instrumentation contract, [observability-for-tool-calls.md](../skills/using-mcp-engineering/observability-for-tool-calls.md).
   Decide transport (stdio / HTTP), state what session state the server may assume across a reconnect, scope per-user / per-project trust boundaries, and specify the per-call telemetry that will later let an operator answer "was that one execution or four?" (idempotency-key tracing, retry visibility).

9. **Composition check** — read [composition-and-namespaces.md](../skills/using-mcp-engineering/composition-and-namespaces.md).
   If this server runs alongside other MCP servers in one agent context, check tool-name collisions and namespace your surface accordingly.

10. **Smell sweep** — read [mcp-server-smells.md](../skills/using-mcp-engineering/mcp-server-smells.md).
    Run the anti-pattern catalog as a checklist against the proposed surface: overlapping-tools, tool-as-CRUD-mirror, error-as-stack-trace, parameter-the-agent-cannot-fill, return-shape-that-blows-budget, retry-amplification, schema-drift-without-bump, namespace-collision, resource-that-should-be-a-tool, tool-that-should-be-a-resource. "No smells found" without enumerating the catalog is a skipped check.

11. **Dispatch the architect agent** — dispatch the **`mcp-server-architect`** SME agent (per the SME Agent Protocol, `meta-sme-protocol:sme-agent-protocol`) with the workflow framing and your draft surface. The agent returns a structured surface proposal with **Confidence / Risk / Information-Gaps / Caveats**. Use it to harden — not replace — your own derivation; reconcile any disagreement explicitly rather than silently adopting either side.

12. **Consistency gate** — run the gate below. Failures are blocking. If the gate fails, fix the surface and rerun; do not emit the specification with silent waivers.

## Dispatching the architect agent

The `mcp-server-architect` agent is the SME for surface construction. Dispatch it once you have a draft tool inventory (after step 10), giving it the agent-voice workflow and the draft surface. It re-derives the inventory independently and returns finding-grade observations on naming, granularity, primitive selection, idempotency, and return shapes, with Confidence / Risk / Information-Gaps / Caveats. Treat its output as a second derivation to reconcile against yours: where it disagrees, record both positions and the resolution. A proposal the agent rubber-stamps with no disagreement is a signal the agent is reading the surface the way you wrote it — re-prompt with a sharper frame.

## Output

Emit a **tool inventory and surface specification**. For each tool, at minimum:

```
tool: claim_issue
  intent (agent-voice): "Take ownership of an unclaimed issue so no other agent works it concurrently."
  parameters:
    issue_id      (string, required) — from a prior find_unclaimed_work result
    expected_version (int, required) — the version the agent last observed; rejects stale claims
  success shape: { issue_id, claimed_by, lease_expires_at }   [bounded, ≤ ~80 tokens]
  error envelopes:
    retry-safe:         transient store unavailable — retry with same args
    retry-with-changes: version_conflict — issue changed; re-read and retry with new expected_version
    fatal:              issue_not_found — surface to user
  idempotency:  fails-second-call (a second claim on a held lease returns version_conflict)
  concurrency:  serialised on issue_id via claim-lease
```

Plus surface-level sections: primitive-selection decisions (tools / resources / prompts / sampling), transport + session-state posture, schema-versioning policy, observability instrumentation contract, namespace, and the smell-sweep result. If the user asked for a written artifact, write it (e.g. `mcp-server-design.md` at repo root or a path they specify); otherwise present it inline.

## Consistency Gate

Run this checklist before declaring the surface specification done. Failures are blocking, not advisory.

- **Every tool has an agent-voice intent statement** describing the effect an agent cares about, not the implementation.
- **Every tool with a side effect declares an idempotency guarantee** from the closed set (no-op-after-first / adds-twice / fails-second-call / requires-claim-lease). "It depends" is not a guarantee.
- **Every tool's return shape has a stated context-budget profile** (bounded / paginated / explicitly-may-be-truncated).
- **Every error envelope tells the agent what to do** across the three classes (retry-safe / retry-with-changes / fatal). A stack trace is none of these.
- **Non-backward-compatible schema changes bump the server capability** visibly in the protocol.
- **Every tool's concurrency contract is stated.** "Undefined" is a defect, not a contract.
- **The tool inventory is derived from the agent workflow, not the host data model.** A CRUD mirror (`get_X` / `update_X` / `delete_X`) fails this check unless the workflow genuinely is CRUD.
- **The smell catalog from [mcp-server-smells.md](../skills/using-mcp-engineering/mcp-server-smells.md) was run as a checklist**, each anti-pattern named and checked.

## Anti-Overconfidence

The first design always *feels* clean. It is not. Tools named after database tables (`get_issue`, `update_issue`, `delete_issue`) look like a coherent CRUD surface and are, almost always, a mis-modeling — the agent wants verbs from the workflow (`claim_issue`, `release_claim`, `close_issue`), not verbs from the data model. Error envelopes copied from a REST API ("400 Bad Request", message="invalid field") look professional and are, almost always, useless to an agent that has no way to know which field. Schemas that "obviously" will not change get re-interpreted by the next model release in ways you could not have anticipated. If the surface you produced is a one-to-one mirror of the host's tables, stop and re-derive from the workflow before running the gate.

## Downstream Handoffs (suggest after completion)

- **Adversarial pre-ship review** — `/review-mcp-server` against the produced specification, to surface the disagreements the architect's own frame cannot see.
- **Tool-catalog-only audit** — `/audit-mcp-tools` once tools are implemented, to re-check names and return shapes against the live surface.
- **Client-side tool-loop and prompt design** — `/llm-specialist` owns when to call these tools and how to recover from their errors in the agent's reasoning.
- **Multi-stage workflow decomposition** — `/procedural-architecture` if the surface expresses a staged procedure (claim → work → close) whose stages need formal structure.
- **Audit-grade telemetry** — `/audit-pipelines` if the tool-call telemetry must carry cryptographic provenance.

## Scope Boundaries

Covered: forward-design of the MCP server surface — tool inventory, primitive selection, output shape, idempotency and concurrency contracts, error envelopes, versioning posture, transport/auth/observability posture.

Not covered: general REST/GraphQL API design (`/web-backend`); client-side prompt/tool-loop design (`/llm-specialist`); whether the project should expose MCP at all (a product decision); implementation of the server runtime (deferred to engineering).
