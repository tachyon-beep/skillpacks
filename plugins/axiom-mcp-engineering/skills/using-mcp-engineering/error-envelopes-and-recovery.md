---
name: error-envelopes-and-recovery
description: Use when an agent gives up after a tool failure, when a tool returns a 500 with a stack trace in the body, when error strings say "Internal server error" or "400 Bad Request: invalid field" with no recovery hint, when the agent retries an error it should not retry (or fails to retry one it should), when you cannot tell whether a failure is the agent's fault or the server's, when input-validation errors crash the connection instead of letting the model self-correct, or when you are auditing whether every error path tells the agent what to do next — retry, change args, or surface to the user.
---

# Error Envelopes and Recovery

## The core asymmetry

A REST error is read once, by a human, while writing client code. That human can read a stack trace, grep the source, ask a colleague, and edit their code to never trigger the error again. None of that is true for an MCP error.

**An MCP error is read on every turn by a model that has no source code, no debugger, no colleague, and exactly one move: decide what to do next from the bytes you returned.** The error string is not a log line. It is a prompt fragment injected into the agent's reasoning. Whatever recovery the agent attempts — retry, fix arguments, give up and tell the user — it derives *entirely* from the shape and content of what you returned. If your error does not encode the recovery, the agent invents one, and the invented recovery is usually wrong: it retries a fatal error forever, or abandons a transient blip that a single retry would have cleared, or hallucinates a different argument because your message said "invalid field" without naming the field.

The discipline of this sheet: **every error your server emits is classified, and every class carries an explicit recovery hint that the agent can act on without guessing.** A stack trace is not a recovery hint. "Internal server error" is not a recovery hint. "400 Bad Request" is not a recovery hint. These are the three default failures this sheet exists to close.

## Two error channels — know which one you are using

MCP has **two distinct ways to signal failure**, and conflating them is the first mistake. As of revision **2025-11-25**, the spec is explicit about the split.

### 1. Protocol errors (JSON-RPC error objects)

These are failures of the *protocol machinery*: malformed JSON-RPC, unknown method, the requested tool does not exist, the server is shutting down. They use the JSON-RPC 2.0 `error` object with standard codes:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "error": {
    "code": -32602,
    "message": "Unknown tool: claim_isue",
    "data": { "did_you_mean": ["claim_issue"] }
  }
}
```

Standard JSON-RPC codes: `-32700` parse error, `-32600` invalid request, `-32601` method not found, `-32602` invalid params, `-32603` internal error. Server-defined codes live in the `-32000` to `-32099` range. A protocol error means **the call never reached your tool's business logic** — the request itself was wrong, or the machinery is broken.

**Critical rule: a protocol error is for the host/client, not primarily for the model.** Many hosts surface a protocol error to the user or abort the turn rather than feeding it back to the model for self-correction. That is exactly right for "you called a tool that does not exist" and exactly wrong for "the issue you tried to claim is already claimed." The latter is a business outcome, not a protocol failure.

### 2. Tool execution errors (`isError: true` in the result)

These are failures of the *tool's work*: the issue was already claimed, the file did not exist, the input validation failed, the upstream API timed out. The call reached your logic and your logic decided "no." These are returned as a **successful JSON-RPC response** whose `result` carries `isError: true`:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "isError": true,
    "content": [
      { "type": "text", "text": "{\"class\":\"retry_with_changes\",\"reason\":\"issue_already_claimed\",\"hint\":\"Issue PROJ-412 is claimed by agent-7 until 2026-06-14T18:00:00Z. Call list_unclaimed_issues to find available work, or call force_release if you have lease authority.\",\"retryable\":false,\"details\":{\"issue_id\":\"PROJ-412\",\"current_holder\":\"agent-7\",\"lease_expires\":\"2026-06-14T18:00:00Z\"}}" }
    ]
  }
}
```

**Why this matters (from the 2025-11-25 spec):** tool execution errors are returned *inside the result* specifically so the model sees them and can self-correct. Input-validation failures in particular are returned as Tool Execution Errors, **not** protocol errors, precisely so the model can read what was wrong and fix its arguments on the next turn. If you raise a `-32602 invalid params` protocol error when the agent passes a malformed date, you have routed a self-correctable mistake into a channel the model may never see. The agent cannot fix what it is not shown.

**The decision rule:**

| Situation | Channel |
| --- | --- |
| Tool does not exist / unknown method | Protocol error (`-32601`) |
| JSON-RPC frame is malformed | Protocol error (`-32700` / `-32600`) |
| Server is shutting down / not initialized | Protocol error (`-32603` or server-defined) |
| Required parameter missing or wrong type | **Tool execution error** (`isError: true`) — model self-corrects |
| Business rule said no (already claimed, not found, conflict) | **Tool execution error** |
| Upstream dependency failed / timed out | **Tool execution error** |
| Auth/permission denied for this specific call | **Tool execution error** with a fatal-class envelope (model cannot fix; surface to user) |

If you are unsure, ask: *does the model have any move that could make this call succeed or recover gracefully?* If yes → tool execution error so the model can see it and act. If no, and the request itself was structurally invalid before reaching your logic → protocol error.

## The three error classes (minimum) — each tells the agent what to do

Every tool execution error MUST be one of three classes. The class *is* the recovery instruction. This is the heart of the discipline and the relevant clause of the MCP Consistency Gate: **every error envelope is one of retry-safe / retry-with-changes / fatal with a recovery hint (a stack trace is none).**

### Class 1: `retry_safe` — transient, retry with the same arguments

The failure was not the agent's fault and was not permanent. A network blip to an upstream service, a momentary lock contention, a rate-limit window, a 503 from a dependency. **The agent's correct move is to retry the identical call** — possibly after a backoff the hint specifies.

```json
{
  "class": "retry_safe",
  "reason": "upstream_timeout",
  "hint": "The issue tracker did not respond within 5s. This is transient. Retry the same call. If it fails 3 times, treat as a fatal outage and tell the user the tracker is unreachable.",
  "retryable": true,
  "retry_after_ms": 2000,
  "max_suggested_retries": 3,
  "details": { "upstream": "tracker-api", "timeout_ms": 5000 }
}
```

The hint does the work the agent cannot do for itself: it says *retry the same call*, it bounds the retries, and it tells the agent what to conclude if retries are exhausted. Without `max_suggested_retries` and the exhaustion instruction, an agent can retry-loop a dead dependency indefinitely. **A `retry_safe` error is only honest if the tool is idempotent under retry** — see `idempotency-and-atomicity.md`. If retrying the same call could double-execute a side effect, the error is not `retry_safe`, no matter how transient the cause; downgrade it or make the tool idempotent first.

### Class 2: `retry_with_changes` — the input was wrong, in a stated way

The agent can recover, but not by retrying as-is. Something about the arguments was wrong, and the hint says *exactly what* and *exactly how to fix it*. This is the class that input validation, business-rule conflicts, and stale-version errors fall into.

```json
{
  "class": "retry_with_changes",
  "reason": "validation_failed",
  "hint": "Parameter 'due_date' must be ISO-8601 (e.g. 2026-06-20). You passed 'next Friday'. Resolve the relative date to an absolute date and retry.",
  "retryable": false,
  "details": {
    "field": "due_date",
    "received": "next Friday",
    "expected_format": "YYYY-MM-DD",
    "example": "2026-06-20"
  }
}
```

Note `retryable: false` — *do not retry as-is*. The recovery is a **changed** call, not a repeated one. The hint names the field, shows what was received, states the expected shape, and gives an example. An agent reading this can fix its arguments deterministically. Compare to the failure this closes: `"400 Bad Request: invalid input"`. That string tells the agent there is *a* problem with *some* field and gives it nothing to act on — so it guesses, and the guess is often a different valid-looking-but-wrong value, burning a turn.

Stale-version conflicts belong here too: "you passed `expected_version: 4` but the issue is now at version 6; re-read the issue and retry with the current version." That is a `retry_with_changes` with a clear, mechanical recovery.

### Class 3: `fatal` — no agent-side recovery; surface to the user

The agent cannot fix this by retrying or by changing arguments. The user lacks permission, the resource was permanently deleted, the operation is structurally impossible, a hard dependency is configured wrong. **The agent's correct move is to stop trying and tell the user**, with the hint as the explanation.

```json
{
  "class": "fatal",
  "reason": "permission_denied",
  "hint": "This server is scoped to project ALPHA. The requested issue belongs to project BETA, which this connection cannot access. Do not retry. Tell the user the server lacks access to project BETA and that they may need to connect a BETA-scoped server.",
  "retryable": false,
  "details": { "requested_project": "BETA", "server_scope": "ALPHA" }
}
```

The hint explicitly says *do not retry* and tells the agent what to communicate. The failure this closes is the un-classed 500: an agent that receives an opaque server error cannot tell `fatal` from `retry_safe`, so it either retries a hopeless call (wasting turns and possibly amplifying side effects) or abandons a recoverable one (giving up on work it could have completed). The class removes the guess.

### Why three is the *minimum*

Three classes cover the agent's three possible moves: **same call again**, **different call**, **stop**. You may add sub-reasons within a class (`upstream_timeout` vs `rate_limited` within `retry_safe`, each with a different `retry_after_ms`), but every error must collapse to one of the three top-level classes, because the agent's decision tree has exactly three branches. An error that does not tell the agent which branch to take has failed its only job.

## The envelope schema

Standardize one envelope across every tool in the server. Heterogeneous error shapes across tools force the agent to parse N formats and are themselves a smell (`error-as-stack-trace` / envelope-inconsistency in `mcp-server-smells.md`). A minimal, sufficient schema:

```json
{
  "class":      "retry_safe | retry_with_changes | fatal",   // REQUIRED — the recovery branch
  "reason":     "machine_token",                              // REQUIRED — stable, greppable, NOT prose
  "hint":       "human/agent-readable recovery instruction",  // REQUIRED — what to DO, in agent voice
  "retryable":  true,                                          // REQUIRED — retry the SAME call?
  "retry_after_ms": 2000,                                      // OPTIONAL — present for retry_safe with backoff
  "max_suggested_retries": 3,                                  // OPTIONAL — bounds a retry loop
  "details":    { }                                            // OPTIONAL — structured, machine-usable specifics
}
```

Rules that make it work:

- **`reason` is a stable token, not prose.** `issue_already_claimed`, not "The issue appears to already be claimed by someone." Tokens let you build golden-conversation regression tests (`testing-mcp-servers.md`) that assert on the reason without brittle string matching, and let observability (`observability-for-tool-calls.md`) count error classes. Prose goes in `hint`.
- **`hint` is in agent voice and says what to *do*.** "Resolve the relative date to an absolute date and retry" is an instruction. "due_date was invalid" is an observation. The agent acts on instructions.
- **`details` is machine-usable.** Field names, received vs expected values, IDs, expiry timestamps. The agent (and your tests) can read structured fields without parsing prose.
- **`retryable` is redundant with `class` but cheap insurance.** Some hosts route on a boolean; some models attend to an explicit flag. State it; do not make the agent infer retryability from the class name.
- **Carry it in the result's text content as JSON** (with `isError: true`), or, if your tool declares an `outputSchema`, in `structuredContent` so the host can validate it. The 2025-11-25 spec supports `structuredContent` alongside the textual `content`; a typed error envelope in `structuredContent` is the strongest form because the host can validate its shape.

## Worked example 1 — Python MCP server, before and after

A `claim_issue` tool, as commonly written first, then corrected.

### Before — the three default failures, all present

```python
# mcp_server.py — DO NOT SHIP THIS
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("issue-tracker")

@mcp.tool()
async def claim_issue(issue_id: str, agent_id: str) -> str:
    """Claims an issue."""                      # implementation-voice intent, not agent-voice
    resp = await httpx.AsyncClient().post(
        f"https://tracker.internal/issues/{issue_id}/claim",
        json={"agent": agent_id},
    )
    resp.raise_for_status()                      # raises -> framework returns a stack trace in the body
    return resp.text                             # success path returns raw upstream blob, unbounded
```

What an agent sees when the issue is already claimed: an exception propagates, and the host surfaces something like:

```
Traceback (most recent call last):
  File "mcp_server.py", line 11, in claim_issue
    resp.raise_for_status()
httpx.HTTPStatusError: Client error '409 Conflict' for url '...'
```

This is **stack-trace-in-body** (failure 1). The agent cannot tell `409 Conflict` from a transient `503`. It has no class, no hint, no recovery branch. It guesses — usually it retries the conflict forever or gives up on the whole task. And on a real `500`, the same code path produces an opaque trace that is **"Internal server error"-equivalent** (failure 2): no recovery information at all.

### After — classified envelopes, recovery hints, correct channels

```python
# mcp_server.py — classified, recoverable
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
import httpx, json

mcp = FastMCP("issue-tracker")

def err(cls: str, reason: str, hint: str, *,
        retryable: bool, retry_after_ms: int | None = None,
        max_retries: int | None = None, **details):
    """One envelope shape for the whole server."""
    body = {"class": cls, "reason": reason, "hint": hint, "retryable": retryable}
    if retry_after_ms is not None:
        body["retry_after_ms"] = retry_after_ms
    if max_retries is not None:
        body["max_suggested_retries"] = max_retries
    if details:
        body["details"] = details
    # isError=True so the MODEL sees it and can self-correct (2025-11-25 semantics)
    return {"isError": True, "content": [TextContent(type="text", text=json.dumps(body))]}

@mcp.tool()
async def claim_issue(issue_id: str, agent_id: str) -> dict:
    """Marks an issue as claimed by this agent so no other agent picks it up.
    Idempotent for the SAME agent_id: re-claiming an issue you already hold is a
    no-op success. Claiming an issue another agent holds fails (retry_with_changes)."""
    if not issue_id or not issue_id.strip():
        return err("retry_with_changes", "missing_issue_id",
                   "Parameter 'issue_id' is required and was empty. "
                   "Call list_unclaimed_issues to get a valid id, then retry.",
                   retryable=False, field="issue_id")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"https://tracker.internal/issues/{issue_id}/claim",
                json={"agent": agent_id},
            )
    except (httpx.ConnectError, httpx.ReadTimeout):
        return err("retry_safe", "upstream_timeout",
                   "The tracker did not respond within 5s. This is transient. "
                   "Retry the same call. After 3 failures, tell the user the tracker is unreachable.",
                   retryable=True, retry_after_ms=2000, max_retries=3,
                   upstream="tracker-api", timeout_ms=5000)

    if resp.status_code == 409:
        holder = resp.json().get("current_holder")
        if holder == agent_id:                                  # idempotent re-claim by same agent
            return {"isError": False, "structuredContent":
                    {"issue_id": issue_id, "claimed_by": agent_id, "status": "already_held_by_you"}}
        return err("retry_with_changes", "issue_already_claimed",
                   f"Issue {issue_id} is held by {holder}. You cannot claim it. "
                   "Call list_unclaimed_issues to find available work, then retry with a different id.",
                   retryable=False, issue_id=issue_id, current_holder=holder)

    if resp.status_code == 403:
        return err("fatal", "permission_denied",
                   f"This connection cannot claim issues in the project owning {issue_id}. "
                   "Do not retry. Tell the user the server lacks access to that project.",
                   retryable=False, issue_id=issue_id)

    if resp.status_code >= 500:
        return err("retry_safe", "upstream_5xx",
                   "The tracker returned a server error. This is usually transient. "
                   "Retry the same call. After 3 failures, tell the user the tracker is degraded.",
                   retryable=True, retry_after_ms=3000, max_retries=3,
                   upstream_status=resp.status_code)

    resp.raise_for_status()
    return {"isError": False, "structuredContent":
            {"issue_id": issue_id, "claimed_by": agent_id, "status": "claimed"}}
```

What changed against the gate:

- The docstring is an **agent-voice intent statement** (the effect the agent cares about) and **declares the idempotency guarantee** (`no-op-after-first` for the same agent). Both are gate clauses.
- Every failure path is **classified** and carries a **hint that names the recovery branch**. No stack trace reaches the agent.
- The 409-by-same-agent path is an **idempotent success**, not an error — retry under network blip after a successful claim does not surface a spurious conflict.
- Channels are correct: missing `issue_id` is a tool execution error (`retry_with_changes`) so the model self-corrects, not a `-32602` protocol error it might never see.

## Worked example 2 — TypeScript server with a typed error envelope and `outputSchema`

The strongest form: the error envelope is part of a declared `outputSchema`, so the host can validate it and the model gets a typed `structuredContent`. JSON Schema **2020-12** is the default dialect as of 2025-11-25.

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { CallToolRequestSchema } from "@modelcontextprotocol/sdk/types.js";

type ErrClass = "retry_safe" | "retry_with_changes" | "fatal";

function envelope(
  cls: ErrClass, reason: string, hint: string,
  retryable: boolean, extra: Record<string, unknown> = {}
) {
  const structured = { ok: false, error: { class: cls, reason, hint, retryable, ...extra } };
  return {
    isError: true,
    content: [{ type: "text", text: JSON.stringify(structured.error) }],
    structuredContent: structured,
  };
}

const server = new Server({ name: "deploy-control", version: "1.2.0" });

// outputSchema lets the host validate the shape — including the error branch.
const promoteOutputSchema = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  type: "object",
  oneOf: [
    { properties: { ok: { const: true }, release: { type: "string" } }, required: ["ok", "release"] },
    {
      properties: {
        ok: { const: false },
        error: {
          type: "object",
          properties: {
            class: { enum: ["retry_safe", "retry_with_changes", "fatal"] },
            reason: { type: "string" },
            hint: { type: "string" },
            retryable: { type: "boolean" },
          },
          required: ["class", "reason", "hint", "retryable"],
        },
      },
      required: ["ok", "error"],
    },
  ],
};

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  if (req.params.name !== "promote_release") {
    // Unknown TOOL is a PROTOCOL error — the call never reaches business logic.
    throw { code: -32601, message: `Unknown tool: ${req.params.name}` };
  }
  const { release, expected_stage } = (req.params.arguments ?? {}) as {
    release?: string; expected_stage?: string;
  };

  if (!release) {
    // Validation failure -> tool execution error, so the model can fix args.
    return envelope("retry_with_changes", "missing_release",
      "Parameter 'release' is required. Call list_releases to get a valid release id, then retry.",
      false, { field: "release" });
  }

  const current = await currentStage(release);   // e.g. throws on network failure
  if (current === null) {
    return envelope("retry_safe", "stage_lookup_timeout",
      "Could not read the current stage within 5s. Transient. Retry the same call; after 3 tries, tell the user the deploy controller is unreachable.",
      true, { retry_after_ms: 2000, max_suggested_retries: 3 });
  }
  if (current !== expected_stage) {
    return envelope("retry_with_changes", "stage_conflict",
      `Release ${release} is at stage '${current}', not '${expected_stage}'. Re-read the release with get_release and retry with expected_stage='${current}'.`,
      false, { release, actual_stage: current, you_sent: expected_stage });
  }
  if (!(await callerMayPromote(req))) {
    return envelope("fatal", "promote_not_authorized",
      "This connection is not authorized to promote releases to production. Do not retry. Tell the user a human with deploy authority must promote.",
      false, { release });
  }

  const promoted = await promote(release);
  return { isError: false, structuredContent: { ok: true, release: promoted }, content: [
    { type: "text", text: `Promoted ${promoted}.` }] };
});
```

This server distinguishes the channels correctly (unknown tool → `-32601` protocol error; everything reachable → typed `isError` envelope), declares an `outputSchema` whose `oneOf` includes the error branch so the host can validate it, and gives every failure a class + actionable hint. The `stage_conflict` hint even tells the agent the exact `expected_stage` value to retry with — a `retry_with_changes` so precise the agent's next call is deterministic.

## Architect view (constructive)

When you design a tool, design its error space *first*, alongside its success shape. For each tool, enumerate:

1. **Every way it can fail**, including the ways its dependencies fail (timeout, 5xx, 4xx, auth, not-found, conflict, validation).
2. **Which class each failure is** — and if the same underlying condition maps to different classes for different callers (a 403 might be `fatal` for one scope but `retry_with_changes` for another), make the distinction explicit.
3. **The recovery hint for each** — what the agent should *do*, written in agent voice, naming any sibling tool that helps (`list_unclaimed_issues`, `get_release`).
4. **Whether `retry_safe` is honest** — only if the tool is idempotent under retry. If not, you have an idempotency problem to fix (`idempotency-and-atomicity.md`) before you can offer a `retry_safe` envelope.

The error envelope is part of the tool's contract, as load-bearing as its parameters. A tool whose error space is undesigned is a tool whose failure behavior the agent will discover at runtime, in production, by guessing.

## Critic view (adversarial)

Audit every tool's error paths, not just its happy path. Findings carry **severity (blocker / major / minor / nit) + evidence** (the specific tool, the response excerpt, the conversation fragment). Look for:

- **Stack trace in the body** → *blocker*. Evidence: the traceback excerpt. The agent cannot recover; this re-creates the exact failure this sheet exists to close.
- **Un-classed error** (`"Internal server error"`, bare `500`, raw upstream message) → *blocker* or *major*. Evidence: the response. The agent cannot tell which of its three moves to make.
- **REST-style message with no field named** (`"400: invalid input"`, `"validation error"`) → *major*. Evidence: the message. A `retry_with_changes` that does not say *what* to change is barely better than a fatal.
- **Wrong channel** — validation failure raised as a `-32602` protocol error the model never sees → *major*. Evidence: the protocol error object. Self-correctable mistakes routed away from the model.
- **`retry_safe` on a non-idempotent tool** → *blocker*. Evidence: the envelope plus the idempotency contract (or its absence). Telling the agent to retry a call that double-executes is an instruction to corrupt state.
- **Inconsistent envelopes across tools** → *minor* to *major* depending on spread. Evidence: two tools' differing error shapes. Forces the agent to parse N formats.
- **Missing retry bound** on a `retry_safe` envelope → *minor*. Evidence: envelope without `max_suggested_retries`. Permits unbounded retry loops against a dead dependency.

A critic pass that returns "errors look fine" without quoting at least the failure-path responses of the highest-traffic tools is rubber-stamping. The architect always writes a clean happy path; the error paths are where the surface rots unobserved.

## Red flags — STOP

If any of these is true, stop and fix the error contract before shipping or signing off:

- **An exception can propagate to the transport.** Any uncaught error becomes a stack trace or opaque 500 in the agent's context. Every tool entry point has a catch-all that emits a classified envelope.
- **An error response has no `class` field.** The agent cannot pick a recovery branch. Non-negotiable.
- **You wrote "Internal server error" / "Something went wrong" / "Unexpected error."** These are non-hints. What should the agent *do*? If you cannot answer, you have not finished the error.
- **A validation error is raised as a protocol error.** The model may never see it and cannot self-correct. Route input/business failures through `isError: true`.
- **A `retry_safe` error sits on a tool that is not idempotent under retry.** You are instructing the agent to duplicate a side effect. Fix idempotency or change the class.
- **Two tools in the same server return differently-shaped errors.** Standardize the envelope.
- **The error message leaks an internal path, a SQL fragment, a stack frame, or a secret.** Not just useless — a disclosure risk. The `hint` is for the agent; internal diagnostics go to your logs (`observability-for-tool-calls.md`), keyed by a correlation id you *can* put in `details`.

## Common mistakes

- **Treating MCP errors like REST errors.** A human reads a REST 400 and edits their code; an agent reads an MCP 400 and must recover *this turn* from the bytes alone. Port the discipline, not the format.
- **Using protocol errors for business failures.** "Already claimed," "not found," "conflict" are tool execution errors (`isError: true`) so the model sees them. Reserve protocol errors for unknown-method / malformed-frame / server-not-ready.
- **Stack trace as the error.** It tells the agent there is a problem and nothing about recovery. It also leaks internals. Log the trace server-side keyed by a correlation id; return a classified envelope.
- **"Internal server error" / opaque 500.** Indistinguishable from transient *and* from fatal. The agent guesses and guesses wrong. Classify it.
- **`retry_with_changes` that does not name the change.** "Invalid field" without saying *which* field, *what* was received, *what* is expected. The agent re-guesses, burns a turn. Name the field; show received vs expected; give an example.
- **No retry bound on `retry_safe`.** The agent retries a dead dependency forever. Always include `max_suggested_retries` and what to conclude on exhaustion.
- **`retry_safe` on a non-idempotent tool.** The most dangerous mistake here — you are telling the agent to corrupt state under retry. Cross-check against `idempotency-and-atomicity.md`.
- **Heterogeneous envelopes across the server.** Every tool inventing its own error shape forces the agent to parse N formats. One envelope, server-wide.
- **Forgetting that the hint is read by a model.** Write it in agent voice as an *instruction* ("retry the same call", "re-read the issue and retry"), naming the sibling tools that help. Not a log line; a move.

## Rationalizations and their counters

- *"It's just a 500, the host will handle retries."* The host does not know if your 500 is transient or permanent — only you do, and only your envelope can tell the agent. An un-classed 500 makes the agent or host guess, and the guess amplifies side effects on non-idempotent tools.
- *"The stack trace helps with debugging."* It helps *your* debugging if it is in *your logs*. In the agent's context it is noise the model cannot act on and a disclosure of your internals. Log it server-side; return an envelope.
- *"The error message is human-readable, that's enough."* The reader is not human. "Invalid date format" is human-readable and agent-useless: it does not name the field, show the received value, or give the expected shape. Readable ≠ actionable-by-a-model.
- *"We'll add structured errors later; ship the happy path now."* The error space is where the surface fails in production, on the median retry, on the model version you have not tested. Shipping the happy path with un-classed errors ships a surface that breaks the first time anything goes wrong — which is the first hour.
- *"Classifying every error is over-engineering for an internal tool."* Three classes is not over-engineering; it is the minimum that matches the agent's three possible moves. Skipping it does not save work — it relocates the work to runtime, where the agent does it badly and you debug it from production transcripts.
- *"The model is smart, it'll figure out the recovery."* The model figures out recovery *from what you return*. Give it a stack trace and it figures out the wrong recovery confidently. The intelligence is downstream of your bytes; do not outsource the contract to the model's guesswork.

## Consistency Gate clause this sheet owns

> Every error envelope is one of **retry_safe / retry_with_changes / fatal** with a recovery hint. **A stack trace is none of these. "Internal server error" is none of these.**

A deliverable that emits any error path without a class and an actionable hint fails this gate clause. The failure is blocking, not advisory. Cross-references: idempotency under retry (`idempotency-and-atomicity.md`) determines whether `retry_safe` is honest; the error-as-stack-trace smell (`mcp-server-smells.md`) is the catalogued anti-pattern; per-call error-class telemetry (`observability-for-tool-calls.md`) proves the classes are emitted as designed; golden-conversation tests (`testing-mcp-servers.md`) assert on `reason` tokens to catch envelope regressions across model versions.
