---
name: transport-reliability
description: Use when stdio bytes are dropped, interleaved, or corrupted; when a slow agent client stalls the server or the server stalls writing; when an MCP session reconnects mid-conversation and the server has forgotten what it was doing; when SSE streams die and the agent never sees the rest of a long tool result; when you cannot tell whether to use stdio or Streamable HTTP, whether the legacy HTTP+SSE dual-endpoint transport still applies, what session state survives a reconnect, how event-ID resumption works, or how to apply backpressure without deadlocking. Framing, transport selection, reconnection, resumability, and session-state recovery for an MCP server.
---

# Transport Reliability

## Overview

**The transport is the only layer where an MCP server can lose data that the application layer will never see go missing.** A dropped byte on stdio does not raise an exception — it desynchronizes the JSON-RPC framing, and every subsequent message is garbage. A reconnect does not announce "I have forgotten everything" — the server keeps a session object in memory that the client's new connection knows nothing about. A slow client does not send an error — it just stops reading, your write buffer fills, and your server blocks inside a tool handler holding a database lock. These are not application bugs. They are transport-layer contract violations that manifest three layers up as "the agent got confused" or "the tool hung."

This sheet covers the two transports the current spec defines — **stdio** (local) and **Streamable HTTP** (remote) — the **deprecated HTTP+SSE dual-endpoint transport** it replaced and why you will still meet it, the framing rules that keep bytes aligned, the reconnection and resumability model, the precise question of **what session state a server may assume across a reconnect**, and backpressure under slow clients. The two failure modes it exists to close are **dropped/desynchronized bytes** and **lost session state on reconnect** — both silent, both blamed on the wrong layer.

Currency: written to MCP protocol revision **2025-11-25** (prior widely-deployed revision **2025-06-18**; original transport spec **2024-11-05**). Base wire format is JSON-RPC 2.0. Streamable HTTP is the current HTTP transport; HTTP+SSE is deprecated but not extinct.

## When to Use

- stdout on a stdio server is emitting log lines, progress bars, or `print()` debug output, and the client's JSON-RPC parser is choking.
- Messages arrive truncated, concatenated, or interleaved; a large tool result corrupts the next request.
- A tool handler hangs and you suspect the client stopped reading (backpressure / write-blocked).
- An agent reconnects mid-conversation (network blip, host restart, tab reload) and the server behaves as if the prior session never happened — or worse, as if it is still mid-operation.
- A long-running streamed result (progress notifications, large paginated payload) dies partway and the agent never receives the tail.
- You are choosing a transport and need the decision rule: stdio vs Streamable HTTP, and whether legacy HTTP+SSE matters for your deployment.
- You are deciding what to persist server-side so a reconnect is recoverable, and what the spec actually lets you assume survives.

Do not use this sheet for: tool naming/shape ([tool-api-design.md](tool-api-design.md)); retry-safety of side effects ([idempotency-and-atomicity.md](idempotency-and-atomicity.md) — transport reconnect and tool idempotency interact, but the *guarantee* lives there); OAuth/authorization (HTTP-only, see [authentication-and-trust.md](authentication-and-trust.md)); per-call telemetry ([observability-for-tool-calls.md](observability-for-tool-calls.md)).

## Core Principle

> The transport guarantees framed, ordered delivery *within a single connection*, and nothing across connections. Every byte your server writes to a JSON-RPC channel must be a newline-delimited JSON-RPC message and nothing else. Every piece of state your server keeps in a session object is state it will lose on the next reconnect unless you have explicitly made it recoverable. A transport that "usually works" is a transport that has not yet met a slow client or a dropped connection.

## 1. stdio framing — the channel discipline that has no error path

stdio is the default local transport. The client launches the server as a subprocess and speaks JSON-RPC over the child's **stdin (client→server) and stdout (server→client)**. The framing contract is exact:

- Messages are **newline-delimited JSON** (one JSON-RPC message per line; the message itself must contain no embedded literal newlines — serialize compactly).
- **stdout carries protocol messages only.** Nothing else. Not a banner, not a progress bar, not a stray `print()`, not a library's startup chatter.
- **stderr is for logging — all levels, not just errors** (this is explicit in 2025-11-25; earlier mental models that "stderr = errors only" are wrong). The client may capture, forward, or ignore stderr, but it is never parsed as protocol.
- stdio implementations **SHOULD NOT use the OAuth flow**; pull credentials from the environment instead.

The dangerous property: **there is no framing error path.** If anything non-protocol reaches stdout, the client's line parser does not raise "malformed frame" — it tries to `JSON.parse` a log line, fails or mis-parses, and the stream is now desynchronized. Every subsequent message is misaligned. The symptom appears far from the cause: "the third tool call returned nothing" when the real fault was a `print("debug")` two calls earlier.

**The single most common stdio bug is a library or dependency writing to stdout.** A logging config that defaults to stdout, a `print()` left in a handler, a C extension that writes a warning, an `import` that prints a deprecation notice — any of these silently poison the channel. You must redirect everything to stderr at process start, before any other code runs.

### Example: Python stdio server with stdout protected (real tools)

This uses the official `mcp` Python SDK. **The protection that actually holds is not a clever redirect — it is the floor: all logging to stderr, every dependency audited for stdout writes, and a framing test that fails CI if anything but the SDK reaches stdout.** That is what the example below leads with, because it is the form you can copy and ship.

The reason the floor is the right default — and the reason the "reassign `sys.stdout` to stderr" trick is a trap with this SDK — is mechanical: `mcp.server.stdio.stdio_server()` captures the protocol channel by **dup-ing the current stdout file descriptor at call time** (`stdout_fd = os.dup(sys.stdout.fileno())` in `src/mcp/server/stdio.py`). If you have already pointed `sys.stdout` at stderr before that call, the SDK dups *stderr's* fd and writes the entire JSON-RPC protocol to stderr — the channel you tried to protect is now poisoned, silently, by the protection itself. So: do **not** reassign `sys.stdout` before `stdio_server()`. Keep fd 1 as the protocol channel and keep everything else off it.

```python
# server.py — stdio MCP server with a protected channel.
# Discipline (in priority order): (1) ALL logging to stderr, (2) audit every dependency
# for stdout writes, (3) a framing test (below) that fails if anything but the SDK hits stdout.
# Do NOT reassign sys.stdout: stdio_server() dups the CURRENT stdout fd at call time, so any
# pre-call reassignment redirects the protocol itself, not your stray prints.
import sys
import logging

logging.basicConfig(
    stream=sys.stderr,            # all log levels to stderr — never stdout
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("issue-mcp")

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

app = Server("issue-mcp")

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.ContentBlock]:
    # Agent-voice intent (claim_issue): "reserve an issue so no other agent picks it up."
    # Idempotency: requires-claim-lease (see idempotency-and-atomicity.md). Concurrency:
    # serialised on issue_id. This sheet's concern is only that the RESULT reaches the client intact.
    if name == "claim_issue":
        issue_id = arguments["issue_id"]
        log.info("claim_issue id=%s", issue_id)        # -> stderr, safe
        # print(f"claimed {issue_id}")                 # <- would CORRUPT the channel; never do this
        claimed = await _claim(issue_id, lease_owner=arguments["agent_id"])
        # Bounded return shape (output-shape-and-pagination.md): one small object.
        return [types.TextContent(type="text", text=_json(claimed))]
    raise ValueError(f"unknown tool: {name}")

async def main():
    # stdio_server() dups fd 1 HERE. fd 1 must still be the real stdout at this point.
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

if __name__ == "__main__":
    anyio.run(main)
```

> If you genuinely must neutralize a dependency that writes to fd 1 directly (a C extension, not Python-level `print`), do not touch `sys.stdout` — preserve the protocol channel at the fd level *before* anything can write to it, and hand that saved descriptor to the SDK rather than letting it dup a reassigned stream:
>
> ```python
> import os, sys
> # Verified pattern: snapshot the REAL stdout fd, then move fd 1 to point at stderr so
> # rogue native writes to fd 1 land on stderr. Reattach the saved fd as sys.stdout so
> # stdio_server() dups the protocol channel — not stderr — when it calls os.dup(fileno()).
> _real_fd = os.dup(1)              # save the true protocol channel
> os.dup2(2, 1)                     # any direct fd-1 write now goes to stderr (safe)
> sys.stdout = os.fdopen(_real_fd, "w", buffering=1)  # SDK will dup THIS at call time
> ```
>
> This is an escape hatch, not the default. The dependency-proof floor — logging to stderr, audited dependencies, and the framing test below — protects the channel without any fd surgery, and it is what almost every server should ship. The official SDK exposes no `MCP_STDOUT_FD` env var or stream override; the only lever is *which fd `sys.stdout` names at the moment `stdio_server()` runs.*

### Framing test you can actually run

A golden-conversation harness ([testing-mcp-servers.md](testing-mcp-servers.md)) catches this. Minimum viable framing assertion:

```bash
# Feed one initialize + one tool call, capture stdout, assert every line parses as JSON-RPC.
printf '%s\n%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"claim_issue","arguments":{"issue_id":"X-1","agent_id":"a"}}}' \
  | python server.py 2>/dev/null \
  | python -c 'import sys,json;[json.loads(l) for l in sys.stdin if l.strip()];print("FRAMING OK")'
```

If a dependency prints to stdout, this fails with a `JSONDecodeError` — which is exactly the desync the agent would otherwise hit silently.

## 2. Streamable HTTP — the current remote transport

Streamable HTTP (introduced in 2025-03-26, current in 2025-11-25) replaced the original **HTTP+SSE dual-endpoint** transport. The shape:

- **A single endpoint** (e.g. `POST/GET /mcp`). The client `POST`s a JSON-RPC message. The server replies either with a single JSON response (`Content-Type: application/json`) or, when it needs to stream (progress notifications, server-initiated requests, multiple messages), upgrades the response to **Server-Sent Events** (`Content-Type: text/event-stream`).
- **GET** opens a long-lived SSE stream for server→client messages outside a request/response pair (notifications, sampling requests back to the host).
- **Sessions** are identified by an `Mcp-Session-Id` header the server issues on `initialize`; the client echoes it on every subsequent request.
- **Origin validation is mandatory.** The server **MUST validate the `Origin` header and return HTTP 403** on an invalid Origin — this is the DNS-rebinding defense for locally-bound servers. Bind to `127.0.0.1`, not `0.0.0.0`, for local HTTP servers.
- **Resumability** is built on SSE event IDs: the server attaches an `id:` to streamed events; if the stream drops, the client reconnects with `Last-Event-ID` and the server replays from after that point. Event IDs encode stream identity, so the server may disconnect a GET stream at will and the client can poll/resume.

Contrast with the **deprecated HTTP+SSE** transport you will still encounter in older servers/clients: it used **two endpoints** — a GET `/sse` to open the stream and a separate POST endpoint (advertised via an initial `endpoint` event) for client→server messages. It is deprecated, brittle under reconnection (no clean resumability model), and you should not build new servers on it. You support it only for backward compatibility with old clients, and even then a compatibility shim is preferable to a parallel code path.

### Example: Streamable HTTP server with Origin validation and resumable streaming (Node/TS)

```typescript
// streamable-http.ts — current transport, with the two non-negotiables:
//   (1) Origin validation -> 403, and (2) event-ID resumability for long results.
import express from "express";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { randomUUID } from "node:crypto";

const ALLOWED_ORIGINS = new Set([
  "https://app.example.com",
  "http://127.0.0.1:3000",
]);

const server = new McpServer({ name: "issue-mcp", version: "1.4.0" });

// Tool: scan_open_issues. Agent-voice intent: "list issues an agent could claim now."
// Return shape: PAGINATED (output-shape-and-pagination.md) — bounded page + cursor. The
// transport concern below is that a long stream of progress can resume if the SSE drops.
server.registerTool(
  "scan_open_issues",
  {
    description: "List open, unclaimed issues the agent can claim (one bounded page).",
    inputSchema: { project: { type: "string" }, cursor: { type: "string", nullable: true } },
  },
  async ({ project, cursor }, { sendNotification }) => {
    const page = await listOpenIssues(project, cursor, 50);
    for (const [i, issue] of page.items.entries()) {
      // Progress notifications stream over SSE; each carries an event id so a mid-scan
      // disconnect can resume from Last-Event-ID instead of re-emitting the whole scan.
      await sendNotification({
        method: "notifications/progress",
        params: { progress: i + 1, total: page.items.length, message: issue.id },
      });
    }
    return { content: [{ type: "text", text: JSON.stringify({ items: page.items, nextCursor: page.nextCursor }) }] };
  },
);

const app = express();
app.use(express.json());

// (1) Origin validation FIRST. Invalid Origin -> 403. This is the spec-mandated
//     DNS-rebinding defense; skipping it is a security blocker, not a nicety.
app.use("/mcp", (req, res, next) => {
  const origin = req.get("origin");
  if (origin !== undefined && !ALLOWED_ORIGINS.has(origin)) {
    res.status(403).json({ error: "invalid_origin" });
    return;
  }
  next();
});

// Session registry: maps Mcp-Session-Id -> transport. This map IS the server's session
// state. Section 4 is about what survives when an entry is recreated on reconnect.
const transports = new Map<string, StreamableHTTPServerTransport>();

app.all("/mcp", async (req, res) => {
  const sid = req.get("mcp-session-id");
  let transport = sid ? transports.get(sid) : undefined;
  if (!transport) {
    transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      // enableResumability + an event store let dropped SSE streams replay from Last-Event-ID.
      // Without an event store, a dropped stream loses every event after the break — silently.
      onsessioninitialized: (id) => transports.set(id, transport!),
    });
    await server.connect(transport);
  }
  await transport.handleRequest(req, res, req.body);
});

app.listen(3000, "127.0.0.1"); // bind to loopback for local servers, never 0.0.0.0
```

The two lines that prevent silent data loss: **Origin → 403** (security; without it a malicious web page can drive your local server) and the **event store / resumability** (without it, every dropped SSE stream loses its tail and the agent sees a truncated result with no error).

## 3. Reconnection and resumability — the model

There are two distinct "reconnect" events and the server must handle them differently:

| Event | What dropped | What the client does | What the server must do |
|---|---|---|---|
| **Stream resume** | A single SSE stream broke mid-flight (same session) | Reconnects the GET/stream with `Last-Event-ID` | Replay buffered events after that ID from the event store; if no store, the tail is lost |
| **Session reconnect** | The whole connection/session ended (subprocess relaunch, `Mcp-Session-Id` gone, host restart) | Re-runs `initialize`, gets a **new** session | Treat as a fresh session; recover any durable state from the backing store, never from the dead in-memory session object |

The hard rule: **resumability within a session is the transport's job (event IDs); recovery across sessions is the application's job (durable state).** Conflating them is the source of the second failure mode.

stdio has no resumability layer — if the subprocess dies, the session is gone and the client relaunches it as a brand-new process with empty memory. Plan accordingly: a stdio server's in-memory session state is *always* lost on restart.

## 4. What session state the server may assume across a reconnect

**The default answer is: none.** This is the discipline.

A new connection — even one presenting a previously-issued `Mcp-Session-Id` that the server no longer recognizes — must be safe to treat as fresh. Concretely:

- **In-memory session objects are not durable.** The `transports` map above, an in-process cache of "issues this agent claimed," an open transaction, a half-built result — all gone on session reconnect. If the server crashed and restarted, the map is empty.
- **A reconnecting client may present a stale `Mcp-Session-Id`.** The server MAY honor it if it still holds that session; it MAY return **HTTP 404** to signal "that session is gone, re-initialize." It must never assume the in-memory state behind that ID still exists.
- **Mid-operation state must live in a durable store, keyed by something the client can re-present** (issue ID, claim ID, idempotency key) — not by the session ID, which the client cannot reconstruct after a session reconnect.
- **Anything the agent must be able to resume after a reconnect must be reconstructable from durable, client-presentable keys.** "Which issues did this agent claim?" must be answerable from the database (`SELECT issue_id WHERE lease_owner = :agent_id`), not from a session-scoped set.

This is where transport reliability and [idempotency-and-atomicity.md](idempotency-and-atomicity.md) interlock. A claim lease keyed by `agent_id` in the database survives a reconnect; a claim tracked in a session-scoped Python set does not. After a reconnect the agent may legitimately retry the operation that was in flight when the connection dropped — and that retry must be safe, which is the idempotency guarantee, not a transport feature. **The transport gives you no exactly-once delivery across a reconnect; the only thing that makes reconnect-then-retry safe is the tool's declared idempotency.**

### Anti-pattern, concretely

```python
# WRONG: session-scoped state. Lost on every reconnect; the agent's claims "vanish."
class Session:
    def __init__(self):
        self.claimed = set()          # in-memory, per-connection — NOT durable

# RIGHT: durable, keyed by a client-presentable key. Survives reconnect and crash.
async def claim_issue(issue_id: str, agent_id: str):
    # Lease row in the DB; idempotency = requires-claim-lease. After a reconnect the agent
    # re-presents (issue_id, agent_id); we can tell it already holds the lease -> no-op-after-first.
    return await db.upsert_lease(issue_id, owner=agent_id, expires_in="10m")

async def my_open_claims(agent_id: str):
    # Reconstructable after reconnect from a durable, client-presentable key.
    return await db.list_leases(owner=agent_id)
```

## 5. Backpressure under slow clients

A slow client is one that reads from the channel slower than the server writes. Under stdio this is a full OS pipe buffer; under HTTP/SSE it is a full socket send buffer plus the framework's response buffer. When the buffer fills, the server's write **blocks** (or, in an async runtime, awaits indefinitely).

The danger: the blocking write is usually inside a tool handler that holds resources — an open transaction, a row lock, a claimed lease. A slow client thus reaches across the transport and stalls your data layer. One stalled agent can hold a lock that blocks every other agent. This is how "the client's terminal scrolled slowly" becomes "the whole server deadlocked."

Disciplines:

- **Never hold a lock across a streaming write.** Acquire → mutate → commit/release → *then* stream the result. The result write must not be in the critical section.
- **Bound every send.** Use a write timeout (HTTP) or a bounded queue with a drop/close policy (SSE). A client that cannot keep up gets its slow stream closed (resumable via `Last-Event-ID`), not the privilege of stalling the server.
- **Bound the producer to the consumer.** When generating progress notifications in a loop, await each send so a backed-up channel slows production rather than growing an unbounded in-memory buffer (which trades a stall for an OOM). The TS example in §2 awaits each `sendNotification` for exactly this reason.
- **Prefer pagination over streaming for large results** ([output-shape-and-pagination.md](output-shape-and-pagination.md)). A bounded page with a cursor has no backpressure surface; a 10k-row stream does. Stream only when the agent genuinely needs incremental delivery.
- **Set per-request timeouts.** A handler that has awaited a write for N seconds should fail with a *retry-safe* error envelope ([error-envelopes-and-recovery.md](error-envelopes-and-recovery.md)), releasing its resources, rather than awaiting forever.

## Common mistakes

- **Writing anything but protocol to stdout.** A `print()`, a progress bar, a library banner, a logging config that defaults to stdout. Symptom: parser desync, "later tool calls return nothing." Fix: all logging to stderr; audit dependencies; assert framing in tests.
- **Treating session state as durable.** Storing claims/transactions/results in an in-memory session object and assuming a reconnecting agent still has them. Symptom: "the agent's work vanished after a network blip." Fix: durable store keyed by client-presentable keys.
- **No resumability event store on Streamable HTTP.** Streaming a long result over SSE with no event IDs / store. Symptom: dropped stream → truncated result, no error. Fix: enable resumability with an event store; client resumes via `Last-Event-ID`.
- **Skipping Origin validation.** Serving Streamable HTTP without the mandated `Origin` check / 403, often bound to `0.0.0.0`. Symptom: any web page can drive your local server (DNS rebinding). Fix: validate Origin → 403; bind to `127.0.0.1`.
- **Holding a lock across a streaming write.** Tool acquires a lease, then streams the result while holding it. Symptom: one slow client deadlocks all agents. Fix: commit/release before streaming; bound sends.
- **Building new servers on legacy HTTP+SSE.** Using the deprecated dual-endpoint transport for new work. Symptom: brittle reconnection, no clean resumability. Fix: Streamable HTTP; treat HTTP+SSE as backward-compat only.
- **Assuming reconnect = exactly-once.** Believing the transport will not re-deliver an in-flight operation after a reconnect. Symptom: double execution under reconnect-then-retry. Fix: declare and enforce tool idempotency ([idempotency-and-atomicity.md](idempotency-and-atomicity.md)) — the transport guarantees nothing across connections.
- **Embedded newlines in stdio messages.** Pretty-printed JSON or multi-line strings written to stdout. Symptom: one logical message parsed as several broken frames. Fix: compact, single-line serialization per message.

## Red flags — STOP

If any of these is true, stop and fix it before shipping:

- There is a `print()`, `console.log`, `fmt.Println`, or default-stdout logger anywhere reachable from a stdio server's request path — **STOP**, the channel is poisoned.
- The server keeps "what this agent is doing" only in an in-memory session object — **STOP**, a reconnect erases it.
- A Streamable HTTP server has no `Origin` validation or is bound to `0.0.0.0` — **STOP**, this is an open DNS-rebinding hole.
- A long streamed result has no event-ID resumability path — **STOP**, every dropped stream silently truncates.
- A tool streams its result while holding a lock, lease, or open transaction — **STOP**, one slow client can deadlock the server.
- You are relying on the transport to make a reconnect-then-retry safe instead of on the tool's idempotency guarantee — **STOP**, the transport gives you nothing across connections.
- You are building new functionality on the deprecated HTTP+SSE dual-endpoint transport — **STOP**, use Streamable HTTP.
- No test asserts that a tool call emits *only* well-formed JSON-RPC frames on stdout — **STOP**, your framing is unverified.

## Counters to the rationalizations

- *"It's just a debug print, I'll remove it later."* On stdio there is no "just" a print — it desynchronizes the channel and corrupts every later message. There is no error; you will debug the wrong call. Remove it now, and add the framing test so the next one can't ship.
- *"Reconnects basically never happen."* They happen on every host restart, every laptop sleep, every network blip, every tab reload, every deploy. "Never" means "not in my five-minute test." The agent population over a week hits it constantly.
- *"The session object is fine, the client always keeps its session."* The client cannot keep a session your process lost on crash, and a session reconnect issues a *new* session by design. Durable, client-presentable keys are the only thing that survives.
- *"Origin validation is a browser thing, my server is local."* Local is exactly the threat model — a malicious web page in the user's browser can POST to `127.0.0.1`. The spec mandates the check and the 403 precisely for local servers.
- *"Streaming is fine, clients read fast."* Until one doesn't, and your write blocks inside a critical section, and the lock it holds blocks every other agent. Bound your sends; the cost of assuming a fast client is a server-wide deadlock.
- *"Resumability is over-engineering for a single result."* A truncated tool result with no error is worse than a failure: the agent reasons over a partial payload as if it were complete. The event store is cheap; a silently-wrong agent is not.
- *"HTTP+SSE still works, why migrate?"* It works until a reconnect, which it has no clean model for. You are choosing the transport the spec deprecated for reliability reasons; new work pays that debt for no benefit.

## For the architect

When you design the transport layer, your deliverable states, per server: **(a)** transport (stdio / Streamable HTTP, and any HTTP+SSE backward-compat); **(b)** for stdio, the stdout-protection mechanism and the framing test that enforces it; **(c)** for HTTP, the Origin allowlist + 403 behavior and the bind address; **(d)** the resumability story (event store yes/no; what a dropped stream does); **(e)** the durable-state inventory — every piece of state that must survive a reconnect, its backing store, and the client-presentable key it is recoverable by; **(f)** the backpressure policy — send bounds, what happens to a slow client, and proof that no lock is held across a streaming write. An architect who cannot name what survives a reconnect has not designed the transport; they have deferred a production incident.

## For the critic

Audit the transport against the corpus, **severity + evidence** per finding ([mcp-server-smells.md](mcp-server-smells.md), §smells):

- **blocker** — stdout reachable by non-protocol writes on a stdio server (cite the line); missing Origin validation / `0.0.0.0` bind on HTTP (cite config); lock/lease held across a streaming write (cite the handler); reconnect-then-retry safety resting on the transport rather than declared idempotency (cite the tool + its idempotency clause).
- **major** — session-only state with no durable backing (cite the field and the reconnect scenario that loses it); long stream with no resumability (cite the tool and the truncation it produces); new feature on deprecated HTTP+SSE.
- **minor** — pretty-printed/multi-line stdout JSON; missing per-request write timeout; no framing test.
- **nit** — logging level config that *could* regress to stdout; bind address documented but not asserted.

Evidence is a code line, a config value, a session/reconnect scenario, or a stream-drop trace — not "transport looks fragile." A clean transport audit on a server with an in-memory `transports` map and no event store is the critic reading the surface the way the architect wrote it; re-run with the reconnect and slow-client scenarios explicitly applied.
