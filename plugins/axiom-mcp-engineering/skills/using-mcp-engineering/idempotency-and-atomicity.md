---
name: idempotency-and-atomicity
description: Use when an MCP tool double-executes after a network blip, a timeout, or an agent retry; when two concurrent tool calls corrupt shared state, lost-update or interleave; when you cannot answer "was that one execution or four?"; when a side-effecting tool has no idempotency guarantee, no idempotency key, no expected_version, no claim lease; when retries amplify side effects (charges twice, posts twice, increments twice); when a tool needs exactly-once-under-retry semantics or a stated concurrency contract.
---

# Idempotency and Atomicity

## Overview

**An MCP tool is called by a client that retries by default, on a transport that can lose the response after the side effect has already happened, possibly concurrently with another agent calling the same tool. If a side-effecting tool does not have an explicit idempotency guarantee and an explicit concurrency contract, it has bugs you have not seen yet — not the absence of bugs.**

The defining asymmetry of MCP versus a human-authored API: a human client retries occasionally and notices when something double-fires. An LLM host retries *as a matter of course* — on a timeout, on a dropped stream, on a `retry-safe` error envelope it was told to retry, sometimes on its own reasoning ("that didn't seem to work, let me try again"). Under MCP revision **2025-11-25**, Streamable HTTP GET streams support resumption and **servers may disconnect at will**; a tool result delivered over a stream that drops before the client acknowledges it is, from the client's perspective, indistinguishable from a tool that never ran. The client's only safe move is to retry. Your tool's only safe move is to make that retry harmless.

This sheet closes two failure modes precisely:

1. **Retry-after-network-blip double-execution.** The tool ran, performed its side effect, and the response was lost before the client saw it. The client retries with identical arguments. Without an idempotency mechanism, the side effect happens twice.
2. **Concurrent calls corrupting state.** Two agents (or one agent on two turns, or one agent the host parallelised) call tools that read-modify-write the same state. Without a concurrency mechanism, you get lost updates, torn writes, or interleaved partial states.

Both are *contract* problems before they are *implementation* problems. The Consistency Gate (from the router) requires that **every side-effecting tool declares an idempotency guarantee** (one of `no-op-after-first` / `adds-twice` / `fails-second` / `requires-claim-lease`) and **every tool states a concurrency contract**. This sheet gives you the patterns to back those declarations and the discipline to refuse to ship without them.

## The agent-voice intent comes first

Before any idempotency mechanism, state the tool's intent in **agent voice** — the effect an agent cares about, not the implementation. The idempotency guarantee is *about that effect*, and you cannot reason about "calling this twice" until you have named what "once" accomplishes.

- Implementation voice: "Inserts a row into the `payments` table." → Twice = two rows. So what? The agent does not know what a row is.
- Agent voice: "Charges the customer for order O." → Twice = charged twice. Now the idempotency requirement is obvious and load-bearing.

The intent statement is the input to the idempotency decision. Write it first.

## The four idempotency guarantees (pick exactly one per side-effecting tool)

Every side-effecting tool's declaration is one of these. "It depends" is not on the list; if it depends, you have two tools or an unmodeled parameter.

| Guarantee | Meaning | Mechanism | Concurrency note |
| --- | --- | --- | --- |
| **`no-op-after-first`** | Calling twice with the same args leaves the same final state as calling once. The canonical safe default. | Idempotency key, or naturally-idempotent write (`SET status = 'closed'`), or upsert keyed on a stable identity. | Usually safe to call concurrently with itself *if* keyed; the key serialises. |
| **`adds-twice`** | Each call appends/increments. Non-idempotent by design (e.g. "post a comment"). The agent must coordinate; the server must say so. | None server-side; declare it loudly so the host does not blind-retry. Offer a client-supplied idempotency key to *let* callers dedupe. | Concurrent calls each add; if order matters, that is a separate contract. |
| **`fails-second`** | First call succeeds; an identical second call returns a `retry-with-changes` or `fatal` envelope (uniqueness-protected). | Unique constraint on a business key; the second insert violates it and you translate the violation into a structured envelope. | The DB constraint *is* the concurrency control; exactly one writer wins. |
| **`requires-claim-lease`** | The effect is "take exclusive ownership of X for a bounded time." Idempotency and concurrency are the same mechanism: the lease. | Claim-lease pattern (below): conditional acquire, fenced token, TTL, explicit release. | The lease *is* the concurrency contract. |

Write the chosen word into the tool's manifest entry. A critic reading the manifest must be able to find it without reading the source.

## Pattern 1 — Idempotency keys (the default for `no-op-after-first` and `fails-second`)

An idempotency key is a client-chosen (or client-passed-through) opaque string that identifies *this logical operation*. The server records, atomically, "I have processed key K and here is the result." A retry with the same key returns the *recorded result* without re-running the side effect.

Three rules that are routinely gotten wrong:

1. **The key must be supplied by the caller, not generated by the server.** A server-generated key changes on every retry and is useless. Expose an `idempotency_key` parameter (or accept the JSON-RPC request `id` as a fallback — but the request id is *not* stable across host-level retries that re-issue the request, so prefer an explicit parameter).
2. **The dedup record and the side effect must commit in the same transaction** (or be linked by a saga/outbox). If you write the side effect, then crash before recording the key, the retry re-runs it. If you record the key, then crash before the side effect, you return success for work that never happened. Both halves, one commit.
3. **Store and return the original result**, not just a "duplicate" flag. The agent on retry wants the same structured answer it would have gotten the first time, so its reasoning continues unbroken.

### Example 1 — idempotent charge tool, Python MCP server (SQLite-backed)

A `charge_order` tool. Intent (agent voice): *"Charges the customer for order O so the order can be fulfilled."* Guarantee: **`no-op-after-first`**. Concurrency contract: *"Safe to call concurrently with itself; the idempotency key is serialised by a unique index, so exactly one charge occurs per (order_id, idempotency_key)."*

```python
# server.py — uses the official `mcp` Python SDK (FastMCP), MCP revision 2025-11-25.
import sqlite3, json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("billing")
db = sqlite3.connect("billing.db", isolation_level=None)  # explicit txn control
db.execute("""
  CREATE TABLE IF NOT EXISTS idempotency (
    key         TEXT PRIMARY KEY,          -- caller-supplied, globally unique
    tool        TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
  )""")
db.execute("""
  CREATE TABLE IF NOT EXISTS charges (
    order_id TEXT PRIMARY KEY,             -- one live charge per order (business rule)
    amount_cents INTEGER NOT NULL,
    state TEXT NOT NULL
  )""")

def _error(kind: str, message: str, recovery: str) -> dict:
    # Structured error envelope — see error-envelopes-and-recovery.md.
    # kind is one of: retry-safe / retry-with-changes / fatal
    return {"isError": True, "error": {"kind": kind, "message": message, "recovery": recovery}}

@mcp.tool()
def charge_order(order_id: str, amount_cents: int, idempotency_key: str) -> dict:
    """Charge the customer for an order so it can be fulfilled.

    Idempotency: no-op-after-first. Re-calling with the same idempotency_key
    returns the original charge result and does NOT charge again.
    Concurrency: safe to call concurrently with itself; serialised on idempotency_key.
    """
    db.execute("BEGIN IMMEDIATE")  # acquire write lock up front; no read-then-write race
    try:
        row = db.execute(
            "SELECT result_json FROM idempotency WHERE key = ?", (idempotency_key,)
        ).fetchone()
        if row is not None:                 # retry path — return recorded result, do NOT re-charge
            db.execute("COMMIT")
            return json.loads(row[0])

        existing = db.execute(
            "SELECT state FROM charges WHERE order_id = ?", (order_id,)
        ).fetchone()
        if existing is not None:
            db.execute("ROLLBACK")
            return _error(
                "retry-with-changes",
                f"Order {order_id} already has a charge in state {existing[0]}.",
                "Do not retry this charge. Check order state with get_order before charging.",
            )

        # --- the actual side effect ---
        db.execute(
            "INSERT INTO charges(order_id, amount_cents, state) VALUES (?, ?, 'captured')",
            (order_id, amount_cents),
        )
        result = {"order_id": order_id, "amount_cents": amount_cents, "state": "captured"}
        # dedup record + side effect commit TOGETHER
        db.execute(
            "INSERT INTO idempotency(key, tool, result_json) VALUES (?, 'charge_order', ?)",
            (idempotency_key, json.dumps(result)),
        )
        db.execute("COMMIT")
        return result
    except sqlite3.IntegrityError:
        # Two concurrent calls with the same key: one wins the unique index, the loser lands here.
        db.execute("ROLLBACK")
        winner = db.execute(
            "SELECT result_json FROM idempotency WHERE key = ?", (idempotency_key,)
        ).fetchone()
        if winner:
            return json.loads(winner[0])    # converge on the winner's result
        return _error("retry-safe", "Transient contention.", "Retry with the same idempotency_key.")
    except Exception:                        # noqa: BLE001 — never leak a traceback to the agent
        db.execute("ROLLBACK")
        return _error("retry-safe", "Charge did not complete.",
                      "Retry with the same idempotency_key; it is safe and will not double-charge.")
```

What this buys: a blind retry after a lost response returns the *same* result and charges nothing extra (`no-op-after-first`). Two concurrent calls with the same key converge on one charge (the `IntegrityError` branch). Two calls with *different* keys for the same order hit the `charges.order_id` PRIMARY KEY and the second gets a `retry-with-changes` envelope telling the agent exactly what to do. No path leaks a stack trace.

## Pattern 2 — Optimistic locking with `expected_version` (the default for read-modify-write tools)

When a tool's effect is "change X based on its current value," two concurrent callers can lost-update each other: both read version 5, both compute from it, both write, the second silently clobbers the first. The fix is **optimistic concurrency control**: the agent passes the version it based its decision on; the write succeeds only if the stored version still matches; otherwise the tool returns a `retry-with-changes` envelope carrying the *current* state so the agent can re-decide.

This is strictly better than a blind partial-update tool for agent clients, because the agent *has* the version (it read the record on a prior turn) and the failure is recoverable by the agent itself — exactly the recovery contract the host needs.

### Example 2 — `update_issue` with `expected_version`, TypeScript MCP server

Intent (agent voice): *"Update an issue's fields so the change reflects the agent's current understanding."* Guarantee: **`fails-second`** for a stale write (the second writer with an old version is rejected, not silently merged). Concurrency contract: *"Read-modify-write protected by optimistic locking on `expected_version`; concurrent writers do not lost-update — the loser receives the current state and must re-decide."*

```typescript
// server.ts — official @modelcontextprotocol/sdk, MCP revision 2025-11-25.
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { db } from "./db.js"; // assume a small wrapper over your store

const server = new McpServer({ name: "tracker", version: "2.0.0" });

server.registerTool(
  "update_issue",
  {
    title: "Update issue",
    description:
      "Update an issue's fields so the change reflects your current understanding. " +
      "You MUST pass expected_version (the version you last read). " +
      "Idempotency: fails-second on a stale version — if someone else changed the issue " +
      "since you read it, this returns the current state and does NOT apply your change. " +
      "Concurrency: optimistic locking on expected_version; concurrent writers never lost-update.",
    inputSchema: {
      issue_id: z.string(),
      expected_version: z.number().int().describe("The version field from your last read of this issue."),
      fields: z.object({ status: z.string().optional(), assignee: z.string().optional() }),
    },
  },
  async ({ issue_id, expected_version, fields }) => {
    // Single atomic conditional write. The WHERE clause is the lock.
    const updated = await db.run(
      `UPDATE issues
          SET status = COALESCE(?, status),
              assignee = COALESCE(?, assignee),
              version = version + 1
        WHERE id = ? AND version = ?`,
      [fields.status ?? null, fields.assignee ?? null, issue_id, expected_version],
    );

    if (updated.changes === 1) {
      const row = await db.get(`SELECT * FROM issues WHERE id = ?`, [issue_id]);
      return { structuredContent: { ...row } }; // success: agent sees the new version
    }

    // changes === 0: either the issue is gone, or the version moved on.
    const current = await db.get(`SELECT * FROM issues WHERE id = ?`, [issue_id]);
    if (!current) {
      return {
        isError: true,
        content: [{ type: "text", text: JSON.stringify({
          kind: "fatal",
          message: `Issue ${issue_id} does not exist.`,
          recovery: "Do not retry. List issues with list_issues to find a valid id.",
        })}],
      };
    }
    // Stale write — hand back current state so the agent can re-decide. retry-with-changes.
    return {
      isError: true,
      content: [{ type: "text", text: JSON.stringify({
        kind: "retry-with-changes",
        message: `Stale update: expected version ${expected_version}, current is ${current.version}.`,
        recovery: `Re-read the issue (provided below), reconcile your change against it, ` +
                  `then call update_issue again with expected_version=${current.version}.`,
        current_state: current,
      })}],
      structuredContent: { current_state: current },
    };
  },
);
```

Why `expected_version` and not a server-side merge: the agent is the only party that knows *why* it wanted the change. A silent merge ("apply my status, keep their assignee") can produce a state neither writer intended. Returning the current state and forcing the agent to re-decide keeps the human-meaningful invariant intact and is recoverable on the agent side — the host gets a `retry-with-changes` envelope, which is exactly what it is built to act on.

## Pattern 3 — Claim leases (for `requires-claim-lease`)

When the effect is "take exclusive ownership of X so no one else works on it" — claiming an issue, locking a deploy slot, reserving a worker — idempotency and concurrency collapse into one mechanism: the lease. A claim lease is a **conditional acquire** (succeeds only if unheld or held by you), a **TTL** (so a crashed holder does not block forever), a **fencing token** (so a revived stale holder cannot act after its lease expired), and an **explicit release**.

```python
@mcp.tool()
def claim_issue(issue_id: str, agent_id: str, lease_seconds: int = 300) -> dict:
    """Take exclusive ownership of an issue so other agents do not work on it concurrently.

    Idempotency: requires-claim-lease. Re-calling while you already hold the lease EXTENDS it
    and returns the same fencing token (no-op-ish). Calling while another agent holds it FAILS.
    Concurrency: the lease IS the concurrency contract — at most one holder at a time.
    """
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute(
            "SELECT holder, expires_at, fence FROM leases WHERE issue_id = ?", (issue_id,)
        ).fetchone()
        now = db.execute("SELECT datetime('now')").fetchone()[0]

        if row and row[0] != agent_id and row[1] > now:
            db.execute("ROLLBACK")
            return _error("retry-safe",
                f"Issue {issue_id} is claimed by {row[0]} until {row[1]} (UTC).",
                "Work a different issue, or retry after the lease expires.")

        fence = (row[2] + 1) if row else 1   # monotonically increasing fencing token
        db.execute("""
            INSERT INTO leases(issue_id, holder, expires_at, fence)
            VALUES (?, ?, datetime('now', ? || ' seconds'), ?)
            ON CONFLICT(issue_id) DO UPDATE
              SET holder=excluded.holder, expires_at=excluded.expires_at, fence=excluded.fence
        """, (issue_id, agent_id, str(lease_seconds), fence))
        db.execute("COMMIT")
        return {"issue_id": issue_id, "holder": agent_id, "fence": fence, "lease_seconds": lease_seconds}
    except Exception:
        db.execute("ROLLBACK")
        return _error("retry-safe", "Could not acquire lease.", "Retry; acquisition is safe to repeat.")
```

The **fencing token** is the non-obvious part. A holder whose lease expired (it stalled for 6 minutes on a 5-minute lease) might wake up and try to commit work. Downstream writes must carry the fence and the target must reject any write whose fence is lower than the highest it has seen. Without fencing, TTL leases are not safe — they only *reduce* the race window. State in the concurrency contract whether downstream writes are fenced; an unfenced lease is a `major` critic finding, not a clean one.

## State a concurrency contract per tool

The router's gate is explicit: **"Undefined" is a defect, not a contract.** A concurrency contract answers three questions in one or two sentences, in the tool description:

1. **Safe to call concurrently with itself?** (e.g. "yes, serialised on `idempotency_key`" / "yes, optimistic-locked on `expected_version`" / "no, undefined behaviour — host must serialise.")
2. **Safe to call concurrently with related tools?** (e.g. "`close_issue` and `update_issue` on the same id race; close wins via the same version check.")
3. **What is the isolation boundary?** (the row, the issue id, the whole table, global.) The agent and the critic both need to know the granularity at which two calls can collide.

Put it in the description so a model with no source code can read it, *and* in the manifest entry so the critic can audit it without running the server.

## Common mistakes

- **Server-generated idempotency keys.** A UUID minted inside the handler changes on every retry and deduplicates nothing. The key must come from the caller and be stable across that caller's retries.
- **Recording the dedup key and performing the side effect in separate transactions.** A crash between them gives you either re-execution (side effect first, key lost) or phantom success (key first, side effect lost). One commit, or an outbox/saga.
- **`SELECT ... ; if exists return; else INSERT` without a lock or unique constraint.** Two concurrent calls both read "not exists," both insert. The check-then-act is a race. Use `BEGIN IMMEDIATE` / `SELECT ... FOR UPDATE` / a unique index that turns the second insert into a catchable violation.
- **Treating optimistic-lock failure as a 500.** A stale `expected_version` is `retry-with-changes`, not an internal error. Return the current state so the agent re-decides; that is the whole point.
- **TTL leases without fencing tokens.** TTL alone narrows the race window; it does not close it. A revived stale holder will corrupt state. Fence every downstream write.
- **Declaring `no-op-after-first` for an operation that is actually `adds-twice`.** "Post a comment" is not idempotent just because you wish it were. Either add a real idempotency key or declare `adds-twice` and let the host stop blind-retrying.
- **Idempotency record table that grows forever.** Keys need a retention/TTL policy or the dedup table becomes the next incident. State the retention window; ensure expiry is longer than the maximum plausible retry horizon.
- **Relying on the JSON-RPC request `id` as the idempotency key.** Host-level retries may re-issue the logical request with a *new* id. The id deduplicates within one connection's framing, not across a reconnect. Use an explicit business-level key.
- **Stating the concurrency contract in a code comment instead of the tool description.** The model that decides whether to parallelise calls reads the description, not your source. The critic auditing the manifest reads the manifest, not your source.

## Red flags — STOP

If any of these are true, stop and fix before shipping:

- A side-effecting tool's manifest entry has **no idempotency guarantee word** (`no-op-after-first` / `adds-twice` / `fails-second` / `requires-claim-lease`).
- A tool's concurrency contract is **"undefined," blank, or absent.**
- You **cannot answer "if the agent retries this after a timeout, what happens?"** with a one-word guarantee.
- The dedup record and the side effect are in **different transactions / different services with no outbox.**
- A read-modify-write tool has **no `expected_version`** (or equivalent CAS) and no other serialisation.
- A lease has a **TTL but no fencing token**, and downstream writes do not check the fence.
- The idempotency key is **generated server-side** or is **the JSON-RPC request id.**
- A stale-version or already-done outcome returns a **stack trace or "Internal Server Error"** instead of a `retry-with-changes` / `fatal` envelope with a recovery hint.
- There is **no golden-conversation regression test that exercises the retry path** (call → simulate lost response → re-call → assert single effect). A retry-safety claim with no replay test is unproven.

## Counters to the rationalizations used to skip this

- *"The transport is reliable; responses don't get lost."* MCP 2025-11-25 Streamable HTTP servers **may disconnect at will** and GET streams support resumption *precisely because* delivery is not guaranteed. The host retries by design. Reliability of the happy path does not remove the retry.
- *"The agent won't call it twice."* The host's retry policy is not yours to assume. Hosts retry on timeout, on dropped streams, on `retry-safe` envelopes you yourself emit, and sometimes on the model's own initiative. "Won't" is a hope; "harmless if it does" is an engineering property.
- *"We've never seen a double-execution in testing."* You have never seen the median-input retry-after-blip in a deterministic test, because your test transport does not drop responses. Absence in a happy-path harness is not evidence of safety; it is evidence the harness does not exercise the failure. Add the lost-response replay test (router sheet 11).
- *"Concurrency isn't a problem; only one agent uses this."* Today. Multi-agent hosts, host-side parallel tool calls, and the same agent acting on two turns all produce concurrency. "One caller" is a deployment assumption, not a tool contract — and the gate requires a contract.
- *"Idempotency keys are overkill for this internal tool."* The cost of a key is one parameter, one unique index, one transaction. The cost of a double-charge / double-post / double-deploy in agent context is an incident the agent cannot diagnose because, from its side, it only called once.
- *"We'll add the version check later when it matters."* It already matters the first time two turns touch the same record, and "later" lands after the lost-update incident. Optimistic locking is cheaper to add at design time than to retrofit under an inconsistency you cannot reproduce.
- *"The database transaction handles it."* A transaction makes a *single call* atomic. It does nothing about a *second call* (the retry) — that is a new transaction. Atomicity within a call is necessary and not sufficient; you also need idempotency across calls.

## For the critic

When auditing for this sheet, produce **finding / severity / evidence** triples (per the SME Agent Protocol and the router's gate):

- **blocker** — a side-effecting tool with a real-money / irreversible effect (charge, deploy, delete, external post) and no idempotency mechanism. Evidence: the handler source showing an unconditional side effect, plus the absent guarantee word in the manifest.
- **major** — read-modify-write tool with no `expected_version` / CAS; TTL lease without fencing; dedup record and side effect in separate transactions; concurrency contract stated as "undefined."
- **minor** — idempotency guarantee present but not surfaced in the tool *description* (only in a comment); idempotency table with no retention policy.
- **nit** — guarantee word present but phrased ambiguously; concurrency contract correct but missing the isolation-granularity sentence.

Evidence is the tool name, the parameter list, the handler excerpt, or a retry trace — never a vibe. If the architect declared `no-op-after-first` and you can exhibit an input ordering that produces two effects, record the disagreement with both positions and the resolution; do not silently downgrade. A retry-safety claim with no golden-conversation regression test that simulates a lost response is, at minimum, a `major` "unproven claim" finding regardless of how the code reads.

## Related sheets

- [error-envelopes-and-recovery.md](error-envelopes-and-recovery.md) — the `retry-safe` / `retry-with-changes` / `fatal` envelope classes this sheet emits. Idempotency and error classification are two halves of the same retry contract.
- [output-shape-and-pagination.md](output-shape-and-pagination.md) — the recorded result you return on a deduped retry must still respect the context-budget profile.
- [observability-for-tool-calls.md](observability-for-tool-calls.md) — how to actually answer "was that one execution or four?": idempotency-key tracing, retry visibility, duplicate-execution surfacing.
- [testing-mcp-servers.md](testing-mcp-servers.md) — the golden-conversation regression that simulates a lost response and asserts a single effect. A retry-safety claim is unproven without it.
- [transport-reliability.md](transport-reliability.md) — why the response can be lost in the first place (Streamable HTTP disconnect-at-will, stdio framing, reconnection); the source of the retries this sheet defends against.
