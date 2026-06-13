---
name: testing-mcp-servers
description: Use when an MCP server's test plan is "we asked Claude and it worked once", when a release has no regression coverage for tool descriptions or return shapes, when a new model revision (2025-06-18 → 2025-11-25) re-reads your tools and something silently breaks, when retries double-execute and no test catches it, when MCP Inspector shows green but production agents misuse the surface, when error envelopes are untested, when structuredContent / outputSchema drift goes unnoticed, or when you cannot prove a tool is exercised before cutting a release — golden-conversation regression, replay traces, deterministic harnesses, MCP Inspector, agent-driven eval.
---

# Testing MCP Servers

## Overview

**"We asked Claude and it worked once" is a demo, not a test. An MCP server is read by a non-deterministic client that retries by default and changes its reading of your tools every model revision — so the only thing that proves the surface works is a test that re-runs the exact conditions and fails loudly when the surface regresses. A demo proves the surface worked once, for one model, on one input, with no retry. None of those conditions hold in production.**

This sheet defines the test pyramid for an MCP server and the discipline that keeps it honest. The four layers, cheapest to most expensive:

1. **Deterministic protocol harness** — drive the server with raw JSON-RPC, no model. Fast, hermetic, runs on every commit. Catches schema, envelope, idempotency, pagination, and concurrency bugs.
2. **MCP Inspector** — interactive and CI smoke validation that the server speaks the protocol, lists its primitives, and round-trips a call. Catches "the server doesn't start" and "the schema is malformed."
3. **Golden-conversation regression** — a captured, canonical agent↔server transcript, replayed, asserted. Catches "the tool description now reads differently to the model" and "the return shape changed." **At least one per shipping tool** (Consistency Gate).
4. **Agent-driven eval** — a real model, a task, a rubric, run N times for a pass-rate. Catches "the surface is technically correct but the agent still misuses it." Expensive, non-deterministic, gated to release — not every commit.

The architect reads this sheet to build the harness as the surface is designed: every tool ships with a golden conversation and a protocol-level test the same day it ships. The critic reads the same sheet to audit whether the test suite actually exercises the surface — whether the golden conversations cover retry paths, whether the protocol harness asserts the idempotency guarantee the architect *claimed*, whether the agent-eval pass-rate is reported with a denominator. Same corpus, different epistemics: the architect proves it works, the critic proves the proof is real.

This is calibrated to MCP revision **2025-11-25** (prior widely-deployed: 2025-06-18). The base wire format is JSON-RPC 2.0 over a stateful, capability-negotiated connection; JSON Schema **2020-12** is the default dialect. Tools may return `structuredContent` against a declared `outputSchema`; input-validation failures come back as **Tool Execution Errors** (`isError: true` results), not protocol errors, precisely so the model can self-correct — which means **your error envelopes are part of the testable surface, not an exception path you can ignore.**

## When to Use

Use this sheet when:

- The test strategy for a shipping MCP server is a manual demo against one model — "I asked Claude to do X and it worked."
- A release is being cut with no regression coverage tied to tool descriptions, return shapes, or error envelopes.
- A new protocol revision or model release is imminent and the team cannot say which tools would break if descriptions are re-read.
- Retries are known to occur but no test asserts what happens on the second call.
- MCP Inspector shows the server is healthy but production agents still misuse tools.
- You are the critic and need to judge whether an existing test suite proves the surface works or merely proves it ran once.
- A CI pipeline runs `pytest` but none of the tests touch the MCP boundary.

Do not use this sheet for:

- Testing the *underlying system* the server operates on (the database, the simulation) at full fidelity — that is ordinary application testing; this sheet is about the **MCP surface**.
- Whole-system deterministic replay of the operated system → use `/determinism-and-replay`. Golden-conversation replay of the *MCP server* is in scope here.
- Client-side agent-loop evaluation (does the agent reason well?) → use `/llm-specialist`. This sheet evaluates whether the *server surface* survives a real agent, not whether the agent is good.
- Load/perf testing of the transport → that is performance engineering; this sheet asserts correctness, not throughput.

## Core Principle

> A test for an MCP server is a re-runnable assertion over the surface as a client sees it. If it cannot run unattended, cannot fail, or cannot tell you *which tool* regressed, it is a demo. A demo is allowed to exist; it is not allowed to be the test strategy.

The corollary the rest of this sheet defends: **non-determinism in the client is not an excuse to skip testing — it is the reason to test in layers.** You make the cheap layers deterministic (drive the server directly, no model), and you make the expensive non-deterministic layer (real agent) statistical (a pass-rate over N runs, not a single anecdote). What you never do is let the non-determinism collapse the whole pyramid into one manual demo.

## The Test Pyramid for an MCP Server

```
                  ┌─────────────────────────┐
                  │   agent-driven eval      │   real model, rubric, pass-rate over N
                  │   (release gate, slow)   │   catches: agent still misuses correct surface
                  ├─────────────────────────┤
                  │  golden-conversation     │   captured transcript, replayed, asserted
                  │  regression (per tool)   │   catches: description / return-shape drift
                  ├─────────────────────────┤
                  │     MCP Inspector        │   protocol smoke; lists + round-trips a call
                  │   (CI smoke + manual)    │   catches: malformed schema, won't start
                  ├─────────────────────────┤
                  │  deterministic protocol  │   raw JSON-RPC, no model, hermetic, every commit
                  │  harness (most tests)    │   catches: envelope/idempotency/pagination/concurrency
                  └─────────────────────────┘
```

Most of your assertions live in the bottom layer because it is fast, deterministic, and exercises the contract directly. The pyramid is *inverted* — and broken — when the only "test" is a single agent-driven anecdote at the top and nothing underneath.

### Layer 1 — Deterministic protocol harness

Drive the server over its real transport (stdio or Streamable HTTP) with hand-authored JSON-RPC, asserting on the raw responses. No model in the loop, so it is deterministic and runs on every commit. This is where you assert the Consistency Gate guarantees mechanically:

- **Idempotency**: call a side-effecting tool twice with the same arguments and assert the declared guarantee — `no-op-after-first` means the second result equals the first and the DB row count did not change; `fails-second-call` means the second call returns a `retry-with-changes` envelope; `requires-claim-lease` means the second caller without the lease is rejected.
- **Error envelopes**: drive each declared error path (missing arg, bad arg, conflict, not-found) and assert the result is a Tool Execution Error (`isError: true`) carrying a classified envelope with a recovery hint — not a protocol-level JSON-RPC error, not a stack trace.
- **Return-shape / context-budget profile**: assert bounded tools stay under their token ceiling on a large fixture; assert paginated tools return a cursor; assert `structuredContent` validates against the declared `outputSchema` (2020-12).
- **Concurrency contract**: fire concurrent calls and assert the stated contract holds (serialised-on-key actually serialises; "safe concurrent" produces no interleaving corruption).
- **Capability negotiation**: assert `initialize` advertises the capabilities the surface depends on, and that a non-backward-compatible change bumps what the client sees.

### Layer 2 — MCP Inspector

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) is the official dev tool: launch your server, list tools/resources/prompts, invoke a tool, watch the wire. Use it two ways:

- **Interactively** while developing a tool — does the description render sensibly, does the input schema accept what you expect, does `structuredContent` come back validating against `outputSchema`.
- **In CI as a smoke gate** — `mcp-inspector --cli` mode runs non-interactively: start the server, assert `tools/list` succeeds and returns the expected tool names, invoke one canonical call, assert it round-trips. This catches the "server doesn't start" and "schema is malformed" failures before any heavier test runs.

Inspector is necessary but shallow: it proves the server speaks the protocol, not that the surface is *good*. Treating a green Inspector run as "tested" is the same mistake as the manual demo, one rung up.

### Layer 3 — Golden-conversation regression

A **golden conversation** is a captured, canonical transcript of an agent using a tool to accomplish a realistic task: the assistant's tool-call request (name + arguments) and the server's exact response, frozen as a fixture. The test replays the captured tool calls against the live server and asserts the responses still match (modulo declared-volatile fields like timestamps and IDs, which are normalized).

What this catches that nothing else does:

- A tool **description** edit that changes how the model reads the tool — caught because a re-recorded conversation diverges from the golden, forcing a human to look.
- A **return-shape** change (a renamed field, a newly-nested object, a list that became paginated) — caught as a fixture diff.
- A **schema drift** across protocol/model revisions — when you bless the suite against 2025-11-25 and a future revision re-reads the surface, the golden run flags it.

**At least one golden conversation per shipping tool** is the Consistency Gate line, and it is a floor, not a target. The high-value goldens are the *retry* and *error-recovery* paths: a conversation where the first call times out and the agent retries, asserting the second call honors the idempotency guarantee; a conversation where the agent passes a bad argument, gets a `retry-with-changes` envelope, and recovers on the next turn. Those are the conversations production actually runs and demos never do.

### Layer 4 — Agent-driven eval

Put a real model in front of the server, give it a task, score the outcome against a rubric, and run it N times to get a **pass-rate with a denominator**. This is the only layer that catches "the surface is technically correct but the agent still misuses it" — overlapping tools the agent confuses, a parameter the agent cannot fill, an intent statement that reads ambiguously. It is non-deterministic and slow, so it is a **release gate**, not a per-commit test, and its result is statistical: "18/20 task completions, 2 failures both on the `release_claim`/`close_issue` confusion" is a finding; "I ran it and it worked" is not.

## Example 1 — Deterministic protocol harness over stdio (Python)

A hermetic pytest that spawns the server over stdio, drives it with raw JSON-RPC, and asserts the **idempotency guarantee** and the **error envelope** — no model involved, runs on every commit. Uses the official `mcp` Python SDK client.

```python
# tests/test_protocol_harness.py
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(
    command="python", args=["-m", "issue_server"], env={"ISSUE_DB": ":memory:"},
)

@pytest.fixture
async def session():
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()          # capability negotiation; fails loudly if absent
            yield s

@pytest.mark.anyio
async def test_claim_issue_is_no_op_after_first(session):
    """Declared guarantee: claim_issue is no-op-after-first for the SAME holder.
    This is the assertion the manual demo never makes — the retry path."""
    first = await session.call_tool("claim_issue", {"issue_id": "ISSUE-1", "holder": "agent-a"})
    assert first.isError is False
    assert first.structuredContent["status"] == "claimed"
    lease = first.structuredContent["lease_id"]

    # The retry: same args, same holder. Network blipped, the agent re-sent.
    second = await session.call_tool("claim_issue", {"issue_id": "ISSUE-1", "holder": "agent-a"})
    assert second.isError is False
    assert second.structuredContent["lease_id"] == lease   # SAME lease — no double-claim
    assert second.structuredContent["status"] == "claimed"

@pytest.mark.anyio
async def test_claim_by_second_holder_returns_recoverable_envelope(session):
    """A different holder racing for the same issue must get a CLASSIFIED error
    the agent can act on — not a 500, not a stack trace."""
    await session.call_tool("claim_issue", {"issue_id": "ISSUE-2", "holder": "agent-a"})
    contested = await session.call_tool("claim_issue", {"issue_id": "ISSUE-2", "holder": "agent-b"})

    assert contested.isError is True                       # Tool Execution Error, not protocol error
    env = contested.structuredContent
    assert env["error_class"] == "retry-with-changes"      # NOT "fatal", NOT unclassified
    assert env["held_by"] == "agent-a"                     # the recovery hint: who holds it
    assert "lease expires" in env["recovery_hint"]         # tells the agent what to do next

@pytest.mark.anyio
async def test_list_issues_is_paginated_and_bounded(session):
    """Return-shape / context-budget profile: this tool is declared PAGINATED.
    Seed many rows; assert the first page is bounded and returns a cursor."""
    for i in range(500):
        await session.call_tool("create_issue", {"title": f"issue {i}"})
    page = await session.call_tool("list_issues", {})
    assert len(page.structuredContent["items"]) <= 50      # bounded page
    assert page.structuredContent["next_cursor"] is not None  # cursor present, not the whole table
```

Why each assertion is load-bearing: the first test is the *exact* condition that causes double-execution in production (retry after a blip) and that a demo never reproduces. The second asserts the error is a recoverable, classified envelope — the thing the Consistency Gate requires and a stack trace fails. The third asserts the declared context-budget profile mechanically, so a future change that makes `list_issues` return the whole table fails the build instead of blowing the agent's context in production.

## Example 2 — Golden-conversation regression with a normalizer (Python)

A golden conversation is stored as JSON: the sequence of tool calls and the expected responses. The test replays the calls against the live server and diffs against the golden, normalizing declared-volatile fields. This is the per-tool regression that catches description and return-shape drift.

```jsonc
// tests/golden/claim_then_close.golden.json
{
  "tool": "claim_issue",          // the tool this golden protects (one per shipping tool, min)
  "revision": "2025-11-25",       // protocol revision this was blessed against
  "turns": [
    { "call": "claim_issue", "args": {"issue_id": "ISSUE-7", "holder": "agent-a"},
      "expect": {"isError": false,
                 "structuredContent": {"status": "claimed", "lease_id": "<UUID>", "issue_id": "ISSUE-7"}} },
    // The retry turn — the high-value path. Same args; must return the SAME lease.
    { "call": "claim_issue", "args": {"issue_id": "ISSUE-7", "holder": "agent-a"},
      "expect": {"isError": false,
                 "structuredContent": {"status": "claimed", "lease_id": "<UUID>", "issue_id": "ISSUE-7"}} },
    { "call": "close_issue", "args": {"issue_id": "ISSUE-7", "lease_id": "<UUID>", "resolution": "fixed"},
      "expect": {"isError": false,
                 "structuredContent": {"status": "closed", "issue_id": "ISSUE-7"}} }
  ]
}
```

```python
# tests/test_golden_conversations.py
import json, glob, re
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

UUID_RE = re.compile(r"^[0-9a-f-]{36}$")
GOLDENS = sorted(glob.glob("tests/golden/*.golden.json"))

def normalize(value, expected):
    """Collapse declared-volatile fields to placeholders so they don't cause
    false diffs, WITHOUT hiding real shape changes. <UUID> matches any UUID;
    a missing/renamed field still fails because the KEY set is compared."""
    if expected == "<UUID>":
        assert isinstance(value, str) and UUID_RE.match(value), f"expected a UUID, got {value!r}"
        return "<UUID>"
    if isinstance(expected, dict):
        assert isinstance(value, dict)
        assert set(value) == set(expected), f"shape drift: keys {set(value)} != {set(expected)}"
        return {k: normalize(value[k], expected[k]) for k in expected}
    return value

@pytest.mark.anyio
@pytest.mark.parametrize("path", GOLDENS, ids=[p.split("/")[-1] for p in GOLDENS])
async def test_golden_conversation(path):
    golden = json.load(open(path))
    params = StdioServerParameters(command="python", args=["-m", "issue_server"],
                                   env={"ISSUE_DB": ":memory:"})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            for i, turn in enumerate(golden["turns"]):
                result = await s.call_tool(turn["call"], turn["args"])
                exp = turn["expect"]
                assert result.isError == exp["isError"], f"{path} turn {i}: isError mismatch"
                normalize(result.structuredContent, exp["structuredContent"])  # raises on shape drift
```

The `normalize` function is the discipline that makes goldens survive: it collapses genuinely-volatile fields (UUIDs, timestamps) to placeholders so the test is stable, while comparing the **key set** at every level so a renamed or dropped field — the actual regression you care about — still fails loudly. A golden that ignores the whole response is theatre; a golden that pins volatile fields literally is flaky. The middle path is the one that catches drift.

To **re-bless** after an intentional change: run the suite in record mode (capture live responses to a `*.candidate.json`), diff candidate against golden by eye, and promote only after a human confirms the change is intended. A golden you can regenerate without looking is not a regression test.

## Common Mistakes

- **"We asked Claude and it worked once."** One model, one input, one run, no retry. It proves the surface is not *completely* broken. It proves nothing about the median input, the retry path, the next model revision, or the second tool. Counter: a demo is the *origin* of a golden conversation — capture it, normalize it, assert it, and now it is a test.
- **Green MCP Inspector treated as "tested."** Inspector proves the server speaks the protocol and lists its tools. It does not prove the surface is unambiguous, idempotent, or budget-bounded. It is layer 2 of 4.
- **Goldens that pin volatile fields literally.** Timestamps and generated IDs change every run; a golden that compares them literally is flaky and gets disabled within a week. Normalize declared-volatile fields; compare structure.
- **Goldens that ignore the response body.** The opposite failure: asserting only `isError == false` and discarding the shape. That golden passes through any return-shape regression. Compare the key set at every level.
- **No retry or error path in any golden.** The demo paths are happy paths. Production runs retries and recoveries. A suite that only covers happy paths is blind to the exact failures this pack exists to prevent.
- **Idempotency claimed in the description, never asserted in a test.** The Consistency Gate requires a declared guarantee; a test must *call the tool twice* and assert it. An unasserted guarantee is a comment.
- **Agent-eval reported as an anecdote.** "It worked in eval" with no N, no denominator, no rubric is the manual demo wearing a lab coat. Report pass-rate over N with the failure modes enumerated.
- **Tests that hit the live external system.** A harness that talks to the real database/API is slow, non-hermetic, and flaky. Run the server against an in-memory or substituted backend so the test isolates the *surface*, not the world. (Cross-ref `/determinism-and-replay` external-effects-substitution for the underlying system.)
- **No re-test gate on protocol/model revision.** The surface is read fresh on every revision. Re-running the golden suite against a new revision *before* deploying is the cheapest possible version-drift insurance, and skipping it makes every new model a release candidate by default.

## Red flags — STOP

If any of these is true, the server is not tested and must not be called production-ready:

- The entire test plan is a manual interaction with one model.
- No test in the suite drives the server over its real transport with raw JSON-RPC.
- No tool with a side effect has a test that calls it twice and asserts the idempotency guarantee.
- No error path has a test asserting a classified, recoverable envelope (stack traces / "internal error" reaching the client unasserted).
- There is not at least one golden conversation per shipping tool.
- No golden exercises a retry or error-recovery path.
- Agent-eval results are reported without a denominator (N) or a rubric.
- The suite has never been re-run against the current protocol revision.
- CI runs `pytest` but no test touches the MCP boundary.
- (Critic) A findings list claims "tests pass" without naming which tools are covered and which are not.

## Counters to the rationalizations used to skip this

- *"It's non-deterministic, you can't test an LLM client."* You don't test the client. You test the **surface** deterministically (layers 1–3, no model) and the agent interaction statistically (layer 4, pass-rate). Non-determinism is the argument *for* layering, not against testing.
- *"The MCP Inspector shows it's working."* Inspector is a protocol smoke test. It cannot see ambiguity, idempotency, or budget. Green Inspector + nothing else = one of four layers.
- *"Writing golden conversations is too slow."* The demo you already ran *is* a golden conversation; capturing it is `git add` plus a normalizer you write once. The slow thing is debugging the production incident that a 20-line golden would have caught.
- *"The tools are simple CRUD, they don't need tests."* CRUD-shaped tools are the ones the agent confuses (`update_issue` called twice) and the ones whose return shape grows with the table. Simple-looking is exactly where the retry and budget bugs hide.
- *"We'll add tests after launch."* The surface is read by every model on every turn from launch onward. "After launch" means "after the first incident." The golden suite is cheapest to write the day the tool ships, while the canonical conversation is fresh.
- *"A new model revision won't change how it reads our tools."* It demonstrably does — that is the entire reason the protocol is date-revisioned and the goldens carry a `revision` field. Re-run the suite before you trust a new revision.
- *"It passed once, that's evidence enough."* Passing once is consistent with passing 5% of the time. Without N and a denominator you have an anecdote, and an anecdote is not evidence about a non-deterministic client.

## Critic Checklist

When auditing an MCP server's test suite, produce findings with severity (blocker / major / minor / nit) and evidence (the test file, the missing assertion, the uncovered tool):

- **blocker** — a shipping side-effecting tool with no test that asserts its idempotency guarantee; no protocol-harness test at all; error paths return unasserted stack traces; zero golden conversations.
- **major** — goldens exist but none cover retry/error-recovery; agent-eval reported without N; return-shape budget profile never asserted; suite never re-run against current revision.
- **minor** — goldens pin volatile fields (flaky); Inspector smoke gate missing from CI; concurrency contract claimed but untested.
- **nit** — golden fixtures missing the `revision` field; test names don't identify the tool under test.

Evidence is mandatory: name the tool, cite the test file (or its absence), quote the missing assertion. "Tests look thin" is a vibe, not a finding. And if the architect's claim ("every tool is covered") and the critic's audit agree on a 20-tool surface after a single pass, the audit is theatre — re-run it.
