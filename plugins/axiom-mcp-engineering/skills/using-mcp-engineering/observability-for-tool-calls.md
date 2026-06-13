---
name: observability-for-tool-calls
description: Use when you cannot tell whether a slow or retried tool ran once or four times, when a side effect appears to have happened twice but the logs only show one request, when an idempotency key is enforced but never traced through to whether it suppressed a duplicate, when a dashboard shows tool latency but not retry amplification, when progress notifications and cancellations vanish without a trace, when a deprecated parameter is "removed" but nothing counts how often agents still pass it, or when a post-incident review asks "was that one execution or four?" and the telemetry has no answer.
---

# Observability for Tool Calls

**You cannot operate an MCP server you cannot count. If your telemetry cannot answer "was that one execution or four?" then under retry your tools are running an unknown number of times, and you will only discover the number from the damage.**

The defining operational fact of an MCP server is that its client is a retrying, non-deterministic agent over an unreliable transport. A 40-second tool call that the host gives up waiting on is not a failure the agent observes as "slow" — it is a timeout the agent observes as "no result," and the agent's default behaviour is to call again. The server, meanwhile, may have completed the original call, completed it twice, or be mid-flight on all three. Whether the *effect* happened once or three times depends entirely on the tool's idempotency guarantee (see `idempotency-and-atomicity.md`). Whether *you can tell* which happened depends entirely on this sheet.

General application observability — request count, p99 latency, error rate — is necessary and insufficient here. It counts *requests* and the question is about *executions*. It groups by *endpoint* and the question is about a *logical operation* spread across a retry family. It records *that an error occurred* and the question is about *what the agent did next*. This sheet is the per-call telemetry discipline that closes the specific blindness: **the inability to distinguish one execution from four.**

This sheet serves two readers over the same corpus. The **architect** instruments forward: chooses the correlation identifiers, the span shape, the cardinality budget, the events that make duplicate execution *visible by construction* rather than reconstructable by archaeology. The **critic** reads adversarially: takes a deployed server's telemetry and asks which incidents it could *not* reconstruct — pulls a real slow-call timestamp and tries to answer "one or four?" from what was actually recorded, and files a finding with severity and evidence when the answer is "cannot tell."

## The Core Question, Stated Precisely

"Was that one execution or four?" decomposes into four distinct observable quantities, and a telemetry system that conflates them cannot answer it:

1. **Tool-call attempts** — how many times the host sent a `tools/call` request for this logical operation. This is what the agent did. It is visible at the transport/protocol layer.
2. **Handler entries** — how many times your tool handler function began executing. A request that arrives after the agent gave up still entered the handler. This is visible only if the *server* instruments it.
3. **Effect commits** — how many times the side effect actually committed (row inserted, payment captured, file written). This is the number that matters to the world. It is visible only at the resource boundary (DB, filesystem, downstream API).
4. **Effects the agent observed** — how many results the agent actually received and acted on. A completed effect whose result never reached the agent (transport dropped, host timed out) is an *invisible* execution — the most dangerous kind.

The incident "one execution or four?" is the gap between (1)/(2) and (3). Retry amplification is when (1) > (3) and the tool is idempotent (good — retries were absorbed). Duplicate execution is when (3) > (4) or (3) > 1 for a logically-once operation (bad — the effect ran more than the agent intended). **A telemetry system that records only (1) and a single aggregate latency cannot place an incident anywhere on this scale.** Instrument all four, correlated, or do not claim observability.

## Correlation Identifiers: The Spine

Everything in this sheet hangs off correlation. Three identifiers, each answering a different "same as what?" question:

- **`request_id`** — the JSON-RPC message id. Unique per `tools/call`. Distinguishes attempt 1 from attempt 2. This is the protocol's own id; capture it, do not invent it.
- **`idempotency_key`** — the application-level key the agent (or your client wrapper) supplies to mark "these attempts are the same logical operation." Two requests with the same idempotency key are a *retry family*. This is what lets you collapse four attempts into one logical operation. If the agent cannot supply one, derive a content hash of the meaningful arguments and treat collisions as a retry family (documenting the false-positive risk).
- **`trace_id` / `span_id`** — OpenTelemetry distributed-trace context, propagated from the host if the host supplies it, else minted at handler entry. This stitches *handler → DB → downstream API* into one causal tree so you can see effect commits under the attempt that caused them.

The non-negotiable invariant: **every log line, span, and metric exemplar carries `idempotency_key` (or its derived hash) so that a retry family is queryable as one unit.** Without this, "one or four?" is a manual log-grep correlated by timestamp and arguments — exactly the archaeology this discipline replaces.

```python
# Handler-entry instrumentation: the four quantities, correlated.
# Real tool: an MCP server exposing `create_issue` over a tracker.
import hashlib, json, time
from opentelemetry import trace, metrics

tracer = trace.get_tracer("tracker-mcp")
meter  = metrics.get_meter("tracker-mcp")

# Counters keyed so a retry family is collapsible. Low cardinality on labels:
# tool name and outcome only — NEVER idempotency_key as a label (unbounded).
handler_entries = meter.create_counter("mcp.tool.handler_entries")
effect_commits  = meter.create_counter("mcp.tool.effect_commits")
dup_suppressed  = meter.create_counter("mcp.tool.duplicate_suppressed")

def derive_idempotency_key(args: dict, supplied: str | None) -> tuple[str, bool]:
    if supplied:
        return supplied, True
    # Fallback: content hash of meaningful args. derived=True flags the
    # false-positive risk (two genuinely-distinct calls hashing equal).
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16], False

async def create_issue(args: dict, ctx) -> dict:
    idem_key, supplied = derive_idempotency_key(args, ctx.request_meta.get("idempotencyKey"))
    with tracer.start_as_current_span("tool.create_issue") as span:
        # (2) Handler entry — every entry, including post-timeout retries.
        span.set_attribute("mcp.tool", "create_issue")
        span.set_attribute("mcp.request_id", ctx.request_id)
        span.set_attribute("mcp.idempotency_key", idem_key)
        span.set_attribute("mcp.idempotency_key.supplied", supplied)
        handler_entries.add(1, {"tool": "create_issue"})

        # Idempotency check BEFORE the effect; record suppression as a
        # first-class event so retry-absorption is VISIBLE, not silent.
        existing = await store.find_by_idempotency_key(idem_key)
        if existing is not None:
            span.set_attribute("mcp.idempotency.outcome", "suppressed_duplicate")
            span.add_event("duplicate_suppressed", {"original_request_id": existing.request_id})
            dup_suppressed.add(1, {"tool": "create_issue"})
            return {"issue_id": existing.issue_id, "deduplicated": True}

        # (3) Effect commit — the number that matters to the world.
        issue = await store.insert_issue(args, idem_key=idem_key, request_id=ctx.request_id)
        span.set_attribute("mcp.idempotency.outcome", "committed")
        span.set_attribute("mcp.effect.issue_id", issue.issue_id)
        effect_commits.add(1, {"tool": "create_issue"})
        return {"issue_id": issue.issue_id, "deduplicated": False}
```

With this in place, "one or four?" is a query, not a forensic exercise: count `handler_entries` where `idempotency_key = K` (attempts that ran the handler), count `effect_commits` where `idempotency_key = K` (effects that committed), count `duplicate_suppressed` where `idempotency_key = K` (retries absorbed). Four entries, one commit, three suppressed = "four attempts, one execution, retries correctly absorbed." Four entries, four commits, zero suppressed = "four executions — idempotency is not working, here is your incident."

## The Idempotency-Key Trace: Proving Suppression Worked

Declaring a tool `no-op-after-first` is a contract (Consistency Gate). *Observing that the contract held under real retries* is operations. The gap between them is where silent duplicate execution lives: an idempotency check that has a race window, a key that is scoped too narrowly, a TTL that expired between attempt 1 and attempt 4.

The discipline: **the idempotency check emits a structured event for every outcome — `committed`, `suppressed_duplicate`, `key_expired`, `key_collision_suspected` — and these events are queryable by key.** A dashboard panel titled "duplicate suppression by tool" that shows `suppressed_duplicate` events trending up is your retry-amplification monitor working *as designed*. A panel showing `committed` count exceeding `1` for any single idempotency key is your duplicate-execution alarm.

The trap to avoid: logging "idempotency check passed" with no distinction between "no prior call existed, committed fresh" and "prior call existed, suppressed duplicate." Those are the two halves of the question. Collapsing them re-creates the blindness this sheet exists to remove. Make the outcome an enum on the span and a label-free counter, never a free-text log message.

A subtle, real failure this catches: an idempotency key with a 5-minute TTL on a tool that the host retries after a 6-minute timeout. Attempt 1 commits, key recorded. Attempt 2 arrives at minute 6, key has expired, second commit. Telemetry shows `committed` twice for the same logical operation — `effect_commits` for key K = 2 — and the `key_expired` event names the cause. Without per-key effect counting, this is invisible until a user reports the duplicate.

## Retry Visibility: Counting the Family, Not the Member

Standard APM counts requests and shows you 4× the traffic during a retry storm, attributing it to "load." That framing is wrong for MCP: the four requests are *one* logical operation the agent attempted four times because it could not see the first three results. The right unit is the **retry family** — all attempts sharing an idempotency key (or, when no key is available, all `tools/call` for the same tool with equal arguments within a correlation window).

Two metrics make retry behaviour legible:

- **Retry-amplification ratio** = `handler_entries / distinct(idempotency_key)`, per tool, per time window. A ratio of 1.0 means no retries. A ratio of 3.5 means the average logical operation is hitting your handler 3.5 times. A *spike* in this ratio is the leading indicator of a transport problem, a too-aggressive host timeout, or a tool that is genuinely too slow — and it is invisible to raw request-count dashboards.
- **Result-delivery gap** = `effect_commits − results_acked`. The agent observed `results_acked` results; the world saw `effect_commits` effects. A positive gap means effects committed that the agent never saw — the invisible-execution case. On a non-idempotent tool, a positive gap is a near-certain duplicate-execution incident in progress, because the agent will retry the operation it never saw succeed.

Tie the amplification ratio to your host's actual timeout. If the host abandons calls at 30s and your tool's p95 is 28s, you are one slow Tuesday away from a retry storm on a tool whose p95 is *already inside the timeout's blast radius*. The dashboard panel that pairs "tool p95 latency" with "host abandonment threshold" as a horizontal marker line is worth more than any single latency number, because it shows the *margin*, which is the operationally actionable quantity.

## Progress and Cancellation: The Long-Running Tool

The 2025-11-25 protocol gives long-running tools two facilities that, uninstrumented, become operational dark matter:

- **Progress notifications** (`notifications/progress`, sent against a `progressToken` the client supplies in request metadata) let a tool report incremental progress so the host knows it is alive and need not abandon it. The new experimental **Tasks** utility extends this to durable requests with polling and deferred result retrieval — the same observability obligations apply to a task's lifecycle.
- **Cancellation** (`notifications/cancelled`) lets the host tell the server "stop, the agent no longer wants this." A cancellation that the server *receives but does not act on* is the worst case: the effect commits after the agent has moved on, and the result is never observed — guaranteed invisible execution.

Instrument both as lifecycle events on the call's span:

```python
# Progress + cancellation lifecycle, instrumented so an abandoned-but-still-running
# call is VISIBLE. Real tool: `export_dataset`, a multi-minute export over the
# 2025-11-25 streamable-HTTP transport.
async def export_dataset(args, ctx):
    with tracer.start_as_current_span("tool.export_dataset") as span:
        span.set_attribute("mcp.idempotency_key", ctx.idempotency_key)
        progress_token = ctx.request_meta.get("progressToken")
        span.set_attribute("mcp.progress.token_present", progress_token is not None)
        cancelled = ctx.cancellation_event  # set when notifications/cancelled arrives

        total = await store.count_rows(args["query"])
        for i, chunk in enumerate(store.iter_chunks(args["query"])):
            if cancelled.is_set():
                # CRITICAL: record that we honoured cancellation. A cancellation
                # received but not recorded is indistinguishable from one ignored.
                span.add_event("cancelled_honoured", {"rows_done": i, "rows_total": total})
                span.set_attribute("mcp.outcome", "cancelled")
                await rollback_partial_export(args)   # leave no half-effect
                return {"status": "cancelled", "rows_exported": 0}
            await write_chunk(chunk)
            if progress_token:
                await ctx.report_progress(progress_token, progress=i, total=total)
                span.add_event("progress_sent", {"progress": i, "total": total})

        span.set_attribute("mcp.outcome", "completed")
        return {"status": "complete", "rows_exported": total}
```

The critic's question for any long-running tool: **if the host cancels at 80% and the server keeps going, does the telemetry show it?** If the only record is "call completed," then a cancelled-but-completed call is indistinguishable from a normally-completed one, the agent's intent (stop) was silently overridden, and the partial or full effect is an invisible execution. The `cancelled_honoured` / `cancelled_ignored` distinction must be in the trace, with severity *major* when absent on any tool that has side effects.

## Dashboards That Answer the Question

A dashboard is not "graphs of metrics." It is a set of pre-answered operational questions. The minimum set for an MCP server, each phrased as the question it answers:

| Panel | Answers | Built from |
| --- | --- | --- |
| Executions vs attempts, per tool | "Was that one execution or four?" | `effect_commits` vs `handler_entries` vs `tools/call` count, grouped by `idempotency_key` |
| Duplicate-suppression rate | "Is idempotency actually absorbing retries?" | `duplicate_suppressed` counter, per tool |
| Per-key effect count > 1 alarm | "Did any logical operation commit twice?" | `effect_commits` grouped by `idempotency_key`, alert on max > 1 |
| Retry-amplification ratio + timeout margin | "Are we one slow day from a retry storm?" | `handler_entries / distinct(idempotency_key)`, with host-timeout marker on the latency panel |
| Result-delivery gap | "Did effects commit that the agent never saw?" | `effect_commits − results_acked` |
| Cancellation honour rate | "When the agent cancels, do we stop?" | `cancelled_honoured` vs `cancelled_ignored`, per tool |
| Error-class distribution | "What kind of errors, and are agents recovering?" | error-envelope class (`retry-safe` / `retry-with-changes` / `fatal`) as a label; correlate `retry-safe` errors with subsequent retry families |
| Deprecated-parameter usage | "Can we actually remove this field?" | counter incremented when a deprecated param is non-null on input |

Two cross-cutting rules:

1. **Cardinality discipline.** `idempotency_key`, `request_id`, `trace_id`, and `issue_id` are high-cardinality. They belong on **spans and log lines and metric exemplars**, *never* as metric labels. Metric labels are bounded sets: tool name, outcome enum, error class. Putting `idempotency_key` on a counter label is the classic metrics-explosion outage — it turns observability into the incident. The query "effects per key" runs over traces/exemplars, not over a labelled metric.

2. **Stdio vs HTTP transport.** On **stdio** (local), stdout is the JSON-RPC channel and writing logs there corrupts the protocol stream — the 2025-11-25 spec permits **all log levels to stderr**, so emit telemetry to stderr or an out-of-band sink, never stdout. On **streamable HTTP**, propagate W3C `traceparent` from the incoming request so the host's trace and your server's trace are one tree; and remember the spec requires the server to validate `Origin` and return **HTTP 403** on a bad one — that 403 is a security event and belongs on a panel, distinct from application errors.

## Deprecated-Parameter Tracing: Closing the "We Removed It" Loop

A recurring cross-reference with `schema-versioning-and-drift.md`: a parameter is "deprecated" but the team cannot remove it because nobody knows whether agents still send it. Instrument it. A counter incremented whenever the deprecated parameter is non-null on input, labelled by tool and parameter name (both bounded), turns "we think it's unused" into "zero hits in 30 days across all model versions, safe to remove" or "still 400/day, removal would break callers." This is the observable precondition for a non-backward-compatible schema change — and per the Consistency Gate, such a change bumps the server capability. The telemetry tells you *when* the bump is safe.

## Common Mistakes

- **Counting requests, not executions.** The single most common error: a dashboard with request rate, latency, error rate, and no notion of effect commits or idempotency keys. It looks complete and cannot answer the one question that defines MCP operations. Symptom: an incident review where "one or four?" is settled by guessing.
- **Logging "idempotency check passed" without the outcome.** Collapsing `committed` and `suppressed_duplicate` into one log message destroys the retry-absorption signal. The two halves of the question must be separately countable.
- **Idempotency key as a metric label.** High-cardinality label on a counter → metric explosion → your observability stack is now the outage. Keys go on spans/exemplars; only bounded enums go on labels.
- **No result-delivery accounting.** Measuring effect commits but not whether the agent received the result misses the invisible-execution case entirely — the effect committed, the agent never saw it, the agent will retry. This is precisely how a `no-op-after-first` tool with an expiring key produces a real duplicate.
- **Cancellation received but not recorded.** A cancelled-but-completed call logged as "completed" silently overrides agent intent and hides an effect. Every cancellation must produce a `cancelled_honoured` or `cancelled_ignored` event.
- **Treating retry amplification as load.** Scaling the server in response to a retry storm treats the symptom and amplifies the cause: more capacity means more handlers running the duplicate, especially on non-idempotent tools. The amplification ratio, not raw request rate, is the metric to alert on.
- **Stdout logging on stdio transport.** Writing telemetry to stdout on a stdio server corrupts the JSON-RPC frame stream. Use stderr (all levels permitted per 2025-11-25) or an out-of-band sink.
- **No timeout margin on the dashboard.** Latency panels without the host-abandonment threshold marked show you a number but not the margin. The margin is what predicts the next retry storm; the bare number does not.
- **Trace context dropped at the handler boundary.** Not propagating `traceparent` from an HTTP request, or not minting a span at handler entry, breaks the causal chain from attempt → effect → downstream call. You then have spans you cannot correlate and are back to timestamp archaeology.

## Critic Checklist (severity + evidence required)

Apply against a deployed server's actual telemetry, not its design doc:

- Pull a real slow-call timestamp from the last 30 days. Reconstruct "one execution or four?" *using only what was recorded.* If you cannot, severity **blocker** for any tool with irreversible side effects, **major** otherwise; evidence is the timestamp and the missing quantity (e.g., "no per-key effect_commits counter").
- For each side-effecting tool, confirm the idempotency outcome is a queryable enum, not a log string. Missing → **major**; evidence is the log line that conflates committed and suppressed.
- Confirm no high-cardinality identifier is a metric label. Violation → **major** (metrics-explosion risk); evidence is the counter definition.
- For each long-running tool, confirm cancellation produces a distinct honoured/ignored event. Missing on a side-effecting tool → **major**; evidence is the span shape on a cancelled call.
- Confirm a result-delivery gap is computable (effects vs acks). Not computable → **major** for non-idempotent tools; evidence is the absence of an ack counter.
- Confirm the latency dashboard marks the host abandonment threshold. Missing → **minor**; evidence is the panel.
- Confirm deprecated parameters have usage counters before any removal is scheduled. Removal scheduled without the counter → **blocker**; evidence is the change ticket and the absent metric.

A clean pass on a server that cannot reconstruct a single real slow-call incident is rubber-stamping. Find the incident the telemetry cannot explain, or assume you have not looked.
