---
description: Dispatch the failure-scenario-analyst to enumerate what breaks under partition/loss/reorder/clock-skew/slow-node (DESIGN mode), or to attribute an observed anomaly (split-brain, lost write, duplicate processing, stuck saga, retry storm, stale read) to the guarantee that broke (ANOMALY mode). Consolidates a dated report with next actions; does not implement fixes.
allowed-tools: ["Read", "Glob", "Grep", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[design_or_anomaly]"
---

# Analyze Failure Modes Command

You are analysing how a distributed design fails. The output is a *failure-mode report* — either a forward enumeration of the fault scenarios a design is exposed to and its defenses (DESIGN mode), or a backward attribution of an observed anomaly to the first guarantee that broke (ANOMALY mode).

This command does NOT design, scaffold, or implement fixes. For design, use the `using-distributed-systems` skill. For a structured critique of a finished design package, use `/review-distributed-design`. This command's report names *which sheet to read, which numbered artifact to update, and which invariant test to add* — it does not write those changes itself.

## Invocation Path

`/analyze-failure-modes` resolves the input (a design package, or an anomaly description plus telemetry), selects the mode, dispatches the `failure-scenario-analyst` agent, and consolidates a dated report.

## Mode Selection

| Signal | Mode |
|--------|------|
| Argument points at a `distributed-system/` artifact set, an HLD, or "what breaks if…" | DESIGN |
| Argument names a symptom (split-brain, lost write, duplicate, stuck saga, retry storm, stale read) or a paste of an incident/log | ANOMALY |
| Ambiguous | Ask via AskUserQuestion |

```bash
ARG="${1:-}"

if [ -z "${ARG}" ]; then
  # AskUserQuestion:
  # "What are we analysing?
  #    (a) DESIGN — a design or artifact set; enumerate the faults it is exposed to. Give the path.
  #    (b) ANOMALY — an observed symptom; attribute it to the broken guarantee. Describe the symptom + give telemetry."
  :
fi

if [ -d "${ARG}" ] || [ -f "${ARG}" ]; then
  echo "Mode: DESIGN (input is a path)"
  MODE="design"
else
  echo "Mode: ANOMALY (input is a symptom description)"
  MODE="anomaly"
fi
```

## Locate the Spec

Both modes attribute to channels named in the system's `99-distributed-system-specification.md` and its source artifacts. Locate them:

```bash
ls distributed-system/ 2>/dev/null
ls "${ARG}/../distributed-system/" 2>/dev/null
ls "${ARG}/distributed-system/" 2>/dev/null
```

If no spec is available, the agent operates in *spec-inferred* mode: it reconstructs the implied channels from the design or telemetry, at lower confidence, and flags every guarantee it had to infer. An anomaly attributed against an inferred spec is a finding in itself — the system had no named guarantee to break.

## DESIGN Mode Workflow

### Step 1 — Frame the blast radius

Read `00-scope-and-goals.md` for the declared tier (XS/S/M/L/XL). The tier sets which fault classes are in scope and which artifacts are required. Enumerate the fault classes the analyst must exercise:

| Fault class | The question | Primary sheet |
|-------------|--------------|---------------|
| Network partition | Which side keeps serving? Does the other lose writes or go split-brain? | `consistency-models-and-cap.md`, `failure-models-and-fallacies.md` |
| Message loss | What is lost-vs-delivered-but-unacked? Does a retry duplicate? | `delivery-and-ordering-semantics.md`, `idempotency-and-deduplication.md` |
| Reordering | Does out-of-order delivery corrupt state? Is there a causal/ordering guarantee? | `time-clocks-and-ordering.md`, `delivery-and-ordering-semantics.md` |
| Clock skew | Does anything depend on wall-clock agreement (TTLs, leases, last-writer-wins)? | `time-clocks-and-ordering.md` |
| Slow node / gray failure | Does a slow-but-alive node get treated as dead? Does a queue grow unbounded? | `resilience-patterns.md`, `backpressure-and-flow-control.md` |
| Node crash / restart | Is in-flight work lost? Is a partial transaction left dangling? | `replication-and-quorums.md`, `sagas-and-distributed-transactions.md` |
| Leader failure | Is failover safe (no two leaders, no lost committed writes)? | `consensus-and-coordination.md`, `replication-and-quorums.md` |
| Partition imbalance / hot shard | Does one shard tip over and cascade? | `partitioning-and-sharding.md`, `backpressure-and-flow-control.md` |

### Step 2 — Dispatch the failure-scenario-analyst

```
Use the Task tool with subagent_type: "failure-scenario-analyst"
Provide:
  - the design / artifact set path
  - the spec (`99-`) or "spec-inferred mode"
  - the declared tier and the in-scope fault classes
  - "DESIGN mode: for each fault class, enumerate the concrete scenario, the
     guarantee it tests, the defense the design relies on, and whether that
     defense is NAMED+TRACEABLE+COSTED+TESTED or a silent gap."
Receive:
  - a fault-scenario matrix: scenario → exposed guarantee → defense → gap severity
```

### Step 3 — Consolidate the DESIGN report

(See report template below; populate the "Fault-Scenario Matrix" section.)

## ANOMALY Mode Workflow

### Step 1 — Confirm the anomaly is real

Before attributing, sanity-check the symptom against telemetry. A misread dashboard is the most common "anomaly".

| Symptom | Confirmation question | Disconfirming evidence |
|---------|----------------------|------------------------|
| Split-brain | Did two nodes both act as leader/owner in overlapping windows? | One was fenced; the "second leader" was read-only |
| Lost write | Was the write *acked* and then absent, or never acked? | Never acked = not lost, just failed |
| Duplicate processing | Did the same logical operation apply its effect twice? | Two distinct operations; or effect was idempotent (no harm) |
| Stuck saga | Is a saga instance past its deadline with no terminal state? | Still within compensation backoff window |
| Retry storm | Is request volume self-amplifying (retries begetting retries)? | A genuine load spike, not amplification |
| Stale read | Did a read return a value older than an acked write the same client saw? | Read from a replica within declared staleness bound — in-contract |

If the symptom does not survive confirmation, report NOT-AN-ANOMALY and stop: the design is behaving as specified, or the telemetry was misread.

### Step 2 — Localise to the first broken guarantee

The visible symptom is downstream of the break. Walk back to the *first* guarantee that failed — the symptom (e.g. lost write) is usually propagation of an earlier violation (e.g. a quorum that committed on too few replicas, or a failover that lost an unreplicated tail).

| Symptom | Most-likely first broken guarantee | Channel / sheet |
|---------|-----------------------------------|-----------------|
| Split-brain | No fencing / quorum on leadership | `consensus-and-coordination.md` → `04-coordination-spec.md` |
| Lost write | Write acked below durable quorum, or failover dropped unreplicated tail | `replication-and-quorums.md` → `03-replication-spec.md` |
| Duplicate processing | Non-idempotent handler under at-least-once delivery | `idempotency-and-deduplication.md` → `07-idempotency-spec.md` |
| Stuck saga | Missing/failed compensation, or non-idempotent step | `sagas-and-distributed-transactions.md` → `08-transaction-spec.md` |
| Retry storm | No backpressure / no jittered backoff / no circuit breaker | `backpressure-and-flow-control.md`, `resilience-patterns.md` → `11-`, `10-` |
| Stale read | Read consistency weaker than the client's contract; un-scoped "strong consistency" claim | `consistency-models-and-cap.md` → `01-consistency-contract.md` |

These are starting hypotheses, not verdicts — the analyst confirms against the actual telemetry and may attribute elsewhere.

### Step 3 — Dispatch the failure-scenario-analyst

```
Use the Task tool with subagent_type: "failure-scenario-analyst"
Provide:
  - the confirmed symptom and the telemetry / logs / incident notes
  - the spec (`99-`) or "spec-inferred mode"
  - "ANOMALY mode: confirm the symptom is real, walk back to the FIRST broken
     guarantee, attribute it to the channel and its source sheet/artifact,
     and rank alternative attributions by evidence."
Receive:
  - an attribution: symptom → first broken guarantee → channel → confidence,
    with the evidence trail and ranked alternatives
```

### Step 4 — Consolidate the ANOMALY report

(See report template below; populate the "Attribution" section.)

## Report Template

Write to `<input dir or working dir>/failure-mode-analysis-YYYY-MM-DD.md`:

```markdown
# Failure-Mode Analysis Report

- **Analysed**: <design path | anomaly description>
- **Mode**: DESIGN | ANOMALY
- **Spec**: <99- reference, or "spec-inferred">
- **Tier**: <XS/S/M/L/XL from 00->
- **Result**: <DESIGN: N gaps (S0/S1/S2)> | <ANOMALY: ATTRIBUTED to <channel> | UNATTRIBUTED | NOT-AN-ANOMALY>

## Fault-Scenario Matrix        (DESIGN mode)
| Scenario | Exposed guarantee | Defense relied on | NAMED? TRACEABLE? COSTED? TESTED? | Gap severity |
|----------|-------------------|-------------------|-----------------------------------|--------------|
[Output from failure-scenario-analyst]

## Attribution                  (ANOMALY mode)
- **Symptom (confirmed)**: <split-brain | lost write | …>
- **First broken guarantee**: <the guarantee, not the symptom>
- **Channel**: <consistency / replication / coordination / idempotency / delivery / …>
- **Source artifact**: <NN-…-spec.md + section>
- **Evidence trail**: <telemetry that pins the break to this guarantee>
- **Ranked alternatives**: <other channels that could produce this symptom, by evidence>

## Next Actions
- **Sheet to read**: <sibling sheet that owns the broken/missing guarantee>
- **Artifact to update**: <NN-…-spec.md — what must be NAMED/changed>
- **Invariant test to add**: <the property/linearizability/fault-injection test from
  testing-distributed-systems.md that would have caught this; -> 12-test-strategy.md>
- **Re-gate**: <which consistency-gate channels must re-pass after the change>

## Confidence Assessment
[Output from failure-scenario-analyst]

## Risk Assessment
[Output from failure-scenario-analyst]

## Information Gaps
[Output from failure-scenario-analyst]

## Caveats
- This analysis enumerates and attributes; it does not patch.
- Attribution is a hypothesis ranked by evidence; reproduction (often via deterministic-simulation
  testing, see testing-distributed-systems.md) is required to confirm.
- A symptom within the declared contract (e.g. a stale read inside the staleness bound) is
  NOT-AN-ANOMALY — it is the design working as specified.
- This report covers the scenarios/telemetry provided. Other faults may exist through unexercised channels.

## Result Statement
<plain-language summary, 1-3 sentences, suitable for designer / on-call / project lead>
```

## Failure-Mode Handling

| Failure | Action |
|---------|--------|
| `99-` missing | Spec-inferred mode; every inferred guarantee is itself a finding (no named contract to break) |
| Anomaly symptom does not survive confirmation | NOT-AN-ANOMALY; report design-is-as-specified and stop |
| Symptom is in-contract (e.g. stale read inside staleness bound) | NOT-AN-ANOMALY; the contract, not the system, is the surprise — route to `01-` |
| Telemetry insufficient to localise | UNATTRIBUTED; name the missing signal (e.g. no quorum-ack log, no causal trace) under Information Gaps |
| Multiple guarantees plausibly broke | Rank by evidence; name the cheap experiment to discriminate |
| Symptom is a downstream propagation only | Walk back to T₀ guarantee; never name the propagation as the cause |
| Guidance forces an artifact above declared tier | Tier promotion: the artifact becomes required; flag, do not waive |
| Anomaly spans determinism/replay of the cluster | Cross-reference `axiom-determinism-and-replay` for the replay-debug loop |

## Scope Boundaries

Covered: forward fault enumeration with defense/gap assessment (DESIGN), backward attribution of an observed anomaly to the first broken guarantee (ANOMALY), and a dated report naming the next sheet/artifact/test.

Not covered: implementing fixes (downstream — the report names the artifact); a full design critique against all failure modes (use `/review-distributed-design`); broker/event-sourcing mechanics (`axiom-event-driven-architecture`); deployment/rollback of a fix (`axiom-devops-engineering`).

## Downstream Handoffs

- A DESIGN report's gaps feed the `using-distributed-systems` sheets named per row; each gap closes by emitting/updating the cited `NN-` artifact and re-gating.
- An ANOMALY report's "Next Actions" names the sheet to read, the artifact to update, and the invariant test to add (`12-test-strategy.md`); after the change, re-run the consistency gate on the affected channel.
- If reproduction needs a deterministic cluster harness, hand off to `axiom-determinism-and-replay`.
- A consolidated risk picture across many such reports rolls up into `axiom-solution-architect`'s consumption of `99-`.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Reporting the symptom (lost write) as the cause | The cause is the first broken guarantee; the symptom is propagation |
| Attributing without confirming the anomaly is real | Step 1 of ANOMALY mode first; many "anomalies" are in-contract behaviour |
| Treating an in-bound stale read as a bug | Check `01-`'s declared staleness; within bound = NOT-AN-ANOMALY |
| Single-channel attribution when several are plausible | Rank by evidence; name the discriminating experiment |
| DESIGN enumeration that stops at "NAMED" | A defense must be NAMED *and* traceable to the failure model *and* costed *and* tested |
| Enumerating faults out of tier scope as blockers | Tier sets scope; out-of-tier faults are noted, in-tier gaps are findings |
| Writing the fix into the artifact | This command localises and routes; it does not patch |
| Spec-inferred run reported at full confidence | No named spec = lower confidence; flag every inferred guarantee |
| Report not dated/signed | Provenance matters; date-stamp the filename and sign the report |
```