---
description: Drive the distributed-system design workflow — declare tier, walk the required sheets, emit the numbered artifact set, run the consistency gate
allowed-tools: ["Read", "Write", "Glob", "Grep", "Bash", "Task", "AskUserQuestion", "Skill"]
argument-hint: "[system_or_brief]"
---

# Design Distributed System Command

You are driving the distributed-system design workflow for a system (or a hardening pass on an
existing one). The output is a numbered `distributed-system/` artifact set — one spec per correctness
channel — governed by a Consistency Gate. This command does NOT implement code, does NOT pick the
consistency contract for the user (it *elicits* it), and does NOT do deployment, ops, or single-service
API design. It produces the design-of-record that downstream packs consume.

## Invocation Path

`/design-distributed-system` is a Claude Code slash command. It orchestrates the
`using-distributed-systems` router: it establishes scope and tier, walks the tier-required reference
sheets in dependency order, optionally dispatches the `distributed-design-reviewer` agent as a gap
pre-pass, consolidates the `99-distributed-system-specification.md`, and runs the Consistency Gate,
recording pass/fail/waiver per check.

For a focused single-channel design pass without the full workflow, invoke the relevant sheet from the
`using-distributed-systems` skill directly (e.g. `replication-and-quorums.md`). Architecture risk
consolidation across the whole solution belongs to `/axiom-solution-architect`, which consumes the
`99-` spec produced here.

## Preconditions

The command takes a single argument: a system name, a path to a system directory, or a free-text design
brief.

### Resolve the argument

```bash
INPUT="${ARGUMENTS}"

if [ -z "${INPUT}" ]; then
  # Use AskUserQuestion to collect:
  # "What system is this for? Provide a name (e.g., 'order-fulfilment'),
  #  a path (e.g., 'services/orders/'), or a one-paragraph brief describing
  #  the nodes, the data, and the SLAs."
  :
fi

# Path → verify directory; name/brief → treat as scope seed
if [ -d "${INPUT}" ]; then
  echo "Designing for existing system at: ${INPUT}"
elif [ -f "${INPUT}" ]; then
  echo "ERROR: ${INPUT} is a file. Provide a directory, a system name, or a brief."
  exit 1
else
  echo "Treating as system name / brief: ${INPUT}"
  echo "Will create distributed-system/ for the artifact set."
fi
```

### Check for an existing distributed-system workspace

```bash
ls distributed-system/ 2>/dev/null
```

**Resume vs fresh protocol:**

If `distributed-system/` already holds artifacts (00–13, 99), ask the user via AskUserQuestion:

1. **Augment** — fill in artifacts missing for the declared tier; leave existing ones alone.
2. **Replace** — archive existing `distributed-system/` to `distributed-system.bak.YYYY-MM-DD/`, start fresh.
3. **Targeted** — (re-)emit a single channel's spec (e.g. just `03-replication-spec.md`), re-run only the
   affected gate checks, leave the rest.
4. **Promote** — the system has outgrown its tier; re-declare the tier, then emit the newly-required
   artifacts (tier promotion, never a waiver — see Step 1).

If no workspace exists, proceed without asking.

## Workflow

### Step 1 — Establish scope and declare the tier → `00-scope-and-goals.md`

This is router-owned. Elicit and record, in `00-scope-and-goals.md`:

- The system, its nodes/services, the data each owns, and the trust boundary (single-org? cross-org?).
- The SLAs that matter: availability target, latency budget, durability/RPO/RTO, consistency expectations
  **per data class** (not globally).
- The **declared tier** by blast radius. Tier is authoritative; the router defines it:

| Tier | Shape | Required artifacts |
|------|-------|--------------------|
| XS | One app calling one replicated managed datastore; retries + idempotency only | 01, 02, 07, 10, 13 |
| S | Single-region, single-leader replication, a few services, a queue | XS + 03, 09, 11 |
| M | Cross-service transactions/sagas, sharded data, real downstream consumers | S + 04, 05, 08, 12 |
| L | Multi-region / active-active, quorum consensus, clock-sensitive ordering | M + 06, gate runs strict |
| XL | Cross-org / partial-trust / Byzantine / regulated | L + BFT in 04, signed delivery in 09, full 12 |

**Tier promotion, not waiver.** If, while walking a sheet, its guidance forces an artifact above the
declared tier (e.g. the failure model reveals a clock-sensitive ordering requirement at declared tier M),
the artifact becomes *required* and the tier is promoted. You do not waive your way back under tier; you
record the promotion in `00-` and emit the artifact.

Do NOT pick the consistency contract here — that is elicited in `01-`. `00-` records the *requirements*
the contract must satisfy.

### Step 2 — Optional gap pre-pass via `distributed-design-reviewer`

For a brownfield system or an existing partial design, dispatch the reviewer agent before walking the
sheets, so its findings shape the artifacts rather than contradict them:

```
Use the Task tool with subagent_type: "distributed-design-reviewer"
Provide: the system's design artifacts / code map, any partial spec set, the declared tier
Receive: a gap report — unnamed guarantees, untraceable choices, missing-cost channels, severity-rated
```

Incorporate findings into the relevant numbered artifacts as you author them. Critical findings (an
unnamed consistency guarantee on a data class, a delivery channel with no idempotency) block the gate;
high findings block that channel's gate check; medium and low inform but do not block.

### Step 3 — Walk the tier-required sheets in dependency order

The spike comes first: you cannot design replication, ordering, or transactions until you know what
*fails* and what consistency you owe. Walk in this order, emitting each artifact from its sheet. Skip any
artifact not required at the declared tier (per the Step 1 table).

| Order | Sheet | Artifact | Notes |
|-------|-------|----------|-------|
| 1 | `consistency-models-and-cap.md` | `01-consistency-contract.md` | Per-data-class guarantee; the contract the rest must honour |
| 2 | `failure-models-and-fallacies.md` | `02-failure-model.md` | **The spike** — what fails, how detected, what is assumed-never |
| 3 | `replication-and-quorums.md` | `03-replication-spec.md` | S+; leader/quorum topology, R/W set sizes |
| 4 | `consensus-and-coordination.md` | `04-coordination-spec.md` | M+; leader election, locks, BFT at XL |
| 5 | `partitioning-and-sharding.md` | `05-partitioning-spec.md` | M+; shard key, rebalancing, hot-shard policy |
| 6 | `time-clocks-and-ordering.md` | `06-ordering-spec.md` | L+; logical/hybrid clocks, the ordering guarantee |
| 7 | `idempotency-and-deduplication.md` | `07-idempotency-spec.md` | Idempotency keys, dedupe window, the inbox |
| 8 | `sagas-and-distributed-transactions.md` | `08-transaction-spec.md` | M+; saga steps, compensations, the outbox |
| 9 | `delivery-and-ordering-semantics.md` | `09-delivery-spec.md` | S+; delivery semantics, ordering guarantee, signing at XL |
| 10 | `resilience-patterns.md` | `10-resilience-spec.md` | Timeouts, retries+jitter, breakers, bulkheads |
| 11 | `backpressure-and-flow-control.md` | `11-backpressure-spec.md` | S+; bounded queues, shed/throttle policy |
| 12 | `testing-distributed-systems.md` | `12-test-strategy.md` | M+; fault injection, the invariants under partition |
| 13 | `cost-and-when-not-to-distribute.md` | `13-cost-and-boundary.md` | Always; cost recorded, the "do not distribute" check |

Dependency rules while walking:

- `01-` and `02-` are prerequisites for *every* later artifact: each later spec names which consistency
  guarantee (from `01-`) and which failure (from `02-`) it defends against. A spec that cannot trace to
  both is incomplete.
- `07-idempotency-spec.md` and `09-delivery-spec.md` are a pair: at-least-once delivery (`09-`) is only
  correct if every consumer is idempotent (`07-`). Author them together.
- `08-transaction-spec.md`'s outbox and `07-`'s inbox are the same correctness device on two sides; keep
  them consistent.
- If `12-test-strategy.md` cannot state a falsifiable invariant for a guarantee declared in another spec,
  that guarantee is not yet gateable — return to the source spec.

### Step 4 — Cross-reference out, never duplicate

While authoring, route — do not re-derive — anything outside this pack's boundary:

- Broker internals / event sourcing / CQRS mechanics → the event-driven-architecture pack. This pack owns
  the *correctness contract* of delivery (semantics, ordering, idempotency, outbox/inbox); it references
  the broker pack for mechanics.
- Deterministic-simulation *testing* of the cluster → `axiom-determinism-and-replay` (cross-reference from
  `12-`); this pack does not own seeds/snapshots/replay loops.
- Single-service API internals (REST/GraphQL, one service's data model) → `axiom-web-backend`.
- Deployment / CI-CD / rollout → `axiom-devops-engineering`.

A spec that starts re-explaining broker offsets or replay snapshots has drifted; replace the prose with a
cross-reference.

### Step 5 — Consolidate → `99-distributed-system-specification.md`

Router-owned consolidation. Assemble the `99-` spec from the numbered artifacts:

- A channel table: one row per correctness channel (consistency, delivery, ordering, idempotency,
  transactions, replication, …) with its **named guarantee**, the failure it defends, its recorded cost,
  and the test/invariant that proves it.
- The declared tier and any promotions recorded in Step 1.
- The list of cross-referenced packs and what each owns.
- Open risks handed to `/axiom-solution-architect` for whole-solution risk consolidation.

If `99-` does not exist yet, generate a draft from the artifacts; the user reviews and signs off before
the gate is declared final.

### Step 6 — Run the Consistency Gate

Invoke the Consistency Gate procedure from `using-distributed-systems/SKILL.md` (router-owned). For each
correctness channel, the gate asks four questions and records a result:

1. **Named** — is the guarantee named precisely (e.g. "read-your-writes on the session key",
   "at-least-once with idempotent consumers"), not "mostly consistent" / "should be fine" / an un-scoped
   global "we use strong consistency"?
2. **Traceable** — does it trace to a specific failure in `02-failure-model.md`?
3. **Costed** — is its cost (latency, availability, dollars, operational burden) recorded in `13-` or the
   channel's spec?
4. **Tested** — is there a test or invariant in `12-` (or the channel's spec) that fails if the guarantee
   is violated?

Record **pass / fail / waiver** per check in the gate report. A silent un-named choice is the exact
failure mode this pack exists to prevent — it is a FAIL, not a pass-by-default. A waiver must name who
accepted the risk and why; an unexplained waiver is a fail. At tier L and above the gate runs strict:
waivers on consistency, delivery, and ordering channels are not permitted.

Failures are addressed before declaring the design complete; this command does NOT bypass the gate.

## Output Location

Artifacts land in `distributed-system/` at the repo root (or `distributed-system/<system>/` when one
repo hosts several systems). The `00-` and `99-` files are router-owned; `01-`–`13-` come from their
sheets. No code is emitted by this command.

## Downstream Handoffs (suggest after completion)

- Whole-solution risk consolidation — `/axiom-solution-architect` consumes the `99-` spec and folds these
  channels into the architecture risk register and ADRs.
- Deterministic cluster testing — `/scaffold-replay-system` (axiom-determinism-and-replay) if `12-`
  calls for deterministic-simulation testing of the cluster.
- Broker mechanics — the event-driven-architecture pack for the implementation of the delivery semantics
  named in `09-`.
- Service internals — `/scaffold-api` or `/review-api` (axiom-web-backend) for each service's own design.
- Deployment — `/design-deployment` (axiom-devops-engineering) once the design is stable.

## Scope Boundaries

Covered: scope + tier declaration, the numbered correctness-channel specs, optional gap pre-pass,
consolidation, the Consistency Gate.

Not covered: code/implementation, deployment/ops/CI-CD, single-service API internals, broker/event-sourcing
mechanics, determinism/replay machinery. Each is cross-referenced, never duplicated.

## Common Mistakes (in the design pass)

| Mistake | Fix |
|---------|-----|
| Declaring "we use strong consistency" globally | Consistency is per-data-class; name the guarantee per class in `01-` |
| Designing replication/ordering before the failure model | `02-` is the spike; it gates every later artifact |
| Picking the consistency contract for the user | The contract is *elicited* into `01-`; the command does not decide it |
| Waiving an artifact that guidance forced above tier | That is tier promotion, not a waiver; promote in `00-` and emit it |
| At-least-once delivery without idempotent consumers | `09-` and `07-` are a pair; at-least-once requires `07-` |
| Outbox in `08-` and inbox in `07-` drift apart | Same correctness device, two sides; keep them consistent |
| A guarantee with no invariant in `12-` | Untestable guarantee is not gateable; add the invariant or drop the claim |
| Re-explaining broker offsets / replay snapshots in a spec | Out of boundary; replace prose with a cross-reference |
| Skipping `13-cost-and-boundary.md` because "it's obvious" | Cost is a deliverable; the "do not distribute" check lives here |
| Treating a silent un-named choice as a gate pass | Un-named = FAIL; the gate exists to surface exactly this |
| Marking a waiver without an owner or reason | Waiver names who accepted the risk and why; otherwise it is a fail |
| Running L-tier gate with consistency waivers | At L+ the gate is strict; consistency/delivery/ordering waivers are not permitted |

## Cross-Pack Notes

- `axiom-solution-architect:using-solution-architect` — consumes the `99-` spec for whole-solution risk
  consolidation and ADRs; this is the primary downstream consumer.
- `axiom-determinism-and-replay:scaffold-replay-system` — deterministic-simulation testing of the cluster;
  cross-referenced from `12-test-strategy.md`.
- `axiom-web-backend:using-web-backend` — the internals of any single service named in the topology.
- `axiom-devops-engineering:design-deployment` — rollout of the designed system; out of this pack's scope.
