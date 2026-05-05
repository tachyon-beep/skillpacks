---
name: partial-replay-from-trail
description: Use when designing replay-from-audit (reconstructing system state from N entries to answer "given what happened, what did the system look like?"). Covers replay scope honesty, what state outside the trail is required, and the boundary against replay-for-debugging owned by axiom-determinism-and-replay. Produces `08-replay-capability.md`.
---

# Partial Replay from an Audit Trail

## Overview

**Replay-from-audit reconstructs system state from N entries to answer: "given what we know happened, what did the system look like?" It is provenance's operational counterpart — an instrument the auditor uses, the incident responder uses, the regulator uses. It is *not* replay-for-debugging (which is the simulation-determinism problem); it is replay-as-evidence.**

This sheet specifies what can and cannot be replayed from a trail, what state outside the trail is required, how to express replay scope honestly, and the boundary between this kind of replay and the non-determinism problem owned elsewhere.

## When to Use

Use this sheet when:

- A consumer of the trail (auditor, regulator, investigator) needs to reconstruct system state from the chain.
- Designing the replay tooling that walks the trail and emits a snapshot.
- Producing `08-replay-capability.md`.

Do not use this sheet for:

- Replay for debugging non-deterministic simulators or stochastic systems → that is `axiom-determinism-and-replay` (architectural design of replayable systems) or `yzmir-simulation-foundations:check-determinism` (verification of an existing simulation). The two problems share vocabulary and are different concerns. See *Boundary* below.

## Core Principle

> Replay-from-audit produces an *evidentiary* state — what the trail says the system looked like — not a *behavioural* state. The trail records decisions; replay aggregates decisions; aggregation is honest about what the trail does and does not cover. A "fully replayable" claim without a state-coverage proof is not a claim, it is a wish.

## Boundary: Replay-from-Audit vs Replay-for-Debugging

| Property | Replay-from-audit (this sheet) | Replay-for-debugging |
|----------|-------------------------------|----------------------|
| Goal | Reconstruct state from evidence | Reproduce a bug; investigate behaviour |
| Source | The audit trail | A captured execution log, possibly extensive |
| Determinism requirement | Decisions in trail are facts; replay is aggregation | The full execution must be deterministic to reproduce |
| Time treatment | Wall-clock from `decided_at` (with provenance from `03-`) | Logical / monotonic time critical for ordering |
| Adversary model | Adversaries against the trail, not the system | Bugs, race conditions, scheduling |
| Owner | this pack | `axiom-determinism-and-replay` (architecture); `yzmir-simulation-foundations:check-determinism` (verification) |

The two share vocabulary because both walk a sequence of records. They diverge on what the records are *for*. An auditor doesn't care that the system is deterministic; they care that the trail is. A debugger needs the system deterministic too.

This sheet covers replay-from-audit only. If determinism of the system is the question, use `axiom-determinism-and-replay`. The two packs cross-link rather than overlap: a deterministically-replayable system that also keeps an audit trail is a system where both "what was decided" (this pack) and "given the recorded inputs, what would happen" (determinism pack) are answerable, and the answers do not need to match — they answer different questions.

## What Can Be Replayed

A trail of N decisions can reconstruct:

1. **Sequence of decisions made.** Trivial; the trail is the sequence.
2. **State directly produced by decisions.** If a decision's `output` is "transition case C from state A to state B," replaying decisions of that type rebuilds case state.
3. **Aggregates over decisions.** Counts, sums, distributions, time-series — the trail is the source-of-truth for these per `audit-aware-logging-vs-observability.md` (Q3 → bytes domain).
4. **Provenance for any single decision.** With `inputs_commitment`, `ruleset_version`, `code_version`, the cause-effect closure of `05-` resolves on demand.

What can be replayed *with the registries* (`03-` / `05-` content-addressed stores):

5. **The exact inputs the producer saw.** If `inputs_commitment.ref.hash` resolves to bytes still held in the inputs registry.
6. **The exact ruleset that fired.** If `ruleset_version` resolves to bytes in the ruleset registry.
7. **A re-run of the decision.** Given inputs + ruleset + code (via SLSA-resolvable build), execute the producer with those inputs against that ruleset and confirm the output. This is the strongest form of provenance verification.

## What Cannot Be Replayed

Honesty about gaps is the core property. Replay cannot reconstruct:

1. **State outside the trail's scope.** Decisions that *weren't audited* didn't go in the trail; their effects are not in the replay. The boundary in `09-` is the source-of-truth for what's missing.
2. **Effects of non-decided actions.** If the system also performed actions that were not "decisions" (a maintenance script ran, a data migration occurred), those don't appear in the trail and don't appear in the replay.
3. **External-world state.** Other systems' decisions, third-party data changes, user actions outside the system are absent unless they were captured as `inputs_commitment` for some decision.
4. **State for deletions / RTBF events.** After a redaction or cryptographic erasure, the original content cannot be replayed for that entry. The trail records "decision occurred at time T with hash H1 → H2" but cannot reproduce the bytes.
5. **The exact runtime environment.** `code_version` resolves to the build; the build resolves (with SLSA) to the source. The runtime environment (host, kernel, library versions if not pinned) may differ. For decisions where this matters (floating-point or model nondeterminism — see `decision-provenance.md`), the trail records what was, not what currently is.

State the limitations explicitly in `08-`. "We can replay decisions" is true; "we can replay system state" is overclaim if the system has out-of-trail effects.

## Replay Scope Models

Three useful scopes:

### Pointwise replay

Reconstruct the state at a single moment in time. "What did the system think about subject S as of date D?"

**Procedure:**

1. Filter trail entries to `decided_at ≤ D` AND scope predicate (subject S).
2. Apply each entry in time order, updating an in-memory state model.
3. The state at the end is the replay output.

**Caveats:**

- The state model is a *projection* — its schema is decided by the replay, not by the trail. State the projection schema in `08-`.
- Pointwise replay assumes idempotent application of entries; design the state model accordingly.

### Range replay

Reconstruct a sequence of state changes over a time window. "What was the case progression from Monday to Friday?"

**Procedure:**

1. Filter trail entries by predicate AND time range.
2. Apply each in order, capturing a snapshot at each entry boundary (or at sampling intervals).
3. Output is a series of state snapshots plus the entries that caused each transition.

**Caveats:**

- Snapshot frequency is a budget tradeoff (every entry is precise but expensive; sampled is cheap but lossy).
- Aligns with regulator's typical question — "show me the timeline of decisions for this case."

### Selective replay

Reconstruct only what is needed to answer a specific question. "Did rule R fire for any case during Q4?"

**Procedure:**

1. Filter trail entries by `decision_type` and `evidence` content (rule R was named).
2. Project to the answer (yes/no with citations, or count, or list).
3. Output is the answer with the supporting entries as evidence.

**Caveats:**

- Performance-friendly; this is the dominant query pattern in incident response.
- Requires the schema in `01-` to expose the queried fields (rule firings in `evidence`, etc.) at canonical-encoding boundaries.

## State-Coverage Proofs

The honest replay tool produces, alongside the state, a *coverage statement*:

```
{
  replay_scope: {
    decision_types: [...],
    time_range: {from, to},
    predicate: <as applied>
  },
  trail_segment: {
    first_entry_id, last_entry_id,
    chain_anchor_used: <hash>,
    chain_anchor_provenance: <signature, source>
  },
  entries_processed: <count>,
  entries_covered_by_chain_verification: <count>,  // matches if no gaps
  gap_markers_in_segment: [<list of gap_marker entries>],
  redactions_in_segment: [<list of redacted entries>],
  out_of_scope_state: [
    "User profile mutations not captured in trail",
    "Cron-driven maintenance actions not captured in trail",
    ...
  ],
  replay_completed_at: <RFC 3339 UTC>,
  replay_signed_by: <key_id>,
  replay_signature: <bytes>
}
```

The state output without this statement is not evidence; it is an assertion. The statement makes the replay self-describing — a reader knows what was covered and what was not, what was verified and what was assumed.

## Replay Verification

A replay can be verified by re-running it. The honest replay tool is *deterministic*: same trail segment + same registries + same replay code = same state and same coverage statement. This makes replay results byte-comparable.

Two parties verifying separately should produce identical state and identical coverage statements; if not, one of them used a different trail segment, registry version, or replay code version. State the deterministic property of the replay tool in `08-` and treat its code version as itself a `code_version` with provenance.

A replay tool that produces non-deterministic output (caches that affect ordering, threading that affects accumulation order, locale-dependent formatting) is unfit for evidence.

## Replay and Performance Budget

Long-trail replay is expensive. See `performance-budget-for-audit-grade-pipelines.md` for amortisation strategies. The dominant patterns:

- **Periodic checkpoints.** A signed snapshot at scheduled intervals; replays start from the most recent checkpoint instead of the chain head. The checkpoint itself is an audit entry (`decision_type: "audit.checkpoint"`) with its own integrity.
- **Incremental replay.** Replay maintains state across queries; subsequent queries incrementally extend rather than restarting.
- **Pre-built indexes.** Indexes derived from the trail (e.g., "all entries about subject S") are observability projections per `09-`, used to *select* trail entries cheaply but never *replace* the trail in the replay step.

## Adversarial Replay

A trail that has been tampered with may produce a "successful" replay that reflects the tamper, not reality. Defences:

- The replay's coverage statement names the chain anchor used. A verifier checking the replay can verify the anchor's provenance independently.
- The replay processes gap-marker and redaction entries explicitly; their presence in the coverage statement is informative.
- Replays are typically run by parties who have separate access to the chain anchors (the regulator, the auditor, an investigator). This means the replay's chain-of-evidence is independent of the producer.

A replay run by the producer themselves, against their own anchors, with no external verifier, is internal evidence — useful for investigation, weak as audit.

## Spec Output (`08-replay-capability.md`)

Answers, in order:

1. **Replay scope models supported.** Pointwise, range, selective — which are first-class in tooling.
2. **State projection schemas.** Per `decision_type` or per query class, what fields are projected from `output` into the replay state.
3. **State-coverage statement schema.** The structured output that accompanies every replay.
4. **What is not in scope.** Explicit list: out-of-trail effects, runtime environment dependencies, deleted/redacted content.
5. **Replay tool determinism.** How determinism is achieved; the tool's own code-version.
6. **Checkpointing strategy.** Cadence, scope, signing.
7. **Verification protocol.** How a third party reproduces a replay.
8. **Cross-pack handoff.** Index/observability stack used to *select* entries; replay tool itself owned by this pack.

## Common Mistakes

| Mistake | Why it fails | Fix |
|---------|--------------|-----|
| "Fully replayable" claim without coverage statement | Implicit overclaim; auditor probes and finds gaps | Explicit coverage statement, named not-in-scope items |
| Replay tool with non-deterministic output | Two replays disagree; evidence is unverifiable | Deterministic; code-versioned; reproducibility test in CI |
| Replay reads only the trail, not registries | `inputs_commitment.ref` not resolvable; replays incomplete | Replay tool reads trail + registries + code SLSA |
| Replay over a chain segment without anchor verification | Tamper undetected; replay echoes the tamper | Coverage statement names anchor; anchor independently verifiable |
| State projection schema implicit (whatever the replay code does) | Schema drift between replay versions | State projection schema explicit in `08-`, versioned |
| Replay used as observability dashboard | Performance impossible; query patterns wrong | Replay produces evidence; observability serves dashboards; see `09-` |
| Pointwise replay ignores gap markers | Misleading state at the gap boundary | Replay processes gap markers explicitly; coverage statement reports them |
| Replay caches break determinism | Order-of-cache-population leaks into output | Caches are deterministic functions of input; or no caches in evidence-grade replay |

## The Bottom Line

**Replay-from-audit is evidence aggregation, not state reproduction. Pointwise / range / selective scopes serve different consumer questions. State-coverage statements are mandatory output — what was processed, what was anchored, what was gapped, what was redacted, what was out of scope. The replay tool is itself deterministic and code-versioned. Replay is independent of debugging-replay (the determinism problem); cite the boundary clearly.**

---

**Retrieval test (run at end of build):** "An investigator wants the case-state timeline for case C from Q3 last year. The chain has two gap markers in that window, one redaction, and a checkpoint dated end-of-Q2. Walk through what the replay tool produces and what the coverage statement says."
