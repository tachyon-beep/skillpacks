---
name: audit-aware-logging-vs-observability
description: Use when delineating audit-grade events from ordinary observability events — a stable, testable rule a developer can apply to a new event class, not a slogan. Prevents audit obligations from being crammed into the observability stack and observability noise from being routed through the audit pipeline. Produces `09-audit-vs-observability-boundary.md`.
---

# Audit-Aware Logging vs Observability

## Overview

**Audit-grade pipelines and ordinary observability share infrastructure (storage, query, dashboards) and require different properties. Conflating them produces audit logs that lose data on backpressure, observability dashboards that grind under retention obligations, and threat models that protect the wrong bytes.**

This sheet is the boundary rule. It tells a developer reading it whether a new event class belongs in the audit pipeline or in the observability stack — and the rule is *testable*, not aspirational.

This sheet is written last in the spec sequence (it is artifact `09-`) because it draws boundaries against the rest. Read the other sheets first; this one decides what they exclude.

## When to Use

Use this sheet when:

- A team is producing application logs and asking whether they should "also be the audit log."
- A new event class is proposed and the routing question — audit, observability, or both — needs to be answered.
- Observability infrastructure (Splunk, Datadog, Loki, Elastic) is being considered as the audit store.
- Producing `09-audit-vs-observability-boundary.md`.

## Core Principle

> Different correctness properties want different infrastructures. Audit answers "did the right thing happen, and can we prove it?" Observability answers "is the system healthy, and where is it slow?" Audit must not lose data; observability must not lose usefulness. The two operations conflict under load — that is when you find out which infrastructure you actually built.

## The Boundary Rule

The rule, applied to any candidate event class:

```
Q1: Does someone (regulator, auditor, customer, court) potentially ask "prove it"?
    Yes → audit pipeline
    No  → observability

Q2: Would loss of one of these events under backpressure be tolerable?
    Yes → observability
    No  → audit pipeline

Q3: Is the value of the event derived from the bytes themselves (the decision)
    or from aggregate patterns (latency, throughput, error rate)?
    Bytes  → audit pipeline
    Patterns → observability
```

If Q1 is yes, the answer is audit, regardless of Q2 and Q3. If Q1 is no, Q2 and Q3 jointly route — both pointing to audit means this is a *gray-zone event* and needs explicit handling (typically dual-write, see below). Both pointing to observability means it stays in observability.

The rule is testable: a developer with a candidate event class can apply Q1-Q3 and arrive at a single answer. A boundary that requires judgement at every event class is no boundary; it is a slogan.

### Examples

| Event class | Q1 | Q2 | Q3 | Routing |
|-------------|----|----|----|---------|
| Policy engine emits allow/deny | Yes | No | Bytes | Audit |
| Policy engine evaluation latency | No | Yes | Patterns | Observability |
| Policy engine threw exception | No (typically) | Yes | Patterns | Observability |
| Policy engine threw exception that produced a default-deny | Yes (the deny IS the decision) | No | Bytes | Audit (record the deny + exception evidence) |
| User login succeeded | Yes (often regulated) | No | Bytes | Audit |
| Login response time histogram | No | Yes | Patterns | Observability |
| KMS key rotation | Yes (per `04-`) | No | Bytes | Audit |
| KMS request count | No | Yes | Patterns | Observability |
| Database row written | No | Yes | Patterns | Observability |
| Database row written for an entry that ALSO went to audit | Already audited | — | — | Just observability |
| Configuration change deployed | Yes (often regulated) | No | Bytes | Audit |
| Pod restart | No | Yes | Patterns | Observability |
| Approval workflow transition | Yes | No | Bytes | Audit |
| Cache hit/miss ratio | No | Yes | Patterns | Observability |

## Why Property Conflicts Are Real

Audit and observability are often colocated and break when treated as the same problem.

| Property | Audit demands | Observability demands |
|----------|---------------|------------------------|
| Loss under backpressure | Forbidden | Acceptable (sampling, drop) |
| Schema mutation | Chain-breaking event | Routine (add fields freely) |
| Read-time mutation | Forbidden | Routine (downsampling, rollups) |
| Retention | Months to decades, regulatory | Days to weeks, cost-driven |
| Encryption-at-rest with app-side keys | Often required | Rarely required |
| Operator visibility | Restricted | Broad |
| Cardinality | Bounded by `decision_type` enum | Often high (per-user, per-request) |
| Query patterns | Time-range, specific id, integrity verification | Aggregate, ranking, anomaly |
| Correctness metric | Did anything get lost? | Was the dashboard fast enough? |

Conflating these means: the observability stack drops audit events under load (Q2 violated); the audit retention obligation crashes observability cost; encryption-at-rest with operator-blind keys breaks the on-call's ability to investigate; the schema migration that cleans up observability fields breaks the chain.

## What "Audit-Aware Logging" Looks Like

Audit-aware logging produces *observability* events that mirror or reference audit events without becoming them. It is the bridge between the two stacks.

Patterns:

### Pattern 1 — Audit emits, observability subscribes

The audit pipeline writes an entry; an observability sink subscribes (event bus, log forwarder) and receives a *projection* of the entry: timestamp, decision_type, producer_id, output (only for low-sensitivity decisions), no `inputs_commitment`, no `entry_hash`, no chain pointers.

The projection is allowed to drop, sample, or summarise. The audit entry is the source of truth.

### Pattern 2 — Observability emits, audit references

For events where Q1 is no but the observability event has audit relevance (e.g., a code deployment triggers downstream behaviour change), the audit pipeline holds an entry that *references* the observability event by id. The observability event lives or dies on its own retention; the audit reference persists. The audit reference doesn't try to carry the observability event's bytes — only the id and enough context for an investigator to look it up if the observability data still exists.

### Pattern 3 — Dual-write for gray-zone events

Some events satisfy Q1 weakly (regulator might ask, depending on incident type). Examples: authentication failures, capacity-driven request rejections, automated retries that change behaviour. These dual-write: full audit entry plus observability projection. The dual-write itself is structured so neither side blocks the other (audit write is durable; observability is fire-and-forget).

Document each gray-zone event class in `09-`. The default for ambiguity is audit (Q1 → conservative yes).

## What Audit-Aware Logging Is *Not*

| Not this | Why |
|----------|-----|
| "Tag observability events with `audit: true`" | Tags don't produce canonical bytes, chain integrity, or retention discipline |
| Writing to both stacks identically | The properties conflict; one stack will be wrong for one purpose |
| Using the audit store as observability | Cardinality and query-pattern mismatch will make it unusable |
| Using observability as the audit store | Loss-under-load and schema-mutation properties violate audit |
| "We have a SIEM, that's our audit log" | SIEMs are observability tools with audit-shaped marketing; verify against the boundary rule |

## Backpressure and Drop Policy

The backpressure question is the fastest way to test whether a stack is audit-grade.

For the audit pipeline:

> Under sustained load that exceeds write capacity, the *system* slows down. Decisions wait until they can be durably audited. The audit pipeline is on the critical path of the decision producer.

For observability:

> Under sustained load, observability sheds: drops, samples, rolls up. The system continues; the observability is degraded. Observability is *not* on the critical path.

This is the cleanest test. Ask of any candidate stack: under sustained overload, does it drop, or does it block? If it drops, it is observability. If it blocks the producer, it is audit.

A pipeline that drops audit events under load is not an audit pipeline. It is a probabilistic record of decisions, which is not a record at all.

### Audit fan-out

The need for backpressure-on-audit conflicts with the desire for fan-out (multiple downstream consumers of the same audit stream). The discipline:

- The producer writes durably to the *primary* audit store. This write is on the critical path.
- The primary store fans out to consumers asynchronously. Consumer downstream of the primary may lag, fail, or fall behind without affecting the producer.
- Consumers maintain their own checkpoints and recover by re-reading from the primary. They never become the only copy.

This makes the primary an append-only log product or a durable database; consumers can be observability sinks, search indexes, ML feature stores, or anything else.

## Spec Output (`09-audit-vs-observability-boundary.md`)

Answers, in order:

1. **The Q1-Q3 rule restated for the team.** The rule itself; an example list of recently-classified events.
2. **Gray-zone register.** Decision-type-by-decision-type list of dual-write events, with reason.
3. **Stack identity.** Names of the audit stack and the observability stack; what is in each.
4. **Backpressure policy.** Audit blocks; observability sheds; the names of the components that enforce each.
5. **Audit-to-observability projection.** What fields cross the boundary, with rationale (low-sensitivity-only, no integrity-relevant fields).
6. **Observability-to-audit reference.** When and how observability events are referenced by audit entries.
7. **Operational ownership.** Who owns the audit stack vs the observability stack; access controls; on-call separation.
8. **Migration discipline.** Schema changes in observability stay in observability; do not bleed into audit. Vice versa: audit schema changes do not propagate to observability via casual code paths.
9. **Cross-pack handoff.** Observability tooling / dashboards owned outside this pack — name the pack or the team.

## Common Mistakes

| Mistake | Why it fails | Fix |
|---------|--------------|-----|
| Audit stack is the observability stack | Q2 violation: drops under load | Separate stacks; backpressure policy explicit |
| "We tag audit events differently in the same stack" | Same retention, same schema rules; tag is decoration | Different stack identities, different ownership |
| SIEM marketed as audit log | Loss-under-load, mutation, schema flexibility violate audit properties | Verify against boundary rule; use a true audit store for audit |
| Observability emits canonical-encoded events for "compatibility" | The bytes are the chain's, not observability's; canonical encoding leaks where it shouldn't | Projection is a *summary*, not a copy of the audit bytes |
| Gray-zone events resolved per-developer | Inconsistent classification, drift over time | Gray-zone register in `09-` is authoritative; new gray-zone events update the register |
| Dual-write where audit blocks on observability | Audit pipeline now drops on observability outage | Audit is the durable path; observability is fire-and-forget |
| Dropping audit "just for now" during incident | Incidents are when audit matters most | Block; raise capacity; never drop |
| Encrypting observability with audit keys | Operators lose dashboards; on-call breaks | Different stacks have different threat models; different keys |

## The Bottom Line

**Audit answers "prove it"; observability answers "is it healthy". Different properties (loss tolerance, schema mutability, retention, encryption, ownership) want different infrastructures. The Q1-Q3 boundary rule is testable; gray-zone events go to a register, not to vibes; backpressure tests stack identity (block vs shed); audit-aware logging is a one-way projection from audit to observability and a reference-only direction from observability to audit. Conflating the two breaks both.**

---

**Retrieval test (run at end of build):** "Our team wants to add a new event: 'config change deployed by user X via tool Y at time T.' Apply the Q1-Q3 rule. Where does it go and why?"
