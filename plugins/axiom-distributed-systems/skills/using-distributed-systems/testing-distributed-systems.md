---
name: testing-distributed-systems
description: Use when example-based tests pass but the system loses writes, splits brain, or duplicates effects under partition, reorder, or clock skew — when "it works in staging" is the only durability argument, when a claimed consistency level was never checked, or when fault injection and chaos are ad hoc. Produces `12-test-strategy.md`.
---

# Testing Distributed Systems

## Overview

**A distributed bug is a function of failure timing, not input value, so the tests that find it must inject failure on a schedule — not feed inputs and assert outputs.** A green unit suite and a clean staging week tell you nothing about what happens when a leader pauses for a GC stall mid-commit, a packet is delivered twice after a retry, or two nodes each believe they hold the lock. This sheet specifies the test discipline that exercises the failure model in `02-failure-model.md` and proves the guarantees the system claims: fault injection, invariant and linearizability checking, deterministic simulation, and production verification. The deliverable is `12-test-strategy.md`.

## When to Use

Use this sheet when:

- The system is at tier M or above (cross-service transactions, sharding, real consumers) — `12-` is required from M up.
- A consistency level is claimed in `01-consistency-contract.md` but nothing checks it — "we use strong consistency" with no linearizability test is an un-verified assertion.
- A bug reproduces "only under load" or "only sometimes after a deploy" — that is a timing-dependent fault, not a flake.
- The failure model in `02-` lists partitions, reordering, duplication, or clock skew, but the test suite injects none of them.
- The team's durability argument is "it worked in staging" or "the happy path passes."
- An idempotency (`07-`), saga (`08-`), or delivery (`09-`) guarantee was specified and needs a test that proves the invariant holds under the faults that threaten it.

Do not use this sheet for:

- The deterministic-simulation *machinery* itself (seeded scheduler, single-threaded sim loop, snapshot/restore) — that is `axiom-determinism-and-replay`. This sheet says *what* to simulate and *what invariants* to assert; it cross-references that pack for *how* to make the sim deterministic.
- Naming the faults that can occur — that is `failure-models-and-fallacies.md` (`02-`). This sheet injects the faults that sheet declares.
- Choosing the resilience mechanisms under test (timeouts, retries, circuit breakers) — that is `resilience-patterns.md` (`10-`). This sheet verifies they work.
- Load/performance/capacity testing in isolation — relevant only where it overlaps fault injection (slow disk, backpressure). Pure throughput benchmarking is out of scope.

## Core Principle

> Test the guarantee, not the happy path. For every guarantee named in `01-`/`07-`/`08-`/`09-`, there is one fault in `02-` that breaks it and one invariant that catches the break — write that test, run it under that fault, and let it fail before you trust it.

## Why Example-Based Tests Miss Distributed Bugs

A unit test fixes inputs and asserts outputs; concurrency, network state, and clock state are mocked away or assumed nominal. But distributed correctness lives precisely in the states that get mocked away. The bug classes that example tests structurally cannot reach:

| Bug class | Why a unit test can't see it | The fault that surfaces it |
|-----------|------------------------------|----------------------------|
| Lost write under partition | Network is mocked as reliable; the minority-partition write that gets dropped never happens | Partition during write, then heal |
| Split-brain | Single process under test; two leaders requires two nodes and a partition between them | Partition isolating the leader from its quorum |
| Duplicate effect | Retry path is not exercised; the "second delivery" of an at-least-once message never arrives | Message duplication after timeout-and-retry |
| Stale read | Replica lag is zero in a single-node test | Latency injection on the replication channel |
| Reorder-induced corruption | Messages arrive in send order in-process | Reorder injection on the channel |
| Clock-skew anomaly | One clock; no skew possible | Clock skew between nodes |
| Saga non-termination | Compensation path runs cleanly when nothing fails mid-saga | Node kill between saga steps |

The shared property: each bug requires a *specific timing of a specific failure*, not a specific input. You cannot enumerate inputs to find it. You must inject the failure and search the timing space. This is why fault injection, not more example tests, is the floor.

## Fault Injection: Inject the Faults Your `02-` Model Assumes

Fault injection is the floor of distributed testing. The rule is a derivation, not a menu: **the set of faults you inject is exactly the set of faults `02-failure-model.md` says can happen.** If `02-` lists a fault and the suite never injects it, that fault's handling is untested and the gate fails. If you inject a fault `02-` doesn't list, either the fault is real (fix `02-`) or you are wasting test budget.

The canonical fault catalog, mapped to what each surfaces:

| Fault | Injected by | Surfaces |
|-------|-------------|----------|
| Network partition | iptables/`tc`, toxiproxy, mesh fault rules | Split-brain, lost writes, quorum loss, failover correctness |
| Latency injection | toxiproxy, `tc netem delay` | Stale reads, timeout tuning, backpressure (`11-`), slow-path correctness |
| Packet loss | `tc netem loss` | Retry behaviour, idempotency (`07-`), at-least-once handling |
| Duplication | toxiproxy duplicate, app-level replay | Idempotency (`07-`), exactly-once-effect claims (`09-`) |
| Reorder | `tc netem reorder`, toxiproxy | Ordering guarantees (`06-`/`09-`), causal-consistency claims |
| Node kill (crash-stop) | SIGKILL, pod delete, VM stop | Failover, leader election, durability of acknowledged writes |
| Process pause (crash-recovery / GC stall) | SIGSTOP, `kill -STOP`, cgroup freeze | Lease/lock expiry, zombie-leader split-brain, fencing (`04-`) |
| Clock skew | libfaketime, clock-skew sidecar | Lease validity, TTL correctness, timestamp-ordering anomalies (`06-`) |
| Disk slow / disk full / fsync stall | `tc`-equivalent for I/O, cgroup io.max, loopback full FS | WAL durability, commit-path stalls, backpressure (`11-`) |
| Byzantine / corrupt message | app-level fault injector, fuzzed payloads | BFT claims (`04-`, XL tier), signature/auth verification (`09-`) |

Two faults teams routinely forget, and which find the worst bugs:

- **Process pause (SIGSTOP), not just kill.** A killed node is gone and its lease expires cleanly. A *paused* node still holds its lease in the cluster's view, then wakes up after the lease expired and a new leader was elected — now there are two leaders. Split-brain almost always comes from pause-then-resume, not crash. If `02-` admits GC stalls or VM migrations, you must inject pause.
- **Asymmetric partition.** A can reach B but B cannot reach A. This breaks naive failure detectors and causes flapping leadership. A symmetric partition is the easy case; asymmetric is where the subtle failover bugs live.

Fault injection is run at two scales: **integration tests** (a few nodes, deterministic fault schedule, run in CI on every PR) and **soak/chaos** (a full cluster, statistical fault schedule, run continuously — see below).

The shape of every fault-injection test is the same: bring up the cluster, start a workload that records a history, inject a fault from the schedule, heal, then run the oracle against the recorded history. The fault and the oracle are the only things that vary:

```text
test(fault, oracle):
    cluster   = bring_up(nodes)
    history   = start_recording_clients(cluster)   # real-time start/end + result per op
    inject(fault)                                   # one fault from the 02- catalog
    run_for(duration)
    heal()
    quiesce()                                       # let replication/recovery settle
    assert oracle.check(history, cluster.state)     # linearizability OR named invariant
    # a violation here is a reproduced defect — under DST, the seed reproduces it exactly
```

The mistake to avoid: stopping at "the cluster came back up." Liveness (it recovered) is necessary but not sufficient; the oracle checks *safety* (it stayed correct), which is where the silent bugs hide.

**Failure detectors are part of the system under test.** Many distributed bugs are not in the consensus or replication logic but in the failure *detector* — the heartbeat/timeout machinery that decides a node is dead. Asymmetric partitions and process pauses specifically attack the detector: a too-eager detector evicts a healthy-but-slow node (flapping, unnecessary failover); a too-lax one lets a paused-then-resumed node believe it is still leader. Inject latency and pause specifically to stress the detector's thresholds, and assert that detection decisions match reality.

## Chaos Engineering Done Properly

Chaos engineering is not "break random things in prod and see what happens." Done properly it is a controlled experiment with four required elements. Without all four it is just outage manufacturing.

1. **A steady-state hypothesis.** Define the measurable normal: "p99 write latency < 50ms and zero lost-write invariant violations." The experiment tests whether the hypothesis holds *during* the fault. No hypothesis → no pass/fail → not an experiment.
2. **A bounded blast radius.** Start in a pre-prod cluster or a single cell/shard with 1% of traffic. Have an abort condition and an automated rollback that fires the instant the steady-state hypothesis is violated beyond a threshold. The blast radius is the maximum harm an experiment may do; it must be set *before* the experiment runs.
3. **A single injected variable.** Inject one fault from the catalog at a time. If you inject partition + latency + node-kill simultaneously and the hypothesis fails, you have learned nothing about *which* one your system can't handle.
4. **Game days, scheduled and reviewed.** Run experiments as deliberate exercises with the on-call team present, observing whether *humans and runbooks* respond correctly, not only whether the system does. The output is a list of findings and fixes, fed back into `02-` and the resilience spec.

The maturity ladder: deterministic fault injection in CI (cheap, every PR) → automated chaos in staging (nightly, broad fault coverage) → controlled chaos in production with bounded blast radius (the steady-state hypothesis, abort conditions, game days). Production chaos is the *last* rung, earned after the lower rungs are green — not the entry point.

## Invariant and Linearizability Checking: Check the Contract the System Claims

Injecting faults is half the test; the other half is a checker that fails when a guarantee is violated. A fault injection run with no invariant checker tells you only whether the system *crashed*, not whether it stayed *correct*. Most distributed bugs are silent: the system keeps serving, but it lost a write or returned a stale read. You need an oracle.

**Linearizability checking (Jepsen + Knossos/Elle).** Record a history of concurrent client operations with their real-time start/end timestamps and results, then check whether that history is consistent with the claimed consistency model. Knossos checks linearizability by searching for a valid sequential ordering; Elle checks for serializability/snapshot-isolation anomalies by inferring the transaction dependency graph and looking for cycles (and, unlike Knossos, tells you *which* anomaly — G0/G1/G2 — and produces a minimal witness). The checker must check the model the system *claims in `01-`*: checking for linearizability against a system that only promised causal consistency produces false alarms; checking for nothing against a system that claimed linearizability is the failure this pack exists to prevent.

**The invariant catalog.** Beyond model-checking, assert the concrete cross-cutting invariants directly. Each invariant maps to a guarantee, the fault that threatens it, and the spec that owns it:

| Invariant | Guarantee it protects | Threatening fault | Owning spec |
|-----------|----------------------|-------------------|-------------|
| No lost writes (every acknowledged write is readable after heal) | Durability of acks | Partition during write | `03-`, `01-` |
| No split-brain (at most one leader accepts writes at any time) | Single-writer / consensus safety | Pause-then-resume, asymmetric partition | `04-` |
| Monotonic reads hold (a client never sees time go backward) | Session consistency | Replica lag, failover to stale replica | `01-` |
| No duplicate effects (at-least-once delivery, exactly-once effect) | Idempotency | Duplication, retry | `07-`, `09-` |
| Saga always terminates (commit or full compensation, never stuck) | Transaction atomicity-as-saga | Node kill mid-saga | `08-` |
| No phantom / no orphaned resource | Cleanup correctness | Crash between allocate and register | `08-`, `10-` |

The discipline: every guarantee named in `01-`/`07-`/`08-`/`09-` appears as a row in *this* table with a named checker. A guarantee with no checker is an un-tested claim, and the consistency gate fails it.

Some invariants are checked online (assert as operations stream — cheap, catches violations fast); others are checked offline as a reconciliation pass over the full recorded history (more thorough, catches anomalies that only emerge from the global ordering). Lost-write and no-duplicate-effect are natural online checks; linearizability and saga-termination are offline checks over the history. Specify which is which, because the online checks are the ones you can also run continuously in production (see below).

### Where this sits in the test pyramid

Distributed tests are slower and flakier-looking than unit tests, so teams under-invest in them and over-rely on the fast layer that cannot see the bugs. Resist the inversion:

| Layer | Catches | Cost | Cadence |
|-------|---------|------|---------|
| Unit / single-node | Logic errors, encoding, idempotency-key derivation | Cheap | Every commit |
| Fault-injection integration (few nodes, deterministic schedule) | Partition/reorder/dup/pause handling, failover, invariant violations | Moderate | Every PR |
| DST (whole cluster, seeded) | Rare interleavings, clock-sensitive bugs, reproducible | High to build, cheap to run | Continuous / overnight seed sweep |
| Chaos / soak (full cluster, statistical) | Emergent, dependency, scale-dependent faults | High | Nightly staging → gated prod |
| Production verification (canary, shadow, continuous invariants) | What only prod's real traffic and dependencies expose | Ongoing | Always |

The non-negotiable layer for a tier-M+ system is fault-injection integration: it is where a guarantee-plus-fault-plus-invariant test actually runs on every change. Unit tests alone are the happy-path trap this sheet exists to break.

## Deterministic Simulation Testing

Fault injection on real nodes searches the timing space *stochastically* — you re-run and hope the bad interleaving recurs. When it does recur, you often can't reproduce it. Deterministic simulation testing (DST), as built by FoundationDB and TigerBeetle, removes the stochasticity: run the *entire cluster* — every node, the network, the clock, the disk — as a single-threaded simulation driven by one seeded pseudo-random fault schedule. The simulated clock advances logically, messages are delivered by the sim scheduler, and faults (partition, drop, reorder, pause, disk-stall) are injected per a seed-derived schedule.

What DST buys you, and why it is the gold standard at L/XL:

- **Reproducibility.** A failing run is a *seed*. Re-run the seed, get the identical interleaving, every time. The "I can't reproduce that" that example-based distributed testing is famous for simply does not occur.
- **Speed.** A simulated cluster runs faster than wall-clock (no real network, no real fsync) and can compress hours of logical time into seconds, exploring millions of fault schedules overnight.
- **Coverage of rare interleavings.** Seed-space search finds the one-in-a-billion pause-at-exactly-this-byte interleaving that real-node chaos would take years to hit.

The cost and the prerequisite: the system code must be written to run *inside* the simulator — all I/O, time, and randomness go through an injectable interface so the sim can control them. This is a deep architectural commitment, not a test you bolt on later. The simulation's determinism machinery — the seeded scheduler, the single-threaded sim loop, the I/O substitution layer, snapshot/restore — is exactly what `axiom-determinism-and-replay` specifies; **cross-reference that pack for the determinism design; this sheet owns only the fault-schedule generation and the invariant assertions checked at each simulated step.** DST is required treatment at L (where reproducibility of clock-sensitive bugs is non-negotiable) and is the backbone of the XL test strategy.

DST and Jepsen are complementary, not alternatives: DST proves your *own* code correct under controlled faults and is reproducible; Jepsen black-box-checks the *assembled system* (including dependencies you didn't write) against its claimed model. A serious L/XL strategy runs both.

## Production Verification

Pre-production tests check the system you *think* you shipped. Production verification checks the system that is *actually running*, with real traffic, real data volumes, and real dependency behaviour. It is not optional at M+, because some faults (true geographic latency, real dependency outages, data-dependent hot shards) only exist in production.

- **Canary deploys.** Roll the new version to a small slice (one cell, 1% of traffic), run the steady-state hypothesis and invariant checks against it, and auto-rollback on violation before the blast radius widens. The canary is a continuously-running experiment, not a one-time gate.
- **Shadow traffic (dark launch).** Mirror real production requests to the new system without using its responses; compare its outputs (and invariant state) against the live system. Surfaces correctness and ordering divergences under the real input distribution with zero user-facing risk.
- **Continuous verification.** Run lightweight invariant checks against production continuously — a reconciliation job that asserts "every acknowledged write is present in N replicas," "no two leaders," "no saga older than its SLA still pending." These are the same invariants as the test catalog, run forever, against reality. They are the last line that catches what every prior layer missed.

The reconciliation job deserves a design note: it must be cheap enough to run often (so divergence is caught in minutes, not at the next audit), and it must *act* on a violation — page, halt the affected writer, or trigger automated repair — not merely log. A reconciliation check that logs "lost write detected" into a dashboard nobody watches is not verification; it is a record of the outage you are about to have. Treat its alerts at the same severity as a hard outage, because a silent-correctness violation is worse than a crash: a crash fails loud, a lost write fails quiet and propagates.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| Testing only the happy path and mocking the network away | The bug lives in the timing of a failure; mocking the network deletes the entire bug surface | Inject every fault `02-` lists; assert invariants under each |
| "It worked in staging" as the durability argument | Staging has no real partitions, no real clock skew, no real load-induced GC pauses; a clean week proves nothing about the failure model | Run fault injection in CI and chaos in staging; require invariant checks to pass *under fault*, not just at rest |
| Chaos with no hypothesis and no blast-radius limit | Without a steady-state hypothesis there is no pass/fail; without a blast radius it is just an outage you caused | Steady-state hypothesis + bounded blast radius + single variable + abort condition, every experiment |
| Claiming strong consistency with no linearizability/invariant check | An un-verified consistency claim is the exact silent-choice failure this pack exists to catch; the system can violate it on every partition and pass every test | Run Jepsen/Knossos/Elle against the model `01-` claims; one checker per claimed guarantee |
| Killing nodes but never pausing them | Crash-stop expires leases cleanly; the split-brain bug needs pause-then-resume, which kill never produces | Inject SIGSTOP/freeze; assert no-split-brain under pause-then-resume |
| Injecting only symmetric partitions | Asymmetric (A→B works, B→A doesn't) breaks failure detectors that symmetric partitions don't | Inject asymmetric partitions; assert failover correctness |
| Fault injection with no invariant oracle | Tells you only whether it crashed, not whether it lost a write or returned a stale read — most distributed bugs are silent | Pair every fault run with a linearizability or invariant checker |
| Flaky distributed test quarantined and ignored | A "flaky" failure under fault injection is usually a real timing-dependent bug the suite finally caught | Treat a fault-injection failure as a reproduced defect; with DST, the seed reproduces it exactly |
| DST bolted on after the architecture is set | DST requires all I/O/time/randomness behind injectable interfaces; retrofitting is a rewrite | Commit to the injectable-I/O architecture at design time if L/XL; cross-ref determinism pack |

## Spec Output

`12-test-strategy.md` must contain, as a checkable list:

1. **Fault-to-`02-` trace** — a table mapping each fault in `02-failure-model.md` to the test that injects it and the tool used; any `02-` fault with no injecting test is flagged as a gap.
2. **Invariant catalog** — every guarantee from `01-`/`07-`/`08-`/`09-` listed with its concrete invariant, the threatening fault, and the named checker that asserts it.
3. **Linearizability/consistency check** — which model (from `01-`) is checked, with which tool (Jepsen+Knossos/Elle or equivalent), and the generator/operation profile used.
4. **Fault-injection harness** — the CI-level integration suite: nodes, fault schedule, which faults, run frequency (per-PR).
5. **Chaos protocol** — the steady-state hypothesis, blast-radius limit, abort/rollback condition, single-variable rule, and game-day cadence.
6. **DST decision** — whether deterministic simulation is in scope (required at L+); if yes, the seed strategy, the injectable-I/O surface, and a cross-reference to the `axiom-determinism-and-replay` artifacts; if no, the tier-based justification.
7. **Production verification** — the canary policy, shadow-traffic plan (if any), and the continuous-verification invariant jobs running against prod.
8. **Failure response** — what happens when a fault test or invariant check fails: that it is a reproduced defect (with the seed, under DST), not a flake to quarantine.
9. **Tier-coverage statement** — confirmation that the strategy meets the declared tier's requirements (full `12-` and signed-delivery/BFT tests at XL).

A reviewer checks each item off; a missing item fails the consistency gate's test-coverage check for the affected channel.

## When to Re-emit

Re-emit `12-test-strategy.md` when:

- **`02-failure-model.md` changes** — a new fault is admitted (e.g., the system moves to a cloud with VM migrations, adding process-pause). The fault-to-`02-` trace must grow a row, or the gate fails. This is the most common re-emit trigger.
- **A guarantee changes in `01-`/`07-`/`08-`/`09-`** — a new or strengthened guarantee needs a new invariant row and checker. Strengthening `01-` from causal to linearizable *requires* a new linearizability check; the old suite no longer proves the claim.
- **Tier promotion** — crossing into L makes DST and clock-skew testing required; crossing into XL makes signed/authenticated-delivery and BFT tests required. Re-emit with the new mandatory sections.
- **A new replication, sharding, consensus, or saga mechanism is added** (`03-`/`04-`/`05-`/`08-` change) — the new mechanism has its own failure modes and invariants; the strategy must cover them.
- **A production incident reveals an untested interleaving** — fold the reproducing fault schedule into the suite (as a committed seed under DST) so the regression cannot recur silently.

Affected downstream: a `12-` change feeds `99-distributed-system-specification.md`'s consolidated test-coverage section and the consistency gate's per-channel "is there a test/invariant for it" check.

## Cross-References

- [failure-models-and-fallacies.md](failure-models-and-fallacies.md) — `02-` declares the faults; this sheet injects exactly that set. The fault-to-`02-` trace is the contract between the two.
- [consistency-models-and-cap.md](consistency-models-and-cap.md) — `01-` declares the consistency level; the linearizability/invariant checker must check *that* level, no stronger, no weaker.
- [idempotency-and-deduplication.md](idempotency-and-deduplication.md) — `07-`'s no-duplicate-effect guarantee is verified here under duplication and retry injection.
- [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md) — `08-`'s saga-always-terminates invariant is verified here under node-kill-mid-saga injection.
- [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md) — `09-`'s ordering and exactly-once-effect claims are verified here under reorder and duplication injection.
- [resilience-patterns.md](resilience-patterns.md) — `10-` defines the timeouts/retries/circuit-breakers; this sheet's latency and partition injection verifies they behave as specified.
- [cost-and-when-not-to-distribute.md](cost-and-when-not-to-distribute.md) — DST and full chaos infrastructure are a real cost; `13-` records it and is one input to the "don't distribute" decision.
- **`axiom-determinism-and-replay`** — owns the determinism machinery DST depends on (seeded scheduler, I/O substitution, snapshot/restore). This sheet owns the fault schedule and invariant assertions; that pack owns making the simulation deterministic.
- **`axiom-devops-engineering`** — owns where the chaos/fault harness runs in CI/CD and how canary rollouts are wired; this sheet owns what the experiments assert.
