---
name: queueing-theory-for-procedures
description: Analyst-cluster sheet — Little's Law, M/M/1 utilization intuition, and bottleneck identification for multi-stage procedural flows. Teaches when closed-form queueing earns its cost over structural review alone. Worked 3-stage approval flow with utilization per stage, bottleneck identification, and three redesign options.
---

# Queueing Theory for Procedures

**Structural review tells you whether the shape is right. Queueing theory tells you whether the shape will scale.**

A procedure whose stage graph is structurally sound — dependencies declared, grain consistent, decision points properly gated — can still fail at production volume. The failure mode is not architectural; it is capacity. Consumers arrive faster than stages can serve them, queues build, and wait time climbs. Queueing theory is the lens that makes that failure predictable from first principles before it becomes a crisis.

---

## When This Earns Its Cost

**Earns its cost when:** stages have finite capacity or service time, multiple consumers flow through simultaneously, and backlogs are a feature of the system rather than a transient. Code-review queues, support-ticket pipelines, multi-stage approval flows, intake processes, content-moderation pipelines — any procedure where demand is continuous, where different consumers share the same stage capacity, and where a consumer who arrives when a stage is busy waits rather than being immediately served. In these contexts, the structural properties of the procedure (which this pack audits first) are necessary but not sufficient; without utilization data, you cannot distinguish a sound procedure that is merely under-resourced from one whose staging is wrong.

**Does not earn its cost when:** each consumer's path is independent, there is no shared stage capacity, and there is no queue to form. One-shot setup wizards, single-user troubleshooting trees, individual onboarding flows — procedures where one consumer executes in isolation. A second consumer running the same wizard concurrently is not competing with the first for stage capacity; they are running independent instances. Applying M/M/1 reasoning to such procedures produces numbers that mean nothing. Run the structural audit from `decomposition-fundamentals.md` instead; utilization is not the question.

---

## Three Concepts at Depth

### 1. Little's Law: L = λW

Little's Law is a relationship between three observables in any stable system that has consumers flowing through it:

- **L** — average number of consumers in the system (either waiting or being served) at any moment
- **λ** (lambda) — average arrival rate (consumers per unit time)
- **W** — average time a consumer spends in the system (wait time plus service time)

The law: **L = λW**

The intuition is easier than the algebra. Imagine standing at the entrance to a stage and watching for exactly one minute. On average, λ new consumers arrive. Each of them will spend W minutes in the system. The number currently inside is therefore λ × W — the same arrivals per minute, each occupying W minutes of system time. No distributional assumptions required; Little's Law holds for any arrival process and any service-time distribution as long as the system is stable (arrivals do not permanently exceed capacity).

**Working an example from one minute of observation.** You time a triage stage for one hour. Sixty tickets arrive (λ = 1/minute). At any moment you glance at the queue, an average of 3 tickets are either waiting or being worked. L = 3. Little's Law: W = L/λ = 3/1 = 3 minutes. That is the average total time each ticket spends in the triage stage. You did not need to time individual tickets; you measured L and λ and the law gave you W.

The practical consequence: **if W is too long and you cannot directly observe it, measure L and λ instead**. Both are countable from observation; W falls out algebraically. This is why Little's Law is a diagnostic tool, not merely a theoretical result.

---

### 2. M/M/1 Utilization Intuition

The M/M/1 model describes a single-server stage with random (Poisson) arrivals at rate λ and exponentially distributed service times with rate μ (average service time = 1/μ). Its key insight is not its closed-form derivations — those are in any operations-research textbook — but its behavior near capacity.

**Utilization** is the fraction of time the server is busy:

**ρ = λ / μ**

If arrivals arrive at 10 per hour (λ = 10) and the server can handle 20 per hour (μ = 20), utilization is ρ = 0.5 — the server is busy half the time, and queues stay short. The M/M/1 result for average wait time in queue is:

W_q = ρ / (μ(1 − ρ))

The critical feature of this formula is the **(1 − ρ)** in the denominator. As ρ approaches 1, that denominator approaches zero, and wait time approaches infinity. The blowup is nonlinear:

| Utilization (ρ) | Relative wait time (arbitrary units) |
|-----------------|--------------------------------------|
| 0.50            | 1×                                   |
| 0.70            | 2.3×                                 |
| 0.80            | 4×                                   |
| 0.90            | 9×                                   |
| 0.95            | 19×                                  |
| 0.99            | 99×                                  |

**80% utilization is fine. 95% utilization is on fire.**

The move from 80% to 90% does not add 10 percentage points of wait; it more than doubles it. The move from 90% to 95% doubles it again. Procedures that feel acceptable at moderate load become unworkable under modest increases in demand, not because the structure changed but because the utilization knee is approached from below.

This is why capacity decisions require more than "add more consumers" instinct — you need to know where on the ρ curve the bottleneck stage currently sits, because the intervention cost scales differently depending on whether you are at ρ = 0.75 or ρ = 0.92.

---

### 3. Bottleneck Identification

In a multi-stage flow, the bottleneck is the stage with the highest utilization. This is not a heuristic; it is a consequence of flow conservation. All consumers that enter the flow must pass through every required stage. The stage that is busiest relative to its capacity is the one that limits throughput for the entire flow.

**Capacity added at a non-bottleneck stage is wasted** — it increases the capacity of a stage that is not the constraint. The flow rate does not increase because the bottleneck stage is still the binding limit. Adding reviewers to stage 2 when stage 3 is at 95% utilization improves nothing; tickets pile up faster in front of stage 3.

**Decisions belong at the bottleneck.** Before spending on capacity (headcount, tooling, additional servers), compute utilization per stage. The investment goes where ρ is highest. Everything else is theater.

The identification procedure is mechanical:

1. For each stage, measure or estimate λ (arrival rate), μ (service rate), and c (number of servers/agents/reviewers).
2. Compute ρ_stage = λ / (c × μ).
3. The stage with the highest ρ is the bottleneck.
4. If ρ_bottleneck > 0.85, the procedure is operating in the nonlinear wait-time region; any further increase in arrival rate will produce disproportionate queue growth.

---

## Worked Example: 3-Stage Approval Flow

A team runs a software-change approval procedure with three sequential stages: triage → review → sign-off. Every change that enters triage goes through review and then sign-off. The team is complaining that approvals are slow but cannot agree on where to invest. You have one week of data.

### Given

| Stage        | Arrival rate λ   | Service time (mean) | Servers c |
|--------------|------------------|---------------------|-----------|
| Triage       | 60 / hour        | 2 minutes           | 1         |
| Review       | 60 / hour        | 4 minutes           | 4         |
| Sign-off     | 60 / hour        | 5 minutes           | 1         |

*Note: λ is 60/hour at every stage because every change that enters triage proceeds through the full flow.*

### Computing Utilization Per Stage

**Stage 1 — Triage:**
- λ = 60/hr = 1/min; service time = 2 min, so μ = 0.5/min; c = 1
- ρ = λ / (c × μ) = 1 / (1 × 0.5) = **2.0**

A utilization above 1.0 means the stage cannot handle the arrival rate at current capacity. The triage stage, with one server handling two-minute tasks at one arrival per minute, is *overwhelmed* — ρ = 2.0. The queue at triage grows without bound.

Wait — that appears to contradict the team's experience of "approvals are slow" rather than "approvals completely fail." Let's recheck: 60/hr = 1/min arrival, 2 min service time, 1 server. The server would need to process 1 item per minute but takes 2 minutes each. That is genuinely ρ = 2.0, unstable. This is the diagnosis: the team likely has informal overflow handling (senior engineers triaging ad hoc, items waiting overnight) that masks but does not resolve the structural overload.

**Stage 2 — Review:**
- λ = 1/min; service time = 4 min, so μ = 0.25/min; c = 4
- ρ = λ / (4 × 0.25) = 1 / 1.0 = **1.0**

Four reviewers handling 4-minute reviews at one arrival per minute: ρ = 1.0. This stage is at the knife-edge of capacity — any variance in arrivals or service time will create a queue, and wait times diverge. The stage is at the knee.

**Stage 3 — Sign-off:**
- λ = 1/min; service time = 5 min, so μ = 0.2/min; c = 1
- ρ = λ / (1 × 0.2) = 1 / 0.2 = **5.0**

Sign-off with one server handling 5-minute tasks at one arrival per minute: ρ = 5.0, completely unstable.

### What the Math Is Telling You

The bottleneck is not where the team thought it was. Both triage (ρ = 2.0) and sign-off (ρ = 5.0) are structurally overloaded — their utilization exceeds 1.0, meaning queues grow without bound under sustained load. Review (ρ = 1.0) is at the knife-edge of capacity: exactly balanced at current staffing, where any variance tips it into overload.

When ρ ≥ 1.0 the system is not in steady state; M/M/c wait-time formulas no longer apply. The numbers above are load-factor readings, not steady-state utilizations — they tell you the structural mismatch (this stage cannot absorb the arrival rate at current staffing) but not the wait time, which grows without bound as long as the overload persists.

The immediate consequence: adding reviewers to stage 2 (the obvious target because reviewers are visible and their time is observable) does nothing. The approval flow is blocked before items reach review and blocked again before they exit. Items will accumulate in front of triage and in front of sign-off regardless of how fast review runs.

### Three Redesign Options

**Option A: Add capacity at the bottleneck stages.**

For triage (ρ = 2.0): requires a second triage server to reach ρ = 1.0, or a third to reach ρ = 0.67 (comfortable). For sign-off (ρ = 5.0): requires five sign-off approvers to reach ρ = 1.0, or six to reach ρ = 0.83. Cost is real — headcount or time allocation. Gain is proportional and predictable: halving the bottleneck ρ halves the wait contribution from that stage. This is the right answer when the structural staging is sound and the only issue is under-investment relative to arrival rate.

**Option B: Batch differently to parallelize the bottleneck stage.**

Sign-off at ρ = 5.0 suggests sign-off is happening on every individual change. If the sign-off decision is stateless — each change is evaluated independently — then batching weekly does not reduce the per-item service time but reduces the coordination overhead and may allow async sign-off rather than synchronous. Alternatively, if sign-off can be delegated to reviewers above a confidence threshold, the arrival rate at sign-off drops below 1/min. This is safe when sign-off's value is marginal for high-confidence reviews; it is not safe when sign-off is a regulatory gate. If sign-off is capturing something review is not, batch or delegate. If sign-off is capturing exactly what review already captured, move to Option C.

**Option C: Restructure the staging itself.**

Triage at ρ = 2.0 means one server spending two minutes on every incoming change. Ask: what is triage actually doing that review could not do? If triage is a routing decision (assign to reviewer pool), and the review stage already handles capacity-based routing internally via the four-server pool, then triage as a discrete stage is adding overhead without adding value proportional to its cost. Collapsing triage into review intake — having each reviewer claim from a shared queue directly — eliminates a stage and removes ρ = 2.0 from the flow entirely.

The same question applies to sign-off: inspect the exit artifact of the review stage before investing in sign-off capacity. If review's exit artifact already captures what sign-off is checking, sign-off is structurally redundant — a decomposition smell, not a capacity problem. Eliminating a redundant stage beats adding capacity to it.

This is the structural redesign option: when the bottleneck stage exists because of how the procedure was decomposed, not because of how much work it does, restructuring the decomposition beats adding headcount. Apply this before investing in capacity; see `decomposition-fundamentals.md` for the grain-consistency check that would have caught these issues in design.

---

## Boundary to Discrete-Event Simulation

The M/M/1 and M/M/c models assume Poisson arrivals and exponentially distributed service times. Real procedural flows frequently violate both:

- **Non-exponential service times.** Code review does not have memoryless service time; a large change takes longer than a small one, and reviewer expertise affects duration in non-random ways. Exponential distribution is a thin-tailed model that underestimates the frequency of long reviews.
- **Complex routing.** If some changes route from triage directly to sign-off (skipping review), or if rejected changes loop back to a prior stage, the M/M/c model no longer applies.
- **State-dependent arrivals.** If the team slows submissions when the queue is visibly long, the Poisson arrival assumption breaks.

When any of these conditions hold, closed-form queueing gives answers with false precision. The right tool is discrete-event simulation, which models each consumer individually, draws service times from empirical distributions, and handles complex routing directly. See `discrete-event-simulation-for-procedures.md` for when DES earns its cost and how to structure the model.

The practical test: if you can describe the flow with a utilization table (arrival rate, service rate, server count per stage) and the distributions are roughly exponential, use queueing theory. If you need to capture service-time variance, routing dependencies, or behavior-under-load feedback, use DES.

---

## Cross-References

- [discrete-event-simulation-for-procedures.md](discrete-event-simulation-for-procedures.md) — when closed-form queueing fails: non-exponential service times, complex routing, state-dependent arrivals; how DES addresses each limitation.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — when the question moves from staged-discrete queueing to continuous-time dynamics, control theory, or formal verification; handoff to `yzmir-simulation-foundations` for ODE-level reasoning.
- [decomposition-fundamentals.md](decomposition-fundamentals.md) — the structural-soundness prerequisite: a procedure must satisfy the five structural properties before utilization analysis is meaningful; queueing theory applied to a structurally broken decomposition produces accurate numbers for the wrong shape.
