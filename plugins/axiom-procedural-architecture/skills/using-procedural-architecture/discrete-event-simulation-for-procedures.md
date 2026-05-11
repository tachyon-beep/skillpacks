---
name: discrete-event-simulation-for-procedures
description: Analyst-cluster sheet — when DES earns its cost over closed-form queueing, the three questions DES answers well (throughput under realistic distributions, sensitivity, what-if redesigns), a static-review/queueing/DES boundary table, and a worked support-ticket pipeline with fast-track lane showing why M/M/1 misleads.
---

# Discrete-Event Simulation for Procedures

**DES answers questions about procedural flow that no closed-form math can.**

When a procedure has non-exponential service times, state-dependent routing, or complex priority rules, the elegant closed forms of queueing theory become comfortable lies — they give precise-looking numbers for a system that does not behave the way the model assumes. Discrete-event simulation models each consumer individually, advances time event by event, and lets you characterise the full distribution of outcomes across many replications. That power has a cost. Before paying it, be deliberate about whether the question actually requires it.

---

## When This Earns Its Cost

**Earns its cost when:** service times are non-exponential, routing is state-dependent, or priority rules are complex. Examples: a procedure where stage 2's service time depends on the type of work that came out of stage 1 (so service time is not memoryless — knowing you spent 15 minutes in stage 1 tells you something about stage 2); a flow with priority lanes where high-priority work jumps the queue (the M/M/c model has no concept of priority discipline); a pipeline with rework loops where items re-enter an earlier stage on failure (closed-form models assume a consumer passes through each stage at most once); batching behaviour where work is held until a minimum batch size accumulates; or blocking and overflow where a full downstream queue causes an upstream stage to stall. In all of these situations, the Markovian assumptions of standard queueing models break down in ways that are not conservative — they can produce numbers that are misleading in either direction.

**Does not earn its cost when:** the system is simple enough that queueing theory's closed form gives a good answer. If arrivals are roughly Poisson, service times are roughly exponential, routing is deterministic and linear, and there are no priority lanes, then `queueing-theory-for-procedures.md` gives you Little's Law and M/M/c utilization in minutes with no model-building overhead. If the question is not about capacity dynamics at all — if you need to know whether the procedure shape is correct, whether stages are in the right order, whether decisions are properly gated — then the critic cluster answers it with structural review. DES is the right tool when queueing theory's assumptions fail and the question is a flow question, not a shape question.

---

## What DES Actually Is

A discrete-event simulation maintains a clock and an event queue. The clock jumps from event to event — arrivals, service starts, service completions, departures — skipping idle time rather than advancing in fixed increments. Each consumer is an entity with its own state (priority class, time of arrival, work type, stages visited, rework count). The model draws service times from a specified distribution for each stage and routes each entity according to rules that can depend on that entity's state and the current system state.

**This sheet teaches recognition, not implementation.** The goal is knowing when DES is the right tool and what questions to put to it — not knowing how to build a DES model. For actual model-building, general-purpose simulation libraries handle the mechanics: SimPy (Python, process-oriented) and ciw (Python, network-of-queues interface) are well-suited to procedural flows. The investment in learning either library is justified once you have confirmed that the flow question cannot be answered with queueing theory's closed forms.

The key methodological discipline is **replication**. A single simulation run is a sample path — one possible history. To characterise the distribution of outcomes (average wait, 95th-percentile wait, throughput variance), you run the model N times with different random seeds and summarise across runs. Reporting the output of a single run as if it were a distributional answer is a common misuse of DES.

A second discipline is **warm-up**. DES starts from an empty system, which is rarely the state of a real pipeline at the moment of interest. Statistics collected during the warm-up period — before the model reaches its operational steady state — contaminate the output. The standard practice is to discard the first portion of each replication before collecting measurements. The length of the warm-up period is a modelling decision, not a default; it depends on how long the system takes to reach its typical load level from empty. Ignoring warm-up produces optimistic wait-time estimates because the system is less congested at the start than under sustained load.

---

## The Three Questions DES Answers Well

### 1. Throughput Under Realistic Distributions

What is the 95th-percentile wait when service times are not memoryless? Exponential service time is thin-tailed: it underestimates the frequency of very long service events. Real procedural stages often have heavy tails — the occasional complex case takes many times longer than the median. DES lets you fit an empirical distribution to observed service times (lognormal, Weibull, or a kernel density estimate from historical data) and run the model with that distribution. The result is a wait-time distribution per stage, per priority class, and across the end-to-end flow. Queueing theory gives you an average; DES gives you a distribution with meaningful percentiles.

### 2. Sensitivity

Which parameter, if it changed, changes the answer most? This is the analyst's core question before making an investment recommendation. DES supports it directly: hold all other parameters constant, vary one (arrival rate, fast-track fraction, mean service time at one stage), and observe the change in the output distribution. Parameters to which the output is highly sensitive are where measurement effort and process intervention should concentrate. Parameters to which the output is insensitive can be estimated loosely without affecting the quality of the recommendation. Sensitivity analysis with DES replaces the analyst's intuition with a reproducible ranked list.

### 3. What-If Redesigns

What happens if you split a stage in two, add a dedicated fast-track lane, change the priority threshold, or add a rework gate? Structural changes that alter routing rules or stage topology are straightforward to model in DES and impossible to handle cleanly in closed-form queueing — each structural change would require deriving a new analytical model. In DES, you modify the routing rules or stage configuration and re-run. The comparison between baseline and redesign runs is your evidence base for the redesign decision. This is the reason to invest in a DES model even when queueing theory could answer the steady-state question: if you know you will be evaluating multiple redesign options, the DES investment amortises across all of them.

---

## The Static-Review / Queueing / DES Boundary Table

| Question type | Right tool |
|---|---|
| Is the procedure shape sound? Are stages in the right order? Are decision points properly gated? Are dependencies declared? | Static review — critic cluster (`decomposition-fundamentals.md`, `dependency-and-ordering-audit.md`) |
| What is average wait under steady state, Poisson arrivals, roughly exponential service, linear routing? | Queueing theory — `queueing-theory-for-procedures.md` |
| What is the 95th-percentile wait under bursty arrivals, non-exponential service, or priority lanes? | DES — this sheet |
| What happens under a specific structural redesign (add fast-track lane, add rework gate, change priority threshold)? | DES — this sheet |
| Which input parameter most affects end-to-end wait? | DES sensitivity analysis — this sheet |
| What happens at execution time under continuous dynamics, control theory, or ODE-level state? | `yzmir-simulation-foundations` (outside this pack) — see `procedural-boundary-and-handoffs.md` |

The table has a simple read: when the question is about shape, use structural review; when the Markovian assumptions hold, use queueing theory; when they fail or the question involves redesign options, use DES; when the question leaves the discrete-event regime entirely and enters continuous dynamics, hand off to `yzmir-simulation-foundations`.

A common error is reaching for DES when queueing theory is sufficient, then investing weeks in model-building to confirm what the M/M/c formula would have told you in an hour. The boundary table is a pre-flight check: ask which row your question sits in before deciding which tool to open.

---

## Worked Example: Support-Ticket Pipeline with Fast-Track Lane

### Setup

A support team handles tickets via a single-agent pipeline. The arrival distribution is Poisson (arrivals are independent and random). Tickets arrive at λ = 20 per hour in the baseline; a lower-load variant (λ = 6/hr) is used to illustrate the steady-state case without overload. Two priority classes:

- **Normal tickets** (80% of arrivals, λ_n = 16/hr): mean service time 10 minutes, exponentially distributed. The routing rule is first-come-first-served within this class.
- **Fast-track tickets** (20% of arrivals, λ_f = 4/hr): mean service time 3 minutes, exponentially distributed. The priority discipline is pre-emptive: a fast-track arrival goes to the head of the line ahead of all waiting normal tickets.

The agent handles one ticket at a time.

### Why Queueing Theory's Closed Form Misleads Here

A naive M/M/1 analysis ignores priority classes and treats all tickets as identical. Aggregate arrival rate λ = 20/hr. Weighted mean service time = 0.8 × 10 + 0.2 × 3 = 8.6 minutes, so μ ≈ 7 tickets per hour. Utilization ρ = 20/7 ≈ 2.86 — the system is overloaded in this parameterisation, which is deliberately extreme to illustrate the point.

Back up to a feasible parameterisation: λ = 6/hr, same priority split (λ_n = 4.8/hr, λ_f = 1.2/hr), same service times. Weighted mean service time = 8.6 min = 0.143 hr, μ ≈ 7/hr. ρ = 6/7 ≈ 0.86. M/M/1 average wait in queue: W_q = ρ / (μ(1 − ρ)) ≈ 0.86 / (7 × 0.14) ≈ 0.88 hr ≈ 53 minutes. The M/M/1 model reports one number: everyone waits about 53 minutes on average.

**This single average conceals the priority dynamic.** Fast-track tickets jump the queue; normal tickets absorb the displacement. The average masks a distribution where fast-track tickets have very short waits and normal tickets have substantially longer waits than the average suggests. A team that designs SLAs from the M/M/1 average will overpromise for normal tickets and may not notice until normal-ticket satisfaction data arrives.

The M/M/1 model also has no concept of the fast-track fraction as a parameter. At 20% fast-track, normal tickets are displaced occasionally. If fast-track demand grows to 40% — a plausible shift if the fast-track criteria are loosely defined — normal wait time climbs steeply. The closed-form model cannot surface this sensitivity at all; it has no parameter to vary.

### What a DES Model Would Reveal

A DES model of this system — single agent, two priority classes, pre-emptive priority discipline — produces the following across replications.

*(The model inputs: one agent, Poisson arrivals at λ = 6/hr total, exponential service time per class as specified above, pre-emptive head-of-queue discipline for fast-track. Run 1,000 replications of 2,000 simulated hours each, discard first 200 hours as warm-up.)*

**Wait-time distributions per class.** Fast-track tickets: low mean, low variance, short 95th-percentile (they skip the queue). Normal tickets: higher mean, high variance, long 95th-percentile that is materially worse than the M/M/1 average. The distribution is the answer; the M/M/1 average is not.

**Sensitivity to fast-track fraction.** Hold arrival rate and service times constant. Vary the fast-track fraction from 5% to 50% in increments. Plot normal-ticket 95th-percentile wait against fast-track fraction. The result is typically nonlinear: normal wait is manageable up to some fraction, then climbs steeply as fast-track pre-emptions become frequent. That inflection point is the operational threshold — the maximum fast-track fraction the team can accommodate before normal-ticket SLAs become structurally unachievable.

**Redesign options evaluated in the same model:**

- *Dedicated fast-track agent*: split the single agent into two roles — one handles only fast-track, one handles only normal. Fast-track wait drops to near-zero (dedicated server, low utilization). Normal wait drops because fast-track pre-emptions are eliminated. The cost is idle time on the fast-track agent when fast-track demand is low. DES quantifies the idle-time penalty at different fast-track fractions, giving the team a break-even point: is the SLA improvement on normal tickets worth the idle-time cost of a dedicated agent at current fast-track volume?

- *Non-pre-emptive priority*: fast-track tickets go to the head of the queue but do not interrupt a ticket currently in service. This reduces fast-track wait only slightly (the pre-emption penalty per ticket is one service time, and mean service time is 3 minutes for fast-track) while meaningfully reducing the tail variance for normal tickets. A DES comparison of pre-emptive vs non-pre-emptive discipline at the same parameters makes this tradeoff visible.

- *Raising the fast-track threshold*: if the classification criteria are tightened so that only 10% of tickets qualify as fast-track, the pre-emption frequency drops and normal-ticket 95th-percentile wait falls. DES sensitivity analysis on the fast-track fraction is the direct evidence for this policy decision.

### The Shape of the DES Question

The right way to frame a DES analysis of this system is: run 1,000 replications of the baseline model, extract the wait-time distribution per priority class, identify the 95th percentile for normal tickets, then run the same 1,000 replications under each redesign option. The comparison across replications is the evidence base. Do not report means from a single run; report distributional summaries from replicated runs.

The question the team needs answered — "are our SLAs achievable under realistic load with this priority discipline?" — requires DES.
The queueing-theory average cannot answer it because:

- The priority discipline violates M/M/1's assumptions (no concept of pre-emptive classes).
- The team's real exposure is in the tail, not the average — a mean-wait SLA hides normal-ticket degradation.
- The fast-track fraction is an operational policy variable whose sensitivity cannot be probed with a single closed-form.

---

## Cross-References

- [queueing-theory-for-procedures.md](queueing-theory-for-procedures.md) — when to use closed-form queueing instead: Poisson arrivals, exponential service, linear routing, no priority lanes. Little's Law and M/M/c utilization analysis. The boundary section in that sheet describes when its assumptions break and DES becomes necessary.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — when the question leaves the discrete-event regime entirely: continuous-time dynamics, ODE-level state, control theory. Handoff to `yzmir-simulation-foundations` for that class of question.
- [process-algebra-and-workflow-nets.md](process-algebra-and-workflow-nets.md) — sibling analyst sheet for soundness verification: whether the procedure can deadlock, livelock, or reach an unresolvable state. Structural correctness is a prerequisite before performance analysis is meaningful; verify soundness first, then apply DES.
