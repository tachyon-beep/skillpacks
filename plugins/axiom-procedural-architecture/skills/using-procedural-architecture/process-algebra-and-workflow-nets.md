---
name: process-algebra-and-workflow-nets
description: Analyst-cluster sheet — when formal procedural verification earns its cost (regulated, safety-critical, concurrency-heavy) vs the informal checklist in procedural-invariants-and-correctness. Workflow nets at recognition depth: three soundness properties (proper completion, no orphan tokens, no dead transitions), ASCII Petri-net diagrams, and a worked incident-response example that exposes an orphan-token violation the informal checklist would have missed. Process calculi (CSP/CCS/π-calculus) at recognition level only.
---

# Process Algebra and Workflow Nets

**When "this procedure cannot deadlock" must be provable, not just plausible, reach for the formal model.**

A structurally sound decomposition — invariants checked, smells cleared, dependencies declared — is a strong argument for correctness, but it is still an argument by inspection. There is a class of procedure where that is not good enough: procedures where a soundness failure carries legal liability, human harm, or regulatory consequence. For those procedures, formal verification replaces the auditor's judgment with a machine-checkable proof. This sheet teaches when to pay that cost and what the model looks like at recognition depth.

---

## When This Earns Its Cost

**Earns its cost:**

- **Regulated processes** — finance, healthcare, aviation. A claim that a procedure "cannot deadlock" made to a regulator must survive external audit. An informal checklist is an assurance; a verified workflow-net model is a proof. When the audit standard requires demonstrable correctness, the model provides the artifact.

- **Safety-critical procedures** — incident response, deployment rollback, emergency shutdown. When a termination failure does not mean an annoying Slack message but means a service stays degraded, a rollback hangs in a liminal state, or a patient is not discharged from a stage they should have exited, the informal checklist is insufficient. These procedures need deadlock freedom and liveness proven, not assured by inspection.

- **Procedures with concurrency interactions that hand-analysis cannot enumerate** — when two or more execution paths interact through shared decision points or shared exit artifacts, the combinatorial space of interleavings grows faster than an auditor can trace. Formal models enumerate all of it.

**Does not earn its cost:**

- **Internal team workflows** where the worst-case soundness failure is an annoying Slack message asking "what do we do now?" The cost of building and maintaining a formal model exceeds the cost of the occasional deadlock. Use `procedural-invariants-and-correctness.md`'s informal checklist instead.

- **One-shot user flows** where retry-from-scratch is acceptable. If the entire procedure restarts cleanly on failure with no accumulated state and no external consequence, formal proof is overhead without payoff.

The earn-it gate is a three-question checklist. It follows after the model description.

---

## Workflow Nets in 200 Words

A **workflow net** is a Petri-net specialization designed for modelling procedures. The key constraint: exactly one **source place** (the procedure start) and exactly one **sink place** (the procedure end). Between them, **transitions** fire when their input places hold **tokens**, consuming input tokens and producing output tokens. A token is a marker — it represents an active case, a consumer, or a thread of execution moving through the net.

The vocabulary:

- **Place** — a state a case can be in. Drawn as a circle.
- **Transition** — an event or activity that moves a case between states. Drawn as a box.
- **Token** — a marker indicating a case currently occupies a place. Drawn as `•`.

A minimal sequential procedure:

```
(• start )
    |
    v
[ validate ]
    |
    v
( validated )
    |
    v
[ approve ]
    |
    v
( end )
```

A token begins in `start`. When `validate` fires, the token moves to `validated`. When `approve` fires, the token moves to `end`. The procedure terminates. That is the happy path. Soundness asks whether it always works.

### The Three Soundness Properties

A workflow net is **sound** if and only if all three properties hold for every possible execution:

**1. Proper completion.** Starting from one token in the source place, it is always possible to reach a state where one token is in the sink place. Every execution path terminates.

**2. No orphan tokens at termination.** When the sink place holds a token, no tokens remain anywhere else in the net. There are no leftover tokens in side-branches or intermediate places — no dangling execution threads.

**3. Every transition is reachable.** From the source place with one token, every transition in the net can fire in at least one execution. There is no dead code in the net — no transition that can never fire regardless of the execution path taken.

If any of these three properties fails, the net is unsound. A net that satisfies properties 1 and 3 but violates property 2 has an orphan-token problem: the procedure "completes" but leaves a thread of execution unresolved — exactly the failure the worked example below exposes.

---

## Process Calculi in 100 Words

**CSP, CCS, and π-calculus** are algebraic frameworks for modelling concurrent processes. They describe systems where two or more execution threads interact through synchronized communication. This sheet teaches **recognition, not authoring**: when someone reaches for a process calculus, they are modelling concurrent interaction — parallel threads that must synchronize — not sequential flow. If your procedure is sequential (one token, one path, no parallelism), a workflow net is the right model and process calculi are overkill. If the procedure has genuinely parallel threads that must synchronize at join points, that is when process calculi become relevant. Milner's *Communicating and Mobile Processes* and Hoare's *Communicating Sequential Processes* (both freely available) are the standard entry points for actual modelling.

---

## The Earn-It Gate

Before investing in formal modelling, answer three questions:

1. **Is there a regulatory requirement?** Does the applicable standard (SOX, ISO 13485, DO-178C, or equivalent) require demonstrable procedural correctness rather than self-attested correctness?

2. **Does failure-cost exceed formalization-cost?** Estimate the cost of a soundness failure (legal exposure, human harm, remediation, reputational damage) and the cost of building and maintaining the formal model. If failure-cost is not clearly larger, skip formalization.

3. **Are there concurrency interactions that hand-analysis cannot enumerate?** Count the number of distinct execution paths including parallel branches. If the space is enumerable by inspection in an hour, run the informal checklist. If it is not, the formal model earns its cost by covering the cases you cannot trace manually.

**If all three answers are "no," skip formalization.** Use `procedural-invariants-and-correctness.md`'s five-invariant checklist instead. It catches termination failures, reachability gaps, and implicit carried state — the same structural problems the formal model proves, without the modelling overhead that an informal context does not justify.

---

## Worked Example: Incident-Response Workflow Net

### The Procedure

A simplified incident-response procedure with a fast-path rollback option:

1. **Detect** — an alert fires.
2. **Triage** — determine severity.
3. On high severity: **rollback** immediately. On low severity: **investigate** first.
4. After rollback: **verify** the system is stable.
5. After investigation: **patch** the root cause, then verify.
6. After verify: **close** the incident.

### Initial Net (with the bug)

```
(• detect )
     |
     v
[ triage ]
     |
   /   \
  /     \
high   low
  |     |
  v     v
[roll] [investigate]
  |     |
  v     |
(rolled)(investigating)
  |     |
  v     |
[verify][patch]
  |     |
  v     v
(verified)(patched)
  |     |
  +--+--+
     |
     v
  [close]
     |
     v
  ( end )
```

### The Soundness Violation

Trace the low-severity path:

1. Token starts in `detect`.
2. `triage` fires → token moves. Because severity is low, the token goes to `investigating`.
3. `patch` fires → token moves to `patched`.
4. At the join before `close`, the net merges both the `verified` and `patched` places. In the low-severity path, `verified` never receives a token — `verify` was on the rollback branch only.
5. `close` requires tokens in **both** `verified` and `patched` (it is an AND-join). On the low-severity path, `verified` is empty. `close` cannot fire.

**Result: deadlock.** The token sits in `patched` forever. The incident is never closed.

This is an **orphan-token violation of soundness property 2** combined with a **liveness failure of soundness property 1**: `close` cannot fire because the AND-join requires a token the low-severity path never produces. A smell-walk of the prose procedure would likely not catch this — the two paths look structurally parallel and complete. The formal model makes the missing token explicit.

### The Fix

Add a `verify` stage to the low-severity path, or convert the join before `close` from an AND-join to an OR-join (whichever matches the procedure's actual intent):

```
(• detect )
     |
     v
[ triage ]
     |
   /   \
  /     \
high   low
  |     |
  v     v
[roll] [investigate]
  |     |
  v     v
(rolled)(investigating)
  |     |
  v     v
[verify][patch]
  |     |
  v     v
(verified)(patched)
  |     |
  +--+--+
     |
   [join]   ← OR-join: fires when either token arrives
     |
     v
  [close]
     |
     v
  ( end )
```

Now both paths converge cleanly at an OR-join. `close` fires when either path's token arrives. Soundness property 2 holds: when the sink fires, no orphan tokens remain. Soundness property 1 holds: both paths reach the sink. Soundness property 3 holds: every transition is reachable — the high-severity path reaches `rollback` and `verify`; the low-severity path reaches `investigate` and `patch`.

**The key takeaway:** the orphan-token failure existed in the original prose procedure. It was invisible to smell-walk because both branches "complete" in the narrative sense. The workflow net exposed it by requiring every token to be consumed before the sink can fire.

---

## Cross-References

- [procedural-invariants-and-correctness.md](procedural-invariants-and-correctness.md) — the informal correctness path; five invariants (termination, definedness, reachability, no implicit carried state, defined uncertainty branch) that cover the same structural properties without a formal model. Use this first; escalate to the present sheet only when the earn-it gate triggers.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — when a soundness failure's remediation requires domain-content judgment about what the procedure should actually do at the failing decision point; document the gap and hand off per that protocol.
- [queueing-theory-for-procedures.md](queueing-theory-for-procedures.md) — a different analyst question: not "can this procedure deadlock?" but "will this procedure keep up with demand?" Use queueing theory when the question is capacity; use workflow nets when the question is soundness.
- [discrete-event-simulation-for-procedures.md](discrete-event-simulation-for-procedures.md) — a third analyst question: realistic throughput under non-exponential service times and complex routing. DES and workflow nets address orthogonal concerns; both may be warranted for a safety-critical high-volume procedure.
