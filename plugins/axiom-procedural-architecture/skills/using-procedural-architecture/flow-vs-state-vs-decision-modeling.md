---
name: flow-vs-state-vs-decision-modeling
description: Analyst-cluster sheet — the five common procedural modelling formalisms (flowchart, state machine, BPMN, decision table, sequence diagram) with what each reveals and hides, a choosing heuristic table, and a worked user-signup example showing the same procedure in three abstractions.
---

# Flow vs State vs Decision Modeling

**Choosing the wrong abstraction for a procedure costs more than choosing none — it leads you to ask the wrong questions about it.**

"Abstraction," as used in this sheet, means *modelling formalism*: a structured notation for representing a procedure on a whiteboard, in documentation, or in a tool. Each formalism foregrounds certain procedural properties and backgrounds others. When it mismatches the procedure's dominant character, the questions you ask of the diagram are questions the diagram cannot answer.

This sheet teaches recognition: given a procedure you need to model, which formalism surfaces what you need to know?

---

## The Five Abstractions

### 1. Flowchart

**What it is.** Boxes are stages or decisions; arrows are transitions; diamond shapes are decision points with labeled exits. The diagram traces a single execution thread from start to end through a branching path.

**Best when:**
- The procedure is primarily sequential with branches.
- Order matters and the sequence of stages is the main thing to communicate.
- State is implicit — the procedure does not need to track where a long-lived entity is between sessions.
- Decisions are the main feature: the diagram reader needs to see which condition leads to which path.

**What it reveals.** The complete set of decision points, the conditions at each branch, and the paths through the procedure. A flowchart is a direct encoding of the procedure's decision structure. Reviewing it against a MECE audit (see `branching-and-mece-review.md`) tells you whether every branch is covered and mutually exclusive.

**What it hides.**
- *Parallelism.* A flowchart has one token, one path. If two stages can execute concurrently, the flowchart cannot represent that without conventions (parallel tracks, swim lanes) that belong to BPMN or Petri nets, not to a basic flowchart.
- *Persistent state.* The flowchart has no notion of an entity pausing between sessions. If a user can abandon mid-procedure and return tomorrow, the flowchart shows the path but not the fact that the procedure has a long-lived suspended state that must be managed.

---

### 2. State Machine

**What it is.** An entity moves through a fixed, named set of states. Transitions between states are triggered by events. Each state is explicit, named, and persistent. The model answers: where is this entity right now, and what events will move it forward?

**Best when:**
- There is a long-lived entity whose current status must be tracked over time.
- State changes are event-driven, not clock-driven.
- The entity can be in different states simultaneously with different obligations: a document can be `draft`, `under_review`, `approved`, or `rejected`, and each status implies different permitted operations.
- The procedure spans multiple sessions, handoffs, or time gaps.

**What it reveals.** The full lifecycle of an entity. Long-lived suspended states — the places where an entity waits for an external trigger — become explicit named states rather than invisible pauses in a flowchart arrow. The transition diagram makes it immediately apparent which states are terminal, which are reversible, and which transitions are one-way. For correctness checks on lifecycle models (termination, reachability, defined uncertainty), see `procedural-invariants-and-correctness.md`.

**What it hides.**
- *Ordering of different entities' work.* A state machine describes what happens to one entity. If the procedure involves multiple entities whose processing must be coordinated — a document that is waiting on an approval that is waiting on a compliance check — the state machine shows each entity's lifecycle in isolation. The sequencing and dependency between entities' state transitions is not visible without additional notation.

---

### 3. BPMN (Business Process Model and Notation)

**What it is.** A formal workflow notation with swim lanes (one per actor or system), gateways (parallel split, exclusive choice, event-based), events (start, intermediate, end), and explicit handoffs between lanes.

**Best when:**
- The procedure involves multiple actors with clear handoffs between them.
- Who owns which stage is the central question.
- Process automation tools (BPMN engines) will execute the model directly.

**What it reveals.** Multi-actor coordination. Who owns which stage. Where the procedure crosses an organisational or system boundary. Parallel gateways make concurrent execution explicit. Event-based gateways model asynchronous triggers (waiting for a message, a timer, an error).

**What it hides.**
- *Numerical reasoning about capacity.* BPMN shows the shape of the flow; it says nothing about how long stages take, how many consumers flow concurrently, or whether stages have sufficient capacity to handle demand. For capacity questions, use queueing theory (`queueing-theory-for-procedures.md`) or DES (`discrete-event-simulation-for-procedures.md`).
- *Concurrency soundness.* BPMN's parallel gateways allow the modeller to create unreachable states or orphan tokens — soundness violations that are not visible from the diagram. For soundness verification, use workflow nets (`process-algebra-and-workflow-nets.md`), which make the same structure formally checkable.

---

### 4. Decision Table

**What it is.** A table where rows are cases (or condition combinations), columns are conditions or actions, and each row reads: "when conditions X, Y, Z hold, take actions A, B, C."

**Best when:**
- The procedure is primarily conditional logic concentrated at one decision point.
- Multiple conditions combine in ways that a flowchart diamond tree would make hard to audit.
- The audience needs to verify completeness and mutual exclusivity of cases against MECE criteria.

**What it reveals.** The complete combinatorial space of conditions and their corresponding actions. Missing rows (unhandled combinations) and conflicting rows (same conditions, different actions) are immediately visible by inspection. For auditing row completeness and mutual exclusivity, see `branching-and-mece-review.md`.

**What it hides.**
- *Ordering.* A decision table has no concept of sequence. It describes what to do given a combination of conditions; it does not describe the stages you pass through to reach that decision point, or what happens after the action is taken.
- *State.* A decision table is stateless: it evaluates inputs at a moment in time and produces outputs. It cannot represent entities with lifecycles, long-lived suspended states, or history-dependent behaviour.

---

### 5. Sequence Diagram

**What it is.** Actors or systems appear as vertical lifelines. Horizontal arrows represent messages (synchronous calls, asynchronous events, returns). Time runs downward. The diagram reads as a script of interactions: who sends what to whom, and in what order.

**Best when:**
- The procedure is fundamentally interactive: multiple actors or systems exchanging messages, with the timing and ordering of those messages as the primary concern.
- You need to expose communication protocols between systems.
- The audience is technical and needs to see the exact message sequence for implementation (API calls, service interactions, protocol specification).

**What it reveals.** The message-level coordination between actors. Which actor initiates which exchange. What responses are synchronous vs asynchronous. The ordering of messages across the system boundary.

**What it hides.**
- *Branching.* A sequence diagram typically shows one scenario (happy path, or one error path). Conditional branches require separate diagrams or inline notation that rapidly clutters the diagram. Complex branching logic belongs in a flowchart or decision table.
- *Persistent state.* A sequence diagram shows what messages are exchanged; it does not show the entity's state between messages. If the entity can be in "waiting for payment confirmation" for three days, the sequence diagram shows the payment-confirmation message arriving but not the three-day suspended state around it.

---

## The Choosing Heuristic

| Dominant procedure character | Preferred abstraction |
|---|---|
| Sequential execution with branches; order is the main concern | Flowchart |
| Long-lived entity with persistent state; lifecycle tracking | State machine |
| Multiple actors with clear handoffs; who owns which stage | BPMN or sequence diagram |
| Conditional logic concentrated at one decision point | Decision table |
| Message exchange between systems; timing and ordering | Sequence diagram |

Most real procedures have more than one dominant character. The question to ask is: *what is the most important thing to make visible?* That determines the primary abstraction. Secondary properties can be modelled separately or deferred to a downstream sheet (see Cross-References).

---

## Worked Example: User Signs Up for a Service

A single procedure — a user creating an account — modelled in three abstractions. The procedure has these stages: collect email and password, verify email address (user clicks a link in an email), collect payment details, activate account.

---

### Abstraction 1: Flowchart

```
[ User enters email + password ]
            |
            v
     < Email valid? >
      Yes /    \ No
         /      \
        v        v
[ Send verification email ]  [ Show validation error ]
        |
        v
< User clicks link? >
   Yes /    \ No (expires after 48 hrs)
      /      \
     v        v
[ Collect payment ]  [ Send reminder; expire pending account ]
     |
     v
< Payment valid? >
   Yes /    \ No
      /      \
     v        v
[ Activate account ]  [ Show payment error ]
     |
     v
  [ Done ]
```

**What this reveals.** The sign-up path and its decision points — two exit conditions (email invalid, payment invalid), one expiry (no click within 48 hours). Branching is auditable.

**What this hides.** The "User clicks link?" diamond is instantaneous on the diagram. The 48-hour suspended state — where the account exists in a pending status waiting for an external event — must be stored, cleaned up, and re-triggered, but the flowchart treats it as a trivial arrow. A developer reading only this diagram may not build the expiry mechanism. The diagram also gives no hint that verification and payment involve separate external systems with their own failure modes.

---

### Abstraction 2: State Machine

Entity: User account.

```
States:
  created
  email_verification_pending
  verified
  payment_pending
  active
  suspended
  churned

Transitions:
  created              --[email submitted]-->          email_verification_pending
  email_verification_pending --[link clicked]-->       verified
  email_verification_pending --[48hr expiry]-->        churned
  verified             --[payment details submitted]--> payment_pending
  payment_pending      --[payment confirmed]-->         active
  payment_pending      --[payment failed]-->            verified       (retry)
  active               --[subscription lapsed]-->       suspended
  suspended            --[payment resumed]-->           active
  suspended            --[90-day no-action]-->          churned
```

**What this reveals.** `email_verification_pending` and `payment_pending` are named states with explicit forward triggers (link click, payment confirmation) and explicit failure or expiry exits. A developer reading this diagram immediately knows which states must be stored and cleaned up. The full lifecycle — including churn and reactivation — is visible.

**What this hides.** The state machine describes one entity in isolation. The question "what does the web app say to the auth service, and in what order?" is not answerable from this diagram; the multi-system coordination is invisible.

---

### Abstraction 3: Sequence Diagram

```
User          Web App         Auth Service    Email Service   Payment Service
  |--POST /signup->|                |               |                |
  |                |--create_user-->|               |                |
  |                |<--user_id------|               |                |
  |                |--send_verify_email------------>|                |
  |                |<--email_sent------------------|                |
  |<--202 pending--|                |               |                |
  |  (48 hours later, user clicks link in email)
  |--GET /verify?token=...|         |               |                |
  |                |--verify_token->|               |                |
  |                |<--verified-----|               |                |
  |<--redirect /payment|            |               |                |
  |--POST /payment->|               |               |                |
  |                |--charge_card---------------------------------->|
  |                |<--charge_ok-----------------------------------|
  |                |--activate_user->|              |                |
  |                |<--activated-----|              |                |
  |<--200 active---|                |               |                |
```

**What this reveals.** The multi-system coordination: four actors (Auth, Email, Payment, Web App) with explicit message ordering. A developer reading this knows the API surface — what the web app calls, in what order. The 48-hour gap between the 202 response and the verification click is visible as a structural pause.

**What this hides.** The branching when payment fails. This is the happy path; "what happens if `charge_card` returns an error?" requires a separate diagram. The suspended waiting state and the question of what happens if the user never clicks are not surfaced here.

---

The right abstraction depends on what must be visible at this stage of the work. Use the flowchart when auditing failure-path coverage; use the state machine when designing account lifecycle storage; use the sequence diagram when specifying the API contract. Use the wrong one and the diagram will answer a question you were not asking while leaving your actual question unanswered.

---

## Cross-References

- [decision-flow-design.md](decision-flow-design.md) — when the procedure is primarily a decision structure: designing decision points for MECE coverage, exit-artifact discipline, and decision-flow composition. Use when the flowchart or decision-table formalism is right and you need to design the decision logic itself.
- [process-algebra-and-workflow-nets.md](process-algebra-and-workflow-nets.md) — when soundness matters: whether the procedure can deadlock, livelock, or leave orphan execution threads. Use when you have chosen BPMN as your formalism and need a proof-level soundness check, not just structural review.
- [queueing-theory-for-procedures.md](queueing-theory-for-procedures.md) — when capacity matters: whether the procedure will scale under load. Use after choosing your modelling formalism; the formalism describes the shape, queueing theory tells you whether that shape holds under volume.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — when the right abstraction is in a downstream pack: continuous-time dynamics, ODE-level state, control theory, or simulation environments that go beyond the procedural modelling formalisms covered here. Document the gap and hand off per that protocol.
