---
name: procedural-invariants-and-correctness
description: Critic-side correctness gate for a proposed decomposition — five structural invariants (termination, definedness of exit artifact, reachability, no implicit carried state, defined "I don't know" branch), each with its check, failure mode, and a concrete violation example. Supplies a minimal numbered checklist that an auditor can run mechanically against any decomposition before approving it. Defers to process-algebra-and-workflow-nets.md for formal verification in safety-critical contexts.
---

# Procedural Invariants and Correctness

**Before declaring a decomposition done, run a minimal set of correctness checks. They catch the failures that no amount of polishing the prose will catch.**

A structurally tidy-looking decomposition — stages named, exit artifacts declared, decision points enumerated, smells triaged — can still fail the moment a consumer executes it. The five invariants in this sheet are not about polish or grain; they are about soundness. A decomposition that violates any one of them is broken in a way that will surface at runtime, silently or loudly. The checklist in this sheet is the last gate before handoff.

Run this sheet after `decomposition-smells.md`, `dependency-and-ordering-audit.md`, and `branching-and-mece-review.md` have been applied and their findings have been actioned. This sheet is not a substitute for those passes — it is the correctness gate that follows them.

---

## The Five Invariants

---

### Invariant 1: Termination

**Check.** Every execution path through the decomposition reaches a declared exit stage. There is no path that loops indefinitely or returns to a prior stage without a declared convergence condition.

**Failure mode.** Infinite loops from unguarded retries: a stage that fails routes back to itself or to an earlier stage with no progress condition and no maximum-attempt declaration. Escalation chains that cycle — stage A fails, routes to stage B for remediation, stage B fails, routes back to stage A — with no path that exits the cycle.

**Violation example.** An "identity verification" stage routes to a "retry submission" stage on failure. "Retry submission" routes back to "identity verification." Neither stage declares a maximum attempt count, a divergent path after N failures, or a terminal error state. The consumer is in a declared loop with no exit. A consumer following the decomposition literally cannot determine when to stop.

**The check to run.** Graph the decomposition as a directed graph of stages and decision point branches. Starting from the start stage, enumerate all execution paths. Mark a path as terminated when it reaches a stage that declares itself a procedure end state. Flag every path that re-enters a previously visited stage without a declared convergence condition (a finite loop counter, a conditional that must eventually evaluate differently, or a sub-procedure with its own termination guarantee).

---

### Invariant 2: Definedness of Exit Artifact

**Check.** Every stage produces an exit artifact that is defined well enough for the next consumer to use it. No stage "completes" while producing an undefined, absent, or ambiguously specified artifact.

**Failure mode.** A stage whose completion criterion is "stage is done" without specifying what artifact it produces. A stage that produces an artifact under the happy path but produces nothing — or an undeclared partial artifact — on a non-happy path. A downstream stage that expects an artifact in a specific format, and an upstream stage that produces one in a different format, with no declared transformation between them.

**Violation example.** Stage 4 of a "configure database" procedure is named "Apply schema migrations." Its declared exit artifact is "migrations applied." Stage 5, "Verify schema integrity," expects an artifact of "schema version: v3.2.1 confirmed." Stage 4 produces a completion signal but does not produce a schema version record — the format that Stage 5 needs. Stage 5 cannot begin from Stage 4's output. The exit artifact is defined by name but not by the structure the downstream stage requires.

**The check to run.** For each stage, read the exit artifact declaration and then read every downstream stage that consumes it. For each downstream consumer, verify that what the upstream stage produces is sufficient for what the downstream stage declares as its required input. Any mismatch — format, content, presence on non-happy paths — is a definedness failure.

---

### Invariant 3: Reachability

**Check.** Every stage in the decomposition is reachable from the start stage by following declared transitions. No stage exists that cannot be arrived at by any execution path.

**Failure mode.** Orphan-state smell formalised: a stage that was added during drafting, survives revision, but is never wired as the successor of any other stage or decision point branch. A stage reachable only through a branch that the declared audience parameters make impossible — a branch that fires only when a precondition is never true for the declared audience.

**Violation example.** A "license renewal" procedure contains a stage named "Enterprise override approval." Examining the decision point branches: this stage is the successor of a "license type = enterprise_legacy" branch. The procedure's declared audience parameters specify the procedure applies only to individual and small-team licenses. The "enterprise_legacy" branch can never be taken by the declared audience. The "Enterprise override approval" stage is unreachable in practice, silently bypassed, and will atrophy — its exit artifact declaration will fall out of sync with the actual enterprise path when one is eventually added.

**The check to run.** Graph traversal: from the start stage, enumerate every reachable stage by following all declared transitions including error paths and all branches at every decision point. Every stage in the decomposition's stage list that is not present in the reachable set is an orphan. For stages reachable only through branches, verify the branch condition is achievable given the declared audience parameters.

Cross-reference: `decomposition-smells.md` smell 7 (orphan-state) provides the canonical diagnostic signal and false-positive caveat for this invariant.

---

### Invariant 4: No Implicit Carried State

**Check.** Every input a stage uses is either produced by an explicitly named earlier stage or declared as an audience parameter supplied before the procedure begins. No stage silently depends on state that is not declared in its input list and not traced to a producer.

**Failure mode.** Hidden coupling: stage B behaves differently depending on whether stage A ran before it, but stage B does not declare stage A's output as an input and stage A does not declare the dependency as part of its exit artifact. The procedure works in practice because the stages happen to run in the right order — until someone skips a stage, reruns a stage out of sequence, or the environment changes in a way that breaks the implicit dependency.

**Violation example.** A "deploy application" procedure has Stage 3 "Set environment variables" and Stage 7 "Run smoke tests." Stage 7 declares its inputs as "server address" and "test suite path." Stage 7 is actually testing against environment variables that Stage 3 set — if Stage 3 is skipped because the environment was previously configured, the smoke tests pass against stale configuration and the deployment is silently wrong. Stage 3's side effect (environment variable state) is not declared in Stage 7's input list and not declared as part of Stage 3's exit artifact. The coupling is invisible.

**The check to run.** This is the consumer-side mirror of Check 4 (no hidden coupling) from `dependency-and-ordering-audit.md`, applied as an invariant gate rather than an advisory finding. For each stage, enumerate all state it consumes: declared inputs, environment state, service state, file state, configuration state. For each item, trace it to either a named earlier stage's exit artifact or a named audience parameter. Any consumed state with no traceable producer is implicit carried state. Severity is not reduced because the procedure happens to work in the current environment — the invariant is violated regardless.

Cross-reference: `dependency-and-ordering-audit.md` Check 4 is the sibling audit procedure; this invariant mirrors its no-hidden-coupling check at the correctness-gate level.

---

### Invariant 5: Defined "I Don't Know" Branch

**Check.** Every decision point in the decomposition has a declared branch for the case where the consumer cannot decide. No decision point leaves the consumer at a dead end when the information needed to choose is absent, ambiguous, or in conflict.

**Failure mode.** A decision point with two or three options that collectively assume the consumer has a clear, definitive answer. No branch for "I have conflicting information," "the required input was not produced," "I am not qualified to decide this," or "the decision point preconditions are not met." The consumer arrives, cannot choose any declared branch, and has no declared path forward — the procedure is effectively terminated at that decision point without a declared exit.

**Violation example.** A "network access provisioning" procedure has a decision point "Is the requester's role senior-engineer or above?" with branches "Yes — proceed to elevated access stage" and "No — proceed to standard access stage." A requester with a role of "contractor-senior" arrives. Their seniority is ambiguous under the org's current role taxonomy. Neither branch is clearly applicable. The procedure offers no "I don't know / ambiguous case" branch — no escalation path, no default, no hold-for-clarification state. The consumer stops. A decision point without a defined uncertainty path is a potential procedure dead-end for any input that is genuinely ambiguous.

**The check to run.** For every decision point, enumerate its declared branches. Verify that one of the following is present: (a) a branch explicitly labeled for the case where the consumer cannot determine which branch applies, (b) a declared default that applies when no other branch condition is decidable, or (c) a declared escalation path that routes the undecidable case to a stage where additional information is gathered or a qualified decision-maker resolves it. A decision point with only substantive branches and no uncertainty path fails this invariant.

---

## The Minimal Checklist

Run this checklist against the complete decomposition after all five invariants are understood. Each item is a yes/no question. A "no" answer is a correctness failure — record the failing item, identify the specific stage or decision point involved, and treat it as a blocking finding before approving the decomposition.

1. Does every execution path reach a declared exit stage? If not, name the paths that loop or terminate without a declared procedure end.

2. For every stage, on every execution path including non-happy paths, does the stage produce an exit artifact that is defined well enough for its downstream consumers to use? If not, name the stage and the mismatch with the downstream consumer's input declaration.

3. Is every stage in the stage list reachable from the start stage by following declared transitions — including all decision point branches and error paths — given the declared audience parameters? If not, name the unreachable stages and the branch condition that makes them unreachable.

4. For every stage, are all of the state items it consumes either (a) declared inputs traced to a named earlier stage's exit artifact, or (b) audience parameters declared before the procedure begins? If not, name the stage and the specific state items with no traceable producer.

5. For every decision point, is there a declared branch or path for the case where the consumer cannot determine which branch applies? If not, name the decision point and describe the undecidable case it lacks a branch for.

An auditor who can answer "yes" to all five items has verified the minimal correctness properties of the decomposition. A decomposition that passes this checklist is not guaranteed to be well-designed — grain, ordering optimality, and audience calibration are separate concerns — but it is structurally executable.

---

## When Formal Verification Earns Its Cost

For most procedures, the checklist above is sufficient. The five invariants are structural properties that can be checked by inspection and graph traversal against the declared decomposition.

Two contexts justify the cost of formal verification:

**Regulated procedures.** When a procedure is subject to regulatory audit, a claim of correctness based on informal inspection is often insufficient. A formal model provides a verifiable artifact that survives external scrutiny.

**Safety-critical procedures.** When a termination failure, a definedness gap, or an implicit carried state dependency results in harm — not inconvenience, but harm — the informal checklist is not a strong enough argument. Deadlock freedom, liveness, and reachability need formal proof, not assurance by inspection.

For both contexts, `process-algebra-and-workflow-nets.md` is the correct next sheet. It formalises termination and reachability as soundness and safeness properties of workflow nets, and provides the model-checking entry point for regulatory and safety-critical procedure validation.

---

## Cross-references

- [decomposition-smells.md](decomposition-smells.md) — orphan-state (smell 7) is the pattern-level version of Invariant 3 (reachability); re-entrancy-blindness (smell 9) is a related structural gap that this sheet does not duplicate. Use that catalog for smell naming and false-positive calibration.
- [process-algebra-and-workflow-nets.md](process-algebra-and-workflow-nets.md) — the formal verification path for regulated and safety-critical procedures; soundness (no deadlock, proper termination) and safeness (no duplicate tokens / state explosion) are the formal counterparts of Invariants 1 and 3 in this sheet.
- [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md) — sibling critic sheet; its Check 4 (no hidden coupling) is the ordering-specific version of Invariant 4 (no implicit carried state). The two sheets are complementary: the audit sheet finds hidden coupling as an ordering defect; this sheet calls the same property a correctness invariant that blocks handoff.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — when a correctness failure's remediation requires domain-content judgment (what the undecidable case should route to, whether a given state item is truly implicit or legitimately out of scope), document the gap and hand off per the protocol in that sheet.
