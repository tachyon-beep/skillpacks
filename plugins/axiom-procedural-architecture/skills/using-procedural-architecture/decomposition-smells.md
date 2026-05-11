---
name: decomposition-smells
description: Canonical catalog of nine decomposition smells (god-step, mystery-step, decision-without-information, audience-amnesia, ladder-of-trivials, premature-commitment, orphan-state, fake-branch, re-entrancy-blindness), each with definition, diagnostic signal, false-positive caveat, and recommended remediation. This is the definitive source other critic and producer sheets defer to for smell names, severity calibration, and fix patterns.
---

# Decomposition Smells

**Smells are not bugs; they are evidence that something is likely wrong and needs verification.**

A smell is a structural pattern that correlates with defects. When you find one, you have not found a bug — you have found a place where a bug is probable. The smell may be fine: context may explain it, the audience parameters may justify it, the alternatives may be worse. But "probably fine" is not verification. A smell demands that you look deliberately and either confirm the defect or record why the pattern is acceptable here.

The nine smells in this catalog are the first-pass audit for any proposed decomposition. They are a checklist, not a verdict. Work through the list, note each smell present, then evaluate each finding against the procedure's context. A decomposition with three smells that are all justified is better than one that looks clean because nobody checked.

---

## How to Use This Catalog

Run the catalog as a structured pass after the producer delivers a draft decomposition. For each smell:

1. **Identify**: Is the structural pattern present?
2. **Apply the diagnostic signal**: Does the signal fire in this decomposition?
3. **Apply the false-positive caveat**: Is the pattern explained by a legitimate design choice?
4. **If not**: Apply the remediation.

When a smell maps to an ordering defect found by `dependency-and-ordering-audit.md`, or a branching defect found by `branching-and-mece-review.md`, cite the smell name in that sheet's finding. The smell's severity guidance takes precedence over the generic tiers in the sibling sheets when they conflict.

Document your smell findings in the same format as those sheets when the output will be handed back to a producer:

```
Smell: [smell name]
Stage reference: [stage name and position]
Signal fired: [which diagnostic signal applies]
False-positive applies: [yes/no — if yes, state why and close the finding]
Remediation: [specific action; see individual smell entries for patterns]
```

---

## The Nine Smells

---

### 1. God-Step

**Definition.** A single stage that hides multiple responsibilities — multiple decision points, multiple exit artifacts, or multiple failure modes — behind one name. The consumer cannot complete it in one focused attempt because "one focused attempt" has no fixed ending.

**Diagnostic signal.** Ask the target consumer: "When is this stage done?" If they cannot answer in one sentence without listing sub-stages, or if the stage's exit artifact requires a paragraph to describe, the stage is a god-step. Secondary signal: the stage has more than one failure mode that leads to different recovery paths.

**False-positive caveat.** A stage may have substantial internal complexity that is legitimately invisible to the consumer. "Provision Kubernetes cluster" is a god-step by name but acceptable when: (a) the audience is opinionated about cluster shape, (b) the implementation hides the sub-steps correctly behind a single tool call or script with a single verifiable exit artifact, and (c) the failure modes collapse into a single "cluster provisioning failed" state with a documented recovery path. The test is not internal complexity — it is whether the consumer must manage that complexity explicitly. If they execute one command and verify one artifact, the stage's grain is correct from their perspective regardless of what happens underneath.

**Remediation.** Split the god-step into constituent stages, each with a single exit artifact and a single unambiguous completion criterion. Alternatively, if the complexity is correctly hidden, expose that hiding explicitly: name the tool, script, or API that encapsulates the sub-steps, specify the machine-verifiable exit artifact format, and collapse the failure modes into a declared error state with a documented recovery path. Do not split a stage that is already correctly encapsulated — splitting it would produce a ladder-of-trivials. The distinction is whether the consumer must make sub-decisions; if not, the encapsulation is doing its job.

---

### 2. Mystery-Step

**Definition.** A stage whose purpose cannot be stated — a stage that describes what it does mechanically but not why it exists in the procedure. The audience can follow the instructions but cannot judge whether the stage succeeded, whether it is necessary, or what to do when it fails unexpectedly.

**Diagnostic signal.** Read the stage description to someone unfamiliar with the domain. If they can follow the instructions but still do not know what the stage is for, the stage is a mystery-step. A tighter signal: if the documentation describes mechanics ("run this command," "edit this file") without stating the outcome the stage achieves in the procedure's larger goal, the why is missing.

**False-positive caveat.** A stage whose purpose is genuinely obvious to the entire realistic audience is not a mystery. "git commit" in a procedure for experienced software engineers does not need a why explanation — the audience universally understands what committing achieves. The smell fires when the audience must reason about the why to execute correctly, not when the author finds the why obvious. Evaluate against the declared audience parameters: if the audience's prerequisites include full familiarity with the stage's purpose, the why can be omitted without creating a mystery.

**Remediation.** State the why in one sentence: "This stage ensures that [downstream stage] has [specific precondition] so that [outcome]. Without it, [failure mode]." If you cannot write that sentence, the stage may be vestigial — a historical artifact with no current function. Investigate: run the procedure without the stage in a test environment and observe whether the outcome changes. If it does not, consider deletion. "We've always done this" is not a purpose; it is a prompt to verify that the stage is still doing what it was originally intended to do.

---

### 3. Decision-Without-Information

**Definition.** A decision point that is forced before the inputs it depends on exist. The audience is asked to choose without the information that would make the choice meaningful, producing a decision that is either a guess or a forced default that must be revisited later.

**Diagnostic signal.** The consumer asks "how am I supposed to know this?" at the decision point. More formally: for each precondition of the decision point, check whether a prior stage has produced it or the audience's declared prerequisites include it. If a precondition cannot be traced to a producer, the decision is asking for information that does not exist yet.

**False-positive caveat.** Some decisions are deliberately early because they are cheap to revise and act as a shaping constraint on subsequent stages. Choosing a project name before detailed design is complete is not a decision-without-information smell — the name is a low-reversal-cost choice that can be changed cheaply, and the procedure needs it to proceed. The smell fires when the decision is expensive to revise and the missing information is materially relevant to the choice. A decision that the consumer can safely make with defaults and revisit without cost is not premature in the dangerous sense; it is premature-commitment (smell 6) only if revision is costly.

**Remediation.** Identify which earlier stage would produce the missing precondition. If that stage exists but is ordered after the decision point, reorder (see `dependency-and-ordering-audit.md` for the precondition-producer trace). If no stage produces the precondition, add an information-gathering stage before the decision point. If the decision is blocking and the information cannot be obtained at this point in the procedure, replace the decision with a default that the consumer can override later — declare the override path explicitly, not as an afterthought.

---

### 4. Audience-Amnesia

**Definition.** A decomposition that forgets its declared audience parameters mid-flow — a procedure that was designed for one audience but, at some stage, assumes prerequisites, working-memory capacity, or error tolerance that belong to a different audience.

**Diagnostic signal.** A novice consumer who could follow the first half of the procedure cannot enter stage N without out-of-band knowledge. An expert consumer is guided through stages that patronize their competence. More precisely: identify the declared audience parameters (prerequisites, working-memory capacity, error cost, reversibility appetite) and verify that every stage in the procedure is consistent with those parameters. A stage that implicitly raises the required prerequisites above the declared level is the amnesia point.

**False-positive caveat.** A deliberate audience shift within a procedure is not audience-amnesia — it is an intentional escalation path. If the procedure declares "at stage 8, you need a senior engineer to review the following decision," the audience has not been forgotten; an explicit handoff has been declared. The smell fires when the audience shift is implicit — when the procedure silently assumes competence at stage N that was not declared in the audience parameters and was not acquired through earlier stages.

**Remediation.** Re-derive the grain and content of the offending stage from the audience parameters as declared at the start of the procedure. If the stage genuinely requires a different audience, declare that explicitly as a handoff: name the new audience, state what they need to know, and describe how control returns (or whether it does). If the stage can be reformulated to match the declared audience without losing its function, reformulate it — add scaffolding stages before high-error-cost actions, provide explicit exit artifacts at the working-memory level of the declared audience, and remove the silent competence assumption. Full audience parameter definitions and their grain implications are in `audience-modeling-for-procedures.md`.

---

### 5. Ladder-of-Trivials

**Definition.** An over-decomposed sequence of stages each doing so little that the consumer would execute them all in a single mental beat. The procedure's grain is too fine for the declared audience, producing noise that obscures the actual structure.

**Diagnostic signal.** The consumer reads consecutive stages and thinks "I would just do these together — they're one move." Formally: if three or more consecutive stages each have exit artifacts that the target consumer would produce in a single continuous action without pausing to verify intermediate state, the sequence is a ladder-of-trivials cluster.

**False-positive caveat.** Fine grain is correct for high-error-cost, low-working-memory audiences. What looks like a ladder-of-trivials for a senior engineer is correct granularity for a novice who needs explicit intermediate verification — and required for an LLM agent that needs machine-verifiable checkpoints to avoid drift. The smell fires only when the grain is finer than the declared audience actually needs: when the audience's working-memory can comfortably span the cluster and the error cost is low enough that intermediate verification adds no safety value. Evaluate against the declared audience parameters before merging.

**Remediation.** Identify the ladder cluster — the consecutive trivial stages — and merge them into a single stage with a richer exit artifact that incorporates all the intermediate artifacts as components or verification criteria. The merged stage's name should describe the coherent unit of work, not enumerate its sub-steps. Verify the merged stage passes the grain-size test: can the target consumer execute it in one focused attempt with no ambiguity about what "done" looks like? If not, the merge went too far — a god-step is waiting on the other side. Full grain calibration guidance is in `granularity-calibration.md`.

---

### 6. Premature-Commitment

**Definition.** An expensive, difficult-to-reverse decision made before the cheap information-gathering stages that would make it informed. The consumer commits to an artifact whose reversal cost is high before the procedure has provided the scaffolding needed to make that commitment safely.

**Diagnostic signal.** The consumer makes a decision at stage N, and at stage N+K (K ≥ 1) they encounter information that, had they known it at stage N, would have changed their choice — but reversing the stage-N artifact is now expensive. The structural signal: rank stages by reversal cost and check whether the ranking is non-decreasing through the decomposition. A high-reversal-cost stage that precedes a lower-reversal-cost stage that would have produced information relevant to it is a premature-commitment.

**False-positive caveat.** Some commitments must precede some information-gathering by necessity — the information-gathering requires the committed infrastructure to exist. Registering a domain name before designing the site architecture is a genuine premature commitment in abstract, but it is operationally unavoidable and the reversal cost (changing the domain) is predictable and accepted. The smell fires when the early commitment was avoidable — when reordering or adding a low-cost information-gathering stage before the commitment would have meaningfully reduced the risk of a wrong commitment. "We needed to commit early to learn what we were committing to" is occasionally true; apply it skeptically.

**Remediation.** Identify the cheap information-gathering stage that would scaffold the commitment and move it before the commitment stage. If no such stage exists in the decomposition, add one. If the commitment cannot be deferred without breaking the procedure's logic, add an explicit cheapening mechanism: a provisional commitment with a declared revision window, a reversibility gate before the commitment stage ("if you are uncertain, do not proceed — escalate"), or a lower-cost proxy commitment that can be upgraded. The `dependency-and-ordering-audit.md` Check 3 (cheap-decisions-early, expensive-decisions-gated) is the systematic tool for finding premature-commitment instances.

---

### 7. Orphan-State

**Definition.** A stage with no path in, no path out, or both. A stage that cannot be reached from the procedure's start by following declared transitions, or from which no transition leads toward the procedure's declared end state.

**Diagnostic signal.** Graph traversal from the procedure's start stage cannot reach the orphan. Alternatively, traversal from the orphan stage cannot reach any end state. In less formal decompositions: a stage appears in the list but is never referenced as the next stage after any other stage, or it has no defined next stage. The smell also fires for conditional paths — a stage reachable only through a branch that the declared audience parameters make impossible.

**False-positive caveat.** A stage may be deliberately unreachable from the happy path but reachable from an error path. "Rollback" stages are the canonical example: they appear in the procedure as error-path destinations but are not on the forward traversal from start to success. The test is whether the stage is reachable from the full graph — including error transitions — and whether all declared paths from it lead somewhere meaningful. An error-path stage with a declared inbound transition (from the error condition) and a declared outbound transition (to a recovery state or a procedure end) is not an orphan. The smell fires when there is genuinely no declared transition in or out.

**Remediation.** For a stage with no path in: either add the missing inbound transition from the stage that logically precedes it, or delete the stage if it belongs to an earlier draft and was never correctly connected. For a stage with no path out: declare the next stage explicitly, or if the stage is a terminal state, mark it as a procedure end state with a declared exit artifact. Do not leave a stage without explicit connectivity under the assumption that the consumer "will know what to do next" — that assumption is audience-amnesia in transition form.

---

### 8. Fake-Branch

**Definition.** A decision point whose options all converge — all branches lead to the same next stage with no meaningful difference in the exit artifact, the stage behavior, or the downstream routing. The decision consumed the consumer's attention without performing a decision.

**Diagnostic signal.** Trace the execution path from each option of the decision point forward. Identify the first point at which paths produce a detectable difference: different stage behavior, different exit artifact content, or different downstream routing. If all options converge before producing any detectable difference, the branch is fake. Partial fake branch: some options converge immediately while others diverge — the converging pair is a fake branch; the diverging option is legitimate.

**False-positive caveat.** A decision that affects only the current stage's behavior but legitimately produces identical downstream routing is not a fake branch — it is a branch that collapses after exactly one stage. If "option A" and "option B" cause stage N to behave differently, produce artifacts with different content, and then both route to stage N+1, the decision was real: the exit artifacts carry the consequence of the choice even though the routing did not diverge. The smell fires only when the decision produces no detectable consequence at any stage — not now, not downstream.

**Remediation.** Delete the decision point if no genuine difference exists. If the options are intended to produce different behavior but currently do not, implement the divergence — make the exit artifacts actually differ, or route to genuinely different downstream stages. If the decision is meaningful but the branches converge quickly, consider whether the options should be collapsed into a configurable parameter of a single stage rather than a branching structure. The `branching-and-mece-review.md` Check 4 — Fake Branches provides the detailed audit procedure for finding and characterizing fake branches in complex decision point sets.

---

### 9. Re-Entrancy-Blindness

**Definition.** A procedure with no handling for back-and-retry — a procedure that assumes single-pass execution and does not define what happens when a consumer must re-enter a stage after a failure, a partial completion, or a change of mind.

**Diagnostic signal.** The procedure assumes each stage runs exactly once and succeeds on first attempt. No stage declares whether re-entry is idempotent, requires a restart from a defined state, or is an error. When a consumer fails at stage N and attempts to re-enter, they have no declared guidance — they must guess whether to restart stage N from scratch, resume from where they left off, or begin the entire procedure again.

**False-positive caveat.** Some procedures legitimately have no re-entry semantics because re-entry is genuinely impossible or meaningless. A single-pass setup wizard run by an automated provisioner in a clean environment, where failure means the entire environment is discarded and the wizard runs fresh, does not need re-entry declarations — the audience parameters define a context where re-entry does not occur. The smell fires when the declared audience parameters admit the possibility of re-entry — when the audience could reasonably fail mid-procedure and want to resume — and the procedure provides no guidance. Evaluate the audience's error cost and recovery options: audiences with high error cost and limited recovery options need re-entry semantics most urgently.

**Remediation.** For each stage, declare one of three re-entry behaviors:
- **Idempotent**: the stage can be re-run from the beginning at any time and will produce the same exit artifact without harmful side effects. State this explicitly — "safe to re-run."
- **Restart from defined state**: the stage is not idempotent, but re-entry is possible if the consumer first restores a declared prior state (delete the partial artifact, revert a configuration, restore a backup). Declare the restore procedure.
- **Error — do not re-enter**: the stage cannot be re-entered and failure requires escalation or procedure termination. Declare the escalation path.

Add these declarations to the stage's exit artifact specification or as a dedicated re-entry note below each stage. Procedures with irreversible stages (premature-commitment candidates) need re-entry declarations most urgently — an irreversible stage that is also re-entrancy-blind is a double structural defect.

---

## Cross-references

- [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md) — the critic-side ordering audit; its Check 2 (premature commitment) and Check 4 (hidden coupling) surface instances of the premature-commitment and orphan-state smells. Use this catalog for severity calibration and smell naming when assembling findings from that audit.
- [branching-and-mece-review.md](branching-and-mece-review.md) — the critic-side branching audit; its Check 4 (fake branches) is the detailed audit procedure for the fake-branch smell. A fake-branch finding from that sheet should be cited against this catalog entry for authoritative severity guidance.
- [granularity-calibration.md](granularity-calibration.md) — the producer-side grain calibration sheet; its two pathologies (god-step and ladder-of-trivials) are the producer's framing of smells 1 and 5 in this catalog. The granularity-calibration sheet provides worked examples of what each pathology looks like at three audience levels; this catalog provides the adversarial diagnostic signals and false-positive caveats.
- [audience-modeling-for-procedures.md](audience-modeling-for-procedures.md) — the six audience parameters (prerequisites, working-memory capacity, error cost, reversibility appetite, latency tolerance, recovery options); authoritative for evaluating whether audience-amnesia (smell 4) has occurred and whether the re-entrancy-blindness (smell 9) false-positive caveat applies.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — when a smell's remediation requires domain-content judgment that is out of scope for this structural audit, document the gap and hand off per the protocol in that sheet.
