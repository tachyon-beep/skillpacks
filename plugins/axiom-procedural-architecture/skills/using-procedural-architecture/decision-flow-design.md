---
name: decision-flow-design
description: When to force a decision now versus defer it; information-readiness gating; escape-hatch discipline for "Other"; re-asking under state change. Producer-side sheet — owns where decision points are placed and whether they fire. The critic-side mirror is branching-and-mece-review.md.
---

# Decision Flow Design

**A decision asked before its inputs exist is noise; a decision deferred past its impact is debt.**

Every decision point in a procedure has two failure modes: firing too early (the audience cannot meaningfully choose, so they guess or stall) and firing too late (the choice that would have simplified downstream stages was never committed to, so uncertainty propagates). This sheet gives the producer tools to find the right moment for each decision point and define what happens when that moment is disrupted.

---

## 1. Forced-Choice vs. Deferred

**The axis is not "reversible vs. irreversible" — it is cost-of-deferral crossed with cost-of-revision.**

A forced-choice|forced choice commits the audience to an option immediately. A deferred decision holds the option open until inputs are richer. The table below shows when each is correct — neither is a default.

Deferral has its own cost: uncertainty held open blocks downstream stages, forks the procedure's live state, and may force the audience to carry mental reservations through work that should be settled. A decision deferred too long becomes debt — every stage that executes under an unresolved assumption is a stage that may need to be re-executed if the assumption resolves badly.

| Cost of deferral | Cost of revision (reversal cost) | Decision | |
|---|---|---|---|
| Low | Low | Defer until inputs are richer — no urgency and cheap to fix | Defer |
| Low | High | Defer and scaffold — high cost to revise, but deferral is cheap; gather the inputs that make revision unnecessary | Defer + gate |
| High | Low | Force with a sensible default — commit now, cheap to change if wrong | Force |
| High | High | Force and validate — commit early, but run a precondition check first to reduce the chance of needing revision | Force + preflight |

"High cost of deferral" means: downstream stages cannot proceed, options remain open that carry live cost (infrastructure provisioned against unknown topology, legal exposure held open, two engineering teams waiting on a decision), or the audience loses context if forced to return later.

"High reversal cost" means: the choice commits infrastructure, encodes into a schema, generates a paper trail, or requires coordination with other parties to undo.

---

## 2. Information-Readiness Gating

**Every decision point must declare its preconditions; if they are not met, the decision is premature.**

A precondition is a piece of state that must exist before the audience can choose meaningfully. It may come from:

- An earlier stage's exit artifact (e.g., the topology-decision-record that gates replication configuration).
- The audience's declared prior state (e.g., "cloud provider credentials already exist").
- A scan or check that runs as part of the current stage (e.g., a secrets-detection pass whose result conditions a later question).

**If preconditions are not ready, the decision point has two legal options:**

1. **Gather first** — insert a stage that produces the missing input before the decision point fires.
2. **Supply a sensible default** — commit to a default that is safe for most audiences and allow revision later at low cost. This is only valid when reversal cost is low.

What is never valid: asking the audience to choose despite missing inputs, relying on the audience to notice the gap themselves, or silently routing to a default without declaring it. Silent defaults are hidden forced-choices; they violate dependency correctness (see [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md)) and break the audit trail.

**Documenting preconditions.** Each decision point in a decomposition should carry an explicit preconditions list alongside its option set. Example format:

```
Decision point: credential-delivery method
Preconditions:
  - instance-provisioning-record exists (stage 3 complete)
  - org security policy document available to the audience
Options: Secrets manager / Environment variable / Other
```

If a precondition cannot be met in the current session (e.g., the audience does not yet have the security policy), the decision is deferred and the stage that calls for it is held.

---

## 3. Escape-Hatch Discipline

**"Other" is required when the domain is open-ended; it must be refused when every option must route somewhere defined.**

An escape hatch|"Other" in an option set serves a structural function: it catches audience members who fall outside the enumerated choices. Without it, they are trapped. With it but without a defined next stage, they fall off the edge of the procedure. The discipline is in knowing which situation applies.

The asymmetry: an absent "Other" in an under-enumerated option set traps audience members who legitimately fall outside the listed choices. But an "Other" that has no defined downstream handling is an orphan path — the audience chose it and the procedure has no next stage for them.

**When "Other" is required:**

- The domain has legitimate novelty — new technology, non-standard configurations, edge cases that any enumeration will miss.
- The cost of trapping the audience (forcing a wrong option) exceeds the cost of handling divergence (routing to a specialist, deferring the decision, escalating to a human).
- The procedure has a defined escalation path that "Other" can route to.

**When "Other" must be refused:**

- Legal or regulatory forced-choice: every option has a compliance implication; "Other" puts the procedure in an unaudited state. The option set must be complete or the procedure must not be offered.
- Payment-method selection or any downstream stage that requires a concrete type to process: "Other" is not a payment type; the system has no handler for it. Every option must route to a defined handler, full stop.
- Safety-critical routing: a troubleshooting tree where "Other" means "symptom not classified" is dangerous if the next stage requires a specific diagnosis.

**What "Other" always requires when it is permitted:**

- A defined exit artifact (even if minimal: a description field, a reason code).
- A defined next stage (escalation queue, human review, deferred-decision holding state).
- A documented rationale in the exit artifact so downstream stages and the audit trail can reconstruct what happened.

**This sheet owns** the producer decision of whether "Other" should exist in an option set at all. The critic-side audit of whether the *enumerated* options are actually MECE — and whether "Other" is present where coverage requires it — belongs to [branching-and-mece-review.md](branching-and-mece-review.md).

---

## 4. Re-asking Under State Change

**When earlier information is invalidated, the procedure must declare a policy; carrying conflicting state forward silently is never the right policy.**

A state change that invalidates an earlier choice creates a fork in the procedure's history. Four policies exist; each has consequences:

| Policy | Mechanism | Consequence |
|---|---|---|
| **Re-ask** | Present the decision point again with the new context | Correct but may disrupt flow; audience must re-confirm; prior downstream work may be wasted |
| **Silent re-route** | Automatically select the appropriate option given new state | Fast but requires the procedure to know the new selection deterministically; not valid when the choice has audience-policy dimensions |
| **Abort and restart** | Discard all accumulated state from the invalidated point forward and restart the affected stages | Safe but expensive; appropriate when the conflicting state is load-bearing for downstream artifacts already produced |
| **Carry forward with conflict flag** | Mark the conflicting state and continue; surface it at the next gate stage | Only valid when a later gate stage is guaranteed to resolve the conflict before any irreversible action; never valid as a terminal state |

**The procedure document must declare which policy applies at each re-asking point.** "We'll handle it" is not a policy. The policy must be stated before the procedure is deployed, not discovered when the conflict arises.

Conflicts typically arise from three sources: a scan or check later in the procedure finding state inconsistent with an earlier declaration (the classic is a secrets-detected scan after "no secrets" was declared); an external change arriving between stages (a policy update, a provisioning failure, an upstream service topology change); or the audience discovering mid-procedure that an earlier answer was wrong.

---

## 5. Worked Example: CI Pipeline Configuration Wizard

**Three decision points from a single wizard — each analyzed across all four dimensions.**

**Audience parameters:**
- Prerequisites: git repository exists; cloud deployment target identified
- Working-memory capacity: mid-level engineer, unfamiliar with the org's CI tooling
- Error cost: medium (broken CI delays team; not a production data incident)
- Reversibility appetite: medium (willing to revise, prefers not to redo infra work)

---

**Decision 1: Which test runner?**

```
Stage:      language-and-framework confirmation (stage 2)
Decision:   test-runner selection
Preconditions:
  - language confirmed (stage 1 exit artifact: language-declaration-record)
  - framework confirmed (stage 1 exit artifact: framework-declaration-record)
Options: pytest / Jest / RSpec / Other (escalate to team lead)
```

- **Forced or deferred?** Deferred. Language and framework are not known until stage 1 exits; asking test-runner selection before that produces meaningless answers (the audience can only guess). Cost of deferral is low — nothing depends on this until the CI config is generated. Stage 1 is cheap, so deferral is cheap. → **Defer until stage 1 exits.**
- **Preconditions:** language-declaration-record and framework-declaration-record. If either is absent, stage 1 is incomplete; the wizard must not advance to this decision point.
- **"Other" handling:** Required. The ecosystem has legitimate edge cases (Vitest, Bun, custom runners). "Other" routes to a text-capture stage with a prompt to name the runner and a note that the wizard cannot validate its config; the output is flagged for manual review before the CI config is committed.
- **Re-asking:** If the language is later changed (e.g., the audience discovers the repository contains a second language), this decision point is re-asked for the new language. The prior test-runner answer is preserved for the original language; a second answer is appended. The procedure carries two runner declarations as a valid plural state.

---

**Decision 2: Deploy on merge or manual gate?**

```
Stage:      deployment-policy selection (stage 3)
Decision:   deployment trigger
Preconditions:
  - deployment target confirmed (available from audience prior state)
Options: Deploy on merge to main / Manual trigger only
```

- **Forced or deferred?** Forced. This is a project-level policy choice with low reversal cost (a single config value, changeable in minutes) and high cost of deferral (downstream stages for environment credentials and notification wiring depend on whether deployments are automatic or manual — they configure different hooks). Defaulting to "manual gate" is safe for almost all audiences; the choice is cheap to revisit. → **Force with a sensible default (manual gate).**
- **Preconditions:** Only the deployment target confirmed. The wizard has this from the audience's prior state or from a prompt at stage 1.
- **"Other" handling:** Refused. The downstream stage that configures deployment hooks requires a concrete trigger type; there is no handler for "something else." The option set is complete for the domain.
- **Re-asking:** Not applicable. If the audience changes their mind about deployment policy, it is a low-cost config update and not a structural invalidation of downstream work. The procedure documents the chosen value in the deployment-policy-record and notes it is revisable.

---

**Decision 3: Which secrets backend?**

```
Stage:      secrets configuration (stage 4)
Decision:   secrets-backend selection
Preconditions:
  - secrets-scan result exists (stage 3 exit artifact: secrets-scan-report)
  - audience has answered: does this project use secrets at all?
Options: Vault / AWS Secrets Manager / GitHub Secrets / No secrets in this pipeline
  (if precondition: secrets detected → "No secrets" option is suppressed)
```

- **Forced or deferred?** Deferred, then forced with a preflight. The question cannot be asked until the secrets-scan runs (precondition). Once the scan result is in hand, the choice is forced — downstream stages for credential injection differ completely by backend, and leaving it open blocks all of them. High reversal cost (migrating a secrets backend mid-pipeline is a coordination event) means the question must be scaffolded before committing. → **Defer until scan exits; then force with preflight (review scan findings before presenting options).**
- **Preconditions:** secrets-scan-report from stage 3. If the scan has not completed, this decision point does not fire.
- **"Other" handling:** Refused for standard pipelines. Every backend option routes to a concrete credential-injection template; "Other" has no template and produces an inoperable CI config. If the org's actual backend is absent from the list, the wizard must be extended — the option set is a system invariant, not an audience choice.
- **Re-asking under state change:** This is the hard case. If the audience declares "no secrets" at stage 2 but the secrets-scan (stage 3) finds an API key committed in `.env.example`, the "No secrets in this pipeline" declaration is invalidated. Policy: **abort the decision point, re-ask.** The wizard presents the scan findings with severity, retracts the "no secrets" exit artifact from stage 2, and re-asks the decision. Any downstream work from stage 2's artifact (e.g., a draft CI config generated under the "no secrets" assumption) is discarded. Carrying the conflicting state forward — generating a CI config that has no credential injection for a pipeline that demonstrably needs it — is a security defect, not a workflow shortcut.

---

## Cross-references

- [decomposition-fundamentals.md](decomposition-fundamentals.md) — reversibility-ordered staging and the reversal-cost convention that feeds the forced-vs-deferred matrix; dependency correctness, which is the structural property that information-readiness gating enforces.
- [branching-and-mece-review.md](branching-and-mece-review.md) — the critic-side mirror: audits whether an option set is actually MECE, whether "Other" is missing where coverage requires it, and whether branches are fake. This sheet decides whether a decision point should exist and what its preconditions are; that sheet checks whether the resulting option set is well-formed.
- [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md) — adversarial preconditions audit: checks that every input a stage or decision point consumes is produced by an earlier stage or declared prior state; catches premature-commitment and hidden-coupling violations that information-readiness gating exists to prevent.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — the question of what the *right* options are at a decision point (which secrets backend is technically superior, which test runner the team should standardize on) is domain-content judgment, not procedural-architecture judgment; that boundary is defined here and the handoff protocol is in that sheet.
