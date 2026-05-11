---
name: audience-modeling-for-procedures
description: The six canonical audience parameters — prerequisites, working-memory capacity, error cost, reversibility appetite, latency tolerance, recovery options — with definitions, elicitation guidance, decomposition impact, the YAML declaration template, and a worked deploy-to-staging example showing the parameter-to-design-choice mechanism for three distinct audiences.
---

# Audience Modeling for Procedures

**Every decomposition has an audience. Treating the audience as implicit is the single most common procedural-design failure.**

A procedure designed for "the user" is designed for nobody. The same goal — deploy a service to staging, configure a database, onboard a new team member — decomposes differently depending on who is executing it. Not just finer or coarser in grain (see [granularity-calibration.md](granularity-calibration.md) for that) but different in which stages exist, which decision points fire, how much defensive scaffolding each stage carries, whether stages are synchronous or async, and whether recovery instructions are built in or omitted entirely. Audience is not a soft preference; it is an engineering parameter with structural consequences.

This sheet defines the six audience parameters, gives elicitation guidance for each, and shows how each one drives specific design choices. The YAML declaration block must be filled out before any decomposition begins; the Consistency Gate in [SKILL.md](SKILL.md) treats an undeclared audience as a blocking defect.

---

## The Six Audience Parameters

### 1. Prerequisites

**Prerequisites are everything the consumer already knows, has access to, or has completed before the procedure begins.**

Prerequisites bound what the decomposition may assume without a stage. A stage that uses a tool, reads an artifact, or makes a decision that relies on knowledge not in the prerequisites list is hiding a dependency — it will silently fail for any consumer who arrives without that prerequisite.

**Elicitation.** Ask three questions: What tools and credentials must already exist? What conceptual knowledge does the consumer need to interpret outputs? What earlier procedures must have completed? The third question is easy to miss: "has deployed to this service before" is a prerequisite that rules out a wide class of confusion about what "deployment acknowledged" looks like.

**Decomposition impact.** A prerequisite that exists means the corresponding stage can be omitted. A prerequisite that is absent means a stage must exist to produce it — or the procedure must refuse to start. Do not paper over absent prerequisites with a vague "ensure you have X before starting"; name the stage that produces X, or declare it explicitly as out-of-scope with a hard precondition check at the procedure's entry.

**Worked signals:**
- Senior engineer has the deployment manifest format memorized → the stage "review manifest schema" can be omitted; a link to the schema is sufficient.
- LLM agent has the manifest schema in its tool catalog → the stage is omitted entirely; the schema is in-context.
- Novice on first day has neither → a stage must exist that walks through the schema field by field, with example values.

---

### 2. Working-Memory Capacity

**Working-memory capacity is how many open items — active constraints, intermediate artifacts, unresolved decisions — the consumer can hold simultaneously before grain must shrink.**

This is the primary driver of stage size. A consumer with high working-memory can carry the full procedure context across many stages without losing their place; one with low working-memory drops items mid-stage and either guesses or restarts. See [granularity-calibration.md](granularity-calibration.md) for the full treatment of how working-memory drives grain selection.

**Elicitation.** Classify as `low`, `medium`, or `high` based on the consumer's familiarity with the domain and the procedure's inherent cognitive load. Low: novice, or LLM agent without persistent memory, or human in a high-distraction context. Medium: mid-level engineer doing something familiar but not routine. High: senior practitioner with deep domain familiarity executing a procedure they own.

**Decomposition impact.** Low working-memory requires smaller stages with explicit, machine-verifiable exit artifacts. Medium working-memory allows stages that bundle related sub-tasks but must still produce named artifacts at every exit. High working-memory supports coarse stages where the consumer self-manages sub-steps — but the exit artifact must still be named and unambiguous.

**Note.** Working-memory capacity and error cost interact: low working-memory at high error cost is the intersection that demands the finest grain and the most explicit verification scaffolding. See [granularity-calibration.md](granularity-calibration.md) for the full interaction table.

---

### 3. Error Cost

**Error cost is what happens if the consumer chooses wrong at a decision point — specifically, how much of the choice can be undone, and at what cost.**

Three levels:

- **Low** — the consumer can retry immediately with no lasting consequence. A wrong branch loses at most a few seconds; the correct option is available again.
- **Medium** — the consumer loses some work or some time. Reverting requires deliberate effort (a rollback, a config change, re-running a sub-procedure) but is fully recoverable.
- **High** — the choice is irrecoverable, regulated, or safety-critical. Examples: a schema migration that drops a column in production, a document submission to a regulatory body, a physical action that cannot be undone. Wrong here is not just expensive — it may be outside the consumer's ability to remediate at all.

**Elicitation.** For each major decision point in the proposed decomposition, ask: "What is the worst outcome if the consumer makes the wrong choice here?" Map that outcome to a level. A procedure that contains even one High-error-cost decision point must be treated as High overall, because the structural safeguards required by a single high-cost point propagate upstream (the scaffolding stages must exist before the consumer reaches that point).

**Decomposition impact.** Low error cost allows the procedure to place decision points confidently and accept that mistakes will be discovered quickly. Medium error cost requires explicit verification stages after irreversible actions — checkpoints that confirm the correct choice was made before proceeding. High error cost requires pre-commitment scaffolding: a gate stage that forces the consumer to review consequences before the irreversible action fires, an escalation path for uncertainty, and a dry-run or preview stage where technically feasible.

---

### 4. Reversibility Appetite

**Reversibility appetite is how willing the consumer is to back up and redo — not whether backing up is possible, but whether the consumer will actually do it.**

This is distinct from error cost. A consumer may face medium error cost but have low reversibility appetite (they are anxious about backing up, prefer never to be put in that position, and will choose wrong rather than acknowledge they need to restart). Conversely, a high-reversibility-appetite consumer actively prefers to defer expensive decisions and revisit them after gathering more information.

**Elicitation.** Ask: "If you reach stage 7 and realize you made the wrong choice at stage 3, what would you do?" The answer characterizes appetite, not just capability. An LLM agent with declared error-handling logic can technically restart; its reversibility appetite depends on whether the procedure designer has given it a recovery path to invoke.

Three levels:

- **Low** — the consumer will not voluntarily restart; they will rationalize a bad choice rather than back up and redo stages.
- **Medium** — the consumer will back up and redo a stage if given an explicit recovery path, but will not improvise one.
- **High** — the consumer actively prefers to defer expensive decisions and revisit them after gathering more information; backing up is a normal part of their workflow.

**Decomposition impact.** Low reversibility appetite means decisions that could technically be deferred should be forced early and validated — the consumer will not restart mid-procedure. It also means the procedure should front-load the information needed to make expensive decisions correctly, even at the cost of slower progress. High reversibility appetite allows deferral: "we will come back to this after we have the environment-readiness report" is acceptable when the consumer is genuinely willing to revisit.

---

### 5. Latency Tolerance

**Latency tolerance is how long the consumer will wait at a stage before the procedure must show progress, hand off, or fail.**

Some consumers can context-switch during a long-running stage and return when it completes. Others are locked into the procedure and experience waiting as a failure: they interpret silence as an error and take action (retry, restart, escalate) that puts the procedure in an inconsistent state.

**Elicitation.** Ask: "If stage N takes five minutes with no visible output, what does the consumer do?" The answers cluster: a senior engineer opens a second terminal and monitors independently; an LLM agent hits a timeout and re-runs the stage; a novice interprets the silence as a hang and ctrl-C's it.

Three levels:

- **Low** — the consumer is locked into the procedure and will interpret silence as an error; they cannot context-switch and will abort if there is no visible progress.
- **Medium** — the consumer tolerates stage-level waits of seconds to a few minutes, but needs explicit closure at each exit — a stage that spans multiple sessions is too long.
- **High** — the consumer can context-switch during a long-running stage and return later; they will open a parallel monitoring window and tolerate async completion.

**Decomposition impact.** Low latency tolerance requires stages that either complete quickly or produce visible progress signals on a defined cadence. A polling stage (LLM agent: "poll convergence every 15 seconds until complete or timeout at 10 minutes") is a structural response to low latency tolerance — it turns a blocking wait into a loop with observable state. High latency tolerance allows async stages where the consumer starts the work and checks back; the exit artifact is produced out-of-band. For novices specifically: low latency tolerance does not mean stages must be fast, it means stages must communicate — a progress marker or expected-duration note per stage is not documentation polish, it is a structural safety mechanism that prevents premature abort.

---

### 6. Recovery Options

**Recovery options are what the consumer can do if a stage fails — not what the procedure ideally offers, but what is actually available to this consumer in this context.**

A senior engineer who owns the infrastructure can manually inspect logs, roll back a deployment, restart a service, or escalate to a colleague. An LLM agent can re-run the wizard from a specified stage, invoke a declared error-handling tool, or escalate to a human via an escalation API. A novice can call a senior or restart from the beginning — those may be their only two options.

**Elicitation.** Enumerate the consumer's actual recovery tools: access level, tools available, escalation paths, whether they can interpret raw error output. Do not assume; ask or observe. Recovery options that exist in theory but require context the consumer lacks are not recovery options for this consumer.

**Decomposition impact.** Narrow recovery options mean each stage must include more defensive scaffolding — precondition checks, explicit failure paths, and escalation instructions. A stage with a narrow-recovery consumer must never fail silently; failure output must be human-readable or machine-parseable without requiring the consumer to interpret raw state. Wide recovery options reduce the defensive scaffolding budget: the consumer can handle partial failures, so the procedure can afford to be less defensive per-stage and more aggressive about progress.

---

## The Audience-Parameter Declaration Block

**Fill this out before writing any stage.** An audience parameter left blank is an implicit assumption; implicit assumptions are the primary cause of procedural defects that only appear when a different audience tries to execute the procedure.

```yaml
audience:
  prerequisites:
    - <one item per line; what they already have/know/have done>
  working_memory_capacity: <low | medium | high> — <one-sentence justification>
  error_cost: <low | medium | high> — <one-sentence justification with example consequence>
  reversibility_appetite: <low | medium | high> — <one-sentence justification>
  latency_tolerance: <low | medium | high> — <one-sentence justification>
  recovery_options:
    - <one item per line; what they can do if a stage fails>
```

**Minimum viable declaration — a worked fill-out for the LLM-agent audience:**

```yaml
audience:
  prerequisites:
    - deployment CLI available in PATH
    - manifest schema loaded in tool catalog
    - staging environment credentials injected as env vars
    - no memory of prior deployments (each run is stateless)
  working_memory_capacity: low — no persistent context across turns; stage state must be fully encoded in declared artifacts
  error_cost: medium — silent state drift between stages can produce incorrect deployment without explicit failure signal
  reversibility_appetite: medium — can execute rollback if target state and rollback command are declared precisely
  latency_tolerance: low — stages must complete within a predictable timeout or surface a progress signal; open-ended blocking causes drift
  recovery_options:
    - re-run from a specified stage if that stage's preconditions are still satisfiable
    - invoke declared escalation API with error payload
    - abort and report failure with stage ID, artifact state, and error output
```

---

## Worked Example: Deploy a Service to Staging — Three Audiences

`granularity-calibration.md` shows how these three audiences produce different grain: senior at 4–5 stages, LLM at 8–10, novice at 15+. This example focuses on the other five parameters — how prerequisites change which stages can be omitted, how error cost and recovery options change defensive scaffolding, how latency tolerance changes sync vs async stage choices, and how reversibility appetite affects how early expensive decisions are forced.

**Goal (identical for all three):** Deploy service version N to the staging environment and confirm it is running.

---

### Audience A: Senior Engineer

```yaml
audience:
  prerequisites:
    - has deployed this service before; knows the manifest format without reference
    - owns the staging environment and can interpret deployment logs directly
    - has rollback credentials and knows the rollback procedure
  working_memory_capacity: high — can hold full deployment context across all stages
  error_cost: medium — staging failures are recoverable without production impact
  reversibility_appetite: high — comfortable rolling back or retrying from any point
  latency_tolerance: high — can context-switch during long stages; opens a second terminal
  recovery_options:
    - manually inspect logs and diagnose from first principles
    - execute rollback directly without guidance
    - restart from any stage without losing context
```

**Stage decisions driven by these parameters:**

- **Prerequisites allow omitting the manifest-review stage.** The senior has the schema memorized. No stage "review manifest schema" is needed; the manifest is produced and committed without a guided review gate.
- **High reversibility appetite allows deferring topology decisions.** The decision "streaming or logical replication?" can be deferred to a later stage that has the environment-readiness report in hand, because the senior is comfortable revisiting if the environment does not support the first choice.
- **No dry-run stage is inserted.** Error cost is medium, not high; the senior can recover from a bad deployment. A dry-run adds latency with no structural safety benefit here.
- **Recovery instructions are omitted per stage.** The senior does not need "if this stage fails, run `kubectl rollout undo`" at every exit; that information exists in their memory. Adding it is clutter, not scaffolding.
- **Latency tolerance allows a single async polling note.** "Monitor deployment progress; return when all replicas converge" is sufficient. No cadence is specified; the senior will use whatever monitoring tool they prefer.

---

### Audience B: LLM Agent

This audience uses the same declaration shown in § Minimum viable declaration — a worked fill-out for the LLM-agent audience above. The stage decisions below show how that parameter set shapes the deploy-to-staging decomposition.

**Stage decisions driven by these parameters:**

- **Prerequisites do not allow omitting the manifest-validation stage — they add structure to it.** The schema is in the tool catalog, which means the validation stage can be a machine-verifiable check (does the manifest parse against schema? all required fields present?). The validation stage produces a `validation-report.json` exit artifact; the deploy stage requires `all_passed: true` from that artifact before firing. A senior would check this mentally; the LLM gets it as a declared precondition.
- **Low latency tolerance forces a dry-run stage before deploy.** Because error cost is medium and latency tolerance is low, a blocking deployment stage that silently hangs is worse than a fast failure from a dry-run. A `--dry-run` flag on the deployment CLI produces a preview artifact in seconds; if it fails, the agent can surface the error and abort without ever touching live state. The stage is inserted between manifest-validation and deploy specifically to keep the deploy stage's blast radius narrow.
- **Low latency tolerance forces a polling stage to replace the async wait.** "Monitor progress" becomes a discrete stage: "Poll convergence every 15 seconds; emit convergence-status.json after each poll; stage completes when `converged: true` or when 10-minute timeout fires." The timeout is a declared exit criterion — not an open-ended wait.
- **Low reversibility appetite means the rollback path is declared inline.** At the evaluate-outcome stage, the agent does not improvise rollback; it invokes the declared rollback command from the procedure, with the prior version tag as a parameter. The prior version tag is captured as an explicit artifact in stage 1 precisely because the agent cannot reconstruct it from first principles mid-procedure.
- **Recovery options are narrow, so every stage failure has a declared next action.** Each stage in the LLM decomposition carries: `on_failure: escalate with stage_id and artifact_state`. No stage exits silently to an unspecified state.

---

### Audience C: Novice on First Day

```yaml
audience:
  prerequisites:
    - has read the deployment overview doc; has credentials provisioned
    - has never executed this procedure; does not know the manifest format
  working_memory_capacity: low — unfamiliar domain; loses track of context if stages are long
  error_cost: medium — novice cannot distinguish a recoverable error from a critical one; fear of "breaking production" is real even in staging
  reversibility_appetite: low — does not trust their own judgment; wants explicit confirmation before proceeding; will not voluntarily back up
  latency_tolerance: medium — needs stage closure; long waits without progress feedback increase anxiety and drive premature abort
  recovery_options:
    - call a senior engineer
    - restart from the beginning of the procedure
```

**Stage decisions driven by these parameters:**

- **Absent prerequisites add stages.** The novice does not know the manifest format. A stage exists — "Review the manifest schema" — that prints the schema alongside an annotated example and asks the novice to confirm each field matches. This stage has no analog in the senior or LLM decompositions; it exists purely because a prerequisite is absent.
- **Low reversibility appetite forces the expensive decision earlier and adds a gate.** For the senior, topology decisions can be deferred until the environment-readiness report exists. For the novice, deferral is dangerous: the novice will not back up willingly. Instead, the procedure front-loads a "confirm what you are about to deploy" review stage — an irreversible-action gate with an explicit escalation path ("if anything looks wrong, stop and message your team lead") — before any state-changing action. The novice does not defer; they confirm.
- **Medium latency tolerance adds progress annotations, not polling.** Unlike the LLM, the novice is not a machine that will drift silently. But they will abort if they see nothing for five minutes. Each long-running stage carries an explicit expected-duration note ("this typically takes 3–7 minutes; the progress bar will increment") and a specific signal to look for before proceeding. This is not documentation polish; it is a structural anxiety-management mechanism.
- **Narrow recovery options saturate every stage with escalation paths.** Every stage failure has one instruction: "Stop and message #deployments-help with the command you ran and the output you see." There is no per-stage diagnosis guidance — the novice cannot act on it. The escalation path is the recovery option; the procedure surfaces it at every failure exit.
- **Low reversibility appetite means the rollback stage is not offered.** The novice does not know how to execute a rollback safely. A stage that says "if something is wrong, roll back" is a god-step for this audience. Instead, the failure path is "stop and escalate" and a senior handles remediation. The novice's procedure terminates at the escalation point; recovery is handed off, not delegated.

**Cross-referencing grain.** The grain consequences of these same parameter sets — why senior decomposes at 4–5 stages and novice at 15+ — are in [granularity-calibration.md](granularity-calibration.md). This example deliberately focuses on the five parameters that do not directly drive grain: how prerequisites add or remove stages, how reversibility appetite forces or defers decisions, how latency tolerance shapes async vs sync stage design, how error cost inserts dry-run and gate stages, and how recovery options set the defensive scaffolding budget per stage.

---

## Cross-references

- [granularity-calibration.md](granularity-calibration.md) — how working-memory capacity and error cost drive stage size; the interaction table and the two grain pathologies (god-step, ladder-of-trivials); the grain-specific consequences of the same three audiences shown here.
- [decomposition-fundamentals.md](decomposition-fundamentals.md) — the five structural properties a decomposition must satisfy; reversibility-ordered staging, which is directly shaped by reversibility appetite; the Postgres example with an explicit audience-parameter declaration.
- [decision-flow-design.md](decision-flow-design.md) — how reversibility appetite and latency tolerance feed into the forced-vs-deferred decision matrix; information-readiness gating; re-asking policy when state changes.
- [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md) — auditing whether stages that assume prerequisites not in the declaration are hiding a dependency; the critic-side enforcement of what this sheet establishes on the producer side.
- [decomposition-smells.md](decomposition-smells.md) — audience-amnesia smell: a decomposition whose audience parameters are implicit, inconsistently applied, or contradict the observed design. The smell is named here; the catalog entry is there.
- [branching-and-mece-review.md](branching-and-mece-review.md) — auditing whether option sets at decision points are still MECE when audience parameters shift (e.g., an option that only applies to a consumer with rollback credentials is not MECE for the novice audience).
- [procedural-invariants-and-correctness.md](procedural-invariants-and-correctness.md) — the soundness checklist includes audience-parameter completeness as a precondition; a procedure declared sound without a filled-out audience declaration has passed a check that did not run.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — audience parameters do not include domain content (what constitutes a passing smoke test, what the correct replication topology is for a given SLA); those are domain-content questions for the relevant domain pack. This sheet models the consumer; the domain pack adjudicates the content the consumer acts on.
