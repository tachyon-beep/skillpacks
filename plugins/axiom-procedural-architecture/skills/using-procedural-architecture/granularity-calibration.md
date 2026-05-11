---
name: granularity-calibration
description: Setting grain size by the consumer of the procedure, not the author's intuition — the grain-size question, the two grain pathologies (ladder-of-trivials and god-step), how grain shifts when the audience shifts, and the working-memory and error-cost parameters that drive every grain decision.
---

# Granularity Calibration

**Grain size is set by the consumer of the procedure, not by the author's idea of "the right amount of detail."**

The author's natural instinct is to decompose at the level of their own competence — the level at which they would personally find the procedure obvious to follow. That instinct produces the correct decomposition for exactly one audience: someone who is already as expert as the author. For everyone else — a less-experienced engineer, a first-day novice, an LLM agent that drifts under ambiguity — it produces either a ladder of stages the author finds non-trivial but the novice cannot execute, or a stage so coarse that the novice cannot determine where it ends.

This sheet gives the producer the single test that determines grain, the two pathologies that result when the test is skipped, and three worked decompositions of the same goal for three different audiences.

---

## 1. The Grain-Size Question

**For every stage in a proposed decomposition, ask: can the consumer execute this in one focused attempt, with no ambiguity about what "done" looks like?**

Three answers, three actions:

- **Yes** — grain is right. The stage is sized correctly for this consumer.
- **No** — split. The stage is a god-step: it contains hidden decision points, multiple artifacts, or multiple failure modes. The consumer cannot execute it in one focused attempt because "one focused attempt" does not have a fixed ending.
- **Trivially, in two seconds** — merge. The stage is a trivial-step inside a ladder-of-trivials cluster. The consumer would do this and the next two stages in a single mental beat. Consolidate them into one stage with a richer exit artifact.

The phrase "one focused attempt" is load-bearing. It means: the consumer can pick up this stage with its declared inputs, execute it to its declared exit artifact, stop, and not wonder whether they are done. If they must make sub-decisions to determine whether they are done, the stage has hidden decision points. If the exit artifact requires a paragraph to describe but the stage description is a single verb, the stage is not named at the right grain.

The phrase "no ambiguity about what 'done' looks like" rules out exit artifacts like "configure the service" or "set up the database." Those are goals, not artifacts. An unambiguous exit artifact names a specific object in a specific state: `deployment-manifest.yaml` committed to the release branch, health check returning HTTP 200 at the staging endpoint, seed-data checksum verified against the fixture hash.

---

## 2. The Two Pathologies

**Grain failures come in exactly two directions: too fine and too coarse. Both are diagnosable from the consumer's behavior, not from the author's intuition.**

### Too Fine: The Ladder-of-Trivials

**A chain of stages each doing almost nothing is a ladder-of-trivials; the consumer would execute the entire chain in a single mental beat.**

The signal is not that any individual stage is short. The signal is that the consumer's natural chunking collapses several of them into one unit. When an experienced consumer reads the procedure and thinks "I would just do steps 3, 4, and 5 together — they're all one move," that is the ladder smell. The consumer is doing the consolidation in their head that the author should have done in the decomposition.

The underlying cause is usually author-side decomposition at sub-task level rather than at the stage level the consumer actually operates. The author enumerates every physical action; the consumer's working model groups those actions into coherent units.

Diagnostic signals:
- Five or more consecutive stages, each taking under thirty seconds for the target consumer.
- No distinct exit artifact per stage — each stage's exit is just "the next micro-action is now possible."
- A consumer who can describe the whole chain in one verb phrase ("set up the repo remote") even though the decomposition has it as five stages.

The smell name in the catalog is `ladder-of-trivials`; the catalog with diagnostic signals and fix patterns lives in [decomposition-smells.md](decomposition-smells.md). This section defines the smell briefly; that sheet provides the full entry including the re-entrancy interaction and the rookie mistake of splitting the merge to avoid the name.

### Too Coarse: The God-Step

**A single stage hiding multiple decisions, multiple artifacts, and multiple failure modes is a god-step; the consumer cannot state its exit criterion in one sentence.**

The one-sentence exit criterion is the practical test. If you hand the decomposition to the target consumer and ask "when is this stage done?" and they cannot answer in one sentence without listing sub-steps, the stage is a god-step. "When I've deployed the service" is not an exit criterion. "When the service is deployed" — is that after the manifest is committed? After CI passes? After the load balancer has updated? After smoke tests pass? Each of those questions is a hidden stage.

God-steps are dangerous in proportion to error cost. A god-step in a procedure where errors are cheap and reversible is an inconvenience — the consumer will fumble through and fix it. A god-step in a procedure where errors are expensive and irreversible (production database migrations, infrastructure provisioning, security-configuration changes) is a structural defect: the consumer cannot verify intermediate state, cannot identify where a failure occurred, and cannot safely retry from a known checkpoint.

Diagnostic signals:
- You cannot write the exit artifact in one line. "The service is running" is not an exit artifact. `health-check-result: HTTP 200 at staging endpoint, latency < 500ms` is.
- Multiple things can fail independently within the stage, each requiring a different recovery action.
- The consumer would naturally ask a follow-up question before they know they're done: "Does that mean I also need to...?"

The full catalog entry for god-step is in [decomposition-smells.md](decomposition-smells.md). This sheet's job is to identify the smell at grain-choice time; the smell catalog's job is to provide the adversarial checklist for the critic pass.

---

## 3. Shifting Grain When the Audience Shifts

**The same procedure decomposes at one grain for a senior engineer, another for an LLM agent, and another for a novice — the goal is identical; the audience parameter is the variable.**

This is not a preference for more or less detail. It is an engineering decision driven by working-memory capacity, error cost, and recovery options — the audience parameters defined fully in [audience-modeling-for-procedures.md](audience-modeling-for-procedures.md). This section shows how those parameters translate into concrete grain choices for the same goal.

**Goal: deploy a service to staging.**

---

### Audience A: Senior Engineer

**Audience-parameter summary:**
- Prerequisites: has deployed this service before; owns the staging environment; can read deployment logs without guidance.
- Working-memory capacity: high — can hold the full deployment context across all stages simultaneously.
- Error cost: medium — staging failures are recoverable without production impact.
- Reversibility appetite: high — comfortable rolling back manually.
- Recovery options: can diagnose failures from first principles without scaffolding.

**Grain choice: coarse — 4 to 5 stages, each large but with unambiguous exit criteria.**

```
Stage 1 — Prepare deploy artifact
  Exit artifact: versioned image tag pushed to registry; SHA in deployment-manifest.yaml

Stage 2 — Verify staging environment health
  Exit artifact: environment-readiness-check result (all systems nominal or blocking items listed)

Stage 3 — Roll out
  Exit artifact: deployment acknowledged by orchestrator; new replica set converging

Stage 4 — Smoke test
  Exit artifact: smoke-test-report (pass/fail per test case, total coverage count)

Stage 5 — Rollback or finalize
  Exit artifact: either rollback-record (reason, prior version restored) or finalization-record
                (deployment marked stable, changelog updated)
```

**Why this grain works here:** The senior engineer's working-memory can span the full deployment without losing context. The exit criteria are unambiguous at this granularity — "orchestrator acknowledges the rollout" is a concrete, verifiable state, not a vague verb. Splitting further would produce a ladder-of-trivials: stages like "push the image," "tag the image," and "update the manifest" that this audience executes as a single mental beat.

---

### Audience B: LLM Agent

**Audience-parameter summary:**
- Prerequisites: has access to CLI tools and APIs; no persistent memory of prior deployments.
- Working-memory capacity: low across turns — benefits from explicit structure that constrains drift between stages.
- Error cost: medium — agent errors can cascade silently if intermediate state is not verified explicitly.
- Reversibility appetite: medium — can execute rollback if the target state is declared precisely.
- Recovery options: limited to declared recovery paths; cannot improvise from first principles.

**Grain choice: medium — 8 to 10 stages, each with declared exit artifact in a specific, machine-verifiable format.**

```
Stage 1 — Resolve current version
  Exit artifact: current-version-record.json { service: string, tag: string, commit: string }

Stage 2 — Build and push image
  Exit artifact: registry-push-receipt.json { image: string, sha256: string, registry_url: string }

Stage 3 — Update deployment manifest
  Exit artifact: manifest-diff.patch committed to branch; PR or direct push SHA

Stage 4 — Run pre-deploy validation
  Exit artifact: validation-report.json { checks: [...], all_passed: bool }

Stage 5 — Trigger deployment
  Exit artifact: deployment-id from orchestrator API { id: string, status: "pending" }

Stage 6 — Poll convergence
  Exit artifact: convergence-status.json { replicas_ready: int, replicas_desired: int,
                 converged: bool } (poll until converged or timeout)

Stage 7 — Run smoke tests
  Exit artifact: smoke-test-results.json { tests: [...], pass_count: int, fail_count: int }

Stage 8 — Evaluate outcome
  Exit artifact: deploy-outcome.json { success: bool, action: "finalize" | "rollback",
                 reason: string }

Stage 9 — Execute finalization or rollback
  Exit artifact: either finalization-record.json or rollback-record.json with prior version tag
```

**Why this grain works here:** The LLM cannot carry open context reliably across many turns without drift. Each stage produces a machine-verifiable artifact in a declared format, eliminating the need for the agent to interpret ambiguous prose state. "Converged" has a precise definition (replicas_ready == replicas_desired) rather than being left to the agent's judgment. Stage 6 is a separate stage specifically because polling convergence is a blocking I/O operation with its own failure mode (timeout); bundling it into the trigger stage would create a god-step where one failure mode (trigger rejected) is indistinguishable from another (converge timed out).

**What shifted from Audience A:** More stages, smaller each. Every intermediate artifact is named with a format constraint. The judgment calls that the senior engineer handles implicitly ("is this rollout stable?") are replaced with explicit numeric thresholds.

---

### Audience C: Novice on First Day

**Audience-parameter summary:**
- Prerequisites: has read the deployment overview doc; has credentials provisioned but has never used them.
- Working-memory capacity: low — unfamiliar domain; can hold at most 2–3 active constraints; will lose track of context if stages are too long.
- Error cost: high subjectively — the novice cannot distinguish a recoverable error from a critical one; fear of "breaking production" is real even in staging.
- Reversibility appetite: low — wants explicit verification at each stage before proceeding; does not trust their own judgment on "is this safe to continue?"
- Recovery options: none without explicit scaffolding — cannot diagnose failures from first principles; requires a defined escalation path for anything unexpected.

**Grain choice: fine — 15+ stages, each tiny, with explicit verification at each stage; irreversible actions get additional scaffolding stages before them.**

```
Stage 1 — Confirm environment access
  Exit artifact: successful login output saved to terminal (copy the last line)

Stage 2 — Confirm you are on the right branch
  Exit artifact: git branch output showing feature branch, not main

Stage 3 — Pull latest changes
  Exit artifact: "Already up to date." or merge-commit SHA shown in terminal output

Stage 4 — Confirm no local uncommitted changes
  Exit artifact: git status output showing "nothing to commit"

Stage 5 — Identify the current version tag
  Exit artifact: version string written to a scratch note (you will need it in stage 14)

Stage 6 — Run the build script
  Exit artifact: build script exits 0; image name printed to terminal

Stage 7 — Verify the image was created
  Exit artifact: docker images output showing the new image tag in the first row

Stage 8 — Push the image to the registry
  Exit artifact: push output ends with "digest: sha256:..." line

Stage 9 — Confirm the push succeeded in the registry UI
  Exit artifact: image tag visible in registry web console

--- [IRREVERSIBLE ACTION GATE — read before proceeding] ---
Stage 10 — Review what you are about to deploy
  Exit artifact: you have read the changelog for this version and confirmed it matches
                 what you expect to go to staging; if anything looks wrong, stop here
                 and message your team lead

Stage 11 — Update the deployment manifest
  Exit artifact: deployment-manifest.yaml saved with new image tag; diff reviewed

Stage 12 — Commit and push the manifest
  Exit artifact: commit SHA shown; push accepted with no errors

Stage 13 — Trigger deployment via the deployment tool
  Exit artifact: deployment-tool output shows "Deployment accepted, ID: XXXXX"

Stage 14 — Monitor deployment progress
  Exit artifact: deployment-tool status shows "All replicas running (3/3)"
                 (wait up to 10 minutes; if not reached, stop and escalate)

Stage 15 — Run the smoke-test script
  Exit artifact: smoke-test script exits 0; "All tests passed" in output

Stage 16 — Notify the team that staging is updated
  Exit artifact: message posted to #deployments Slack channel with deployment ID
                 and smoke-test result
```

**Why this grain works here:** The novice cannot execute "verify staging environment health" — that sentence contains no action the novice knows how to take. Every stage here names the exact tool, the exact command output to look for, and the exact exit artifact. Stages 10–11 are scaffolding stages around the irreversible point (manifest commit and deployment trigger): stage 10 is a review gate with an explicit escalation path, because the novice cannot judge "does this look right?" without being told what to look for. Stage 14 specifies the timeout and escalation path, because "wait for it to converge" is a god-step for someone who has never seen a deployment timeout before.

**What shifted from Audience B:** More stages still. The exit artifacts are human-observable (terminal output text, console visual confirmation) rather than machine-verifiable JSON. The irreversible-action gate before stage 11 is explicit scaffolding that neither the senior engineer nor the LLM agent need. The recovery path is always "stop and escalate" rather than "execute rollback."

---

## 4. Working-Memory and Error-Cost as the Underlying Parameters

**Grain is set by two parameters above all others: working-memory capacity and error cost. Audience competence mediates both but does not replace either.**

Working-memory capacity sets the ceiling on stage size. A consumer with high working-memory can track the full context of a large stage — all its sub-decisions, intermediate states, and potential failure modes — without losing their place. A consumer with low working-memory drops context mid-stage and either guesses or has to restart. The grain-size question ("can they execute this in one focused attempt?") is asking: does this stage fit within the consumer's working memory?

Error cost sets the floor on stage size. When errors are expensive and hard to reverse, small stages are not just convenient — they are a safety mechanism. Explicit checkpoints after each stage let the consumer verify intermediate state before committing to the next irreversible action. A god-step in a high-error-cost procedure is a structural defect because it removes those checkpoints: the consumer cannot verify that sub-step 3 of 7 succeeded before sub-step 4 makes an irreversible change.

The relationship:
- High working-memory + low error cost → coarser grain is safe. The consumer can carry the context and can recover if something goes wrong.
- Low working-memory + high error cost → finer grain is required. The consumer needs each stage to be atomic and verifiable; mistakes at this intersection are both likely and expensive.
- High working-memory + high error cost → medium grain with explicit checkpoints. The consumer can carry the context but the stakes demand explicit verification before irreversible actions.
- Low working-memory + low error cost → medium to fine grain. Recovery is easy, but the consumer still cannot carry too much context.

The full parameter set — prerequisites, working-memory capacity, error cost, reversibility appetite, latency tolerance, recovery options — is defined and quantified in [audience-modeling-for-procedures.md](audience-modeling-for-procedures.md). This sheet shows how those parameters translate into the grain decision; that sheet defines the parameters themselves.

---

## Cross-references

- [audience-modeling-for-procedures.md](audience-modeling-for-procedures.md) — the six audience parameters (prerequisites, working-memory capacity, error cost, reversibility appetite, latency tolerance, recovery options); how to declare them explicitly before decomposing.
- [decomposition-smells.md](decomposition-smells.md) — full catalog entries for ladder-of-trivials and god-step, including diagnostic signals, fix patterns, and the edge cases that confuse the two smells.
- [decomposition-fundamentals.md](decomposition-fundamentals.md) — grain consistency as one of the five structural properties of a good decomposition; the effort-ratio signal (max:min > 5:1 indicates a grain problem); the relationship between grain consistency and the other four properties.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — when grain calibration reveals that a stage's content is a domain-content question (what does a good health check cover? what constitutes a passing smoke test?) rather than a structural question; the handoff protocol to the relevant domain pack.
