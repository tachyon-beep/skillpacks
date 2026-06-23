---
description: Dispatch the distributed-design-reviewer agent to produce a severity-rated gap review of a distributed system design or codebase against the 13 channels.
allowed-tools: ["Read", "Glob", "Grep", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[design_dir_or_path]"
---

# Review Distributed Design Command

You are reviewing a distributed system's design (or codebase) against the 13 channels this pack governs: consistency, failure model, replication, coordination, partitioning, ordering, idempotency, transactions, delivery, resilience, backpressure, test strategy, and cost/boundary. The output is a *severity-rated gap review* — per channel, is the guarantee NAMED, is it TRACEABLE to the failure model, is the COST recorded, and is there a TEST/INVARIANT for it.

This command does NOT design, scaffold, or implement fixes. For forward design, use the `using-distributed-systems` skill. The review's "Next Actions" name the resolving sheet and the numbered artifact the designer must update; the designer does the editing.

## Invocation Path

`/review-distributed-design` is a Claude Code slash command. The command does not perform the review itself — it resolves the subject, locates the contract and tier, dispatches the `distributed-design-reviewer` agent via the `Task` tool, then consolidates the agent's findings into a dated review report and writes it to disk. Readers seeing this command invoked should expect: command resolves the subject → command locates the 99- spec / declared tier → command hands the workspace to the agent → agent walks the 13 channels and the consistency gate → command consolidates results.

## Core Principle

**A guarantee that is not named, traced, costed, and tested is not a guarantee — it is a hope.** The review's job is to find the silent, un-named consistency choices this pack exists to prevent. "Mostly consistent", "should be fine", and an un-scoped "we use strong consistency" are FAILURES, not designs.

## Preconditions

The command accepts one optional argument: a path to a `distributed-system/` artifact set, an HLD document, or the root of a brownfield codebase. If none is supplied, ask.

### Resolve the subject

```bash
SUBJECT="${ARGUMENTS:-}"

if [ -z "${SUBJECT}" ]; then
  # Use AskUserQuestion to collect:
  # "What am I reviewing? Provide one of:
  #    - a distributed-system/ artifact set (numbered specs + 99-),
  #    - an HLD / design document (single file or directory), or
  #    - the root of a brownfield codebase to review against the 13 channels."
  :
fi

if [ ! -e "${SUBJECT}" ]; then
  echo "ERROR: ${SUBJECT} not found."
  exit 1
fi
```

### Classify the subject and locate the contract

The review's authority — the declared tier and the per-channel consistency contract — comes from the system's `99-distributed-system-specification.md` and `00-scope-and-goals.md`. Locate them:

```bash
# Artifact-set mode: numbered specs present
ls "${SUBJECT}/99-distributed-system-specification.md" 2>/dev/null
ls "${SUBJECT}/00-scope-and-goals.md" 2>/dev/null
ls "${SUBJECT}/"??-*.md 2>/dev/null

# Or a sibling artifact set next to an HLD / codebase
ls "${SUBJECT}/distributed-system/" 2>/dev/null
ls distributed-system/ 2>/dev/null
```

Determine the **review mode**:

- **artifact-set** — a numbered `distributed-system/` set with a `99-`. Highest confidence: the contract and tier are declared and each channel has (or should have) a spec to check against.
- **hld** — a design document with no numbered set. Medium confidence: the contract is inferred from prose; many gate checks (test/invariant, cost recorded) are evidenced by cross-artifact references the prose obscures.
- **brownfield** — a codebase with no spec. **spec-inferred mode**, lowest confidence: the channels are inferred from code (retry wrappers, quorum config, transaction boundaries, queue consumers, clock reads). The review reports what the code *appears* to guarantee, not what was *intended*.

If no `99-` and no `00-` are present, the tier is **undeclared**. The agent runs in spec-inferred mode and must propose the tier it infers from blast radius (XS–XL), flagging that an undeclared tier is itself a Critical finding — tier is what makes the required-artifact set checkable.

## Workflow

### Step 1 — Frame the scope and tier

Determine and record:

- **Review mode** — artifact-set / hld / brownfield (from Preconditions). Bounds confidence; belongs in Caveats.
- **Declared tier** — XS / S / M / L / XL from `00-` (or inferred). The tier sets which of the 13 artifacts are *required*. Tier is authoritative: if any channel's guidance forces an artifact above the declared tier, that artifact is required (tier promotion, not a waiver) — and a system whose code exhibits, e.g., cross-service sagas while declaring tier S is *mis-tiered*, itself a finding.
- **Required-artifact set** for the tier:
  - XS → 01, 02, 07, 10, 13
  - S → XS + 03, 09, 11
  - M → S + 04, 05, 08, 12
  - L → M + 06, gate runs strict
  - XL → L + BFT in 04, signed/authenticated delivery in 09, full 12
- **What is missing** — which required artifacts are absent, which present-but-thin, which channels the code touches but no spec covers.

If the declared tier and the evident blast radius disagree, flag prominently before any channel-level finding — a mis-declared tier mis-scopes the entire review.

### Step 2 — Dispatch the distributed-design-reviewer agent

```
Use the Task tool with subagent_type: "distributed-design-reviewer"
Provide:
  - the subject path and the review mode (artifact-set / hld / brownfield)
  - the 99- spec and 00- scope, or "spec-inferred mode"
  - the declared (or inferred) tier and its required-artifact set
  - any specific channels or concerns the user flagged in the invocation
Receive:
  - a per-channel gap review keyed to the 13 channels, each rated against the
    four gate questions (named / traceable / costed / tested)
  - a severity-rated findings list with file:line or section evidence
  - a `## Summary (machine-readable)` block (verdict / counts / mode / tier)
```

The agent walks the 13 channels against the available evidence, applies the consistency gate per channel, and returns severity-rated findings. This command resolves the subject, frames the tier, and consolidates output — it does not perform the channel checks itself.

### Step 3 — Consolidate the review report

Write the report to `${SUBJECT}/review-$(date +%Y-%m-%d).md` (or next to the source file if `${SUBJECT}` is a single file, or to a working directory if the subject is read-only):

```markdown
# Distributed Design Review

- **Subject**: <path>
- **Mode**: artifact-set | hld | brownfield (spec-inferred)
- **Contract**: <99- reference, or "spec-inferred">
- **Declared tier**: <XS|S|M|L|XL from 00-, or "inferred: <tier>">
- **Verdict**: PASS | PASS-WITH-FINDINGS | FAIL | TIER-MISMATCH

## Summary (machine-readable)
[Copy the agent's Summary block verbatim — verdict, severity counts, mode, tier, required-vs-present artifacts.]

## Tier & Required Artifacts
[Required set for the declared tier; which are present / thin / missing; any tier-promotion or tier-mismatch finding.]

## Per-Channel Gate Results
[The agent's table: one row per channel (01–13), each scored named / traceable / costed / tested, with the resolving sheet named.]

## Findings — Critical
[Severity-rated findings with file:line or section evidence and the resolving sheet + numbered artifact to update.]

## Findings — High
[...]

## Findings — Medium
[...]

## What the design does well
[Genuine strengths only — no rubber-stamping.]

## Next Actions
- **Per finding**: <resolving sheet> → update <NN-artifact.md> → re-emit 99- → re-run gate
- **Re-gate scope**: which channels' gate rows must re-pass after the fixes land

## Confidence Assessment
[From the agent.]

## Risk Assessment
[From the agent.]

## Information Gaps
[From the agent — what could not be evidenced in this mode.]

## Caveats
- This review covers the artifacts/code provided; other code paths may violate channels not exercised here.
- In spec-inferred mode the contract is reconstructed from code; intent is unknown and the inferred tier may understate blast radius.
- Severity is the reviewer's call; the *response* to a finding (fix / waive / defer) is the signatory's.

## Verdict Statement
<plain-language summary, 1–3 sentences, suitable for designer / architect / on-call>
```

If a review for today already exists, append a `-v2` / `-v3` suffix rather than overwriting — prior reviews are the change history.

### Step 4 — Suggest follow-ups

Based on the verdict:

- **PASS / PASS-WITH-FINDINGS** — name each finding's resolving sheet; the designer reads it, updates the numbered artifact, re-emits `99-`, and re-runs the gate on the affected channel rows.
- **FAIL** — a required artifact is missing or a load-bearing channel is un-named/un-traced/un-tested. Route to the `using-distributed-systems` skill to author the missing spec(s); do not patch prose around the gap.
- **TIER-MISMATCH** — the declared tier under-scopes the system. Re-declare the tier in `00-`, which promotes new artifacts into the required set; the review must then re-run against the larger set.

## Failure-Mode Handling

| Failure | Action |
|---------|--------|
| No `99-` and no `00-` | spec-inferred mode; infer the tier from blast radius and flag undeclared-tier as Critical |
| HLD only, no numbered set | hld mode; gate checks for cost-recorded and test/invariant are weak — record in Information Gaps |
| Brownfield, no spec at all | spec-inferred mode; report what the code appears to guarantee, lowest confidence |
| Declared tier below evident blast radius | TIER-MISMATCH verdict; re-scope before channel-level findings stand |
| A required artifact for the tier is absent | Critical finding; route to `using-distributed-systems` to author it |
| Channel present in code but no spec covers it | Finding: silent un-named choice — the exact failure mode this pack prevents |
| Gate row "should be fine" / "mostly consistent" | FAIL that row; the qualifier is the failure, not a guarantee |
| Subject is read-only | Write the report to a working directory, not next to the source |
| Determinism/replay testing of the cluster raised | Out of scope here — cross-reference `axiom-determinism-and-replay` |
| Broker internals / event-sourcing mechanics raised | Out of scope — cross-reference the event-driven pack; this pack owns the delivery *correctness contract* only |

## Output Location

Reports are written with a date-stamped filename adjacent to the subject (or in a working directory when the subject is read-only). They are not modified after writing; a subsequent review produces a new dated report — the prior reports are the audit trail.

## Downstream Handoffs

- A PASS-WITH-FINDINGS report's "Next Actions" names each resolving sheet; the designer updates the numbered artifact, re-emits `99-`, and re-gates the affected channels.
- **Architecture risk consolidation** — `axiom-solution-architect` consumes this pack's `99-` spec; a FAIL here is an input to the SAD's risk register.
- **Deployment / CI / rollout concerns** surfaced in the review → `axiom-devops-engineering` (out of this pack's scope).
- **Single-service API internals** flagged → `axiom-web-backend`.
- **Deterministic-simulation testing of the cluster** → `axiom-determinism-and-replay`.
- **Broker/event-sourcing/CQRS mechanics** behind a delivery finding → the event-driven pack.

## Scope Boundaries

**Covered:** subject resolution and tier framing; dispatch of the `distributed-design-reviewer` agent; per-channel gate results across the 13 channels; severity-rated, evidence-cited findings; a dated, machine-readable review report; handoff routing.

**Not covered:** designing or authoring the missing specs (use the `using-distributed-systems` skill); implementing code fixes; deployment/ops, single-service API design, determinism/replay mechanics, or broker internals (each cross-referenced to its owning pack).

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Reviewing prose for tone instead of the four gate questions | Score every channel on named / traceable / costed / tested; a fluent paragraph with no test is still a FAIL |
| Accepting an un-scoped "we use strong consistency" | Demand the *scope*: which keys, which operations, under which partition — global un-scoped strong consistency is a failure |
| Taking the declared tier at face value when the code disagrees | Check blast radius against the code; a tier-mismatch mis-scopes the whole review |
| Treating a missing required artifact as Medium | A missing required artifact for the declared tier is Critical — the system is un-gateable on that channel |
| Relabelling severity under ship-date pressure | Severity is the reviewer's; the response (fix/waive/defer) is the signatory's — don't let them merge |
| Rubber-stamping a clean-looking design | A review that finds nothing is suspicious; name specific strengths or look harder |
| Implementing fixes inside the review | The review localises gaps and names the resolving sheet; the designer does the editing |
| Reviewing broker mechanics or replay internals here | Out of scope — cross-reference the event-driven and determinism packs; review only the delivery *correctness contract* |
| Spec-inferred review presented at full confidence | Brownfield/HLD modes reconstruct intent from artifacts; flag the confidence drop in Caveats and Information Gaps |
| Report not dated or not signed | Provenance matters; date-stamp and attribute, and never overwrite a prior day's review |
