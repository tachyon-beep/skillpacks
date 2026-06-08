---
description: Construct or refresh a RAID log (Risks, Assumptions, Issues, Dependencies) from the current state of a project or program — each risk scored for exposure (probability × impact) with a named owner, an explicit response, and a trigger; each dependency expressed as a dated, owned contract row that auto-flags as a risk when promised-by clears past needed-by; the load-bearing assumptions surfaced for validation; each issue given a severity, action owner, and resolution path; emitted as clean markdown tables ordered by exposure and severity, with a review cadence and escalation thresholds that keep the log a living tool rather than a graveyard.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[project_path]"
---

# Build RAID Command

You are constructing or refreshing a **RAID log** — Risks, Assumptions, Issues, Dependencies — from the current state of a project or program. The output is a *living management artifact*: clean markdown tables the user can save and review on cadence, not a kickoff relic that gets written once and forgotten. This command is **outcome-first, not template-first**: you populate the tables from gathered state (or from what you ask for), you do not emit a blank skeleton for the user to fill in.

This command produces an artifact. It does not edit code, design a schedule, or run a delivery audit. For a deeper delivery-health audit against all 13 pack sheets, dispatch the `delivery-health-reviewer` agent (see *Optional* below). For the formal RSKM (Risk Management) and DAR (Decision Analysis) process areas required in regulated contexts, route to `/axiom-sdlc-engineering` — this command runs the *operational* RAID log, not the formal process definition.

## Invocation Path

`/build-raid` is a Claude Code slash command. It takes an optional `project_path` argument; if omitted, it targets the current working directory. Typical uses: standing up a RAID log for a project or program that lacks one, or **refreshing an existing one** — re-scoring risks as conditions change, advancing dependency statuses, closing resolved issues, and re-validating assumptions.

```bash
INPUT="${ARGUMENTS}"
TARGET="${INPUT:-.}"
echo "Building RAID from state under: ${TARGET}"
```

## Two modes: construct vs refresh

Before anything else, look for an existing RAID log so you do not silently overwrite a living tool.

```bash
grep -rliE "raid[ _-]?log|risks.*assumptions.*issues.*dependencies" "${TARGET}" \
  --include="*.md" --include="*.markdown" 2>/dev/null | head
```

- **Refresh mode** (an existing RAID log is found): read it, then *carry it forward*. Re-score every open risk against current conditions (a risk whose trigger has fired becomes an issue; a risk whose conditions have passed gets closed with a note). Advance each dependency's status. Close issues that are resolved and record the resolution. Re-test each assumption — a falsified assumption becomes a risk or an issue. Preserve owners and history; do not reset the log to kickoff state. This *is* the discipline that keeps a RAID log alive.
- **Construct mode** (none found): build fresh from gathered state.

If a RAID log is found, confirm the path with the user via AskUserQuestion before refreshing (Refresh-in-place / Build-fresh-alongside / Cancel) so an existing artifact is never clobbered without consent.

## Gather current state

Read the project's reality before inventing entries. A RAID log assembled from real signals beats a generic one every time.

```bash
# Planning docs, charters, READMEs, roadmaps
grep -rliE "charter|roadmap|milestone|risk|dependenc|assumption|blocker" "${TARGET}" \
  --include="*.md" 2>/dev/null | head -40

# Issue-tracker exports (CSV/JSON) and status notes
find "${TARGET}" -type f \( -iname "*issues*.csv" -o -iname "*issues*.json" \
  -o -iname "*status*.md" -o -iname "*standup*.md" \) 2>/dev/null | head

# Recent activity — what is moving, what is stuck
git -C "${TARGET}" log --oneline -30 2>/dev/null
git -C "${TARGET}" log --since="3 weeks ago" --pretty="%s" 2>/dev/null | head -40
```

Read what these surface: open risks and blockers already named, dependencies between teams or components, assumptions baked into the plan, and issues currently in flight. Commit messages and status notes often name blockers and waiting-on relationships that were never logged.

**Then ask for what is missing.** A RAID log needs human knowledge the repo does not hold: who owns each risk, what the real probability and impact are, which dependencies have a promised date, and which assumptions are load-bearing. Use AskUserQuestion to fill the gaps — do not fabricate owners, dates, or scores. Scale the questioning to the work: a single squad needs a handful of risks and two or three dependencies; a multi-team program needs cross-project dependencies and an escalation chain. **Match the rigor to the scale and stakes — lean for a project, more predictive structure for a program — without making either dogmatic.**

## Assemble the RAID log

Emit four tables plus a cadence/escalation block. Order **risks by exposure descending** and **issues by severity descending** so the things that matter sit at the top.

### Risks

Each risk row carries: probability (1–5), impact (1–5), **exposure = probability × impact**, a **named owner**, a **response** drawn from {avoid, reduce, transfer, accept}, and a **trigger / leading indicator** — the observable signal that the risk is materializing.

- **Acceptance must be explicit and must carry a trigger.** "Accept" is a decision, not a shrug: it names what would have to be true to revisit the decision. A risk marked `accept` with no trigger is not a managed risk.
- A risk whose trigger has already fired is no longer a risk — promote it to the Issues table.

| ID | Risk | Prob (1-5) | Impact (1-5) | Exposure | Owner | Response | Trigger / leading indicator |
|----|------|:---------:|:------------:|:--------:|-------|----------|------------------------------|
| R1 | A third-party API may miss its latency SLA under our peak load | 4 | 4 | 16 | Integration lead | Reduce — add a cache + fallback path | p95 latency in staging exceeds 400ms under load test |
| R2 | The shared component team may not finish the auth library by integration week | 3 | 5 | 15 | Program coordinator | Reduce — agree a thin-slice interface early; stub behind a flag | Their burndown shows < 60% done two weeks out |
| R3 | A known browser-compat gap affects < 2% of users on a legacy client | 2 | 2 | 4 | Frontend owner | Accept — trigger: that segment exceeds 5% of traffic, then reduce | Analytics shows legacy-client share climbing above 5% |

### Assumptions

List the **load-bearing** assumptions — the ones that, if false, change the plan. Each names a validation path and an owner. A validated assumption can be retired; a falsified one becomes a risk or an issue.

| ID | Assumption (load-bearing) | Why it matters | How to validate | Owner | Status |
|----|---------------------------|----------------|-----------------|-------|--------|
| A1 | The upstream data feed stays in its current schema for the release window | All ingestion downstream depends on it | Get written confirmation from the feed owner | Data lead | Unvalidated |
| A2 | Two engineers stay allocated through delivery, not pulled to support | The forecast assumes that throughput | Confirm allocation with the resourcing manager | Delivery lead | Validated |

### Issues

Issues are risks that have materialized, or problems already present. Each carries a **severity**, an **action owner**, and a **resolution path**. Order by severity descending.

| ID | Issue | Severity | Action owner | Resolution path | Status |
|----|-------|----------|--------------|-----------------|--------|
| I1 | The staging environment is down, blocking integration testing | High | Platform owner | Restore staging; add a health alert so it surfaces sooner next time | Open |
| I2 | A downstream consumer reported a breaking change in our payload | Med | API owner | Ship a versioned endpoint; coordinate the consumer's cutover | In progress |

### Dependencies

Each dependency is a **dated, owned contract** — not a vague "we need X from Y." Seven columns: provider, consumer, what, needed-by, promised-by, interface/acceptance, status.

**Auto-flag the late ones.** Compare `promised-by` against `needed-by` for every row: if `promised-by` is *later than* `needed-by`, the dependency will arrive late by its own admission — flag it (mark the status and raise a matching risk in the Risks table). A dependency with no promised date is also a flag: an undated commitment is not a commitment.

| ID | Provider | Consumer | What | Needed-by | Promised-by | Interface / acceptance | Status |
|----|----------|----------|------|-----------|-------------|------------------------|--------|
| D1 | Platform team | Our team | Auth service endpoint | Week 6 | Week 5 | OpenAPI contract + a passing smoke test | On track |
| D2 | Vendor | Our team | Signed data-sharing terms | Week 4 | Week 7 | Countersigned agreement | **LATE — promised after needed; raised as R-new** |
| D3 | Our team | Reporting team | Event schema v2 | Week 8 | (none given) | JSON schema + sample payloads | **Undated — chase a promised date** |

When a dependency is flagged late or undated, add a corresponding risk to the Risks table (owner = the seam owner, trigger = the slip date) so it is scored and escalated like any other risk rather than living only in the dependency table.

### Review cadence and escalation thresholds

A RAID log is a management tool *only if it is reviewed on cadence and risks escalate before they become issues.* Emit both:

- **Review cadence** — concrete and scaled to the work. For a single project: review the log in the weekly team sync; re-score the top risks; refresh dependency statuses. For a program: a fortnightly RAID review at the governance board, with project-level logs rolling up into a program log. State the cadence explicitly in the artifact.
- **Escalation thresholds** — a concrete exposure level mapped to who hears about it. For example: *exposure ≥ 12 escalates to the sponsor at the next governance review; exposure ≥ 20, or any risk whose trigger has fired, escalates immediately, out of cadence.* Tune the numbers to the project's risk appetite, but make the threshold concrete and name the recipient.

## The graveyard anti-pattern (state this in the artifact)

Close the output with a short, explicit note:

> **This RAID log is only worth the page it sits on if it is reviewed.** The failure mode is the *graveyard*: a RAID log written once at kickoff, never re-scored, never escalated, never closed — a compliance artifact, not a management tool. The cure is the review cadence above. A risk that is never re-scored is not being managed; a dependency whose status never changes is not being tracked; an assumption never re-tested is just a hope. Review on the cadence, or do not keep the log.

## Output format and saving

Default to emitting the full RAID log as **clean markdown inline** in your response — the four tables, the cadence/escalation block, and the graveyard note — ordered with risks by exposure and issues by severity. Then **offer to save it** (e.g., `RAID.md` under the project) via AskUserQuestion; only write the file with the `Write` tool if the user accepts, or when refreshing an existing log in place. Do not write a file unprompted in construct mode.

## Hard constraints

1. **Generic illustrations only.** Every example row above is a placeholder; when you populate the real log, use the project's actual state and asked-for inputs — never invent client, employer, or vendor specifics, and never carry a real organization's data into an example.
2. **Never fabricate owners, dates, or scores.** If the gathered state does not yield them, ask. An owner-less risk and a dateless dependency are the two failure modes a RAID log exists to prevent — do not reintroduce them by guessing.
3. **Scale rigor to the work.** Lean for a project, more predictive structure for a program. Do not impose program-scale governance on a single squad, and do not under-equip a multi-team program with a single flat list.

## Optional: dispatch the delivery-health reviewer

If the user wants a deeper audit than a RAID log — flow metrics, WIP discipline, forecast defensibility, reporting honesty, dependency exposure, and (at program scale) outcome accountability and governance cadence — offer to dispatch the `delivery-health-reviewer` agent:

```
Task(subagent_type="delivery-health-reviewer",
     description="Delivery-health audit using the RAID log as one input",
     prompt="Audit the delivery health of the project at ${TARGET} against the axiom-program-management sheets. Use the RAID log just produced as one input. Report findings with severity and the sheet that closes each gap. Follow the SME Agent Protocol (Confidence, Risk, Information Gaps, Caveats).\n\n${RAID_MARKDOWN}")
```

The agent supplements the RAID log; it does not replace it. Present the RAID log first, then the audit.

## Cross-references

- `risk-issues-and-raid.md` — the authoritative sheet: operational RAID log, exposure scoring (probability × impact), the risk→issue conversion, review cadence, and the escalation path. Load this for the scoring and escalation discipline behind this command.
- `dependencies-and-coordination.md` — dated, owned dependency commitments with named provider and consumer, blocked-work management, and integration-point discipline. Load this for the dependency contract model (and `cross-project-dependencies-and-integration.md` when the dependency graph spans multiple projects in a program).
- `status-reporting-and-metrics.md` — how RAID feeds an honest status report without the watermelon effect.
- `/axiom-sdlc-engineering` — the **formal** RSKM (Risk Management) and DAR (Decision Analysis) process areas for regulated contexts. That pack defines the process; this command runs the operational RAID log inside it.
