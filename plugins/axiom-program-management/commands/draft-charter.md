---
description: Draft a lean delivery charter (project) or a structured program charter — outcome-first, leading with the measurable outcome and success metrics rather than a scope list, then scope boundaries (in / out / deferred), stakeholders and the single accountable sponsor or SRO with decision rights, a seed RAID of the highest-exposure risks, the delivery cadence (and program governance forum), and a forecast basis stated as a date range with its load-bearing assumptions — gathering any missing inputs by asking and reading available project context.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "AskUserQuestion", "Write"]
argument-hint: "[project_or_program_name]"
---

# Draft Charter Command

You are drafting a **charter** — the lean founding artifact that says what an effort exists to produce, who owns the outcome, what is in and out, what could sink it, how it will run, and how the date will be forecast. This command **generates a tailored artifact**, it does not fill in a blank template. The charter leads with the **outcome**, never with a scope list, and the forecast is a **range**, never a single date.

This is a `/draft-charter` slash command. It takes an optional name argument. If omitted, infer the effort's name from context or ask.

## Two artifacts — branch first

Before drafting anything, decide whether this is a **PROJECT** or a **PROGRAM**, because they produce different artifacts at different rigor:

- **PROJECT** — one team (or a small number) delivering one coherent thing. Output: a **one-page delivery charter**. A single accountable **sponsor**. Cadence is the delivery rhythm (sprint / kanban). No governance forum section.
- **PROGRAM** — several projects coordinated toward one outcome no single project owns. Output: a **structured program charter / brief** (longer, more sections filled). A single accountable **Senior Responsible Owner (SRO)**. Cadence includes a **governance forum** and the decisions that forum owns.

Detect from context where you can (multiple teams / projects named, a portfolio, a cross-team deadline → program; one squad, one backlog → project). If it is genuinely ambiguous, ask one question to settle it. **Scale the rigor to the answer** — lean-leaning for a project, more predictive structure for a program; do not impose program governance on a single squad, and do not leave a nine-team program with a one-pager.

## Gather inputs before drafting

Read available project context first — it is cheaper than asking, and grounds the charter in reality:

```bash
# Existing planning / charter docs
ls README* docs/ 2>/dev/null
grep -rl -i "charter\|objective\|outcome\|okr\|milestone\|roadmap" docs/ README* 2>/dev/null

# Open work and themes, if an issue tracker or backlog is present
ls .github/ISSUES* docs/backlog* 2>/dev/null

# Recent direction from history
git log --oneline -20 2>/dev/null
```

Then **ask concise, batched questions** (via AskUserQuestion) only for what context could not supply. The questions worth asking, in priority order:

1. **The outcome** — what measurable change in the world does this effort exist to produce? (Push past "ship feature X" to "behavior / value Y changes.")
2. **How success is measured** — the one or two metrics that would prove the outcome landed.
3. **What is explicitly out / deferred** — the boundaries, not just the inclusions.
4. **The single accountable owner** — sponsor (project) or SRO (program). Exactly one name.
5. **The top one to three risks** — the few highest-exposure things that could sink it.
6. **Forecast inputs** — is there historical throughput / a comparable past effort to forecast from, or is this a cold start? Any fixed external deadline?

Ask only what you need. A charter with three honest unknowns flagged as assumptions beats a charter with six confident fabrications.

## The artifact — six sections, outcome-first

Generate the charter as clean markdown in a fenced block the user can copy and save. Lead with the outcome. The six sections, in order:

### 1. Outcome and success metrics — *(`benefits-realization-and-outcomes.md`)*

State the **measurable outcome the effort exists to produce** — the behavior changed or the value realized — **not the outputs** (features shipped, tickets closed). This section comes first and carries the most weight; it is the section most likely to come out wrong, because the natural instinct is to list deliverables. Resist it. Then give the **success metric(s)**: the one or two measures, each with a baseline and a target, that would let someone outside the team confirm the outcome happened. For a program, tie these to the benefits the program is accountable for realizing *past the point of delivery*.

### 2. Scope — in / out / deferred — *(`scope-and-backlog-management.md`)*

Three explicit lists. **In** — what this effort commits to. **Out** — what it explicitly will not do (so a future "small addition" is a visible trade, not a silent drift). **Deferred** — what is real but later, named so it is parked rather than forgotten. Boundaries are the point of this section; a scope list with no "out" is scope drift waiting to happen.

### 3. Stakeholders and decision rights — *(`stakeholder-and-communication.md` + `program-structure-and-governance.md`)*

The key stakeholders (for a program, mapped by power / interest). **Exactly one accountable owner**: the sponsor (project) or SRO (program) — the single name on the hook for the outcome. Then the **decision rights**: who decides what, and which decisions are reserved to the owner versus delegated to the delivery team. For a program, this is where the governance roles live; pull structure from `program-structure-and-governance.md`.

### 4. Top risks — seed RAID — *(`risk-issues-and-raid.md`)*

The **few highest-exposure risks** only — this seeds a RAID log, it does not exhaustively enumerate one (that is `/build-raid`). For each: the risk, its exposure (probability × impact, even if coarse — high / med / low is fine at charter stage), and a one-line response or owner. Pick the risks that would actually change the plan if they fired, not the boilerplate ones.

### 5. Cadence and governance — *(`program-structure-and-governance.md`)*

The **delivery cadence** — the rhythm the work runs on (sprint length / kanban with WIP limits / hybrid) and the heartbeat events that matter. For a **program**, add the **governance forum**: who sits on it, how often it meets, and — crucially — **the decisions it owns** (not just "it meets" but "it decides scope trades over £X, cross-project priority, and go/no-go at each gate"). A governance section that names a meeting but no decisions is theatre.

### 6. Forecast basis and key assumptions — *(`estimation-and-forecasting.md`)*

**How the date will be forecast**, stated as a **range with a confidence level, never a single committed date**. Name the method: throughput-based forecasting or Monte-Carlo over historical throughput where history exists; an explicit cold-start approach (reference-class / comparable effort, with the range widened to reflect the unknown) where it does not. Then the **load-bearing assumptions** the forecast rests on — the things that, if false, move the date. A single hard date with no interval contradicts the forecasting discipline this section cites; do not emit one.

## Constraints on the generated artifact

- **Outcome-first, not template-first.** Do not emit a blank form for the user to fill. Generate real, tailored content from the gathered inputs; flag genuine unknowns as stated assumptions rather than inventing facts.
- **Forecast is a range.** Section 6 never collapses to one date.
- **Generic illustrations only.** If you show an example metric or risk to guide the user, make it an obvious placeholder (e.g. "reduce median onboarding time from BASELINE to TARGET") — never a real client, organization, or domain.
- **Scale to the branch.** A project charter is one page and lean; a program charter is a structured brief with governance and benefits filled in.

## Closing seam — hand off to /axiom-planning

After the charter, name the **chosen top workstream** — the single highest-value item the effort should start with — and state explicitly that it is handed to **`/axiom-planning`** to be turned into an executable, codebase-validated implementation plan. The charter owns the outcome, the scope, and the forecast; `/axiom-planning` owns the plan for the item at the top. Make this seam visible so the user knows where the executable plan comes from.

## After generating

1. Confirm the branch chosen (project vs program) and why.
2. Present the charter as a fenced markdown block the user can save (offer to `Write` it to a path if the user wants the file on disk).
3. State the named top workstream and the `/axiom-planning` hand-off.
4. Note any assumptions that, if wrong, would change the charter — especially in the outcome metric and the forecast range.

## Cross-references

- `using-program-management` — the router; load for the management discipline behind each section.
- `benefits-realization-and-outcomes.md` — outcome vs output, benefits mapping, success metrics (section 1).
- `scope-and-backlog-management.md` — in / out / deferred, lean scope control (section 2).
- `stakeholder-and-communication.md` — power / interest mapping, lean RACI, accountability (section 3).
- `program-structure-and-governance.md` — SRO, decision rights, governance forum and its decisions (sections 3, 5).
- `risk-issues-and-raid.md` — exposure scoring, seed RAID, escalation (section 4); `/build-raid` for the full log.
- `estimation-and-forecasting.md` — throughput / Monte-Carlo forecasting, date ranges, cold-start (section 6).
- `/axiom-planning` — turns the chosen top workstream into the executable implementation plan (closing seam).
