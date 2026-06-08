---
description: Turn a problem or opportunity into a PRD with falsifiable acceptance criteria — problem-first never solution-first, with the success metric drawn from the product workspace's metrics.md scoreboard, named bet-level non-goals as a scope fence, at least one guardrail criterion that must not degrade, and a handoff section that names the top item for /axiom-planning and routes solution shaping to /axiom-solution-architect — reading the docs/product/ workspace and repo context first, then asking only for the gaps. Generates a tailored PRD, not a blank template; emits no dated commitment, no implementation steps, no architecture.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "AskUserQuestion", "Write"]
argument-hint: "[feature_or_problem]"
---

# Write PRD Command

You are writing a **PRD** — the handoff spec that turns a decided product bet into something `/axiom-planning` can plan and the operating loop's `ACCEPT` step can later judge. This command **generates a tailored artifact**, it does not fill in a blank template. The PRD leads with the **problem**, never the solution, and every acceptance criterion is **falsifiable** — binary against a pre-committed threshold, bounded by a date, with an explicit reject branch. A criterion that cannot fail is not a criterion; a PRD whose success cannot be falsified is the build trap wearing a spec.

This is a `/write-prd` slash command. It takes an optional `[feature_or_problem]` argument. If omitted, infer the subject from workspace context or ask.

The PRD is **not** a sixth workspace artifact. It is the handoff spec referenced by ID from `current-state.md` (`awaiting acceptance against PRD-0006`). It pairs with — and does not replace — the Product Decision Record: the **PDR records the decision** to make the bet; the **PRD specifies** it. This command writes the PRD and links it to the PDR; it does not re-make the discovery call.

## Read the workspace before asking

This command runs **inside** the ownership operating loop, so it pulls from the git-versioned product workspace rather than inventing. Read it first — it is cheaper than asking and it keeps the PRD consistent with decisions already on record:

```bash
# The product workspace (default docs/product/, may be configured elsewhere)
ls docs/product/ 2>/dev/null
cat docs/product/metrics.md 2>/dev/null        # the metric scoreboard: north-star + guardrails, BASELINE + current readings
cat docs/product/roadmap.md 2>/dev/null         # which Now/Next/Later bet is this?
ls docs/product/decisions/ 2>/dev/null          # the PDR that decided this bet (Decision: PDR-NNNN)
cat docs/product/current-state.md 2>/dev/null   # what is in flight; existing PRD IDs to number from

# Repo context that grounds the problem in reality
ls README* docs/ 2>/dev/null
grep -rl -i "prd\|problem\|acceptance\|outcome" docs/ 2>/dev/null
git log --oneline -15 2>/dev/null
```

Pull these specifically — they are the things a context-blind PRD gets wrong:

- **The success metric and its guardrail come from `metrics.md`.** The PRD *names* which existing metric moves and by how much; it does not design a metric. If the metric this bet needs is **not yet on the scoreboard**, that is a gate: flag that a metric must be added to `metrics.md` before the bet can be accepted, and do not fabricate a BASELINE. Pull at least one relevant **guardrail** to become a must-not-degrade criterion.
- **The PDR comes from `decisions/`.** The PRD assumes the is-this-worth-solving / for-whom call was made and recorded; record its ID in the header (`Decision: PDR-NNNN`). If no PDR exists, **flag it** — the bet has no provenance — rather than reverse-engineering the discovery decision here.
- **The bet's tier comes from `roadmap.md`** (Now / Next / Later). **The PRD number** continues the existing sequence from `current-state.md` / `decisions/`.

If there is no workspace yet, say so and proceed with what context supplies — but note that without `metrics.md` the success metric is unanchored and the PRD cannot reach `ready-for-planning` until the scoreboard exists. Point the user at `/own-product` to bootstrap the workspace.

## Gather the gaps — batched, only what context could not supply

Then **ask concise, batched questions** (via AskUserQuestion) only for what the workspace and repo did not answer. The problem statement has four parts and the criteria need a measurable target; ask for the missing ones, in priority order:

1. **Who** — the specific user or segment whose problem this is. Not "users"; the segment whose job is blocked.
2. **The problem (their pain)** — the job they are trying to do and where it breaks, stated as their pain, not as the absence of your feature.
3. **The desired outcome** — the changed behavior or state that means the problem is solved, in the user's terms. This is what the success metric measures.
4. **What success looks like, measurably** — the metric, its BASELINE → TARGET, and the date by which the move must show. (Anchor to `metrics.md` if it answered; ask only for the TARGET and date if the metric and baseline are on the scoreboard.)
5. **What is explicitly out** — the bet-level non-goals a reasonable reader might assume are in scope.
6. **Why now** — what makes this worth a slot (the story, not a WSJF score).

Ask only what you need. A PRD with two honest unknowns flagged in **Open questions / assumptions** beats one with two confident fabrications. Do not ask for the *solution* — that is not a PRD input.

## The artifact — the canonical PRD template

Generate the PRD as clean markdown in a fenced block the user can copy and save, in **exactly** this structure (the load-bearing minimum — a spec, not an essay, not a plan):

```markdown
# PRD-0006 — <bet name>            Status: ready-for-planning
Decision: PDR-0005   Bet (roadmap.md): Now   Target metric (metrics.md): <name>

## Problem
Who · the problem (their pain) · desired outcome · why now.   (3–6 sentences.)

## Success metric (the signal the bet paid off)
The ONE metric from metrics.md this bet moves, its BASELINE, its TARGET, and the
date by which the move must show. This is the falsification condition.

## Acceptance criteria (falsifiable)
Numbered, binary, each with a pre-committed threshold and a reject branch.

## Non-goals (this bet)
Explicitly out of scope for THIS bet — the scope fence, not the product's anti-goals.

## Constraints & guardrails
Hard limits the solution must respect (a guardrail from metrics.md must not degrade;
a compliance / latency / data bound). NOT a design — a boundary the design lives inside.

## Open questions / assumptions
What is unresolved and what the bet assumes; each is a risk the plan inherits.

## Handoff
What goes to /axiom-planning (the top item), what goes to /axiom-solution-architect
(solution shape), tracker IDs.
```

**Problem first, solution never.** Lead with who → the problem → desired outcome → why now, and stop before the solution. The tell for a solution-first spec: delete the proposed solution and see whether the problem still reads as a problem. If it collapses into "we haven't built X yet," there was no problem. State the problem so well that several solutions are visibly possible, then hand the choice down to `/axiom-solution-architect`.

## The falsifiability gate — enforce it, do not narrate it

Every acceptance criterion must pass three tests, and a criterion missing any one is not a criterion:

1. **Binary against a pre-committed threshold** — resolves to pass or fail, not to a direction. "Faster" is a direction; "p95 ≤ TARGET" is a threshold.
2. **Bounded by a date** — the observation has a window. "Eventually" is not falsifiable.
3. **Has an explicit reject branch** — names what happens when the threshold is missed. The reject branch is what makes it a *test* and not a *hope*.

At least one criterion must be a **guardrail pulled from `metrics.md`** that must *not* degrade — a bet can hit its success metric *by* harming something else, and a win that hides a harm is not a win. A worked criteria block (generic illustration):

```markdown
## Acceptance criteria (falsifiable)
1. SUCCESS — <north-star metric> rises BASELINE → TARGET within N days of release,
   measured on the metrics.md scoreboard.
   Reject branch: below TARGET at N days → bet rejected; open follow-up PDR.
2. GUARDRAIL — <guardrail from metrics.md> stays within <ceiling/floor> over the
   same window.
   Reject branch: breached → bet rejected even if (1) passes.
3. SCOPE — The change reaches 100% of the segment in the Problem (not a cohort-gated
   experiment) by release.
   Reject branch: gated to a subset → this criterion is unmet.
```

`Status: ready-for-planning` is a **claim about falsifiability**, not about word count. A PRD with a directional criterion ("conversion trends up," "checkout is smoother") is **not ready**, no matter how complete it looks, because the thing it hands to planning can never be accepted. If you cannot make a criterion falsifiable from the inputs, leave the status as `draft` and name the missing measurement in Open questions.

## Constraints on the generated artifact

- **Problem-first, not template-first.** Generate real, tailored content from the gathered inputs; flag genuine unknowns as stated assumptions rather than inventing facts. Do not emit a blank form.
- **One metric, not a dashboard.** Exactly one headline success metric, drawn from the `metrics.md` scoreboard. A PRD that lists six "success metrics" has chosen none.
- **No how.** Conspicuously absent — and kept absent — are implementation steps, file lists, architecture, and task breakdowns. Those are the plan (`/axiom-planning`) and the design (`/axiom-solution-architect`). A PRD that contains them is a PRD-as-plan: it does two siblings' jobs badly and dissolves the seam.
- **No dated commitment.** The PRD's why-now is a story, not a WSJF score; the dated delivery commitment comes from `/axiom-program-management`'s forecast, never from this spec.
- **Generic illustrations only.** Any example metric, criterion, or non-goal is an obvious placeholder (BASELINE / TARGET / `<bet name>`) — never a real client, organization, or domain.
- **Non-goals are bet-level.** They fence *this* spec's scope (a later bet may cross them) — not the product-level anti-goals in `vision.md` (what the product refuses to *become*).

## Closing seam — hand off the top item

After the PRD, make the three edges of the seam explicit so the user knows where the build comes from and where it does not:

1. **Name the top item** — the single highest-value workstream in this PRD — and state that it is handed to **`/axiom-planning`** to become an executable, codebase-validated implementation plan. The PRD owns *what / why* and the bar; planning owns the plan for the item at the top.
2. **Route solution / architecture shaping to `/axiom-solution-architect`** — the solution shape, the component design, the ADR. The PRD names the constraints the design lives inside, never the design.
3. **State the forecast does not come from here** — the committed bet also goes to `/axiom-program-management` for sequencing and the dated forecast. The PRD emits no date.

The PRD's `Handoff` section records only the linkage (which item, which tracker ID, which sibling owns what) — never a copy of the plan.

## After generating

1. State whether a workspace was found and what was pulled from it (the metric + guardrail from `metrics.md`, the `PDR-NNNN` from `decisions/`, the bet tier from `roadmap.md`) versus asked for.
2. Present the PRD as a fenced markdown block the user can save (offer to `Write` it to `docs/product/` or alongside the tracker item if the user wants the file on disk).
3. Declare the **falsifiability status**: `ready-for-planning` only if every criterion is binary + dated + has a reject branch and the success metric is anchored to `metrics.md`; otherwise `draft`, naming exactly what blocks it (an unanchored metric, a directional criterion, a missing PDR).
4. State the named top item and the `/axiom-planning` hand-off, plus the `/axiom-solution-architect` route for solution shape.
5. Note any assumption that, if wrong, changes the bet — especially in the success metric and its TARGET.

## Cross-references

- `using-product-management` — the router; load for the ownership discipline behind the PRD and the operating loop it sits inside.
- `prd-and-acceptance-criteria.md` — the spec behind this command: problem-first discipline, the falsifiability test, the seam table, and bet-level non-goals.
- `product-state-and-continuity.md` — the workspace artifact schemas (`metrics.md`, `roadmap.md`, `decisions/`, `current-state.md`) this command reads, and the PDR template the PRD links to by ID.
- `product-metrics-and-experimentation.md` — owns metric design, instrumentation, and kill-the-bet logic; this command *names* which metric moves, it does not design the discipline.
- `product-discovery-and-opportunity.md` — owns the is-this-worth-solving / for-whom call the PRD assumes was made and recorded as a PDR.
- `vision-strategy-and-roadmap.md` — owns product-level anti-goals (what the product refuses to become), distinct from this PRD's bet-level non-goals.
- `/axiom-planning` — turns the PRD's top item into an executable, codebase-validated implementation plan (closing seam).
- `/axiom-solution-architect` — owns the solution shape, architecture, and ADRs (the *how*); the PRD routes solution-elaboration there.
- `/axiom-program-management` — sequences and delivers the committed bet; the dated commitment comes from its forecast, never from this PRD.
