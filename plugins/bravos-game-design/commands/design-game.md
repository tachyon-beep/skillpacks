---
description: Dispatch the game-design-architect agent to turn a design brief into a full design package - experience thesis, causally distinct routes, closed ruleset, responsibility screen, and the cheapest valid test with a precommitted decision rule.
allowed-tools: ["Read", "Glob", "Grep", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[brief_or_path]"
---

# Design Game Command

You are turning a design brief into a game design package: an experience thesis, two or three causally distinct mechanic routes with a recommendation, the recommended route traced at five scales and closed at the edges, a responsibility screen, a balance sketch, and the cheapest test that could falsify the riskiest assumption — with the disposition rule decided before the test runs.

This command does NOT critique an existing external design (`/review-game-design`), plan a playtest for an already-designed game (`/plan-playtest`), or implement anything. It designs.

## Invocation Path

`/design-game` is a Claude Code slash command. The command does not perform the design itself — it resolves the brief, establishes maturity and constraints, dispatches the `game-design-architect` agent via the `Task` tool, then writes the returned package to a dated design file. Readers seeing this command invoked should expect: command resolves the brief → command frames maturity/medium/constraints → agent runs the full-cycle workflow from the `using-game-design` skill → command writes the package to disk.

## Core Principle

**Work backward from the experience, never forward from a clever component — and end in a test, not a victory lap.** A package is complete when its riskiest assumption has a named prototype, a method, and a precommitted mapping from outcomes to Keep / Rewire / Simplify / Shelve / Kill / Retest. "The design is done" is never the deliverable; "the design is testable" is.

## Preconditions

The command accepts one optional argument: an inline brief, or a path to a brief / existing design material. If none is supplied, ask.

### Resolve the brief

```bash
BRIEF="${ARGUMENTS:-}"

if [ -z "${BRIEF}" ]; then
  # Use AskUserQuestion to collect the minimum viable brief:
  #  - desired experience / kind of fun / player promise
  #  - intended players (ages, relationships, access needs) and context
  #  - medium, player count, session length, components/budget
  #  - protected intent vs replaceable implementation
  #  - current maturity (blank premise / concept / ruleset / prototype /
  #    playtest evidence / live game) and any existing material
  #  - the decision this package must enable
  :
fi

# If BRIEF is a path, read it and any sibling design material.
[ -e "${BRIEF}" ] && ls "${BRIEF}"
```

Only ask for gaps that would materially change the design — the agent states reversible assumptions and proceeds; the command should not run an intake interview the skill forbids.

### Locate prior material

```bash
# Existing design docs, critiques, or playtest notes near the subject
ls game-design/ design/ docs/design/ 2>/dev/null
ls *critique* *review* *playtest* 2>/dev/null
```

If a `game-design-critic` report exists for this design, pass it to the agent — critic findings are binding constraints on the package.

## Workflow

### Step 1 — Frame the engagement

Determine and record:

- **Maturity entry point** — blank premise / concept / ruleset / prototype / playtest evidence / live game. This decides whether the agent invents, repairs, or rewires, and whether diagnosis precedes prescription.
- **Medium** — decides which single medium adapter sheet the agent loads.
- **Responsibility posture** — if the players, intensity, embodiment, data use, monetization, or social power could make access/safety/ethics load-bearing, say so explicitly in the dispatch; the screen must shape the design space, not decorate the output.
- **Deliverable scale** — full package vs one bounded artifact. Someone who needs one discriminating experiment must not receive a design document.

### Step 2 — Dispatch the game-design-architect agent

```
Use the Task tool with subagent_type: "game-design-architect"
Provide:
  - the brief (or path) and all located prior material
  - maturity entry point, medium, constraints, protected intent
  - any critic findings or playtest observations (binding constraints)
  - the decision the package must enable
Receive:
  - the design package per the agent's Output Format: thesis, responsibility
    screen, routes with per-decision confidence/risk, recommended design with
    edge closure, five-scale causal trace, balance sketch, cheapest valid
    test with precommitted decision rule, and the four SME protocol sections
```

The agent runs the skill's full-cycle workflow. The command frames and consolidates — it does not design.

### Step 3 — Write the package

Write to `game-design/design-$(date +%Y-%m-%d).md` (or next to the brief file, or a working directory if the location is read-only). If a package for today exists, suffix `-v2` / `-v3` — prior packages are the design history, never overwritten.

### Step 4 — Suggest follow-ups

- **Package complete** → run the cheapest valid test as specified; the precommitted decision rule maps results to the next dispatch.
- **Design needs adversarial review before testing** (high-stakes, serious-game, or monetized designs) → `/review-game-design` on the fresh package.
- **The test itself needs elaboration** (recruiting, measures, evidence handling) → `/plan-playtest`.
- **The premise contained a causal conflict the agent flagged** → resolve the conflict with the human designer before any further design spend.

## Failure-Mode Handling

| Failure | Action |
|---------|--------|
| Brief has no stated experience or audience | Agent derives a candidate thesis and marks it first-to-confirm; command surfaces it as the package's top open question |
| Brief is a solution ("make me a deck-builder") with no problem | Dispatch anyway; the agent works the thesis out of the components and flags theme-as-experience if that is all there is |
| Critic findings exist but conflict with the brief | Pass both; the agent treats findings as binding and surfaces the conflict rather than silently picking a side |
| Responsibility screen returns a no-go (e.g., unsafe live play, minors + adult monetization) | The package reports the no-go and required controls INSTEAD of an elaborated design; do not design around a refused screen |
| User asks the command to certify the design as fun | Refuse the framing: fun is a hypothesis; point at the package's test and its decision rule |
| Output location is read-only | Write to a working directory and say where |

## Scope Boundaries

**Covered:** brief resolution and framing; dispatch of `game-design-architect`; a dated, complete design package on disk; follow-up routing.

**Not covered:** critiquing external designs (`/review-game-design`); standalone playtest planning (`/plan-playtest`); implementation, UI design, narrative prose, or engine work (routed to engineering packs, `lyra-ux-designer`, `lyra-creative-writing`).

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Running an intake interview before dispatch | Collect only what materially changes the design; the agent states reversible assumptions and proceeds |
| Accepting "make it fun" as the thesis | The thesis names who, what experience, through which behavior, with observable signs — push once, then let the agent derive and flag |
| Dropping critic findings from the dispatch | Findings are binding constraints; a package that relitigates them silently is defective |
| Treating the returned package as validated | The package ends in an unrun test; say so in the summary to the user |
| Writing the package over a prior day's version | Date-stamp and suffix; design history is evidence for later diagnosis |
