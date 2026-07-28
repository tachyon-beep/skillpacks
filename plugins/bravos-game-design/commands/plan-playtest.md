---
description: Design the cheapest valid test for a game design - identify the riskiest assumption, choose the lowest-fidelity prototype that can falsify it, and emit a playtest plan with precommitted decision rules mapping outcomes to Keep/Rewire/Simplify/Shelve/Kill/Retest.
allowed-tools: ["Read", "Glob", "Grep", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[design_path] [riskiest_assumption]"
---

# Plan Playtest Command

You are designing the cheapest valid test for a game design: naming the riskiest assumption, choosing the lowest-fidelity prototype capable of falsifying it, specifying the method and measures, and — before any result exists — committing the decision rule that maps support / contradiction / inconclusive to Keep, Rewire, Simplify, Shelve, Kill, or Retest unchanged.

This command drives the `using-game-design` skill's test-design step directly (no agent dispatch): read `prototyping-and-playtesting.md` and `test-and-review-deliverables.md` from the `bravos-game-design` plugin before producing anything, plus `accessibility-safety-and-ethics.md` whenever the test involves human participants beyond the designer's own table.

This command does NOT design the game (`/design-game`), critique it (`/review-game-design`), or run the test. It produces the plan that makes the test's evidence interpretable and its outcome decidable in advance.

## Core Principle

**Precommit or launder.** A playtest whose interpretation is decided after the results arrive will launder enthusiasm into proof — politeness, laughter, and "everyone had fun" will be read as validation because nothing was at stake in advance. The plan's job is to make one claim falsifiable, name what would support, contradict, or leave it unresolved, and bind each outcome to a disposition before anyone rolls a die.

## Preconditions

The command accepts up to two arguments: a path to the design material, and (optionally) the assumption to test. If the design path is missing, ask.

### Resolve the design and its claims

```bash
DESIGN="${1:-}"
ASSUMPTION="${2:-}"

if [ -z "${DESIGN}" ] || [ ! -e "${DESIGN}" ]; then
  # Use AskUserQuestion: "Which design is this test for? Provide the design
  # package, ruleset, or concept doc."
  :
fi

# Prior evidence — what has already been tested and what it supported
ls "${DESIGN%/*}"/*playtest* "${DESIGN%/*}"/*evidence* "${DESIGN%/*}"/*review* 2>/dev/null
```

### Identify the riskiest assumption

If `ASSUMPTION` was not supplied, derive candidates from the design's own claims — the experience thesis, each claimed dynamic, and each "this will be fun/tense/strategic" assertion — and rank by (impact if false) × (current uncertainty). Confirm the top candidate with the user only when two candidates genuinely compete; otherwise state the ranking and proceed.

**The riskiest assumption is usually experiential, not structural.** Rule closure, legal-action coverage, and obvious dominance can be checked synthetically without a human playtest — if those are the open questions, say so and check them instead of planning a human test the design does not yet deserve.

## Workflow

### Step 1 — Classify the claim under test

| Claim type | Valid test | Invalid test |
|-----------|-----------|--------------|
| Structural (closure, legal actions, dominance, consequence propagation) | Solo / synthetic play, both-hands-open walkthrough | Recruiting humans to find what a probe finds cheaper |
| Dynamic (does the mechanic produce the behavior — bluffing, negotiation, escalation) | Humans playing under observation, behavior counted | Asking players whether they bluffed |
| Experiential (does the behavior produce the feeling, for THESE players) | The intended players, minimal designer contamination, behavioral + return signals | Friends reporting enjoyment to the designer's face; any LLM judgment of fun |
| Learning/transfer (does play change real-world performance) | Delayed, authentic-context assessment | In-game score as evidence of learning |

A test may only claim what its type can support. An LLM's judgment that play was fun — including this session's — is never evidence of human enjoyment, chemistry, usability, or willingness to return.

### Step 2 — Choose the cheapest capable prototype

Lowest fidelity that can still falsify the claim: both-hands-open solo play → paper/proxy components → scripted facilitation → digital mock. Name what the chosen fidelity CANNOT show and record it in the plan — a prototype card is never sufficient evidence of human experience on its own.

### Step 3 — Design method and measures

- **Participants:** who, how many sessions, and their relationship to the designer (contamination is a confound to record, not a footnote).
- **Protocol:** what is fixed (rules version, teach script, designer's role — ideally silent), what varies.
- **Measures:** behavioral counts and observable events precommitted per the thesis's observable signs — not post-hoc impressions. Enjoyment self-reports may be collected but are labeled contaminated context, never the decision measure.
- **Responsibility:** consent, withdrawal without reason, intensity calibration, and data handling wherever participants beyond the designer's own table are involved (`accessibility-safety-and-ethics.md`); minors require guardian consent and age-appropriate stop authority.

### Step 4 — Precommit the decision rule

For the claim under test, write the mapping BEFORE the test:

- **Support looks like:** <specific observable thresholds> → disposition (usually Keep, or proceed to next-riskiest assumption)
- **Contradiction looks like:** <specific observables> → disposition (Rewire / Simplify / Shelve / Kill — named now, with the specific repair direction each implies)
- **Inconclusive looks like:** <what makes the run uninterpretable — rules played wrong, wrong participants, confound fired> → Retest unchanged with the named correction

No disposition is issued now: the unrun test has earned nothing yet, and the plan must say so.

### Step 5 — Write the plan

Write to `${DESIGN%/*}/playtest-plan-$(date +%Y-%m-%d).md` (suffix `-v2`/`-v3` rather than overwrite):

```markdown
# Playtest Plan

- **Design under test**: <path + version/date>
- **Claim under test**: <the riskiest assumption, stated falsifiably>
- **Claim type**: structural | dynamic | experiential | learning
- **Prototype**: <fidelity chosen and why it is the cheapest capable one>
- **What this test cannot show**: <explicit>

## Participants & Protocol
<who, sessions, designer's role, teach script, fixed vs varied>

## Measures (precommitted)
<behavioral counts / observable events tied to the thesis's observable signs>

## Responsibility
<consent, withdrawal, calibration, data handling — or the recorded judgment that the test is designer-solo and none apply>

## Decision Rule (precommitted)
- Support: <observables> → <disposition>
- Contradiction: <observables> → <disposition + repair direction>
- Inconclusive: <conditions> → Retest unchanged with <correction>

## Confidence Assessment
<how discriminating this test actually is, and the confounds that survive it>

## Risk Assessment
<what a false positive or false negative here costs the project>

## Information Gaps
<what remains unknown even if the test lands cleanly>

## Caveats
<standing limits: enjoyment self-reports contaminated; single-session evidence; participant pool skew>
```

### Step 6 — Suggest follow-ups

- Plan ready → run it; results + the precommitted rule produce the disposition mechanically.
- Results in hand and confusing → `/review-game-design` on the evidence (the critic audits evidence chains).
- Contradiction disposition fires → `/design-game` with the findings as constraints.

## Failure-Mode Handling

| Failure | Action |
|---------|--------|
| User wants "a playtest" with no claim | Derive and rank candidate assumptions from the design's own claims; a test without a claim is a party, not evidence |
| The open questions are all structural | Say so; specify the synthetic probes instead of a human test |
| User asks to fold "and check if it's fun" into a structural test | Split them: one test, one claim; fun is experiential and needs the intended players |
| Design has already "validated" the claim via friends | The prior evidence is contaminated context; plan the discriminating test as if unvalidated, and say why |
| Participants include minors or vulnerable groups | Responsibility section becomes load-bearing: guardian consent, stop authority, data minimization — or recommend not running the test |
| Designer insists on being present and explaining | Record it as a named confound in the plan and in Inconclusive conditions |

## Scope Boundaries

**Covered:** riskiest-assumption identification, claim classification, prototype selection, method/measures, precommitted decision rules, responsibility screening, a dated plan on disk.

**Not covered:** designing or repairing the game (`/design-game`); critiquing the design (`/review-game-design`); running sessions, recruiting, or analyzing results (the plan makes those interpretable; humans execute them).

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Planning a human playtest for a structural question | Closure and dominance are synthetic checks; save humans for what only humans can show |
| Measures invented after watching the session | Measures are precommitted from the thesis's observable signs; late measures = laundering |
| "We'll see how it feels" as the decision rule | Every outcome maps to a disposition before the test; otherwise the test decides nothing |
| Reading laughter, politeness, or comprehension as support | Behavioral counts against precommitted signs; self-reports are context, never the measure |
| One test carrying five claims | One claim per test; stack tests, not claims |
| Skipping responsibility because "it's just a playtest" | Any participants beyond the designer's own table get consent, withdrawal, and calibration by default |
