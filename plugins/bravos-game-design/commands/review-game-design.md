---
description: Dispatch the game-design-critic agent to produce a severity-rated critique of a game design, ruleset, balance model, or playtest evidence against the pack's failure-mode catalog, with each finding citing the resolving reference sheet.
allowed-tools: ["Read", "Glob", "Grep", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[design_path]"
---

# Review Game Design Command

You are reviewing a game design — a concept, ruleset, design document, balance model, playtest plan, or playtest evidence — against the failure-mode catalog the `bravos-game-design` pack governs: dominance collapse, claimed dynamics with no causal source, flatness patched with disconnected systems, numerical symmetry sold as balance, welfare-adverse retention/monetization, ceremonial accessibility and safety, evidence laundering, unfalsifiable experience claims, and open rule-closure holes. The output is a severity-rated critique in which every finding names the reference sheet that closes it.

This command does NOT design, repair, or replace mechanics. For forward design or redesign, use `/design-game`. The critique's findings become that command's binding constraints.

## Invocation Path

`/review-game-design` is a Claude Code slash command. The command does not perform the critique itself — it resolves the subject, establishes the player promise and maturity, dispatches the `game-design-critic` agent via the `Task` tool, then writes the returned critique to a dated review file. Readers seeing this command invoked should expect: command resolves the subject → command frames promise/audience/maturity → agent walks the twelve review dimensions → command writes the dated critique.

## Core Principle

**Severity by experience blast radius: which part of the player promise collapses, or which participant gets hurt.** A critique is not a list of preferences — taste is labeled taste, deliberate unconventional choices serving the stated experience are defended, and a zero-finding review of a nontrivial design is treated as a defect of the review.

## Preconditions

The command accepts one optional argument: a path to the design material (file or directory). If none is supplied, ask.

### Resolve the subject

```bash
SUBJECT="${ARGUMENTS:-}"

if [ -z "${SUBJECT}" ]; then
  # Use AskUserQuestion to collect:
  # "What am I reviewing? Provide the design material — a design doc, a
  #  ruleset, a balance model, a playtest plan, or playtest evidence
  #  (file or directory)."
  :
fi

if [ ! -e "${SUBJECT}" ]; then
  echo "ERROR: ${SUBJECT} not found."
  exit 1
fi
```

### Locate the promise and context

```bash
# The stated player promise / experience thesis, if it exists anywhere
grep -ril -E "experience|promise|vision|thesis|audience|players" "${SUBJECT}" 2>/dev/null | head

# Prior critiques and playtest evidence — the review history
ls "${SUBJECT}"/*review* "${SUBJECT}"/*critique* "${SUBJECT}"/*playtest* 2>/dev/null
```

Determine the **review inputs**:

- **Player promise** — stated in the material, supplied by the user, or absent. If absent, the agent infers it and the inference is itself a finding.
- **Audience and context** — ages, relationships, access needs, session context. Severity depends on who is affected; if unknown, the agent flags the gap.
- **Maturity** — premise / ruleset / prototype / playtest-evidence / live. Bounds which checks are fair: closure findings apply from prototype maturity up; a napkin concept is not faulted for missing tie rules.

## Workflow

### Step 1 — Frame the review

Record: subject scope (which files), the promise being reviewed against and its provenance (stated vs inferred), audience/context, medium, maturity, and any dimensions the user specifically flagged (e.g., "we're worried about the economy"). User-flagged concerns get reviewed but never limit the walk — the catalog runs in full.

### Step 2 — Dispatch the game-design-critic agent

```
Use the Task tool with subagent_type: "game-design-critic"
Provide:
  - the subject path(s) and review scope
  - the player promise (or "infer and flag"), audience, medium, maturity
  - prior critiques / playtest evidence located above
  - any user-flagged concerns
Receive:
  - a severity-rated findings list across the twelve dimensions, each
    finding citing the resolving sheet
  - cross-cutting patterns, "what the design does well"
  - the four SME protocol sections (Confidence / Risk / Gaps / Caveats)
  - a plain-language result statement
```

The agent walks the twelve dimensions and rates severity by blast radius. The command frames and consolidates — it does not perform the checks.

### Step 3 — Write the critique

Write to `${SUBJECT}/review-$(date +%Y-%m-%d).md` (or next to the source file, or a working directory if read-only). If a review for today exists, suffix `-v2` / `-v3` — prior reviews are the change history, never overwritten.

### Step 4 — Suggest follow-ups

Based on the findings:

- **Critical findings on the core loop or a claimed dynamic** → `/design-game` with the critique attached; the findings are its binding constraints. Do not patch prose around a structural failure.
- **Responsibility findings (welfare, safety, access)** → these outrank everything; surface them to the human designer before any redesign spend.
- **Evidence findings** ("proven" claims the tests cannot support) → `/plan-playtest` to design the test that would actually discriminate.
- **Bounded closure holes only** (ties, exhaustion, termination) → the designer can fix directly with the router's bounded-edit checklist; no full redesign needed.

## Failure-Mode Handling

| Failure | Action |
|---------|--------|
| No stated player promise anywhere | Agent infers the strongest promise the artifact implies and reports the absence as a finding — an unstated promise cannot be designed toward |
| Subject is a live game with real players | Findings on welfare/retention mechanics carry live blast radius — flag that severity assumes players are currently exposed |
| User asks for "just a balance check" | Run the full catalog; report balance findings first but never suppress responsibility or evidence findings discovered en route |
| Evidence of harm-adjacent design (tilt monetization, minors + pressure loops) | Report at Critical regardless of what else the design does well; recommend human-owner escalation before ship decisions |
| The design is deliberately unconventional (no victory, one-shot, discomfort as intent) | Not a finding when it serves the stated experience — the agent defends it; check the defense is present, not silently dropped |
| Review returns zero findings | Reject and re-dispatch with the zero-finding rule quoted — a clean bill on a nontrivial design means the walk failed |
| Subject is read-only | Write the critique to a working directory and say where |

## Scope Boundaries

**Covered:** subject and promise resolution; dispatch of `game-design-critic`; a dated, severity-rated critique on disk with sheet citations; follow-up routing.

**Not covered:** redesign or repair (`/design-game`); playtest execution or planning (`/plan-playtest`); UI/interface audits (`lyra-ux-designer`), prose critique (`lyra-creative-writing`), implementation review (engineering packs).

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Letting a user-flagged concern narrow the walk | Flagged concerns are emphasis, not scope; the twelve dimensions always run |
| Accepting the design's own validation claims as context | Validation claims are review subjects, not review inputs — audit the evidence chain |
| Softening a Critical because the team is about to ship | Severity is the reviewer's; the response (fix / accept / defer) is the designer's — do not merge the two |
| Reviewing a premise as if it were a ruleset | Scale checks to maturity; premature closure findings poison the report's credibility |
| Writing the critique into the design doc | Separate dated file; the design doc belongs to the designer |
| Skipping "what the design does well" | Required — a critique with no named strengths reads as reflexive opposition and gets ignored |
