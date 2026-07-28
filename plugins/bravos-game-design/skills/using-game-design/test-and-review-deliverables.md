---
name: test-and-review-deliverables
description: "Use when producing a test or review artifact - prototype card, playtest plan, evidence diagnosis, balance review, responsibility review, or disposition ledger. Provides the smallest artifact that can test, diagnose, or decide the current design question; preserve uncertainty."
---

# Test and Review Deliverables

Use the smallest artifact that can test, diagnose, or decide the current design
question. Omit empty sections and preserve uncertainty.

## Contents

- Prototype card
- Playtest plan
- Evidence diagnosis
- Balance review
- Balance model
- Accessibility, safety, and ethics review
- Disposition ledger
- Critical design review

## Prototype card

```markdown
## Prototype — [Name]

**Riskiest claim:**
**Support:**
**Contradiction:**
**Inconclusive when:**
**Participants and context:**
**Included fidelity:**
**Explicit substitutions:**
**Procedure:**
**Observed data:**
**Intervention and stop rules:**
**Decision after test:**
**Human participants:** If yes, attach the complete Playtest plan below; this
prototype card alone is not a consent, recruitment, privacy, withdrawal,
compensation, safety, or follow-up protocol.
```

## Playtest plan

```markdown
## Playtest plan

**Question:**
**Claim and causal theory:**
**Participants, recruitment, and compensation:**
**Build/rules/scenario:**
**Consent, withdrawal, access, privacy, and follow-up:**
**Protocol:**
**Permitted interventions:**
**Stopping conditions and safety escalation:**
**Behavioral observations:**
**Debrief prompts:**
**Confounds:**
**Analysis method and decision owner:**
**Support / contradiction / inconclusive:**
**Decision rule mapping outcomes to Keep / Rewire / Simplify / Shelve / Kill / Retest unchanged:**
```

## Evidence diagnosis

```markdown
## Diagnosis

**Verdict:**
**Protected intent:**
**Observed evidence:**
**Participant report:**
**Inference:**
**Alternative explanations:**
**Confidence and limits:**
**Discriminating next test:**
**Recommended disposition:**
```

## Balance review

```markdown
## Balance review

**Balance target:**
**States and player counts examined:**
**Strongest option or exploit:**
**Counterplay and legibility:**
**Feedback/snowball behavior:**
**Multiplayer pathology:**
**Model assumptions:**
**Structural repair before tuning:**
**Test:**
```

## Balance model

```markdown
## Balance model — [Decision or system]

**Decision this model informs:**
**Owner and version:**

### Variables and parameters

| Name | Meaning and unit | Range/distribution | Source or calibration | Confidence |
|---|---|---|---|---|
| | | | | |

### State, transitions, payoffs, and invariants

[Equations, tables, state-transition rules, stock/flow relations, or executable
model definition.]

### Scenarios and policies

| Scenario/player count | Strategy or policy | Starting state | Output metric | Result |
|---|---|---|---|---|
| | | | | |

### Sensitivity and extremes

| Changed parameter/assumption | Range tested | Material outcome change | Design implication |
|---|---|---|---|
| | | | |

**Assumptions about information, skill, and behavior:**
**Calibration evidence:**
**Excluded phenomena:**
**Structural uncertainty:**
**Decision threshold and next test:**
```

## Accessibility, safety, and ethics review

```markdown
## Responsibility review

**Participants and intended intensity:**
**Load-bearing access requirements:**

| Material barrier or hazard | Exposed participants | Likelihood | Severity | Prevention | Calibration / stop / exit | Response and follow-up | Owner |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

**Privacy/economic/conduct concerns:**
**Unresolved expert boundary:**
**Representative test:**

### Current-authority verification when high-stakes

**Jurisdiction(s), ages, setting, responsible organization, data, and payment context:**

| Current primary authority | Checked on | Mandatory / advisory | Requirement or guidance used | Qualified reviewer / owner | Status |
|---|---|---|---|---|---|
| | | | | | |

**Unverified or jurisdiction-dependent questions:**
```

## Disposition ledger

```markdown
## Disposition ledger

| Idea | Local value | Global job | Evidence | Decision | Reopen when |
|---|---|---|---|---|---|
| | | | Keep / Rewire / Simplify / Shelve / Kill / Retest unchanged | |
```

## Critical design review

```markdown
## Verdict

[One or two sentences with the highest-leverage finding.]

## Protected intent

[What the design is trying to preserve.]

## Strongest objection

[Causal failure, affected players, and evidence basis.]

## What already works

[Mechanics with a defensible causal job.]

## Recommended intervention

**Defect class:** [Structural / bounded]

[For a structural issue, give two or three causally distinct candidates with
trade-offs and recommend one. For a confirmed bounded wording, arithmetic,
parameter, timing, or edge defect, give one smallest complete correction.]

## Cheapest valid test

**Claim:**
**Support:**
**Contradiction:**
**Inconclusive:**
**Decision rule:** [Map results to Keep / Rewire / Simplify / Shelve / Kill /
Retest unchanged.]

## Uncertainty

[What cannot yet be concluded.]
```

Do not bury the verdict beneath a generic design lecture. Do not expand every
template when a single mechanic card or prototype card is sufficient.
