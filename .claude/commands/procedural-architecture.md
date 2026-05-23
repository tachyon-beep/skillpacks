---
description: Structural reasoning for staged procedures — design, critique, and analyze decompositions of any procedure into composable stages with explicit dependencies, decision points, and flow properties. Pattern pack — applies wherever you build a wizard, troubleshooting tree, training curriculum, configuration flow, decision pipeline, or workflow.
---

# Procedural Architecture Routing

**A procedure is a directed structure of stages with state, flow, decision points, and capacity. This pack owns the SHAPE of the procedure, not its rendering, its content, or its continuous-time dynamics. For code-implementation plans specifically use `/axiom-planning`; for system shape rather than procedure shape use `/system-architect`; for continuous-time dynamics use `/simulation-foundations`.**

Use the `using-procedural-architecture` skill from the `axiom-procedural-architecture` plugin to route to the right specialist sheet. Content authority lives in `plugins/axiom-procedural-architecture/skills/using-procedural-architecture/SKILL.md` — this wrapper is a thin pointer.

## Sheets

**Producer cluster (build the decomposition):**

- **decomposition-fundamentals** - core properties of a good decomposition: MECE-ish coverage, grain consistency, dependency correctness, reversibility-ordered staging, progressive disclosure
- **decision-flow-design** - when to ask which question; forced-choice vs deferred; information-readiness gating; escape hatches
- **granularity-calibration** - picking grain size by working-memory capacity, error cost, and audience competence
- **audience-modeling-for-procedures** - audience as explicit parameter (prerequisites, working memory, error cost, reversibility appetite, latency tolerance, recovery options) not implicit assumption

**Critic cluster (audit the decomposition):**

- **dependency-and-ordering-audit** - preconditions-met-before-use; no-premature-commitment; cheap-decisions-early; hidden coupling
- **branching-and-mece-review** - coverage, exclusivity, "Other" handling, fake-branch detection
- **decomposition-smells** - the authoritative 9-smell catalog (god-step, mystery-step, decision-without-information, audience-amnesia, ladder-of-trivials, premature-commitment, orphan-state, fake-branch, re-entrancy-blindness) with false-positive caveats and remediation
- **procedural-invariants-and-correctness** - minimal soundness checklist before declaring done

**Analyst cluster (flow / capacity / soundness):**

- **queueing-theory-for-procedures** - Little's Law, M/M/1 / M/M/c intuitions, utilisation-knee, bottleneck identification
- **discrete-event-simulation-for-procedures** - when DES earns its cost over closed-form queueing
- **process-algebra-and-workflow-nets** - workflow-net soundness; when the formal model earns its cost
- **flow-vs-state-vs-decision-modeling** - choosing the right abstraction (flowchart vs state machine vs workflow net vs decision table)

**Boundary sheet:**

- **procedural-boundary-and-handoffs** - where this pack stops; cross-pack handoffs; colonisation-smell catalog

## Commands

- `/axiom-procedural-architecture:decompose-procedure` - producer pipeline; goal + audience in, structured decomposition out with stages, dependencies, decision points, and exit artifacts
- `/axiom-procedural-architecture:review-decomposition` - critic pipeline; proposed decomposition in, severity-rated findings list out with evidence per finding and a machine-readable YAML summary
- `/axiom-procedural-architecture:analyze-procedure` - analyst pipeline; routed by question class (queueing / DES / workflow-net soundness)

## Agents

- `decomposition-architect` - producer SME: forward construction of decompositions; refuses to emit without audience declaration; Anti-Overconfidence Protocol
- `decomposition-critic` - critic SME: adversarial audit of proposed decompositions; refuses zero-finding verdicts on non-trivial input; Anti-Rubber-Stamp Protocol

## Cross-references

- Code-implementation-plan instance of this discipline → `/axiom-planning`
- System shape rather than procedure shape → `/system-architect`
- Recovering an existing procedure from a codebase → `/system-archaeologist`
- Continuous-time dynamics, ODEs, control loops → `/simulation-foundations`
- Emergent player-driven game flow → `/simulation-tactics`
- Rendering a finished procedure as prose → `/technical-writer`
- Rendering a finished procedure as a wizard UI → `/ux-designer`
- Rendering a procedure as site information architecture → `/site-designer`
