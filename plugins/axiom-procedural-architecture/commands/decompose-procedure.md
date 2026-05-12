---
description: Producer pipeline — take a goal description and audience parameters, produce a structured decomposition with stages, dependencies, decision points, and exit artifacts.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[goal_description_or_file]"
---

# Decompose Procedure Command

<!-- sections: When to use|When to Use · Required input|Required Input · Output format|Output Format -->

You are running the producer pipeline for `axiom-procedural-architecture`: goal description + audience parameters → structured decomposition (stage list, decision-point inventory, dependency graph, exit artifacts). Not for critiquing an existing procedure (use `/review-decomposition`) or soundness/capacity analysis (use `/analyze-procedure`).

## When to Use

Use when designing a new wizard, troubleshooting tree, training curriculum, configuration flow, decision pipeline, or any staged procedure with gates and handoffs.

Do **not** use to critique an existing procedure → `/review-decomposition`. Do **not** use for capacity, soundness, or formal correctness → `/analyze-procedure`.

## Required Input

Before running, collect both:

1. **Goal description** — what the procedure accomplishes, for whom, in what context. Inline text or file path.
2. **Audience parameter declaration** — the YAML block from `audience-modeling-for-procedures.md`:

```yaml
audience:
  role: ""                   # e.g. "junior developer", "ops engineer"
  prior_knowledge: []        # e.g. ["basic CLI", "reads JSON"]
  failure_tolerance: ""      # "low" | "medium" | "high"
  context: ""                # e.g. "incident response", "onboarding"
  cognitive_load_budget: ""  # "tight" | "moderate" | "generous"
```

If either is missing, use `AskUserQuestion` before proceeding.

## Pipeline

Run steps in order. Read each sheet before using its guidance — do not rely on memory.

**Step 1 — Router orientation.** Read `using-procedural-architecture` SKILL.md: confirm available sheets, the Consistency Gate checklist, and which checks this pipeline must satisfy.

**Step 2 — Audience modeling.** Read `audience-modeling-for-procedures.md`. Fill the audience block from user input. If any field is unclear, ask — audience mismatches drive grain and branching errors.

**Step 3 — Draft stage list.** Read `decomposition-fundamentals.md`. For each stage declare: name, single-sentence purpose, inputs, exit artifact, and preliminary dependency links.

**Step 4 — Calibrate granularity.** Read `granularity-calibration.md`. Merge stages too fine for the audience; split stages that exceed the cognitive-load budget; flag any stage with an ambiguous exit artifact.

**Step 5 — Place decision points.** Read `decision-flow-design.md`. For each branch point record: triggering condition, exhaustive option set (MECE), declared preconditions, and destination stage per branch.

**Step 6 — Consistency Gate.** Run the gate from the router. Verify: every stage has inputs + exit artifact + dependency declaration; every decision point has an exhaustive option set + preconditions; no circular dependencies; granularity matches audience; all branches terminate. Fix failures and re-run; do not emit with known gate failures.

**Step 7 — Emit.** Write the decomposition document (see Output Format).

## Output Format

A self-contained document with four sections:

**Audience Parameter Declaration** — the filled YAML block, no prose.

**Stage List** — one entry per stage:
```
Stage N: <Name>
  Purpose:       <single sentence>
  Inputs:        <list>
  Exit artifact: <concrete deliverable or verified condition>
  Depends on:    <stage numbers or "none">
```

**Decision-Point Inventory** — one entry per decision point:
```
DP-N (after Stage X): <triggering question>
  Preconditions: <what must hold>
  Options: A. <label> → Stage Y  |  B. <label> → Stage Z
```

**Dependency Graph** — ASCII or textual adjacency list:
```
Stage 1 → Stage 2 → DP-1 → [A: Stage 3, B: Stage 4]
Stage 3 → Stage 5 → END
Stage 4 → Stage 5
```

## Hand-off

After emitting, recommend: run `/review-decomposition` on the output as an adversarial check — it stress-tests MECE branching, dependency ordering, grain consistency, and boundary conditions the producer pipeline does not catch.
