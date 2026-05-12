---
description: Critic pipeline — take a proposed decomposition in any reasonable format and produce a severity-rated findings list with evidence per finding.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit"]
argument-hint: "[decomposition_text_or_file]"
---

# Review Decomposition Command

<!-- sections: When to Use · Required Input · Pipeline · Output Format · Hand-off -->

You are running the critic pipeline for `axiom-procedural-architecture`: proposed decomposition + audience parameters → severity-rated findings list with evidence, plus a machine-readable summary. Not for producing a decomposition (use `/decompose-procedure`); not for capacity or soundness analysis (use `/analyze-procedure`).

## When to Use

Use when you have a proposed decomposition — a stage list, pasted spec, prose description, diagram caption, or any structured description of a staged procedure — and want an adversarial audit before adopting or publishing it.

Do **not** use to produce a decomposition from scratch → `/decompose-procedure`. Do **not** use for capacity, soundness, or formal correctness reasoning → `/analyze-procedure`.

## Required Input

Before running, collect both:

1. **Decomposition under review** — the proposed decomposition in any reasonable format (inline text, markdown, file path, diagram caption).
2. **Audience parameter declaration** — the YAML block describing the intended audience. If absent, the **first finding** is automatically:

```
Finding: audience-amnesia
Severity: high
Evidence: No audience parameters supplied with the decomposition.
Remediation: Declare audience block (prerequisites, working_memory_capacity, error_cost,
             reversibility_appetite, latency_tolerance, recovery_options) before re-evaluating.
```

## Pipeline

Run steps in order. Read each sheet before applying its guidance — do not rely on memory.

**Step 1 — Router orientation.** Read `using-procedural-architecture` SKILL.md. Confirm the available sheets, locate the critic cluster (sheets 5–8), and note any Consistency Gate checks that apply to this pipeline.

**Step 2 — Dependency and ordering audit.** Read `dependency-and-ordering-audit.md`. Run all four ordering checks against the decomposition:
  - Are dependencies declared for every stage?
  - Are there any forward references or implied orderings not made explicit?
  - Do any circular dependencies exist?
  - Are handoff artifacts between stages well-defined?

**Step 3 — MECE and branching review.** Read `branching-and-mece-review.md`. Run all four MECE/branching checks:
  - Is each decision point's option set mutually exclusive?
  - Is the option set collectively exhaustive (no missing branches)?
  - Do branch labels carry enough information to decide without re-reading earlier stages?
  - Do all branches terminate at a defined stage or END?

**Step 4 — Decomposition smells walkthrough.** Read `decomposition-smells.md`. Walk through all nine smells one by one. For each smell that fires, record: smell name, affected stage or decision point, and the evidence string from the decomposition.

**Step 5 — Invariants and correctness checklist.** Read `procedural-invariants-and-correctness.md`. Run the minimal checklist. Flag any invariant violations.

**Step 6 — Aggregate and rate.** Collect all findings from steps 1–5. Assign severity:
  - **high** — structural break: missing exit artifacts, circular deps, non-exhaustive branches, violated invariants.
  - **medium** — clarity or maintainability risk: ambiguous stage names, implicit dependencies, smell pattern match.
  - **low** — style or improvement opportunity: redundant labeling, grain inconsistency, minor naming issues.

**Step 7 — Emit.** Write the findings report (see Output Format) and the machine-readable summary block.

## Output Format

A self-contained critique document with two parts:

**Part 1 — Findings List** (one heading per finding):

```
### Finding N: <slug-name>
Severity: high | medium | low
Location: Stage X / DP-Y / global
Evidence: <verbatim excerpt or precise description from the decomposition>
Remediation: <specific corrective action>
```

Findings ordered by severity (high → medium → low).

**Part 2 — Machine-readable Summary** (YAML or JSON, immediately after the findings list):

```yaml
review_summary:
  total_findings: N
  by_severity:
    high: N
    medium: N
    low: N
  top_findings:
    - slug: <name>
      severity: <level>
      location: <stage or dp>
    - slug: <name>
      severity: <level>
      location: <stage or dp>
    - slug: <name>
      severity: <level>
      location: <stage or dp>
  recommended_remediations:
    - <remediation 1>
    - <remediation 2>
    - <remediation 3>
```

## Hand-off

After emitting the findings report, recommend: run `/decompose-procedure` with the same goal and audience parameters to produce a revised decomposition that addresses the findings — the producer pipeline applies the same sheets from the constructive side and will naturally avoid the patterns flagged here.
