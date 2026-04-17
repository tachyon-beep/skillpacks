---
description: Critique an existing solution design package against the 10 canonical failure modes - dispatches the solution-design-reviewer agent and returns a severity-rated findings list with evidence
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[solution_architecture_path]"
---

# Review Solution Design Command

You are running a review of an existing solution design package. Your job is to locate the artifacts, dispatch the `solution-design-reviewer` agent, and summarise the findings for the user.

## Preconditions

Locate the design artifacts:

```bash
# Argument-supplied path, or default
ls solution-architecture/ 2>/dev/null
ls "${ARGUMENT}" 2>/dev/null
```

If neither a consolidated SAD (`99-*.md`) nor the numbered artifacts exist, stop and report "No design package found at <path>."

## Protocol

1. **Identify scope:** full numbered artifact set, just the SAD, or a specific artifact? Note in the review.
2. **Dispatch:** use the Task tool to launch the `solution-design-reviewer` agent with the workspace path.
3. **Report:** present the agent's findings — Critical / High / Medium sections, with evidence references — plus Confidence Assessment, Risk Assessment, Information Gaps, and Caveats.

## Output Location

Write the review to `solution-architecture/review-YYYY-MM-DD.md` (or the same directory as the source if no workspace).

## Scope Boundaries

Covered: review against the 10 canonical failure modes with evidence-cited findings.

Not covered: rewriting the design (that is `/design-solution`), ADR lifecycle governance, security threat modelling.
