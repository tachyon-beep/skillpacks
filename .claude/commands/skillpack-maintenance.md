---
description: Systematic maintenance and enhancement of Claude Code skill packs — investigative domain analysis, structure review with fitness scorecard, behavioral (RED-GREEN-REFACTOR) testing, and scoped quality improvements across skills, commands, agents, hooks, and reference sheets
---

# Skillpack Maintenance Routing

**Maintenance is behavioral validation, not syntactic checking — test whether components guide Claude correctly, not whether they parse. The router is a five-stage workflow (Investigate → Scorecard → Test → Discuss → Execute); new skills go through `superpowers:writing-skills`, never inline. For creating a brand-new pack from scratch, design first; this pack maintains existing packs.**

Use the `using-skillpack-maintenance` skill from the `meta-skillpack-maintenance` plugin to route through the maintenance workflow. Content authority lives in `plugins/meta-skillpack-maintenance/skills/using-skillpack-maintenance/SKILL.md` — this wrapper is a thin pointer.

## When to Use

- Enhancing an existing plugin (e.g. "refresh yzmir-deep-rl", "remediate this pack")
- Adding, removing, or modifying components (commands, agents, hooks, reference sheets)
- Identifying coverage gaps against a domain map
- Validating component quality under pressure and edge cases

**Don't use** for: creating a brand-new plugin from scratch (design first), or authoring a brand-new skill (use `superpowers:writing-skills`, which runs its own RED-GREEN-REFACTOR loop per skill).

## Sheets

- **analyzing-pack-domain** — domain investigation: user scope, domain/coverage map, component inventory, gap analysis, with copy-pasteable inventory commands
- **reviewing-pack-structure** — structure review and fitness scorecard (Critical / Major / Minor / Pass), duplicate handling, router/slash-wrapper alignment audit, model-selection guide
- **testing-skill-quality** — behavioral testing methodology: the pressure / adversarial-edge / real-world-complexity gauntlet, subagent test-runner mechanics, per-component test reports
- **implementing-fixes** — execution and versioning: structural fixes, content enhancements, component creation (not skills), version-bump rules, completion summary

## Workflow

Investigation → Scorecard → Testing → Discussion → Execution. Load the briefing sheet at each stage, present the scorecard and test findings, get user approval before execution, then apply only the approved scope and bump the version.
