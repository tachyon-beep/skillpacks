# Axiom Planning

**Version:** 1.1.0
**Category:** Software Engineering
**License:** CC BY-SA 4.0

## Overview

Axiom Planning provides TDD-validated implementation planning with a quality gate for plan validation. The planning workflow:

```
brainstorming → implementation-planning → plan-review → executing-plans
```

**implementation-planning** creates comprehensive plans with anti-rationalization defenses.

**plan-review** validates plans against codebase reality before execution - catching hallucinations, convention violations, and risks before any code is written.

## Skills

### implementation-planning

**When to use:** Creating comprehensive implementation plans for multi-step features requiring documentation for handoff

**Key features:**
- Complete code examples (not pseudocode or "add validation")
- Exact file paths for every file touched
- Exact commands with expected output
- Atomic task breakdown (one action per step)
- Definition of Done checklists
- Anti-rationalization defenses

### plan-review

**When to use:** After implementation-planning, before executing the plan (high-risk or high-complexity work)

**Invoke:** `/review-plan [plan_file]`

**Architecture:** Spawns 4 specialized reviewer agents in parallel, then synthesizes findings:

| Reviewer | Focus |
|----------|-------|
| **Reality** | Symbol existence, path validity, version compatibility, CLAUDE.md conventions |
| **Architecture** | Blast radius, one-way doors, complexity, patterns, technical debt |
| **Quality** | Test strategy, observability, edge cases, security anti-patterns |
| **Systems** | Second-order effects, feedback loops, failure modes, timing assumptions |

**Philosophy:** Accuracy over speed. Token-intensive by design - reserve for high-risk work. Simplified single-reviewer mode available for lower-risk plans.

**Output:** JSON report with verdict (APPROVED, APPROVED_WITH_WARNINGS, CHANGES_REQUESTED) plus human-readable summary with prioritized issues

## What's Different from superpowers:writing-plans

This plugin was created by systematically fixing issues found in `superpowers:writing-plans`:

| Issue | Fix Applied |
|-------|-------------|
| Description includes workflow details | Changed to pure triggering conditions |
| Time estimates present ("2-5 minutes") | Removed, replaced with "atomic action" concept |
| No "When to Use" section | Added comprehensive decision guidance |
| No "Common Mistakes" section | Added table with 8 common planning mistakes |
| No rationalization defenses | Added Red Flags table with 5 rationalizations |
| Advocates `@` syntax for cross-references | Changed to requirement markers (burns less context) |
| Prescriptive tone ("questionable taste") | Softer professional tone |
| No TDD testing evidence | Includes test scenarios document |

## Installation

From the skillpacks marketplace root:

```bash
/plugin install axiom-planning
```

Or add to your project's `.claude/plugins` directory.

## Usage

**Invoke the skill:**

```
I need to create an implementation plan for [feature description]
```

Claude will recognize the multi-step nature and use the implementation-planning skill.

**Manual invocation:**

```
Use the implementation-planning skill to create a plan for [feature]
```

**Output:** Plan saved to `docs/plans/YYYY-MM-DD-<feature-name>.md`

## Testing Status

**✅ VALIDATED:** This skill has undergone full RED-GREEN-REFACTOR validation per TDD methodology.

**Test Results:**
- RED Phase: 3/5 scenarios showed baseline failures (60% failure rate)
- GREEN Phase: 5/5 scenarios passed with skill (100% compliance)
- REFACTOR Phase: No new loopholes discovered
- **Status:** Production ready

## Design Principles

### Atomic Task Breakdown

Each step is one action with clear Definition of Done:
- Focused on single outcome
- Independently testable
- Committable as logical unit

### Complete Code Examples

Plans include runnable code, not pseudocode:
```python
# ❌ BAD: Vague pseudocode
Add validation logic

# ✅ GOOD: Complete code
def validate_input(data: dict) -> bool:
    required_fields = ["name", "email"]
    return all(field in data for field in required_fields)
```

### Exact Commands

Every test or build command is specific:
```bash
# ❌ BAD: Vague
Run tests

# ✅ GOOD: Exact
pytest tests/auth/test_login.py::test_jwt_token_generation -v
```

### Anti-Rationalization

The skill explicitly counters common excuses:
- "They'll figure out the details" → Details ARE the plan
- "File path is obvious" → State it explicitly anyway
- "Standard validation" → Show the code
- "Quick, combine steps" → Keep atomic

## Real-World Impact

**Problem:** Implementation plans often fail because they're too vague:
- "Add validation" without showing what validation means
- "Update the config" without specifying which file
- "Run tests" without exact command

**Result:** Developer implementing plan gets stuck, wastes time searching codebase, or implements wrong thing.

**This skill prevents that** by enforcing completeness standards and catching rationalization patterns that lead to shortcuts.

## File Manifest

```
axiom-planning/
├── .claude-plugin/
│   └── plugin.json                  # Plugin metadata
├── skills/
│   ├── implementation-planning/
│   │   └── SKILL.md                 # Plan creation skill (TDD-validated)
│   └── plan-review/
│       └── SKILL.md                 # Plan validation skill
├── agents/
│   ├── plan-review-reality.md       # Hallucination detection
│   ├── plan-review-architecture.md  # Patterns, complexity, debt
│   ├── plan-review-quality.md       # Testing, observability, security
│   ├── plan-review-systems.md       # Second-order effects, failure modes
│   └── plan-review-synthesizer.md   # Collates findings into verdict
├── commands/
│   └── review-plan.md               # /review-plan slash command
└── README.md                        # This file
```

## Contributing

Found a rationalization pattern not covered? Discovered a new loophole?

1. Document the scenario and agent behavior
2. Test the scenario to verify the issue
3. Update Red Flags or Common Mistakes tables in SKILL.md
4. Submit PR with evidence

## Version History

**1.1.0** (2026-02-03)
- Added plan-review skill - quality gate between planning and execution
- Multi-reviewer architecture: 4 specialized agents + synthesizer
  - Reality reviewer (hallucination detection)
  - Architecture reviewer (patterns, complexity, debt)
  - Quality reviewer (testing, observability, security)
  - Systems reviewer (second-order effects, failure modes)
  - Synthesizer (collates into prioritized verdict)
- Added /review-plan command with cost warning and simplified mode
- Token-intensive by design - thoroughness over speed
- JSON output with human-readable summary

**1.0.0** (2026-01-25)
- Initial release with implementation-planning skill
- TDD-validated (RED-GREEN-REFACTOR): 100% GREEN phase pass rate

## Related Skills

**Prerequisites:**
- Understanding of TDD methodology helps (but not required)
- Familiarity with git workflows

**Complementary skills:**
- `superpowers:executing-plans` - Execute plans created with this skill
- `superpowers:subagent-driven-development` - Alternative execution path
- `superpowers:test-driven-development` - TDD fundamentals
- `superpowers:brainstorming` - Use before planning when requirements unclear

## License

CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International)

You are free to:
- Share and adapt this work
- Use commercially

Under these terms:
- Attribution required
- ShareAlike (derivatives must use same license)
- No warranties provided

**Faction name note:** "Axiom" is from Altered TCG and not covered by this license. See main skillpacks LICENSE_ADDENDUM.md for details.
