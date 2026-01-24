# Axiom Planning

**Version:** 1.0.0
**Category:** Software Engineering
**License:** CC BY-SA 4.0

## Overview

Axiom Planning provides TDD-validated implementation planning guidance designed to resist common rationalization patterns that lead to incomplete or vague plans. Built from comprehensive analysis of the original `superpowers:writing-plans` skill with all identified issues corrected.

## Skills

### implementation-planning

**When to use:** Creating comprehensive implementation plans for multi-step features requiring documentation for handoff

**Key features:**
- ✅ Complete code examples (not pseudocode or "add validation")
- ✅ Exact file paths for every file touched
- ✅ Exact commands with expected output
- ✅ Atomic task breakdown (one action per step)
- ✅ Definition of Done checklists
- ✅ Anti-rationalization defenses

**Improvements over original:**
- Fixed CSO (Claude Search Optimization) - description focuses on triggers only
- Removed time estimates (violates best practices)
- Added "When to Use" section with clear decision guidance
- Added "Common Mistakes" table
- Added "Red Flags" rationalization table
- Fixed cross-reference guidance (no `@` syntax that burns context)
- Softer, more professional tone
- Includes TDD test scenarios for validation

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
│   └── plugin.json              # Plugin metadata
├── skills/
│   └── implementation-planning/
│       └── SKILL.md             # Main skill (TDD-validated)
└── README.md                    # This file
```

## Contributing

Found a rationalization pattern not covered? Discovered a new loophole?

1. Document the scenario and agent behavior
2. Test the scenario to verify the issue
3. Update Red Flags or Common Mistakes tables in SKILL.md
4. Submit PR with evidence

## Version History

**1.0.0** (2026-01-25)
- Initial release
- Fixed all issues from superpowers:writing-plans analysis:
  - CSO description (pure triggers)
  - Removed time estimates
  - Added When to Use, Common Mistakes, Red Flags sections
  - Fixed cross-reference guidance
  - Softer professional tone
- TDD-validated (RED-GREEN-REFACTOR): 100% GREEN phase pass rate
- Production ready

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
