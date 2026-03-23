# Implementing Fixes

**Purpose:** Execute approved changes across all component types with version management and git commit.

## Prerequisites

You should have completed and gotten approval for:
- Domain analysis and component inventory
- Structure review and scorecard
- Behavioral testing results
- User discussion and approval of scope

**Do NOT proceed without user approval of the scope of work.**

---

## Critical Checkpoint: New Skills

**STOP:** Did you identify gaps requiring new skills?

**If YES → Exit this workflow NOW:**

1. For EACH skill gap:
   - Use `superpowers:writing-skills`
   - RED: Test scenario WITHOUT the skill
   - GREEN: Write skill addressing gaps
   - REFACTOR: Close loopholes
   - Commit that ONE skill
2. Repeat for ALL skill gaps
3. Return here AFTER all new skills are tested and committed

**New commands and agents CAN be created here.** Only new SKILLS require the separate workflow.

**Proceeding past this checkpoint assumes:**
- ✓ Zero new skills needed, OR
- ✓ All new skills already created via superpowers:writing-skills

---

## Execution by Component Type

### 1. Skills

**Structural fixes:**
- Remove duplicate skills (preserve unique value)
- Update router to reflect current specialists
- Fix broken cross-references

**Content enhancements:**
- Address gauntlet failures
- Add missing anti-patterns
- Strengthen pressure-resistance language
- Improve examples

**Reference sheets:**
- Update to reflect current state
- Add missing guidance
- Fix inaccuracies

### 2. Commands

**Create new commands:**
```yaml
---
description: [What this command does]
allowed-tools: [Tool, List]
argument-hint: "[args]"
---

# Command Name

[Command content]
```

**Fix existing commands:**
- Update descriptions for clarity
- Adjust tool restrictions
- Fix argument-hint accuracy
- Remove overlap with skills

**Remove obsolete commands:**
- Delete file
- Update any references

### 3. Agents

**Create new agents:**
```yaml
---
description: [What this agent specializes in]
model: sonnet
tools: [Read, Grep, Glob, Bash, Write]
---

# Agent Name

[Agent system prompt]

## When to Activate

<example>
User: "[matching task]"
Action: Activate - [reason]
</example>

<example>
User: "[non-matching task]"
Action: Do NOT activate - [reason, handoff]
</example>

## Scope Boundaries

**I do:** [list]
**I do NOT:** [list, with handoffs]
```

**Fix existing agents:**
- Clarify scope boundaries
- Adjust model selection
- Update activation examples
- Fix tool permissions

### 4. Hooks

**Create new hooks:**
```json
{
  "hooks": {
    "EventType": [{
      "matcher": "Pattern",
      "hooks": [{
        "type": "command",
        "command": "${CLAUDE_PLUGIN_ROOT}/scripts/script.sh"
      }]
    }]
  }
}
```

**Fix existing hooks:**
- Correct event types
- Fix matcher patterns
- Update scripts

**Test hooks after changes:**
- Verify they fire correctly
- Verify they don't fire incorrectly

### 5. Metadata

**Update plugin.json:**
- Description matches current content
- Version reflects changes
- Component counts accurate

---

## Coherence Updates

After component changes, ensure coherence:

**Cross-references:**
- Skills reference related skills/commands/agents
- Commands reference related guidance
- Agents reference handoff targets

**Terminology:**
- Consistent terms across all components
- Canonical names for concepts

**Navigation:**
- Router skill (if exists) lists all specialists
- Clear paths to find components

---

## Version Management

**Assess total impact:**

| Bump | When | Examples |
|------|------|----------|
| Patch (x.y.Z) | Low impact | Typos, formatting, minor clarifications |
| Minor (x.Y.0) | Medium impact | Enhanced guidance, new components, better examples |
| Major (X.0.0) | High impact | Components removed, structural changes, philosophy shifts |

**Decision logic:**
- New components added? → Minor minimum
- Components removed? → Consider Major
- Only fixes/clarifications? → Patch
- Multiple significant enhancements? → Minor
- Changed plugin philosophy? → Major

**Default for maintenance: Minor bump**

---

## Git Commit

**Single commit with all changes:**

```bash
git add plugins/[plugin-name]/
git commit -m "$(cat <<'EOF'
feat(plugin-name): enhance [summary]

Structure:
- Added [count] commands: [names]
- Added [count] agents: [names]
- Removed [count] duplicate skills: [names]
- Updated router

Content:
- Enhanced [skill]: [changes]
- Fixed [command]: [changes]
- Improved [agent]: [changes]

Coherence:
- Updated cross-references
- Aligned terminology
- Updated metadata

Version: [old] → [new] ([patch/minor/major])

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

**Do NOT push** - let user decide when to push.

---

## Red Flags - Execution Phase

| Thought | Reality |
|---------|---------|
| "I'll just write this skill inline" | NO. Skills require superpowers:writing-skills |
| "This command is simple, skip testing" | Simple commands fail edge cases. Test anyway. |
| "I'll add extra improvements while I'm here" | Scope creep. Stick to approved changes. |
| "I'll skip the cross-reference updates" | Broken navigation. Update references. |
| "Version bump doesn't matter" | Version communicates impact. Get it right. |
| "I'll commit each change separately" | Single commit. Track as atomic change. |

**All of these mean: Follow the process. Execute approved scope only.**

---

## Output Summary

```markdown
# Plugin Enhancement Complete: [plugin-name]

## Version: [old] → [new] ([type])

## Summary

| Component | Added | Modified | Removed |
|-----------|-------|----------|---------|
| Skills | [n] | [n] | [n] |
| Commands | [n] | [n] | [n] |
| Agents | [n] | [n] | [n] |
| Hooks | [n] | [n] | [n] |

## Changes

### Skills
- [Change 1]
- [Change 2]

### Commands
- [Change 1]

### Agents
- [Change 1]

### Hooks
- [Change 1]

### Coherence
- [Update 1]

## Git Commit

Created commit: [hash]
Ready to push: Yes
```

---

## Anti-Patterns

| Anti-Pattern | Why Bad | Instead |
|--------------|---------|---------|
| Creating skills without testing | Untested = broken | Use superpowers:writing-skills |
| Scope creep during execution | Unapproved changes | Stick to approved scope |
| Multiple commits | Hard to track | Single atomic commit |
| Skipping verification | Broken components | Test after changes |
| Pushing without user approval | Premature | Let user decide |

---

## The Bottom Line

**Execute approved changes autonomously with high quality.**

- One commit
- Proper versioning
- Complete summary

No shortcuts. No scope creep. Professional execution.
