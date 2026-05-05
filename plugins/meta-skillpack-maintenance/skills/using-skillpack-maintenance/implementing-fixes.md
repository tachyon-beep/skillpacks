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

**Create new commands** (match the marketplace's quoted-array convention):

```yaml
---
description: What this command does (one line, no trailing period)
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[args]"
---

# Command Name

[Command content]
```

**Frontmatter rules:**
- `allowed-tools` is a **quoted JSON-style array** of tool name strings. Verify with `head -10 plugins/*/commands/*.md`.
- Include `"Skill"` if the command should be able to dispatch to specialist skills.
- `argument-hint` is a quoted string showing argument shape, e.g. `"[symptom_or_endpoint]"` or `"<file.py> [function_or_script_args]"`.

**Fix existing commands:**
- Update descriptions for clarity
- Adjust tool restrictions
- Fix argument-hint accuracy
- Remove overlap with skills

**Remove obsolete commands:**
- Delete file
- Update any references

### 3. Agents

**Create new agents** (match the marketplace's two-key frontmatter convention; only add `tools:` if you intend to restrict):

```yaml
---
description: What this agent specializes in. Follows SME Agent Protocol with confidence/risk assessment.
model: sonnet
---

# Agent Name

[Agent system prompt — 1–2 paragraphs of role, scope, and standards.]

**Protocol**: You follow the SME Agent Protocol defined in `meta-sme-protocol:sme-agent-protocol`. Before [acting], READ [relevant inputs]. Your output MUST include Confidence Assessment, Risk Assessment, Information Gaps, and Caveats sections.

## When to Activate

<example>
User: "[matching task]"
Action: Activate — [reason]
</example>

<example>
User: "[non-matching task]"
Action: Do NOT activate — [reason, handoff target]
</example>

## Scope Boundaries

**I do:** [list]
**I do NOT:** [list, with handoffs]
```

**Frontmatter rules:**
- The two near-universal keys are `description` and `model`. Adding a `tools:` key restricts the agent to that exact set; omit it to inherit the parent context. Spot-check shows ~60/65 marketplace agents omit `tools:`.
- For SME-style agents (review / audit / advise / critique), end the description with the phrase **"Follows SME Agent Protocol with confidence/risk assessment."** so callers know to expect the four-section output.
- For non-SME agents (autonomous executors like delinters, formatters), the SME protocol does not apply — describe what they do and the model used.

**Slash-command wrapper.** If the new agent is part of a router pattern, also add or update the `.claude/commands/<name>.md` wrapper at the repo root.

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

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Co-author identifier.** Use the model identifier matching the model that did the work (e.g. `Claude Opus 4.7 (1M context)` or `Claude Sonnet 4.6`). Check `~/CLAUDE.md` or recent repo commits for the project's current convention before committing — older templates referenced an unversioned identifier or included a `🤖 Generated with Claude Code` line; current marketplace convention is the model-identified co-author only.

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
