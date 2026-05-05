---
name: using-skillpack-maintenance
description: Use when maintaining, enhancing, or modifying existing Claude Code plugins - handles skills, commands, agents, hooks, and reference sheets through systematic domain analysis, structure review, behavioral testing, and quality improvements
---

# Plugin Maintenance

Systematic maintenance of Claude Code plugins including skills, commands, agents, hooks, and reference sheets.

## Core Principle

**Maintenance = behavioral validation, not syntactic checking.** Test if components guide Claude correctly, not if they parse correctly.

## Scope: What This Skill Maintains

| Component | Location | Frontmatter (observed in this marketplace) |
|-----------|----------|-------------|
| **Skills** | `skills/*/SKILL.md` | `name`, `description` (and optionally `allowed-tools`) |
| **Reference sheets** | `skills/using-*/*.md` | (none — content files referenced by a router SKILL.md) |
| **Commands** | `commands/*.md` | `description`, `allowed-tools` (quoted array), `argument-hint` |
| **Agents** | `agents/*.md` | `description`, `model` (most repo agents declare ONLY these two — `tools` is rare) |
| **Hooks** | `hooks/hooks.json` | JSON with event matchers |
| **Slash-command routers** | repo-root `.claude/commands/*.md` | thin wrapper exposing a router skill as `/name` |

**Note on `tools:` for agents.** A spot-check of the 32-plugin marketplace shows ~60/65 agents declare only `description` and `model`. Adding a `tools:` key restricts the agent to that exact set; omit it to inherit the parent context. Recommend `tools:` only when restriction is intentional and audited.

## When to Use

**Use for:**
- Enhancing existing plugins (e.g., "refresh yzmir-deep-rl")
- Adding/removing/modifying components
- Identifying coverage gaps
- Validating component quality

**Do NOT use for:**
- Creating new plugins from scratch (design first)
- Creating brand new skills (use `superpowers:writing-skills`)

---

## Reference Sheet Location

All reference sheets are in this skill's directory:
- `analyzing-pack-domain.md` - Domain investigation
- `reviewing-pack-structure.md` - Structure review, scorecard
- `testing-skill-quality.md` - Behavioral testing methodology
- `implementing-fixes.md` - Execution and versioning

When reading `analyzing-pack-domain.md`, find it at:
  `skills/using-skillpack-maintenance/analyzing-pack-domain.md`

---

## Workflow: Review → Discuss → Execute

### Stage 1: Investigation

**Load:** `analyzing-pack-domain.md`

1. **User scope** - Ask about intent, boundaries, target audience
2. **Domain mapping** - What should this plugin cover?
3. **Inventory audit** - What exists? Skills, commands, agents, hooks?
4. **Gap analysis** - What's missing vs. coverage map?

**Output:** Coverage map, component inventory, gaps identified

### Stage 2: Structure Review

**Load:** `reviewing-pack-structure.md`

Generate fitness scorecard:
- **Critical** - Plugin unusable, consider rebuild
- **Major** - Significant gaps or structural issues
- **Minor** - Polish and improvements
- **Pass** - Structurally sound

**Decision gate:** Present scorecard → User decides: Proceed / Rebuild / Cancel

### Stage 3: Behavioral Testing

**Load:** `testing-skill-quality.md`

Test each component with challenging scenarios:
- **Pressure tests** - Does it hold under "just do it quickly" pressure?
- **Edge cases** - Does it handle corner cases?
- **Real-world complexity** - Does it guide correctly in messy situations?

**Output:** Per-component test results (Pass / Fix needed)

### Stage 4: Discussion

Present findings by category:

**Gaps requiring new components:**
- Skills needing `superpowers:writing-skills` (each = separate RED-GREEN-REFACTOR)
- Commands to create
- Agents to create

**Existing components needing fixes:**
- Skills/commands/agents with behavioral failures
- Hooks with issues

**Get user approval before execution.**

### Stage 5: Execution

**Load:** `implementing-fixes.md`

**CRITICAL CHECKPOINT:**
If gaps were identified → Use `superpowers:writing-skills` for EACH new skill first.
Do NOT create new skills inline. They require behavioral testing.

Execute approved changes:
1. Structural fixes (remove duplicates, update router)
2. Content enhancements (fix behavioral failures)
3. Component creation (commands, agents - NOT skills)
4. Version bump and commit

---

## Component-Specific Guidance

### Skills (SKILL.md)

```yaml
---
name: skill-name
description: Use when [trigger condition] — [what the skill does]. Routes to / loads [reference sheets] when [...].
---
```

**Key questions:**
- Does the description start with **"Use when..."** (the dominant repo convention for discoverability)? Browse `plugins/*/skills/*/SKILL.md` to confirm.
- Does the description trigger correct activation?
- Is guidance actionable under pressure?
- Are there missing anti-patterns?

**`allowed-tools` on skills is rare** in this marketplace; most SKILL.md files omit it and let the calling context govern tool access. Only add it if the skill must restrict tool use.

### Commands (commands/*.md)

```yaml
---
description: What this command does (one line, no trailing period)
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[optional_arg]"
---
```

**Frontmatter style observed across this marketplace:**
- `allowed-tools` is a **quoted JSON-style array** of tool name strings — `["Read", "Bash"]`, not `[Read, Bash]`. Verify with `head -10 plugins/*/commands/*.md`.
- `argument-hint` is a quoted string showing argument shape, e.g. `"[symptom_or_endpoint]"` or `"<file.py> [function_or_script_args]"`.
- Most router commands include `"Skill"` in `allowed-tools` so they can dispatch to specialist skills.

**Key questions:**
- Is the command user-invocable (vs. a skill, which is model-invoked)?
- Does it have a clear entry point?
- Are tool restrictions appropriate (and quoted)?

### Agents (agents/*.md)

```yaml
---
description: What this agent specializes in. Follows SME Agent Protocol with confidence/risk assessment.
model: sonnet
---
```

**Frontmatter style observed across this marketplace:**
- The two near-universal keys are `description` and `model` (~60/65 agents declare only these).
- `tools:` is uncommon (~5/65 agents) and acts as a **restriction**. Omit it unless you intend to restrict.
- For SME-style agents (reviewers, auditors, advisors), the description should end with the phrase **"Follows SME Agent Protocol with confidence/risk assessment."** so callers know to expect the four-section output. See `meta-sme-protocol:sme-agent-protocol`.

**SME body convention.** SME agent system prompts in this repo include a `**Protocol**:` line near the top, e.g.:

```markdown
**Protocol**: You follow the SME Agent Protocol defined in `meta-sme-protocol:sme-agent-protocol`. Before reviewing, READ the relevant code. Your output MUST include Confidence Assessment, Risk Assessment, Information Gaps, and Caveats sections.
```

**Key questions:**
- Clear scope boundaries (what it does / doesn't do)?
- Appropriate model selection for complexity?
- Activation examples (positive and negative)?
- If it's an SME agent: does the body cite `meta-sme-protocol:sme-agent-protocol` and require the four output sections?

### Hooks (hooks/hooks.json)

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Write|Edit",
      "hooks": [{"type": "command", "command": "script.sh"}]
    }]
  }
}
```

**Events:** PreToolUse, PostToolUse, UserPromptSubmit, Notification, Stop, SubagentStop, SessionStart, SessionEnd, PreCompact

**Key questions:**
- Correct event type for the use case?
- Matcher pattern accurate?
- Script executable and tested?

### Slash-Command Routers (`.claude/commands/*.md`)

This marketplace exposes router skills (`using-X` skills) as repo-root slash commands so users can invoke them explicitly without competing for skill-discovery context. Per `/home/john/skillpacks/CLAUDE.md`:

> All router skills (`using-X` skills) are available as slash commands in `.claude/commands/` due to skill context limits.

**Example wrapper** (`/home/john/skillpacks/.claude/commands/python-engineering.md`):

```markdown
# Using Python Engineering

## Overview

This meta-skill routes you to the right Python specialist based on symptoms...

## When to Use

Load this skill when:
- Working with Python and encountering problems
- ...
```

**Maintenance check.** When a plugin contains a router skill (any `using-X` SKILL.md), confirm a corresponding `.claude/commands/X.md` exists at the repo root. Missing wrappers mean the router is not user-invocable as a slash command. If a plugin intentionally has no slash-command exposure, document that decision in the plugin's README.

```bash
# List router skills
find plugins -path "*/skills/using-*/SKILL.md"

# List slash-command wrappers
ls .claude/commands/

# Diff to find routers without wrappers
```

---

## Version Bump Rules

| Impact | Bump | Examples |
|--------|------|----------|
| **Low** | Patch (x.y.Z) | Typos, formatting, minor clarifications |
| **Medium** | Minor (x.Y.0) | Enhanced guidance, new components, better examples |
| **High** | Major (X.0.0) | Components removed, structural changes, philosophy shifts |

**Default for maintenance: Minor bump**

---

## Red Flags - Stop and Reconsider

| Thought | Reality |
|---------|---------|
| "I'll write new skills during execution" | NO. Use `superpowers:writing-skills` for each gap |
| "Syntax looks correct, no need to test" | Parsing ≠ effectiveness. Test behavior. |
| "This is a quick fix, skip the process" | Quick untested = broken later |
| "The command/agent is simple enough" | Simple things fail in edge cases. Test anyway. |

---

## Quick Reference

```
Investigation → Scorecard → Testing → Discussion → Execution
     ↓              ↓           ↓           ↓            ↓
  Domain map    Fitness    Behavioral   Present      Apply
  + inventory   rating     validation   + approve    changes
```

**Load briefings at each stage. Test with scenarios. Get approval. Execute.**
