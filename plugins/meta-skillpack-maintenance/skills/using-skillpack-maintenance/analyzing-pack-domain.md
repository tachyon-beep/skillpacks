# Analyzing Plugin Domain

**Purpose:** Establish "what this plugin should cover" and inventory all existing components.

## Workflow: D → B → C → A

### Phase D: User-Guided Scope

**Ask:**
- What is the intended scope of this plugin?
- What boundaries should it respect?
- Target audience: beginners / practitioners / experts?
- Depth: overview / comprehensive / exhaustive?

**Document:**
- User's vision as baseline
- Explicit boundaries (IN scope vs. OUT of scope)
- Success criteria

### Phase B: Domain Mapping

**Generate coverage map from model knowledge:**

1. **Core concepts** - What must this plugin cover?
2. **Key techniques** - Common patterns practitioners need
3. **Advanced topics** - Expert-level material
4. **Cross-cutting** - Testing, debugging, optimization

**Structure:**
```
Foundational:
- [Concept] - Status: Exists / Missing / Needs enhancement

Core:
- [Technique] - Status: ...

Advanced:
- [Topic] - Status: ...
```

**Flag research currency:**
- Stable domains (patterns, algorithms): No research needed
- Evolving domains (AI/ML, security, frameworks): Flag for Phase A

### Phase C: Component Inventory

**Inventory ALL plugin components:**

#### Skills
```bash
find plugins/[name]/skills -name "SKILL.md" -type f
```

For each skill:
- Name and description
- Reference sheets (if router skill)
- Domain area covered

#### Commands
```bash
ls plugins/[name]/commands/*.md 2>/dev/null
```

For each command:
- Description from frontmatter
- argument-hint
- What it enables users to do

#### Agents
```bash
ls plugins/[name]/agents/*.md 2>/dev/null
```

For each agent:
- Description from frontmatter
- Model and tools
- Specialization area

#### Hooks
```bash
cat plugins/[name]/hooks/hooks.json 2>/dev/null
```

For each hook:
- Event type (PreToolUse, PostToolUse, etc.)
- Matcher pattern
- Purpose

#### Plugin Metadata
```bash
cat plugins/[name]/.claude-plugin/plugin.json
```

Note: version, description accuracy, skill count

### Phase A: Research (Conditional)

**Only if Phase B flagged domain as evolving.**

**AI/ML domains:** Survey papers, library docs, benchmarks
**Security domains:** OWASP, NIST, recent CVEs
**Framework domains:** Official docs, migration guides

Update coverage map with:
- New techniques/patterns
- Deprecated approaches
- Version-specific considerations

---

## Output Format

```markdown
# Domain Analysis: [plugin-name]

## User-Defined Scope
- Intent: [description]
- Boundaries: [in/out scope]
- Audience: [level]

## Coverage Map

### Foundational
- [Concept 1] - [Exists/Missing/Needs work]
- [Concept 2] - [Status]

### Core Techniques
- [Technique 1] - [Status]

### Advanced
- [Topic 1] - [Status]

## Component Inventory

### Skills ([count])
| Skill | Description | Status |
|-------|-------------|--------|
| [name] | [desc] | [OK/Issue] |

### Reference Sheets ([count])
| Sheet | Parent Skill | Status |
|-------|--------------|--------|
| [name] | [skill] | [OK/Issue] |

### Commands ([count])
| Command | Description | Status |
|---------|-------------|--------|
| /[name] | [desc] | [OK/Issue] |

### Agents ([count])
| Agent | Description | Status |
|-------|-------------|--------|
| [name] | [desc] | [OK/Issue] |

### Hooks
| Event | Matcher | Purpose | Status |
|-------|---------|---------|--------|
| [event] | [pattern] | [why] | [OK/Issue] |

## Gap Analysis

### Missing Components (High Priority)
- [Gap] - Foundational concept not covered
  - Recommended: [skill/command/agent]
  - Scope: [small/medium/large]

### Missing Components (Medium Priority)
- [Gap] - Advanced topic not covered

### Duplicates/Overlaps
- [Component A] and [Component B] overlap on [topic]

### Obsolete
- [Component] uses deprecated approach

## Research Currency
- Domain stability: [Stable/Evolving]
- Research needed: [Yes/No]
- Currency concerns: [List]
```

---

## Component Type Selection

**When is a gap a skill vs command vs agent?**

| Use Case | Component |
|----------|-----------|
| Model should auto-invoke based on context | Skill |
| User explicitly triggers with `/name` | Command |
| Autonomous specialist for complex subtasks | Agent |
| Automated response to tool events | Hook |

**Examples:**
- "Help with debugging" → Skill (auto-invoked)
- "Deploy this model" → Command (explicit action)
- "Review this code for security" → Agent (autonomous specialist)
- "Format on save" → Hook (automated)

---

## Proceeding

After completing investigation, hand off to `reviewing-pack-structure.md` for scorecard generation.
