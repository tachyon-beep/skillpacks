# Specialist Subagent Integration

For complex codebases, leverage specialist subagents from other skillpacks to improve analysis quality.

---

## When to Invoke Specialists

### During Step 3 (Holistic Assessment)

| Codebase Type | Specialist to Consider | Benefit |
|---------------|----------------------|---------|
| Python-heavy | `axiom-python-engineering:python-code-reviewer` | Python-specific patterns, anti-patterns |
| PyTorch/ML | `yzmir-pytorch-engineering:pytorch-code-reviewer` | ML architecture patterns, memory issues |
| Deep RL | `yzmir-deep-rl:rl-training-diagnostician` | RL-specific architecture concerns |
| Web API | `axiom-web-backend:api-reviewer` | REST/GraphQL patterns, security |

### During Step 6.5 (Code Quality)

| Quality Concern | Specialist | What They Add |
|----------------|------------|---------------|
| Test architecture | `ordis-quality-engineering:test-suite-reviewer` | Test anti-patterns, pyramid issues |
| Flaky tests | `ordis-quality-engineering:flaky-test-diagnostician` | Root cause identification |
| Coverage gaps | `ordis-quality-engineering:coverage-gap-analyst` | Risk-based prioritization |

### During Step 6.6 (Security Surface)

| Security Area | Specialist | Handoff Package |
|--------------|------------|-----------------|
| Threat modeling | `ordis-security-architect:threat-analyst` | STRIDE analysis on critical subsystems |
| Security controls | `ordis-security-architect:controls-designer` | Control recommendations |

---

## Spawning Specialist Pattern

When invoking a cross-pack specialist:

```markdown
## Specialist Invocation - [timestamp]

**Specialist:** [agent name]
**Scope:** [What to analyze]
**Input:** [Files/artifacts to read]
**Expected output:** [What findings to produce]
**Integration:** [How findings feed into archaeological analysis]
```

**Example invocation for Python codebase:**

```markdown
## Specialist Invocation - 2024-01-15 14:30

**Specialist:** axiom-python-engineering:python-code-reviewer
**Scope:** Review auth/ and api/ subsystems for Python anti-patterns
**Input:**
- Read 02-subsystem-catalog.md for context
- Focus on files listed in Auth and API Gateway entries
**Expected output:**
- Python-specific concerns to add to Concerns sections
- Pattern observations to add to Patterns Observed
**Integration:** Merge findings into 05-quality-assessment.md
```

---

## Specialist Output Integration

When specialist returns findings:

1. **Validate findings** - Ensure they cite specific files/lines
2. **Map to subsystems** - Associate findings with catalog entries
3. **Merge appropriately:**
   - Architecture concerns → 02-subsystem-catalog.md (Concerns section)
   - Quality issues → 05-quality-assessment.md
   - Security flags → 07-security-surface.md
   - Test issues → 09-test-infrastructure.md

4. **Document integration:**

```markdown
## Specialist Integration Log - [timestamp]

- Invoked: [specialist name]
- Findings received: [count]
- Integrated into: [documents]
- Discarded: [count with reasoning if any]
```

---

## Cross-Pack Handoff Points

After archaeological analysis, these specialists can continue the work:

| Analysis Output | Handoff To | When to Handoff |
|----------------|------------|-----------------|
| Architecture issues | `axiom-system-architect:architecture-critic` | Option C, G selected |
| Technical debt | `axiom-system-architect:debt-cataloger` | Significant debt identified |
| Security surface | `ordis-security-architect:threat-analyst` | Option E, G selected |
| Test gaps | `ordis-quality-engineering:coverage-gap-analyst` | Option F, G selected |
| API concerns | `axiom-web-backend:api-reviewer` | Web API subsystems identified |

---

## Validation of Technical Accuracy

**Structural validation** (analysis-validator agent) checks contract compliance.
**Technical accuracy validation** requires domain expertise.

### When to Escalate for Technical Accuracy Review

1. **Confidence: Low** on any critical-path subsystem
2. **Patterns Observed** that you're uncertain about
3. **Technology-specific findings** outside your expertise
4. **Conflicting information** between subsystems

### Escalation Options

| Uncertainty Type | Escalation Path |
|-----------------|-----------------|
| Python patterns | Spawn `axiom-python-engineering:python-code-reviewer` |
| ML architecture | Spawn `yzmir-neural-architectures:architecture-reviewer` |
| Security claims | Spawn `ordis-security-architect:threat-analyst` |
| API design | Spawn `axiom-web-backend:api-architect` |
| General architecture | Spawn `axiom-system-architect:architecture-critic` |
| Unclear after specialist | **Escalate to user** with specific questions |

### User Escalation Template

When specialists cannot resolve uncertainty, escalate to user:

```markdown
## User Escalation - [timestamp]

**Subsystem:** [Which catalog entry]
**Uncertainty:** [What you can't determine]
**Investigation done:**
- [Specialist invoked and findings]
- [Files examined]
- [Why uncertainty remains]

**Specific questions:**
1. [Concrete question the user can answer]
2. [Another specific question if needed]

**Impact:** [How this affects confidence/deliverable quality]
```

### Document All Escalations

```markdown
## Technical Accuracy Escalation - [timestamp]

**Concern:** [What you're uncertain about]
**Subsystem:** [Which entry]
**Specialist invoked:** [agent name]
**Resolution:** [What was determined]
**Catalog updated:** [Yes/No with details]
```

**NEVER mark confidence as High if technical accuracy wasn't verified by domain-appropriate means.**
