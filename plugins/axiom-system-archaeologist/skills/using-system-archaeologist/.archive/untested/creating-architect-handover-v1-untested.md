
# Creating Architect Handover

## Purpose

Generate handover reports for axiom-system-architect plugin, enabling seamless transition from analysis (archaeologist) to improvement planning (architect) - synthesizes architecture + quality findings into actionable assessment inputs.

## When to Use

- Coordinator completes architecture analysis and quality assessment
- User requests improvement recommendations or refactoring guidance
- Need to transition from "what exists" (archaeologist) to "what should change" (architect)
- Task specifies creating architect-ready outputs

## Core Principle: Analysis → Assessment Pipeline

**Archaeologist documents neutrally. Architect assesses critically. Handover bridges the two.**

```
Archaeologist → Handover → Architect → Improvements
(neutral docs)  (synthesis) (critical)  (execution)
```

Your goal: Package archaeologist findings into architect-consumable format for assessment and prioritization.

## The Division of Labor

### Archaeologist (This Plugin)

**What archaeologist DOES:**
- Documents existing architecture (subsystems, diagrams, dependencies)
- Identifies quality concerns (complexity, duplication, smells)
- Marks confidence levels (High/Medium/Low)
- Stays neutral ("Here's what you have")

**What archaeologist does NOT do:**
- Critical assessment ("this is bad")
- Refactoring recommendations ("you should fix X first")
- Priority decisions ("security is more important than performance")

### Architect (axiom-system-architect Plugin)

**What architect DOES:**
- Critical quality assessment (direct, no diplomatic softening)
- Technical debt cataloging (structured, prioritized)
- Improvement roadmaps (risk-based, security-first)
- Refactoring strategy recommendations

**What architect does NOT do:**
- Neutral documentation (that's archaeologist's job)
- Implementation execution (future: project manager plugin)

### Handover (This Briefing)

**What handover DOES:**
- Synthesizes archaeologist outputs (architecture + quality)
- Formats findings for architect consumption
- Enables architect consultation (spawn as subagent)
- Bridges neutral documentation → critical assessment

## Output: Architect Handover Report

Create `06-architect-handover.md` in workspace:

```markdown
# Architect Handover Report

**Project:** [System name]
**Analysis Date:** YYYY-MM-DD
**Archaeologist Version:** [axiom-system-archaeologist version]
**Handover Purpose:** Enable architect assessment and improvement prioritization

---

## Executive Summary

**System scale:**
- [N] subsystems identified
- [M] subsystem dependencies mapped
- [X] architectural patterns observed
- [Y] quality concerns flagged

**Assessment readiness:**
- Architecture: [Fully documented / Partial coverage / etc.]
- Quality: [Comprehensive analysis / Sample-based / Not performed]
- Confidence: [Overall High/Medium/Low]

**Recommended architect workflow:**
1. Use `axiom-system-architect:assessing-architecture-quality` for critical assessment
2. Use `axiom-system-architect:identifying-technical-debt` to catalog debt items
3. Use `axiom-system-architect:prioritizing-improvements` for roadmap creation

---

## Archaeologist Deliverables

### Available Documents

- [x] `01-discovery-findings.md` - Holistic scan results
- [x] `02-subsystem-catalog.md` - Detailed subsystem documentation
- [x] `03-diagrams.md` - C4 diagrams (Context, Container, Component)
- [x] `04-final-report.md` - Multi-audience synthesis
- [x] `05-quality-assessment.md` - Code quality analysis (if performed)
- [ ] Additional views: [List if created]

### Key Findings Summary

**Architectural patterns identified:**
1. [Pattern 1] - Observed in: [Subsystems]
2. [Pattern 2] - Observed in: [Subsystems]
3. [Pattern 3] - Observed in: [Subsystems]

**Concerns flagged (from subsystem catalog):**
1. [Concern 1] - Subsystem: [Name], Severity: [Level]
2. [Concern 2] - Subsystem: [Name], Severity: [Level]
3. [Concern 3] - Subsystem: [Name], Severity: [Level]

**Quality issues identified (from quality assessment):**
1. [Issue 1] - Category: [Complexity/Duplication/etc.], Severity: [Critical/High/Medium/Low]
2. [Issue 2] - Category: [Category], Severity: [Level]
3. [Issue 3] - Category: [Category], Severity: [Level]

---

## Architect Input Package

### 1. Architecture Documentation

**Location:** `02-subsystem-catalog.md`, `03-diagrams.md`, `04-final-report.md`

**Usage:** Architect reads these to understand system structure before assessment

**Highlights for architect attention:**
- [Subsystem X]: [Why architect should review - complexity, coupling, etc.]
- [Subsystem Y]: [Specific concern flagged]
- [Dependency pattern]: [Circular dependencies, high coupling, etc.]

### 2. Quality Assessment

**Location:** `05-quality-assessment.md` (if performed)

**Usage:** Architect incorporates quality metrics into technical debt catalog

**Priority issues for architect:**
- **Critical:** [Issue requiring immediate attention]
- **High:** [Important issues from quality assessment]
- **Medium:** [Moderate concerns]

### 3. Confidence Levels

**Usage:** Architect knows which assessments are well-validated vs. tentative

| Subsystem | Confidence | Rationale |
|-----------|------------|-----------|
| [Subsystem A] | High | Well-documented, sampled 5 files |
| [Subsystem B] | Medium | Router provided list, sampled 2 files |
| [Subsystem C] | Low | Missing documentation, inferred structure |

**Guidance for architect:**
- High confidence areas: Proceed with detailed assessment
- Medium confidence areas: Consider deeper analysis before major recommendations
- Low confidence areas: Flag for additional investigation

### 4. Scope and Limitations

**What was analyzed:**
- [Scope description]

**What was NOT analyzed:**
- Runtime behavior (static analysis only)
- Security vulnerabilities (not performed)
- Performance profiling (not available)
- [Other limitations]

**Guidance for architect:**
- Recommendations should acknowledge analysis limitations
- Security assessment may require dedicated review
- Performance concerns should be validated with profiling

---

## Architect Consultation Pattern

### Option A: Handover Only (Document-Based)

**When to use:**
- User will engage architect separately
- Archaeologist completes, then user decides next steps
- Asynchronous workflow preferred

**What to do:**
1. Create this handover report (06-architect-handover.md)
2. Inform user: "Handover report ready for axiom-system-architect"
3. User decides when/how to engage architect

### Option B: Integrated Consultation (Subagent-Based)

**When to use:**
- User requests immediate improvement recommendations
- Integrated archaeologist + architect workflow
- User says: "What should we fix?" or "Assess the architecture"

**What to do:**

1. **Complete handover report first** (this document)

2. **Spawn architect as consultant subagent:**

```
I'll consult with the system architect to assess the architecture and provide improvement recommendations.

[Use Task tool with subagent_type='general-purpose']

Task: "Use the axiom-system-architect plugin to assess the architecture documented in [workspace-path].

Context:
- Architecture analysis is complete
- Handover report available at [workspace-path]/06-architect-handover.md
- Key deliverables: [list deliverables]
- Primary concerns: [top 3-5 concerns from analysis]

Your task:
1. Read the handover report and referenced documents
2. Use axiom-system-architect:assessing-architecture-quality for critical assessment
3. Use axiom-system-architect:identifying-technical-debt to catalog debt
4. Use axiom-system-architect:prioritizing-improvements for roadmap

Deliverables:
- Architecture quality assessment
- Technical debt catalog
- Prioritized improvement roadmap

IMPORTANT: Follow architect skills rigorously - maintain professional discipline, no diplomatic softening, security-first prioritization."
```

3. **Synthesize architect outputs** when subagent returns

4. **Present to user:**
   - Architecture assessment (from architect)
   - Technical debt catalog (from architect)
   - Prioritized roadmap (from architect)
   - Combined context (archaeologist + architect)

### Option C: Architect Recommendation (User Choice)

**When to use:**
- User didn't explicitly request architect engagement
- Archaeologist found concerns warranting architect review
- Offer as next step

**What to say:**

> "I've completed the architecture analysis and documented [N] concerns requiring attention.
>
> **Next step options:**
>
> A) **Immediate assessment** - I can consult the system architect (axiom-system-architect) right now to provide:
>    - Critical architecture quality assessment
>    - Technical debt catalog
>    - Prioritized improvement roadmap
>
> B) **Handover for later** - I've created a handover report (`06-architect-handover.md`) that you can use to engage the architect when ready
>
> C) **Complete current analysis** - Finish with archaeologist deliverables only
>
> Which approach fits your needs?"

## Handover Report Synthesis Approach

### Extract from Subsystem Catalog

**Read `02-subsystem-catalog.md`:**
- Count subsystems (total documented)
- Collect all "Concerns" entries (aggregate findings)
- Note confidence levels (High/Medium/Low distribution)
- Identify dependency patterns (high coupling, circular deps)

**Synthesize into handover:**
- List total subsystems
- Aggregate concerns by category (complexity, coupling, technical debt, etc.)
- Highlight low-confidence areas needing deeper analysis
- Note architectural patterns observed

### Extract from Quality Assessment

**Read `05-quality-assessment.md` (if exists):**
- Extract quality scorecard ratings
- Collect Critical/High severity issues
- Note methodology limitations

**Synthesize into handover:**
- Summarize quality dimensions assessed
- List priority issues by severity
- Note analysis limitations for architect awareness

### Extract from Final Report

**Read `04-final-report.md`:**
- Executive summary (system overview)
- Key findings (synthesized patterns)
- Recommendations (if any)

**Synthesize into handover:**
- Use executive summary for handover summary
- Reference key findings for architect attention
- Note any recommendations already made

## Success Criteria

**You succeeded when:**
- Handover report comprehensively synthesizes archaeologist outputs
- Architect input package clearly structured with locations and usage guidance
- Confidence levels documented for architect awareness
- Scope and limitations explicitly stated
- Consultation pattern matches user's workflow needs
- Written to 06-architect-handover.md following format

**You failed when:**
- Handover is just concatenation of source documents (no synthesis)
- No guidance on which documents architect should read
- Missing confidence level context
- Limitations not documented
- Spawned architect without completing handover first
- No option for user choice (forced integrated consultation)

## Integration with Workflow

This briefing is typically invoked as:

1. **Coordinator** completes final report (04-final-report.md)
2. **Coordinator** (optionally) completes quality assessment (05-quality-assessment.md)
3. **Coordinator** writes task specification for handover creation
4. **YOU** read all archaeologist deliverables
5. **YOU** synthesize into handover report
6. **YOU** write to 06-architect-handover.md
7. **YOU** offer consultation options to user (A/B/C)
8. **(Optional)** Spawn architect subagent if user chooses integrated workflow
9. **Coordinator** proceeds to cleanup or next steps

**Your role:** Bridge archaeologist's neutral documentation and architect's critical assessment through structured handover synthesis.

## Common Mistakes

**❌ Skipping handover report**
"Architect can just read subsystem catalog" → Architect needs synthesized input, not raw docs

**❌ Spawning architect without handover**
"I'll just task architect directly" → Architect works best with structured handover package

**❌ Making architect decisions**
"I'll do the assessment myself" → That's architect's job, not archaeologist's

**❌ Forcing integrated workflow**
"I'll spawn architect automatically" → Offer choice (A/B/C), let user decide

**❌ No synthesis**
"Handover is just copy-paste" → Synthesize, don't concatenate

**❌ Missing limitations**
"I'll hide what we didn't analyze" → Architect needs to know limitations for accurate assessment

## Anti-Patterns

❌ **Overstepping into architect role:**
"This architecture is bad" → "This architecture has [N] concerns documented"

❌ **Incomplete handover:**
Missing confidence levels, no limitations section → Architect can't calibrate recommendations

❌ **Forced workflow:**
Always spawning architect subagent → Offer user choice

❌ **Raw data dump:**
Handover is just file paths → Synthesize key findings for architect

❌ **No consultation pattern:**
Just write report, no next-step guidance → Offer explicit A/B/C options

## The Bottom Line

**Archaeologist documents neutrally. Architect assesses critically. Handover bridges professionally.**

Synthesize findings. Package inputs. Offer consultation. Enable next phase.

The pipeline works when each role stays in its lane and handovers are clean.
