
# Creating Architect Handover

## Purpose

Generate handover reports for axiom-system-architect plugin - enables transition from neutral documentation (archaeologist) to critical assessment (architect).

## When to Use

- User requests prioritization: "What should we fix first?"
- User asks for improvement recommendations
- Analysis complete, user needs actionable next steps
- Quality issues identified that need criticality assessment

## Core Principle: Role Discipline

**Archaeologist documents. Architect assesses. Never blur the line.**

```
❌ ARCHAEOLOGIST: "Fix payment errors first, then database config, then documentation"
✅ ARCHAEOLOGIST: "I'll consult the system architect to provide prioritized recommendations"

❌ ARCHAEOLOGIST: "This is critical because it affects payments"
✅ ARCHAEOLOGIST: "Payment error handling issue documented (see quality assessment)"
✅ ARCHITECT: "Critical - payment integrity risk, priority P1"
```

**Your role:** Bridge neutral analysis to critical assessment, don't do the assessment yourself.

## MANDATORY: Stop Before Answering Prioritization

**When user asks:** "What should we fix first?" / "What's most critical?" / "Where should we start?"

**STOP. DO NOT ANSWER. YOU MUST:**

1. **Recognize:** This is a prioritization question (architect domain)
2. **Create handover:** Write 06-architect-handover.md FIRST
3. **Offer consultation:** Present Pattern A/B/C ONLY
4. **Wait for user choice:** Do NOT provide your own priority assessment

**NEVER:**
- Answer the question directly ("Fix X first")
- List issues in priority order as your recommendation
- Explain why something is critical/important
- Create roadmaps or sprint plans
- Say "obviously X is more important than Y"

**If you catch yourself about to answer:** STOP. You're violating role boundaries. Create handover instead.

## The Role Confusion Trap (from baseline testing)

**User asks:** "What should we fix first?"

**Baseline failure:** Archaeologist provides prioritized recommendations directly
- "Fix payment errors first" (made criticality call)
- "Security is priority" (made security judgment)
- "Sprint 1: X, Sprint 2: Y" (made timeline decisions)

**Why this fails:**
- Archaeologist analyzed with neutral lens
- Architect brings critical assessment framework
- Architect has security-first mandate, pressure-resistant prioritization
- Blending roles produces inconsistent rigor

**Correct response:** Recognize question belongs to architect, facilitate handover

## Division of Labor (MANDATORY)

### Archaeologist (This Plugin)

**What you DO:**
- Document architecture (subsystems, diagrams, dependencies)
- Identify quality concerns (complexity, duplication, smells)
- Mark confidence levels (High/Medium/Low)
- **Stay neutral:** "Here's what exists, here's what I observed"

**What you do NOT do:**
- Assess criticality ("this is critical")
- Prioritize fixes ("fix X before Y")
- Make security judgments ("security risk trumps performance")
- Create sprint roadmaps

### Architect (axiom-system-architect Plugin)

**What they DO:**
- Critical assessment ("this is bad, here's severity")
- Technical debt prioritization (risk-based, security-first)
- Refactoring strategy recommendations
- Improvement roadmaps with timelines

**What they do NOT do:**
- Neutral documentation (that's your job)

### Handover (Your Job Now)

**What you DO:**
- Synthesize archaeologist outputs (catalog + quality + diagrams + report)
- Package for architect consumption
- Offer architect consultation (3 patterns below)
- **Bridge the roles, don't blend them**

## Handover Output

Create `06-architect-handover.md`:

```markdown
# Architect Handover Report

**Project:** [Name]
**Analysis Date:** YYYY-MM-DD
**Purpose:** Enable architect assessment and improvement prioritization

## Archaeologist Deliverables

**Available documents:**
- 01-discovery-findings.md
- 02-subsystem-catalog.md
- 03-diagrams.md (Context, Container, Component)
- 04-final-report.md
- 05-quality-assessment.md (if performed)

## Key Findings Summary

**Architectural concerns from catalog:**
1. [Concern from subsystem X] - Confidence: [High/Medium/Low]
2. [Concern from subsystem Y] - Confidence: [High/Medium/Low]

**Quality issues from assessment (if performed):**
- Critical: [Count] issues
- High: [Count] issues
- Medium: [Count] issues
- Low: [Count] issues

[List top 3-5 issues with file:line references]

## Architect Input Package

**For critical assessment:**
- Read: 02-subsystem-catalog.md (architectural concerns)
- Read: 05-quality-assessment.md (code quality evidence)
- Read: 04-final-report.md (synthesis and patterns)

**For prioritization:**
- Use: axiom-system-architect:prioritizing-improvements
- Note: Archaeologist findings are neutral observations
- Apply: Security-first framework, risk-based priority

## Confidence Context

[Confidence levels per subsystem - helps architect calibrate]

## Limitations

**What was analyzed:** [Scope]
**What was NOT analyzed:** [Gaps]
**Recommendations:** [What architect should consider for deeper analysis]

## Next Steps

[Offer consultation patterns - see below]
```

## Three Consultation Patterns

### Pattern A: Offer Architect Consultation (RECOMMENDED)

**When user asks prioritization questions:**

> "This question requires critical assessment and prioritization, which is the system architect's domain.
>
> **Options:**
>
> **A) Architect consultation now** - I'll consult axiom-system-architect to provide:
>    - Critical architecture quality assessment
>    - Risk-based technical debt prioritization
>    - Security-first improvement roadmap
>    - Sprint-ready recommendations
>
> **B) Handover report only** - I'll create handover package, you engage architect when ready
>
> Which approach fits your needs?"

**Rationalization to counter:** "I analyzed it, so I should just answer the question"

**Reality:** Analysis ≠ Assessment. Architect brings critical framework you don't have.

### Pattern B: Integrated Consultation (If User Chooses)

**If user selects Option A:**

1. Create handover report first (`06-architect-handover.md`)
2. Spawn architect as subagent using Task tool
3. Provide context: handover report location, key concerns, deliverables
4. Synthesize architect outputs when returned
5. Present to user

**Prompt structure:**
```
Use axiom-system-architect to assess architecture and prioritize improvements.

Context:
- Handover report: [workspace]/06-architect-handover.md
- Key concerns: [top 3-5 from analysis]
- User needs: Prioritized improvement recommendations

Tasks:
1. Use assessing-architecture-quality for critical assessment
2. Use identifying-technical-debt to catalog debt
3. Use prioritizing-improvements for security-first roadmap

IMPORTANT: Follow architect discipline - no diplomatic softening, security-first prioritization
```

### Pattern C: Handover Only (If User Wants Async)

**If user wants handover without immediate consultation:**

1. Create handover report (`06-architect-handover.md`)
2. Inform user: "Handover ready for architect when you're ready to engage"
3. Provide brief guide on using architect plugin

## Common Rationalizations (STOP SIGNALS)

| Excuse | Why This Fails | What To Do Instead |
|--------|----------------|-------------------|
| "I know the codebase, I can prioritize" | Knowing ≠ Risk framework. You analyzed neutrally, architect assesses critically. | Create handover, spawn architect |
| "Just helping them prioritize for tomorrow" | Helping by guessing priorities = wrong priorities. | Provide correct answer via architect |
| "Architect would say same thing" | Prove it by asking architect. Don't assume. | Handover + consultation takes 5min |
| "User wants MY recommendation" | User wants CORRECT recommendation from right role. | Architect is right role for priority |
| "It's obvious payment errors are critical" | Obvious ≠ risk-based security-first assessment. | Let architect apply framework |
| "I'll save time by combining roles" | Saves 5min, loses discipline forever. | Time saved ≠ correctness maintained |
| "User chose 'handover only' option" | Handover only = no immediate consult. Still need architect eventually. | Create handover, explain architect needed |
| "I'm just summarizing findings" | Listing top 5 issues = implicit priority. | List ALL issues, let architect prioritize |
| "I can infer priority from severity" | Severity ≠ priority. Security-first ordering requires architect. | Document findings, spawn architect |

**ENFORCEMENT:** If you recognize these thoughts, you are about to violate boundaries. STOP. Create handover.

**From baseline testing:** Agents will try to "help" by doing architect's job. Skill must enforce boundary.

## Success Criteria

**You succeeded when:**
- Recognized prioritization question belongs to architect
- Created handover synthesis (not raw file dump)
- Offered architect consultation (3 patterns)
- Maintained role boundary (documented concerns, didn't assess criticality)
- Handover written to 06-architect-handover.md

**You failed when:**
- Directly provided prioritized recommendations
- Made criticality assessments ("this is critical")
- Created sprint roadmaps yourself
- Skipped handover, worked from raw files
- Blended archaeologist and architect roles
- Answered "what should we fix first" directly
- Listed issues in priority order without architect
- Said "obviously X is more important"
- Provided timeline recommendations yourself
- Used time pressure to justify skipping architect

## Integration with Workflow

Typically invoked after final report completion. User requests improvement recommendations. You recognize this requires architect, create handover, offer consultation.

**Pipeline:** Archaeologist → Handover → Architect → Recommendations

