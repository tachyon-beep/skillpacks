---
name: using-system-archaeologist
description: Use when analyzing existing codebases to generate architecture documentation - coordinates subagent-driven exploration with mandatory workspace structure, validation gates, and pressure-resistant workflows
---

# System Archaeologist - Codebase Architecture Analysis

## Overview

Analyze existing codebases through coordinated subagent exploration to produce comprehensive architecture documentation with C4 diagrams, subsystem catalogs, and architectural assessments.

**Core principle:** Systematic archaeological process with quality gates prevents rushed, incomplete analysis.

## When to Use

- User requests architecture documentation for existing codebase
- Need to understand unfamiliar system architecture
- Creating design docs for legacy systems
- Analyzing codebases of any size (small to large)
- User mentions: "analyze codebase", "architecture documentation", "system design", "generate diagrams"

---

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-system-archaeologist/SKILL.md`

Reference sheets like `subsystem-discovery.md` are at:
  `skills/using-system-archaeologist/subsystem-discovery.md`

NOT at:
  `skills/subsystem-discovery.md` ← WRONG PATH

When you see a link like `[subsystem-discovery.md](subsystem-discovery.md)`, read the file from the same directory as this SKILL.md.

---

## Mandatory Workflow

### Step 1: Create Workspace (NON-NEGOTIABLE)

**Before any analysis:**

```bash
mkdir -p docs/arch-analysis-$(date +%Y-%m-%d-%H%M)/temp
```

**Why this is mandatory:**
- Organizes all analysis artifacts in one location
- Enables subagent handoffs via shared documents
- Provides audit trail of decisions
- Prevents file scatter across project

**Common rationalization:** "This feels like overhead when I'm pressured"

**Reality:** 10 seconds to create workspace saves hours of file hunting and context loss.

### Step 1.5: Offer Deliverable Menu (MANDATORY)

**After workspace creation, offer user choice of deliverables:**

**Why this is mandatory:**
- Users may need subset of analysis (quick overview vs. comprehensive)
- Time-constrained scenarios require focused scope
- Different stakeholder needs (exec summary vs. full technical docs)
- Architect-ready outputs have different requirements than documentation-only

Present menu using **AskUserQuestion tool:**

**Question:** "What deliverables do you need from this architecture analysis?"

**Options:**

**A) Full Analysis (Comprehensive)** - Recommended for complete understanding
- All standard documents (discovery, catalog, diagrams, report)
- Optional: Code quality assessment
- Optional: Architect handover report
- Timeline: 2-6 hours depending on codebase size
- Best for: New codebases, major refactoring planning, complete documentation needs

**B) Quick Overview (Essential)** - Fast turnaround for stakeholder presentations
- Discovery findings + high-level diagrams only (Context + Container)
- Executive summary with key findings
- Documented limitations (partial analysis)
- Timeline: 30 minutes - 2 hours
- Best for: Initial assessment, stakeholder presentations, time-constrained reviews

**C) Architect-Ready (Analysis + Improvement Planning)** - Complete analysis with improvement focus
- Full analysis (discovery, catalog, diagrams, report)
- Code quality assessment (mandatory for architect)
- Architect handover report with improvement recommendations
- Optional: Integrated architect consultation
- Timeline: 3-8 hours depending on codebase size
- Best for: Planning refactoring, technical debt assessment, improvement roadmaps

**D) Custom Selection** - Choose specific documents
- User selects from: Discovery, Catalog, Diagrams (which levels?), Report, Quality, Handover
- Timeline: Varies by selection
- Best for: Updating existing documentation, focused analysis

**Document user's choice in coordination plan:**

```markdown
## Deliverables Selected: [Option A/B/C/D]

[If Option D, list specific selections]

**Rationale:** [Why user chose this option]
**Timeline target:** [If time-constrained]
**Stakeholder needs:** [If presentation-driven]
```

**Common rationalization:** "User didn't specify, so I'll default to full analysis"

**Reality:** Always offer choice explicitly. Different needs require different outputs. Assuming full analysis wastes time if user needs quick overview.

### Step 2: Write Coordination Plan

**After documenting deliverable choice, write `00-coordination.md`:**

```markdown
## Analysis Plan
- Scope: [directories to analyze]
- Strategy: [Sequential/Parallel with reasoning]
- Time constraint: [if any, with scoping plan]
- Complexity estimate: [Low/Medium/High]

## Execution Log
- [timestamp] Created workspace
- [timestamp] [Next action]
```

**Why coordination logging is mandatory:**
- Documents strategy decisions (why parallel vs sequential?)
- Tracks what's been done vs what remains
- Enables resumption if work is interrupted
- Shows reasoning for future review

**Common rationalization:** "I'll just do the work, documentation is overhead"

**Reality:** Undocumented work is unreviewable and non-reproducible.

### Step 3: Holistic Assessment First

**Before diving into details, perform systematic scan:**

1. **Directory structure** - Map organization (feature? layer? domain?)
2. **Entry points** - Find main files, API definitions, config
3. **Technology stack** - Languages, frameworks, dependencies
4. **Subsystem identification** - Identify 4-12 major cohesive groups

Write findings to `01-discovery-findings.md`

**Why holistic before detailed:**
- Prevents getting lost in implementation details
- Identifies parallelization opportunities
- Establishes architectural boundaries
- Informs orchestration strategy

**Common rationalization:** "I can see the structure, no need to document it formally"

**Reality:** What's obvious to you now is forgotten in 30 minutes.

### Step 4: Subagent Orchestration Strategy

**Decision point:** Sequential vs Parallel

**Use SEQUENTIAL when:**
- Project < 5 subsystems
- Subsystems have tight interdependencies
- Quick analysis needed (< 1 hour)

**Use PARALLEL when:**
- Project ≥ 5 independent subsystems
- Large codebase (20K+ LOC, 10+ plugins/services)
- Subsystems are loosely coupled

**Document decision in `00-coordination.md`:**

```markdown
## Decision: Parallel Analysis
- Reasoning: 14 independent plugins, loosely coupled
- Strategy: Spawn 14 parallel subagents, one per plugin
- Estimated time savings: 2 hours → 30 minutes
```

**Common rationalization:** "Solo work is faster than coordination overhead"

**Reality:** For large systems, orchestration overhead (5 min) saves hours of sequential work.

### Step 5: Subagent Delegation Pattern

**When spawning subagents for analysis:**

Create task specification in `temp/task-[subagent-name].md`:

```markdown
## Task: Analyze [specific scope]
## Context
- Workspace: docs/arch-analysis-YYYY-MM-DD-HHMM/
- Read: 01-discovery-findings.md
- Write to: 02-subsystem-catalog.md (append your section)

## Expected Output
Follow contract in documentation-contracts.md:
- Subsystem name, location, responsibility
- Key components (3-5 files/classes)
- Dependencies (inbound/outbound)
- Patterns observed
- Confidence level

## Validation Criteria
- [ ] All contract sections complete
- [ ] Confidence level marked
- [ ] Dependencies bidirectional (if A depends on B, B shows A as inbound)
```

**Why formal task specs:**
- Subagents know exactly what to produce
- Reduces back-and-forth clarification
- Ensures contract compliance
- Enables parallel work without conflicts

### Step 6: Validation Gates (MANDATORY)

**After EVERY major document is produced, validate before proceeding.**

**What "validation gate" means:**
- Systematic check against contract requirements
- Cross-document consistency verification
- Quality gate before proceeding to next phase
- NOT just "read it again" - spawn a validator

#### Validation Approach: SPAWN VALIDATION SUBAGENT

**MANDATORY for multi-subsystem work (≥3 subsystems):**
- Spawn dedicated validation subagent using Task tool
- Agent reads document + contract, produces validation report
- Provides "fresh eyes" review - catches errors you're blind to
- Write validation report to `temp/validation-[document].md`

**Self-validation ONLY permitted when ALL conditions met:**
1. Single-subsystem analysis (1-2 subsystems only)
2. Total analysis time < 30 minutes
3. YOU personally did ALL the work (no subagents involved)
4. AND you document validation EVIDENCE (not just checkmarks)

**If ANY condition is not met → SPAWN VALIDATION SUBAGENT. No exceptions.**

#### STOP: Validation Rationalization Blockers

**If you're about to skip spawning a validation subagent, STOP and check:**

❌ "We have 45 minutes, no time for validation"
→ Validation takes 5-10 minutes. You have time. Spawn validator.

❌ "I already reviewed it while writing"
→ Self-review ≠ validation. Fresh eyes catch errors. Spawn validator.

❌ "I'll do systematic self-validation"
→ You completed multiple subsystems. This requires independent validator. Spawn validator.

❌ "Most work is already done, just need to finish"
→ Prior work must be validated before proceeding. Spawn validator.

❌ "Time pressure - I'll document limitations instead"
→ Documented limitations don't excuse skipping validation. Spawn validator.

❌ "It's a simple codebase"
→ Simple ≠ correct. Validation catches format errors regardless of complexity. Spawn validator.

**Reality check:** If your analysis involved ≥3 subsystems or multiple hours of work, you MUST spawn validation subagent. Period.

#### Validation Checklist (for validator subagent)

Validator checks:
- [ ] Contract compliance (all required sections present)
- [ ] Cross-document consistency (subsystems in catalog match diagrams)
- [ ] Confidence levels marked with reasoning
- [ ] No placeholder text ("[TODO]", "[Fill in]")
- [ ] Dependencies bidirectional (A→B means B shows A as inbound)
- [ ] Evidence present (file paths, line numbers cited)

#### When Self-Validation IS Permitted (Rare)

**Only for trivial single-subsystem work, document EVIDENCE:**

```markdown
## Validation Decision - [timestamp]
- Approach: Self-validation (ONLY because: 1-2 subsystems, <30 min work, solo)
- Documents validated: 02-subsystem-catalog.md

**Evidence (REQUIRED):**
- Contract sections verified: Location ✓ (line 5), Responsibility ✓ (line 7)...
- Consistency checks: Subsystem X matches diagram node X ✓
- Specific issues found and resolved: [list any, or "None"]
- Validation took: [X minutes]

- Result: APPROVED with evidence above
```

**Self-validation WITHOUT evidence is not validation - it's rationalization.**

#### Validation Status Meanings

- **APPROVED** → Proceed to next phase
- **NEEDS_REVISION** (warnings) → Fix non-critical issues, document as tech debt, proceed
- **NEEDS_REVISION** (critical) → BLOCK. Fix issues, re-validate. Max 2 retries, then escalate to user.

**Common rationalization:** "Validation slows me down"

**Reality:** Validation catches errors before they cascade. 5-10 minutes validating saves hours debugging diagrams generated from bad data.

**Common rationalization:** "I already checked it, validation is redundant"

**Reality:** "Checked it" ≠ "validated by independent subagent". Your own review misses your own blind spots.

### Step 7: Handle Validation Failures

**When validator returns NEEDS_REVISION with CRITICAL issues:**

1. **Read validation report** (temp/validation-*.md)
2. **Identify specific issues** (not general "improve quality")
3. **Spawn original subagent again** with fix instructions
4. **Re-validate** after fix
5. **Maximum 2 retries** - if still failing, escalate: "Having trouble with [X], need your input"

**DO NOT:**
- Proceed to next phase despite BLOCK status
- Make fixes yourself without re-spawning subagent
- Rationalize "it's good enough"
- Question validator authority ("validation is too strict")

**From baseline testing:** Agents WILL respect validation when it's clear and authoritative. Make validation clear and authoritative.

## Working Under Pressure

### Time Constraints Are Not Excuses to Skip Process

**Common scenario:** "I need this in 3 hours for a stakeholder meeting"

**WRONG response:** Skip workspace, skip validation, rush deliverables

**RIGHT response:** Scope appropriately while maintaining process

**Example scoping for 3-hour deadline:**

```markdown
## Coordination Plan
- Time constraint: 3 hours until stakeholder presentation
- Strategy: SCOPED ANALYSIS with quality gates maintained
- Timeline:
  - 0:00-0:05: Create workspace, write coordination plan (this)
  - 0:05-0:35: Holistic scan, identify all subsystems
  - 0:35-2:05: Focus on 3 highest-value subsystems (parallel analysis)
  - 2:05-2:35: Generate minimal viable diagrams (Context + Component only)
  - 2:35-2:50: Validate outputs
  - 2:50-3:00: Write executive summary with EXPLICIT limitations section

## Limitations Acknowledged
- Only 3/14 subsystems analyzed in depth
- No module-level dependency diagrams
- Confidence: Medium (time-constrained analysis)
- Recommend: Full analysis post-presentation
```

**Key principle:** Scoped analysis with documented limitations > complete analysis done wrong.

### Handling Sunk Cost (Incomplete Prior Work)

**Common scenario:** "We started this analysis last week, finish it"

**Checklist:**
1. **Find existing workspace** - Look in docs/arch-analysis-*/
2. **Read coordination log** - Understand what was done and why stopped
3. **Assess quality** - Is prior work correct or flawed?
4. **Make explicit decision:**
   - **Prior work is good** → Continue from where it left off, update coordination log
   - **Prior work is flawed** → Archive old workspace, start fresh, document why
   - **Prior work is mixed** → Salvage good parts, redo bad parts, document decisions

**DO NOT assume prior work is correct just because it exists.**

**Update coordination log:**

```markdown
## Incremental Work - [date]
- Detected existing workspace from [prior date]
- Assessment: [quality evaluation]
- Decision: [continue/archive/salvage]
- Reasoning: [why]
```

## Common Rationalizations (RED FLAGS)

If you catch yourself thinking ANY of these, STOP:

| Excuse | Reality |
|--------|---------|
| "Time pressure makes trade-offs appropriate" | Process prevents rework. Skipping process costs MORE time. |
| "This feels like overhead" | 5 minutes of structure saves hours of chaos. |
| "Working solo is faster" | Solo works for small tasks. Orchestration scales for large systems. |
| "I'll just write outputs directly" | Uncoordinated work creates inconsistent artifacts. |
| "Validation slows me down" | Validation catches errors before they cascade. |
| "I already checked it" | Self-review misses what fresh eyes catch. |
| "I can't do this properly in [short time]" | You can do SCOPED analysis properly. Document limitations. |
| "Rather than duplicate, I'll synthesize" | Existing docs ≠ systematic analysis. Do the work. |
| "Architecture analysis doesn't need exhaustive review" | True. But it DOES need systematic method. |
| "Meeting-ready outputs" justify shortcuts | Stakeholders deserve accurate info, not rushed guesses. |

**All of these mean:** Follow the process. It exists because these rationalizations lead to bad outcomes.

## Extreme Pressure Handling

**If user requests something genuinely impossible:**

- "Complete 15-plugin analysis with full diagrams in 1 hour"

**Provide scoped alternative:**

> "I can't do complete analysis of 15 plugins in 1 hour while maintaining quality. Here are realistic options:
>
> A) **Quick overview** (1 hour): Holistic scan, plugin inventory, high-level architecture diagram, documented limitations
>
> B) **Focused deep-dive** (1 hour): Pick 2-3 critical plugins, full analysis of those, others documented as "not analyzed"
>
> C) **Use existing docs** (15 min): Synthesize existing README.md, CLAUDE.md with quick verification
>
> D) **Reschedule** (recommended): Full systematic analysis takes 4-6 hours for this scale
>
> Which approach fits your needs?"

**DO NOT:** Refuse the task entirely. Provide realistic scoped alternatives.

## Documentation Contracts

See individual skill files for detailed contracts:
- `01-discovery-findings.md` contract → [analyzing-unknown-codebases.md](analyzing-unknown-codebases.md)
- `02-subsystem-catalog.md` contract → [analyzing-unknown-codebases.md](analyzing-unknown-codebases.md)
- `03-diagrams.md` contract → [generating-architecture-diagrams.md](generating-architecture-diagrams.md)
- `04-final-report.md` contract → [documenting-system-architecture.md](documenting-system-architecture.md)
- `05-quality-assessment.md` contract → [assessing-code-quality.md](assessing-code-quality.md)
- `06-architect-handover.md` contract → [creating-architect-handover.md](creating-architect-handover.md)
- Validation protocol → [validating-architecture-analysis.md](validating-architecture-analysis.md)

## Workflow Summary

```
1. Create workspace (docs/arch-analysis-YYYY-MM-DD-HHMM/)
1.5. Offer deliverable menu (A/B/C/D) - user chooses scope
2. Write coordination plan (00-coordination.md) with deliverable choice
3. Holistic assessment → 01-discovery-findings.md
4. Decide: Sequential or Parallel? (document reasoning)
5. Spawn subagents for analysis → 02-subsystem-catalog.md
6. VALIDATE subsystem catalog (mandatory gate)
6.5. (Optional) Code quality assessment → 05-quality-assessment.md
7. Spawn diagram generation → 03-diagrams.md
8. VALIDATE diagrams (mandatory gate)
9. Synthesize final report → 04-final-report.md
10. VALIDATE final report (mandatory gate)
11. (Optional) Generate architect handover → 06-architect-handover.md
12. Provide cleanup recommendations for temp/
```

**Every step is mandatory except optional steps (6.5, 11). No exceptions for time pressure, complexity, or stakeholder demands.**

**Optional steps triggered by deliverable choice:**
- Step 6.5: Required for "Architect-Ready" (Option C), Optional for "Full Analysis" (Option A)
- Step 11: Required for "Architect-Ready" (Option C), Not included in "Quick Overview" (Option B)

## Success Criteria

**You have succeeded when:**
- Workspace structure exists with all numbered documents
- Coordination log documents all major decisions
- All outputs passed validation gates
- Subagent orchestration used appropriately for scale
- Limitations explicitly documented if time-constrained
- User receives navigable, validated architecture documentation

**You have failed when:**
- Files scattered outside workspace
- No coordination log showing decisions
- Validation skipped "to save time"
- Self-validated multi-subsystem work instead of spawning validator
- Used time pressure to justify skipping independent validation
- Documented validation checkmarks without evidence
- Worked solo despite clear parallelization opportunity
- Produced rushed outputs without limitation documentation
- Rationalized shortcuts as "appropriate trade-offs"

## Anti-Patterns

**❌ Skip workspace creation**
"I'll just write files to project root"

**❌ No coordination logging**
"I'll just do the work without documenting strategy"

**❌ Work solo despite scale**
"Orchestration overhead isn't worth it"

**❌ Skip validation subagent**
"I already reviewed it myself" / "I'll do systematic self-validation"

**❌ Self-validate multi-subsystem work**
"Time constraints mean self-validation is acceptable" (NO - spawn validator)

**❌ Bypass BLOCK status**
"The validation is too strict, I'll proceed anyway"

**❌ Complete refusal under pressure**
"I can't do this properly in 3 hours, so I won't do it" (Should: Provide scoped alternative)

---

## System Archaeologist Specialist Skills

After routing, load the appropriate specialist skill for detailed guidance:

1. [analyzing-unknown-codebases.md](analyzing-unknown-codebases.md) - Systematic codebase exploration, subsystem identification, confidence-based analysis
2. [generating-architecture-diagrams.md](generating-architecture-diagrams.md) - C4 diagrams, abstraction strategies, notation conventions
3. [documenting-system-architecture.md](documenting-system-architecture.md) - Synthesis of catalogs and diagrams into comprehensive reports
4. [validating-architecture-analysis.md](validating-architecture-analysis.md) - Contract validation, consistency checks, quality gates
5. [assessing-code-quality.md](assessing-code-quality.md) - Code quality analysis beyond architecture - complexity, duplication, smells, technical debt assessment
6. [creating-architect-handover.md](creating-architect-handover.md) - Handover reports for axiom-system-architect - enables transition from analysis to improvement planning
