---
name: using-system-archaeologist
description: Use when analyzing existing codebases to generate architecture documentation - coordinates subagent-driven exploration with mandatory workspace structure, validation gates, and pressure-resistant workflows
---

# System Archaeologist - Codebase Architecture Analysis

## Overview

Analyze existing codebases through coordinated subagent exploration to produce comprehensive architecture documentation with C4 diagrams, subsystem catalogs, and architectural assessments.

**Core principle:** Systematic archaeological process with quality gates prevents rushed, incomplete analysis.

---

## Context Conservation (CRITICAL)

**You are a COORDINATOR, not an analyst.** Your primary context is precious - preserve it for orchestration decisions, not detailed code reading.

### The Delegation Imperative

| Task Type | Your Role | Subagent Role |
|-----------|-----------|---------------|
| Subsystem analysis | Spawn `codebase-explorer` | Read files, produce catalog entries |
| Validation | Spawn `analysis-validator` | Check contracts, produce reports |
| Diagram generation | Spawn diagram subagent | Generate Mermaid/PlantUML |
| Quality assessment | Spawn quality subagent | Analyze patterns, produce assessment |
| Security surface | Spawn security subagent | Map boundaries, flag concerns |

### What You DO (Coordinator):
- Create workspace and coordination plan
- Make orchestration decisions (parallel vs sequential)
- Spawn subagents with clear task specifications
- Read subagent outputs (summaries, not raw files)
- Make proceed/revise decisions based on validation
- Synthesize final deliverables from subagent work

### What You DO NOT Do (Delegate Instead):
- Read implementation files directly (subagent reads, you get summary)
- Perform detailed pattern analysis (spawn specialist)
- Write catalog entries yourself (spawn codebase-explorer)
- Validate your own work (spawn analysis-validator)
- Generate diagrams from scratch (spawn diagram agent)

### Rationalization Blockers

| Thought | Reality |
|---------|---------|
| "I'll just quickly read this file" | Spawn subagent. Your context is for coordination. |
| "It's faster if I do it myself" | Subagents preserve your context for later decisions. |
| "Only a few files to check" | "Few" files become many. Delegate from the start. |
| "I need to understand the code" | You need to understand the STRUCTURE. Subagents report findings. |
| "Spawning overhead isn't worth it" | Context exhaustion is worse. Always delegate detailed work. |

---

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

Reference sheets like `analyzing-unknown-codebases.md` are at:
  `skills/using-system-archaeologist/analyzing-unknown-codebases.md`

NOT at:
  `skills/analyzing-unknown-codebases.md` ← WRONG PATH

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

**After workspace creation, present deliverable options using AskUserQuestion tool.**

See [deliverable-options.md](deliverable-options.md) for the complete menu (Options A-G).

**Key options:**
- **A) Full Analysis** - Comprehensive documentation
- **B) Quick Overview** - Fast turnaround, documented limitations
- **C) Architect-Ready** - Full analysis + improvement planning
- **E) Full + Security** - Security surface mapping included
- **F) Full + Quality** - Test infrastructure + dependency analysis
- **G) Comprehensive** - Everything

**Common rationalization:** "User didn't specify, so I'll default to full analysis"

**Reality:** Always offer choice explicitly. Different needs require different outputs.

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

**Common rationalization:** "I'll just do the work, documentation is overhead"

**Reality:** Undocumented work is unreviewable and non-reproducible.

### Step 3: Holistic Assessment First

**Before diving into details, perform systematic scan:**

1. **Directory structure** - Map organization (feature? layer? domain?)
2. **Entry points** - Find main files, API definitions, config
3. **Technology stack** - Languages, frameworks, dependencies
4. **Subsystem identification** - Identify 4-12 major cohesive groups

Write findings to `01-discovery-findings.md`

**Common rationalization:** "I can see the structure, no need to document it formally"

**Reality:** What's obvious to you now is forgotten in 30 minutes.

### Step 4: Subagent Orchestration Strategy

**Decision point:** Sequential vs Parallel vs **Ultralarge per-module track**

**Use SEQUENTIAL when:**
- Project < 5 subsystems
- Subsystems have tight interdependencies
- Quick analysis needed (< 1 hour)

**Use PARALLEL when:**
- Project ≥ 5 independent subsystems
- Large codebase (20K–100K LOC, 10+ plugins/services)
- Subsystems are loosely coupled

**Use ULTRALARGE PER-MODULE TRACK when ANY of:**
- Source LOC > 100,000
- Subsystem candidates > 12
- Test corpus LOC ≥ source LOC
- Doc corpus > 500 markdown files
- Plugin-registry architecture with extensible categories

The ultralarge track abandons single-pass orchestration and switches to manual subsystem partitioning + per-module review-with-scribe. **Use the `/analyze-ultralarge` command, not `/analyze-codebase`.** See:
- Tier criteria & partitioning protocol → [partitioning-ultralarge-repos.md](partitioning-ultralarge-repos.md)
- Per-module loop → [module-by-module-with-scribe.md](module-by-module-with-scribe.md)
- Findings schema (load-bearing) → [findings-schema.md](findings-schema.md)

**Common rationalization:** "Solo work is faster than coordination overhead"

**Reality:** For large systems, orchestration overhead (5 min) saves hours of sequential work. For ultralarge systems, single-pass orchestration produces a thin, sample-driven catalog that misses entire subsystems — the per-module track is non-negotiable at that scale.

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

### Step 6: Validation Gates (MANDATORY)

**After EVERY major document is produced, validate before proceeding.**

#### SPAWN VALIDATION SUBAGENT (Required for ≥3 subsystems)

- Spawn dedicated validation subagent using Task tool
- Agent reads document + contract, produces validation report
- Write validation report to `temp/validation-[document].md`

**Self-validation ONLY permitted when ALL conditions met:**
1. Single-subsystem analysis (1-2 subsystems only)
2. Total analysis time < 30 minutes
3. YOU personally did ALL the work (no subagents involved)
4. AND you document validation EVIDENCE (not just checkmarks)

**If ANY condition is not met → SPAWN VALIDATION SUBAGENT. No exceptions.**

#### Validation Rationalization Blockers

| Excuse | Reality |
|--------|---------|
| "We have 45 minutes, no time for validation" | Validation takes 5-10 minutes. Spawn validator. |
| "I already reviewed it while writing" | Self-review ≠ validation. Spawn validator. |
| "I'll do systematic self-validation" | Multiple subsystems require independent validator. Spawn validator. |
| "Most work is already done" | Prior work must be validated. Spawn validator. |
| "Time pressure - I'll document limitations instead" | Documented limitations don't excuse skipping validation. Spawn validator. |
| "It's a simple codebase" | Simple ≠ correct. Spawn validator. |

#### Validation Status Meanings

- **APPROVED** → Proceed to next phase
- **NEEDS_REVISION** (warnings) → Fix non-critical issues, document as tech debt, proceed
- **NEEDS_REVISION** (critical) → BLOCK. Fix issues, re-validate. Max 2 retries, then escalate to user.

### Step 7: Handle Validation Failures

**When validator returns NEEDS_REVISION with CRITICAL issues:**

1. **Read validation report** (temp/validation-*.md)
2. **Identify specific issues** (not general "improve quality")
3. **Spawn original subagent again** with fix instructions
4. **Re-validate** after fix
5. **Maximum 2 retries** - if still failing, escalate to user

**DO NOT:**
- Proceed to next phase despite BLOCK status
- Make fixes yourself without re-spawning subagent
- Rationalize "it's good enough"
- Question validator authority

---

## Working Under Pressure

### Time Constraints Are Not Excuses to Skip Process

**WRONG response:** Skip workspace, skip validation, rush deliverables

**RIGHT response:** Scope appropriately while maintaining process

**Example scoping for 3-hour deadline:**

```markdown
## Coordination Plan
- Time constraint: 3 hours until stakeholder presentation
- Strategy: SCOPED ANALYSIS with quality gates maintained
- Timeline:
  - 0:00-0:05: Create workspace, write coordination plan
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

### Resuming Prior Work (Sunk-Cost Discipline)

**Common scenario:** "We started this analysis last week, finish it."

Prior work is a *hypothesis*, not a foundation. Validate before continuing.

**Checklist:**

1. **Find existing workspace** - Look in `docs/arch-analysis-*/`.
2. **Read the coordination log** (`00-coordination.md`) - Understand what was done, what remains, and why work stopped.
3. **Assess quality** - Is the prior catalog/diagram/report contract-compliant? Are confidence levels documented? Were validation gates passed?
4. **Make an EXPLICIT continue/archive/salvage decision:**
   - **Continue** - Prior work passes contract + validation. Resume from the last checkpoint. Update coordination log.
   - **Archive** - Prior work is structurally flawed (wrong scope, wrong contract, no validation). Archive the old workspace (`mv docs/arch-analysis-OLD docs/arch-analysis-OLD.archived-$(date +%Y-%m-%d)`), start fresh, document why.
   - **Salvage** - Mixed quality. Keep good artifacts (e.g., discovery findings), redo bad ones (e.g., un-validated catalog). Document salvage decisions in the new coordination log.
5. **For incremental analysis on a moving codebase** (prior analysis was complete but the code has since changed), use the delta-analysis path - see [incremental-analysis.md](incremental-analysis.md).

**DO NOT** assume prior work is correct just because it exists. **DO NOT** silently continue without recording a decision.

**Coordination log entry:**

```markdown
## Resume Decision - [timestamp]
- Detected existing workspace: docs/arch-analysis-2026-05-15-1430/
- Last completed step: 02-subsystem-catalog.md (validated APPROVED)
- Assessment: Catalog contract-compliant, 11/14 subsystems analyzed
- Decision: SALVAGE - keep catalog, redo diagrams (3 missing subsystems), continue from Step 7
- Reasoning: Validator approved catalog; diagrams were skipped at original session end
```

### Extreme Pressure Handling

**If user requests something genuinely impossible** (e.g., "Complete 15-plugin analysis in 1 hour"):

Provide scoped alternatives:

> A) **Quick overview** (1 hour): Holistic scan, plugin inventory, high-level diagram
> B) **Focused deep-dive** (1 hour): Pick 2-3 critical plugins, full analysis of those
> C) **Use existing docs** (15 min): Synthesize existing README.md, CLAUDE.md
> D) **Reschedule** (recommended): Full analysis takes 4-6 hours for this scale

**DO NOT refuse entirely. Provide realistic scoped alternatives.**

---

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
| "Meeting-ready outputs" justify shortcuts | Stakeholders deserve accurate info, not rushed guesses. |

---

## Specialist Subagent Integration

For complex codebases, leverage specialist subagents from other skillpacks.

See [specialist-integration.md](specialist-integration.md) for:
- When to invoke specialists (by codebase type)
- Spawning pattern and output integration
- Cross-pack handoff points
- Technical accuracy escalation

---

## Workflow Summary

```
1. Create workspace (docs/arch-analysis-YYYY-MM-DD-HHMM/)
1.5. Offer deliverable menu (A/B/C/D/E/F/G) - user chooses scope
2. Write coordination plan (00-coordination.md) with deliverable choice
3. Holistic assessment → 01-discovery-findings.md
4. Decide: Sequential or Parallel? (document reasoning)
5. Spawn subagents for analysis → 02-subsystem-catalog.md
6. VALIDATE subsystem catalog (mandatory gate)
6.5. (Optional) Code quality assessment → 05-quality-assessment.md
6.6. (Optional) Security surface mapping → 07-security-surface.md
6.7. (Optional) Test infrastructure analysis → 09-test-infrastructure.md
6.8. (Optional) Dependency analysis → 10-dependency-analysis.md
7. Spawn diagram generation → 03-diagrams.md
8. VALIDATE diagrams (mandatory gate)
9. Synthesize final report → 04-final-report.md
10. VALIDATE final report (mandatory gate)
11. (Optional) Generate architect handover → 06-architect-handover.md
12. Provide cleanup recommendations for temp/
```

**Every step is mandatory except optional steps (6.5-6.8, 11). No exceptions.**

---

## Success Criteria

**You have succeeded when:**
- Workspace structure exists with all numbered documents
- Coordination log documents all major decisions
- All outputs passed validation gates
- Subagent orchestration used appropriately for scale
- Limitations explicitly documented if time-constrained

**You have failed when:**
- Files scattered outside workspace
- No coordination log showing decisions
- Validation skipped "to save time"
- Self-validated multi-subsystem work instead of spawning validator
- Worked solo despite clear parallelization opportunity
- Produced rushed outputs without limitation documentation

---

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

## Documentation Contracts

See individual skill files for detailed contracts:
- `01-discovery-findings.md` contract → [analyzing-unknown-codebases.md](analyzing-unknown-codebases.md)
- `02-subsystem-catalog.md` contract → [analyzing-unknown-codebases.md](analyzing-unknown-codebases.md)
- `03-diagrams.md` contract → [generating-architecture-diagrams.md](generating-architecture-diagrams.md)
- `04-final-report.md` contract → [documenting-system-architecture.md](documenting-system-architecture.md)
- `05-quality-assessment.md` contract → [assessing-code-quality.md](assessing-code-quality.md)
- `06-architect-handover.md` contract → [creating-architect-handover.md](creating-architect-handover.md)
- `07-security-surface.md` contract → [mapping-security-surface.md](mapping-security-surface.md)
- `08-incremental-report.md` contract → [incremental-analysis.md](incremental-analysis.md)
- `09-test-infrastructure.md` contract → [analyzing-test-infrastructure.md](analyzing-test-infrastructure.md)
- `10-dependency-analysis.md` contract → [analyzing-dependencies.md](analyzing-dependencies.md)
- Validation protocol → [validating-architecture-analysis.md](validating-architecture-analysis.md)
- Language/framework patterns → [language-framework-patterns.md](language-framework-patterns.md)
- Deliverable options → [deliverable-options.md](deliverable-options.md)
- Specialist integration → [specialist-integration.md](specialist-integration.md)
- Ultralarge tier criteria & partitioning → [partitioning-ultralarge-repos.md](partitioning-ultralarge-repos.md)
- Ultralarge per-module loop → [module-by-module-with-scribe.md](module-by-module-with-scribe.md)
- Ultralarge findings schema (load-bearing) → [findings-schema.md](findings-schema.md)
