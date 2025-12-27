---
description: Initiate systematic codebase architecture analysis with workspace structure, deliverable menu, and subagent orchestration
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[directory_or_scope]"
---

# Analyze Codebase Command

You are initiating a systematic codebase architecture analysis. Follow the mandatory workflow with quality gates.

## Core Principle

**Systematic archaeological process with quality gates prevents rushed, incomplete analysis.**

10 seconds creating workspace saves hours of file hunting. 5 minutes validating saves hours debugging downstream.

## Mandatory Workflow

### Step 1: Create Workspace (NON-NEGOTIABLE)

```bash
mkdir -p docs/arch-analysis-$(date +%Y-%m-%d-%H%M)/temp
```

**Why mandatory:**
- Organizes all artifacts in one location
- Enables subagent handoffs via shared documents
- Provides audit trail of decisions
- Prevents file scatter across project

### Step 2: Offer Deliverable Menu (MANDATORY)

Use AskUserQuestion to offer:

**A) Full Analysis (Comprehensive)**
- All documents: discovery, catalog, diagrams, report
- Optional: Code quality assessment, architect handover
- Timeline: 2-6 hours depending on codebase size

**B) Quick Overview (Essential)**
- Discovery findings + high-level diagrams only
- Executive summary with documented limitations
- Timeline: 30 min - 2 hours

**C) Architect-Ready (Analysis + Improvement Planning)**
- Full analysis + code quality + architect handover
- Enables transition to improvement planning
- Timeline: 3-8 hours

**D) Custom Selection**
- User picks specific documents needed
- Timeline varies by selection

### Step 3: Write Coordination Plan

Create `00-coordination.md`:

```markdown
## Analysis Configuration
- **Scope**: [directories to analyze]
- **Deliverables**: [Option A/B/C/D selected]
- **Strategy**: [Sequential/Parallel with reasoning]
- **Time constraint**: [if any]
- **Complexity estimate**: [Low/Medium/High]

## Execution Log
- [timestamp] Created workspace
- [timestamp] User selected Option [X]
- [timestamp] [Next action]
```

### Step 4: Holistic Assessment

Before detailed analysis:

1. **Directory structure** - Map organization (feature? layer? domain?)
2. **Entry points** - Find main files, API definitions, config
3. **Technology stack** - Languages, frameworks, dependencies
4. **Subsystem identification** - Identify 4-12 major cohesive groups

Write findings to `01-discovery-findings.md`

### Step 5: Decide Orchestration Strategy

**Use SEQUENTIAL when:**
- Project < 5 subsystems
- Tight interdependencies
- Quick analysis needed (< 1 hour)

**Use PARALLEL when:**
- Project ≥ 5 independent subsystems
- Large codebase (20K+ LOC)
- Loosely coupled subsystems

Document decision in coordination plan.

### Step 6: Execute Analysis

**For each subsystem, produce catalog entry:**

```markdown
## [Subsystem Name]

**Location:** `path/to/subsystem/`

**Responsibility:** [One sentence]

**Key Components:**
- `file1.ext` - [Brief description]

**Dependencies:**
- Inbound: [Subsystems depending on this]
- Outbound: [Subsystems this depends on]

**Patterns Observed:**
- [Pattern 1]

**Concerns:**
- [Issues found, or "None observed"]

**Confidence:** [High/Medium/Low] - [Reasoning with evidence]
```

Write to `02-subsystem-catalog.md`

### Step 7: Validation Gates (MANDATORY)

**After EVERY major document:**

For multi-subsystem work (≥3 subsystems):
- MUST spawn validation subagent
- Subagent reads document + contract
- Produces validation report in `temp/validation-*.md`

**Self-validation ONLY when ALL conditions met:**
1. Single-subsystem (1-2 only)
2. Total time < 30 minutes
3. You did ALL work personally
4. Document EVIDENCE (not just checkmarks)

### Step 8: Generate Deliverables

Based on user's selection:

| Document | File | Required For |
|----------|------|--------------|
| Discovery | `01-discovery-findings.md` | All options |
| Catalog | `02-subsystem-catalog.md` | A, C, D |
| Diagrams | `03-diagrams.md` | All options |
| Report | `04-final-report.md` | A, C |
| Quality | `05-quality-assessment.md` | C |
| Handover | `06-architect-handover.md` | C |

## Handling Time Pressure

**Time constraints are NOT excuses to skip process.**

For 3-hour deadline:

```markdown
## Scoped Analysis
- 0:00-0:05: Create workspace, coordination plan
- 0:05-0:35: Holistic scan, identify all subsystems
- 0:35-2:05: Focus on 3 highest-value subsystems
- 2:05-2:35: Generate minimal viable diagrams
- 2:35-2:50: Validate outputs
- 2:50-3:00: Executive summary with EXPLICIT limitations

## Limitations Acknowledged
- Only 3/14 subsystems analyzed in depth
- No module-level dependency diagrams
- Confidence: Medium (time-constrained)
- Recommend: Full analysis post-presentation
```

**Scoped analysis with documented limitations > complete analysis done wrong.**

## Output Format

Final workspace structure:

```
docs/arch-analysis-YYYY-MM-DD-HHMM/
├── 00-coordination.md      # Strategy and execution log
├── 01-discovery-findings.md # Holistic assessment
├── 02-subsystem-catalog.md  # Detailed subsystem entries
├── 03-diagrams.md           # C4 architecture diagrams
├── 04-final-report.md       # Synthesized report
├── 05-quality-assessment.md # (Optional) Code quality
├── 06-architect-handover.md # (Optional) Improvement planning
└── temp/
    ├── task-*.md            # Subagent task specs
    └── validation-*.md      # Validation reports
```

## Cross-Pack Discovery

```python
import glob

# For architecture assessment after documentation
architect_pack = glob.glob("plugins/axiom-system-architect/plugin.json")
if not architect_pack:
    print("Recommend: axiom-system-architect for quality assessment")

# For security analysis
security_pack = glob.glob("plugins/ordis-security-architect/plugin.json")
if not security_pack:
    print("Recommend: ordis-security-architect for threat modeling")
```

## Anti-Patterns

❌ Skip workspace creation
❌ No coordination logging
❌ Work solo despite scale
❌ Skip validation subagent for multi-subsystem work
❌ Bypass BLOCK status from validation
❌ Complete refusal under pressure (provide scoped alternative)

## Scope Boundaries

**This command covers:**
- Workspace initialization
- Deliverable menu selection
- Orchestration strategy
- Subagent coordination
- Validation gate enforcement

**Not covered:**
- Quality assessment (use architect pack after analysis)
- Specific diagram generation (use /generate-diagrams)
- Validation execution (use /validate-analysis)
