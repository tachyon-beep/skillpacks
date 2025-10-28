# File Manifest - What Each File Does

**Created**: 2025-10-28
**Purpose**: Quick reference for understanding file structure

---

## Files in skillpacks/

```
skillpacks/
├── README.md                           ← START HERE (status, quick start, phase tracking)
├── FILE_MANIFEST.md                    ← This file (what each file does)
│
└── planning/
    ├── GETTING_STARTED.md              ← Implementation workflow (critical reminders, how to work)
    ├── QUICK_CHECKLIST.md              ← Keep open while working (RED-GREEN-REFACTOR checklist)
    ├── REAL_WORLD_TEST_SCENARIOS.md    ← Real test cases from Elspeth (ADR-002→005, VULN-004, VULN-009)
    └── 2025-10-28-security-architect-technical-writer-design.md
                                        ↑ Complete design (10,000 words, every skill detailed)
```

---

## Read Files In This Order

### First Time Arriving in New Repo

1. **README.md** (5 min)
   - Current status and phase
   - What the project is
   - Quick start commands
   - Next 4 skills to implement

2. **planning/GETTING_STARTED.md** (15 min)
   - Implementation workflow (RED-GREEN-REFACTOR)
   - Skills to USE during implementation
   - Common mistakes to avoid
   - Testing methodology
   - Phase 1 detailed plan

3. **planning/2025-10-28-security-architect-technical-writer-design.md** (30 min)
   - Complete design for all 16 skills
   - Every skill detailed with examples
   - Cross-referencing strategy
   - Testing approach by skill type
   - Pick up plan for independent work

4. **planning/REAL_WORLD_TEST_SCENARIOS.md** (10 min, read before Phase 1)
   - Real test cases from Elspeth project
   - ADR-002 → 005 cascading discovery pattern
   - VULN-004, VULN-009 concrete scenarios
   - Measurable success criteria

5. **planning/QUICK_CHECKLIST.md** (1 min, print it!)
   - Keep open while working
   - Checkbox workflow per skill
   - Red flags and reminders

---

## File Purposes

### README.md
**Purpose**: Entry point, current status, navigation hub
**Read When**: First arrival, checking status, lost and need orientation
**Key Content**:
- Current phase and next skills
- Quick start for Claude
- Critical reminders (Iron Law, required skills)
- Phase tracking table
- Exit criteria

---

### planning/GETTING_STARTED.md
**Purpose**: Complete implementation guide with critical context
**Read When**: Starting work, forgot workflow, need detailed guidance
**Key Content**:
- Skills to USE (superpowers:writing-skills, etc.)
- Iron Law reminder
- Step-by-step workflow per skill
- Phase 1 hour-by-hour plan
- Common mistakes and testing shortcuts
- Context from Elspeth (inspiration, not dependencies)
- Future vision notes
- Quick reference commands

**Critical Sections**:
- "CRITICAL: Skills to USE During Implementation"
- "Implementation Workflow (Per Skill)"
- "Phase 1 Detailed Plan"
- "Key Principles from Design Discussion"

---

### planning/2025-10-28-security-architect-technical-writer-design.md
**Purpose**: Complete architectural design and specifications
**Read When**: Need details on specific skill, implementing that skill, lost context
**Key Content**:
- Full design overview (16 skills, 2 packs)
- Every skill detailed:
  - When to use
  - Techniques covered
  - Deliverables
  - Example scenarios
  - YAML frontmatter ready to use
- Cross-referencing strategy (bidirectional knowledge graph)
- Testing strategy by skill type
- Implementation phases (4 phases detailed)
- Pick up plan (repository independence)
- Skill summary table

**Use As**: Reference during implementation, source of truth for design decisions

---

### planning/REAL_WORLD_TEST_SCENARIOS.md
**Purpose**: Concrete test cases from real project evolution
**Read When**: Before implementing Phase 1 skills, when creating RED phase tests
**Key Content**:
- ADR-002 → 005 evolution (cascading discovery pattern)
- VULN-004 (configuration override attack)
- VULN-009 (immutability bypass)
- Each scenario includes:
  - What happened in real project
  - What skilled person would have caught
  - Specific test scenario with baseline failures
  - Measurable success criteria

**Critical Sections**:
- "ADR-002 Evolution: The Cascading Discovery Case"
- "Test Scenario Design" for each case
- "Using These Scenarios" (how to apply)

**Use As**: Source of RED phase test scenarios, proof that skills prevent real issues

---

### planning/QUICK_CHECKLIST.md
**Purpose**: Quick visual checklist for active work
**Read When**: While implementing a skill (keep open in second window)
**Key Content**:
- Checkbox workflow (RED → GREEN → REFACTOR → COMMIT)
- Iron Law reminder (visual)
- Red flags (what not to do)
- Phase 1 skill checkboxes
- Emergency help section

**Use As**: Active working reference, print and tape to monitor

---

### FILE_MANIFEST.md (this file)
**Purpose**: Meta-documentation about documentation
**Read When**: Forgot what a file is for, need to understand structure
**Key Content**:
- File tree with descriptions
- Reading order recommendations
- File purpose summaries
- Critical sections per file

---

## Reading Strategy by Scenario

### "I just arrived and need to start working"
```
1. README.md (5 min - get oriented)
2. GETTING_STARTED.md (15 min - learn workflow)
3. DESIGN.md → Phase 1 section (10 min - understand first 4 skills)
4. Print QUICK_CHECKLIST.md
5. Start first skill: security-architect/using-security-architect
```

### "I'm implementing a specific skill"
```
1. DESIGN.md → Find skill section (read full details)
2. GETTING_STARTED.md → "Implementation Workflow" section
3. Open QUICK_CHECKLIST.md in second window
4. Follow RED-GREEN-REFACTOR with TodoWrite
```

### "I forgot the workflow"
```
1. QUICK_CHECKLIST.md (quick visual reminder)
2. GETTING_STARTED.md → "Implementation Workflow" (detailed)
3. Load superpowers:writing-skills if still stuck
```

### "I'm lost and need orientation"
```
1. README.md (where am I? what phase?)
2. This file (what do these files do?)
3. GETTING_STARTED.md (how do I work?)
```

### "I need details on all skills"
```
1. DESIGN.md → "Security-Architect Pack" section
2. DESIGN.md → "Technical-Writer Pack" section
3. DESIGN.md → "Cross-Referencing Strategy" section
```

---

## Critical Information Locations

| Need | File | Section |
|------|------|---------|
| Current phase status | README.md | "Current Phase" |
| Next skills to do | README.md | "Next 4 Skills" |
| How to implement | GETTING_STARTED.md | "Implementation Workflow" |
| Skill details | DESIGN.md | Skill catalog sections |
| Quick checklist | QUICK_CHECKLIST.md | Entire file |
| Testing methodology | GETTING_STARTED.md | "Step 2: RED Phase" |
| Skills to USE | GETTING_STARTED.md | "CRITICAL: Skills to USE" |
| Phase 1 plan | GETTING_STARTED.md | "Phase 1 Detailed Plan" |
| Cross-references | DESIGN.md | "Cross-Referencing Strategy" |
| Exit criteria | README.md | "Phase 1 Exit Criteria" |
| Real test scenarios | REAL_WORLD_TEST_SCENARIOS.md | All sections |
| ADR-002→005 case | REAL_WORLD_TEST_SCENARIOS.md | "ADR-002 Evolution" |
| VULN test cases | REAL_WORLD_TEST_SCENARIOS.md | "VULN-004", "VULN-009" |

---

## Files You'll Create During Implementation

```
security-architect/
├── using-security-architect/
│   └── SKILL.md
├── threat-modeling/
│   └── SKILL.md
├── (more skills as you progress)

technical-writer/
├── using-technical-writer/
│   └── SKILL.md
├── documentation-structure/
│   └── SKILL.md
│   └── (optional: templates/ directory if needed)
├── (more skills as you progress)
```

---

## Document Evolution

As you progress, update:
- ✅ **README.md**: Phase status, completed skills checkboxes
- ✅ **Git commits**: Clear messages per skill
- ✅ **Git tags**: Phase completion (v0.1-phase1, v0.2-phase2, etc.)

**Don't modify**:
- ❌ DESIGN.md (source of truth, read-only)
- ❌ GETTING_STARTED.md (workflow reference, read-only)
- ❌ QUICK_CHECKLIST.md (checklist template, read-only)

---

## Quick Summary

```
README.md                      → Status & navigation (5 min)
GETTING_STARTED.md             → How to work (15 min)
DESIGN.md                      → Complete specs (30 min, reference)
REAL_WORLD_TEST_SCENARIOS.md   → Real test cases (10 min, read before Phase 1)
QUICK_CHECKLIST.md             → Active working aid (1 min, print it)
FILE_MANIFEST.md               → You are here (this file)
```

**Start with README.md. Read REAL_WORLD_TEST_SCENARIOS.md before Phase 1. Follow the reading order. Trust the process.**
