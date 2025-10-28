# Getting Started - Implementation Guide

**Created**: 2025-10-28
**Status**: Ready to begin Phase 1
**Current Phase**: Foundation (4 skills)

---

## Quick Start (For Claude in New Repo)

```bash
# You're in ~/skillpacks-repo (independent from Elspeth)
# Read these files in order:
1. README.md (current status)
2. docs/DESIGN.md (complete design, 16 skills detailed)
3. docs/GETTING_STARTED.md (this file - how to implement)
```

---

## CRITICAL: Skills to USE During Implementation

**You are creating skills by USING skills.** This is meta-level skill development.

### Required Skills to Load

**ALWAYS use these skills when implementing:**

1. **`superpowers:using-superpowers`** - Mandatory first skill, establishes workflows
2. **`superpowers:writing-skills`** - Core skill creation methodology (TDD for documentation)
3. **`superpowers:testing-skills-with-subagents`** - How to test skills with pressure scenarios

**Invoke at start of EVERY work session:**
```
I'm using the writing-skills skill to create [skill-name].
```

### The Iron Law (From writing-skills)

```
NO SKILL WITHOUT A FAILING TEST FIRST
```

This applies to:
- ✅ New skills
- ✅ Edits to existing skills
- ✅ "Simple additions"
- ✅ "Documentation updates"

**No exceptions. Ever.**

If you wrote skill content before testing → Delete it. Start over with RED phase.

---

## Implementation Workflow (Per Skill)

### Step 1: Create TodoWrite Checklist

**Use TodoWrite tool to create checklist for EACH skill:**

```
RED Phase - Write Failing Test:
- [ ] Create pressure scenarios (3+ combined pressures for discipline skills)
- [ ] Run scenarios WITHOUT skill - document baseline behavior verbatim
- [ ] Identify patterns in rationalizations/failures

GREEN Phase - Write Minimal Skill:
- [ ] Name uses only letters, numbers, hyphens (no parentheses/special chars)
- [ ] YAML frontmatter with only name and description (max 1024 chars)
- [ ] Description starts with "Use when..." and includes specific triggers/symptoms
- [ ] Description written in third person
- [ ] Keywords throughout for search (errors, symptoms, tools)
- [ ] Clear overview with core principle
- [ ] Address specific baseline failures identified in RED
- [ ] Code inline OR link to separate file
- [ ] One excellent example (not multi-language)
- [ ] Run scenarios WITH skill - verify agents now comply

REFACTOR Phase - Close Loopholes:
- [ ] Identify NEW rationalizations from testing
- [ ] Add explicit counters (if discipline skill)
- [ ] Build rationalization table from all test iterations
- [ ] Create red flags list
- [ ] Re-test until bulletproof

Quality Checks:
- [ ] Small flowchart only if decision non-obvious
- [ ] Quick reference table
- [ ] Common mistakes section
- [ ] No narrative storytelling
- [ ] Supporting files only for tools or heavy reference

Deployment:
- [ ] Commit skill to git
- [ ] Update README.md status
```

### Step 2: RED Phase (Baseline Without Skill)

**For each skill, BEFORE writing anything:**

1. **Create test scenario** based on skill type:
   - **Discipline skill**: Pressure scenario (time + sunk cost + authority)
   - **Technique skill**: Application scenario (use the technique)
   - **Pattern skill**: Recognition scenario (when to apply)
   - **Reference skill**: Retrieval scenario (find and use info)

2. **Run subagent WITHOUT skill**:
   ```
   Use Task tool to dispatch subagent
   Give them scenario WITHOUT loading skill
   Document what they do (verbatim rationalizations)
   ```

3. **Analyze baseline behavior**:
   - What did they miss?
   - What rationalizations did they use?
   - What patterns emerged?

4. **Document failures** → These become the skill content

### Step 3: GREEN Phase (Write Skill)

**Now and ONLY now, write the skill:**

1. **Create directory**: `security-architect/[skill-name]/` or `technical-writer/[skill-name]/`

2. **Write SKILL.md** addressing baseline failures:
   ```markdown
   ---
   name: skill-name-with-hyphens
   description: Use when [specific triggering conditions] - [what skill does]
   ---

   # Skill Name

   ## Overview
   Core principle in 1-2 sentences.

   ## When to Use
   Bullet list with SYMPTOMS

   ## [Core Content]
   Address RED phase failures

   ## Common Mistakes
   From RED phase

   ## Examples
   One excellent example
   ```

3. **Test WITH skill**:
   ```
   Run same scenarios WITH skill loaded
   Verify agent now complies
   Document any remaining issues
   ```

### Step 4: REFACTOR Phase (Close Loopholes)

1. **Add pressure** (if discipline skill):
   - Combined pressures (time + sunk cost)
   - Authority pressure ("PM says skip this")
   - Exhaustion pressure (end of long session)

2. **Find new rationalizations**:
   - Run harder scenarios
   - Document new excuses
   - Add explicit counters

3. **Build rationalization table**:
   ```markdown
   | Excuse | Reality |
   |--------|---------|
   | "Too simple to test" | Tests take 30 seconds |
   ```

4. **Re-test until bulletproof**

### Step 5: Commit

```bash
git add security-architect/[skill-name]/
git commit -m "Implement security-architect/[skill-name] (Phase N)

- RED: Baseline testing documented [brief summary]
- GREEN: Skill addresses [key failures]
- REFACTOR: Tested against [pressure types]
"

# Update README.md
# Mark skill complete in phase tracking
git add README.md
git commit -m "Update status: [skill-name] complete"
```

---

## Phase 1 Detailed Plan

See `docs/DESIGN.md` section "Pick Up Plan > Phase 1 Work Plan" for:
- Hour-by-hour breakdown for each skill
- Specific test scenarios
- Expected effort (2-4 hours per skill)
- Exit criteria

**Phase 1 Skills**:
1. `security-architect/using-security-architect` (meta, 2-3 hrs)
2. `security-architect/threat-modeling` (core technique, 3-4 hrs)
3. `technical-writer/using-technical-writer` (meta, 2-3 hrs)
4. `technical-writer/documentation-structure` (core reference, 3-4 hrs)

**Phase 1 Exit Criteria**:
- ✅ All 4 skills pass RED-GREEN-REFACTOR
- ✅ Meta-skills route correctly
- ✅ Cross-references work
- ✅ Personal use validated
- ✅ Committed to git

---

## Key Principles from Design Discussion

### 1. Layered Architecture
- **Core skills** = Universal (any project)
- **Extension skills** = Specialized (high-security/regulated)
- **Meta-skills** = Routing/discovery

### 2. Universal Core with High-Security Extensions
- Base skills work anywhere
- Extensions add depth for classified/regulated contexts
- Clear separation enables gradual adoption

### 3. Bidirectional Cross-References
- Security-architect ↔ Technical-writer
- Creates knowledge graph
- No hard dependencies (skills work standalone)

### 4. Meta-Awareness Pattern
Example: `compliance-awareness-and-mapping` teaches:
- "Frameworks vary by jurisdiction/industry"
- ALWAYS ask "What frameworks apply HERE?"
- Universal pattern (discover → map → trace) not exhaustive lists

### 5. Cross-Cutting Ownership
- Each pack owns its perspective
- Security architects: How to document threats
- Technical writers: How to handle sensitive material
- Both reference each other

---

## Common Mistakes to Avoid

### ❌ Writing Skill Before Testing
- **Wrong**: Write SKILL.md → Test it
- **Right**: RED (baseline) → GREEN (write) → REFACTOR (harden)

### ❌ Batching Multiple Skills
- **Wrong**: Write 3 skills → Test all 3
- **Right**: Test skill 1 → Write skill 1 → Test skill 1 → Commit → Next skill

### ❌ Skipping Pressure Testing
- **Wrong**: "Academic review is enough"
- **Right**: Test under time pressure, sunk cost, authority override

### ❌ Generic Examples
- **Wrong**: "Configure the system appropriately"
- **Right**: "Set `API_TIMEOUT=30` in `.env`"

### ❌ Forgetting TodoWrite
- **Wrong**: Work through checklist mentally
- **Right**: TodoWrite for EACH skill's RED-GREEN-REFACTOR checklist

---

## Testing Shortcuts (Don't Use These)

| Excuse | Why It Fails |
|--------|--------------|
| "Skill is obviously clear" | Clear to you ≠ clear to agents under pressure |
| "It's just a reference" | References have gaps, test retrieval |
| "Testing is overkill" | 15 min testing saves hours debugging |
| "I'll test if problems emerge" | Problems = agents can't use skill in production |
| "Academic review is enough" | Reading ≠ using under pressure |

**Reality**: Every untested skill has issues. Always. Test first.

---

## Context from Origin Project (Elspeth)

These skill packs were designed alongside **Elspeth** (security-first LLM orchestration platform). Key patterns that informed design:

### Security Patterns from Elspeth
- **Bell-LaPadula MLS**: No read up, no write down (influenced `classified-systems-security`)
- **Fail-fast security**: Validate at construction, not runtime (influenced `secure-by-design-patterns`)
- **Defense-in-depth**: Three-layer validation (influenced `security-controls-design`)
- **Immutability enforcement**: Frozen security properties (influenced `classified-systems-security`)

### Documentation Patterns from Elspeth
- **ADR quality**: 14 ADRs, release-quality (influenced `documentation-structure`)
- **Compliance documentation**: ISM/IRAP, SOC2 (influenced `compliance-awareness-and-mapping`)
- **ATO preparation**: SSP/SAR/POA&M (influenced `operational-acceptance-documentation`)

### Testing Patterns from Elspeth
- **TDD for refactoring**: RED-GREEN-REFACTOR for complexity reduction (directly influenced writing-skills usage)
- **Mutation testing**: Bulletproof test suites (influenced REFACTOR phase approach)

**Note**: These are INSPIRATIONS, not dependencies. Skills are universal and work for any project.

---

## Future Vision (Not Current Scope)

### Web of Complementary Skill Packs

**Vision**: Create ecosystem of skill packs that enhance each other:
- security-architect + technical-writer (current project)
- performance-engineer + reliability-engineer (future)
- data-architect + privacy-engineer (future)
- compliance-specialist + audit-readiness (future)

**Pattern**: Each pack is universal core + specialized extensions, with bidirectional cross-references to related packs.

**Not in scope for current project** - but design enables this future.

---

## Quick Reference Commands

### Starting Work Session
```bash
# In new Claude session
cd ~/skillpacks-repo
cat README.md              # Check status
cat docs/DESIGN.md         # Full context
cat docs/GETTING_STARTED.md  # This file

# Announce usage
"I'm using the writing-skills skill to create [skill-name]."
```

### During Implementation
```bash
# Create skill directory
mkdir -p security-architect/threat-modeling
cd security-architect/threat-modeling

# After RED-GREEN-REFACTOR complete
git add .
git commit -m "Implement security-architect/threat-modeling (Phase 1)"

# Update status
vim ../README.md  # Mark skill complete
git add ../README.md
git commit -m "Update status: threat-modeling complete"
```

### Phase Completion
```bash
# All skills in phase done
git tag v0.1-phase1
git push origin main --tags

# Update README for next phase
vim README.md  # "Phase 2 In Progress"
```

---

## Current Status (At Handoff)

**Date**: 2025-10-28
**Phase**: Ready to begin Phase 1
**Completed**: Design document, getting started guide
**Next Action**: Move to independent repo, implement first skill

**Todo State from Design Session**:
All brainstorming phases complete:
- ✅ Phase 1: Understanding (requirements gathered)
- ✅ Phase 2: Exploration (layered architecture chosen)
- ✅ Phase 3: Design Presentation (all sections validated)
- ✅ Phase 4: Design Documentation (design written)
- ✅ Phase 5: Worktree Setup (not needed - orthogonal project)
- ✅ Phase 6: Planning Handoff (complete)

**Ready for**: Implementation Phase 1 (4 foundation skills)

---

## Success Mantra

```
RED (watch agent fail without skill)
  ↓
GREEN (write skill addressing failures)
  ↓
REFACTOR (close loopholes under pressure)
  ↓
COMMIT (one skill at a time)
```

**The Iron Law**: NO SKILL WITHOUT FAILING TEST FIRST

---

## Need Help?

1. **Stuck on testing?** → Review `superpowers:testing-skills-with-subagents`
2. **Unsure about structure?** → Review `docs/DESIGN.md` skill details
3. **Forgot workflow?** → Review this file's "Implementation Workflow"
4. **Lost context?** → Read README.md → DESIGN.md → This file

---

**You have everything you need to work independently. Trust the process. Follow RED-GREEN-REFACTOR. Ship one skill at a time.**

Good luck! 🚀
