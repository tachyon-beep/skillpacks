# TDD Methodology for Skill Development

## Overview

This document captures the Test-Driven Development (TDD) methodology used to create the `axiom-system-archaeologist` skillpack. This approach adapts software TDD principles to process documentation.

**Core insight:** Skills are documentation that guide agent behavior. TDD for skills means testing agent behavior WITHOUT the skill (RED), writing the skill to address failures (GREEN), and iteratively improving (REFACTOR).

## Why TDD for Skills?

**The problem:** Skills written without testing often:
- Sound good but don't change behavior
- Miss critical rationalizations agents use
- Include unnecessary content (gold-plating)
- Lack specificity in guidance

**The solution:** RED-GREEN-REFACTOR ensures:
- Skills address actual failure modes (not hypothetical ones)
- Every guideline maps to observed rationalization
- Minimal effective content (no more, no less)
- Validated behavioral change

## The Process

### Phase 1: RED (Baseline Testing)

**Goal:** Document how agents behave WITHOUT the skill present

**Steps:**

1. **Create pressure scenarios** (5-7 scenarios covering common failure modes)
   - Time pressure (deadlines, urgency)
   - Authority pressure (stakeholders, visibility)
   - Sunk cost (incomplete prior work)
   - Complexity overwhelm (large/unfamiliar systems)
   - Validation resistance (bypassing quality gates)
   - Combined pressures (multiple simultaneous)

2. **Run baseline tests** (spawn subagents WITHOUT skill)
   - Give realistic task descriptions with pressure indicators
   - Let subagent complete task naturally
   - Capture verbatim output and reflections

3. **Document failures and rationalizations**
   - What process steps were skipped?
   - What shortcuts were taken?
   - What exact words did agent use to rationalize? (verbatim)
   - What pressures triggered which failures?

4. **Identify patterns across scenarios**
   - Universal failures (appear in all/most scenarios)
   - Context-specific failures (only under certain pressures)
   - Surprising positive behaviors (what already works)
   - Rationalization categories (efficiency, quality, self-validation)

**Artifacts produced:**
- `test-scenarios.md` - Detailed pressure test descriptions
- `baseline-results.md` - Complete failure documentation with verbatim quotes
- Pattern analysis identifying critical skill requirements

**Time investment:** 30-60 minutes per scenario (2-5 hours total for 5 scenarios)

### Phase 2: GREEN (Minimal Skill Writing)

**Goal:** Write skill that addresses ONLY the specific failures found in baseline

**Principles:**

1. **Address specific failures, not hypothetical ones**
   - Every "must" or "mandatory" maps to observed baseline failure
   - Every rationalization table entry comes from verbatim quote
   - No speculative guidance

2. **Use baseline language**
   - Agents said "Time pressure makes trade-offs appropriate" → Skill addresses this exact phrase
   - Agents said "Working solo is faster" → Skill counters this specific rationalization

3. **Make mandatory things mandatory**
   - If agents universally skip X (workspace, validation, logging) → Mark as "NON-NEGOTIABLE"
   - Use strong language ("MANDATORY", "NO EXCEPTIONS") for universal failures
   - Explain WHY (addresses "this feels like overhead" rationalization)

4. **Provide positive alternatives for Extreme pressure**
   - If baseline shows task refusal under extreme pressure
   - Provide scoped alternatives that maintain process
   - Example: "Can't do complete analysis in 1 hour" → "Here's how to do scoped analysis in 1 hour WITH quality gates"

**Structure:**

```markdown
## Mandatory Workflow

### Step N: [Process Step]

[What to do concretely]

**Why this is mandatory:**
[Addresses "feels like overhead" rationalization]

**Common rationalization:** "[Exact quote from baseline]"

**Reality:** [Counter-argument with evidence/reasoning]

## Common Rationalizations (RED FLAGS)

| Excuse | Reality |
|--------|---------|
| "[baseline quote]" | [Why this fails] |
```

**Artifacts produced:**
- `SKILL.md` - Minimal skill addressing baseline failures
- Clear, actionable workflow with mandatory steps
- Rationalization table with baseline quotes

**Time investment:** 1-2 hours for initial skill draft

### Phase 3: GREEN Testing (Validation)

**Goal:** Verify skill changes agent behavior for critical scenarios

**Steps:**

1. **Select critical scenarios** (2-3 most important from baseline)
   - Scenario showing universal failures (time pressure)
   - Scenario showing context-specific failures (incremental work)
   - Optional: Scenario showing surprising behavior (extreme pressure)

2. **Run WITH skill present**
   - Same exact task description as baseline
   - Skill is now available to agent
   - Capture outputs and reflections

3. **Compare baseline vs with-skill**
   - What changed?
   - What improved?
   - What new behaviors emerged?
   - Any unexpected adaptations?

4. **Document GREEN results**
   - Side-by-side comparison table
   - Verdict: PASS/FAIL for each scenario
   - Identify any remaining gaps

**Success criteria:**
- Process steps previously skipped are now followed
- Rationalizations from baseline no longer appear
- Agent explicitly cites skill guidance
- Quality improvements without complete refusal

**Artifacts produced:**
- `green-results.md` - Comparison of baseline vs with-skill
- Identification of remaining gaps for REFACTOR

**Time investment:** 15-30 minutes per critical scenario (30-90 min total)

### Phase 4: REFACTOR (Close Loopholes)

**Goal:** Address any new rationalizations or gaps found in GREEN testing

**Common gaps:**

1. **Acceptable adaptations** (self-validation under time pressure)
   - Document when adaptation is OK vs not OK
   - Provide decision criteria

2. **New rationalizations** (creative bypasses)
   - Add explicit counters to rationalization table
   - Strengthen mandatory language if needed

3. **Ambiguous guidance** (multiple interpretations)
   - Add concrete examples
   - Specify decision criteria more clearly

4. **Missing context** (when to use A vs B)
   - Add decision trees or flowcharts
   - Provide use-case matching

**Iteration:**
- Make changes to skill
- Re-run problem scenarios
- Verify closure
- Repeat until bulletproof

**Artifacts produced:**
- Updated `SKILL.md` with refinements
- Updated `green-results.md` showing improvements

**Time investment:** 30-60 minutes per iteration (1-2 iterations typical)

## Example: axiom-system-archaeologist Development

**RED Phase (2.5 hours):**
- Created 5 pressure scenarios
- Ran all 5 baselines
- Documented 100+ failures and rationalizations
- Identified 4 universal failures (workspace, coordination, validation, orchestration)

**GREEN Phase (2 hours):**
- Wrote 280-line skill addressing all baseline failures
- Tested Scenario 1 (time pressure) - PASSED
- Identified validation adaptation gap

**REFACTOR Phase (30 minutes):**
- Added guidance on when self-validation is acceptable
- Clarified validation gate concept
- Documented decision criteria for validation approach

**Total time:** ~5 hours for complete, validated router skill

## Best Practices

### DO:

✅ Run baselines BEFORE writing skill (iron law)
✅ Capture verbatim rationalizations
✅ Address every universal failure
✅ Use strong language for mandatory steps
✅ Test critical scenarios in GREEN phase
✅ Iterate until no new rationalizations

### DON'T:

❌ Write skill without baseline testing
❌ Add hypothetical guidance not grounded in failures
❌ Use weak language for universal failures
❌ Skip GREEN testing (assume skill works)
❌ Stop after one GREEN test (test multiple pressures)
❌ Ignore acceptable adaptations (document criteria)

## Metrics of Success

**Quantitative:**
- X% reduction in process violations (workspace creation: 0% → 100%)
- Y agent behaviors changed (from baseline to GREEN)
- Z rationalizations eliminated

**Qualitative:**
- Agent explicitly cites skill guidance in reflections
- Agent follows process under pressure (not just absence of pressure)
- Acceptable adaptations are documented and justified
- Users report improved analysis quality

## Scaling to Other Skills

This methodology scales to all skill types:

**Discipline skills** (TDD, verification, code-review)
- Heavy RED testing with pressure combinations
- Strong mandatory language
- Extensive rationalization tables

**Technique skills** (debugging, optimization, refactoring)
- Baseline shows missing steps or wrong approaches
- GREEN shows technique applied correctly
- Examples from baseline failures

**Pattern skills** (architecture, design, modularity)
- Baseline shows pattern misidentification
- GREEN shows correct pattern selection
- Decision criteria from baseline confusion

**Reference skills** (APIs, syntax, commands)
- Baseline shows retrieval failures or incorrect usage
- GREEN shows correct information access
- Gaps from baseline show missing documentation

## Future Work

**For axiom-system-archaeologist:**
- Complete TDD cycle for 4 remaining skills (analyzing-unknown-codebases, generating-architecture-diagrams, documenting-system-architecture, validating-architecture-analysis)
- Each follows same RED-GREEN-REFACTOR process
- Each targets 2-3 hours total development time

**Methodology improvements:**
- Automated baseline comparison (script to diff outputs)
- Pressure taxonomy (standardized pressure types)
- Rationalization library (common patterns across skills)
- Success metrics framework (standardized measurement)

## Conclusion

TDD for skills transforms skill development from "write documentation and hope" to "observe failures, address them, verify fix." The investment (5 hours) produces validated, effective skills that demonstrably change agent behavior under pressure.

**Bottom line:** If you didn't watch agents fail without the skill, you don't know if the skill works.
