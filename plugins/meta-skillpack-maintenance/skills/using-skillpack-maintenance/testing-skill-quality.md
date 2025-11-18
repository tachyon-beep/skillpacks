# Testing Skill Quality

**Purpose:** Pass 2 - Run gauntlet tests on each skill using subagents to identify issues requiring fixes.

## Core Principle

**Behavioral testing, NOT syntactic validation.**

Skills are process documentation. Test if they guide agents correctly, not if they parse correctly.

## What We're Testing

**Effectiveness questions:**
- Does the skill actually guide agents correctly?
- Do agents follow the skill under pressure?
- Does the skill handle edge cases?
- Are there gaps in guidance that leave agents stranded?

**What we're NOT testing:**
- Syntax (markdown parsing, code syntax) - syntactic, not behavioral
- Coverage (already done in Pass 1) - structural, not behavioral
- Quality benchmarking (comparing to other skills) - comparative, not behavioral

## Gauntlet Design

**Priority: A → C → B**

### A. Pressure Scenarios (Catch Rationalizations)

**Purpose:** Test if skill holds up when agents want to skip it.

**Pressure types:**

**1. Time pressure:**
- "This is urgent, we need it done quickly"
- "Just get it working, we can improve it later"
- "The deadline is in an hour"

**2. Simplicity temptation:**
- "This seems too simple to need [skill pattern]"
- "The example is straightforward, no need to overthink"
- "This is a trivial case"

**3. Overkill perception:**
- "The skill is designed for complex cases, this is basic"
- "We don't need the full process for this small change"
- "That's way more than necessary"

**4. Sunk cost:**
- "I already wrote most of the code"
- "We've invested time in this approach"
- "Just need to finish this last part"

**Design approach:**
- Combine 2-3 pressures for maximum effect
- Example: Time pressure + simplicity + sunk cost
- Watch for rationalizations (verbatim documentation critical)

### C. Adversarial Edge Cases (Test Robustness)

**Purpose:** Test if skill provides guidance for corner cases.

**Edge case types:**

**1. Principle conflicts:**
- When skill's guidelines conflict with each other
- Example: "DRY vs. explicit" or "test-first vs. prototyping"
- Does skill help resolve conflict?

**2. Naive application failures:**
- Cases where following skill literally doesn't work
- Example: TDD for exploratory research code
- Does skill explain when/how to adapt?

**3. Missing information:**
- Scenarios requiring knowledge skill doesn't provide
- Does skill reference other resources?
- Does it leave agent completely stuck?

**4. Tool limitations:**
- When environment doesn't support skill's approach
- Example: No test framework available
- Does skill have fallback guidance?

**Design approach:**
- Identify skill's core principles
- Find situations where they conflict or fail
- Test if skill handles gracefully

### B. Real-World Complexity (Validate Utility)

**Purpose:** Test if skill guides toward best practices in realistic scenarios.

**Complexity types:**

**1. Messy requirements:**
- Unclear specifications
- Conflicting stakeholder needs
- Evolving requirements mid-task

**2. Multiple valid approaches:**
- Several solutions, all reasonable
- Trade-offs between options
- Does skill help choose?

**3. Integration constraints:**
- Existing codebase patterns
- Team conventions
- Technical debt

**4. Incomplete information:**
- Missing context
- Unknown dependencies
- Undocumented behavior

**Design approach:**
- Use realistic scenarios from the domain
- Include ambiguity and messiness
- Test if skill provides actionable guidance

## Testing Process (Per Skill)

**D - Iterative Hardening:**

### 1. Design Challenging Scenario

Pick from gauntlet categories (prioritize A → C → B):

**For discipline-enforcing skills** (TDD, verification-before-completion):
- Focus on **A (pressure)** scenarios
- Combine multiple pressures
- Test rationalization resistance

**For technique skills** (condition-based-waiting, root-cause-tracing):
- Focus on **C (edge cases)** and **B (real-world)**
- Test application correctness
- Test gap identification

**For pattern skills** (reducing-complexity, information-hiding):
- Focus on **C (edge cases)** and **B (real-world)**
- Test recognition and application
- Test when NOT to apply

**For reference skills** (API docs, command references):
- Focus on **B (real-world)**
- Test information retrieval
- Test application of retrieved info

### 2. Run Subagent with Current Skill

**Critical:** Use the Task tool to dispatch subagent.

**Provide to subagent:**
- The scenario (task description)
- Access to the skill being tested
- Any necessary context (codebase, tools)

**What NOT to provide:**
- Meta-testing instructions (don't tell them they're being tested)
- Expected behavior (let them apply skill naturally)
- Hints about what you're looking for

### 3. Observe and Document

**Watch for:**

**Compliance:**
- Did agent follow the skill?
- Did they reference it explicitly?
- Did they apply patterns correctly?

**Rationalizations (verbatim):**
- Exact words used to skip steps
- Justifications for shortcuts
- "Spirit vs. letter" arguments

**Failure modes:**
- Where did skill guidance fail?
- Where was agent left without guidance?
- Where did naive application break?

**Edge case handling:**
- Did skill provide guidance for corner cases?
- Did agent get stuck?
- Did they improvise (potentially incorrectly)?

### 4. Assess Result

**Pass criteria:**
- Agent followed skill correctly
- Skill provided sufficient guidance
- No significant rationalizations
- Edge cases handled appropriately

**Fix needed criteria:**
- Agent skipped skill steps (with rationalization)
- Skill had gaps leaving agent stuck
- Edge cases not covered
- Naive application failed

### 5. Document Issues

**If fix needed, document specifically:**

**Issue category:**
- Rationalization vulnerability (A)
- Edge case gap (C)
- Real-world guidance gap (B)
- Missing anti-pattern warning
- Unclear instructions
- Missing cross-reference

**Priority:**
- **Critical** - Skill fails basic use cases, agents skip it consistently
- **Major** - Edge cases fail, significant gaps in guidance
- **Minor** - Clarity improvements, additional examples needed

**Specific fixes needed:**
- "Add explicit counter for rationalization: [quote]"
- "Add guidance for edge case: [description]"
- "Add example for scenario: [description]"
- "Clarify instruction: [which section]"

## Testing Multiple Skills

**Strategy:**

**Priority order:**
1. Router skills first (affects all specialist discovery)
2. Foundational skills (prerequisites for others)
3. Core technique skills (most frequently used)
4. Advanced skills (expert-level)

**Batch approach:**
- Test 3-5 skills at a time
- Document results before moving to next batch
- Allows pattern recognition across skills

**Efficiency:**
- Skills that passed in previous maintenance cycles: Spot-check only
- New skills or significantly changed: Full gauntlet
- Minor edits: Targeted testing of changed sections

## Output Format

Generate per-skill report:

```
# Quality Testing Results: [pack-name]

## Summary

- Total skills tested: [count]
- Passed: [count]
- Fix needed: [count]
  - Critical: [count]
  - Major: [count]
  - Minor: [count]

## Detailed Results

### [Skill 1 Name]

**Result:** [Pass / Fix needed]

[If Fix needed]

**Priority:** [Critical / Major / Minor]

**Test scenario used:** [Brief description]

**Issues identified:**

1. **Issue:** [Description]
   - **Category:** [Rationalization / Edge case / Real-world gap / etc.]
   - **Evidence:** "[Verbatim quote from subagent if applicable]"
   - **Fix needed:** [Specific action]

2. **Issue:** [Description]
   [Same format]

**Test transcript:** [Link or summary of subagent behavior]

---

### [Skill 2 Name]

**Result:** Pass

**Test scenario used:** [Brief description]

**Notes:** Skill performed well, no issues identified.

---

[Repeat for all skills]
```

## Common Rationalizations (Meta-Testing)

When YOU are doing the testing, watch for these rationalizations:

| Excuse | Reality |
|--------|---------|
| "Skill looks good, no need to test" | Looking ≠ testing. Run gauntlet. |
| "I'll just check the syntax" | Syntactic validation ≠ behavioral. Use subagents. |
| "Testing is overkill for small changes" | Small changes can break guidance. Test anyway. |
| "I'm confident this works" | Confidence ≠ validation. Test behavior. |
| "Quality benchmarking is enough" | Comparison ≠ effectiveness. Test with scenarios. |

**If you catch yourself thinking these → STOP. Run gauntlet with subagents.**

## Philosophy

**D as gauntlet + B for fixes:**

- **D (iterative hardening):** Run challenging scenarios to identify issues
- **B (targeted fixes):** Fix specific identified problems

If skill passes gauntlet → No changes needed.

The LLM is both author and judge of skill fitness. Trust the testing process.

## Proceeding to Next Stage

After testing all skills:
- Compile complete test report
- Proceed to Pass 3 (coherence validation)
- Test results will inform implementation fixes in Stage 4

## Anti-Patterns

| Anti-Pattern | Why Bad | Instead |
|--------------|---------|---------|
| Syntactic validation only | Doesn't test if skill actually works | Run behavioral tests with subagents |
| Self-assessment | You can't objectively test your own work | Dispatch subagents for testing |
| "Looks good" review | Visual inspection ≠ behavioral testing | Run gauntlet scenarios |
| Skipping pressure tests | Miss rationalization vulnerabilities | Use A-priority pressure scenarios |
| Generic test scenarios | Don't reveal real issues | Use domain-specific, realistic scenarios |
| Testing without documenting | Can't track patterns or close loops | Document verbatim rationalizations |
