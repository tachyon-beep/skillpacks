
# Assessing Code Quality

## Purpose

Analyze code quality indicators beyond architecture with EVIDENCE-BASED findings - produces quality scorecard with concrete examples, file/line references, and actionable recommendations.

## When to Use

- Coordinator delegates quality assessment after subsystem catalog completion
- Need to identify specific, actionable quality issues beyond architectural patterns
- Output feeds into architect handover or improvement planning

## Core Principle: Evidence Over Speculation

**Good quality analysis provides concrete evidence. Poor quality analysis makes educated guesses.**

```
❌ BAD:  "Payment service likely lacks error handling"
✅ GOOD: "Payment service process_payment() (line 127) raises generic Exception, loses error context"

❌ BAD:  "Functions may be too long"
✅ GOOD: "src/api/orders.py:process_order() is 145 lines (lines 89-234)"

❌ BAD:  "Error handling unclear"
✅ GOOD: "3 different error patterns: orders.py uses exceptions, payment.py returns codes, user.py mixes both"
```

**Your goal:** Document WHAT YOU OBSERVED, not what you guess might be true.

## Implementation Review Requirement (MANDATORY)

**YOU MUST READ IMPLEMENTATION CODE to assess code quality.**

**Banned shortcuts:**
- ❌ Producing findings based solely on `wc -l` (file/function size)
- ❌ Using `grep -c` to count patterns without reading context
- ❌ Citing cyclomatic complexity tools without reviewing actual code
- ❌ Inferring quality from file structure or import statements
- ❌ Using "minimal viable quality assessment" as justification

**If you haven't opened and read the actual source files:**
- DO NOT produce quality findings
- DO write: "Implementation not reviewed - cannot assess code quality"
- DO recommend: "Quality assessment requires implementation review"

**Verification check:**
- Can you quote specific code snippets from the files in your findings?
- Can you cite line numbers where you observed issues?
- If NO to either: You haven't done quality assessment, you've done structural analysis.

**From pressure testing:** Agents will use grep/wc/file-structure to fake quality analysis. This is BANNED.

## The Speculation Trap (from baseline testing)

**Common rationalization:** "I don't have full context, so I'll make educated guesses to provide value."

**Reality:** Speculation masquerading as analysis is worse than saying "insufficient information."

**Baseline failure mode:**
- Document says "Error handling likely mixes patterns"
- Based on: Filename alone, not actual code review
- Should say: "Reviewed file structure only - implementation details not analyzed"

## Weasel Words - BANNED

If you catch yourself writing these, STOP:

| Banned Phrase | Why Banned | Replace With |
|---------------|------------|--------------|
| "Likely" | Speculation | "Observed in file.py:line" OR "Not analyzed" |
| "May" | Hedge | "Found X" OR "Did not review X" |
| "Unclear if" | Evasion | "Code shows X" OR "Insufficient information to assess X" |
| "Appears to" | Guessing | "Lines 12-45 implement X" OR "Not examined" |
| "Probably" | Assumption | Concrete observation OR admit limitation |
| "Should" | Inference | "Currently does X" (observation) |
| "Suggests" | Reading between lines | Quote actual code OR acknowledge gap |

**From baseline testing:** Agents will use these to fill gaps. Skill must ban them explicitly.

## Evidence Requirements

**Every finding MUST include:**

1. **File path** - Exact location
   - ✅ `src/services/payment.py`
   - ❌ "payment service"

2. **Line numbers** - Specific range
   - ✅ "Lines 127-189 (63 lines)"
   - ❌ "Long function"

3. **Code example or description** - What you saw
   - ✅ "Function has 8 nested if statements"
   - ✅ "Quoted: `except Exception as e: pass`"
   - ❌ "Complex logic"

4. **Severity with reasoning** - Why this level
   - ✅ "Critical: Swallows exceptions in payment processing"
   - ❌ "High priority issue"

**If you didn't examine implementation:**
- ✅ "Reviewed imports and structure only - implementation not analyzed"
- ❌ "Function likely does X"

## Severity Framework (REQUIRED)

**Define criteria BEFORE rating anything:**

**Critical:**
- Security vulnerability (injection, exposed secrets, auth bypass)
- Data loss/corruption risk
- Blocks deployment
- Example: Hardcoded credentials, SQL injection, unhandled exceptions in critical path

**High:**
- Frequent source of bugs
- High maintenance burden
- Performance degradation
- Example: 200-line functions, 15% code duplication, N+1 queries

**Medium:**
- Moderate maintainability concern
- Technical debt accumulation
- Example: Missing docstrings, inconsistent error handling, magic numbers

**Low:**
- Minor improvement
- Style/cosmetic
- Example: Verbose naming, minor duplication (< 5%)

**From baseline testing:** Agents will use severity labels without defining them. Enforce framework first.

## Observation vs. Inference

**Distinguish clearly:**

**Observation** (what you saw):
- "Function process_order() is 145 lines"
- "3 files contain identical validation logic (lines quoted)"
- "No docstrings found in src/api/ (0/12 functions)"

**Inference** (what it might mean):
- "145-line function suggests complexity - recommend review"
- "Identical validation in 3 files indicates duplication - recommend extraction"
- "Missing docstrings may hinder onboarding - recommend adding"

**Always lead with observation, inference optional:**
```markdown
**Observed:** `src/api/orders.py:process_order()` is 145 lines (lines 89-234) with 12 decision points
**Assessment:** High complexity - recommend splitting into smaller functions
```

## "Insufficient Information" is Valid

**When you haven't examined something, say so:**

✅ **Honest limitation:**
```markdown
## Testing Coverage

**Not analyzed** - Test files not reviewed during subsystem cataloging
**Recommendation:** Review test/ directory to assess coverage
```

❌ **Speculation:**
```markdown
## Testing Coverage

Testing likely exists but coverage unclear. May have integration tests. Probably needs more unit tests.
```

**From baseline testing:** Agents filled every section with speculation. Skill must make "not analyzed" acceptable.

## When You Haven't Read Implementation Code

**If coordinator requests quality assessment but you've only reviewed structure:**

**DO:**
1. State clearly: "Cannot assess code quality - implementation not reviewed"
2. Explain: "Quality assessment requires reading source code, not just file structure"
3. Offer alternative: "Can provide architectural analysis or recommend quality review scope"

**DO NOT:**
1. Produce "findings" based on file sizes, function counts, or import analysis
2. Call structural metrics "quality observations"
3. Use time pressure as justification for structural-only "assessment"

**Example response:**
```
I've reviewed the subsystem catalog (file structure and imports only). Code quality assessment requires reading implementation code - I haven't done this yet.

**Options:**
1. I can read implementation files now and provide quality assessment (estimate: 2 hours)
2. Coordinator can assign this to a specialist subagent
3. We can defer quality assessment to post-delivery phase

**Not viable:** Producing quality findings without reading code.
```

**Structural metrics (line counts, function counts) are NOT quality findings.**

**From pressure testing:** "Minimal viable" became "structural metrics masquerading as quality analysis." This is banned.

## Output Contract

Write to `05-quality-assessment.md`:

```markdown
# Code Quality Assessment

**Analysis Date:** YYYY-MM-DD
**Scope:** [What you reviewed - be specific]
**Methodology:** [How you analyzed - tools, sampling strategy]
**Confidence:** [High/Medium/Low with reasoning]

## Severity Framework

[Define Critical/High/Medium/Low with examples]

## Findings

### [Category: Complexity/Duplication/etc.]

**Observed:**
- [File path:line numbers] [Concrete observation]
- [Quote code or describe specifically]

**Severity:** [Level] - [Reasoning based on framework]

**Recommendation:** [Specific action]

### [Repeat for each finding]

## What Was NOT Analyzed

- [Explicitly list gaps]
- [No speculation about these areas]

## Recommendations

[Prioritized by severity]

## Limitations

- Sample size: [N files from M subsystems]
- Methodology constraints: [No tools, time pressure, etc.]
- Confidence assessment: [Why Medium/Low]
```

## Success Criteria

**You succeeded when:**
- Every finding has file:line evidence
- No weasel words ("likely", "may", "unclear if")
- Severity framework defined before use
- Observations distinguished from inferences
- Gaps acknowledged as "not analyzed"
- Output follows contract

**You failed when:**
- Speculation instead of evidence
- Vague findings ("functions may be long")
- Undefined severity ratings
- "Unclear if" everywhere
- Professional-looking guesswork

## Common Rationalizations (STOP SIGNALS)

| Excuse | Reality |
|--------|---------|
| "I'll provide value with educated guesses" | Speculation is not analysis. Be honest about gaps. |
| "I can infer from file names" | File names ≠ implementation. Acknowledge limitation. |
| "Stakeholders expect complete analysis" | They expect accurate analysis. Partial truth > full speculation. |
| "I reviewed it during cataloging" | Catalog focus ≠ quality focus. If you didn't examine for quality, say so. |
| "Professional reports don't say 'I don't know'" | Professional reports distinguish facts from limitations. |

## Integration with Workflow

Typically invoked after subsystem catalog completion, before or alongside diagram generation. Feeds findings into final report and architect handover.

