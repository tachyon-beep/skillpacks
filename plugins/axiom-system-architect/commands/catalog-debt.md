---
description: Catalog technical debt with execution discipline - deliver first, explain after
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write"]
argument-hint: "[assessment_file_or_directory]"
---

# Catalog Debt Command

You are creating a technical debt catalog from architecture assessment findings. Execute with delivery discipline.

## Core Principle

**Delivered partial catalog > perfect undelivered catalog.**

Produce the document. Don't explain why you're producing it.

## The Iron Law

**Time allocation priority:**
1. **Deliver** (80% of time)
2. **Decide** (10% of time)
3. **Explain** (10% of time, after delivery)

**NOT:**
1. ~~Explain~~ (50% of time)
2. ~~Analyze trade-offs~~ (30% of time)
3. ~~Deliver~~ (20% of time, if at all)

## Catalog Entry Requirements

### Minimum Viable Entry (REQUIRED for all items)

**Every cataloged item MUST have:**
- **Name** - Clear, specific title
- **Evidence** - At least 1 file path showing the issue
- **Impact** - 1 sentence business + technical consequence
- **Effort** - T-shirt size (S/M/L/XL) or day estimate
- **Category** - Security, Architecture, Code Quality, or Performance

**Time per entry:** 3-5 minutes

### Enhanced Entry (If Time Allows)

**Nice-to-have additions:**
- Multiple evidence citations
- Code examples
- Specific recommendations
- Dependencies between items
- ROI calculations

**Time per entry:** 10-15 minutes

**Under time pressure: Minimum for ALL critical/high > Enhanced for SOME**

## Output Format (Contract)

```markdown
# Technical Debt Catalog

**Coverage:** [X of Y items analyzed]
**Priority Focus:** Critical + High priority items
**Status:** [Z] medium/low priority items identified, detailed analysis pending
**Complete Catalog Delivery:** [Date if partial]
**Confidence:** HIGH for items below, MEDIUM for pending analysis

## Critical Priority (Immediate Action Required)

### [Item Name]
**Evidence:** `path/to/file.py:line`, `path/to/other.js:line`
**Impact:** [Business + Technical consequences]
**Effort:** [S/M/L/XL or days]
**Category:** Security / Architecture / Code Quality / Performance
**Details:** [Specific description with examples]

[Repeat for all critical items]

## High Priority (Next Quarter)

[Same structure for high priority items]

## Medium Priority

[Same structure or abbreviated entries]

## Low Priority

[Brief list if not fully analyzed]

## Pending Analysis

**Medium Priority:** [X items identified]
- [Item name]: [1-sentence description]

**Low Priority:** [Y items identified]
- [Listed but not analyzed]

## Limitations

This catalog analyzes [X] of [Y] identified technical debt items.

**Not included in this analysis:**
- [What was excluded and why]

**Complete catalog delivery:** [Date]
```

## Scoping Under Time Pressure

### The Three Options

**Option A: Complete analysis, miss deadline**
- Choose when: Deadline is negotiable, completeness is critical
- Never choose when: Stakeholder has hard deadline

**Option B: Partial analysis with limitations (RECOMMENDED)**
- Document critical/high priority items fully
- Note medium/low items identified but not analyzed
- Explicit limitations section
- Delivery date for complete catalog

**Option C: Quick list without proper analysis**
- Choose when: NEVER. This damages credibility.

### Time-Boxing Pattern (90 minutes)

| Time | Activity | Output |
|------|----------|--------|
| 0-5 min | Decide scope (A/B/C) | Option B: Critical + High |
| 5-15 min | Document structure | Template with sections |
| 15-75 min | Catalog items | 10-12 items @ 5 min each |
| 75-85 min | Limitations section | Explicit scope |
| 85-90 min | Executive summary | Stakeholder-ready intro |

## Evidence Requirements

**Every item needs evidence.**

❌ Bad:
```markdown
### Authentication Issues
The auth system has problems.
**Effort:** Large
```

✅ Good (Minimum):
```markdown
### Weak Password Hashing (MD5)
**Evidence:** `src/auth/password.py:23` - uses MD5, should be bcrypt
**Impact:** Password database breach exposes user passwords immediately
**Effort:** M (2-3 days to migrate + test)
**Category:** Security
```

## Red Flags - STOP

If you catch yourself:
- Writing paragraphs about what you "would" do
- Explaining trade-offs before producing document
- Analyzing hypothetical scenarios
- Describing your methodology

**STOP. Start writing the actual catalog.**

## Rationalization Blockers

| Excuse | Reality |
|--------|---------|
| "Stakeholder needs my reasoning" | Stakeholder needs the catalog. Reasoning comes after. |
| "Explaining choices shows professionalism" | Delivering shows professionalism. |
| "Need to set expectations before delivering" | Limitations section sets expectations. Write it in the document. |
| "Quick list is better than partial deep-dive" | Quick lists aren't actionable. Partial proper analysis is. |

## Cross-Pack Discovery

```python
import glob

# For improvement prioritization after cataloging
priority_cmd = glob.glob("plugins/axiom-system-architect/commands/prioritize-improvements.md")
if priority_cmd:
    print("Next step: /prioritize-improvements to create roadmap")
```

## Output Location

Write to `docs/arch-analysis-*/06-technical-debt-catalog.md`

## Scope Boundaries

**This command covers:**
- Technical debt cataloging
- Evidence-based documentation
- Priority classification
- Time-boxed execution

**Not covered:**
- Architecture assessment (use /assess-architecture)
- Improvement prioritization (use /prioritize-improvements)
- Codebase exploration (use axiom-system-archaeologist)
