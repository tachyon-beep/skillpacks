---
description: Review documentation for clarity, structure, completeness, and audience fit
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[documentation_file_or_path]"
---

# Review Documentation Command

You are reviewing technical documentation for clarity, structure, and completeness.

## Core Principle

**Good writing = easy scanning + clear actions + adapted to reader's context.**

Structure determines findability. If readers can't find it, you haven't documented it.

## Review Framework

### 1. Structure Assessment

**Check Document Type Completeness**:

| Type | Required Sections |
|------|-------------------|
| **ADR** | Context, Decision, Alternatives, Consequences, Status |
| **API** | Auth, Rate Limiting, Pagination, Endpoints, Errors |
| **Runbook** | Prerequisites, Safety, Procedure, Verification, Rollback |
| **README** | Description, Installation, Usage Example |

**Structure Quality**:
- [ ] Clear heading hierarchy
- [ ] Logical section order
- [ ] Table of contents (if >500 words)
- [ ] Cross-references where needed

### 2. Clarity Assessment

**Active Voice Check**:
- [ ] Actions use active voice ("Run X" not "X should be run")
- [ ] Subject performs action directly
- [ ] WHO does WHAT is clear

**Concrete Examples Check**:
- [ ] Every instruction has runnable example
- [ ] Commands are copy-pasteable
- [ ] Config examples show actual values
- [ ] Expected output shown

**Scannability Check**:
- [ ] Short paragraphs (3-5 sentences max)
- [ ] Bullet points for lists
- [ ] Tables for comparisons
- [ ] Code blocks for commands
- [ ] Bold used sparingly for emphasis

### 3. Audience Assessment

**Audience Identification**:
- Who is the primary reader?
- What do they need? (HOW vs WHY vs WHAT)

**Audience Fit**:

| Audience | Needs | Style |
|----------|-------|-------|
| **Developers** | HOW it works (APIs, code, architecture) | Technical precision, code examples |
| **Operators** | HOW to run it (config, monitoring, troubleshooting) | Step-by-step, commands, verification |
| **Executives** | WHY it matters (impact, cost, risk) | High-level, business terms |

- [ ] Content matches audience needs
- [ ] Jargon appropriate for audience
- [ ] Detail level appropriate

### 4. Completeness Assessment

**Information Completeness**:
- [ ] All required sections present
- [ ] No "TODO" or placeholder text
- [ ] Links work (not broken)
- [ ] Examples are current (not deprecated)

**Action Completeness**:
- [ ] Reader knows what to do next
- [ ] Prerequisites are listed
- [ ] Success criteria defined
- [ ] Error handling documented

## Review Process

### Step 1: Identify Document Type

Determine: ADR, API docs, runbook, README, or other

### Step 2: Check Structure

Apply type-specific completeness checklist

### Step 3: Check Clarity

Apply active voice, concrete examples, scannability checks

### Step 4: Check Audience Fit

Verify content matches audience needs

### Step 5: Identify Issues

Categorize by severity:
- **Critical**: Missing required content, incorrect information
- **Major**: Clarity issues, structure problems
- **Minor**: Style improvements, polish

## Output Format

```markdown
## Documentation Review: [Document Name]

### Summary

**Document Type**: [ADR/API/Runbook/README/Other]
**Target Audience**: [Developers/Operators/Executives/Mixed]
**Overall Quality**: [Strong/Acceptable/Needs Work]

### Structure Assessment

**Completeness**: [X/Y required sections present]

| Required Section | Status | Notes |
|-----------------|--------|-------|
| [Section 1] | Present/Missing | [Details] |
| [Section 2] | Present/Missing | [Details] |

**Structure Issues**:
- [Issue description and location]

### Clarity Assessment

| Check | Status | Examples |
|-------|--------|----------|
| Active voice | Pass/Fail | [Quote passive sentence if fail] |
| Concrete examples | Pass/Fail | [What's missing] |
| Scannability | Pass/Fail | [Wall of text location] |

**Clarity Issues**:
- [Issue description and location]

### Audience Assessment

**Identified Audience**: [Who this is written for]
**Audience Fit**: [Good/Partial/Poor]

**Issues**:
- [Too technical for executives at line X]
- [Missing code examples for developers]

### Prioritized Recommendations

**Critical** (Fix Before Publishing):
1. [Issue + specific action]

**Major** (Fix Soon):
1. [Issue + specific action]

**Minor** (Polish):
1. [Issue + specific action]

### Specific Fixes

#### Issue 1: [Title]

**Location**: [File:Line or Section]
**Problem**: [What's wrong]
**Fix**: [Specific change]

**Before**:
```
[Current text]
```

**After**:
```
[Improved text]
```
```

## Common Issues Quick Reference

| Issue | Category | Fix |
|-------|----------|-----|
| Passive voice | Clarity | "X should be run" → "Run X" |
| Abstract instruction | Clarity | "Configure appropriately" → "Set `API_KEY=abc123`" |
| Wall of text | Scannability | Break into bullets, add headings |
| Missing prerequisites | Completeness | Add "Prerequisites" section |
| Broken links | Completeness | Update or remove |
| Wrong audience level | Audience | Add/remove technical detail |
| No usage example | Completeness | Add concrete example |
| Missing error handling | Completeness | Document error cases |

## Scope Boundaries

**This command covers:**
- Documentation structure review
- Clarity assessment
- Audience fit evaluation
- Improvement recommendations

**Not covered:**
- Technical accuracy verification (requires domain knowledge)
- Writing new documentation (use /write-docs)
- Security documentation review (use ordis-security-architect)
