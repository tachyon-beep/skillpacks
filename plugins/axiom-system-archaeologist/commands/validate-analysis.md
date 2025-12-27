---
description: Run validation gate on architecture analysis documents against output contracts
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write"]
argument-hint: "[document_to_validate]"
---

# Validate Analysis Command

You are running a validation gate on architecture analysis documents. Validation ensures contract compliance and cross-document consistency.

## Core Principle

**Validation catches errors before they cascade. 5-10 minutes validating saves hours debugging.**

Validation is NOT optional. It's a mandatory quality gate.

## When Validation is Required

**MUST spawn validation subagent when:**
- Multi-subsystem work (≥3 subsystems)
- Multiple hours of analysis work
- Multiple subagents contributed to document
- Any doubt about completeness

**Self-validation ONLY when ALL conditions met:**
1. Single-subsystem analysis (1-2 only)
2. Total analysis time < 30 minutes
3. You personally did ALL work (no subagents)
4. You document EVIDENCE (not just checkmarks)

## Validation Checklist

### For Subsystem Catalog (`02-subsystem-catalog.md`)

**Contract Compliance:**
```
[ ] Each entry has H2 heading (## Subsystem Name)
[ ] Location field with backticks and path
[ ] Responsibility as single sentence
[ ] Key Components as bulleted list with descriptions
[ ] Dependencies with "Inbound:" and "Outbound:" labels
[ ] Patterns Observed as bulleted list
[ ] Concerns section present (issues OR "None observed")
[ ] Confidence level with reasoning AND evidence
[ ] No extra sections added
[ ] Sections in correct order
```

**Cross-Document Consistency:**
```
[ ] All subsystems in discovery also in catalog
[ ] Dependency claims are bidirectional
    (If A depends on B, B shows A as inbound)
[ ] No placeholder text ([TODO], [TBD], [Fill in])
[ ] Evidence cited (file paths, line numbers)
```

### For Diagrams (`03-diagrams.md`)

**Contract Compliance:**
```
[ ] Context diagram shows external actors
[ ] Container diagram shows technology choices
[ ] Component diagrams for significant containers
[ ] Each diagram has description
[ ] Limitations section present
```

**Cross-Document Consistency:**
```
[ ] All catalog subsystems appear in diagrams
[ ] Dependencies in diagrams match catalog
[ ] Container names consistent across levels
[ ] No orphan components (not in catalog)
```

### For Final Report (`04-final-report.md`)

**Contract Compliance:**
```
[ ] Executive summary present
[ ] Subsystem overview section
[ ] Architectural patterns section
[ ] Concerns/recommendations section
[ ] Confidence assessment
```

**Cross-Document Consistency:**
```
[ ] Statistics match catalog counts
[ ] Patterns summarize catalog findings
[ ] Concerns aggregate from individual entries
[ ] Diagrams referenced correctly
```

## Validation Process

### Step 1: Identify Document to Validate

Read the document and determine type:
- Subsystem catalog
- Diagrams
- Final report
- Quality assessment
- Handover report

### Step 2: Load Relevant Contract

Each document type has specific requirements. Apply correct checklist.

### Step 3: Execute Validation

For each check:
1. Verify requirement is met
2. Note specific evidence (line numbers, sections)
3. Mark PASS, WARNING, or FAIL

### Step 4: Cross-Reference Other Documents

Check consistency with:
- `01-discovery-findings.md`
- `02-subsystem-catalog.md`
- `03-diagrams.md`

### Step 5: Produce Validation Report

## Validation Status Meanings

| Status | Meaning | Action |
|--------|---------|--------|
| **APPROVED** | All checks pass | Proceed to next phase |
| **NEEDS_REVISION** (warnings) | Non-critical issues | Fix or document as limitations, proceed |
| **NEEDS_REVISION** (critical) | Blocking issues | STOP. Fix issues. Re-validate. |

**Maximum 2 retries before escalation to user.**

## Validation Report Format

Write to `temp/validation-[document-name].md`:

```markdown
# Validation Report: [Document Name]

**Validated:** [timestamp]
**Document:** [path]
**Validator:** [self/subagent]

## Contract Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| H2 headings | PASS | Lines 5, 45, 89 |
| Location field | PASS | All 6 entries |
| Confidence levels | WARNING | Entry 3 missing evidence |

## Cross-Document Consistency

| Check | Status | Evidence |
|-------|--------|----------|
| Discovery ↔ Catalog | PASS | All 6 subsystems present |
| Bidirectional deps | FAIL | Subsystem A→B, but B missing A inbound |

## Issues Found

### Critical (BLOCKING)
1. **Bidirectional dependency mismatch**
   - Location: Subsystem B, Dependencies section
   - Expected: "Inbound: Subsystem A"
   - Found: No mention of Subsystem A
   - Fix: Add inbound dependency

### Warnings (Non-blocking)
1. **Missing evidence in confidence**
   - Location: Subsystem C, Confidence section
   - Issue: Says "High" but no files cited
   - Fix: Add specific files read

## Validation Result

**Status:** NEEDS_REVISION (1 critical, 1 warning)

**Required Actions:**
1. Fix bidirectional dependency in Subsystem B
2. Add evidence to Subsystem C confidence

**After fixes:** Re-run validation
```

## Common Validation Failures

| Failure | Cause | Fix |
|---------|-------|-----|
| Extra sections | "Helpful" additions | Remove; follow contract exactly |
| Missing Concerns | Skipped section | Add "None observed" |
| No evidence | Confidence without citations | Cite specific files/lines |
| Orphan in diagram | Component not in catalog | Add to catalog or remove |
| One-way dependency | Forgot inbound side | Add bidirectional reference |
| Placeholder text | Rushed completion | Replace with actual content |

## Self-Validation Evidence Format

When self-validating (rare, small work only):

```markdown
## Validation Decision
- Approach: Self-validation
- Justification: 1-2 subsystems, <30 min work, solo

**Evidence (REQUIRED):**
- Contract sections verified:
  - Location ✓ (line 5)
  - Responsibility ✓ (line 7)
  - Key Components ✓ (lines 9-12)
  - Dependencies ✓ (lines 14-16)
  - Patterns ✓ (lines 18-20)
  - Concerns ✓ (line 22)
  - Confidence ✓ (line 24)
- Consistency checks:
  - Subsystem X matches discovery ✓
  - Dependencies bidirectional ✓
- Issues found and resolved: None

**Result:** APPROVED with evidence above
```

**Self-validation WITHOUT evidence is NOT validation.**

## Scope Boundaries

**This command covers:**
- Contract compliance checking
- Cross-document consistency
- Validation report generation
- Status determination (APPROVED/NEEDS_REVISION)

**Not covered:**
- Content quality assessment (use axiom-system-architect)
- Fixing issues (return to analysis phase)
- Technical accuracy verification (requires domain knowledge)
