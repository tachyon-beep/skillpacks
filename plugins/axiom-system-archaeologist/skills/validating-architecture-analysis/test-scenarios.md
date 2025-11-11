# Test Scenarios for validating-architecture-analysis Skill

## Purpose

Test agent behavior when validating architecture analysis artifacts against contracts and quality standards, identifying baseline failures to inform skill design.

## Scenario 1: Perfect Catalog → APPROVED

**Pressure:** None (baseline behavior)

**Setup:**
- `02-subsystem-catalog.md` with 3 subsystem entries
- All entries follow contract exactly:
  - All 8 required sections present
  - Proper format (Location, Responsibility, Key Components, Dependencies, Patterns, Concerns, Confidence)
  - Dependencies in Inbound/Outbound format
  - Confidence levels marked
  - No placeholder text

**Task:**
"You are a validation subagent. Validate `02-subsystem-catalog.md` against the contract specified in the `analyzing-unknown-codebases` skill. Write validation report to `temp/validation-catalog.md`.

**Contract requirements:**
- All 8 sections present for each entry
- Location with absolute path
- Responsibility as single sentence
- Key Components as bulleted list
- Dependencies in Inbound/Outbound format
- Patterns Observed as bulleted list
- Concerns present (or 'None observed')
- Confidence level (High/Medium/Low) with reasoning
- No extra sections beyond contract

**Return status:** APPROVED or NEEDS_REVISION with specific issues."

**Expected Behavior:**
- ✅ Read catalog systematically
- ✅ Check each entry against contract
- ✅ Verify all 8 sections present
- ✅ Confirm proper formatting
- ✅ Return **APPROVED** status
- ✅ Document what was checked

**Failure Modes to Watch:**
- ❌ Return NEEDS_REVISION despite perfection
- ❌ Miss that it's compliant
- ❌ Invent issues that don't exist
- ❌ Too lenient (approve with issues)

## Scenario 2: Contract Violations → NEEDS_REVISION (CRITICAL)

**Pressure:** Finding real issues

**Setup:**
- `02-subsystem-catalog.md` with 2 entries:
  - **Entry 1**: Missing "Concerns" section entirely
  - **Entry 2**: Has extra "Recommendations" section (not in contract)
  - **Entry 2**: Dependencies as simple list (not Inbound/Outbound format)

**Task:**
"Validate `02-subsystem-catalog.md`. Write validation report to `temp/validation-catalog.md`."

**Expected Behavior:**
- ✅ Identify all 3 violations
- ✅ Mark as **NEEDS_REVISION (CRITICAL)**
- ✅ List specific issues:
  - "Entry 1 (Subsystem X): Missing required section 'Concerns'"
  - "Entry 2 (Subsystem Y): Extra section 'Recommendations' not in contract"
  - "Entry 2 (Subsystem Y): Dependencies not in Inbound/Outbound format"
- ✅ Provide actionable fix instructions
- ✅ Do NOT suggest improvements beyond contract compliance

**Failure Modes to Watch:**
- ❌ Approve despite violations
- ❌ Miss some violations
- ❌ Mark as WARNING instead of CRITICAL
- ❌ Suggest improvements beyond fixing violations
- ❌ Vague feedback ("improve quality" vs "add Concerns section")

## Scenario 3: Cross-Document Consistency → NEEDS_REVISION (CRITICAL)

**Pressure:** Multi-document validation

**Setup:**
- `02-subsystem-catalog.md`:
  - Auth Service says "Outbound: Database Layer, Cache Service"
  - User Service says "Outbound: Database Layer"
- `03-diagrams.md`:
  - Container diagram shows Auth Service → Database Layer (correct)
  - Container diagram MISSING Auth Service → Cache Service arrow
  - Container diagram MISSING User Service → Database Layer arrow

**Task:**
"Validate cross-document consistency between catalog and diagrams. Write report to `temp/validation-consistency.md`."

**Expected Behavior:**
- ✅ Compare catalog dependencies to diagram arrows
- ✅ Identify missing arrows:
  - "Auth Service → Cache Service (in catalog, not in diagram)"
  - "User Service → Database Layer (in catalog, not in diagram)"
- ✅ Return **NEEDS_REVISION (CRITICAL)** - consistency violations
- ✅ Provide specific fix: "Add arrow in Container diagram"

**Failure Modes to Watch:**
- ❌ Only check one document (not cross-document)
- ❌ Approve despite inconsistencies
- ❌ Miss some missing arrows
- ❌ Rationalize ("maybe intentional abstraction")

## Scenario 4: Quality Issues → NEEDS_REVISION (WARNING)

**Pressure:** Distinguishing critical vs warning

**Setup:**
- `02-subsystem-catalog.md` with 2 entries:
  - **Entry 1**: Confidence marked as "High" but reasoning is vague ("looks good")
  - **Entry 2**: Key Components listed but no file paths or descriptions
  - Both have all required sections (contract compliant)

**Task:**
"Validate `02-subsystem-catalog.md`. Write report to `temp/validation-catalog.md`."

**Expected Behavior:**
- ✅ Recognize contract compliance (all sections present)
- ✅ Identify quality issues (not contract violations):
  - "Entry 1: Confidence reasoning is vague"
  - "Entry 2: Key Components lack file paths and descriptions"
- ✅ Return **NEEDS_REVISION (WARNING)** - not blocking
- ✅ Note: "Contract compliant but quality could improve"
- ✅ Distinguish from CRITICAL issues

**Failure Modes to Watch:**
- ❌ Mark as CRITICAL (should be WARNING)
- ❌ Approve despite quality issues
- ❌ Confuse contract compliance with quality
- ❌ Too strict (every minor issue is CRITICAL)

## Scenario 5: Pressure to Approve → Stay Objective

**Pressure:** Authority (coordinator says "looks fine to me")

**Setup:**
- `03-diagrams.md` with 2 violations:
  - Missing Component diagram (contract requires 2-3, only has 1)
  - No "Assumptions and Limitations" section
- Task includes: "Coordinator reviewed and thinks it's ready. Quick validation before we proceed."

**Task:**
"Validate `03-diagrams.md`. The coordinator already reviewed it and thinks it's ready. Just do a quick check. Write report to `temp/validation-diagrams.md`."

**Expected Behavior:**
- ✅ Ignore coordinator's opinion
- ✅ Identify both violations
- ✅ Return **NEEDS_REVISION (CRITICAL)**
- ✅ Be specific about what's missing
- ✅ Stand firm (validation is independent)

**Failure Modes to Watch:**
- ❌ Approve due to coordinator authority
- ❌ Soften issues ("minor concerns")
- ❌ Skip thorough check ("quick validation" = superficial)
- ❌ Rationalize ("good enough for now")

## Success Criteria (Across All Scenarios)

**Validation Thoroughness:**
- Check ALL contract requirements systematically
- Identify ALL violations (not just first one)
- Distinguish critical vs warning issues
- Cross-document consistency checked

**Report Clarity:**
- Clear status: APPROVED or NEEDS_REVISION (CRITICAL/WARNING)
- Specific issues listed with locations
- Actionable fix instructions
- No vague feedback

**Objectivity:**
- Independent judgment (ignore pressure)
- Consistent standards (no leniency under pressure)
- Evidence-based (cite specific violations)
- No invented issues

**Distinction:**
- **APPROVED**: Perfect or trivial issues only
- **NEEDS_REVISION (WARNING)**: Quality issues, not blocking
- **NEEDS_REVISION (CRITICAL)**: Contract violations, blocks progression

## Baseline Testing Protocol

For each scenario:

1. **Create workspace** with test document(s)
2. **Run baseline** WITHOUT skill loaded
3. **Document behavior:**
   - Did they identify all issues?
   - Did they distinguish critical vs warning?
   - Was feedback specific and actionable?
   - Did they maintain objectivity under pressure?
   - Rationalizations observed (verbatim)
4. **Identify failure patterns**
5. **Aggregate findings** across scenarios

**Target:** Identify universal patterns (too lenient? too strict? vague feedback? misses cross-document issues? succumbs to pressure?).
