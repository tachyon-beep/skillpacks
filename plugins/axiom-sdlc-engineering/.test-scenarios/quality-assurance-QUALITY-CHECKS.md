# Quality Assurance Skill - Quality Checks

## Date
2026-01-24

## Checklist from writing-skills Skill

### Frontmatter ✅

- [x] Name uses only letters, numbers, hyphens (no parentheses/special chars)
  - ✅ "quality-assurance" - valid
- [x] YAML frontmatter with only name and description (max 1024 chars)
  - ✅ 2 fields only, well under limit (~200 chars)
- [x] Description starts with "Use when..." and includes specific triggers/symptoms
  - ✅ "Use when deciding test strategy, struggling with code reviews, shipping without tests, or conflating verification with validation"
- [x] Description written in third person
  - ✅ Third person throughout
- [x] Description does NOT summarize workflow
  - ✅ Only triggers (test strategy, code reviews, shipping without tests, VER/VAL confusion)

### Content Structure ✅

- [x] Clear overview with core principle
  - ✅ "Core principle: Verification ≠ Validation"
- [x] When to Use section with symptoms
  - ✅ Bulleted list with 7 specific symptoms
- [x] Quick Reference table
  - ✅ Table mapping situations to reference sheets and key decisions

### Keywords for Search (CSO) ✅

Error messages/symptoms included:
- ✅ "LGTM rubber stamps"
- ✅ "tests pass but customers report bugs"
- ✅ "same defects recurring"
- ✅ "manual testing taking days"
- ✅ "ice cream cone anti-pattern"
- ✅ "validation theater"
- ✅ "defect whack-a-mole"

### Addressing Baseline Failures ✅

All 6 gaps from RED phase addressed:

1. **Frameworks/Methodologies** ✅
   - Exception Protocol (TEST-HOTFIX)
   - Review Metrics (20-40% finding rate)
   - Test Pyramid Economics
   - VER/VAL distinction
   - RCA Methods (5 Whys, Fishbone)

2. **Social/Cultural Dynamics** ✅
   - Psychological Safety Playbook (peer-reviews.md)
   - Rubber Stamp anti-pattern
   - Social pressure addressed
   - Reviewer accountability

3. **Project Level Context** ✅
   - Level 2/3/4 requirements clearly delineated
   - Level scaling reference sheet
   - Escalation/de-escalation criteria

4. **Root Causes** ✅
   - 5 Whys methodology
   - Fishbone diagram
   - Fault Tree Analysis
   - RCA quality requirements

5. **Risk Assessment** ✅
   - Risk matrices in scenarios
   - Probability/impact analysis
   - Cost comparisons (2-4 hours testing vs 20-40 hours debugging)

6. **Metrics/Measurement** ✅
   - Defect escape rate (<10% target)
   - Review finding rate (20-40% target)
   - Test coverage (>70% Level 3)
   - MTTR by severity
   - Metrics enforcement with audit triggers

### Code/Examples ✅

- [x] One excellent example (not multi-language)
  - ✅ Real-world examples throughout reference sheets
  - ✅ Concrete scenarios, not generic templates
  - ✅ Before/after comparisons (ice cream cone → pyramid)

### Supporting Files ✅

- [x] Supporting files only for tools or heavy reference
  - ✅ 6 reference sheets (each 190-280 lines, appropriate for on-demand loading)
  - ✅ All reference material (test pyramid, RCA methods, UAT process)
  - ✅ Main SKILL.md at 446 lines (appropriate overview + quick reference)

### Common Mistakes Section ✅

- [x] Common mistakes with fixes
  - ✅ Table with 7 common mistakes, why they fail, better approach

### Anti-Patterns ✅

- [x] Anti-patterns documented
  - ✅ 5 major anti-patterns with detection, red flags, why it fails, counter
  - ✅ Test Last, Rubber Stamp Reviews, Ice Cream Cone, Defect Whack-a-Mole, Validation Theater

### Flowcharts ❌ NOT NEEDED

- [x] Small flowchart only if decision non-obvious
  - ✅ No flowcharts (tables and bullet lists more appropriate for this skill type)
  - ✅ Decision points clearly expressed in Quick Reference table

### REFACTOR Phase ✅

- [x] Loopholes closed
  - ✅ 5 loopholes identified and closed
  - ✅ Re-tested and verified bulletproof
  - ✅ No new rationalizations emerged

### Integration ✅

- [x] Cross-references to other skills
  - ✅ Links to axiom-python-engineering, ordis-quality-engineering, design-and-build, requirements-lifecycle

---

## Token Efficiency Check

**Main SKILL.md**: 446 lines (~3,100 words)
- Target: <500 words for frequent skills, or appropriate for reference
- ✅ PASS: This is a reference skill (not frequently-loaded), appropriate length

**Reference sheets**: 6 files, 190-280 lines each (~1,400-2,000 words each)
- ✅ PASS: Separated for on-demand loading, not all loaded at once

**Total skill content**: ~12,500 words across 7 files
- ✅ PASS: Comprehensive reference justified for CMMI process area coverage

---

## Writing Style Check

- [x] No narrative storytelling
  - ✅ Focused on reusable patterns and frameworks
- [x] Active voice, clear language
  - ✅ Direct, actionable guidance throughout
- [x] No emojis (unless user requested)
  - ✅ None used
- [x] Professional tone
  - ✅ Appropriate for enterprise/CMMI context

---

## Overall Quality Assessment

**Rating**: ✅ PRODUCTION-READY

**Strengths**:
1. Clear VER/VAL distinction (core principle)
2. Comprehensive anti-pattern catalog
3. Loophole-resistant (5 loopholes closed and verified)
4. Level-based scaling (L2/3/4)
5. Enforcement mechanisms (metrics, audits, escalation)
6. Integration with prescription document

**No major gaps identified**

**Recommendation**: READY FOR DEPLOYMENT

---

**Last Updated**: 2026-01-24
**Quality Check Status**: PASSED
