# Test Scenarios for documenting-system-architecture Skill

## Purpose

Test agent behavior when synthesizing subsystem catalogs and architecture diagrams into final stakeholder-ready documentation, identifying baseline failures to inform skill design.

## Scenario 1: Complete Inputs → Comprehensive Report

**Pressure:** None (baseline behavior)

**Setup:**
- Complete `02-subsystem-catalog.md` (5 subsystems, all sections filled)
- Complete `03-diagrams.md` (Context, Container, 3x Component diagrams)
- All confidence levels marked
- Clear dependencies documented

**Task:**
"You're in `docs/arch-analysis-test-report-1/`. Synthesize the subsystem catalog and architecture diagrams into a final architecture report. Write to `04-final-report.md`.

**Expected output:**
- Executive summary (2-3 paragraphs)
- System overview (purpose, scope, technology stack)
- Architecture diagrams with context
- Subsystem descriptions (synthesized from catalog)
- Key findings (patterns, concerns, recommendations)
- Appendices (methodology, confidence levels, assumptions)
- Navigation structure (table of contents, cross-references)"

**Expected Behavior:**
- ✅ Read both catalog and diagrams
- ✅ Create coherent narrative (not just concatenation)
- ✅ Executive summary at top (high-level takeaways)
- ✅ Diagrams embedded with explanatory text
- ✅ Cross-reference catalog ↔ diagrams
- ✅ Table of contents for navigation
- ✅ Synthesize patterns across subsystems
- ✅ Aggregate concerns and recommendations

**Failure Modes to Watch:**
- ❌ Simple concatenation (paste catalog + diagrams without synthesis)
- ❌ No executive summary
- ❌ Diagrams without context
- ❌ No cross-references
- ❌ Missing navigation (TOC, links)
- ❌ Concerns buried (not elevated to findings)

## Scenario 2: Incomplete Inputs → Graceful Documentation

**Pressure:** Uncertainty (gaps in analysis)

**Setup:**
- `02-subsystem-catalog.md` with:
  - 3 complete subsystems (high confidence)
  - 2 incomplete subsystems (medium/low confidence, missing details)
- `03-diagrams.md` with:
  - Context and Container diagrams
  - Only 1 Component diagram (time constraint)
  - Documented limitations

**Task:**
"Generate final architecture report acknowledging incompleteness. Write to `04-final-report.md`."

**Expected Behavior:**
- ✅ Document scope limitations explicitly
- ✅ Mark confidence levels throughout
- ✅ "Partial Analysis" section listing what's missing
- ✅ Recommendations for future work
- ✅ Don't hide gaps or uncertainties

**Failure Modes to Watch:**
- ❌ Present incomplete analysis as complete
- ❌ No confidence indicators
- ❌ Missing "future work" section
- ❌ Bury limitations in footnotes

## Scenario 3: Stakeholder Pressure → Executive-Friendly Format

**Pressure:** Audience (C-level executives, non-technical)

**Setup:**
- Complete catalog and diagrams
- Task says: "Presentation for CTO in 2 hours, needs to be exec-readable"

**Task:**
"Create architecture report for executive audience. Prioritize clarity over technical depth. Write to `04-final-report.md`."

**Expected Behavior:**
- ✅ Strong executive summary (business value, risks, recommendations)
- ✅ Diagrams simplified or annotated for non-technical readers
- ✅ Technical jargon explained
- ✅ Focus on patterns, risks, decisions over implementation details
- ✅ Visual hierarchy (bold key findings)
- ✅ Actionable recommendations section

**Failure Modes to Watch:**
- ❌ Dense technical documentation unsuitable for execs
- ❌ No business context
- ❌ Diagrams without explanatory text
- ❌ Missing risk assessment

## Scenario 4: Large System → Navigation and Organization

**Pressure:** Complexity (15 subsystems across 5 categories)

**Setup:**
- `02-subsystem-catalog.md` with 15 subsystems
- `03-diagrams.md` with faction-based grouping and 3 Component diagrams
- Multiple patterns identified across categories

**Task:**
"Create comprehensive architecture report for large marketplace system. Ensure navigability. Write to `04-final-report.md`."

**Expected Behavior:**
- ✅ Detailed table of contents (multi-level)
- ✅ Group subsystems by category in report
- ✅ "Architecture at a Glance" summary section
- ✅ Pattern catalog (patterns identified across system)
- ✅ Dependency analysis (which subsystems are most connected?)
- ✅ Cross-references between sections
- ✅ Appendix structure (don't overwhelm main body)

**Failure Modes to Watch:**
- ❌ Flat list of 15 subsystems (hard to navigate)
- ❌ No grouping or categorization
- ❌ Missing table of contents
- ❌ Overwhelming main body (no appendices)
- ❌ Patterns not synthesized

## Scenario 5: Technical Depth Choice → Developer Audience

**Pressure:** Audience (software engineers joining project)

**Setup:**
- Complete catalog and diagrams
- Task says: "Onboarding documentation for new developers"

**Task:**
"Create architecture report for developer onboarding. Include technical depth developers need. Write to `04-final-report.md`."

**Expected Behavior:**
- ✅ Technology stack section (languages, frameworks, tools)
- ✅ Development workflow implications
- ✅ Key files and entry points
- ✅ Dependency management details
- ✅ Testing approach (from catalog concerns)
- ✅ Common patterns to follow
- ✅ Known issues and workarounds

**Failure Modes to Watch:**
- ❌ Executive summary style (too high-level for devs)
- ❌ Missing technical details
- ❌ No file/code references
- ❌ Patterns described abstractly without code context

## Success Criteria (Across All Scenarios)

**Synthesis Quality:**
- Coherent narrative (not concatenation)
- Patterns identified across subsystems
- Concerns elevated to findings
- Recommendations based on analysis

**Navigation:**
- Table of contents (multi-level for large systems)
- Cross-references between sections
- Clear section structure
- Appendices for supporting detail

**Audience Adaptation:**
- Executive summary for all audiences
- Depth adjusted to audience needs
- Jargon explained or avoided as appropriate
- Visual hierarchy (bold, headings, lists)

**Completeness Transparency:**
- Confidence levels marked
- Limitations documented
- Future work section
- Assumptions explicit

**Technical Accuracy:**
- Diagrams match catalog
- Dependencies consistent
- Patterns correctly identified
- No invented information

## Baseline Testing Protocol

For each scenario:

1. **Create workspace** with test inputs (catalog + diagrams)
2. **Run baseline** WITHOUT skill loaded
3. **Document behavior:**
   - Report structure and organization?
   - Synthesis vs concatenation?
   - Navigation elements present?
   - Audience appropriateness?
   - Rationalizations observed (verbatim)
4. **Identify failure patterns**
5. **Aggregate findings** across scenarios

**Target:** Identify universal patterns (synthesis quality? navigation? audience adaptation? completeness transparency?).
