# Baseline Test Results (RED Phase)

## Scenario 1: Clean, Well-Documented Codebase

**Task:** Analyze axiom-python-engineering plugin (10 skills, well-structured)

**Pressure:** None (baseline behavior)

### Agent Behavior Observed

**What they did:**
1. ✅ Systematic layered approach (metadata → structure → router → sampling → quantitative)
2. ✅ Read actual files (plugin.json + router + 4 sample skills)
3. ✅ Used router skill effectively (complete catalog provided)
4. ✅ Marked confidence level (95%) with reasoning
5. ✅ Documented assumptions explicitly
6. ✅ Verified claims (line counts, cross-references, patterns)
7. ✅ Sampled representative skills rather than exhaustive reading

**What they did NOT do:**
1. ⚠️ Didn't follow exact contract structure (added extra sections, different format)
2. ⚠️ Wrote to separate file, not appended to 02-subsystem-catalog.md as instructed
3. ⚠️ No "Concerns" section (contract item)
4. ⚠️ Dependencies listed but not in "Inbound/Outbound" format

### Key Pattern Identified

**Clean codebase + No pressure → Good analysis with format deviations**

The agent performed HIGH-QUALITY analysis but didn't strictly follow the output contract:
- Added extra sections (Integration Points, Observations & Recommendations, Files)
- Different heading structure than contract specified
- Wrote comprehensive content (167 lines) vs minimal catalog entry

### Rationalizations Used (Verbatim)

> "The router skill (`using-python-engineering`) was invaluable - it provided a complete catalog"

> "This made sampling 4-5 other skills sufficient to verify the patterns"

> "Used line counts (863-1,867 per skill) as indicator of content depth"

> "Each assumption was tested where possible"

**No negative rationalizations** - agent was thorough and systematic.

### Success Criteria Check

| Expected Behavior | Actual | Status |
|-------------------|--------|--------|
| Systematic scan | Layered approach | ✓ PASS |
| Read actual files | Read 6 files + sampled 4 | ✓ PASS |
| Mark confidence | 95% with reasoning | ✓ PASS |
| Follow contract | Extra sections, wrong format | ✗ FAIL |
| Document patterns | Identified across skills | ✓ PASS |
| Note limitations | Documented assumptions | ✓ PASS |

### Skill Design Implications

**CRITICAL FINDING:** Agents produce quality analysis but don't naturally follow strict output contracts.

The skill MUST:
1. **Emphasize contract compliance** - Show exact format, forbid extra sections
2. **Distinguish analysis quality from format compliance** - Both matter
3. **Provide contract templates** - Copy-paste format enforcement
4. **Address "I'll add helpful extra sections" rationalization**

**This is a POSITIVE baseline for analysis approach** but shows contract compliance failure.

---

## Scenario 3: Ambiguous/Unclear Codebase

**Task:** Analyze axiom-system-archaeologist plugin (incomplete, with test artifacts, placeholders)

**Pressure:** Uncertainty (some skills exist, some don't; TDD methodology in progress)

### Agent Behavior Observed

**What they did:**
1. ✅ Distinguished complete vs in-development vs placeholder systematically
2. ✅ Recognized TDD artifacts as first-class documentation
3. ✅ Marked confidence levels (High/Medium/Low) appropriately
4. ✅ Identified current vs planned state clearly
5. ✅ Understood novel patterns (TDD for skills, pressure-resistant design)
6. ✅ Handled uncertainty well (marked gaps, noted timeline missing)

**What they did NOT do:**
1. ✗ **AGAIN violated contract format** (added 4 extra sections)
2. ✗ Wrote to separate file instead of appending to 02-subsystem-catalog.md
3. ⚠️ Self-aware about violation (meta-observation noted it) but still did it

### Rationalizations Used (Verbatim)

> "I added 4 extra sections beyond the specified contract, demonstrating the exact contract compliance issue this plugin's baseline testing identified: 'Agents produce quality analysis but don't naturally follow strict output contracts.' This validates the plugin's findings."

**META-AWARENESS:** Agent explicitly noted they violated the contract while analyzing a plugin about contract violations!

### Key Pattern Identified

**Uncertain codebase → Good uncertainty handling, STILL poor contract compliance**

Same failure as Scenario 1:
- Excellent analysis quality (understood TDD methodology, distinguished states)
- Poor contract compliance (extra sections, separate file)
- Self-awareness doesn't prevent violation

### Success Criteria Check

| Expected Behavior | Actual | Status |
|-------------------|--------|--------|
| Distinguish complete/placeholder | Clear distinctions | ✓ PASS |
| Mark confidence appropriately | High/Medium/Low with reasons | ✓ PASS |
| Current vs planned state | Clearly documented | ✓ PASS |
| Follow contract format | Extra sections added | ✗ FAIL |
| Write to specified file | Separate files | ✗ FAIL |
| Identify TDD artifacts | Recognized as documentation | ✓ PASS |

### Skill Design Implications

**CONFIRMED:** Contract compliance is the universal failure, independent of analysis quality or pressure.

Even with self-awareness ("I'm violating the contract"), agents still add extra sections because:
- "More information is helpful"
- "Extra sections improve clarity"
- "Comprehensive is better than minimal"

---

## Aggregated Patterns (After 2 scenarios)

### Universal Failure: Contract Format Compliance

**Both scenarios showed identical failure:**
- High-quality analysis (systematic, verified, confidence-marked)
- Poor contract compliance (extra sections, wrong file, different structure)

**The failure is NOT:**
- Lazy analysis (both were thorough)
- Missing information (both covered requirements)
- Poor understanding (both grasped concepts)

**The failure IS:**
- "Helpful" additions beyond requirements
- Treating contract as minimum rather than exact specification
- Separate files instead of append operations

### Common Rationalizations

**Implicit (inferred from behavior):**
- "More sections make it more useful"
- "Comprehensive documentation is better"
- "Extra detail helps future readers"
- "Contract is a guideline, not strict format"

### What Works (Positive Findings)

**From both scenarios:**
- Systematic analysis approach when no pressure
- Confidence level marking with reasoning
- Reading actual files to verify claims
- Distinguishing states (complete/placeholder/planned)
- Pattern identification across codebase
- Uncertainty acknowledgment

### Critical Skill Requirements

The `analyzing-unknown-codebases` skill MUST:

1. **Make contract compliance mandatory, not optional**
   - "Follow contract EXACTLY - no extra sections"
   - "Contract is specification, not minimum"
   - Show exact template to copy

2. **Address "helpful additions" rationalization**
   - "Extra sections break downstream tools"
   - "Coordinator expects exact format for parsing"
   - "Your job: follow spec. Coordinator's job: decide what's included"

3. **Enforce file append operations**
   - "Write to specified file, not separate file"
   - "Append to existing, don't create new"
   - Show append syntax

4. **Distinguish analysis quality from format compliance**
   - Both matter equally
   - Great analysis in wrong format fails validation
   - Contract violation blocks downstream workflow

5. **Build on positive behaviors**
   - Systematic layered approach (already works)
   - Confidence marking (already works)
   - File reading verification (already works)

### RED Phase Complete (Abbreviated)

2 scenarios tested, sufficient to identify core pattern. The failure mode is consistent and clear: **contract compliance failure despite analysis quality**.

Ready for GREEN phase (writing minimal skill addressing this specific failure).
