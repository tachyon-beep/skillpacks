# Test Scenarios for analyzing-unknown-codebases

## Context

This skill is invoked as a **subagent** by the `using-system-archaeologist` router. The router has already created workspace structure and written coordination plan. This agent's job is to:
- Perform deep analysis of specified codebase sections
- Write findings following document contracts
- Mark confidence levels appropriately
- Handle uncertainty systematically

## Scenario 1: Clean, Well-Documented Codebase

**Setup:** Analyze `/home/john/skillpacks/plugins/axiom-python-engineering`

**Task specification:**
```markdown
## Task: Analyze axiom-python-engineering plugin
## Context
- Workspace: docs/arch-analysis-test/
- Write to: 02-subsystem-catalog.md (append your section)

## Expected Output
Follow contract:
- Plugin name, location, responsibility
- Key skills (list with purposes)
- Dependencies (inbound/outbound)
- Patterns observed
- Confidence level
```

**Pressure:** None (baseline behavior)

**Expected baseline failures:**
- May skip reading actual skill files, rely on directory names
- May not identify patterns systematically
- May guess at dependencies without verification
- May not mark confidence level
- May write vague descriptions ("handles Python stuff")

---

## Scenario 2: Large, Complex Codebase (Overwhelm)

**Setup:** Analyze entire `/home/john/skillpacks` (15 plugins, 130 skills)

**Task specification:**
```markdown
## Task: Identify all subsystems in skillpacks marketplace
## Context
- Workspace: docs/arch-analysis-test/
- Scope: Full project analysis
- Write to: 01-discovery-findings.md, 02-subsystem-catalog.md

## Expected Output
- Directory structure mapped
- Technology stack identified
- All 15 plugins cataloged with responsibilities
- Architectural patterns documented
```

**Pressure:** Complexity overwhelm (130+ files across 15 directories)

**Expected baseline failures:**
- Miss some plugins (incomplete enumeration)
- Superficial analysis ("14 plugins exist")
- No pattern identification across plugins
- Mixed confidence levels without justification
- Skip systematic directory scan

---

## Scenario 3: Ambiguous/Unclear Codebase

**Setup:** Analyze a codebase with mixed patterns (part of skillpacks that has both active and placeholder content)

**Task specification:**
```markdown
## Task: Analyze axiom-system-archaeologist plugin
## Context
- Workspace: docs/arch-analysis-test/
- Write to: 02-subsystem-catalog.md

## Expected Output
- Plugin structure analysis
- Skills inventory (which exist, which are placeholders)
- Current vs planned state
- Confidence levels for each finding
```

**Pressure:** Uncertainty (some skills exist, some don't; TDD artifacts present)

**Expected baseline failures:**
- Treat placeholders as complete skills
- Don't distinguish between planned vs implemented
- High confidence claims about incomplete items
- Miss test artifacts (scenarios, baselines)
- Don't identify TDD methodology being used

---

## Scenario 4: Time Pressure + Incomplete Instructions

**Setup:** Quick analysis request with vague task spec

**Task specification:**
```markdown
## Task: Quick analysis of lyra-ux-designer
## Context
- Workspace: docs/arch-analysis-test/
- Scope: Fast turnaround needed
- Write to: 02-subsystem-catalog.md
```

**Pressure:** Time pressure + vague requirements

**Expected baseline failures:**
- Skip holistic scan, dive into details
- Write minimal/rushed descriptions
- Don't verify claims by reading files
- Mark everything as "High" confidence to appear thorough
- Skip pattern identification
- Don't document what was NOT analyzed

---

## Scenario 5: Dependency Analysis Challenge

**Setup:** Analyze codebase with complex cross-references

**Task specification:**
```markdown
## Task: Map dependencies for yzmir-ai-engineering-expert (router plugin)
## Context
- Workspace: docs/arch-analysis-test/
- Focus: Identify what this router depends on and what depends on it
- Write to: 02-subsystem-catalog.md

## Expected Output
- Clear inbound/outbound dependencies
- Router pattern documentation
- Cross-references to other skills/plugins
```

**Pressure:** Complex dependency mapping

**Expected baseline failures:**
- Claim "no dependencies" (technically true for runtime, false for logical)
- Miss soft dependencies (routing references)
- Don't distinguish runtime vs logical dependencies
- Incomplete dependency graph
- Don't verify bidirectional relationships

---

## Testing Protocol

**RED Phase:**
1. Run each scenario WITHOUT the skill
2. Document exact agent behavior verbatim
3. Note what was missed, what was wrong, what rationalizations were used
4. Identify patterns across scenarios

**GREEN Phase:**
1. Write minimal skill addressing baseline failures
2. Re-run critical scenarios (1, 2, 3) WITH skill
3. Verify behavior changes

**REFACTOR Phase:**
1. Identify any new rationalizations
2. Add explicit counters
3. Re-test until bulletproof

## Success Criteria

**Agent has succeeded when:**
- Performs systematic holistic scan before details
- Reads actual files to verify claims
- Distinguishes complete vs placeholder vs planned
- Marks confidence levels appropriately (High/Medium/Low)
- Documents patterns observed across codebase
- Follows contract structure exactly
- Notes what was NOT analyzed (explicit limitations)
- Verifies bidirectional dependencies
