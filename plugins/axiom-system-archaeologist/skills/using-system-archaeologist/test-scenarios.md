# Test Scenarios for using-system-archaeologist

## Scenario 1: Quick Architecture Request (Time Pressure)

**User request:** "I need architecture documentation for this project ASAP - we have a meeting in an hour."

**Pressures applied:**
- Time constraint (urgency)
- Authority (meeting implies stakeholders)

**Expected baseline failures (without skill):**
- Skip holistic assessment, jump straight to file scanning
- No workspace creation
- Generate diagrams without proper subsystem identification
- Miss critical architectural decisions
- No validation of outputs

**Success criteria (with skill):**
- Creates proper workspace structure
- Performs holistic assessment first
- Documents decision to do rapid analysis given time constraint
- Still validates outputs before presenting
- Produces navigable report even if abbreviated

---

## Scenario 2: Incremental Update Request (Sunk Cost Pressure)

**Setup:** Existing analysis workspace at `docs/arch-analysis-2025-11-10/`

**User request:** "The API subsystem has changed significantly. Update the analysis."

**Pressures applied:**
- Sunk cost (existing work to preserve)
- Ambiguity (how much has changed?)

**Expected baseline failures (without skill):**
- Re-run full analysis from scratch (ignoring existing work)
- OR update without proper validation
- OR make assumptions about what changed without checking
- No coordination log update

**Success criteria (with skill):**
- Detects existing workspace
- Asks user to confirm workspace
- Reads coordination log to understand prior work
- Spawns focused subagent for API subsystem only
- Updates only relevant sections
- Validates updated sections
- Documents incremental work in coordination log

---

## Scenario 3: Complex Codebase (Overwhelm Pressure)

**Setup:** Large monorepo with 500K+ LOC, 15+ microservices, multiple languages

**User request:** "Analyze this codebase and document the architecture."

**Pressures applied:**
- Complexity overwhelm
- Uncertainty about where to start

**Expected baseline failures (without skill):**
- Get lost in details without holistic view
- Analyze sequentially even though parallel would be better
- No clear subsystem boundaries identified
- Incomplete analysis (miss some services)
- No decision documentation about strategy

**Success criteria (with skill):**
- Creates workspace
- Performs holistic scan first (directory structure, tech stack, entry points)
- Identifies all 15+ microservices
- Documents decision: "Large system (500K LOC, 15 services) → parallel analysis strategy"
- Spawns multiple parallel subagents for independent services
- Coordinates findings into coherent whole
- Documents confidence levels and unknowns

---

## Scenario 4: Validation Failure Handling (Authority Pressure)

**Setup:** Subagent produces incomplete subsystem catalog (missing dependencies section)

**Pressures applied:**
- Progress momentum (want to keep moving forward)
- Authority (validator is "just another agent")

**Expected baseline failures (without skill):**
- Ignore validation failure
- Proceed to diagram generation with incomplete data
- Rationalize: "The subsystem catalog is mostly complete, good enough"
- OR argue with validator: "Dependencies section isn't critical"

**Success criteria (with skill):**
- Reads validation report
- Does NOT proceed to next phase
- Spawns subagent again with specific fix instructions
- Re-validates
- Only proceeds after APPROVED status
- Maximum 2 retries, then escalates to user

---

## Combined Pressure Scenario 5: All At Once

**Setup:** Large codebase, time pressure, existing incomplete analysis

**User request:** "We started analyzing this last week but didn't finish. The deadline is today - can you complete the architecture documentation?"

**Pressures applied:**
- Time (deadline today)
- Sunk cost (existing incomplete work)
- Complexity (large codebase)
- Authority (deadline implies stakeholders)

**Expected baseline failures (without skill):**
- Rush through without validation
- Skip holistic assessment (assume prior work covered it)
- Generate quick diagrams without proper subsystem analysis
- Don't check if prior analysis is complete/correct
- Produce incoherent report mixing old and new findings

**Success criteria (with skill):**
- Finds and reads existing workspace
- Assesses what's complete vs incomplete
- Documents strategy: "Deadline pressure + incomplete work → validate existing, complete gaps, prioritize validation"
- Runs validation on existing documents first
- Identifies gaps
- Completes missing work
- Validates all outputs
- Documents time constraints influenced strategy but didn't compromise quality gates

---

## Testing Protocol

1. **Baseline (RED)**: Run each scenario WITHOUT skill loaded
   - Document exact agent behavior verbatim
   - Note rationalizations used
   - Identify which pressures triggered which failures

2. **With Skill (GREEN)**: Run same scenarios WITH skill loaded
   - Verify agent complies with protocols
   - Note any new rationalizations
   - Check success criteria met

3. **Refactor**: For each new rationalization found
   - Add explicit counter to skill
   - Re-test to verify closure
   - Repeat until bulletproof
