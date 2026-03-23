# Quality Assurance Skill - RED Baseline Test Results

**Test Date**: 2026-01-24
**Tester**: Claude Code (without quality-assurance skill)
**Purpose**: Document what happens when QA requests are made WITHOUT specialized guidance

## Test Methodology

For each scenario, I simulated the response Claude Code would give without access to the quality-assurance skill, then analyzed:
1. What response was given
2. What's missing or inadequate
3. What specific guidance would help

---

## Scenario 1: Skip Tests to Ship Faster

**User Request**: "We need to ship this tomorrow but don't have tests. Can we skip them just this once and add them later?"

**Context**: Sprint deadline tomorrow, demo to customer, tests not written yet, Level 3 project

### Simulated Response WITHOUT Skill

"I understand the time pressure. Here are some options:

1. **Ship without tests** - You can deploy with manual testing, but document the technical debt
2. **Write minimal smoke tests** - Cover critical paths only to reduce risk
3. **Negotiate deadline** - Ask for 2-3 more days to add proper test coverage

If you ship without tests, make sure to:
- Create tickets to add tests later
- Do thorough manual testing
- Have a rollback plan
- Monitor closely after deployment

The risk is that bugs may reach production, but sometimes business needs require trade-offs. Just make sure stakeholders understand the increased risk."

### What's Missing/Inadequate

**Critical Gaps**:
1. **No explicit pushback on "just this once"** - Fails to identify the slippery slope pattern
2. **Accepts technical debt as normal** - No guidance on why "later" rarely happens
3. **No risk quantification** - Doesn't help team assess actual risk of shipping untested code
4. **Missing project level context** - Level 3 project means tests are REQUIRED, not optional
5. **No alternative solutions** - Doesn't suggest scope reduction, feature flagging, or beta rollout
6. **Lacks organizational perspective** - No mention of how this decision affects team norms long-term

**Inadequate Guidance**:
- Suggests "create tickets to add tests later" without warning this rarely works
- Doesn't address root cause (why are tests not written alongside code?)
- No mention of writing tests FIRST next time (TDD)
- Missing guidance on what "thorough manual testing" means
- No framework for deciding when shipping without tests is acceptable vs unacceptable

### What Would Help

**Specific Guidance Needed**:
1. **Decision framework** - When is it OK to skip tests vs when it's unacceptable (project level)
2. **Risk assessment** - How to quantify actual risk (what could break? customer impact?)
3. **Alternative strategies** - Scope reduction, feature flags, beta rollout, demo vs production
4. **Slippery slope prevention** - How to ensure "just this once" doesn't become the norm
5. **Root cause analysis** - Why weren't tests written? How to prevent this next sprint?
6. **Contract negotiation** - How to push back on deadline with data
7. **Minimal viable testing** - If you MUST ship, what's the absolute minimum safety net?

---

## Scenario 2: Rubber Stamp Reviews - "LGTM"

**User Request**: "Our code reviews aren't catching bugs. How do we make them more effective?"

**Context**: Code reviews taking 5 minutes, no meaningful feedback, social pressure not to block teammates, defects escaping to production

### Simulated Response WITHOUT Skill

"Code reviews should be thorough. Here are some tips:

1. **Use a checklist** - Create a standard checklist for reviewers to follow
2. **Take more time** - Don't rush through reviews
3. **Look for common issues** - Check for null pointer exceptions, error handling, edge cases
4. **Run the code** - Pull the branch and test it locally
5. **Enforce standards** - Use linters and formatters to catch style issues automatically

You might also want to:
- Pair program for complex changes
- Require 2 reviewers for critical code
- Add automated tests to catch bugs earlier
- Document coding standards so reviewers know what to look for

The key is making reviews a priority, not just a checkbox to tick before merging."

### What's Missing/Inadequate

**Critical Gaps**:
1. **Doesn't address social dynamics** - Misses the core issue: pressure not to block teammates
2. **No psychological safety discussion** - Reviewers may fear being seen as "difficult"
3. **Missing organizational context** - What does "effective review" mean for this team's project level?
4. **No metrics** - How to measure review effectiveness? (defect escape rate, time to find bugs)
5. **Doesn't diagnose root cause** - Why are reviewers rushing? Workload? Incentives? Culture?
6. **No role clarification** - Is reviewer responsible for finding bugs or just ensuring readability?

**Inadequate Guidance**:
- Suggests "take more time" without addressing WHY reviewers are rushing
- Checklist advice is generic, not tailored to team's actual defect patterns
- Doesn't explain HOW to give feedback that won't create conflict
- Missing guidance on what reviewers should vs shouldn't focus on
- No process improvements (review size limits, async vs synchronous, tools)

### What Would Help

**Specific Guidance Needed**:
1. **Social dynamics playbook** - How to give critical feedback without damaging relationships
2. **Reviewer responsibilities** - What is reviewer accountable for? (clarity, correctness, performance?)
3. **Review taxonomy** - Different review depths for different change types (hotfix vs new feature)
4. **Metrics framework** - How to measure review quality (defect detection rate, turnaround time)
5. **Cultural interventions** - How to make "blocking" socially acceptable
6. **Process design** - Review size limits, time allocation, async vs pair review
7. **Root cause diagnosis** - Why are reviews ineffective? (Skill gap? Time pressure? Incentives?)
8. **Training path** - How to teach developers to review effectively

---

## Scenario 3: Test Pyramid Inverted - Ice Cream Cone

**User Request**: "Our tests are all manual and take forever. How do we fix this?"

**Context**: Mostly manual E2E tests, few unit tests, regression testing takes 2 days, Level 2 project

### Simulated Response WITHOUT Skill

"Manual testing is slow and doesn't scale. You should automate more. Here's how:

1. **Add unit tests** - Test individual functions and classes
2. **Automate E2E tests** - Use Selenium, Playwright, or Cypress
3. **Add integration tests** - Test how components work together
4. **Run tests in CI** - Catch issues before they reach QA

Start by:
- Writing unit tests for new code
- Automating your most frequent manual test cases
- Using test frameworks like pytest, Jest, or JUnit
- Setting up a CI pipeline to run tests automatically

Over time, build up your test suite so you have:
- Lots of fast unit tests (seconds)
- Some integration tests (minutes)
- A few critical E2E tests (tens of minutes)

This 'test pyramid' structure gives you fast feedback while still catching integration issues."

### What's Missing/Inadequate

**Critical Gaps**:
1. **No diagnosis of current state** - Doesn't ask WHY tests are all manual
2. **Missing migration strategy** - How do you transition from manual to automated without stopping development?
3. **No resource/skill assessment** - Does team know how to write automated tests?
4. **Lacks prioritization** - Which tests to automate first? (high value, high frequency)
5. **No cost/benefit analysis** - Automation has upfront cost, when is it worth it?
6. **Missing risk management** - What if automated tests are flaky? False positives?

**Inadequate Guidance**:
- Describes test pyramid but doesn't explain WHY it's better
- "Write unit tests for new code" doesn't address the legacy codebase problem
- Doesn't address test data management, test environments, or flakiness
- No guidance on test granularity or what to test at each level
- Missing advice on how to sell this to management (time/cost investment)

### What Would Help

**Specific Guidance Needed**:
1. **Anti-pattern recognition** - Name the problem: "Ice Cream Cone" anti-pattern
2. **Migration playbook** - Step-by-step: How to transition from manual to automated
3. **Prioritization framework** - Which tests to automate first? (ROI calculation)
4. **Test pyramid economics** - Why unit tests are cheaper than E2E tests
5. **Team capability assessment** - Does team have automation skills? Training needed?
6. **Flakiness prevention** - How to avoid creating unreliable automated tests
7. **Legacy code strategy** - How to add tests to untested code (characterization tests, seams)
8. **Management communication** - How to justify upfront investment in automation

---

## Scenario 4: Verification vs Validation Confusion

**User Request**: "All our tests pass, but customers say the feature doesn't work right. What did we miss?"

**Context**: Team conflates testing with user acceptance, tests pass but customers find issues, Level 3 project

### Simulated Response WITHOUT Skill

"It sounds like your tests are passing but they're testing the wrong things. Here's what to check:

1. **Test coverage** - Are you testing all the important scenarios?
2. **Edge cases** - Did you test boundary conditions and error cases?
3. **User acceptance testing** - Have actual users tried the feature?
4. **Requirements clarity** - Did you understand what the customer wanted?

To prevent this:
- Involve users earlier in testing
- Write acceptance criteria before coding
- Test with real data, not just mock data
- Ask customers to review features before release
- Add end-to-end tests that simulate real user workflows

The gap between 'tests pass' and 'feature works' usually means:
- Tests don't match real-world usage
- Requirements weren't clear
- Developers made assumptions about user needs

Consider adding a beta testing phase where real users try features before full release."

### What's Missing/Inadequate

**Critical Gaps**:
1. **Doesn't distinguish verification vs validation** - Missing the fundamental conceptual framework
2. **No definition of "right"** - Who defines "right"? (specs, user needs, business value?)
3. **Lacks product development process** - Where does validation happen in the workflow?
4. **No discussion of BDD/ATDD** - Behavior-driven development could prevent this
5. **Missing stakeholder involvement** - When should product owner/users be involved?
6. **Doesn't address project level** - Level 3 means formal validation is REQUIRED

**Inadequate Guidance**:
- Suggests "involve users earlier" but doesn't say HOW or WHEN
- "Write acceptance criteria" is good but doesn't explain who writes them or format
- Doesn't explain the difference between developer testing and acceptance testing
- No guidance on test scenarios vs test cases vs acceptance tests
- Missing validation gate concept (someone must approve before release)

### What Would Help

**Specific Guidance Needed**:
1. **Verification vs Validation definition** - Building it right vs building the right thing
2. **Validation gate design** - Who approves? What criteria? When in process?
3. **Acceptance criteria format** - Given/When/Then, examples, non-examples
4. **Role clarification** - Developer tests vs QA tests vs acceptance tests
5. **BDD/ATDD practices** - How to write executable specifications
6. **Stakeholder involvement** - Product owner review, user testing, beta programs
7. **Project level requirements** - Level 3 needs formal validation checkpoint
8. **Communication protocol** - How to surface validation failures early (not at release)

---

## Scenario 5: Defect Whack-a-Mole - No Root Cause

**User Request**: "We keep fixing the same types of bugs. How do we stop this?"

**Context**: Same bugs recurring, team firefighting constantly, no time for root cause analysis, Level 3 project

### Simulated Response WITHOUT Skill

"Recurring bugs are frustrating. Here are some ways to prevent them:

1. **Track bug patterns** - Categorize bugs to see what types keep happening
2. **Improve testing** - Add tests to catch these bugs earlier
3. **Code reviews** - Have reviewers watch for these specific issues
4. **Static analysis** - Use linters and type checkers to catch common mistakes
5. **Refactoring** - Fix the underlying code quality issues

For specific bug types:
- Null pointer exceptions → Add null checks, use Optional types
- Race conditions → Review concurrency design, add locks/synchronization
- Off-by-one errors → Write boundary tests, use standard library functions
- Security issues → Security training, use security scanning tools

You might also want to:
- Hold retrospectives to discuss recurring issues
- Create coding standards to prevent known problems
- Pair program on complex areas
- Reduce technical debt that makes bugs easy to introduce

The key is to treat recurring bugs as symptoms of deeper issues, not isolated incidents."

### What's Missing/Inadequate

**Critical Gaps**:
1. **No systematic root cause analysis method** - Doesn't teach 5 Whys, fishbone diagrams, etc.
2. **Missing process improvements** - Focuses on technical fixes, not process/culture changes
3. **No defect taxonomy** - Doesn't categorize by root cause (requirements, design, code, process)
4. **Lacks prevention mindset** - Focuses on detection (tests, reviews) not prevention
5. **No organizational learning** - How to share learnings across team? Knowledge base?
6. **Doesn't address "no time" constraint** - Team is firefighting, when do they do root cause analysis?

**Inadequate Guidance**:
- "Track bug patterns" is good but doesn't say HOW (categories? metrics? tools?)
- Suggests refactoring but doesn't explain how to prioritize what to refactor
- "Treat as symptoms" is right idea but no framework for diagnosis
- Doesn't address why team has "no time" (too much WIP? Poor planning? Understaffed?)
- Missing guidance on preventive measures vs detective measures vs corrective measures

### What Would Help

**Specific Guidance Needed**:
1. **Root cause analysis methods** - 5 Whys, Fishbone, Fault Tree Analysis
2. **Defect taxonomy** - Requirements, Design, Code, Process, Environment
3. **Prevention hierarchy** - Eliminate, Substitute, Engineer Out, Detect, Mitigate
4. **Process improvements** - How to make time for RCA in sprint planning
5. **Organizational learning** - Defect knowledge base, team retrospectives, training
6. **Metrics for improvement** - How to measure if recurring defects are decreasing
7. **Prioritization framework** - Which bugs to RCA first? (frequency × severity)
8. **Systemic fixes** - Architecture changes, process changes, tool changes, not just code fixes

---

## Summary: Critical Gaps Without quality-assurance Skill

### Pattern 1: Lacks Frameworks and Methodologies
Without the skill, responses are ad-hoc advice without systematic frameworks:
- No decision frameworks for when to skip tests
- No root cause analysis methods (5 Whys, fishbone)
- No verification vs validation distinction
- No test pyramid economics

### Pattern 2: Misses Social/Cultural Dynamics
Technical advice without addressing human factors:
- Doesn't address social pressure in code reviews
- Misses "slippery slope" of "just this once"
- No guidance on psychological safety for giving critical feedback
- Doesn't help with stakeholder negotiations

### Pattern 3: No Project Level Context
Ignores that different project levels have different QA requirements:
- Level 3 projects REQUIRE tests, not optional
- Validation gates are mandatory at higher levels
- Risk tolerance varies by project criticality

### Pattern 4: Surface Fixes, Not Root Causes
Treats symptoms instead of diagnosing underlying issues:
- "Write more tests" without asking WHY tests weren't written
- "Take more time on reviews" without addressing time pressure root cause
- "Automate tests" without understanding why they're manual
- "Involve users" without process design for HOW

### Pattern 5: Missing Risk Assessment
No framework for quantifying and managing risk:
- Can't help team assess risk of shipping without tests
- Doesn't explain cost/benefit of test automation
- No guidance on acceptable vs unacceptable risk by project level

### Pattern 6: Lacks Metrics and Measurement
No guidance on how to know if you're improving:
- How to measure code review effectiveness?
- How to track defect escape rate?
- How to measure test automation ROI?
- How to know if root cause analysis is working?

---

## Conclusion

Without the quality-assurance skill, Claude Code provides:
- **Generic advice** that sounds reasonable but lacks depth
- **Technical solutions** without addressing process/culture issues
- **Tactical fixes** without strategic frameworks
- **Individual actions** without organizational context

The skill would need to provide:
1. **Systematic frameworks** for decision-making and analysis
2. **Social/cultural guidance** for navigating human dynamics
3. **Project level context** for risk assessment and requirements
4. **Root cause methodologies** for preventing recurring issues
5. **Metrics and measurement** for tracking improvement
6. **Process design** for integrating QA into workflow

This baseline demonstrates that QA questions require specialized expertise beyond general software engineering knowledge.
