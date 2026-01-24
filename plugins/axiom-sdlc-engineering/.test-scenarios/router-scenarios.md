# using-sdlc-engineering Router - Test Scenarios

## Purpose

Test that the router skill correctly:
1. Routes user requests to appropriate skills
2. Detects CMMI maturity level (from CLAUDE.md, conversation, or defaults to L3)
3. Handles ambiguous requests
4. Coordinates with other skillpacks
5. Provides clear guidance when multiple skills apply

## Scenario 1: Basic Routing - Requirements

### Context
- New project, no existing CMMI practices
- User wants to set up requirements tracking
- CMMI level not specified (should default to L3)

### User Request
"How do I track requirements for this project?"

### Expected Behavior
- Detects default to CMMI Level 3
- Routes to `requirements-lifecycle` skill
- Mentions traceability as key concern for L3

### Baseline Behavior (RED)
[To be filled: What does Claude do WITHOUT the router skill?]

### With Skill (GREEN)
[To be filled: What does Claude do WITH the router skill?]

### Loopholes Found (REFACTOR)
[To be filled: What rationalizations/failures occurred?]

---

## Scenario 2: Ambiguous Request - "Better Quality"

### Context
- Existing project with technical debt
- User frustrated with bugs
- No CMMI level specified

### User Request
"We need better quality. How do we improve?"

### Expected Behavior
- Recognizes ambiguity (could be: quality-assurance, design-and-build, governance-and-risk)
- Asks clarifying question OR provides overview of quality-related skills
- Default to L3
- Suggests starting with quality-assurance skill (most direct)

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 3: CMMI Level Detection - CLAUDE.md

### Context
- Project has CLAUDE.md with: `CMMI Target Level: 2`
- User asks about code review process

### User Request
"What code review process should we use?"

### Expected Behavior
- Detects Level 2 from CLAUDE.md
- Routes to `quality-assurance` skill
- Passes Level 2 context to skill
- Recommends basic peer review (not Level 3 formality)

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 4: CMMI Level Detection - Conversation

### Context
- No CLAUDE.md configuration
- User explicitly mentions level in request

### User Request
"We need to meet CMMI Level 4 requirements for our medical device software. How do we set up metrics?"

### Expected Behavior
- Detects Level 4 from user message
- Routes to `quantitative-management` skill
- Passes Level 4 context
- Emphasizes statistical process control

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 5: Multi-Skill Request - New Project Setup

### Context
- Brand new project, no code yet
- User wants comprehensive CMMI setup
- Level 3 target

### User Request
"I'm starting a new project and want to follow CMMI Level 3 from day one. Where do I start?"

### Expected Behavior
- Recognizes need for multiple skills (lifecycle-adoption, requirements-lifecycle, design-and-build, quality-assurance)
- Routes to `lifecycle-adoption` FIRST (bootstrapping guidance)
- Provides roadmap of other skills to use next
- Mentions platform-integration for tooling setup

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 6: Cross-Skillpack Coordination - Python Testing

### Context
- Python project using pytest
- User asks about testing approach
- Could use axiom-python-engineering OR axiom-sdlc-engineering

### User Request
"How should I approach testing in my Python project?"

### Expected Behavior
- Recognizes overlap with axiom-python-engineering
- Routes to `quality-assurance` for test STRATEGY (what to test, coverage policy)
- Mentions axiom-python-engineering for pytest IMPLEMENTATION (how to write tests)
- Clear separation: SDLC = "what/why", Python = "how"

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 7: Pressure - "This is Too Much Process"

### Context
- Small 2-person team
- User perceives CMMI as heavyweight
- Time pressure: needs to ship in 2 weeks

### User Request
"We're just two developers. Isn't CMMI overkill? We need to ship fast."

### Expected Behavior
- Acknowledges team size concern
- Routes to `lifecycle-adoption` skill
- Emphasizes Level 2 (minimal viable) vs. Level 3
- Shows how CMMI adapts to team size
- Counters "process = slow" rationalization

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 8: Platform-Specific Question

### Context
- Team uses GitHub
- Want to implement traceability
- Level 3 target

### User Request
"How do I set up requirements traceability in GitHub?"

### Expected Behavior
- Routes to `platform-integration` skill (GitHub section)
- References `requirements-lifecycle` for WHAT traceability means
- Shows GitHub-specific implementation (issue refs, projects, labels)

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 9: Adoption on Existing Project

### Context
- 6-month-old project with existing codebase
- No documentation, no tests, no formal process
- Management wants "royal commission standard"

### User Request
"We have an existing project with no process. How do we adopt CMMI without stopping development?"

### Expected Behavior
- Routes to `lifecycle-adoption` skill (this is the primary use case)
- Mentions parallel tracks approach (new work follows new process, legacy exempt)
- Default to Level 3 (royal commission suggests high quality)
- References other skills for specific practices

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 10: Risk and Decision - ADR Question

### Context
- Technical decision needed: microservices vs. monolith
- Level 3 project
- Need defensible rationale

### User Request
"Should we use microservices or a monolith? We need to document this decision."

### Expected Behavior
- Routes to `governance-and-risk` skill (DAR process)
- Shows ADR template
- Guides through alternatives analysis
- Mentions Level 3 requires formal decision documentation

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 11: Metrics Question - "What Should We Measure?"

### Context
- Level 3 project
- No metrics currently collected
- User overwhelmed by options

### User Request
"What metrics should we track? I don't want to create measurement theater."

### Expected Behavior
- Routes to `quantitative-management` skill
- Emphasizes GQM (Goal-Question-Metric) approach
- For Level 3: trend analysis and baselines (not full SPC)
- Warns against vanity metrics

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 12: Conflicting Pressures - Fast Ship vs. Audit Trail

### Context
- Medical device project (FDA regulated)
- Deadline in 1 month
- User feels tension between speed and compliance

### User Request
"We need to ship this medical device software in a month, but also need full traceability for FDA. Can we do both?"

### Expected Behavior
- Recognizes medical device = Level 3-4 rigor required
- Routes to `lifecycle-adoption` for realistic timeline assessment
- References `requirements-lifecycle` for traceability approach
- References `governance-and-risk` for regulatory risk
- Acknowledges tension, provides risk-based prioritization

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Success Criteria

Router skill is ready when:

- [ ] All 12 scenarios routed correctly
- [ ] CMMI level detected from all sources (CLAUDE.md, conversation, default)
- [ ] Ambiguous requests handled with clarification or overview
- [ ] Cross-skillpack coordination clear (SDLC vs. implementation skillpacks)
- [ ] Rationalization table complete for "process is overkill" objections
- [ ] Flowchart added if routing logic non-obvious
- [ ] Red flags section for common misrouting patterns
