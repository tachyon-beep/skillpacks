# design-and-build - Test Scenarios

## Purpose

Test that design-and-build skill correctly:
1. Guides architecture decision records (ADRs)
2. Implements configuration management (branching, merging)
3. Manages technical debt systematically
4. Provides build/integration strategies
5. Prevents cowboy coding and integration hell

## Scenario 1: Quick Fix vs Proper Architecture

### Context
- Production bug needs fix
- **Time pressure**: Customer escalation, fix needed today
- **Sunk cost**: Already spent 2 hours on hack
- Level 3 project requires ADR for architectural changes

### User Request
"There's a production bug. I have a quick fix but it adds technical debt. Do I really need an ADR for this?"

### Expected Behavior
- Design & architecture reference sheet
- ADR decision criteria (when required vs. optional)
- For quick fix: document as tech debt, ADR for proper fix
- Hotfix process vs. regular development
- Counters "emergency = skip process" rationalization

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 2: Git Chaos - No Branch Strategy

### Context
- Team of 6, no consistent branching
- Merge conflicts daily
- Force pushes losing work
- **Frustration**: Morale low

### User Request
"Our Git workflow is a mess. What branching strategy should we use?"

### Expected Behavior
- Configuration management reference sheet
- GitFlow vs. GitHub Flow vs. Trunk-based (choose based on context)
- Branch protection rules
- Merge strategy (squash vs. merge commits)
- References platform-integration for implementation
- Counters "Git is too complicated" fear

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 3: Technical Debt Spiral

### Context
- 2-year-old codebase
- Every feature breaks something else
- **Exhaustion**: Team spending 70% time on bugs
- Level 3 project

### User Request
"We're drowning in technical debt. How do we track and pay it down without stopping feature development?"

### Expected Behavior
- Technical debt management reference sheet
- Debt register (tracking, categorization)
- Paydown strategy (20% rule, debt sprints)
- Debt metrics (ratio, trend)
- Parallel tracks: new features + debt paydown
- Addresses "can't afford to stop" pressure

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 4: CI/CD Setup - Where to Start?

### Context
- Manual builds taking hours
- No automated testing in build
- Level 2 project, wants to improve

### User Request
"We're building manually. How do we set up CI/CD the right way?"

### Expected Behavior
- Build & integration reference sheet
- CI/CD pipeline stages (build → test → integration → deploy)
- Start simple (build + unit tests), add complexity gradually
- Tool recommendations (GitHub Actions, Azure Pipelines)
- References platform-integration for specifics

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 5: Resume-Driven Design

### Context
- Developer wants to use latest trendy tech
- **Authority pressure**: "Everyone is using microservices"
- Project is simple CRUD app
- Level 3 requires ADR

### User Request
"Should we use microservices for this project? I've heard they're best practice."

### Expected Behavior
- Design & architecture reference sheet
- Problem-first technology selection (not trend-first)
- ADR template for alternatives analysis
- Microservices vs. monolith trade-offs
- References governance-and-risk for DAR process
- Counters "resume-driven development" anti-pattern

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Success Criteria

design-and-build skill is ready when:

- [ ] All 5+ scenarios provide actionable guidance
- [ ] ADR template and decision criteria clear
- [ ] Branching strategies compared (GitFlow, GitHub Flow, trunk-based)
- [ ] Technical debt tracking and paydown
- [ ] CI/CD pipeline guidance
- [ ] Anti-patterns: Cowboy Coding, Architecture Astronaut, Resume-Driven Design
- [ ] Reference sheets complete (6 sheets)
