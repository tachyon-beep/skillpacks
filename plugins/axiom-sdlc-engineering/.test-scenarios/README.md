# Test Scenarios for axiom-sdlc-engineering

## TDD Methodology for Skills

Following the **writing-skills** skill, we apply RED-GREEN-REFACTOR to all skill creation:

### The Cycle

1. **RED Phase**: Run scenarios WITHOUT the skill
   - Document baseline behavior
   - Capture rationalizations verbatim
   - Identify failure patterns

2. **GREEN Phase**: Write minimal skill
   - Address specific baseline failures
   - Re-run scenarios
   - Verify compliance

3. **REFACTOR Phase**: Close loopholes
   - Find new rationalizations
   - Add explicit counters
   - Build rationalization tables
   - Re-test until bulletproof

### Scenario Structure

Each scenario file contains:

```markdown
# Skill Name - Test Scenarios

## Scenario N: [Name]

### Context
[Project state, pressures, constraints]

### User Request
[Exact user message]

### Expected Behavior
[What should happen with the skill]

### Baseline Behavior (RED)
[What actually happened WITHOUT skill - to be filled during RED phase]

### With Skill (GREEN)
[What happened WITH skill - to be filled during GREEN phase]

### Loopholes Found (REFACTOR)
[New rationalizations discovered - to be filled during REFACTOR phase]
```

### Pressure Types

According to writing-skills, combine multiple pressures:

- **Time pressure**: "Need this by end of day"
- **Sunk cost**: "Already spent 2 weeks on this"
- **Authority pressure**: "VP wants it this way"
- **Exhaustion**: "Been working on this for days"
- **Scope pressure**: "Just add one more thing"
- **Informality bias**: "This is overkill"

### Success Criteria

A skill is ready when:
- [ ] All baseline behaviors documented
- [ ] All rationalizations captured
- [ ] Skill addresses specific failures
- [ ] No new loopholes found in extended testing
- [ ] Rationalization table complete

## Scenario Files

- `router-scenarios.md` - using-sdlc-engineering router skill
- `lifecycle-adoption-scenarios.md` - Adopting CMMI on existing projects
- `requirements-lifecycle-scenarios.md` - Requirements management and development
- `design-and-build-scenarios.md` - Technical solution, configuration management
- `quality-assurance-scenarios.md` - Verification and validation
- `governance-and-risk-scenarios.md` - Decision analysis and risk management
- `quantitative-management-scenarios.md` - Measurement and analysis
- `platform-integration-scenarios.md` - GitHub/Azure DevOps implementation

## Testing Log

Track testing progress:

| Skill | RED Complete | GREEN Complete | REFACTOR Complete | Status |
|-------|-------------|----------------|-------------------|--------|
| router | ❌ | ❌ | ❌ | Not started |
| lifecycle-adoption | ❌ | ❌ | ❌ | Not started |
| requirements-lifecycle | ❌ | ❌ | ❌ | Not started |
| design-and-build | ❌ | ❌ | ❌ | Not started |
| quality-assurance | ❌ | ❌ | ❌ | Not started |
| governance-and-risk | ❌ | ❌ | ❌ | Not started |
| quantitative-management | ❌ | ❌ | ❌ | Not started |
| platform-integration | ❌ | ❌ | ❌ | Not started |
