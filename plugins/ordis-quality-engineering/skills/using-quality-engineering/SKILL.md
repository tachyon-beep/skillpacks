---
name: using-quality-engineering
description: Routes to quality engineering skills - E2E testing, performance testing, chaos engineering, test automation, production observability, and reliability patterns
---

# Using Quality Engineering

## Overview

This meta-skill routes you to the right quality engineering skills based on your testing and reliability situation. Load this skill when you need quality engineering expertise but aren't sure which specific skill to use.

**Core Principle**: Different quality challenges require different skills. Match your situation to the appropriate skill, load only what you need.

**Ordis Identity**: Quality engineering is the bulwark against production chaos - systematic testing, controlled failure injection, observability, and verification gates that prevent disorder from reaching users.

## When to Use

Load this skill when:
- Starting any testing or reliability task
- User mentions: "test", "E2E", "integration", "performance", "load test", "chaos", "flaky", "monitoring", "SLO", "production issue"
- You recognize quality/reliability needs but unsure which skill applies
- Setting up quality gates or CI/CD testing

**Don't use for**: Unit testing single functions (use language-specific skills), code review (unless quality-focused)

## Routing by Testing Type

### End-to-End Testing

**Symptoms**: "E2E tests", "integration tests", "full workflow testing", "flaky tests", "test isolation"

**Route to**: `ordis/quality-engineering/e2e-testing-architecture`

**Key Patterns**: Test pyramid, data isolation, avoiding flakiness, selective E2E coverage

**Example**: "Build E2E test suite for checkout flow" → Load e2e-testing-architecture

---

### Performance Testing

**Symptoms**: "Load test", "stress test", "performance regression", "can it handle X users?", "latency", "throughput"

**Route to**: `ordis/quality-engineering/performance-testing-foundations`

**Key Patterns**: Baseline establishment, p95/p99 metrics, load profiles, bottleneck identification

**Example**: "Validate API handles 10k requests/sec" → Load performance-testing-foundations

---

### Chaos Engineering

**Symptoms**: "Test resilience", "failure injection", "what if database fails?", "Game Day", "chaos testing"

**Route to**: `ordis/quality-engineering/chaos-engineering-principles`

**Key Patterns**: Steady-state hypothesis, blast radius control, controlled failures, production chaos

**Example**: "Test if system handles database failure" → Load chaos-engineering-principles

---

### Test Automation Strategy

**Symptoms**: "CI/CD testing", "test pipeline", "test pyramid", "automated testing", "test coverage", "flake management"

**Route to**: `ordis/quality-engineering/test-automation-strategy`

**Key Patterns**: Test pyramid, fast feedback, flake quarantine, test selection

**Example**: "Set up automated testing in CI" → Load test-automation-strategy

---

## Routing by Phase

### Pre-Production (Development/Staging)

**Testing code before production:**

**Route to**:
1. `ordis/quality-engineering/e2e-testing-architecture` - For integrated testing
2. `ordis/quality-engineering/performance-testing-foundations` - For load/stress testing
3. `ordis/quality-engineering/contract-testing` - For microservice boundaries
4. `ordis/quality-engineering/quality-gates-and-verification` - For CI/CD gates

---

### Production (Live Systems)

**Testing and monitoring production:**

**Route to**:
1. `ordis/quality-engineering/production-monitoring-and-observability` - For SLOs, alerting, tracing
2. `ordis/quality-engineering/testing-in-production` - For canary releases, feature flags
3. `ordis/quality-engineering/chaos-engineering-principles` - For production resilience testing
4. `ordis/quality-engineering/reliability-engineering` - For SRE patterns, error budgets

---

## Routing by System Architecture

### Monolithic Applications

**Route to**:
- `ordis/quality-engineering/e2e-testing-architecture` - Full workflow testing
- `ordis/quality-engineering/performance-testing-foundations` - Load testing
- `ordis/quality-engineering/test-automation-strategy` - Testing strategy

**Skip**: contract-testing (not needed for monoliths)

---

### Microservices / Distributed Systems

**Route to**:
- `ordis/quality-engineering/contract-testing` - Testing service boundaries
- `ordis/quality-engineering/production-monitoring-and-observability` - Distributed tracing
- `ordis/quality-engineering/reliability-engineering` - Circuit breakers, retries
- `ordis/quality-engineering/chaos-engineering-principles` - Resilience testing

**Minimize**: e2e-testing-architecture (use contracts instead of E2E where possible)

---

## Routing by Quality Goal

### Preventing Regressions

**Goal**: Ensure new changes don't break existing functionality

**Route to**:
1. `ordis/quality-engineering/test-automation-strategy` - Regression suite design
2. `ordis/quality-engineering/quality-gates-and-verification` - Automated gates
3. `ordis/quality-engineering/e2e-testing-architecture` - Critical path coverage

---

### Improving Reliability

**Goal**: Reduce incidents, improve uptime, meet SLOs

**Route to**:
1. `ordis/quality-engineering/reliability-engineering` - SRE patterns, error budgets
2. `ordis/quality-engineering/production-monitoring-and-observability` - SLO definition
3. `ordis/quality-engineering/chaos-engineering-principles` - Finding weaknesses
4. `ordis/quality-engineering/testing-in-production` - Safe rollout strategies

---

### Establishing Quality Culture

**Goal**: Build testing discipline, quality gates, shift-left

**Route to**:
1. `ordis/quality-engineering/test-automation-strategy` - Testing strategy
2. `ordis/quality-engineering/quality-gates-and-verification` - Gate definition
3. `ordis/quality-engineering/e2e-testing-architecture` - Test design patterns

---

## Specialized Skills

### Test Data Management

**Symptoms**: "Test data", "database seeding", "anonymization", "PII in tests", "data fixtures"

**Route to**: `ordis/quality-engineering/test-data-management`

**When**: Working with databases, sensitive data, realistic test scenarios

**Cross-faction**: Reference `ordis/security-architect/compliance-awareness-and-mapping` for GDPR/HIPAA

---

### Contract Testing

**Symptoms**: "Microservice testing", "API contract", "schema validation", "breaking changes"

**Route to**: `ordis/quality-engineering/contract-testing`

**When**: Distributed systems, avoiding E2E complexity, API versioning

---

### Production Testing

**Symptoms**: "Canary release", "feature flag", "progressive rollout", "blue-green deployment"

**Route to**: `ordis/quality-engineering/testing-in-production`

**When**: Deploying to production, reducing deployment risk

---

### Observability

**Symptoms**: "Monitoring", "alerting", "SLO", "tracing", "logs", "metrics", "on-call"

**Route to**: `ordis/quality-engineering/production-monitoring-and-observability`

**When**: Production systems, on-call, incident response

---

## Core vs Extension Skills

### Core Skills (Universal - Any Project)

Use for **any** project with quality needs:

- `test-automation-strategy` - Testing approach, pyramid, CI/CD
- `e2e-testing-architecture` - Integration and E2E testing
- `performance-testing-foundations` - Load and performance validation
- `quality-gates-and-verification` - Preventing regressions
- `production-monitoring-and-observability` - Production quality signals

### Extension Skills (Specialized Contexts)

Use **only** when context requires:

- `chaos-engineering-principles` - When resilience testing is explicit requirement
- `contract-testing` - When working with microservices/distributed systems
- `test-data-management` - When dealing with complex data needs or sensitive data
- `reliability-engineering` - When SRE practices are needed (SLOs, error budgets)
- `testing-in-production` - When doing progressive rollouts or canary releases

**Decision**: If unsure whether context is "specialized", start with core skills. Specialized needs will be explicit.

---

## Cross-Faction References

Quality engineering often requires skills from other factions:

**Axiom (Python Engineering)**:
- `axiom/python-engineering/pytest-patterns` - Python-specific testing

**Yzmir (AI/ML)**:
- `yzmir/ml-production/model-validation` - ML model testing

**Muna (Documentation)**:
- `muna/technical-writer/operational-acceptance-documentation` - Runbooks for on-call

**Load both factions when**: Language-specific testing needs general testing strategy

---

## Decision Tree

```
Is this quality/testing related?
├─ No → Don't load quality engineering skills
└─ Yes → Continue

What's the testing type?
├─ E2E/Integration → e2e-testing-architecture
├─ Performance/Load → performance-testing-foundations
├─ Resilience/Chaos → chaos-engineering-principles
├─ Test automation → test-automation-strategy
└─ Production monitoring → production-monitoring-and-observability

What's the system architecture?
├─ Monolithic → Focus on E2E, performance, automation
├─ Microservices → ADD: contract-testing, distributed tracing
└─ Both → Use contract-testing for boundaries, E2E for critical paths

What's the phase?
├─ Pre-production → Testing skills (E2E, performance, contracts)
├─ Production → Observability skills (monitoring, testing-in-production)
└─ Both → Load both categories

Specialized needs?
├─ Complex test data → ADD: test-data-management
├─ SRE practices → ADD: reliability-engineering
├─ Progressive rollout → ADD: testing-in-production
└─ None → Core skills sufficient
```

---

## Common Routing Patterns

### Pattern 1: New Feature Testing

```
User: "Add E2E tests for new checkout flow"
You: Loading ordis/quality-engineering/e2e-testing-architecture
```

### Pattern 2: Performance Validation

```
User: "Load test the new API endpoint"
You: Loading ordis/quality-engineering/performance-testing-foundations
```

### Pattern 3: Microservice Integration

```
User: "Test integration between order service and payment service"
You: Loading ordis/quality-engineering/contract-testing
```

### Pattern 4: Production Reliability

```
User: "We need better SLOs and monitoring"
You: Loading ordis/quality-engineering/production-monitoring-and-observability +
     ordis/quality-engineering/reliability-engineering
```

### Pattern 5: CI/CD Pipeline

```
User: "Set up automated testing in our pipeline"
You: Loading ordis/quality-engineering/test-automation-strategy +
     ordis/quality-engineering/quality-gates-and-verification
```

### Pattern 6: Chaos Engineering

```
User: "Test if we can handle database failures"
You: Loading ordis/quality-engineering/chaos-engineering-principles
```

### Pattern 7: Progressive Deployment

```
User: "Deploy with canary release"
You: Loading ordis/quality-engineering/testing-in-production +
     ordis/quality-engineering/production-monitoring-and-observability
```

---

## When NOT to Load Quality Engineering Skills

**Don't load quality engineering skills for**:
- Simple unit tests for pure functions (use language-specific testing)
- Code reviews not focused on quality/testing
- Documentation that isn't testing-related
- Feature implementation without quality requirements

**Example**: "Write unit test for add() function" → No quality engineering skills needed (use language-specific testing)

---

## Quick Reference Table

| Task Type | Load These Skills | Notes |
|-----------|------------------|-------|
| **E2E testing** | e2e-testing-architecture | Focus on critical paths only |
| **Load testing** | performance-testing-foundations | Establish baseline first |
| **Chaos testing** | chaos-engineering-principles | Start non-prod, limit blast radius |
| **CI/CD testing** | test-automation-strategy, quality-gates-and-verification | Fast feedback, zero flakes |
| **Microservice testing** | contract-testing | Avoid E2E where contracts work |
| **Production monitoring** | production-monitoring-and-observability | Define SLOs, symptom-based alerts |
| **SRE practices** | reliability-engineering, production-monitoring-and-observability | Error budgets, graceful degradation |
| **Canary releases** | testing-in-production, production-monitoring-and-observability | Progressive rollout with monitoring |
| **Test data** | test-data-management | Anonymize PII, use factories |
| **Quality gates** | quality-gates-and-verification | Block on failures, not warnings |

---

## Common Mistakes

### ❌ Loading All Skills at Once
**Wrong**: Load all 11 quality-engineering skills for every testing task
**Right**: Load only the skills your situation needs (use decision tree)

### ❌ Skipping Test Automation Strategy
**Wrong**: Jump into E2E tests without thinking about test pyramid
**Right**: Start with test-automation-strategy to plan approach

### ❌ E2E Testing Everything
**Wrong**: Write E2E tests for all functionality
**Right**: E2E for critical paths only, push coverage down pyramid (use contract-testing for boundaries)

### ❌ Testing in Production Without Monitoring
**Wrong**: Canary release without SLO monitoring
**Right**: Load testing-in-production + production-monitoring-and-observability together

### ❌ Not Cross-Referencing Language Skills
**Wrong**: Use only ordis/quality-engineering for Python testing
**Right**: Load axiom/python-engineering for Python specifics + ordis for strategy

---

## Examples

### Example 1: Building E2E Test Suite

```
User: "Create E2E tests for our booking system"

Your routing:
1. Recognize: E2E testing task
2. Load: ordis/quality-engineering/e2e-testing-architecture
3. Apply: Test pyramid, selective coverage, data isolation, condition-based waiting
```

### Example 2: Improving Production Reliability

```
User: "We're having too many incidents. Need better reliability."

Your routing:
1. Recognize: Reliability improvement + production focus
2. Load: ordis/quality-engineering/reliability-engineering (SRE patterns)
3. Load: ordis/quality-engineering/production-monitoring-and-observability (SLOs, alerting)
4. Consider: ordis/quality-engineering/chaos-engineering-principles (find weaknesses)
```

### Example 3: Microservice Testing Strategy

```
User: "How should we test our 10 microservices?"

Your routing:
1. Recognize: Distributed system testing
2. Load: ordis/quality-engineering/test-automation-strategy (overall strategy)
3. Load: ordis/quality-engineering/contract-testing (service boundaries)
4. Load: ordis/quality-engineering/e2e-testing-architecture (critical paths only)
5. Emphasize: Contracts over E2E where possible
```

### Example 4: Simple Unit Test (No Skills Needed)

```
User: "Write unit test for calculateTotal() function"

Your routing:
1. Recognize: Simple unit test, language-specific
2. Decision: No quality engineering skills needed
3. Use: Language-specific testing knowledge (e.g., pytest for Python)
```

---

## Summary

**This skill maps quality engineering tasks → specific skills to load.**

1. Identify testing type (E2E, performance, chaos, automation)
2. Consider phase (pre-prod vs production)
3. Consider architecture (monolith vs microservices)
4. Use decision tree to find applicable skills
5. Load core skills for universal needs
6. Add extension skills for specialized contexts
7. Cross-reference language-specific skills when needed
8. Don't load skills when basic testing knowledge suffices

**Meta-rule**: When in doubt about testing strategy, start with `test-automation-strategy` - it provides the foundation for all other testing decisions.

**Ordis Principle**: Quality engineering is the bulwark against chaos. Every test is a defensive layer, every monitor a watchful sentinel, every gate a barrier against disorder.
