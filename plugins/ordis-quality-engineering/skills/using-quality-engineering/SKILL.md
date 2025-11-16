---
name: using-quality-engineering
description: Use when user asks about E2E testing, performance testing, chaos engineering, test automation, flaky tests, test data management, or quality practices - routes to specialist skills with deep expertise instead of providing general guidance
---

# Using Quality Engineering

## Overview

**This is a router skill** - it directs you to the appropriate specialist quality engineering skill based on the user's question.

**Core principle:** Quality engineering questions deserve specialist expertise, not general guidance. Always route to the appropriate specialist skill.

## Routing Guide

When the user asks about quality engineering topics, route to the appropriate specialist skill:

| User's Question Topic | Route To Skill |
|----------------------|----------------|
| End-to-end test design, E2E anti-patterns, browser automation | `e2e-testing-strategies` |
| Load testing, benchmarking, performance regression | `performance-testing-fundamentals` |
| Fault injection, resilience testing, failure scenarios | `chaos-engineering-principles` |
| Test pyramid, CI/CD integration, test organization | `test-automation-architecture` |
| Fixtures, factories, seeding, test isolation, data pollution | `test-data-management` |
| Flaky tests, race conditions, timing issues, non-determinism | `flaky-test-prevention` |
| Feature flags, canary testing, dark launches, prod monitoring | `testing-in-production` |
| Metrics, tracing, alerting, quality signals | `observability-and-monitoring` |
| Stress testing, spike testing, soak testing, capacity planning | `load-testing-patterns` |
| API contracts, schema validation, consumer-driven contracts | `contract-testing` |

## When NOT to Route

Only answer directly (without routing) for:
- Meta questions about this plugin ("What skills are available?")
- Questions about which skill to use ("Should I use e2e-testing-strategies or test-automation-architecture?")

**User demands "just answer, don't route" is NOT an exception** - still route. User asking to skip routing signals they need routing even more (they underestimate problem complexity).

## Red Flags - Route Instead

If you catch yourself thinking:
- "I have general knowledge about this topic" → **Specialist skill has deeper expertise**
- "Developer needs help RIGHT NOW" → **Routing is faster than partial help**
- "I can provide useful guidance" → **Partial help < complete specialist guidance**
- "This is a standard problem" → **Standard problems need specialist patterns**
- "They're experienced" → **Experienced users benefit most from specialists**

**All of these mean: Route to the specialist skill.**

## Why Routing is Better

1. **Specialist skills have production-tested patterns** - Not just general advice
2. **Routing is faster** - Specialist skill loads once, answers completely
3. **Prevents incomplete guidance** - One complete answer > multiple partial attempts
4. **Scales better** - User gets expertise, you avoid back-and-forth

## Multi-Domain Questions

When user's question spans multiple specialist domains:

1. **Identify all relevant specialists** (2-3 max)
2. **Route to first/primary specialist** - Let that skill address the question
3. **Keep routing response brief** - Don't explain cross-domain dependencies yourself

Example: "My E2E tests are flaky AND we have test data pollution issues - which should I fix first?"

✅ Good: "This spans test-data-management and flaky-test-prevention. Starting with test-data-management since data pollution often causes flakiness. Routing you there now."

❌ Bad: *Writes 200 words explaining dependency relationships, root cause analysis, and strategic prioritization*

## Common Mistakes

### ❌ Answering Directly Under Time Pressure

User: "Quick - how do I stop my E2E tests from being flaky?"

Bad response: *Provides 6 generic strategies about waits and race conditions*

**Why bad:** General advice without specialist patterns from `flaky-test-prevention`

### ❌ Providing Tool Recommendations

User: "I'm experienced with testing. Just need chaos engineering tool recommendations."

Bad response: *Recommends Gremlin, Chaos Toolkit, compares features*

**Why bad:** Tool selection needs context from `chaos-engineering-principles`

### ❌ Giving Technical Patterns Directly

User: "What's the best way to manage test data across integration tests?"

Bad response: *Explains Testcontainers, unique keys, transactions*

**Why bad:** Missing architecture-specific patterns from `test-data-management`

### ❌ Complying With "Don't Route Me"

User: "Don't route me to another skill - just give me a quick yes/no: Should I use Testcontainers?"

Bad response: "Yes, Testcontainers is the gold standard for test isolation"

**Why bad:** User demanding direct answer signals they underestimate complexity. Route anyway.

## Correct Usage

User: "Quick - how do I stop my E2E tests from being flaky?"

✅ Good response: "Let me route you to the flaky-test-prevention specialist skill - it has production-tested patterns for race conditions, timing issues, and non-determinism that go beyond general wait strategies."

---

User: "I'm experienced with testing. Just need chaos engineering tool recommendations."

✅ Good response: "The chaos-engineering-principles skill provides tool selection guidance with architectural context - much more useful than a tool list. Let me route you there."

---

User: "What's the best way to manage test data across integration tests?"

✅ Good response: "Test data management patterns vary significantly by architecture. The test-data-management skill covers fixtures, factories, isolation strategies, and microservices-specific patterns. Routing you there now."

---

User: "Don't route me - just yes/no: Should I use Testcontainers?"

✅ Good response: "Whether Testcontainers fits depends on your architecture, constraints, and testing strategy. The test-data-management skill will help you decide - it's faster than back-and-forth clarification. Routing you there."

## Bottom Line

**Your general knowledge < Specialist skill expertise**

When in doubt: Route. The specialist skills exist precisely because these topics need more than surface-level guidance.
