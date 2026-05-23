---
description: Quality engineering methodology - E2E, API, integration, performance, chaos, flaky tests, observability, mutation testing, coverage gap analysis, modern supply-chain (Trivy/OSV/Syft/Cosign/SLSA). Routes to 21 specialist reference sheets, 5 commands, 3 SME agents.
---

# Quality Engineering Routing

**Quality is engineering discipline, not a final-stage filter. Tests are designed for specific failure modes; flakiness is signal, not noise; quality gates that don't gate are decoration. For threat modeling and security control design use `/security-architect`; for deployment strategy and CI/CD platform mechanics use `/axiom-devops-engineering`.**

Use the `using-quality-engineering` skill from the `ordis-quality-engineering` plugin to route to the right specialist sheet. Content authority lives in `plugins/ordis-quality-engineering/skills/using-quality-engineering/SKILL.md` - this wrapper is a thin pointer.

## When to Use

- Setting up or improving test infrastructure
- Debugging flaky or unreliable tests
- Designing testing strategy for new features
- Performance testing, load testing, capacity planning
- Production reliability, observability, SLI/SLO design
- Security testing integration (SAST, dependency scanning, fuzzing)

**Don't use** for: pure code architecture (`/system-architect`), security threat modeling (`/security-architect`), deployment pipeline mechanics (`/axiom-devops-engineering`), or UX research.

## Sheets

### Test Fundamentals
- **test-isolation-fundamentals** - test independence, idempotence, order-independence, isolation patterns

### API & Integration
- **api-testing-strategies** - REST/GraphQL testing, request validation, API mocking
- **integration-testing-patterns** - database testing, test containers, component integration
- **contract-testing** - consumer-driven contracts, schema validation, Pact

### End-to-End & UI
- **e2e-testing-strategies** - browser automation, E2E anti-patterns, flow prioritization
- **visual-regression-testing** - screenshot comparison, responsive testing

### Performance & Load
- **performance-testing-fundamentals** - benchmarking, performance regression, baselines
- **load-testing-patterns** - stress, spike, soak testing, capacity planning (k6/JMeter/Gatling)

### Test Quality & Maintenance
- **quality-metrics-and-kpis** - coverage delta, quality dashboards, CI/CD gates, vanity-metric avoidance
- **test-maintenance-patterns** - refactoring, page objects, reducing test debt
- **mutation-testing** - test effectiveness, mutation score (Stryker/PITest/mutmut)

### Static Analysis & Security Testing
- **static-analysis-integration** - SAST tools, ESLint, Pylint, quality gates
- **dependency-scanning** - SCA + supply chain (Trivy/OSV-Scanner/Snyk/Dependabot, SBOMs via Syft/CycloneDX/SPDX, Sigstore/Cosign, SLSA provenance)
- **fuzz-testing** - random input testing, security vulnerabilities

### Advanced Techniques
- **property-based-testing** - Hypothesis, fast-check, invariant testing

### Production & Monitoring
- **testing-in-production** - feature flags, canary, dark launches
- **observability-and-monitoring** - metrics, tracing, alerting, SLIs/SLOs, RED/USE/Four Golden Signals
- **chaos-engineering-principles** - fault injection, resilience testing (LitmusChaos/Chaos Mesh/Gremlin/AWS FIS)

### Test Infrastructure
- **test-automation-architecture** - test pyramid, CI/CD integration, progressive testing
- **test-data-management** - fixtures, factories, seeding, data isolation
- **flaky-test-prevention** - race conditions, timing, non-determinism

## Commands

- `/ordis-quality-engineering:audit` - quality metrics audit (coverage, flakiness rate, pass rate, build time) with actionable thresholds
- `/ordis-quality-engineering:diagnose-flaky` - systematic flaky-test decision tree, root-cause-before-retry
- `/ordis-quality-engineering:setup-pipeline` - CI/CD testing pipeline with stages (pre-push/PR/merge/nightly/pre-deploy)
- `/ordis-quality-engineering:analyze-pyramid` - test distribution across unit/integration/E2E, detect inverted pyramid
- `/ordis-quality-engineering:analyze-test-gaps` - map codebase to tests, risk-based prioritization of untested critical code

## Agents

- `coverage-gap-analyst` - reads source and tests, surfaces untested critical paths with confidence/risk
- `flaky-test-diagnostician` - diagnoses intermittent failures; rejects retry-first shortcuts; hands off to `axiom-python-engineering` for pytest-specific syntax
- `test-suite-reviewer` - audits test files for assertion-free tests, sleep-based waits, hidden dependencies, wrong test level, shared mutable state; severity-scored findings

All three agents follow the SME Agent Protocol with Confidence/Risk/Information Gaps/Caveats sections.

## Cross-references

- CI/CD platform mechanics and deployment strategy → `/axiom-devops-engineering`
- Security control design and threat modeling → `/security-architect`
- Pytest-specific syntax and Python test idioms → `/python-engineering`
- Code architecture and refactoring discipline → `/system-architect`
- LLM output evaluation methodology → `/llm-specialist`
