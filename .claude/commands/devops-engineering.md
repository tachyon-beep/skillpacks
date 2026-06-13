---
description: Routes DevOps and platform-engineering questions to specialist reference sheets (CI/CD, deployment, IaC, containers, orchestration, observability, incident response, release/rollback, secrets, environments, DevSecOps, GitOps, reliability)
---

# DevOps Engineering Routing

**This router directs you to specialized DevOps and platform-engineering reference sheets. Each sheet provides deep, production-grade expertise in one slice of the path from a commit to a healthy production system.**

Use the `using-devops-engineering` skill from the `axiom-devops-engineering` plugin to route DevOps questions to the appropriate specialist:

- **ci-cd-pipeline-architecture** - Pipeline stages, verification gates, fast feedback, fan-out/fan-in
- **deployment-strategies** - Blue/green, canary, rolling, zero-downtime cutover
- **release-management-and-rollback** - Release trains, feature flags, rollback triggers and blast-radius control
- **infrastructure-as-code** - Terraform/Pulumi, state, drift, module design
- **containerization** - Image build hygiene, layer caching, multi-stage, minimal/distroless
- **orchestration-and-scheduling** - Kubernetes runtime, probes, resources, scheduling
- **observability-and-monitoring** - Metrics/logs/traces, SLI/SLO, alerting
- **incident-response-and-oncall** - On-call, severity, escalation, post-mortems
- **secrets-and-configuration** - Secret stores, rotation, config layering, environment injection
- **environment-management** - Environment parity, promotion, ephemeral environments
- **devsecops-and-supply-chain** - SBOM, signing, provenance, dependency and pipeline hardening
- **gitops-and-delivery-automation** - Declarative delivery, reconciliation, pull-based deploys
- **reliability-engineering** - SRE practice, error budgets, capacity, resilience

**Commands:** `/design-deployment`, `/review-pipeline`, `/audit-deployment`
**Agents:** deployment-strategist, pipeline-reviewer

**Cross-references to other packs:**
- Security architecture → `ordis-security-architect`
- Audit-grade provenance trails → `axiom-audit-pipelines`
- Quality/test pipelines → `ordis-quality-engineering`
- Process maturity & governance → `axiom-sdlc-engineering`
