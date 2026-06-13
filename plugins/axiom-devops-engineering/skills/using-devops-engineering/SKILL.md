---
name: using-devops-engineering
description: Use when building or fixing the path from commit to running production — CI/CD pipelines and verification gates, zero-downtime deployment, infrastructure as code, container build hygiene, Kubernetes/orchestration runtime, observability (metrics/logs/traces, SLI/SLO), incident response and on-call, release management and rollback, secrets and configuration, environment parity and promotion, DevSecOps and software supply-chain security, GitOps/declarative delivery, and SRE/reliability — routes to 13 specialist reference sheets instead of giving surface-level generic ops advice.
---

# Using DevOps Engineering Skills

## Overview

**This router directs you to specialized DevOps and platform-engineering reference sheets. Each sheet provides deep, production-grade expertise in one slice of the path from a commit to a healthy production system.**

**Core principle:** "Ship it" is not one skill. Getting code to production safely spans pipeline design, deployment strategy, infrastructure, containers, orchestration, observability, incident response, release discipline, secrets, environments, supply-chain security, GitOps, and reliability — each with its own failure modes. Routing to the right specialist beats a competent-but-broad answer that misses the gate, the rollback trigger, or the blast-radius control that actually prevents the outage.

## When to Use

Use this router when encountering:

- **Delivery pipeline**: CI/CD design, verification gates, slow feedback, a "deploy" job that is one step straight to prod
- **Getting to production**: deployment strategy, traffic shifting, canary/blue-green, "how do we undo this in under a minute"
- **Infrastructure**: provisioning, Terraform/OpenTofu/Pulumi, state, drift, declarative desired-state
- **Containers & runtime**: image hygiene, Dockerfiles, Kubernetes probes, autoscaling, resource limits, graceful shutdown
- **Operating it**: observability, SLI/SLO, alerting, incidents, on-call, postmortems, reliability/SRE
- **Discipline & safety**: release identity and rollback, secrets, environment parity, supply-chain security, GitOps

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-devops-engineering/SKILL.md`

Reference sheets like `deployment-strategies.md` are at:
  `skills/using-devops-engineering/deployment-strategies.md`

NOT at:
  `skills/deployment-strategies.md` ← WRONG PATH

When you see a link like `[deployment-strategies.md](deployment-strategies.md)`, read the file from the same directory as this SKILL.md.

---

## Quick Reference - Routing Table

| Symptom / task contains | Route to | Why |
|-------------------------|----------|-----|
| CI/CD pipeline design, "deploy" is one step, slow feedback, no merge gate, migrations break prod | [ci-cd-pipeline-architecture.md](ci-cd-pipeline-architecture.md) | Pipeline stages, verification gates, build-once artifacts, automated rollback |
| Zero-downtime deploy, canary, blue-green, rolling, traffic shifting, "undo in under a minute" | [deployment-strategies.md](deployment-strategies.md) | Progressive delivery, per-strategy verification, automated rollback triggers |
| Provisioning, Terraform/OpenTofu/Pulumi, state locking, drift, clicking the console | [infrastructure-as-code.md](infrastructure-as-code.md) | Declarative desired-state, idempotency, remote state, plan/apply review |
| Huge image, runs as root, secrets in a layer, CVE flood, slow cache, Dockerfile choices | [containerization.md](containerization.md) | Multi-stage builds, distroless/minimal bases, non-root, reproducible builds, scanning, signing |
| Pods get traffic before ready, OOM neighbors, no autoscaling, probes kill slow boots | [orchestration-and-scheduling.md](orchestration-and-scheduling.md) | Kubernetes probes, requests/limits, HPA/VPA, PDBs, graceful shutdown |
| "Site is slow" with no data, customer-tweet-first outages, alert fatigue, no MTTR, no SLO | [observability-and-monitoring.md](observability-and-monitoring.md) | Metrics/logs/traces + OpenTelemetry, SLI/SLO/error budgets, RED/USE, actionable alerts |
| Outage with nobody in charge, recurring incidents, postmortems that blame people, deploy-freeze with no trigger | [incident-response-and-oncall.md](incident-response-and-oncall.md) | Severity levels, incident command, runbooks, blameless postmortems, on-call health |
| Rebuilt per environment, ":latest" in prod, can't undo because the artifact was overwritten, rubber-stamp/freeze CAB | [release-management-and-rollback.md](release-management-and-rollback.md) | Immutable artifact identity, build-once-promote, semver/channels, rollback-vs-roll-forward |
| Hardcoded API key, password in committed .env, secret baked in image, token in CI logs, no rotation | [secrets-and-configuration.md](secrets-and-configuration.md) | Vaults/KMS, rotation, dynamic short-lived creds, config-vs-secret, 12-factor |
| Passed staging then broke prod, silent env drift, no preview/ephemeral env, manual promotion steps | [environment-management.md](environment-management.md) | Environment parity, ephemeral/preview envs, promotion flow, single config contract |
| Unsigned artifacts, unscanned deps, runner with cloud-owner creds, no SBOM, security as a final manual review | [devsecops-and-supply-chain.md](devsecops-and-supply-chain.md) | SAST/DAST/SCA gates, SBOM, signing, provenance/SLSA, runner isolation, policy-as-code |
| CI runs `kubectl apply` at the cluster, manual hotfixes, cluster/repo drift, long-lived cluster creds in CI | [gitops-and-delivery-automation.md](gitops-and-delivery-automation.md) | Git as source of truth, pull-based reconciliation (Argo CD/Flux), drift correction, CI/CD split |
| Falls over when a dependency slows, retry storms, no error budget, untested backups, no DR/RTO/RPO | [reliability-engineering.md](reliability-engineering.md) | SRE error budgets, retries/backoff, circuit breakers/bulkheads, chaos, DR, restore-proven backups |

## Commands

These slash commands drive the most common end-to-end workflows; each dispatches an SME agent and returns structured findings:

- **`/design-deployment`** — design a zero-downtime deployment strategy (strategy selection, verification gates, rollback triggers) for a service. Backed by the `deployment-strategist` agent.
- **`/review-pipeline`** — review an existing CI/CD pipeline for missing stages, anti-patterns, and production readiness. Backed by the `pipeline-reviewer` agent.
- **`/audit-deployment`** — audit a deployment/release path end-to-end (artifact identity, rollback, supply-chain attestation, environment parity) against the pack's discipline.

## Agents

- **`deployment-strategist`** — SME for choosing and hardening a deployment strategy: progressive delivery, traffic shifting, automated metric-driven rollback, and migration safety.
- **`pipeline-reviewer`** — SME for critiquing CI/CD pipelines: gate coverage, feedback speed, build-once-deploy-everywhere, and supply-chain integration.

## When NOT to Use This Pack

Route elsewhere when the question is not about getting code to and keeping it running in production:

- **Test strategy, what/how to test, flaky tests, coverage, test pyramid** → `/quality-engineering`. DevOps cares that the *gate runs and blocks*; quality engineering decides *what the tests should be*.
- **Application/system architecture, service boundaries, NFR design, tech selection, ADRs** → `/solution-architect` (forward design) or `/system-architect` (assessment). DevOps operates the system; it does not design its internal structure.
- **Security threat modeling, control design, OWASP-level review** → `/security-architect`. DevSecOps here is the supply-chain and pipeline slice; deep threat modeling belongs there.
- **Web framework / API implementation (FastAPI, Django, Express, REST/GraphQL)** → `/web-backend`.
- **Runbook / postmortem / ADR writing as prose** → `/technical-writer` for the document; this pack for the operational substance.

## How to Route

**STOP: Do not answer DevOps questions from generic ops knowledge.**

1. **Identify the slice** from the routing table — pipeline, deployment, infra, containers, orchestration, observability, incident, release, secrets, environments, supply-chain, GitOps, or reliability.
2. **State which sheet you're using**: "I'll use the `[sheet-name]` reference sheet for this."
3. **Read that sheet and apply it.** If the problem spans slices (most do), name each sheet and address them in order of blast radius.

## Red Flags - Using Generic Knowledge Instead of Specialists

If you catch yourself doing any of these, STOP and route to a sheet:

- ❌ "I'll just give a generic CI/CD example" instead of reading `ci-cd-pipeline-architecture.md`
- ❌ Recommending a deployment without naming a rollback trigger
- ❌ Suggesting "add monitoring" without SLI/SLO or actionable alerting from `observability-and-monitoring.md`
- ❌ Treating secrets, supply-chain, and config as one vague "security" answer
- ❌ "The sheet isn't available, so I'll answer from memory" — name the sheet and what it covers instead

## Rationalization Table

| Excuse | Reality |
|--------|---------|
| "It's all just DevOps, I can answer broadly" | DevOps spans 13 distinct failure-domain sheets; broad answers miss the gate or the rollback |
| "A quick pipeline snippet is enough" | Snippets without verification gates and rollback are how incidents ship |
| "Monitoring and reliability are the same thing" | Observability tells you something broke; reliability engineering stops one failure cascading |
| "Secrets are just config" | Config is what changes per env; secrets need rotation, vaults, and short-lived credentials |
| "GitOps is just CI deploying to the cluster" | GitOps is pull-based reconciliation with git as source of truth — the opposite of CI push |

## Example Routing

**User**: "Our deploy ships all pods at once and a bad release means an outage until someone reverts and rebuilds."

**Your response**:
"Two slices here:
1. Zero-downtime strategy and fast rollback → I'll use [deployment-strategies.md](deployment-strategies.md).
2. The fact that rollback means a rebuild points at release identity → I'll also use [release-management-and-rollback.md](release-management-and-rollback.md) so the old artifact is immutable and re-deployable.
Let me start with the deployment strategy..."

**User**: "First we hear of an outage is a customer tweet, and our postmortems blame whoever deployed."

**Your response**:
"1. No leading signal → [observability-and-monitoring.md](observability-and-monitoring.md) for SLI/SLO and symptom-based alerting.
2. Blame-naming postmortems → [incident-response-and-oncall.md](incident-response-and-oncall.md) for blameless postmortems and corrective-action tracking.
Starting with the observability gap..."

## Why This Matters

**Without routing**: A competent-sounding ops answer that ships without a gate, a rollback trigger, or an SLO — and causes the incident it was meant to prevent.

**With routing**: Deep, production-grade guidance that names the verification gate, the rollback path, and the blast-radius control for the specific slice in front of you.

---

## DevOps Engineering Reference Sheet Catalog

After routing, read the appropriate reference sheet for detailed guidance:

### Delivery Pipeline & Deployment

1. [ci-cd-pipeline-architecture.md](ci-cd-pipeline-architecture.md) — CI/CD pipeline architecture with verification gates and rollback; build-once artifacts, fast feedback, migration safety, supply-chain attestation in the pipeline
2. [deployment-strategies.md](deployment-strategies.md) — Zero-downtime deployment strategies; canary/blue-green/rolling/dark-launch, traffic shifting, per-strategy verification and automated rollback triggers
3. [release-management-and-rollback.md](release-management-and-rollback.md) — Release management, versioning, and rollback; immutable artifact identity, build-once-promote-everywhere, semver/channels, rollback-vs-roll-forward, change management without theatre
4. [gitops-and-delivery-automation.md](gitops-and-delivery-automation.md) — GitOps and declarative continuous delivery; git as source of truth, pull-based reconciliation (Argo CD/Flux), drift correction, the CI/CD split

### Infrastructure & Runtime

5. [infrastructure-as-code.md](infrastructure-as-code.md) — Infrastructure as Code discipline; declarative desired-state, idempotency, remote state and locking, drift detection, module design (Terraform/OpenTofu/Pulumi)
6. [containerization.md](containerization.md) — Container image hygiene and build discipline; multi-stage builds, distroless/minimal bases, non-root, BuildKit caching, reproducible builds, scanning, signing, registry hygiene
7. [orchestration-and-scheduling.md](orchestration-and-scheduling.md) — Kubernetes/orchestration runtime patterns; liveness/readiness/startup probes, resource requests/limits, HPA/VPA, controlled rollouts, PodDisruptionBudgets, graceful shutdown
8. [environment-management.md](environment-management.md) — Environment parity and promotion; dev/staging/prod parity, ephemeral/preview environments, promotion flow, config drift and a single config contract, data handling across environments

### Operating & Reliability

9. [observability-and-monitoring.md](observability-and-monitoring.md) — Metrics, logs, traces, SLI/SLO; three pillars + OpenTelemetry/OTLP, error-budget design, RED/USE methods, symptom-based actionable alerting, killing alert fatigue
10. [incident-response-and-oncall.md](incident-response-and-oncall.md) — Incident response, on-call, and blameless postmortems; severity classification, incident command, runbooks, corrective-action tracking, on-call rotation health and toil reduction
11. [reliability-engineering.md](reliability-engineering.md) — SRE principles and resilience; error budgets, retries with backoff/jitter, circuit breakers/bulkheads, chaos engineering, disaster recovery with measured RTO/RPO, restore-proven backups

### Security & Configuration

12. [secrets-and-configuration.md](secrets-and-configuration.md) — Secrets management and configuration; vaults/KMS, rotation, dynamic short-lived credentials, config-vs-secret separation, 12-factor config, keeping secrets out of images and repos
13. [devsecops-and-supply-chain.md](devsecops-and-supply-chain.md) — DevSecOps and software supply-chain security; SAST/DAST/SCA gates, SBOM, artifact signing, provenance/SLSA, runner isolation, policy-as-code
