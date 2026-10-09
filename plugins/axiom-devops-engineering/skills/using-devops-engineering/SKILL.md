---
name: using-devops-engineering
description: "Use when building or fixing delivery and operations: artifact identity, rollout health, rollback/restore evidence, infrastructure and runtime reliability."
---

# DevOps Engineering

Connect a verified change to an observed running service and recoverable operations. Model knowledge cannot establish the live artifact, permissions, health or restore state; retrieve that evidence.

## Workflow

1. Identify environment, service, artifact/revision, permissions and current operational state. Read existing pipelines/runbooks and preserve unrelated resources.
2. Define the deployment or operational outcome, health signals, failure threshold, recovery plan and actual approval boundary.
3. Inspect the relevant mechanism: reproducible build, credentials, config/state isolation, supply-chain verification, readiness, rollout, traffic or backup.
4. Make authorized changes and run focused checks. Promote the same immutable artifact where practical; test signature/attestation enforcement at the mechanism that actually enforces it.
5. Observe rollout against declared health/traffic criteria. A configured pipeline is not evidence that deployment or acceptance happened.
6. Exercise rollback/restore where required by risk. Record data/schema incompatibilities and irreversible limits; a backup's existence is not restore proof.
7. Report local validation, integration, publish/deploy and live acceptance as separate states with artifact identity and remaining uncertainty.

## Optional references

[Pipelines](ci-cd-pipeline-architecture.md), [release/rollback](release-management-and-rollback.md), [deployment strategies](deployment-strategies.md), [supply chain](devsecops-and-supply-chain.md), [containers](containerization.md), [orchestration](orchestration-and-scheduling.md), [IaC](infrastructure-as-code.md), [environments](environment-management.md), [secrets](secrets-and-configuration.md), [GitOps](gitops-and-delivery-automation.md), [observability](observability-and-monitoring.md), [reliability](reliability-engineering.md), [incidents](incident-response-and-oncall.md), [GitHub/Azure governance recipes](platform-governance-recipes.md).

Current user authorization applies; do not add a permission checkpoint merely because the work is operational. Retrieve current official documentation for version-sensitive commands/configuration and state unavailable evidence rather than refusing to progress without a sheet.

## Working contract

Use the smallest deliverable that makes the decision or change reviewable. Existing project records can satisfy these fields; do not create duplicate documents. Read only references needed for the unresolved question. Ordinary work does not require loading a specialist, delegating, or asking a routing question.

Use tools and current project evidence where available. Distinguish observed facts, inferences and unknowns. Report material findings with a source, consequence and proposed action; report a supported clean result when appropriate. State checks run, checks omitted and residual uncertainty without mandatory report sections. Honor current user authorization and applicable project policy; ask only when a missing decision materially blocks progress.
