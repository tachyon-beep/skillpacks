---
description: Design a zero-downtime deployment and release strategy with verification gates and automated, metric-driven rollback - grounded in the using-devops-engineering reference sheets and the deployment-strategist agent
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[application_or_service_name]"
---

# Design Deployment Command

You are designing a deployment and release strategy for **$ARGUMENTS** that ships behind a strategy keeping the old version serving until the new one is proven, limits blast radius, and reverts faster than a human can be paged.

## Core Principle

**"Deploy to production" is not a single step - it is a sequence of gates, health checks, gradual rollouts, and automated rollback triggers. If you cannot answer "what flips us back, and how long does it take," you do not have a deployment strategy - you have a big-bang deploy with extra steps.**

## Load the discipline first

This command is a thin entry point. The actual engineering depth lives in the `using-devops-engineering` reference sheets - READ the relevant ones before designing, and ground every recommendation in them rather than improvising:

- **`deployment-strategies.md`** (primary) - the strategy decision (rolling / blue-green / canary / feature-flag dark launch), per-strategy mechanics and verification, automated metric-driven rollback triggers, progressive delivery with Argo Rollouts and Flagger, Gateway API traffic-shifting, and expand/contract migrations. This sheet owns *which mechanism flips traffic and when it flips back*.
- **`release-management-and-rollback.md`** (primary) - artifact identity and immutability (build once, promote everywhere), build-once/promote-unchanged across environments, semantic versioning and release channels, change management without theatre, release health gates, and the **rollback-vs-roll-forward** decision. This sheet owns *artifact identity, promotion, and the revert decision* - rollback is only fast if the previous immutable artifact still exists and you know its digest.
- **`environment-management.md`** - environment parity, ephemeral/preview environments, the promotion flow, and config-drift control, so a change proven in a lower environment is actually proven for production.
- **`observability-and-monitoring.md`** - the SLIs/SLOs and symptom-based signals that a canary's automated rollback analysis depends on. A canary that no automation watches is just a slow big-bang.

For non-trivial designs, **dispatch the `deployment-strategist` agent** (in this pack's `agents/`). It follows the SME Agent Protocol (`meta-sme-protocol:sme-agent-protocol`): it READs existing infrastructure configs and deployment scripts first, then returns a design with explicit Confidence, Risk, Information-Gaps, and Caveats sections. Use the agent when the service is high-stakes, when infrastructure is unfamiliar, or when strategy selection is contested; handle the design inline for simple, well-understood cases.

## Information Gathering

Before designing, determine (use `AskUserQuestion` for gaps you cannot read from the repo):

1. **Infrastructure**: Kubernetes, containers, VMs, serverless?
2. **Traffic volume**: Requests per second, peak load? (Enough for a statistically meaningful canary?)
3. **Risk tolerance**: Can you afford double infrastructure temporarily? Is 30 seconds of badness acceptable, or must revert be instant?
4. **Observability**: Is there a reliable automated success signal (error rate / latency / SLO)? This gates whether a metric-driven canary is even possible - see `observability-and-monitoring.md`.
5. **Artifact discipline**: Is the release one immutable, content-addressed artifact built once and promoted, or rebuilt per environment? (Rollback speed depends on this - see `release-management-and-rollback.md`.)
6. **Database**: Stateful migrations required? Are they expand/contract (backward-compatible) so code and schema can roll back independently?
7. **Current state**: Existing CI/CD, deployment process, environment parity?

## Strategy Selection

Choose by what the failure costs and what you can observe - not by what is fashionable. Apply the decision procedure from `deployment-strategies.md`:

1. **Can you observe success automatically?** No reliable SLO signal -> you cannot run a metric-gated canary safely. Use blue-green (smoke-test gate) or rolling with strong health checks until you have telemetry.
2. **Is instant, total revert the priority?** (payments, auth, anything where seconds of badness are unacceptable) -> **blue-green** (one traffic flip back to known-good).
3. **High-traffic with good SLO data, and gradual exposure worth the complexity?** -> **canary** via Argo Rollouts or Flagger.
4. **Ship code now, turn it on later or for a subset?** -> **feature flags / dark launch**, layered on top of whichever rollout moves the binary.
5. **Otherwise** (cost-sensitive, lower stakes) -> **rolling** with proper readiness gates and `maxUnavailable: 0`.

| Strategy | Rollback speed | Blast radius before proof | Infra cost | Needs good metrics? | Best for |
|----------|---------------|---------------------------|-----------|---------------------|----------|
| **Rolling** | Slow (re-roll instances) | One batch at a time | 1x (+surge) | No | Internal tools, low-stakes stateless services |
| **Blue-green** | Instant (flip traffic) | All-or-nothing at cutover | 2x during deploy | Smoke tests suffice | Critical systems where instant revert matters most |
| **Canary** | Fast (route to stable) | A small % of real traffic | ~1.1x | **Yes** - rollback is metric-driven | High-traffic services with real SLO telemetry |
| **Feature flag / dark launch** | Instant (toggle, no deploy) | Per-user / per-cohort | 1x | Yes for guarded rollout | Decoupling *release* from *deploy*; kill-switches; A/B |

These compose: the mature pattern is rolling/blue-green to move the *binary* + feature flags to control *exposure* + canary analysis to *gate progression*. Pull the per-strategy mechanics (selector flips, traffic weights, surge/unavailable settings, Argo Rollouts / Flagger config) directly from `deployment-strategies.md` rather than reinventing them here.

## Release identity and rollback (from release-management-and-rollback.md)

Strategy selection only delivers safety if the release itself is reversible by construction. Enforce:

- **Build once, promote everywhere.** One immutable, content-addressed artifact (`@sha256:` digest, version+hash wheel, checksummed binary) moves unchanged through every environment. Mutable tags (`:latest`, `:prod`) are pointers, never the thing you deploy or roll back to.
- **Reversible by construction.** Retain the previous N artifacts beyond your rollback window; record provenance (commit SHA, artifact digest, SBOM, who promoted, when) in a queryable release ledger. "Roll back" must resolve to "deploy artifact `@sha256:<prev>`" - no rebuild, no `git revert`, no fingers crossed.
- **Rollback vs roll-forward** is a deliberate, pre-agreed decision, not a 40-minute incident argument. Use the decision framing in `release-management-and-rollback.md`.

## Automated rollback triggers

Rollback must be metric-driven and faster than a page. Ground thresholds in your SLIs (`observability-and-monitoring.md`):

```yaml
rollback_triggers:
  immediate:
    - health_check_failures >= 3
    - error_rate > 10%
  gradual:
    - error_rate > 5% for 3 minutes
    - p99_latency > 2x baseline for 5 minutes
    - pod_restarts > 3 in 5 minutes

rollback_action:
  blue_green: switch_to_previous_version       # flip selector back
  canary:     route_100%_to_stable             # Argo Rollouts / Flagger analysis abort
  rolling:    redeploy_previous_artifact        # deploy retained @sha256:<prev>
```

## Database Migration Strategy

Schema and code must be able to roll back independently. Use expand/contract (three-phase), per `deployment-strategies.md` and `release-management-and-rollback.md`:

```yaml
migration_approach: expand_contract
phase_1_expand:
  description: Add new schema, backward compatible
  changes: [add nullable columns, create new tables, add indexes CONCURRENTLY]
  rollback: drop_new_elements
phase_2_migrate:
  description: Deploy code that writes both, reads new with fallback
  rollback: revert_code   # old schema still present, safe
phase_3_contract:
  description: Drop old schema
  timing: after_phase_2_stable_and_irreversible_window_passed
```

## Output Format

```markdown
# Deployment Design: [Application Name]

## Context
**Infrastructure**: [K8s/Containers/VMs/Serverless]
**Traffic**: [Requests/second] | **Risk tolerance**: [High/Medium/Low]
**Observability**: [SLO signal available? Y/N] | **Artifact discipline**: [immutable/rebuilt-per-env]
**Current State**: [Existing process, environment parity]

## Chosen Strategy
**Strategy**: [Rolling / Blue-Green / Canary / Feature-flag]
**Rationale**: [Why, per the deployment-strategies decision procedure]

## Pipeline & Promotion
[Build-once artifact, promotion across environments, gate sequence]

## Deployment Configuration
[Strategy-specific config pulled from deployment-strategies.md]

## Verification Gates
### Staging
- [ ] Health endpoint 200  - [ ] Smoke tests pass  - [ ] No error-rate increase
### Production
- [ ] Health checks pass  - [ ] Error rate < 1%  - [ ] Latency within baseline  - [ ] Business metrics normal

## Automated Rollback
[Triggers + per-strategy rollback action; previous artifact digest retained]

## Rollback vs Roll-Forward
[Pre-agreed decision criteria]

## Database Migrations
**Approach**: [Expand/contract / Zero-downtime / N/A] + steps

## Implementation Checklist
- [ ] Health check endpoint implemented
- [ ] Immutable artifact retained beyond rollback window
- [ ] Staging parity verified (see environment-management.md)
- [ ] SLIs/SLOs wired for rollback analysis (see observability-and-monitoring.md)
- [ ] Rollback tested in staging
- [ ] On-call notified of deployment process
```

If you dispatched the `deployment-strategist` agent, append its Confidence / Risk / Information-Gaps / Caveats sections verbatim.

## Cross-Pack Discovery

```python
import glob
# Observability / SLOs for canary analysis and monitoring
if glob.glob("plugins/ordis-quality-engineering/plugin.json"):
    print("Available: ordis-quality-engineering for test gates and observability patterns")
# API deployment specifics
if glob.glob("plugins/axiom-web-backend/plugin.json"):
    print("Available: axiom-web-backend for API deployment patterns")
```

## Scope Boundaries

**This command covers:**
- Deployment strategy selection (rolling / blue-green / canary / feature-flag)
- Release identity, immutability, and promotion
- Verification gates and automated rollback trigger design
- Rollback-vs-roll-forward framing
- Expand/contract database migration approach

**Not covered:**
- Pipeline review (use `/review-pipeline`)
- Infrastructure provisioning / IaC (see `infrastructure-as-code.md`)
- Application code changes
