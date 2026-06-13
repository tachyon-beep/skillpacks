---
description: Audit a service or environment for production readiness across observability/SLOs, rollback, secrets, IaC/drift, incident runbooks, and supply-chain gates - severity-rated findings each citing a reference sheet
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[service_or_environment_to_audit]"
---

# Audit Deployment Command

You are auditing the **production readiness** of `$ARGUMENTS` (a service or environment). This is not a pipeline-stage review (see `/review-pipeline`) and not a forward design (see `/design-deployment`). It is a point-in-time assessment: **is this thing safe to be in production right now, and what will hurt us when it breaks at 3am?**

## Core Principle

**Production readiness is not "it deploys". It is: when this fails, can we see it, undo it, and recover from it - without heroics, without leaking secrets, and without rebuilding the universe from memory.**

Every finding you emit MUST be severity-rated and MUST cite the reference sheet that grounds it. A finding with no sheet citation is an opinion, not an audit result.

## Scope: Six Readiness Dimensions

Audit across exactly these six dimensions. Each maps to one or more sheets in `axiom-devops-engineering/skills/using-devops-engineering/`.

| # | Dimension | Primary sheet(s) | The question it answers |
|---|-----------|------------------|--------------------------|
| 1 | Observability & SLOs | `observability-and-monitoring.md` | When users hurt, do we know before they tweet? |
| 2 | Rollback path | `release-management-and-rollback.md`, `deployment-strategies.md` | Can we undo this in under a minute, by construction? |
| 3 | Secrets handling | `secrets-and-configuration.md` | Are credentials vaulted, rotatable, and out of images/logs/repos? |
| 4 | IaC & drift | `infrastructure-as-code.md` | Is the running state what the code says, and can we prove it? |
| 5 | Incident runbooks | `incident-response-and-oncall.md`, `reliability-engineering.md` | When it breaks, is there a command structure and a runbook, or improvisation? |
| 6 | Supply-chain gates | `devsecops-and-supply-chain.md` | Can we prove how the artifact was built, who signed it, and what is inside it? |

Read the relevant sheet before judging a dimension. Do not audit from memory - the sheets carry the calibrated 2026 specifics (OTLP, SLSA provenance, expand/contract, dynamic short-lived credentials).

## Audit Process

### Step 1 - Establish the target and depth

If `$ARGUMENTS` is empty or ambiguous, use **AskUserQuestion** to pin down:
- Whether you are auditing a **single service** or a whole **environment** (prod vs staging).
- Whether this is a **pre-launch gate** (block release on Critical) or a **standing hygiene audit** (prioritise remediation backlog).
- Where the evidence lives: repo paths for IaC, pipeline definitions, the observability stack, the secrets backend.

Do not invent infrastructure. If you cannot find evidence for a dimension, that is itself a finding (see "Absence is a finding" below).

### Step 2 - Gather evidence per dimension

Use **Glob/Grep/Read/Bash** (read-only) to find concrete artifacts. Suggested probes:

**1. Observability & SLOs** - cite `observability-and-monitoring.md`
- Search for dashboard/alert config: `grep -rEi 'alert|slo|sli|error.budget|prometheus|otel|opentelemetry' --include=*.yml --include=*.yaml --include=*.json`
- Check: is there an SLI tied to **user-facing** success (latency/error rate), or only host metrics (CPU/disk)? Are alerts **symptom-based** (RED/USE) or cause-based? Is there a stated, measured SLO and error budget? Does a deploy emit a version marker that dashboards show?
- Red flags: forty-panel dashboards that never answer "are users okay", alerts that auto-resolve and nobody reads, no measurable MTTR.

**2. Rollback path** - cite `release-management-and-rollback.md` and `deployment-strategies.md`
- Check artifact identity: is the deployed version an **immutable, SHA-pinned** artifact, or a moving tag (`:latest`, `:prod`)? Is it **build-once / promote-everywhere**, or rebuilt per environment?
- Check the undo: is there a kept, runnable prior version? Are there **automated rollback triggers** and release-health gates, or is rollback "redeploy the old tag and pray"? Is there a defined **roll-back vs roll-forward** decision rule?
- Check migrations: are schema changes **expand/contract** (backward-compatible), or do code and schema ship coupled so rollback corrupts data?

**3. Secrets handling** - cite `secrets-and-configuration.md`
- `grep -rEi 'password|secret|api[_-]?key|token|aws_secret|private_key' --include=*.env --include=*.yml --include=*.yaml --include=Dockerfile --include=docker-compose*` and inspect committed `.env` files.
- Check: secrets in a vault / cloud KMS / secrets manager vs hardcoded? Baked into image layers? Printed in CI logs? **Rotatable** - can anyone say where a credential is used? Least-privilege service accounts? Dynamic short-lived credentials where possible?

**4. IaC & drift** - cite `infrastructure-as-code.md`
- Locate IaC: `*.tf`, `*.tofu`, Pulumi, CloudFormation. Check for **remote state with locking** (not laptop-local), a `plan`/`apply` review gate, and whether anything is provisioned by hand / console-clicked.
- If safe and the user consents, a read-only `terraform plan`/`tofu plan` reveals drift directly. Otherwise flag the **absence of drift detection** as the finding.

**5. Incident runbooks** - cite `incident-response-and-oncall.md` and `reliability-engineering.md`
- Search for runbooks, on-call docs, severity definitions, postmortem records: `grep -rEi 'runbook|on.?call|sev[0-9]|severity|postmortem|incident.command|rto|rpo'`
- Check: defined severity levels? Incident-command roles (who is in charge)? Per-alert runbooks vs dashboard-thrashing? Blameless postmortems with tracked corrective actions? Stated RTO/RPO and a **restore that has actually been tested** (a backup proven only by existing is not a backup)?

**6. Supply-chain gates** - cite `devsecops-and-supply-chain.md`
- Inspect pipeline definitions for **SAST/DAST/SCA** that **block** (not warn), dependency pinning, **SBOM** generation, artifact **signing**, and build **provenance/SLSA**. Check runner privilege - does a CI runner hold cloud-owner credentials? Do self-hosted runners execute untrusted PRs?
- Red flags: unsigned artifacts, unscanned dependencies in prod, no way to answer "which images are affected by this CVE".

### Step 3 - Rate each finding

Use this severity rubric consistently:

| Severity | Definition | Examples |
|----------|------------|----------|
| **Critical** | Will cause an unrecoverable incident or active breach exposure; blocks production. | Hardcoded prod secret in repo; no rollback path; unsigned artifacts with cloud-owner runner; no backups or never-restored backups. |
| **High** | Materially raises blast radius or MTTR; fix before next deploy. | No user-facing SLI/alerting; moving `:latest` tag in prod; no drift detection; no severity model or incident command. |
| **Medium** | Erodes reliability or hygiene; schedule remediation. | Cause-based alerts only; non-blocking SAST; manual `terraform apply` without review gate; coupled migrations. |
| **Low** | Improvement opportunity; no immediate exposure. | Missing deploy version marker on dashboards; runbooks present but stale; over-broad-but-scoped service account. |

**Absence is a finding.** If a dimension has no discoverable evidence (no SLOs at all, no IaC at all, no runbooks at all), record it as a finding at the severity its absence warrants - usually High or Critical - explicitly noting "no evidence found" rather than silently skipping the dimension.

### Step 4 - Emit the report

Write the report to a file (e.g. `deployment-readiness-audit-<target>.md`) using **Write**, and also summarise inline. Use this structure:

```markdown
## Deployment Readiness Audit: <service_or_environment>

### Verdict
**Production Ready**: Yes / No / Conditional
**Critical**: <n>  **High**: <n>  **Medium**: <n>  **Low**: <n>
**Blocking issues** (must clear before release): <list or "none">

### Dimension Scorecard

| Dimension | State | Worst severity | Sheet cited |
|-----------|-------|----------------|-------------|
| Observability & SLOs | Pass / Gaps / Absent | - | observability-and-monitoring.md |
| Rollback path | Pass / Gaps / Absent | - | release-management-and-rollback.md |
| Secrets handling | Pass / Gaps / Absent | - | secrets-and-configuration.md |
| IaC & drift | Pass / Gaps / Absent | - | infrastructure-as-code.md |
| Incident runbooks | Pass / Gaps / Absent | - | incident-response-and-oncall.md |
| Supply-chain gates | Pass / Gaps / Absent | - | devsecops-and-supply-chain.md |

### Findings

#### Critical
| Finding | Evidence (file:line / "no evidence found") | Impact | Remediation | Sheet |
|---------|--------------------------------------------|--------|-------------|-------|
| ... | ... | ... | ... | <sheet>.md |

#### High
[same columns]

#### Medium
[same columns]

#### Low
[same columns]

### Remediation Order
1. [Critical fixes - block release]
2. [High - before next deploy]
3. [Medium / Low - backlog]
```

Every row in every findings table MUST populate the **Sheet** column with the specific sheet that grounds the judgement. A finding you cannot trace to a sheet does not belong in this audit.

## Calibration and Honesty

- Report only what the evidence supports. Distinguish "verified present", "verified absent", and "could not determine" - never imply you confirmed something you only assumed.
- If a Bash probe (e.g. `terraform plan`) could mutate state or needs credentials, ask via **AskUserQuestion** before running it; default to read-only.
- State your confidence and information gaps. If the secrets backend or observability stack lives outside the repo and you could not inspect it, say so explicitly rather than scoring it Pass.

## Scope Boundaries

**This command covers:**
- Point-in-time production-readiness assessment across the six dimensions above.
- Severity-rated, sheet-cited findings with a remediation order.
- "Absence is a finding" coverage so untouched dimensions are never silently passed.

**Not covered (route elsewhere):**
- Pipeline stage completeness and CI/CD anti-patterns -> `/review-pipeline`
- Designing a new deployment/rollout strategy -> `/design-deployment`
- Deep application-level threat modeling -> `ordis-security-architect` (`/threat-model`, `/security-review`)
- Test-suite quality and coverage thresholds -> `ordis-quality-engineering` (`/audit`)
