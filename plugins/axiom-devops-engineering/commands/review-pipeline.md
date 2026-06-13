---
description: Audit a CI/CD pipeline against the devops-engineering reference sheets - dispatches the pipeline-reviewer agent and returns severity-rated findings, each citing the sheet that resolves it
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[pipeline_file_or_directory]"
---

# Review Pipeline Command

You are auditing a CI/CD pipeline for completeness, supply-chain integrity, and post-deploy observability. Your role is to locate the pipeline artifacts, dispatch the `pipeline-reviewer` agent, and present severity-rated findings where **every finding names the reference sheet that resolves it**.

## Invocation path

`/review-pipeline` is a Claude Code slash command. The command does not perform the audit itself — it dispatches the `pipeline-reviewer` agent via the `Task` tool, then surfaces that agent's findings and writes them to disk. Readers seeing this slash command invoked should expect: command locates the pipeline files → command hands them to the agent → agent walks the audit checks against the three reference sheets → command presents severity-rated, sheet-cited results. For forward design of a new pipeline or deployment strategy (rather than critique of an existing one), use `/design-deployment`.

## Core Principle

**Accuracy over comfort. Evidence over opinion.**

"Deploy to production" is not a single step — it is a sequence of gates, supply-chain attestations, progressive rollouts, and metric-driven rollback triggers, watched by instrumentation that can answer "are users okay" within minutes. A rubber-stamp review is worse than none: it launders an incident-prone pipeline as production-ready. Name what is wrong, with `file:line` evidence, and point at the sheet that fixes it.

## The three reference sheets this audit is graded against

Every finding must cite the sheet (and ideally the section) that resolves it. The sheets live at `plugins/axiom-devops-engineering/skills/using-devops-engineering/`.

| Sheet | Covers | Audit dimension |
|-------|--------|-----------------|
| `ci-cd-pipeline-architecture.md` | Build-once digest-pinned artifacts, the test pyramid in CI, staging↔prod parity, progressive delivery (canary/blue-green), metric-driven auto-rollback, expand/contract migrations, IaC from the pipeline, secrets via OIDC | **Delivery safety** — gates, promotion, rollback |
| `devsecops-and-supply-chain.md` | SAST/DAST/SCA as PR gates, locked + hash-pinned deps, SBOM + VEX, SLSA provenance, cosign keyless signing + admission enforcement, least-privilege OIDC runners, fork-PR isolation, policy-as-code | **Supply-chain integrity** — what shipped, who signed it, what is inside it |
| `observability-and-monitoring.md` | The three pillars + OpenTelemetry/OTLP, SLI/SLO/error-budget, RED + USE, symptom-based actionable alerting, deploy annotation + rollout gating, dashboards that answer "are users okay", killing alert fatigue | **Observability** — can you detect and localize an incident before a customer does |

## Preconditions

Locate the pipeline definition. The command accepts an optional path; if none is supplied, scan the common locations.

```bash
# Argument-supplied path, or scan the repo for pipeline definitions.
TARGET="${ARGUMENTS:-}"

if [ -n "${TARGET}" ]; then
  ls -la "${TARGET}" 2>/dev/null
else
  ls -la .github/workflows/*.yml .github/workflows/*.yaml 2>/dev/null   # GitHub Actions
  ls -la .gitlab-ci.yml 2>/dev/null                                     # GitLab CI
  ls -la Jenkinsfile 2>/dev/null                                        # Jenkins
  ls -la .circleci/config.yml 2>/dev/null                               # CircleCI
  ls -la azure-pipelines.yml 2>/dev/null                                # Azure DevOps
fi
```

Also locate the artifacts the supply-chain and observability dimensions depend on — their absence is itself a finding:

```bash
ls -la **/Dockerfile* 2>/dev/null                  # build stage to audit
ls -la **/*.rego policy/ kyverno/ 2>/dev/null       # policy-as-code / admission
ls -la **/otel*.y*ml **/prometheus*.y*ml 2>/dev/null # instrumentation + SLO rules
```

**Stop conditions:**

- If no pipeline definition is found at the target or in the common locations, stop and report: `No pipeline definition found. Pass a path, or expected one of: .github/workflows/*.yml, .gitlab-ci.yml, Jenkinsfile, .circleci/config.yml, azure-pipelines.yml.`
- If a pipeline exists but there is no Dockerfile / build artifact, no policy/admission config, and no instrumentation config, proceed but warn: the supply-chain and observability dimensions will be audited largely as **absences** (which are valid, usually Critical/High, findings). Record this in Information Gaps.

## Protocol

### Step 1 — Identify scope

Determine what is present:

- **Full pipeline + build + policy + instrumentation** → full three-dimension audit.
- **Pipeline only** (no build/policy/instrumentation config) → audit proceeds; supply-chain and observability findings will mostly be missing-control findings. Flag in Information Gaps.
- **A specific file the user named** (e.g. a single workflow) → targeted audit of that file, with cross-cutting absences still noted.

Record the scope decision — it bounds coverage and belongs in the Caveats section of the output.

### Step 2 — Dispatch the pipeline-reviewer agent

Invoke the `pipeline-reviewer` subagent via the `Task` tool. Pass:

- The pipeline file path(s) and any build/policy/instrumentation files found in preconditions.
- The scope determined in Step 1 (full / pipeline-only / targeted).
- The instruction that this is a **three-dimension audit** against `ci-cd-pipeline-architecture.md`, `devsecops-and-supply-chain.md`, and `observability-and-monitoring.md`, and that **every finding must cite the resolving sheet** (sheet name + section).

The agent walks the audit checks across the three dimensions and returns a severity-rated findings list with `file:line` evidence and a sheet citation per finding, plus a `## Summary (machine-readable)` block at the top of its output. The agent follows the SME Agent Protocol (`meta-sme-protocol:sme-agent-protocol`) and returns Confidence / Risk / Information Gaps / Caveats.

### Step 3 — Present findings

Return the agent's output to the user. Surface, in this order:

- **Summary (machine-readable)** — copy the agent's verdict / severity counts / scope lines to the top so the user gets the verdict at a glance.
- **Executive summary** — production-ready verdict, count of findings by severity, and the per-dimension state (delivery safety / supply-chain integrity / observability).
- **Findings** organised by **Critical / High / Medium / Low**, each with `file:line` (or "absent — no such config") evidence, a specific fix, and the **resolving sheet citation**.
- **What the pipeline does well** — genuine strengths only (no rubber-stamping).
- **Confidence Assessment**, **Risk Assessment**, **Information Gaps**, **Caveats**.

## Audit dimensions and the sheet that grades each

The agent applies these checks; the command exists to make sure all three dimensions are covered and each finding is traceable to a sheet. Severity guidance below; the agent sets final severity from blast radius and context.

### Dimension A — Delivery safety → `ci-cd-pipeline-architecture.md`

| Check | Anti-pattern (finding) | Severity | Cite |
|-------|------------------------|----------|------|
| One immutable, digest-pinned artifact, built once and promoted | `:latest` / per-env rebuild / tag-pinned | Critical | `ci-cd-pipeline-architecture.md` § Stage 1 — Build / Red flags |
| Fast PR gate split from thorough release gate; pyramid right-side-up | All tests on every push; no fast feedback | Medium | § Stage 2 — Test / Fast feedback is a first-class requirement |
| Staging matches prod; promotion gated on automated verification | No staging, or "push to main, it'll be fine" | Critical | § Stage 4–5 / Red flags |
| Progressive delivery (canary/blue-green), never restart-in-place | `restart` / direct redeploy causing downtime | High | § Stage 6 — Progressive delivery |
| Automated metric-driven rollback encoded in the rollout | "We'll watch it manually" / fix-forward | Critical | § Stage 6 / § Stage 7 / Counters to the rationalizations |
| Migrations expand/contract, backward-compatible, UP+DOWN tested | Destructive migration coupled to the rollout | High | § Database migrations: expand/contract |
| IaC plan-review-apply from the pipeline; OpenTofu/Pulumi not BSL Terraform | `terraform apply` by hand; Terraform ≥1.6 by default | Medium | § Infrastructure as Code |

### Dimension B — Supply-chain integrity → `devsecops-and-supply-chain.md`

| Check | Anti-pattern (finding) | Severity | Cite |
|-------|------------------------|----------|------|
| SAST/SCA on every PR, gated on new High/Critical; secret scanning + push-protection | Security as a manual end-stage; scanners as warnings | High | `devsecops-and-supply-chain.md` § Shift-left: SAST, DAST, SCA as PR gates |
| Dependencies locked + hash-pinned; `--frozen`/`npm ci`/`--locked`; confusion-proof | Unpinned deps; floating versions at build time | High | § Dependencies: pin, lock, verify provenance |
| SBOM per artifact, signed, continuously re-scanned (CycloneDX/SPDX) | No SBOM — no CVE blast-radius map | High | § SBOM: the artifact you grep when the next CVE drops |
| Provenance to SLSA Build L3; identity verified before promotion | Cannot prove how the artifact was built | Medium | § Provenance and SLSA |
| Image + SBOM + provenance cosign-keyless signed; `cosign verify` gate AND admission Enforce | Unsigned artifacts; admission in Audit, not Enforce | Critical | § Signing and verification / Red flags |
| CI token `contents: read` by default; OIDC short-lived creds; no static keys | Runner has admin/long-lived cloud keys | Critical | § Least-privilege CI: the runner is your softest target |
| Fork PRs: read-only, no secrets, approval gate | Fork PR runs full pipeline with secrets | Critical | § Least-privilege CI / Red flags |
| Actions + base images pinned by 40-char SHA | Tag-pinned actions (mutable) | Medium | § Least-privilege CI / Red flags |
| Policy-as-code, version-controlled, unit-tested, same bundle CI + admission | Policy is a wiki page; or set to Audit | Medium | § Policy-as-code |

### Dimension C — Observability → `observability-and-monitoring.md`

| Check | Anti-pattern (finding) | Severity | Cite |
|-------|------------------------|----------|------|
| SLIs at the user's edge; SLOs below 100%; error budget runs the speed/stability contract | "We just keep it up" — no SLO | High | `observability-and-monitoring.md` § SLIs, SLOs, and the error budget |
| Alerts on user-facing symptoms, not causes (RED/USE), few enough on-call still reads them | CPU/disk alerts only; alert fatigue | High | § Actionable alerting / § RED and USE |
| Every deploy annotated; rollout gated on the SLI | "Just deploy, we'll watch Grafana" | High | § Watching deploys — closing the unmonitored-deploy gap |
| OpenTelemetry/OTLP instrumentation through a Collector — vendor-neutral, correlated | Vendor SDK lock-in; signals in three uncorrelated tools | Medium | § The three pillars / § vendor-neutral instrumentation with the OTel Collector |
| MTTR measurable (detect/mitigate/resolve timestamped) | "We can't measure MTTR" | Medium | § The production stake / Red flags |
| Dashboards answer "are users okay"; bounded-cardinality metrics | 40-panel dashboards; user_id on a metric (cardinality bomb) | Low | § Dashboards that answer "are users okay" |

## Output Format

The agent returns, and the command surfaces, this shape:

```markdown
## Summary (machine-readable)
verdict: not-production-ready
scope: full | pipeline-only | targeted
critical: N  high: N  medium: N  low: N
delivery_safety: <state>  supply_chain: <state>  observability: <state>

## Pipeline Audit: [pipeline file(s)]

### Executive summary
- Production ready: Yes/No
- Findings: N Critical, N High, N Medium, N Low
- Delivery safety / Supply-chain integrity / Observability: one line each

### Findings

#### Critical
| # | Finding | Evidence (file:line / absent) | Fix | Resolving sheet |
|---|---------|-------------------------------|-----|-----------------|
| 1 | [issue] | [.github/workflows/deploy.yml:42] | [action] | `devsecops-and-supply-chain.md` § Signing and verification |

#### High
[same columns]

#### Medium
[same columns]

#### Low
[same columns]

### What the pipeline does well
- [genuine strength, or "None identified at this scope"]

### Confidence Assessment
### Risk Assessment
### Information Gaps
### Caveats
```

Every row in the findings tables MUST populate the **Resolving sheet** column. A finding with no sheet citation is incomplete — drop it or trace it to the right sheet.

## Scope Boundaries

**This command covers:**

- Delivery-safety audit against `ci-cd-pipeline-architecture.md`
- Supply-chain integrity audit against `devsecops-and-supply-chain.md`
- Observability audit against `observability-and-monitoring.md`
- Dispatching the `pipeline-reviewer` agent and surfacing severity-rated, sheet-cited findings

**Not covered:**

- Designing new pipelines or deployment strategies (use `/design-deployment` and the `deployment-strategist` agent)
- Implementing the fixes (the audit names them; it does not apply them)
- Application code review, infrastructure provisioning, or test-strategy design (route to `/quality-engineering`, `/security-architect`, `/system-architect` per the sheets' cross-references)
