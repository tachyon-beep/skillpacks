---
name: ci-cd-pipeline-architecture
description: Use when setting up or reviewing a CI/CD pipeline, when a "deploy" job is a single step that pushes straight to production, when deployments cause downtime or require manual rollback, when CI feedback takes longer than a coffee break, when a bad merge can reach prod with no gate, when migrations break production, when there is no staging or staging does not match prod, or when an incident happens minutes after a release with no automatic recovery. Covers verification gates, progressive delivery (canary / blue-green), automated metric-driven rollback, environment promotion, build-once-deploy-everywhere artifacts, database migration expand/contract, and supply-chain attestation in the pipeline.
---

# CI/CD Pipeline Architecture

**A pipeline that ends in a step called `deploy` is a loaded gun pointed at production. "Deploy to production" is not an action — it is a sequence of gates, health checks, gradual rollouts, and automated rollback triggers. The single most expensive outage class in software is "the deploy went out, error rate spiked, and a human took eleven minutes to notice and roll back by hand." Every minute of that is revenue, trust, and on-call sleep. The discipline below is what converts a deploy from a gamble into a controlled, observable, reversible transition. It is not slower than shipping fast — it *is* shipping fast, because it is the only way to ship continuously without the periodic week-long incident that eats all the velocity you saved.**

This is engineering discipline, not a tool tour. The tools change; the gate sequence does not.

## The shape: gates, not a step

Every production pipeline is a directed sequence where each stage is a **gate** — it either passes the artifact forward or stops the line. A red gate must halt promotion; a gate that warns and continues is decoration, not a gate.

```
commit ─▶ Build ─▶ Test ─▶ Supply-chain ─▶ Stage ─▶ Verify ─▶ Progressive ─▶ Verify ─▶ Monitor
         (once)   (fast    (sign + SBOM    deploy   stage     rollout to     prod      with auto-
                  feedback) + verify)               (smoke)   prod (canary)  (synthetic) rollback
```

Two properties make this work, and skipping either is the root cause of most pipeline incidents:

1. **Build once, deploy everywhere.** One immutable artifact, identified by content digest, flows through every environment. The bytes you tested in staging are the bytes that serve production. The moment you rebuild per environment, your staging signal is worthless.
2. **Promotion is gated by verification of the *previous* environment, never by a clock or a human's optimism.** "It merged to main" is not a reason to be in production. "Staging passed automated verification" is.

### Fast feedback is a first-class requirement

The PR gate exists to give a developer a verdict before they context-switch — target under ~10 minutes wall-clock. The merge/release gate can be slower and more thorough. Split them:

- **PR gate (fast):** lint, type-check, unit tests, affected-package tests, build. Parallelized, cached, fail-fast (fastest tests first).
- **Release gate (thorough):** full integration + E2E suite, supply-chain attestation, deploy-to-staging + verify.

Conflating the two means either PRs are agonizingly slow (developers stop running CI mentally and batch huge changes) or main is under-tested. Separate them deliberately.

## Stage 1 — Build (produce one immutable artifact)

Build once. Tag by **commit SHA and content digest**, never `latest` (`latest` is mutable, untraceable, and the cause of "works on my machine, breaks in prod"). Use BuildKit/buildx for concurrent stages, cache mounts, and multi-arch output.

**Minimal-image discipline (currency, June 2026):** distroless is now the *baseline*, and it is explicitly the *weakest* of the minimal-image options — it tracks Debian-stable, so its CVE remediation lags. Current production practice is **Wolfi** (glibc, rolling, container-native; built with Melange for packages + Apko for image composition) or **Chainguard Images**, which rebuild on upstream CVE within hours and ship a native SBOM + SLSA-L3 provenance per digest. Pick a minimal base; do not ship a full distro as your runtime layer.

```dockerfile
# Multi-stage build on a Wolfi/Chainguard base. BuildKit assumed.
# syntax=docker/dockerfile:1.7

FROM cgr.dev/chainguard/go:latest AS build
WORKDIR /src
# Cache module downloads across builds (BuildKit cache mount).
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=bind,source=go.mod,target=go.mod \
    --mount=type=bind,source=go.sum,target=go.sum \
    go mod download
COPY . .
RUN --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/app ./cmd/app

# Minimal, rolling, native-SBOM runtime. NOT a full distro, NOT plain distroless.
FROM cgr.dev/chainguard/static:latest
COPY --from=build /out/app /app
USER nonroot
ENTRYPOINT ["/app"]
```

```bash
# Build and push by digest; capture the digest as the artifact identity.
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag registry.example.com/app:${GIT_SHA} \
  --provenance=true --sbom=true \
  --push .

# The digest — not the tag — is what every later stage references.
DIGEST=$(docker buildx imagetools inspect registry.example.com/app:${GIT_SHA} \
  --format '{{json .Manifest.Digest}}')
```

## Stage 2 — Test (the pyramid, in CI)

```
       /\
      /E2E\      ← Few, critical user journeys only (single digits)
     /------\
    / Integ  \   ← API contracts, DB integration, message flows
   /----------\
  /    Unit    \ ← Fast, isolated, the bulk of your assertions
 /______________\
```

**Optimization is the answer to slow tests; deletion is not.** When the suite is too slow:

- **Parallelize** across runners (shard by test file/timing).
- **Run affected tests on PRs**, full suite on merge (Nx/Turborepo/Bazel/`--changed`).
- **Cache** dependencies, build outputs, and test-DB fixtures.
- **Fail fast** — order fastest tests first so a broken build dies in seconds.

An inverted pyramid (mostly E2E) is slow *and* flaky *and* a poor failure localizer. If you find yourself there, that is a quality-engineering problem — route to `/quality-engineering` (analyze-pyramid).

## Stage 3 — Supply-chain attestation (sign, SBOM, verify — in pipeline)

Currency, June 2026: this is now a required gate, not a nice-to-have. The bar is **SLSA v1.0 Build L3** (the federal-procurement floor).

- **Generate an SBOM** with Syft: **CycloneDX 1.5+** (security-workflow-friendly, native SLSA attestation support) or **SPDX 2.3+** (compliance-heavy estates).
- **Sign the image *and* the SBOM** with **Sigstore/cosign 2.x**, keyless via Fulcio (short-lived cert from OIDC identity) + Rekor (public transparency log). No long-lived signing keys to leak.
- **Verify in-pipeline before promotion.** A signature you never check is theater.

```yaml
# GitHub Actions — keyless signing + SBOM attestation + verify gate.
permissions:
  id-token: write   # required for cosign keyless (OIDC)
  packages: write
  contents: read

jobs:
  attest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: sigstore/cosign-installer@v3
      - uses: anchore/sbom-action@v0
        with:
          image: registry.example.com/app@${{ env.DIGEST }}
          format: cyclonedx-json
          output-file: sbom.cdx.json

      # Sign the image (keyless) and attach the SBOM as a signed attestation.
      - run: |
          cosign sign --yes registry.example.com/app@${{ env.DIGEST }}
          cosign attest --yes \
            --predicate sbom.cdx.json --type cyclonedx \
            registry.example.com/app@${{ env.DIGEST }}

  verify-gate:
    needs: attest
    runs-on: ubuntu-latest
    steps:
      - uses: sigstore/cosign-installer@v3
      # GATE: fail the pipeline if the image is not signed by THIS repo's workflow.
      - run: |
          cosign verify \
            --certificate-identity-regexp "https://github.com/${{ github.repository }}/.*" \
            --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
            registry.example.com/app@${{ env.DIGEST }}
          cosign verify-attestation --type cyclonedx \
            --certificate-identity-regexp "https://github.com/${{ github.repository }}/.*" \
            --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
            registry.example.com/app@${{ env.DIGEST }}
```

The cluster should *also* refuse unsigned images at admission (cosign policy controller / Kyverno) — defense in depth, so a manual `kubectl apply` cannot bypass the pipeline's signing.

## Stage 4–5 — Staging deploy + verify

**Staging must match production** in shape: same orchestrator, same deploy mechanism, same migration process, representative (anonymized) data volume. A staging environment that differs in kind from prod is a confidence generator, not a confidence *source*.

Verification is **automated**, never "a human clicks around":

```yaml
verify_staging:        # any failure STOPS the line — no promotion to prod
  - health endpoint returns 200 and reports schema_version == expected
  - critical API contracts respond correctly (smoke suite)
  - migrations applied; row counts and constraints sane
  - background workers draining the queue
  - external integrations reachable (or stubbed deterministically)
```

## Stage 6 — Progressive delivery to production (never a restart)

**NEVER deploy by restarting in place.** That is downtime plus an unrecoverable cutover. Choose a strategy that keeps the old version serving until the new one is *proven*:

| Strategy | How | Best for | Cost |
|---|---|---|---|
| **Canary** | New version gets 1% → 10% → 50% → 100%, metric-gated at each step | High-traffic services with good metrics | Modest extra capacity |
| **Blue-green** | Stand up Green fully, smoke-test at 0% traffic, flip, keep Blue warm for fast rollback | Critical systems, instant rollback needed | ~2× during cutover |
| **Rolling** | Replace instances N at a time with health gates | Cost-sensitive, moderate traffic | None extra (but mixed versions) |

### GitOps + progressive delivery (currency, June 2026)

Pull-based GitOps is the default control plane: git is the single source of truth, an in-cluster controller continuously reconciles desired state. Anchor to the **OpenGitOps** principles (declarative, versioned/immutable, pulled, continuously reconciled).

- **Argo CD** dominates (~60% of clusters; the overwhelming majority of its users run it in production). Pair it with **Argo Rollouts** for canary/blue-green.
- **Flux** is the lighter, fully-CRD alternative — pair it with **Flagger**. Pick by ecosystem fit, not hype.

Both shift traffic via **Gateway API** or a service mesh and drive **automated, metric-driven rollback** off your analysis provider. Traffic standard note (currency): **Gateway API has replaced Ingress** — Ingress-NGINX was retired March 2026; target **Gateway API v1.5** (Feb 2026) via Envoy Gateway, Istio, Cilium, or Kong. Use **native sidecar containers** (restartable init containers) for mesh proxies; drop the old bare-sidecar pattern.

```yaml
# Argo Rollouts: metric-gated canary with AUTOMATED rollback.
# The rollout itself is the gate — it pauses and aborts on bad metrics
# without a human in the loop.
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: app
spec:
  replicas: 10
  strategy:
    canary:
      # Traffic shifting via Gateway API (the post-Ingress standard).
      trafficRouting:
        plugins:
          argoproj-labs/gatewayAPI:
            httpRoute: app-route
            namespace: app
      steps:
        - setWeight: 10
        - pause: {duration: 5m}
        - analysis: {templates: [{templateName: success-rate}]}  # gate
        - setWeight: 50
        - pause: {duration: 10m}
        - analysis: {templates: [{templateName: success-rate}]}  # gate
        - setWeight: 100
      # If any analysis run fails, the rollout ABORTS and shifts
      # 100% of traffic back to the stable ReplicaSet automatically.
  selector:
    matchLabels: {app: app}
  template:
    metadata:
      labels: {app: app}
    spec:
      containers:
        - name: app
          image: registry.example.com/app@sha256:...   # by DIGEST, never a tag
          readinessProbe:
            httpGet: {path: /healthz, port: 8080}
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  metrics:
    - name: success-rate
      interval: 1m
      count: 5
      # ABORT (and roll back) if success rate drops below 99% on the canary.
      successCondition: result[0] >= 0.99
      failureLimit: 1
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(http_requests_total{app="app",code!~"5.."}[2m]))
              /
            sum(rate(http_requests_total{app="app"}[2m]))
```

The point of this manifest: **the rollback decision is encoded in the pipeline, evaluated continuously, and executed by the controller in seconds.** No pager, no human judgement at 3 a.m., no eleven-minute mean-time-to-rollback.

## Stage 7 — Verify production + monitor with auto-rollback

Post-deploy verification is automated synthetic checks plus metric SLOs:

```yaml
verify_production:
  - synthetic transaction over the critical user journey passes
  - p95 latency <= baseline * 1.2
  - error rate < 1%
  - saturation (CPU, mem, DB connections, queue depth) within bounds

auto_rollback_when:
  - canary analysis fails (handled by Rollouts/Flagger above)
  - error rate > 5% sustained 3m
  - p99 latency > 2× baseline sustained
  - readiness failing on N consecutive checks
```

**Observability is vendor-neutral (currency, June 2026).** Instrument with **OpenTelemetry** (CNCF-graduated): traces, metrics, and logs are stable across SDKs; **Profiles** is the stabilizing fourth signal. One wire protocol (**OTLP**), one semantic-convention layer, and the **OTel Collector** (receivers → processors → exporters) so backends are swappable. For high-volume estates, note **OTel Arrow** (compression) and **eBPF zero-code auto-instrumentation**. **Never instrument application code with a vendor SDK** — you will be hostage to that vendor's pricing and a rewrite to leave. The whole metric-driven-rollback loop depends on having trustworthy, portable signals; this is the foundation, not an afterthought.

## Database migrations: expand/contract (this takes 3 deploys, and that is correct)

The most common "the deploy broke prod" cause is a schema change that the old running code cannot tolerate during a rolling/canary cutover, where *both versions run simultaneously*. The fix is **backward-compatible, multi-phase migration** — never a destructive change coupled to a code deploy.

```
1. EXPAND  — additive only. Add nullable column / new table / new index
             (Postgres: CREATE INDEX CONCURRENTLY). Old AND new code both work.
2. MIGRATE — backfill data in batches; dual-write if needed.
3. DEPLOY  — ship code that reads/writes the new shape.
4. CONTRACT— in a LATER deploy, drop the old column/table once no code references it.
```

Each phase is independently deployable and independently reversible. Test the **down** migration in CI, not in the incident:

```yaml
test_migrations:
  - apply UP to a throwaway DB seeded with prod-shaped data
  - run the app test suite against the migrated schema
  - apply DOWN; assert schema + data integrity restored
  - assert UP is backward-compatible: run the PREVIOUS app version's
    smoke suite against the migrated schema (catches the dual-version trap)
```

A migration that cannot be rolled back is a one-way door — treat it like one and gate it accordingly.

## Infrastructure as Code (currency, June 2026)

The pipeline's environments are themselves code. Two currency points that change the default advice:

- **The Terraform / OpenTofu split is permanent.** Terraform ships under the **BSL 1.1 (non-OSI)** license since 1.6 — **stop calling it "open source."** **OpenTofu** is the MPL-2.0 fork, now CNCF-hosted, at ~1.11.x (1.12 in beta), with 3,900+ providers and a roadmap that has *diverged* from HashiCorp's rather than merely tracking it. **Treat OpenTofu as the default open choice, and name BSL as the reason.**
- **Pulumi** remains the general-purpose-language alternative (Python/TS/Go) for teams who want real code, real loops, and real unit tests over HCL.

Whatever you pick: plan in CI, require human review on the plan for prod, apply from the pipeline (never a laptop), and keep state locked and remote.

## Secrets

```yaml
# WRONG — credentials in the workflow, unrotatable, leaked in logs.
env:
  DATABASE_URL: postgresql://user:pass@host/db

# RIGHT — from the secret manager, scoped per environment.
env:
  DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

- Store in a secret manager (cloud KMS / Vault / platform CI secrets); never in git, never in image layers.
- Prefer **OIDC/workload-identity federation** over long-lived cloud keys in CI — short-lived tokens, nothing to leak.
- Different secrets per environment; rotate automatically; enable push-protection secret scanning.

## Common mistakes

| Mistake | Why it bites | Fix |
|---|---|---|
| "Deploy" is one job that restarts the service | Downtime + unrecoverable cutover | Progressive rollout (canary/blue-green) with the old version live |
| Rebuild per environment | Staging tested different bytes than prod ships | Build once, promote one digest everywhere |
| Tag by `latest` | Untraceable, mutable, non-reproducible | Tag by commit SHA + content digest |
| Promote on "merged to main" | No gate; a broken main reaches prod | Gate promotion on automated verification of the prior env |
| Manual click-through verification | Slow, non-repeatable, skipped under pressure | Automated health + synthetic + smoke checks |
| No automated rollback | MTTR = "until a human notices" | Encode rollback in the rollout (metric-gated abort) |
| Destructive migration in the deploy | Old code dies during dual-version cutover | Expand/contract, backward-compatible, 3 deploys |
| Down migration never tested | You write it during the incident | Test UP+DOWN+prev-version-compat in CI |
| One slow CI gate for everything | PRs crawl or main is under-tested | Split fast PR gate from thorough release gate |
| Vendor-SDK instrumentation | Lock-in; rollback metrics held hostage | OpenTelemetry/OTLP, swap backends at the Collector |
| Unsigned images, no SBOM | Can't prove provenance; supply-chain blind | cosign keyless sign + Syft SBOM + in-pipeline verify + admission policy |
| Calling Terraform "open source" | It's BSL 1.1 since 1.6 | Default to OpenTofu (MPL-2.0); name the license as the reason |
| Distroless treated as the gold standard | Debian-stable lag = stale CVEs | Wolfi / Chainguard (rolling, hours-fast CVE rebuilds) |
| One environment | Prod becomes your staging | Minimum staging + production, staging matching prod |

## Red flags — STOP

If you hear yourself (or a teammate) say any of these, the pipeline is about to cause an incident. Stop and fix the gate first.

- "Just push to main, it'll be fine." → No promotion without a passing verification gate.
- "Tests pass locally, skip CI." → Local is not the artifact that ships. Never skip CI.
- "Restart is faster than blue-green." → Downtime is never the fast option once you count the incident.
- "We'll watch it manually after deploy." → Manual watching is not rollback. Encode metric-gated auto-abort.
- "If it breaks we'll fix forward." → Fix-forward under outage pressure is how one bug becomes three. Roll back, then fix.
- "Migrations can run as part of the deploy." → Expand/contract first; never couple a destructive change to a rollout.
- "We'll add staging/rollback/signing later." → Later arrives as an incident. Build the gate before the first prod deploy.
- "Terraform is open source, just use it." → BSL since 1.6. Default OpenTofu unless a deliberate reason says otherwise.
- "Tag it `latest` for now." → `latest` is unreproducible. Digest-pin from the start.

## Counters to the rationalizations

| Rationalization | Counter |
|---|---|
| "It's just an MVP/demo pipeline." | MVP pipelines become production pipelines untouched. The "we'll harden it later" rewrite never gets prioritized over features. Build the gates once, now, when it's cheap. |
| "Staging is expensive." | One Sev-1 caused by an unverified deploy costs more than a year of staging — in incident hours, customer trust, and the follow-up retro tax. Staging is the cheap insurance. |
| "Blue-green/canary doubles our cost." | It doubles capacity for *minutes during cutover*, not steady-state. Compare that to the cost of downtime × traffic × incident duration. The math is not close. |
| "We'll add rollback later." | You need rollback precisely when a deploy is already failing — the one moment you cannot calmly build it. The whole value is that it exists *before* the bad deploy. |
| "Health checks/verification are overkill." | A silent bad deploy serving 5xx to 5% of users for an hour is worse than any deploy you blocked. Verification is what turns "silent" into "caught in 5 minutes." |
| "Our app is too simple for this." | Deployment risk scales with traffic and coupling, not lines of code. A 200-line service in the critical path needs the same gates as a monolith. |
| "Signing/SBOM is compliance theater." | It's the difference between "we patched the CVE in 3 hours" and "we spent two days figuring out which images even contain the vulnerable package." The SBOM is your incident map. |
| "This slows us down." | Continuous delivery *is* this discipline. The teams shipping 50×/day are the ones with the strongest gates — gates are what make high frequency safe. Without them you ship fast until the outage, then freeze for a week. |

## Cross-references

- `/quality-engineering` — test pyramid (analyze-pyramid), flaky-test triage, and CI test-pipeline staging (setup-pipeline).
- `/web-backend` — API contract tests in CI; database migration patterns at the data layer.
- `/security-architect` — threat-modeling the deploy path and CI as an attack surface (supply-chain, secret exfil).
- `/system-architect` — when deployment pain is actually an architecture/coupling problem.
- This pack's `/design-deployment` command and `deployment-strategist` / `pipeline-reviewer` agents for hands-on design and review.

## Quick checklist

- [ ] One immutable artifact, digest-pinned, built once and promoted everywhere
- [ ] Fast PR gate (<~10 min) split from thorough release gate
- [ ] Test pyramid right-side-up; affected-tests on PR, full suite on merge
- [ ] Image + SBOM signed (cosign keyless), verified in-pipeline, enforced at admission
- [ ] Staging matches prod; promotion gated on automated staging verification
- [ ] Progressive rollout (canary/blue-green) via GitOps + Gateway API — never restart-in-place
- [ ] Automated, metric-driven rollback encoded in the rollout (not a human + a pager)
- [ ] Production verification: synthetic journey + SLO checks
- [ ] OpenTelemetry instrumentation (vendor-neutral) feeding the rollback metrics
- [ ] Migrations: expand/contract, backward-compatible, UP+DOWN+prev-version tested in CI
- [ ] IaC: OpenTofu/Pulumi, plan-review-apply from the pipeline, remote locked state
- [ ] Secrets from a manager via OIDC/workload identity; per-env; rotated; scanned

## The bottom line

"Deploy to production" is build-once → verify staging → roll out gradually → verify production → auto-roll-back on bad signal. Each arrow is a gate that can stop the line, and the rollback decision lives in the pipeline, not in a human's reflexes at 3 a.m. Skipping gates to "move fast" is what produces the week-long incident that destroys a quarter of velocity. **This discipline IS moving fast — it is the only way to ship continuously and still sleep.**
