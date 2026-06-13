---
name: release-management-and-rollback
description: Use when a release is rebuilt separately for staging and production, when you cannot say exactly which commit and which bytes are running in prod right now, when "the version" is a moving tag like :latest or :prod, when shipping a fix means cutting a brand-new build instead of promoting one that already passed, when a release cannot be undone because the only artifact that existed was overwritten, when rolling back means rebuilding the old version from source and hoping it compiles the same, when nobody can answer "do we roll back or roll forward" without a 40-minute argument during an incident, when an approval is a Slack thumbs-up nobody can reconstruct later, when change management is either a rubber-stamp CAB that approves everything or a freeze that blocks everything, or when versioning is ad-hoc and consumers cannot tell a breaking change from a patch. Covers immutable artifact and version-identity discipline, build-once promote-everywhere across environments, semantic versioning and release channels, change management without theatre, the rollback-versus-roll-forward decision, automated rollback triggers and release health gates, and making every release reversible by construction.
---

# Release Management and Rollback

## The production stake

A release is the moment a thing you built becomes a thing your users depend on. Everything between `git commit` and "serving production traffic" is release management, and the two ways it kills you are both about *identity* and *reversibility*:

- **You cannot say what is running.** The version in prod is a mutable tag, the build was re-run per environment, and the artifact you tested is not the artifact you shipped. When the incident starts you are debugging bytes you have never seen.
- **You cannot get back.** The previous version's artifact was overwritten, garbage-collected, or never stored as an immutable thing — so "roll back" means "rebuild from source and pray the toolchain, dependencies, and base image resolve identically." Under outage pressure, that rebuild is a second outage.

The discipline this sheet enforces: **a release is one immutable, content-addressed artifact, built once, promoted unchanged across environments, identified by a version you never reassign, and retained long enough that rolling back is selecting a previous artifact — not rebuilding one.** If you cannot point at the exact digest in production, name the previous digest, and route traffic back to it without a build step, you do not have releases. You have a sequence of rebuilds you hope are equivalent.

This is engineering discipline, not a tool tour. Argo CD, cosign, and OpenTofu are mechanisms; the invariants — *build once, promote immutable, version honestly, retain the previous, decide rollback vs roll-forward deliberately* — are the point.

This sheet is the **release-lifecycle and decision** companion to `deployment-strategies` (which mechanism flips traffic) and `ci-cd-pipeline-architecture` (the surrounding gate sequence). It owns artifact identity, promotion, versioning, change management, and the rollback/roll-forward call.

## The two invariants everything else serves

### 1. Build once, promote everywhere (immutability)

A release artifact is built **exactly once**, from one commit, and the *same bytes* move through every environment. Staging tests the artifact; production runs the artifact staging blessed. The instant an environment rebuilds from source, your staging sign-off certifies a different binary — different dependency resolution, different base-image digest, different build-time clock — and every gate upstream proved nothing about what ships.

Immutability means **content-addressed and never overwritten**: a container by `sha256` digest, a Python wheel by version+hash, a Rust binary by a checksummed release asset. Mutable tags (`:latest`, `:prod`, `:stable`) are *pointers*, not identities — fine as a human-readable alias that you re-point, never as the thing you deploy or roll back to.

### 2. Reversible by construction (retention + identity)

Rollback is only fast if the previous release still exists as a deployable artifact and you know its identity. This is a *storage and recordkeeping* discipline, not a heroics discipline:

- The previous N artifacts are retained in the registry (never auto-pruned below your rollback window).
- Every release records its provenance — commit SHA, artifact digest, SBOM digest, who promoted it, when — in a place you can query during an incident (a release ledger / the GitOps repo history).
- "Roll back" resolves to "deploy artifact `@sha256:<prev>`," which already exists and already passed. No `git revert`, no rebuild, no fingers crossed.

## Version identity: name releases so consumers and operators can reason

A version is a *contract*, not a label. Two audiences read it: **consumers** (can I upgrade safely?) and **operators** (which exact thing is running?). They need different precision, so carry both.

| Identifier | Audience | What it promises | Example |
|------------|----------|------------------|---------|
| **SemVer** `MAJOR.MINOR.PATCH` | Consumers / API + library users | MAJOR = breaking, MINOR = additive, PATCH = fix-only | `2.7.0` |
| **Build / release version** | Release process | Monotonic, unique per build, ties to source | `2.7.0+build.4821` or `2.7.0-rc.3` |
| **Commit SHA** | Operators / forensics | Exact source state | `a1b9f3c` |
| **Artifact digest** | Runtime / rollback | Exact bytes deployed | `sha256:9e4f…` |

Rules that keep versions honest:

- **SemVer means what it says or it is noise.** If a "minor" bump breaks consumers, you have lied in the one field they trust. Gate breaking changes behind MAJOR; if you cannot tell whether a change is breaking, you do not understand your contract yet — that is the work, not the version string.
- **One version, one set of bytes, forever.** Never re-cut `2.7.0` with different content. A re-release is `2.7.1` (or `2.7.0+build.N`). Re-pointing a published version is the supply-chain equivalent of changing history.
- **Release channels are aliases over immutable versions.** `stable`, `beta`, `canary`, `lts` are *names that point at a pinned version*. Promotion is re-pointing the channel — `stable -> 2.7.0` — never mutating `2.7.0`. Consumers track a channel; operators pin the version.
- **Derive the version; do not type it.** Tag-driven or conventional-commit-driven version computation (e.g. release-please, semantic-release, `git describe`) removes the "someone forgot to bump" class of error.

## Immutable promotion across environments (concrete)

### Example 1 — build once, sign, promote by digest (GitHub Actions + cosign)

The artifact is built and signed in **one** job. Every later environment references it by the digest emitted here. Nothing downstream ever runs `docker build` again.

```yaml
# .github/workflows/release.yml — build ONCE, sign artifact + SBOM, emit the digest
name: release
on:
  push:
    tags: ["v*.*.*"]            # release is tag-driven; the tag IS the version contract

permissions:
  contents: read
  packages: write
  id-token: write               # OIDC for keyless cosign signing

jobs:
  build-and-sign:
    runs-on: ubuntu-latest
    outputs:
      digest: ${{ steps.push.outputs.digest }}
      version: ${{ steps.version.outputs.value }}
    steps:
      - uses: actions/checkout@v4
      - id: version
        run: echo "value=${GITHUB_REF_NAME#v}" >> "$GITHUB_OUTPUT"

      - uses: docker/setup-buildx-action@v3      # BuildKit: cache mounts, multi-arch

      # Wolfi/Chainguard base: glibc, rolling, rebuilt on upstream CVE within hours.
      # Distroless is the weakest minimal option (Debian-stable lag = stale CVEs).
      - id: push
        uses: docker/build-push-action@v6
        with:
          push: true
          platforms: linux/amd64,linux/arm64
          tags: ghcr.io/acme/orders:${{ steps.version.outputs.value }}   # human alias
          provenance: true                       # SLSA build provenance attestation
          sbom: true

      # SBOM in CycloneDX (security-workflow-friendly; SPDX if compliance-heavy)
      - uses: anchore/sbom-action@v0
        with:
          image: ghcr.io/acme/orders@${{ steps.push.outputs.digest }}
          format: cyclonedx-json
          output-file: sbom.cdx.json

      - uses: sigstore/cosign-installer@v3
      # Sign the IMAGE and the SBOM, keyless (Fulcio cert + Rekor transparency log).
      # A rollback target you cannot verify is not a rollback target.
      - run: |
          cosign sign --yes ghcr.io/acme/orders@${{ steps.push.outputs.digest }}
          cosign attest --yes --predicate sbom.cdx.json --type cyclonedx \
            ghcr.io/acme/orders@${{ steps.push.outputs.digest }}
```

### Example 2 — promotion as a recorded, immutable change (GitOps)

Promotion to an environment is **editing the desired-state repo to pin the digest** — not running a deploy script. Argo CD (or Flux) reconciles the cluster to match. The promotion is a git commit: reviewable, attributable, and itself the rollback mechanism (revert the commit).

```yaml
# clusters/prod/orders/release.yaml — the SINGLE source of truth for what prod runs.
# Promotion = change this digest in a PR. Rollback = set it back to the previous digest.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orders
spec:
  template:
    spec:
      containers:
        - name: orders
          # Pinned by digest — the EXACT bytes that passed staging. Never a moving tag.
          image: ghcr.io/acme/orders@sha256:9e4f2c...   # was sha256:7b1a8d... (prev release)
```

```yaml
# Argo CD verifies the signature/attestation BEFORE it will sync the digest to prod.
# Unsigned or unverifiable artifact => sync is refused => unverifiable bytes never reach prod.
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata: { name: orders-prod }
spec:
  source:
    repoURL: https://github.com/acme/clusters
    path: clusters/prod/orders
  syncPolicy:
    automated: { prune: true, selfHeal: true }   # continuously reconciled (OpenGitOps)
```

The release ledger is now **the git history of the prod repo**: every line says which digest, who changed it, when, and the reverting commit is one click. That is build-once-promote-everywhere and reversible-by-construction in one artifact.

> Why GitOps here, not a deploy script: declarative + versioned + pulled + continuously reconciled (the OpenGitOps principles) makes "what is running" equal to "what the repo says," and makes rollback a revert. A push-based deploy script leaves no immutable record of *what* was pushed *when* by *whom*.

> IaC note: the platform these releases run on is itself versioned state. Pin providers and modules; treat the state as immutable history. Prefer **OpenTofu** (MPL-2.0, CNCF-hosted) as the open default and name the reason — **Terraform ships under the non-OSI BSL 1.1 since 1.6**; it is no longer open source. Pulumi is the general-purpose-language alternative (Python/TS/Go).

## Change management without theatre

Change management exists to make changes *safe and reconstructable*, not to make them *slow*. The failure mode on both ends is well known: a weekly CAB that rubber-stamps everything it cannot actually evaluate (theatre that adds latency and zero safety), or a blanket freeze that blocks low-risk changes and trains everyone to route around the process.

Replace approval-as-ritual with **risk-tiered, evidence-backed change records**. The approval attaches to the *evidence*, and most of it is automated.

| Change class | Example | Gate | Approver | Record |
|--------------|---------|------|----------|--------|
| **Standard** (pre-authorised) | Routine deploy of a passing artifact via canary | Automated: tests + canary analysis pass | None (pre-approved policy) | Auto-logged release record |
| **Normal** | New feature, schema expand step, config change | Peer review + staging evidence | Code owner / on-call lead | PR + linked release record |
| **High-risk** | Destructive migration, auth change, infra topology | Review + rehearsed rollback plan + named owner | Service owner (async, time-boxed) | Change record with rollback plan |
| **Emergency** | Incident hotfix | Ship now, review after | On-call authority | Post-hoc change record within 24h |

Principles that kill the theatre while keeping the safety:

- **Pre-authorise the routine.** A deploy of an artifact that passed every gate via a metric-gated canary is *standard change* — no human approval, because the policy already approved the *class*. Humans approve policy, automation enforces it. This is the only way continuous delivery and change management coexist.
- **The approval is on evidence, and it is reconstructable.** "Did tests pass, is there a rollback path, who owns it if it breaks" — answered by the PR, the pipeline run, and the release record. A Slack 👍 you cannot reconstruct in six months is not change management; it is plausible deniability. The artifact-signing, SBOM attestation, and GitOps history already produce most of the audit trail — use it instead of a separate ceremony.
- **Risk-tier, do not blanket.** Most changes are low-risk and reversible; gating them like high-risk ones trains the org to treat all gates as obstacles. Reserve human approval for the genuinely irreversible (the contract step of a migration, an auth/permissions change, a one-way infra move).
- **A freeze is a scalpel, not a wall.** Freeze the specific risky surface during the specific risky window (peak commerce day) — not all changes for two weeks. A blanket freeze just bunches risk into one giant post-freeze release, which is a big-bang.

## The rollback vs roll-forward decision

When a release is bad, you have two recoveries. Choosing badly under pressure is its own incident. Decide the *policy* before the incident so the on-call decision is a lookup, not a debate.

**Roll back** = return to the previous known-good artifact (re-point the channel / revert the digest / flip blue-green). **Roll forward** = ship a new artifact that fixes the problem.

```
                Is the system currently harming users / data?  (errors, corruption, outage)
                                 |
                    ┌────────────┴────────────┐
                  YES                          NO  (degraded but contained, or caught in canary)
                    |                            |
   Is rollback SAFE and FAST?            Do you understand the fix,
   (prev artifact retained, no           and can you ship + verify it
    irreversible migration ran,          faster than the harm grows?
    no data written in new format)              |
        ┌───────────┴──────────┐        ┌───────┴────────┐
      YES                      NO       YES               NO
        |                       |        |                 |
   >>> ROLL BACK <<<      Roll FORWARD   Roll FORWARD   >>> ROLL BACK <<<
   (fastest stop-bleed)   (rollback would    (low blast,   (default; restore
                           lose/corrupt       fix is known)  known-good, then
                           data — restoring                  fix forward calmly)
                           old code breaks
                           on new-format data)
```

The decision rules, made explicit:

- **Default to rollback when users are being harmed.** It restores a *known-good, already-proven* state in seconds. Roll-forward ships an *unproven* fix into an active incident — you are debugging in production while it bleeds. Rollback first, diagnose second.
- **Rollback is unsafe when the new release wrote irreversible state.** This is exactly why `deployment-strategies` insists on expand/contract migrations: if the bad release ran a *destructive* schema change or wrote data in a format only it understands, reverting the code lands it on a schema/data it cannot read — rollback *is* the second outage. In that case roll forward, and treat the inability to roll back as a process bug to fix (it means an irreversible step shipped coupled to a release).
- **Caught in the canary → almost always roll back.** The blast radius is small and bounded; route the canary's slice back to stable and fix forward off the hot path. The automated trigger should already have done this.
- **Roll forward when the fix is well-understood, small, and faster to verify than the harm grows** — typically config/flag changes (a kill-switch flip is roll-forward that is *as fast as* rollback) or a one-line fix where rebuilding the previous artifact's data compatibility is harder than patching.

Whichever you choose, the *speed* of rollback comes entirely from the invariants at the top: the previous artifact exists, you know its digest, and routing back is a channel re-point or git revert — never a rebuild.

## Automated rollback triggers and release health gates

A release is not "done" at promotion; it is done when it has *proven healthy under real load*. Wire the proof into automation so a bad release reverts faster than a human can be paged. (Mechanism detail — `AnalysisTemplate`, Flagger metric checks, Gateway API traffic-shifting — lives in `deployment-strategies`; here is the release-level contract.)

- **Define release health as an SLO query, not a vibe.** Error rate, p99 latency, and a key business metric, queried by the rollout controller. "We'll watch the dashboard" is not a trigger; it is a hope with a Grafana tab.
- **Gate after promotion, not only before.** Post-promotion / post-canary analysis catches failures that appear only at full production load, and aborts to the previous artifact automatically.
- **Instrument vendor-neutrally.** The metrics that gate releases must outlive any vendor — emit via **OpenTelemetry / OTLP** (one wire protocol, one semantic-convention layer, OTel Collector). Never gate a release on a metric you can only get from a vendor SDK you might leave.
- **Define the bake time.** A release stays observed at full traffic for a defined window before the previous artifact is scaled down / pruned. Pruning the rollback target the instant you promote deletes your safety net.

## Common mistakes

- **Rebuild-per-environment.** Re-running `docker build` for staging then again for prod. The bytes diverge (dependency drift, base-image digest, build clock), so every upstream gate certified a binary you never shipped. Build once; promote the digest.
- **Mutable tags as the deploy/rollback identity.** Deploying `:latest` or `:prod` means "what is running" is unanswerable and "roll back to the previous version" is ambiguous. Tags are aliases; deploy and roll back by digest.
- **Irreversible release.** The previous artifact was overwritten or pruned, so rollback = rebuild-from-source-and-pray. Retain the previous N artifacts; rollback must be *select*, never *rebuild*.
- **SemVer that lies.** A "patch" that breaks consumers, or re-cutting an existing version with new bytes. The version is the one contract consumers trust — breaking it silently is worse than no version at all.
- **Coupling irreversible state changes to a release.** A destructive migration in the same release as the code that needs it removes the rollback path entirely. Expand/contract across separate releases (see `deployment-strategies`); the contract/destructive step ships alone, long after.
- **CAB theatre or blanket freeze.** Rubber-stamping changes nobody can evaluate adds latency and zero safety; freezing everything bunches risk into one big-bang. Risk-tier, pre-authorise the routine, and let automation produce the audit trail.
- **Approval as an unreconstructable 👍.** If you cannot reconstruct who approved what, on what evidence, six months later, you have deniability, not change management. Attach approval to the PR + pipeline run + release record.
- **Roll-forward-by-reflex during an active incident.** Shipping an unproven fix while users are harmed debugs in production. Default to rollback to a proven state; fix forward off the critical path — unless rollback is unsafe (irreversible state) or the fix is a flag flip.
- **Pruning the rollback target at promotion.** Scaling down / GC-ing the previous release the moment the new one goes live deletes your safety net before the new release has proven itself. Keep it warm for the bake window.
- **No post-promotion health gate.** Gating before the flip but not after misses failures that only surface at full production load. Wire SLO-query analysis after promotion with an automatic revert.

## Red flags — STOP

If any of these is true, stop and fix the release discipline before shipping:

- You cannot name the exact artifact digest running in production right now.
- You cannot name the *previous* release's digest, or it no longer exists in the registry.
- Rolling back requires a rebuild from source.
- The artifact in prod was built by a different job than the one that passed staging.
- "The version" is a mutable tag you re-point (`:latest`, `:prod`).
- A release can run a destructive migration in the same deploy as the code that depends on it.
- The release approval is a chat message you could not reconstruct in an audit.
- Nobody can state the rollback-vs-roll-forward policy without convening a meeting.
- The release "health check" is a human watching a dashboard.
- You re-cut a published version number with different bytes.

## The rationalizations (and the counters)

- *"Rebuilding for prod guarantees a clean, prod-specific build."* It guarantees a *different* build. Clean is irrelevant if it is not the bytes you tested. Build once, inject config at runtime; the artifact is environment-agnostic.
- *"Pinning digests is unreadable; tags are friendlier."* Tags are friendly *aliases* — keep them for humans. Deploy and roll back by digest. Readability for operators is not worth ambiguity during an incident.
- *"We'll just roll forward, rollback is scary / complicated."* Rollback is scary *because* you never made it cheap. The fix is the invariants in this sheet, not avoiding rollback. Roll-forward as a default is debugging in production under fire.
- *"We don't need to retain old artifacts; we can always rebuild the old tag."* Rebuilding the old commit does not reproduce the old artifact — dependencies, base images, and toolchains have moved. The only reliable previous version is the one you stored.
- *"SemVer is bureaucracy; we just ship."* Then your consumers cannot tell a safe upgrade from a breaking one, and integrate by trial-and-error against your outages. The contract is the product.
- *"CAB / change records slow us down."* Theatre slows you down; *risk-tiered, automated* change records do not — they pre-authorise the routine and reserve humans for the irreversible. Continuous delivery requires this, it is not opposed to it.
- *"We froze all changes to be safe for the launch."* You moved every change into one giant post-freeze release — the largest, riskiest, least-reversible deploy of the quarter. Freeze the specific risky surface, not all motion.
- *"It passed staging, it's fine — no need to watch it in prod."* Staging is not prod load, prod data, or prod concurrency. Release health is proven *after* promotion, by automated gates, or it is not proven.

## Related sheets

- `deployment-strategies` — the mechanism that flips traffic (rolling / blue-green / canary / flags), per-strategy automated rollback triggers, and expand/contract migrations that keep each release individually reversible.
- `ci-cd-pipeline-architecture` — the surrounding pipeline: test gates, environment promotion, build-once artifacts, and supply-chain attestation where signing/SBOM generation live.
- `infrastructure-as-code` — versioning and immutable promotion of the *platform* the releases run on (OpenTofu/Pulumi, pinned providers, state as history).
