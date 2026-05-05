# Supply-Chain Security

## Overview

Architectural controls for the build and dependency supply chain: source
code, dependencies, build pipelines, artifacts, and the chain of custody
between them. After SolarWinds (2020), Log4Shell (CVE-2021-44228),
ua-parser-js (2021), Codecov (2021), and the xz-utils backdoor
(CVE-2024-3094), supply-chain compromise is a board-level threat. This
skill is the architecture-level treatment; tactical tooling (Dependabot,
Trivy, etc.) is deferred to `security-architecture-review.md` and CI/CD
automation skills.

**Core Principle**: Every artifact you ship is the product of every
artifact you depend on, every machine that built it, every account that
could push to those machines, and every developer with commit rights.
Trust is transitive — and so is compromise.

## When to Use

Load this skill when:

- Designing or reviewing a build/release pipeline
- Selecting model registries, package registries, or container registries
- Threat modeling third-party-code risk
- Implementing or evaluating SLSA, SBOM, signing, or provenance
- Responding to a published advisory affecting a dependency
- User mentions: "supply chain", "SBOM", "SLSA", "Sigstore",
  "cosign", "dependency confusion", "typosquat", "provenance",
  "build attestation"

**Don't use for**:

- Vulnerability management of running systems (`security-architecture-review.md`)
- Compliance mapping (`compliance-awareness-and-mapping.md`)
- Application-level threat modeling (`threat-modeling.md`)

## Reference Frameworks

| Framework | Version (current) | Scope | Reference |
|-----------|-------------------|-------|-----------|
| **SLSA** | v1.1 (Aug 2024); v1.0 (Apr 2023) | Build integrity levels | slsa.dev |
| **SBOM (SPDX)** | 3.0 (Apr 2024); 2.3 widely deployed | Software bill of materials | spdx.dev |
| **SBOM (CycloneDX)** | 1.6 (Apr 2024) | Software/HW/ML/SaaS BOM | cyclonedx.org |
| **in-toto** | 1.0 (2023) Attestation Framework | Build-step attestation | in-toto.io |
| **Sigstore** | GA | Keyless signing & transparency log | sigstore.dev |
| **NIST SP 800-161** | Rev 1 (May 2022) | C-SCRM (federal) | csrc.nist.gov |
| **EO 14028 / OMB M-22-18** | 2021 / 2022 | US federal SBOM mandate | whitehouse.gov |
| **CRA (EU)** | Reg. (EU) 2024/2847 | Cyber Resilience Act, in force Dec 2024, applies Dec 2027 | eur-lex.europa.eu |

Cite the version of each framework you used in any artifact.

---

## Threat Model: Supply Chain

Apply STRIDE to each link in the chain. The chain is at minimum:

```text
Source code ─→ CI build ─→ Test ─→ Artifact ─→ Registry ─→ Deployment
   ^              ^         ^         ^           ^            ^
  Devs         Build      Test    Signing      Storage      Runtime
  Repo hosts   nodes      infra   keys/HSM     ACLs         hosts
```

### CWE / ATT&CK / Standard Threats

| ID | Threat | CWE | MITRE ATT&CK |
|----|--------|-----|--------------|
| SC-01 | Compromised dependency (malicious update pushed) | CWE-1357, CWE-1395 | T1195.001 |
| SC-02 | Dependency confusion (private name resolved to public registry) | CWE-1357 | T1195.002 |
| SC-03 | Typosquatting (`reuests` vs `requests`) | CWE-1357 | T1195.002 |
| SC-04 | Compromised build pipeline (CI runner secrets stolen) | CWE-1395 | T1195.002, T1078.004 |
| SC-05 | Source-repo compromise (commit injection, force-push) | CWE-1395 | T1195.002, T1078 |
| SC-06 | Tampered artifact in registry | CWE-494, CWE-345 | T1195.002 |
| SC-07 | Pulled image/binary not what was built (replay) | CWE-345 | T1195.002 |
| SC-08 | Backdoored toolchain (compiler, base image) | CWE-1395 | T1195.003 |
| SC-09 | Insider commit (xz-utils pattern: trusted maintainer) | CWE-1357 | T1195, T1199 |
| SC-10 | Publishing-account takeover (npm/PyPI account hijack) | CWE-1357 | T1078, T1195 |

**Worked example — xz-utils (CVE-2024-3094, March 2024)**: A long-trusted
maintainer added obfuscated build-time logic to a widely-used compression
library that injected an SSH backdoor into downstream `sshd` builds. This
was SC-09 + SC-08 in combination — insider trust plus toolchain
manipulation. Detection came from a performance regression, not from
signing or scanning. Lessons:

- **Signing alone does not stop insider compromise** — the malicious
  release was signed legitimately.
- **Reproducible builds** would have surfaced the discrepancy between
  source tree and build output.
- **Maintainer concentration risk** is a real threat — single-maintainer
  packages need different controls than multi-maintainer projects.

---

## SLSA: Supply-chain Levels for Software Artifacts

SLSA (pronounced "salsa") defines build-integrity levels. Current spec is
**SLSA v1.1** (Aug 2024); v1.0 dropped the SLSA-4 level — the current top
level is SLSA Build L3.

### SLSA Build Track Levels

| Level | Requirement | Defense against |
|-------|-------------|-----------------|
| **L1** | Build process documented; provenance generated (need not be authenticated) | Mistakes; basic tampering visibility |
| **L2** | Hosted build platform; signed provenance; tamper-evident | Tampering after build; forged provenance |
| **L3** | Hardened, isolated builds; non-falsifiable provenance | Build-platform compromise; forged builds |

**SLSA Source Track** (separate, less mature) covers source-code
controls (code review, branch protection, retention).

### What "Provenance" Means in SLSA

Provenance is a signed statement of *how an artifact was built*:

- The source repository and commit
- The build platform and configuration
- The build steps (recipe)
- The output artifact's digest

The provenance is signed by the build platform and verifiable by anyone
with the public key (or via Sigstore transparency log).

### How to Reach Each Level

**SLSA L1**:

- Move builds out of "any laptop" into a documented CI workflow
- Emit a provenance attestation (SLSA Provenance v1.0 schema) per build,
  even if unsigned

**SLSA L2**:

- Use a hosted CI (GitHub Actions, GitLab CI, Buildkite, etc.) with no
  long-lived build credentials on developer machines
- Sign the provenance — Sigstore keyless signing via OIDC is standard
  (see Sigstore section below)
- Store provenance in a tamper-evident location (transparency log or
  immutable bucket)

**SLSA L3**:

- Build runners are ephemeral, isolated, and untrusted-input-resistant
- Provenance is generated by the build platform itself, not the build
  workflow (so a malicious workflow cannot forge it). GitHub's
  `slsa-github-generator`, GitLab's keyless signing, and reusable
  workflows are common patterns.
- No workflow has access to the signing key; signing is done by an
  isolated component the workflow cannot influence.

### Anti-Patterns

- **Self-issued provenance**: A build that signs its own provenance with
  a key the build itself can read is L1 at best. The signing party must
  be distinct from and isolated from the workflow.
- **Confusing source levels with build levels**: SLSA Build L3 says
  nothing about whether the source was reviewed.
- **Treating SLSA as a compliance checkbox**: SLSA verifies that the
  build is what the source says. It does not verify that the source is
  benign.

---

## SBOM: Software Bill of Materials

An SBOM is a structured inventory of components in an artifact. US
federal contractors are required to provide SBOMs (EO 14028 / OMB M-22-18).
EU CRA will require similar evidence in EU markets.

### Format Selection

| Format | Maintained by | Strengths | Choose when |
|--------|---------------|-----------|-------------|
| **SPDX** | Linux Foundation | License-focused; ISO/IEC 5962:2021 standard | License compliance is primary; long-term archival |
| **CycloneDX** | OWASP | Security-focused; supports VEX, ML, SaaS, services | Vulnerability response is primary; modern dev pipelines |

Both are widely tooled (Syft, sbom-tool, `cdxgen`, etc.). Most projects
should produce **CycloneDX 1.6** or **SPDX 2.3+** at build time.

### What Belongs in an SBOM

- Each direct and transitive dependency (name, version, **hash/digest**,
  license, supplier)
- The build artifact itself (with its digest)
- The build's source commit (linked via provenance)
- For container images: every layer, base image, OS packages
- For ML systems: **model artifacts, datasets, and inference-time
  dependencies** (CycloneDX 1.5+ has ML-BOM extension)

### VEX: Vulnerability Exploitability eXchange

A CVE found in an SBOM does not always mean exploitable. **VEX**
(CycloneDX 1.4+ or OpenVEX) is the structured way to say "we ship CVE-X
but this codepath is not reachable" so downstream consumers can triage.

Without VEX, every disclosed CVE forces the same emergency response.
With VEX, you separate "we know and we're affected" from "we know and
we're not".

### SBOM Lifecycle Anti-Patterns

- **Generate-and-forget**: SBOM produced at first build, never updated.
  An SBOM must be regenerated **per build**.
- **Source-only SBOM**: Generated from `package.json` rather than the
  built artifact. Misses transitive resolution differences and
  build-time fetches.
- **Hash-less SBOM**: Component listed by name + version with no digest.
  Fails to detect rebinding of the same version to different content.

---

## Sigstore: Keyless Signing

Sigstore is a stack — `cosign` (signing tool), `Fulcio` (short-lived
cert authority backed by OIDC), `Rekor` (transparency log) — that lets
you sign artifacts without managing long-lived keys.

### How It Works (Briefly)

1. CI workflow authenticates to Fulcio with an OIDC token (e.g., GitHub
   Actions identity).
2. Fulcio issues a short-lived (≈10 min) X.509 cert binding the OIDC
   identity to a freshly-generated key.
3. The cert + signature are recorded in Rekor (append-only transparency
   log).
4. The private key is discarded after signing.
5. Verifiers check the signature plus the OIDC identity in the cert
   against an expected policy (e.g., "must be signed by the
   `github.com/myorg/myrepo` workflow on `main`").

### What Sigstore Buys You

- **No HSM, no key rotation, no key compromise** to manage.
- **Identity-bound signatures**: signature ties to *who built it*, not
  just *that it was signed*.
- **Public log**: tampering with the log is detectable.

### What Sigstore Does Not Buy You

- It does not verify that the *source* is benign — only that the
  signed identity built the artifact.
- It does not replace SBOM or SLSA — it is the signing primitive used
  by both.

### Verification Policy

Signatures are only as strong as the verification policy. Required:

- **Identity expectation**: `==` match on workflow path, branch, repo
- **OIDC issuer expectation**: only accept the specific provider
- **Policy enforcement at deploy time**: e.g., admission controller
  rejecting unsigned images (Kyverno, Connaisseur, Sigstore Policy
  Controller)

A build pipeline that signs but never verifies is theater.

---

## in-toto: Step-by-Step Attestations

in-toto generalizes "SLSA Provenance" to arbitrary build-step chains.
Each step (lint, test, build, package, sign) emits a signed attestation;
the final consumer verifies the chain matches a declared layout.

Use in-toto when:

- Builds are multi-stage with hand-offs between systems
- You need attestations at finer granularity than "one final provenance"
- You're running federal pilots or EU CRA-aligned regulated builds

For typical SaaS, SLSA L2/L3 provenance from a single build platform is
sufficient and simpler.

---

## Dependency-Layer Threats and Controls

### Dependency Confusion (CWE-1357)

**Threat**: Internal package `corp-utils` is referenced by an internal
project. A public registry (PyPI/npm) accepts an attacker's package
named `corp-utils`. The build resolves to the public one because of
search-path order or version-resolution preferring higher versions.

**Controls** (apply all):

1. **Reserve namespaces** on public registries (squat your own internal
   names).
2. **Scoped names** — npm `@org/...`, Python namespace packages.
3. **Configure resolver** to prefer internal registry, refuse fallthrough
   for internal names.
4. **Lock files committed and verified** in CI (no resolution-time
   surprises).

### Typosquatting

**Threat**: Attacker publishes `reuests`, `colorama-utils`,
`python-dateutil-fix`. Developer or LLM-generated code references the
typo.

**Controls**:

- Allow-list internal-approved packages OR require review for new deps
- Tools that detect typosquats during dependency review (Socket,
  StepSecurity, others)
- Internal mirror with allow-list policy

### Hallucinated Packages (LLM-era)

LLM coding assistants invent package names that do not exist. Attackers
register the most-likely hallucinations. Reviewing code-review patches
for "is this package real, and is it the one I expect" is now part of
the threat model.

### Pinning, Lockfiles, and Hashes

- **Pin versions** in declared deps (no `^` or `~` range in production).
- **Commit lockfiles** (`package-lock.json`, `poetry.lock`,
  `Cargo.lock`, `go.sum`).
- **Hash-pin** when supported (`pip install --require-hashes`,
  `pnpm` content-addressable store, `npm install --frozen-lockfile`).
- **CI verifies** the lockfile matches declared deps and refuses to
  resolve fresh.

### Mirror, Vendor, or Trust?

Three postures, increasing in cost:

1. **Trust the public registry** — accept some supply-chain risk;
   smallest cost; mitigations are scanning + advisories.
2. **Mirror the public registry** with an internal proxy (Artifactory,
   Nexus) — review-on-promotion; rollback control.
3. **Vendor dependencies** into your repo — full audit trail; costliest
   to maintain; common in classified/regulated builds.

Choice depends on threat appetite and regulatory context.

---

## Build-Pipeline Hardening

Treat the build pipeline as a privileged production system. It is.

### Minimum Pipeline Hygiene

- **Pinned actions**: GitHub Actions referenced by SHA, not by tag
  (`uses: actions/checkout@<full-sha>`). Tags are mutable.
- **No secret promotion across stages** unless explicitly scoped.
- **OIDC over long-lived keys**: deployments to AWS/GCP/Azure use
  short-lived federated credentials, not stored access keys.
- **Branch protection**: required reviews, signed commits, no
  force-push to protected branches.
- **No third-party action with full repo access** — review tokens'
  scopes and use minimum.
- **Ephemeral runners** (GitHub-hosted, fresh VM per job) over
  long-lived self-hosted runners.

### Reproducible Builds

The same source produces a bit-for-bit identical artifact, regardless
of build host. Reproducibility:

- Lets multiple independent parties verify a release matches its source.
- Makes xz-utils-style toolchain backdoors observable (output diverges
  from "expected" reproducible build).

Reproducibility costs effort (build env, timestamps, ordering, paths).
For high-assurance contexts (debian/coreutils-style) it is worth it; for
typical SaaS it is aspirational.

---

## Container and OCI Specifics

- **Pin base images by digest**, not tag (`alpine@sha256:...`).
- **Distroless or minimal base** images reduce surface and dependency
  count.
- **Layer SBOM** every image at build (Syft → CycloneDX/SPDX).
- **Sign with cosign**, verify at admission with a policy controller.
- **Vulnerability scan at push** (Trivy, Grype, Clair, Snyk Container);
  set thresholds in CI.
- **Watch the registry** for new tags appearing under your namespace
  that you didn't push.

---

## Federal and Regulatory Anchoring

| Regulation / Memo | Year | Requires |
|-------------------|------|----------|
| **EO 14028** (US) | May 2021 | SBOM, SLSA-aligned practices, attestation |
| **OMB M-22-18 / M-23-16** (US) | 2022 / 2023 | Self-attestation by software vendors to federal agencies |
| **NIST SP 800-218 (SSDF)** | v1.1 (Feb 2022) | Secure Software Development Framework |
| **NIST SP 800-161 Rev 1** | May 2022 | C-SCRM controls (federal supply-chain risk mgmt) |
| **EU CRA** (Reg 2024/2847) | In force Dec 2024; applies Dec 2027 | Vulnerability handling, SBOM for "products with digital elements" sold in EU |

For US federal sales, **CISA Self-Attestation Form** is the operational
artifact. For EU CRA, the technical file must include vulnerability
handling and SBOM-equivalent inventories.

---

## Anti-Patterns

### "We have an SBOM, we're done"

An SBOM produced once and never regenerated, never consumed, never
correlated against a vulnerability feed, is shelfware. SBOM is a
**feed** that drives advisory triage; design the consumer side too.

### "We sign, so we're SLSA L3"

Signing is necessary, not sufficient. SLSA L3 requires *isolated build*
and *non-falsifiable provenance*. A signed build on a self-hosted runner
that workflows can SSH into is not L3.

### "We pinned dependencies"

Pinning without hash verification means re-resolving might fetch
different content for the same version (registry compromise, repackage,
yanked-and-replaced). Hash-pin where the ecosystem supports it.

### "It's open source, it's safe"

xz-utils was open-source. The malicious code was open-source. It was
read by hundreds of distros' security teams and shipped. Open-source
doesn't mean reviewed; "many eyes" is a hypothesis, not a control.

### "Our CI runs in the cloud, so it's hardened"

Hosted CI is hardened compared to a developer laptop, not compared to
a dedicated SLSA-L3 build platform. The level of isolation matters.

---

## Quick Reference Checklist

**Source side**:

- [ ] Branch protection on default branch (review, signed commits,
      no force-push)
- [ ] Internal package names reserved on public registries
- [ ] Lockfiles committed; CI refuses to re-resolve

**Build side**:

- [ ] CI pinned by SHA (actions, base images, builders)
- [ ] No long-lived secrets; OIDC federation for cloud
- [ ] Provenance attestation generated per build (SLSA v1.0+)
- [ ] Signing via Sigstore (or equivalent keyless flow)
- [ ] SBOM generated per build (CycloneDX 1.6 or SPDX 2.3+)

**Artifact side**:

- [ ] Artifact signed; signature in transparency log
- [ ] SBOM published alongside artifact
- [ ] Vulnerability scan with VEX-aware triage
- [ ] Hash recorded for downstream verification

**Deploy side**:

- [ ] Admission controller verifies signature + identity policy
- [ ] Provenance verified against expected source repo + workflow
- [ ] SBOM correlated against advisory feed; alerts wired to oncall

**ML / LLM specifics** (cross-link `llm-and-ai-security.md`):

- [ ] Model artifacts pinned by hash; safetensors preferred
- [ ] Datasets and model weights inventoried in SBOM (ML-BOM)
- [ ] Model load sandboxed (no network, no FS write)

---

## Cross-References

**Use WITH this skill**:

- `threat-modeling.md` — STRIDE applied per pipeline link
- `security-controls-design.md` — designing the verification gates
- `llm-and-ai-security.md` — model/dataset supply-chain extensions

**Use AFTER this skill**:

- `security-architecture-review.md` — concrete tool catalog (SAST/DAST/
  SCA/secrets/container scanners) and how they pair with these
  architectural controls
- `compliance-awareness-and-mapping.md` — mapping to EO 14028, NIST
  800-161, EU CRA

**Cross-faction**:

- `axiom-rust-engineering:audit`, `axiom-python-engineering:delint`
  and equivalents — ecosystem-specific tooling
- `ordis-quality-engineering:setup-pipeline` — CI/CD baseline

---

## Summary

**Supply-chain security is the architecture of provenance**: every
artifact ties back, by signature and hash, to a known build, of a known
source, by a known builder, from known dependencies. SLSA defines the
build-integrity levels, SBOM defines the inventory, Sigstore provides
the signing primitive, in-toto generalizes the chain, and VEX makes the
inventory actionable.

The 2020–2024 incident set (SolarWinds, Log4Shell, ua-parser-js, Codecov,
xz-utils) proved that *trust is transitive*. Architectural defense is
*verification at every link*, not a single signature at the end.

When in doubt: assume any single dependency, maintainer, or build node
is compromised, and check whether your pipeline still produces a
verifiable artifact. If not, harden the link that fails the test.
