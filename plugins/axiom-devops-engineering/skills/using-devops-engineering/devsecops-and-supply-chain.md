---
name: devsecops-and-supply-chain
description: Use when artifacts ship unsigned, when dependencies reach production unscanned, when a CI runner has admin/cloud-owner credentials or a self-hosted runner runs untrusted PRs, when there is no SBOM and a CVE drops with no way to tell which images are affected, when security is a manual review at the end instead of a gate in the pipeline, when "pin your dependencies" is aspirational, when a typosquatted or compromised package could be pulled at build time, when secrets leak through build logs or image layers, when SAST/DAST/SCA findings are warnings nobody blocks on, when policy is a wiki page instead of code, or when you cannot prove how a deployed artifact was built (provenance/SLSA), who signed it, or what is inside it.
---

# DevSecOps and Software Supply Chain

**The artifact running in your production cluster right now — can you prove who built it, from which source commit, on which runner, with which dependencies, and that nobody altered it between build and deploy? If the answer is "we trust the registry," you do not have a supply chain, you have a faith-based delivery system. The expensive failure class here is not a clever zero-day; it is mundane and self-inflicted: a transitive dependency you never scanned ships a known-critical CVE for six weeks because nobody had an SBOM to grep; a CI runner with `AdministratorAccess` gets a malicious PR that exfiltrates the cloud account; an unsigned image is swapped at the registry and the cluster runs it without a blink. Every one of these is preventable by a gate that runs in seconds. DevSecOps is not a security team's job bolted on at the end — it is the engineering discipline of making "secure" the *default path* and the *only path that passes the pipeline*. Shift-left is not a slogan; it is the recognition that a finding caught in a PR costs a comment, and the same finding caught in production costs an incident bridge.**

This is engineering discipline, not a tool tour. Tools rotate; the gates do not. The job is to make every artifact **scanned, signed, attested, and produced under least privilege** — and to make the pipeline *refuse to promote* anything that is not.

> **Scope split with `ci-cd-pipeline-architecture`.** That sheet owns the *deploy* gate sequence (build-once, progressive delivery, rollback) and touches signing/SBOM as one stage. This sheet owns the *security* discipline end to end: shift-left scanning (SAST/DAST/SCA), SBOM as an asset, provenance/SLSA, signing + verification, policy-as-code, and least-privilege CI. Where they meet (cosign, SBOM), this sheet goes deeper. Read both.

## The supply chain is the whole path, not the registry

Every link is an attack surface and every link needs a control:

```
developer ─▶ source ─▶ deps ──▶ build ───▶ artifact ─▶ registry ─▶ admission ─▶ runtime
  (MFA,     (branch    (SCA,   (isolated,  (sign +    (immutable, (verify    (least
  signed    protect,   pin,    least-priv  SBOM +     digest-     sig +      privilege,
  commits)  CODEOWNERS) lock)  runner)     provenance) pinned)     policy)    drop caps)
```

Two framing facts drive everything below:

1. **Most breaches enter through the boring links** — an unscanned dependency, a leaked CI token, an unsigned image — not through novel exploits. Defend the boring links first.
2. **A control you do not *enforce* is documentation, not a control.** A scanner that warns-and-continues, a signature you never verify, a policy in a wiki — these create the *appearance* of security and the *reality* of none. Every control in this sheet ends in a **gate that fails the pipeline** and, where possible, a second enforcement point at admission (defense in depth, so `kubectl apply` from a laptop cannot bypass CI).

## Shift-left: SAST, DAST, SCA as PR gates

"Shift-left" means the finding surfaces while the developer still has the code in their head and the change is one diff — not weeks later in a pen-test report. Three complementary lenses, each a distinct gate:

| Lens | What it sees | When it runs | Tools (current) |
|---|---|---|---|
| **SAST** (static) | Code-level flaws: injection, unsafe deserialization, hardcoded secrets, path traversal | PR, on the diff | Semgrep, CodeQL, language linters with security rules |
| **SCA** (composition) | Known-vulnerable *dependencies* (direct + transitive), license risk | PR + nightly (CVEs land after you merge) | Grype, Trivy, OSV-Scanner, Dependabot/Renovate |
| **DAST** (dynamic) | Runtime behaviour: auth bypass, injection against a *running* instance | Against staging, on release | OWASP ZAP, Nuclei, Burp (gated) |
| **Secret scanning** | Credentials in source/history/logs | Pre-commit + PR + push-protection | gitleaks, trufflehog, platform push-protection |

Discipline points that separate a real gate from theatre:

- **Scan the diff on PRs, the whole tree nightly.** Diff-scoped SAST keeps PR feedback fast; full + SCA nightly catches CVEs disclosed *after* you merged (the dependency you shipped clean on Monday is critical on Friday — only a recurring scan finds that).
- **Fail on severity, not on "any finding."** Block the merge on new High/Critical; track Medium/Low as debt. A gate that blocks on everything gets disabled within a week; calibrate so it blocks on what matters.
- **Baseline existing findings, gate on *new* ones.** Adopting SAST on a mature repo with a hard gate on all findings is how teams turn the gate off. Snapshot the existing set, fail only on newly-introduced issues, burn the baseline down deliberately.
- **Secret scanning needs push-protection AND history scan.** Blocking the push is good; the secret already in git history is still live — scan history and rotate anything found.

```yaml
# .github/workflows/security.yml — shift-left gates on the PR.
# Least privilege by default: read-only token, scoped up only where needed.
permissions:
  contents: read

on:
  pull_request:
  schedule:
    - cron: "0 6 * * *"   # nightly full + SCA — catches post-merge CVE disclosures

jobs:
  sast:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write   # to upload SARIF — nothing more
    steps:
      - uses: actions/checkout@v4
      - name: Semgrep (diff-scoped on PR, full on schedule)
        uses: semgrep/semgrep-action@v1
        with:
          config: "p/default p/secrets p/owasp-top-ten"
        env:
          # GATE: nonzero exit (a blocking finding) fails the job → blocks merge.
          SEMGREP_BASELINE_REF: ${{ github.event.pull_request.base.sha }}

  sca:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4
      - name: Grype dependency scan — fail on High+
        uses: anchore/scan-action@v4
        with:
          path: "."
          fail-build: true          # GATE
          severity-cutoff: high     # block High/Critical; track the rest as debt

  secrets:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0            # full history, not just the tip
      - name: gitleaks
        uses: gitleaks/gitleaks-action@v2   # GATE: any verified secret fails the run
```

## Dependencies: pin, lock, verify provenance

Unscanned dependencies are the single most common supply-chain entry point. The discipline is three layered:

1. **Lock and pin.** A lockfile (`uv.lock`, `Cargo.lock`, `package-lock.json`, `go.sum`) pins the full transitive tree to exact versions *and hashes*. Build with `--frozen` / `npm ci` / `--locked` so a build *fails* rather than silently resolving a new version. An unpinned build is non-reproducible and a window for dependency-confusion and typosquat attacks.
2. **Scan continuously, update on a cadence.** Renovate/Dependabot to surface updates; SCA on every PR *and* nightly so a newly-disclosed CVE in an existing dep is caught even with no code change. Auto-merge patch bumps that pass CI; review majors.
3. **Verify provenance where the ecosystem supports it.** Prefer registries/packages that publish signed provenance (npm provenance attestations, PyPI trusted-publishing/attestations, Sigstore-signed artifacts). Verifying the publisher closes the "someone pushed a malicious version under a name you trust" gap.

Defensive details that bite when skipped:

- **Dependency confusion:** if you have internal packages, configure the registry/scope so a public package of the same name *cannot* shadow your private one. Pin the registry, do not let resolution fall through to the public index for internal names.
- **Allowlist, not just denylist, for licenses.** SCA tools enforce license policy; default to an allowlist of approved licenses so a copyleft or unknown license can't enter unnoticed.
- **Vendoring/mirroring** for critical builds: pull from an internal proxy (Artifactory/Nexus/zizmor-checked actions) so an upstream takedown or compromise can't break or poison a release mid-flight.

## SBOM: the artifact you grep when the next CVE drops

A Software Bill of Materials is the dependency manifest of a *built artifact* — every package, version, and hash actually inside the image, not just what the source declares. Its value is realized at exactly one moment: a critical CVE is announced, and the question is "which of our 300 running images contain the vulnerable package?" With SBOMs you answer in minutes by querying; without them you spend two days rebuilding and re-scanning under incident pressure.

Format choice (currency, June 2026):

- **CycloneDX 1.5+** — security-workflow-friendly, native support for VEX and SLSA attestation. Default for security-driven estates.
- **SPDX 2.3+** — the compliance/legal lingua franca; default where procurement or legal consumes it.

Generate with **Syft**, attach it to the image as a *signed* attestation (not a loose file in a bucket), and re-scan stored SBOMs against the CVE feed continuously — a CVE that lands tomorrow against an image you shipped today is found by re-scanning yesterday's SBOM, with no rebuild.

```bash
# Generate an SBOM from the BUILT image (what actually shipped), not the source tree.
syft registry.example.com/app@${DIGEST} -o cyclonedx-json > sbom.cdx.json

# Re-scan the SBOM (not the image) against today's CVE feed — fast, no pull/rebuild.
grype sbom:sbom.cdx.json --fail-on high

# Pair with VEX so "present but not exploitable" findings don't drown the real ones.
grype sbom:sbom.cdx.json --vex vex.json --fail-on high
```

## Provenance and SLSA: prove how it was built

SBOM answers *what is inside*. **Provenance** answers *how it was made* — which source commit, which builder, which parameters — as a signed, tamper-evident attestation. The framework is **SLSA v1.0**; the bar that matters is **Build Level 3** (the U.S. federal-procurement floor):

| SLSA Build Level | Guarantee | What it demands |
|---|---|---|
| L0 | Nothing | — |
| L1 | Provenance exists | Build emits a provenance document |
| L2 | Signed provenance, hosted build | Authenticated builder, service-generated provenance |
| **L3** | **Non-falsifiable provenance, isolated build** | **Hardened, isolated build platform; provenance the build *cannot* forge; no ambient secrets to the build steps** |

L3 is achievable on common platforms with the SLSA GitHub generator or a hardened build service — you do not need a bespoke build farm. The point of provenance is **verification in the pipeline and at admission**: before promotion, assert the artifact's provenance shows it came from *your* repo, *your* workflow, *your* trusted builder — not a developer's laptop, not a forked PR runner.

## Signing and verification: cosign keyless, then enforce at admission

An unsigned artifact is unauthenticated — anyone with registry write can substitute it and your cluster will run the substitute. Sign the **image and the SBOM and the provenance**, and — this is the half teams skip — **verify before every promotion and at admission**. A signature nobody checks is decoration.

Use **Sigstore/cosign 2.x keyless**: a short-lived certificate from Fulcio bound to your CI's OIDC identity, recorded in the Rekor public transparency log. No long-lived signing key to leak, rotate, or have stolen — the identity *is* the workflow.

```yaml
# Sign image + SBOM + provenance keyless, then GATE on verification.
permissions:
  id-token: write     # REQUIRED for cosign keyless (OIDC) — nothing broader
  contents: read

jobs:
  sign:
    runs-on: ubuntu-latest
    steps:
      - uses: sigstore/cosign-installer@v3
      - name: Sign image + attach SBOM + provenance (keyless via Fulcio/Rekor)
        run: |
          cosign sign --yes registry.example.com/app@${DIGEST}
          cosign attest --yes --type cyclonedx \
            --predicate sbom.cdx.json registry.example.com/app@${DIGEST}
          cosign attest --yes --type slsaprovenance \
            --predicate provenance.json registry.example.com/app@${DIGEST}

  verify-gate:
    needs: sign
    runs-on: ubuntu-latest
    steps:
      - uses: sigstore/cosign-installer@v3
      - name: GATE — refuse to promote unless signed by THIS repo's workflow
        run: |
          cosign verify \
            --certificate-identity-regexp "https://github.com/${{ github.repository }}/.*" \
            --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
            registry.example.com/app@${DIGEST}
          cosign verify-attestation --type cyclonedx \
            --certificate-identity-regexp "https://github.com/${{ github.repository }}/.*" \
            --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
            registry.example.com/app@${DIGEST}
```

Then enforce the *same* assertion at the cluster boundary, so nothing unsigned ever schedules — even a manual apply:

```yaml
# Kyverno ClusterPolicy: admission-time signature + attestation enforcement.
# Defense in depth — the pipeline signs, the cluster refuses anything unsigned.
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-signed-images
spec:
  validationFailureAction: Enforce   # NOT Audit — block, don't just log
  rules:
    - name: verify-cosign-signature
      match:
        any:
          - resources: { kinds: [Pod] }
      verifyImages:
        - imageReferences:
            - "registry.example.com/*"
          attestors:
            - entries:
                - keyless:
                    issuer: "https://token.actions.githubusercontent.com"
                    subject: "https://github.com/your-org/app/*"
                    rekor:
                      url: https://rekor.sigstore.dev
          # Require the SBOM attestation to be present AND signed by the same identity.
          attestations:
            - type: https://cyclonedx.org/bom
              attestors:
                - entries:
                    - keyless:
                        issuer: "https://token.actions.githubusercontent.com"
                        subject: "https://github.com/your-org/app/*"
```

## Policy-as-code: gates that are version-controlled and tested

Security policy that lives in a runbook is unenforced and drifts. Encode it as code — **OPA/Rego**, **Kyverno**, or **Conftest** — so it is reviewed in PRs, tested, and executed automatically at the gate that matters (PR, admission, or terraform-plan).

```rego
# policy/ci.rego — Conftest over a rendered manifest. Tested like any other code.
package main

deny[msg] {
  input.kind == "Pod"
  c := input.spec.containers[_]
  endswith(c.image, ":latest")
  msg := sprintf("container %q uses a mutable :latest tag — pin by digest", [c.name])
}

deny[msg] {
  input.kind == "Pod"
  c := input.spec.containers[_]
  not c.securityContext.runAsNonRoot
  msg := sprintf("container %q may run as root — set runAsNonRoot: true", [c.name])
}

deny[msg] {
  input.kind == "Pod"
  c := input.spec.containers[_]
  c.securityContext.privileged
  msg := sprintf("container %q is privileged — forbidden", [c.name])
}
```

Policy discipline: keep policies in version control next to the code they govern; write **unit tests for the policies themselves** (`conftest verify`, `opa test`) so a loosened rule is a reviewable diff; run the *same* policy bundle in CI and at admission so there is one source of truth, not two that drift.

## Least-privilege CI: the runner is your softest target

CI runners are catnip for attackers — they hold deploy credentials, run code from PRs, and are often configured for convenience. An over-privileged runner turns one malicious PR or one compromised action into account takeover. This is where most "supply chain" incidents actually land.

The non-negotiables:

- **Default-deny token scope.** Set the workflow token to `contents: read` at the top and grant the *minimum* per job (`id-token: write` only on the signing job, `packages: write` only on the publish job). A repo-wide write token on every job is an exfiltration primitive.
- **OIDC / workload-identity federation, never long-lived cloud keys.** The runner exchanges its short-lived OIDC token for a scoped, short-lived cloud credential. Nothing durable to steal from a leaked log or a compromised step.
- **Never run untrusted code with secrets.** `pull_request_target` and self-hosted runners executing fork PRs are the classic exfiltration path — a PR adds a step that prints `${{ secrets.* }}`. Fork PRs run with read-only tokens and no secrets; require approval before any privileged workflow runs on external contributions.
- **Pin actions/images by full commit SHA, not a tag.** `uses: foo/bar@v1` follows a movable tag the upstream (or an attacker who compromised it) can repoint. Pin to a 40-char SHA; let Renovate bump it through a reviewed PR.
- **Ephemeral, isolated runners.** Fresh runner per job, no shared state, no persistent caches that a prior job could poison. For self-hosted, ephemeral VMs/containers — never a long-lived box accumulating credentials.
- **Egress control on runners** where the threat model warrants it — a build step has no business calling an arbitrary IP; restrict outbound so an exfiltration attempt fails closed.

```yaml
# Least-privilege deploy job: OIDC → short-lived cloud creds, no static keys,
# actions pinned by SHA, token scoped to exactly what this job needs.
permissions:
  contents: read          # default-deny at the top

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write      # ONLY to mint the OIDC token for cloud federation
      contents: read
    steps:
      - uses: actions/checkout@a5ac7e51b41094c92402da3b24376905380afc29  # v4, SHA-pinned
      - name: Federate to cloud (no long-lived keys anywhere)
        uses: aws-actions/configure-aws-credentials@e3dd6a429d7300a6a4c196c26e071d42e0343502  # v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/ci-deploy   # narrowly scoped
          aws-region: us-east-1
          # No aws-access-key-id / secret — federation only.
```

## Worked example: the unsigned-image incident, and the gate that stops it

A team ships images tagged `app:prod`. An attacker with leaked registry write pushes a backdoored image under the same tag. The next pull serves the backdoor; nobody notices for days because the cluster trusts the registry implicitly. **Three independent controls from this sheet each stop it cold:**

1. **Digest pinning** (deploy by `@sha256:...`, never `:prod`) — the cluster pulls the exact bytes that were verified, not whatever the tag now points to.
2. **In-pipeline `cosign verify`** — promotion fails because the swapped image is not signed by the repo's workflow identity.
3. **Admission policy (Kyverno)** — even a manual `kubectl apply` of the backdoored image is rejected at the API server because it carries no valid keyless signature.

The lesson is the theme of the whole sheet: **one signature, verified at two boundaries, defeats an attacker who fully owns the registry.** Defense in depth means no single control is load-bearing alone.

## Common mistakes

| Mistake | Why it bites | Fix |
|---|---|---|
| Security review only at the end | Findings arrive after the design is set; expensive to fix, easy to wave through | Shift-left: SAST/SCA/secret gates on every PR |
| Scanner warns and continues | Findings accumulate; nobody reads the wall of warnings | Gate the pipeline on new High/Critical; baseline the rest |
| Hard gate on all findings, day one | Team disables the gate within a week | Baseline existing, fail only on *newly introduced* issues |
| Dependencies pinned but never re-scanned | A CVE disclosed after merge ships silently | Nightly SCA on the full tree, re-scan stored SBOMs vs the feed |
| Unpinned / unlocked builds | Non-reproducible; window for confusion/typosquat | Lockfile + hashes; `--frozen` / `npm ci` / `--locked` |
| No SBOM | A CVE drops and you can't tell which images are affected | Syft SBOM per artifact, signed attestation, continuous re-scan |
| Unsigned artifacts | Anyone with registry write substitutes the image | cosign keyless sign image+SBOM+provenance |
| Signed but never verified | A signature nobody checks is decoration | `cosign verify` gate in pipeline *and* admission policy |
| No provenance | Can't prove the artifact came from your build, not a laptop | SLSA Build L3; verify provenance identity before promote |
| Repo-wide write token on every job | One malicious step/PR exfiltrates everything | `contents: read` default; least scope per job |
| Long-lived cloud keys in CI | Leaked once = persistent account access | OIDC/workload-identity federation, short-lived creds |
| Fork PRs run with secrets | A PR prints `${{ secrets.* }}` and steals them | Read-only token, no secrets, approval gate for forks |
| Actions/images pinned by tag | Upstream (or attacker) repoints the tag | Pin by 40-char commit SHA; Renovate bumps via PR |
| Policy in a wiki | Drifts, unenforced, unreviewable | Policy-as-code (OPA/Kyverno/Conftest), tested, gated |
| `validationFailureAction: Audit` left in place | Violations are logged, never blocked | `Enforce` for the controls that matter |
| Secrets only push-protected | The secret already in history is still live | Scan full history; rotate everything found |

## Red flags — STOP

Say (or hear) any of these and a supply-chain incident is being set up. Stop and fix the gate first.

- "Just turn off the scanner, it's blocking the release." → Then you're shipping a known vulnerability knowingly. Fix it or file an explicit, signed-off risk acceptance — never a silent disable.
- "We'll sign images later." → Later is during the incident where someone swapped one. Signing's whole value is that it exists *before* the swap.
- "The CI runner needs admin so it can deploy." → No. It needs a *scoped, short-lived* role via OIDC. Admin on a runner is account takeover waiting for one bad PR.
- "Pin actions by SHA? The tag is fine." → A tag is mutable; SHA is the artifact. Tag-pinning is trusting the upstream not to be compromised — ever.
- "We don't need an SBOM, we know our deps." → You know your *direct* deps. The CVE will be three levels down in a transitive you've never heard of.
- "Let the fork PR run the full pipeline, it's a contributor." → That's how secrets get exfiltrated. Fork PRs get read-only, no secrets, full stop.
- "Verification at admission is overkill, the pipeline already signs." → The pipeline can be bypassed by a manual apply. Admission is the boundary that can't be.
- "Set the policy to Audit so it doesn't break anything." → Audit blocks nothing. If it's worth a policy, it's worth Enforce.
- "Use `:latest`, we'll pin it before prod." → Mutable tags are unreproducible and swappable. Digest-pin from the first build.

## Counters to the rationalizations

| Rationalization | Counter |
|---|---|
| "Security slows down delivery." | A finding caught in a PR costs a comment; the same finding in prod costs an incident bridge, a CVE disclosure, and a customer email. The gate is the *fast* path — it moves the cost left where it's cheap. |
| "We're too small to be a target." | Supply-chain attacks are largely *untargeted* — automated scans for leaked tokens, dependency confusion, typosquats hit everyone. Small teams with weak controls are the *easiest* targets, not exempt ones. |
| "Signing/SBOM is compliance theater." | It's the difference between "we identified and patched every affected image in 20 minutes" and "we spent two days rebuilding everything to find out what's vulnerable." The SBOM is your incident map; provenance is your tamper alarm. |
| "Our deps are fine, we vet them." | You vetted the *version you added*. SCA exists because a dep you vetted last quarter gets a Critical CVE this week, and a transitive five levels down was never vetted by anyone. |
| "OIDC federation is complicated; static keys just work." | Static keys "just work" right up until one leaks in a log, and then they work for the attacker indefinitely. OIDC is a one-time setup that removes the entire class of stolen-long-lived-credential incidents. |
| "We'll add policy-as-code after we ship." | Policy added after ship is policy applied to a fleet already in violation — now it's a migration project, not a default. Encode it before the first deploy, when compliance is free. |
| "The pipeline already checks this, admission is redundant." | Redundant controls are the *definition* of defense in depth. The redundancy is the point: it's what survives when the primary control is bypassed, misconfigured, or skipped under pressure. |
| "Least privilege on CI is a hassle for developers." | The hassle is bounded and one-time; the breach from an over-privileged runner is unbounded and recurring. Scoped tokens are the cheapest insurance in the entire stack. |

## Cross-references

- This pack's `ci-cd-pipeline-architecture` — the deploy gate sequence (build-once, progressive delivery, rollback) into which these security gates slot.
- `/security-architect` — threat-modeling the build/deploy path as an attack surface; designing the controls (this sheet *implements* them in the pipeline).
- `/quality-engineering` — wiring SAST/SCA/DAST stages into the test pipeline (setup-pipeline) and keeping gates non-flaky.
- `/python-engineering`, `/rust-engineering` — language-level dependency locking, hashing, and lint-based SAST rules at the source.
- `/system-architect` — when supply-chain pain is actually an architecture problem (sprawling deps, no internal proxy, no trust boundaries).

## Quick checklist

- [ ] SAST on every PR, diff-scoped; full tree nightly — gated on new High/Critical
- [ ] SCA on every PR *and* nightly (catches post-merge CVE disclosures); license allowlist
- [ ] Secret scanning: pre-commit + push-protection + full-history scan; rotate anything found
- [ ] Dependencies locked + hash-pinned; builds `--frozen`/`npm ci`/`--locked`; confusion-proof for internal names
- [ ] DAST against staging on release, gated
- [ ] SBOM (CycloneDX 1.5+/SPDX 2.3+) per artifact via Syft, signed attestation, continuously re-scanned
- [ ] Provenance to SLSA Build L3; identity verified before promotion
- [ ] Image + SBOM + provenance signed with cosign keyless (Fulcio/Rekor)
- [ ] `cosign verify` gate in-pipeline AND admission policy (Kyverno/cosign controller) in Enforce mode
- [ ] Policy-as-code (OPA/Kyverno/Conftest), version-controlled, unit-tested, same bundle in CI + admission
- [ ] CI token `contents: read` by default; minimum scope per job
- [ ] OIDC/workload-identity federation; zero long-lived cloud keys in CI
- [ ] Fork PRs: read-only token, no secrets, approval gate
- [ ] Actions/base images pinned by 40-char SHA; bumped via reviewed Renovate PRs
- [ ] Ephemeral, isolated runners; egress controlled where warranted

## The bottom line

A secure supply chain is one where every artifact is **scanned** (SAST/DAST/SCA, before it merges), **inventoried** (SBOM you can grep), **provable** (SLSA provenance you can verify), and **signed** (cosign keyless, checked at two boundaries) — produced by a CI runner that holds **only short-lived, minimal credentials** and runs **no untrusted code with secrets**. Every one of those is a *gate that fails the pipeline*, not a warning. Skipping them to "move fast" buys you exactly one quiet quarter and then the dependency-CVE scramble, the swapped-image incident, or the leaked-CI-token account takeover — each of which costs more than every gate combined. **Secure-by-default IS the fast path: it's the only way to ship continuously without periodically handing your build system to a stranger.**
