---
name: containerization
description: Use when a container image is hundreds of megabytes or ships a full OS, when it runs as root, when secrets or build tokens end up baked into a layer, when image scans flood with CVEs from packages the app never calls, when builds are slow because every change busts the cache, when two builds of the same commit produce different digests, when `latest` is deployed and nobody knows what's actually running, when an unsigned image with no SBOM reaches prod, or when choosing a base image — covers multi-stage builds, distroless/Wolfi/Chainguard minimal bases, non-root, BuildKit layer caching, reproducible builds, image scanning, signing, and registry hygiene.
---

# Containerization

## The production stake

A container image is your deployable artifact, your attack surface, and your supply-chain bill of materials all in one file. A fat image that runs as root with a shell, a package manager, curl, and a build-time secret in layer 3 is not "an app that's containerized" — it is a pre-assembled toolkit for whoever gets a foothold. The moment one process is compromised, the question is what else is in the image: if the answer is `bash`, `apt`, `pip`, `git`, and the cloud credentials used to build it, the blast radius is the whole environment, not one process. Every megabyte you don't need is a CVE you'll triage at 2am, a package an attacker can pivot through, and a layer that slows every pull on every node during an incident-time scale-up.

Container hygiene is not "did it build." It is a **discipline you hold at build time**: the image contains *only the application and its runtime dependencies*, runs as a non-root user with no shell to drop into, carries no secret in any layer, builds reproducibly to a known digest, and arrives in the registry signed with an SBOM you can verify. Break any of those and you have shipped an artifact you cannot reason about.

This sheet is about holding that discipline. It is not a tour of `docker build` flags.

## The invariants

1. **Minimal contents.** The runtime image contains the app and its runtime deps — nothing else. No build toolchain, no package manager, no shell unless an explicit operational need is documented. If you can `kubectl exec ... -- bash` into prod, so can an attacker.
2. **Non-root by default.** The process runs as an unprivileged UID. Root in a container is one misconfiguration (privileged, hostPath, a kernel CVE) away from root on the node.
3. **No secrets in layers.** A layer is forever — squashing or deleting in a later layer does not remove it from history. Secrets enter at *runtime*, never at build time.
4. **Reproducible and provenanced.** The same source produces the same image (modulo signatures), and the image carries an SBOM and signed provenance so you can answer "what is in this digest and who built it."
5. **Deploy by digest, not tag.** Tags are mutable pointers; `@sha256:...` is the artifact. Production references digests.

If any of these is false, you cannot answer basic incident questions about what you're running.

## Base image selection (June 2026 reality)

The base image decides most of your CVE surface before you write a single `COPY`. Distroless is now the **baseline floor, not the frontier** — and explicitly the *weakest* of the serious minimal options, because Google's distroless tracks Debian stable, so its packages lag and accumulate known CVEs between rebuilds.

| Option | What it is | When to reach for it | Caveat |
|--------|-----------|---------------------|--------|
| **Chainguard Images** | Hardened, minimal images built on Wolfi; rebuilt within hours of an upstream CVE; native SBOM + SLSA-L3 provenance per digest | **Default when you can consume them** — lowest CVE surface, signed provenance out of the box | Free tier is `:latest` only; pinned/older tags are a paid catalog |
| **Wolfi** (via Melange + Apko) | A glibc, rolling, container-native "undistro" you compose yourself | You need to build your own minimal base, glibc compatibility, and rolling freshness | You own the build pipeline (Melange = packages, Apko = image assembly) |
| **Distroless** (`gcr.io/distroless/*`) | Google's minimal images: runtime + libs, no shell/pkg-mgr | Baseline floor; fine when you can't adopt Wolfi/Chainguard | Debian-stable lag → stale CVEs; treat as minimum acceptable, not best |
| **Alpine** | Tiny, but musl libc | Only when musl is genuinely fine (not for glibc-linked, DNS-edge-case, or some ML stacks) | musl quirks (DNS, locale, native wheels) bite in production |
| **`ubuntu` / `debian` full** | Full OS | Almost never as a *runtime* base | Hundreds of CVEs you don't use; root + shell + pkg mgr |

The ranking that matters: **Chainguard/Wolfi > distroless > slim-distro > full distro.** Choosing distroless over a full Ubuntu image is the easy 80% win; choosing Wolfi/Chainguard over distroless is the freshness win that keeps your scan clean between releases. Record the choice in an ADR — "we use distroless because we can't yet consume Chainguard's paid pinned tags" is a real, reviewable decision; "we use ubuntu:latest because that's what the tutorial had" is not.

Build with **BuildKit / `docker buildx`** regardless of base: concurrent stage execution, cache mounts, secret mounts, and multi-arch in one tool. The discipline below is base-agnostic; the syntax is BuildKit.

## Multi-stage builds — the mechanism that makes minimal possible

You cannot have a minimal runtime image and a compiler in the same stage. Multi-stage builds split *build environment* from *runtime environment*: a fat builder stage with the full toolchain compiles/installs, and a tiny final stage copies only the resulting artifact. The toolchain, the source, the build caches, and any build-time credentials never reach the shipped image because they live in a stage that is discarded.

### Example 1 — Go service: full builder, distroless runtime, non-root, no secrets

```dockerfile
# syntax=docker/dockerfile:1.7   # enables BuildKit cache & secret mounts

########## build stage — fat, has the whole toolchain ##########
FROM golang:1.24 AS build
WORKDIR /src

# Cache module downloads across builds: this mount persists between builds and
# does NOT become a layer. Editing a .go file does not re-download modules.
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

COPY . .
# Reproducible, statically linked, stripped binary.
# CGO off + -trimpath + pinned ldflags => byte-stable output for a given commit.
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOFLAGS=-trimpath \
    go build -ldflags="-s -w -buildid=" -o /out/app ./cmd/app

########## runtime stage — tiny, no shell, no toolchain, non-root ##########
# distroless static: no shell, no package manager, ships a nonroot user (65532)
FROM gcr.io/distroless/static-debian12:nonroot
# (prefer cgr.dev/chainguard/static:latest when you can consume it — fresher CVEs)

COPY --from=build /out/app /app
USER nonroot:nonroot          # invariant: do not run as root
EXPOSE 8080
ENTRYPOINT ["/app"]           # no shell form — there is no shell to invoke
```

What this image does **not** contain: Go, git, the source tree, build caches, a shell, or a package manager. An attacker who lands RCE in `/app` finds no `bash`, no `curl`, no `apt` to pull a second stage. The binary is the entire userland.

The cache mounts are the layer-caching discipline: module downloads and the compile cache live in BuildKit's cache, *not* in image layers, so a code change reuses them instead of re-downloading the world — and they never ship.

### Build-time secrets: never `COPY`, never `ARG`, never `ENV`

`ARG GITHUB_TOKEN=...` and `COPY .npmrc` bake the secret into a layer permanently. BuildKit secret mounts expose a credential to a single `RUN` and leave nothing in the image:

```dockerfile
# RIGHT — secret is mounted for one command, never persisted to a layer
RUN --mount=type=secret,id=npm_token \
    NPM_TOKEN="$(cat /run/secrets/npm_token)" npm ci --omit=dev
```

```bash
# pass it at build time from your secret store, never from a checked-in file
docker buildx build --secret id=npm_token,env=NPM_TOKEN -t app:build .
```

```dockerfile
# WRONG — both of these are now in the image forever; squashing does not save you
ARG NPM_TOKEN
RUN echo "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > .npmrc && npm ci
COPY .env /app/.env
```

The test: `docker history --no-trunc <image>` and `dive <image>` should never reveal a credential, a `.env`, or a `.npmrc`. If they do, the credential is compromised the moment that image hits a registry — rotate it, don't just rebuild.

## Layer caching and ordering — fast builds without surprises

Layers are cached top-down and invalidated by the first changed instruction; everything after it rebuilds. Order from **least-frequently-changed to most-frequently-changed**: base image, then dependency manifests, then dependency install, then application source last. Copying your whole source tree before installing dependencies means every one-line code change re-installs all dependencies.

```dockerfile
# RIGHT — deps cached independently of source
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --omit=dev   # cached unless manifests change
COPY src ./src                                               # changes here don't bust the install

# WRONG — any source edit re-runs the entire install
COPY . .
RUN npm ci --omit=dev
```

A `.dockerignore` is mandatory: without it `COPY . .` ships `.git`, `node_modules`, local `.env` files, and CI artifacts into the build context — bloating the image, leaking secrets, and busting cache on irrelevant changes.

```gitignore
# .dockerignore
.git
node_modules
**/.env
**/*.log
.github
dist
```

## Reproducible builds — same source, same digest

If two builds of the same commit produce different digests, you cannot prove what's in production matches what was reviewed, and signature/provenance verification becomes theater. Sources of non-determinism and their fixes:

- **Unpinned base** (`FROM node:22`) → pin to a **digest** (`FROM node:22@sha256:...`). Tags move; digests don't.
- **Unpinned packages** (`apt-get install curl`) → pin versions, or use lockfile-driven installs (`npm ci`, `pip install -r requirements.txt --require-hashes`, `uv sync --frozen`).
- **Embedded timestamps / build IDs** → strip them (`-buildid=` in Go, `SOURCE_DATE_EPOCH` for tooling that honors it).
- **Floating mirrors / network installs** → vendor or lock; a remote that changes changes your image.

Wolfi's Melange/Apko toolchain produces byte-reproducible images by design and is the strongest option here; for hand-written Dockerfiles, digest-pinning plus lockfiles plus `SOURCE_DATE_EPOCH` gets you most of the way.

## Scanning, SBOMs, and signing — provenance you can verify

A scan that runs after deploy is an audit log, not a gate. Scan the image **in the pipeline, fail the build on policy violations**, generate an SBOM, sign both the image and the SBOM, and verify the signature before anything pulls it.

The June 2026 standard stack:

- **Scan:** Trivy or Grype against the built image; gate on severity/policy (e.g. fail on fixable HIGH/CRITICAL).
- **SBOM:** generate with **Syft**, in **CycloneDX 1.5+** (security-workflow-friendly, native attestation) or **SPDX 2.3+** (compliance-heavy). Attach it as an attestation, not a loose file.
- **Sign:** **Sigstore / cosign 2.x**, keyless via Fulcio (short-lived certs from OIDC identity) with the signature recorded in the Rekor transparency log. Sign the image *and* attest the SBOM.
- **Provenance:** target **SLSA v1.0 Build L3** — it's the federal-procurement bar and the line where provenance becomes non-forgeable.
- **Verify in-pipeline and at admission:** `cosign verify` in CI, and a cluster admission policy (e.g. policy-controller / Kyverno) that refuses unsigned or unverified digests.

### Example 2 — build, scan, SBOM, keyless-sign, verify (GitHub Actions, BuildKit)

```yaml
# .github/workflows/image.yml — the artifact does not ship unless it passes the gates
name: build-image
on:
  push: { branches: [main] }

permissions:
  contents: read
  packages: write
  id-token: write          # OIDC: keyless cosign signing, no long-lived keys

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3       # BuildKit

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      # Build and push; capture the immutable digest — everything downstream uses it.
      - id: build
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/acme/app:${{ github.sha }}
          provenance: mode=max                    # SLSA provenance attestation
          sbom: true                              # BuildKit-native SBOM attestation

      # Independent SBOM artifact in CycloneDX (security-friendly) for the attestation.
      - uses: anchore/sbom-action@v0
        with:
          image: ghcr.io/acme/app@${{ steps.build.outputs.digest }}
          format: cyclonedx-json
          output-file: sbom.cdx.json

      # Vulnerability gate: fail the build on fixable HIGH/CRITICAL.
      - uses: aquasecurity/trivy-action@0.24.0
        with:
          image-ref: ghcr.io/acme/app@${{ steps.build.outputs.digest }}
          severity: HIGH,CRITICAL
          ignore-unfixed: true
          exit-code: "1"                          # <-- this is the gate, not a report

      - uses: sigstore/cosign-installer@v3

      # Keyless sign the image AND attest the SBOM — bound to the digest, logged in Rekor.
      - run: |
          cosign sign --yes ghcr.io/acme/app@${{ steps.build.outputs.digest }}
          cosign attest --yes --predicate sbom.cdx.json --type cyclonedx \
            ghcr.io/acme/app@${{ steps.build.outputs.digest }}

      # Verify before declaring success — prove the signature is real and from this CI identity.
      - run: |
          cosign verify \
            --certificate-identity-regexp "https://github.com/acme/app/.github/workflows/.+" \
            --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
            ghcr.io/acme/app@${{ steps.build.outputs.digest }}
```

The load-bearing parts: `exit-code: "1"` on Trivy makes the scan a *gate* (a scan that can't fail the build is decoration); keyless signing means there is no private key to leak; and `cosign verify` runs *in the same pipeline* so a green build is a verified build. Pair this with an admission controller that rejects any digest lacking a valid signature, or the verification only protects builds, not deploys.

## Registry hygiene

The registry is where artifacts live and where deploys pull from. Treat it as production infrastructure:

- **Immutable tags for releases.** A release tag must never be overwritten. Mutable tags let a re-push silently change what `:v1.4.2` means after it was signed and reviewed.
- **Deploy by digest, never `latest`.** `latest` is "whatever was pushed most recently" — non-reproducible, unverifiable, and the cause of "works on my node, not yours." Manifests and GitOps reference `@sha256:...`.
- **Retention / GC policy.** Untagged and stale digests accumulate cost and attack surface. Expire them on a schedule; keep signed release digests.
- **Least-privilege access.** CI pushes via OIDC short-lived tokens; runtime pulls read-only. No standing admin credential in a pipeline.
- **Private by default.** Don't push internal images to a public registry, and scan what you pull *in* from public registries before basing on it.

## Common mistakes

| Mistake | Why it bites | Fix |
|---------|-------------|-----|
| Full-OS runtime base (`ubuntu`, `node`, `python`) | Hundreds of unused-package CVEs; shell + pkg mgr for attackers | Distroless floor; Wolfi/Chainguard preferred |
| Single-stage build with toolchain in final image | Ships compiler, source, build caches, sometimes secrets | Multi-stage; copy only the artifact to a minimal runtime |
| Container runs as root | One misconfig/CVE → root on the node | `USER nonroot`; non-root base; drop capabilities |
| Secret via `ARG`/`ENV`/`COPY .env` | Baked into a layer forever; rebuild doesn't remove it | BuildKit `--mount=type=secret`; inject at runtime |
| `COPY . .` before installing deps | Every code change re-installs all dependencies | Copy manifests → install → copy source last |
| No `.dockerignore` | Ships `.git`, `node_modules`, `.env`; cache-busts; leaks | Maintain `.dockerignore` like a `.gitignore` |
| Unpinned base/packages (`FROM node:22`) | Non-reproducible; digest drifts; signing is meaningless | Pin to digest + lockfile installs |
| Scan runs after deploy (or just reports) | It's an audit log, not a gate | Scan in CI, `exit-code: 1` on fixable HIGH/CRITICAL |
| No SBOM / no signature | Can't answer "what's in this digest, who built it" | Syft SBOM + cosign keyless sign + verify in-pipeline |
| Deploying `:latest` | Non-reproducible; unverifiable; signature can't bind | Deploy by `@sha256:` digest |
| Mutable release tags | A re-push changes what a reviewed/signed tag means | Immutable tags; release = digest |
| "Distroless = best, we're done" | Debian-stable lag = stale CVEs between rebuilds | It's the floor; Wolfi/Chainguard for freshness |

## Red flags — STOP

- "Just put the token in an `ARG`, it's only at build time." → It's in the layer forever and the image will hit a registry. Use a secret mount.
- "We need a shell in prod to debug." → That shell is also the attacker's shell. Use an ephemeral debug container, not a fat image.
- "Run it as root, permissions are easier." → Root in a container is a node-takeover waiting for one CVE. Fix the permissions, run non-root.
- "Tag it `latest` and deploy that." → You can't reproduce, verify, or roll back to a thing called "most recent." Deploy the digest.
- "The scan can run nightly after we ship." → Then it never gates anything. Gate in the build or don't bother.
- "It's distroless, the image is fine." → Distroless is the floor. Check its CVE age; Wolfi/Chainguard if it's stale.
- "We'll add signing later." → Until then nobody can prove the image in prod is the one you built. Sign now; it's a one-line keyless step.
- "Squash the layers, the secret's gone." → It's in build history regardless. Rotate the secret and stop baking it in.

## Rationalizations and their counters

- **"A bigger base is more convenient to debug."** Convenience for you at build time is convenience for an attacker at runtime. Debug with ephemeral containers (`kubectl debug`), keep the image minimal.
- **"Pinning digests is annoying, I want updates."** Updates come from rebuilding against a fresh, *scanned* base in CI — not from a tag silently moving under your signed artifact.
- **"Reproducible builds are academic."** They're what makes signing and SBOMs mean anything. Without reproducibility, "verify this is what we built" has no answer.
- **"Scanning will block our releases."** It blocks exactly the releases carrying fixable critical CVEs. That is the feature, not the bug.
- **"Distroless already solved our CVE problem."** It solved the full-OS problem. Run a scan on a month-old distroless image and watch the Debian-stable lag show up.
- **"Non-root breaks our file writes."** It breaks the assumption that you can write anywhere; fix the paths (writable volume, `WORKDIR` ownership) rather than handing the process root.

## The bottom line

The image is the artifact, and an artifact you cannot reason about is a liability you've deployed at scale. Hold the invariants: minimal contents (Wolfi/Chainguard over distroless over full-OS), non-root, no secret in any layer, reproducible to a digest, scanned-and-signed-and-verified before it ships, deployed by digest from a registry you treat as production. Hold those, and a compromised process finds an empty userland and a postmortem finds an SBOM. Break them, and you've handed out a root-running toolkit with your build credentials inside, and you'll find out which CVE mattered the night it's exploited.

## Cross-references

- `ci-cd-pipeline-architecture` (this pack) — where the build/scan/sign/verify gates live in the pipeline.
- `deployment-strategies` (this pack) — deploying signed digests via progressive delivery.
- `infrastructure-as-code` (this pack) — registry, admission policy, and OIDC roles as code.
- `/ordis-security-architect` — supply-chain threat modeling, admission control, secret management posture.
- `/axiom-solution-architect` — recording the base-image and signing-stack decisions as ADRs.
