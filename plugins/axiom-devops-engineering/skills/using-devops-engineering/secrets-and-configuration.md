---
name: secrets-and-configuration
description: Use when an API key is hardcoded in source, when a database password is in a committed .env or docker-compose.yml, when secrets are baked into a Docker image layer, when CI logs print a token, when nobody can rotate a credential because they don't know where it's used, when the same admin password has lived in prod for two years, when staging and prod read config from different places, when a leaked key forces an emergency rotation, when service accounts have far more access than they use, or when deciding what is "config" versus what is a "secret" — covers vaults (Vault / cloud KMS / secrets managers), rotation, least-privilege, dynamic short-lived credentials, config-vs-secret separation, 12-factor config, and keeping secrets out of images and repos.
---

# Secrets and Configuration

## The production stake

A secret committed to git is a secret you must treat as already compromised — and rewriting history does not un-leak it, because clones, forks, CI caches, and the attacker's `git clone` already have it. The recovery from "an AWS key hit a public repo" is not "delete the commit." It is: rotate the key immediately, audit every action that key could have taken in the window it was live, assume the blast radius is everything that key could reach, and explain to someone why the bill spiked or the data left. The cost of a leaked credential is not the leak; it is everything the credential could *do*.

This is why secrets management is **engineering discipline, not a tool you install**. The invariant you hold is simple and absolute: **a secret exists in exactly one authoritative store, is delivered to workloads at runtime, is never written to a repo / image layer / CI log, is scoped to the least it needs, and can be rotated without a code change or a redeploy.** The day a plaintext secret lives in `.env`, a `Dockerfile`, a Helm `values.yaml`, or a CI variable printed to a log, you no longer have secrets management — you have secrets *somewhere*, and "somewhere" is the attack surface.

This sheet is about holding that invariant. It is not a tour of Vault's UI.

## Config versus secret — the distinction the whole discipline rests on

Get this wrong and everything downstream is wrong: you either leak config noise into a vault (operationally painful, expensive) or, far worse, ship a secret as if it were config (catastrophic).

- **Config** is *non-sensitive deployment-varying input*: feature flags, log level, timeouts, the *hostname* of a database, the bucket name, the region, the replica count. Leaking it is embarrassing at most. It belongs in environment variables, ConfigMaps, or a config service.
- **Secret** is *anything whose disclosure causes harm*: passwords, the database **connection string with credentials**, API keys, OAuth client secrets, private keys, signing keys, TLS private keys, encryption keys, session secrets. It belongs in a secrets store and *nowhere else*.

The test: **"If this string appeared in a public Slack channel, would I have to rotate something or call legal?"** If yes, it is a secret. The database *host* is config; the database *password* is a secret; the *connection URL* that splices them together is a secret (because it contains the password) — so store the password as a secret and assemble the URL at runtime.

A surprising amount of leakage comes from treating a compound value (a connection string, a webhook URL with a token in the path) as config because "most of it is just config." If any part is sensitive, the whole value is a secret.

## 12-factor config: read config from the environment, not from the build

The relevant 12-factor principle (factor III): **strict separation of config from code, with config read from the environment at runtime**. The corollary is the litmus test for whether you've done it right:

> Could you open-source the codebase *right now* without leaking any credential?

If the answer is no, config and secrets are entangled into the code. The same artifact (image, binary) must be promotable unchanged from dev to staging to prod — only the injected config and secrets differ. This is what makes the thing you tested in staging *the same thing* that runs in prod. Bake an environment name or a key into the image and you've built a different artifact per environment, which means you never actually tested the prod one.

What this rules out, concretely:
- No `config.prod.json` with real secrets checked in.
- No `ENV SECRET_KEY=...` in a Dockerfile (it's in the layer forever — see below).
- No "build the prod image" step that differs from the staging image except by config baked in.
- No per-environment branches selecting hardcoded credentials.

## Where secrets actually live: vault, cloud secrets manager, or KMS

You have three tiers of store; pick by what you're protecting and what you already run.

| Store | What it is | Reach for it when |
|-------|-----------|-------------------|
| **HashiCorp Vault** / **OpenBao** | Identity-based secrets engine; static *and* **dynamic** secrets, leasing, revocation, transit encryption | You want dynamic short-lived credentials, multi-cloud, fine-grained policy, and you can run it. OpenBao is the MPL-2.0 fork if you need an open license (Vault is BSL 1.1). |
| **Cloud secrets manager** (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault) | Managed store, native IAM integration, built-in rotation hooks | You're single-cloud and want zero ops on the store itself. The default for most teams. |
| **Cloud KMS** (AWS KMS, GCP KMS, Azure Key Vault keys) | Key-management for *encryption keys*, envelope encryption | You're encrypting data/secrets at rest and never want the key material to leave the HSM. KMS holds keys; a secrets manager holds secrets — they compose (KMS encrypts the secrets manager's data). |

**KMS is not a secrets store and a secrets store is not a key manager.** KMS gives you a key you can encrypt/decrypt *with* but ideally never see; a secrets manager hands you the secret value. Use KMS to encrypt the things the secrets manager stores, and to back envelope encryption (e.g. SOPS, sealed-secrets). Don't put your database password directly in KMS as ciphertext blobs you manually decrypt — that's a secrets manager's job.

For Kubernetes specifically: do **not** treat the native `Secret` object as a vault. A k8s `Secret` is base64 (not encryption) and, unless you've explicitly enabled etcd encryption-at-rest, sits in plaintext in etcd readable by anyone with cluster-level access. Use the **External Secrets Operator** (ESO) to sync from a real store into the cluster, or the **Secrets Store CSI driver** to mount them — and enable etcd encryption regardless.

## Dynamic, short-lived credentials — the practice that makes leaks survivable

A static credential is a standing liability: it is valid until someone notices the leak and rotates it, which is usually *after* the damage. The strongest move available is to stop having static credentials at all.

- **Dynamic secrets** (Vault/OpenBao database secrets engine): the app asks the vault for a database credential at startup; the vault *creates a real DB user on the fly* with a short TTL, hands it over, and revokes it when the lease expires. A leaked credential is worthless within minutes, and every credential is unique-per-workload, so a leak is *attributable*.
- **Workload identity / OIDC federation** instead of stored cloud keys: CI runners, k8s pods, and lambdas assume a role via short-lived tokens (GitHub Actions OIDC → AWS role, GKE Workload Identity, IRSA on EKS). There is no long-lived `AWS_SECRET_ACCESS_KEY` to leak because there is no long-lived key.
- **Per-environment, per-service scoping**: prod's secret path is unreadable from staging's identity. The blast radius of a compromised workload is bounded by its policy, not by "whatever the shared key could touch."

The mental shift: stop asking "where do I store this long-lived secret safely" and start asking "can this credential be short-lived and minted on demand instead." When yes, most of the storage problem evaporates.

## Least-privilege — scope the secret *and* what it grants

Two separate least-privilege questions, both mandatory:

1. **Who can read the secret?** Only the workload that needs it, via its identity. Not "all of CI," not "the whole namespace," not a shared human account.
2. **What can the secret *do*?** A database credential that only needs `SELECT` on three tables should not be the `root`/owner login. A cloud key that uploads to one bucket should have an IAM policy for exactly that, not `s3:*`.

Over-privilege is the multiplier on every leak: the same key leaking is a footnote or a front-page breach depending entirely on what it could reach. The discipline is to scope down *before* the incident, because after the incident is too late.

## Example 1 — keeping secrets out of the image, injecting at runtime (BuildKit + runtime env)

The single most common image-layer leak: `ENV` or `COPY` of a secret. **Every layer is immutable and inspectable** — `docker history`, `dive`, or just `docker save | tar` recovers it, even if a later layer "deletes" the file. A secret in any layer is in the image forever.

```dockerfile
# WRONG — secret is baked into a layer, recoverable from the published image forever
FROM cgr.dev/chainguard/python:latest
ENV DATABASE_PASSWORD=s3cr3t-prod-password      # in the layer, leaked to anyone who pulls
COPY prod.env /app/.env                          # same — the file lives in history
```

```dockerfile
# RIGHT — no secret in any layer. Build-time secrets use BuildKit mounts (not persisted);
# runtime secrets are injected by the platform, never present at build.
# syntax=docker/dockerfile:1.7
FROM cgr.dev/chainguard/python:latest-dev AS build
WORKDIR /app
COPY requirements.txt .

# A build-time-only secret (e.g. a private package-registry token) via BuildKit mount:
# it is available during this RUN only and is NOT written to any layer.
RUN --mount=type=secret,id=pip_token \
    PIP_INDEX_URL="https://$(cat /run/secrets/pip_token)@pkgs.internal/simple" \
    pip install --no-cache-dir -r requirements.txt --target /app/deps

FROM cgr.dev/chainguard/python:latest          # distroless-class, minimal, native SBOM
WORKDIR /app
COPY --from=build /app/deps /app/deps
COPY src/ /app/src/
ENV PYTHONPATH=/app/deps
# NO secrets here. DATABASE_PASSWORD arrives at runtime from the orchestrator.
ENTRYPOINT ["python", "/app/src/main.py"]
```

```bash
# Build: the secret is passed to BuildKit, used during build, and never persisted in a layer.
DOCKER_BUILDKIT=1 docker build \
  --secret id=pip_token,env=PIP_TOKEN \
  -t acme/api:1.4.2 .

# Verify nothing leaked into the published image (do this in CI, fail the build on a hit):
dive acme/api:1.4.2                 # inspect layers for stray secret files
docker history --no-trunc acme/api:1.4.2 | grep -i -E 'password|secret|token|key' && exit 1 || true
```

The load-bearing parts: the build-time secret uses `--mount=type=secret` so it exists only for that `RUN` and lands in no layer; the runtime secret is *absent from the image entirely* and injected by the platform. The CI grep is the trip-wire that fails the build if someone reintroduces `ENV SECRET=`.

## Example 2 — runtime injection from a real store: External Secrets Operator + workload identity (Kubernetes / AWS)

This is the production shape: the secret lives in AWS Secrets Manager, the pod authenticates with its *own* identity (IRSA — no stored AWS key), and ESO syncs the value into a k8s Secret that the pod mounts as env. Nothing sensitive is in the manifest, the image, or git.

```yaml
# 1. SecretStore: how the cluster authenticates to AWS. No keys here — uses IRSA (the SA's
#    annotated IAM role assumes via OIDC). There is no long-lived AWS credential to leak.
apiVersion: external-secrets.io/v1
kind: SecretStore
metadata:
  name: aws-secrets
  namespace: payments
spec:
  provider:
    aws:
      service: SecretsManager
      region: eu-west-1
      auth:
        jwt:
          serviceAccountRef:
            name: payments-api          # SA annotated with eks.amazonaws.com/role-arn
---
# 2. ExternalSecret: declares WHICH secret to pull and how to project it. The value never
#    appears in git — only the *reference* (the path in Secrets Manager) does.
apiVersion: external-secrets.io/v1
kind: ExternalSecret
metadata:
  name: payments-db
  namespace: payments
spec:
  refreshInterval: 1h                    # re-pull so rotation propagates without a redeploy
  secretStoreRef:
    name: aws-secrets
    kind: SecretStore
  target:
    name: payments-db-credentials        # the k8s Secret ESO creates/updates
    creationPolicy: Owner
  data:
    - secretKey: DB_PASSWORD
      remoteRef:
        key: prod/payments/db            # path in AWS Secrets Manager
        property: password               # pull just the field we need (least exposure)
---
# 3. Deployment: consumes the synced secret as env at runtime. Image is identical across
#    environments; only the referenced secret differs. No credential in the manifest.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payments-api
  namespace: payments
spec:
  replicas: 3
  selector: { matchLabels: { app: payments-api } }
  template:
    metadata: { labels: { app: payments-api } }
    spec:
      serviceAccountName: payments-api   # the IRSA-bound identity; least-privilege IAM
      containers:
        - name: api
          image: acme/api:1.4.2          # same digest in staging and prod
          env:
            - name: DB_HOST              # CONFIG — non-sensitive, fine inline
              value: payments-db.internal
            - name: DB_PASSWORD          # SECRET — sourced from the synced k8s Secret
              valueFrom:
                secretKeyRef:
                  name: payments-db-credentials
                  key: DB_PASSWORD
```

Why this is the right shape: the AWS secret value lives in exactly one authoritative store; the pod reads it via its *own* identity (a compromised pod can't read another team's secrets); `refreshInterval` means **rotating the secret in AWS propagates automatically** — no code change, no redeploy; and the entire manifest is safe to commit because it contains references, not values. The IAM role behind the SA is scoped to *just* `prod/payments/db`, so the blast radius of a compromised pod is one secret.

## Rotation — a secret you cannot rotate is a permanent liability

Rotation is not a nice-to-have; **the inability to rotate is itself the vulnerability**. If you can't rotate without downtime and a code change, you won't rotate, which means a leaked or stale credential stays live indefinitely.

The non-negotiables:

- **Every secret has a known, automatable rotation path.** Manual "someone remembers to change the prod password annually" is not a rotation strategy.
- **Rotation must not require a redeploy.** This is the payoff of runtime injection + a refresh interval (Example 2): rotate in the store, workloads pick it up. If rotation forces a rebuild, fix the injection first.
- **Support overlapping validity during rotation** (two valid credentials briefly) so rotation is zero-downtime. Cloud secrets managers do this with versioned secrets / `AWSPENDING`→`AWSCURRENT` staging.
- **Dynamic secrets make rotation moot** — a credential with a 15-minute TTL is "rotated" continuously by design. Prefer this where the resource supports it.
- **You must be able to answer "where is this credential used?"** before you can rotate it. If a leaked key forces emergency rotation and nobody knows the seven services that read it, the leak window stays open while you hunt. One authoritative store with read-audit logging answers this.

```bash
# AWS Secrets Manager managed rotation: a Lambda rotates the DB credential on a schedule,
# staging the new value (AWSPENDING) and promoting it (AWSCURRENT) atomically.
aws secretsmanager rotate-secret \
  --secret-id prod/payments/db \
  --rotation-lambda-arn arn:aws:lambda:eu-west-1:1111:function:SecretsManagerRDSRotation \
  --rotation-rules AutomaticallyAfterDays=30
```

## Common mistakes

| Mistake | Why it bites | Fix |
|---------|-------------|-----|
| Secret committed to git (`.env`, `config.prod.json`) | In history + every clone/fork/CI cache forever | Rotate now; move to a store; add a pre-commit scanner |
| `ENV SECRET=` / `COPY secret` in a Dockerfile | Baked into an immutable layer, recoverable from any pull | BuildKit `--mount=type=secret` for build; runtime injection for runtime |
| Secret as a plaintext CI variable, printed in logs | CI logs are widely readable and often retained | Masked secret store / OIDC; never `echo` a secret |
| Treating a k8s `Secret` as encrypted | It's base64; plaintext in etcd without encryption-at-rest | Enable etcd encryption; sync from a real store via ESO/CSI |
| Connection string stored as "config" | The embedded password is a secret | Store the password as a secret; assemble the URL at runtime |
| Long-lived cloud access keys in CI/pods | A standing credential to leak and forget | Workload identity / OIDC short-lived tokens |
| Over-privileged credential (`s3:*`, DB `root`) | A small leak becomes a full breach | Scope IAM/DB grants to exactly what's used |
| Same shared secret across all environments | One leak compromises prod | Per-environment, per-service secrets and identities |
| No rotation path / rotation needs a redeploy | Leaked/stale secrets live indefinitely | Runtime injection + refresh; managed/dynamic rotation |
| Can't answer "where is this secret used?" | Emergency rotation window stays open | One authoritative store with read-audit logging |
| KMS used as a secrets store (or vice versa) | Wrong tool: manual decrypt sprawl / unmanaged keys | KMS encrypts; secrets manager stores; compose them |

## Red flags — STOP

If you catch yourself (or a teammate) saying any of these, stop and fix the discipline before proceeding:

- "I'll just put the key in `.env` and gitignore it." → One missed `.gitignore` and it's in history forever. Use a store.
- "It's only in the Dockerfile temporarily." → Layers are immutable; "temporarily" is "permanently" in the image.
- "Just paste the token into the CI variable for now." → If it's ever logged, it's leaked. Use a masked store or OIDC.
- "The k8s Secret is encrypted, it's base64." → base64 is not encryption. That's plaintext in etcd.
- "We'll rotate it if it ever leaks." → You won't know it leaked, and you can't rotate what you can't locate.
- "Give the service the admin key, it's easier." → That's the difference between a footnote and a breach.
- "Same password in staging and prod is simpler." → Then staging is a backdoor into prod.
- "Just `echo $TOKEN` to debug the pipeline." → That line lives in the build log. Never print a secret.
- "It's only the database *host*, that's not secret." → Confirm it's not the whole connection string with the password in it.

## Rationalizations and their counters

- **"It's an internal-only secret, the network protects it."** Defense in depth: the network boundary fails (misconfig, SSRF, a compromised pod), and then the only thing standing between the attacker and the secret is whether it was stored properly. "Internal" is not a synonym for "safe."
- **"Setting up Vault/ESO is overkill for our size."** Cloud secrets managers are a few API calls and near-zero ops; the floor is "not in git, injected at runtime," which is *less* work than maintaining `.env` files across environments. The expensive option is the leak.
- **"We'll fix the hardcoded keys later."** Every day later is another day the key is in history, another clone, another fork. The leak is already done the moment it's committed; "later" only adds exposure.
- **"Rotating is risky, it might break things."** Not being *able* to rotate is the risk that's already live. Build the rotation path while things are calm, not during the incident that forces it.
- **"Short-lived credentials are more complex to operate."** They're more complex to set up once and far simpler to *survive* — a leaked 15-minute credential is a non-event. Static credentials trade setup ease for incident severity.
- **"Least-privilege slows down development."** Broad grants slow down the postmortem. Scope it now; the cost is a few IAM lines, the alternative is explaining the blast radius.
- **"Nobody's going to scan our repo for keys."** Automated bots scan public (and breached private) repos within *minutes* of a push, specifically for credential patterns. Assume a committed key is found before you finish your coffee.

## The bottom line

A secret lives in one authoritative store, is delivered to workloads at runtime via their own least-privileged identity, never touches a repo / image layer / CI log, and can be rotated — ideally automatically, ideally as a short-lived credential that rotates itself by expiring. Config is read from the environment so one tested artifact promotes unchanged across environments. Hold this and a leaked credential is a minor, attributable, quickly-revoked event. Break it and the question is not *whether* a secret leaks but how much the leaked one could reach, how long it stayed live, and whether you can even tell what it touched. The discipline is cheap; the incident is not.

## Cross-references

- `infrastructure-as-code` (this pack) — state encryption and OIDC federation for CI; never store cloud keys in CI.
- `ci-cd-pipeline-architecture` (this pack) — where OIDC/workload-identity and masked secret injection live in the pipeline.
- `deployment-strategies` (this pack) — promoting one immutable artifact across environments with only config/secrets differing.
- `/ordis-security-architect` — threat-modeling the secret blast radius, IAM least-privilege design, and KMS/envelope-encryption posture.
