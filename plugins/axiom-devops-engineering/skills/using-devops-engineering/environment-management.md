---
name: environment-management
description: Use when a release passed every check in staging and broke in prod, when "it works on staging" precedes an incident, when dev/staging/prod have silently diverged, when nobody can say how staging differs from prod, when a config value is set in one environment and forgotten in another, when developers test against prod data copied to a laptop, when there is no preview/ephemeral environment per pull request, when promoting a change means re-running the same manual steps in three places, when a fix that worked in staging needs "a few tweaks" to work in prod, or when you cannot reproduce a prod-only bug anywhere lower. Covers environment parity (dev/staging/prod), ephemeral/preview environments, the promotion flow, config drift and a single config contract, and data handling across environments.
---

# Environment Management

## The production stake

"It works in staging" is not evidence. It is a claim about an environment that, in most shops, differs from production in ways nobody has written down — a different database version, a different feature-flag default, a memory limit set by hand eighteen months ago, a secret that points at a stubbed payment gateway, a dataset one-hundredth the size with none of the rows that trigger the bug. Every one of those differences is a place where a release can pass staging and fail prod. The gap between environments is exactly the gap between "verified" and "verified somewhere that isn't where it runs."

The discipline this sheet enforces: **the lower environments must differ from production only in ways you have explicitly chosen and can enumerate.** Everything else — same image, same orchestrator version, same config *shape*, same migration path, representative data — is held identical by construction. When you can list every intentional difference on one page, staging becomes evidence. When you cannot, every green staging run is a coin flip you didn't know you were tossing.

This is engineering discipline, not a tour of staging tools. The goal is a single property: **a change that is proven in a lower environment is proven for production, because the only differences are ones that cannot affect the outcome being proven.**

## The parity invariant

Parity does not mean "identical." A staging cluster with prod's full traffic and prod's real customer data would be insane. Parity means **the differences are deliberate, documented, and provably irrelevant to what you are verifying.** Three layers, in priority order:

1. **Artifact parity — non-negotiable, zero tolerance.** The *exact same immutable image* (by `sha256` digest, never a mutable tag) flows dev → staging → prod. You build once and promote the bytes. If staging and prod can resolve `:latest` or `:v2` to different bytes, every test you ran proved nothing about what's in prod. This is the cheapest parity to hold and the most expensive to violate.
2. **Platform parity — held identical by IaC.** Same orchestrator version, same runtime, same proxy/Gateway implementation, same resource-limit *policy*. The same Terraform/OpenTofu modules build every environment; only the *inputs* differ (size, replica count, CIDR). A module with `if env == "prod"` branches is a parity leak — see `infrastructure-as-code`.
3. **Config and data parity — shape identical, values intentional.** Every environment reads the *same set of keys* (the config contract). Values differ on purpose (prod points at the real DB, staging at a staging DB), and the *list* of values-that-differ is the documented diff. Data is *representative* (shape, scale, edge cases) without being a copy of production PII.

The deliverable that makes parity real is a **parity ledger**: a single document that lists every intentional difference between staging and prod and why it's safe. If a difference isn't on the ledger, it's drift — a defect, not a config.

## The environment ladder

| Environment | Lifetime | Data | Who/what changes it | Purpose |
|-------------|----------|------|---------------------|---------|
| **Local/dev** | Per-developer | Synthetic / seeded fixtures | Developer | Fast inner loop; nothing shared |
| **Preview/ephemeral** | Per-PR, auto-destroyed on merge/close | Synthetic or anonymized subset | The PR itself (GitOps) | Review a change in a live env before merge |
| **Staging** | Persistent | Representative, anonymized, prod-scale-ish | Promotion only (no manual edits) | The dress rehearsal — must mirror prod |
| **Production** | Persistent | Real | Promotion + break-glass only | Where users are |

Two rules govern the ladder. **No environment is edited by hand** — every change arrives by promoting code (this is what keeps the ladder from drifting; see GitOps below). And **staging is the one that must mirror prod**, because it is the gate. Dev can be loose; staging cannot, or the gate is decorative.

## Ephemeral / preview environments — kill the "shared staging" bottleneck

A single shared staging environment is a queue: changes pile up, interfere, and you cannot tell whose change broke it. Worse, a developer tests on their laptop ("works on my machine") because staging is contended — and laptop parity is the worst parity there is.

The fix is **ephemeral environments**: a full, isolated, production-shaped environment spun up *per pull request* from the same IaC and the same images, then destroyed automatically. The reviewer clicks a URL and exercises the actual change in a live environment that matches prod's construction, with no contention and no laptop. This is the single highest-leverage move against "works in staging" surprises, because every change gets its own faithful environment instead of fighting for one stale shared one.

### Example 1 — per-PR ephemeral environment with Argo CD ApplicationSet (Pull Request generator)

Argo CD's `ApplicationSet` PR generator creates one Application per open PR and deletes it when the PR closes — declarative, pulled, auto-reconciled (OpenGitOps). The same chart and the same image digest that ship to prod render the preview, so the preview *is* prod-shaped by construction.

```yaml
# applicationset-previews.yaml — one live environment per open PR, auto-destroyed on close
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: orders-api-previews
  namespace: argocd
spec:
  goTemplate: true
  generators:
    - pullRequest:
        github:
          owner: acme
          repo: orders-api
          tokenRef: { secretName: gh-token, key: token }
        requeueAfterSeconds: 60          # poll for opened/closed PRs
  template:
    metadata:
      name: 'preview-pr-{{.number}}'
    spec:
      project: previews
      source:
        repoURL: https://github.com/acme/orders-api.git
        targetRevision: '{{.head_sha}}'  # the PR's exact commit
        path: deploy/chart
        helm:
          # SAME chart as prod; only the values that MUST differ are overridden.
          # The image is pinned by digest built once in CI for this SHA — artifact parity.
          parameters:
            - { name: image.digest,   value: 'sha256:{{.head_sha_image_digest}}' }
            - { name: ingress.host,   value: 'pr-{{.number}}.preview.acme.dev' }
            - { name: replicaCount,   value: '1' }      # intentional diff: cost, on the ledger
            - { name: database.mode,  value: 'ephemeral' }  # seeded synthetic DB, never prod data
      destination:
        server: https://kubernetes.default.svc
        namespace: 'preview-pr-{{.number}}'
      syncPolicy:
        automated: { prune: true, selfHeal: true }   # closes PR -> Application pruned -> env destroyed
        syncOptions: [CreateNamespace=true]
```

The load-bearing details: `targetRevision: {{.head_sha}}` pins the preview to the PR's exact commit; the **same Helm chart** as prod is used (only ledgered values overridden); `prune: true` plus the PR generator means closing the PR *destroys the environment* with no human cleanup. The replica count and ephemeral DB are intentional, ledgered differences — cost and data-safety — that cannot affect the correctness the reviewer is checking.

## Config drift — the single config contract

Config drift is how environments diverge invisibly. Someone sets `MAX_UPLOAD_MB=50` in prod during an incident, never adds it to staging, and six months later a 40MB upload passes staging and fails prod. The root cause is that each environment's config was authored *independently* instead of being instances of one **contract**.

The discipline: **define the full set of config keys once (the contract), then provide a value for every key in every environment — no key may be silently absent.** The application must *fail to start* if a contract key is missing, so a forgotten value is a loud boot failure in CI, not a 2am prod surprise. Secrets are referenced, never inlined (see `ordis-security-architect`).

### Example 2 — typed config contract with fail-fast validation and a per-env value matrix

A schema makes the contract executable: the app loads config through it and crashes on boot if any key is missing or malformed. The *same code* runs in every environment; only the injected values differ — and the schema guarantees the *shape* is identical everywhere.

```python
# config.py — ONE contract. Same keys in every environment; missing/invalid = boot failure.
from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORDERS_", frozen=True)

    environment: str = Field(pattern="^(dev|preview|staging|prod)$")
    database_url: PostgresDsn                 # required everywhere; value differs per env
    max_upload_mb: int = Field(ge=1, le=500)  # the value that bit prod — now contract-enforced
    payment_base_url: str                     # staging -> sandbox, prod -> live; ON THE LEDGER
    feature_new_ranking: bool = False         # default identical across envs; flip via promotion
    otel_exporter_endpoint: str               # vendor-neutral OTLP; same key everywhere

    @field_validator("payment_base_url")
    @classmethod
    def _no_prod_gateway_below_prod(cls, v, info):
        # Guardrail: a lower env must never point at the live payment gateway.
        if info.data.get("environment") != "prod" and "live.payments" in v:
            raise ValueError("non-prod env pointed at LIVE payments gateway — refusing to start")
        return v


settings = Settings()  # raises on startup if ANY contract key is missing or invalid
```

The per-environment values live in version control as a **matrix**, so the diff between environments is a reviewable artifact — this *is* the config half of the parity ledger:

```yaml
# config-matrix.yaml — every key x every env, in git. A blank cell is a build-breaking error.
# The right-hand columns ARE the documented diff between environments.
keys:
  ORDERS_MAX_UPLOAD_MB:       { dev: "10",  preview: "10",  staging: "50",  prod: "50" }   # parity: staging==prod
  ORDERS_FEATURE_NEW_RANKING: { dev: "true", preview: "true", staging: "false", prod: "false" }
  ORDERS_PAYMENT_BASE_URL:    { dev: "http://stub", preview: "http://stub",
                                staging: "https://sandbox.payments", prod: "https://live.payments" }
  ORDERS_DATABASE_URL:        { dev: "secretref://dev/db", preview: "secretref://preview/db",
                                staging: "secretref://staging/db", prod: "secretref://prod/db" }
```

A CI check parses the matrix against the schema and **fails the build if any environment is missing any contract key.** That single check is what converts config drift from a silent prod-only landmine into a red build on the PR that introduced it. Note the deliberate guardrail: `staging` and `prod` share `MAX_UPLOAD_MB=50` (parity on a value that has burned teams), while `payment_base_url` legitimately differs (and the validator refuses to let a lower env aim at the live gateway).

## The promotion flow

Promotion is moving *the same artifact* up the ladder, gaining confidence at each rung. It is not rebuilding per environment, and it is not re-running manual steps in three consoles.

```
  build once  ──>  dev  ──>  preview (per-PR)  ──>  staging  ──>  prod
  (one digest)     │           │                     │            │
                   └─ same image digest promoted unchanged at every rung ─┘
            config differs only by the ledgered matrix; migrations precede the code that needs them
```

Rules that make promotion trustworthy:

1. **Promote the digest, not the source.** `prod` runs the bytes that passed `staging`, identified by `sha256` digest. Re-building "from the same commit" for prod can still produce different bytes (base-image moved, dependency floated) and silently breaks artifact parity. Build once; sign the image and its SBOM with Sigstore/cosign and **verify the signature at the prod gate** — promotion should refuse an unsigned or unknown digest.
2. **Each rung has a gate, and the gate is automated.** Dev → preview: it builds and unit tests pass. Preview → staging: integration/e2e green on the ephemeral env. Staging → prod: smoke + SLO analysis hold (see `deployment-strategies`). A manual "looks fine" is not a gate.
3. **Migrations are promoted ahead of the code that needs them, expand/contract style** (see `deployment-strategies`), so a code rollback never lands on a schema it can't read. Run the migration in staging first against representative data — a migration that's instant on 10k rows can lock a table for minutes on prod's 50M.
4. **Nothing skips a rung, and nothing is edited at a rung.** A hotfix still flows through the ladder (a fast lane, not a bypass). The moment someone edits prod directly, the ledger is wrong and the next promotion may revert their fix — exactly the click-ops failure `infrastructure-as-code` exists to kill.

## Data handling across environments

Data is where parity and *safety* collide. Real production data makes staging faithful; it also makes staging a PII liability and a compliance breach waiting to happen. The resolution is **representative, not real**:

- **Never copy raw production data downward.** A prod dump on a laptop or in staging is a data-protection incident in waiting (GDPR/CCPA, and a juicy target with weaker controls than prod). This is also a hard house constraint: lower environments must not carry real customer or sensitive data.
- **Anonymize/pseudonymize at the boundary.** If you must derive staging data from prod, run it through a masking/synthetic pipeline (deterministic faking of names/emails, hashed identifiers, scrubbed free-text) *before* it leaves the prod boundary — never after it lands somewhere weaker.
- **Match shape and scale, not contents.** The bugs that hide from staging are scale bugs (a query fine at 10k rows, pathological at 50M) and edge-case bugs (the one customer with a null in a "never null" column). Seed synthetic data that *reproduces the distribution and the edge cases*, not a thumbnail of prod.
- **Ephemeral envs get seeded synthetic DBs**, created and destroyed with the environment — never a shared mutable store and never prod-derived.

If a bug only reproduces in prod, the usual root cause is a data-parity gap: staging's data lacks the shape, scale, or edge case that triggers it. Fix the data generator, not just the bug.

## Common mistakes

| Mistake | Why it bites | Fix |
|---------|-------------|-----|
| Rebuilding the artifact per environment | Prod runs different bytes than staging tested | Build once, promote the digest, verify signature |
| Mutable tags (`:latest`, `:v2`) up the ladder | Environments resolve the tag to different bytes | Pin by `sha256` digest end to end |
| Config authored per-env independently | A key set in prod and forgotten in staging | One contract; value matrix in git; CI fails on a blank cell |
| Config missing-key handled with a default | Silent wrong behavior instead of loud failure | Fail-fast: app refuses to boot on a missing contract key |
| Manual edits to staging/prod | Drift; next promotion reverts the edit | No-hand-edits; everything via promotion/GitOps |
| Single shared staging | Contention, interference, "works on my laptop" | Per-PR ephemeral environments |
| Ephemeral envs that aren't prod-shaped | Preview passes, prod fails — same surprise, earlier | Same chart/IaC + same digest; only ledgered diffs |
| Copying prod data to lower envs | PII/compliance breach; weak-control target | Anonymized/synthetic, masked at the prod boundary |
| Staging data tiny vs prod | Scale/edge-case bugs invisible until prod | Representative shape + scale + edge cases |
| Migration coupled to its code on promote | Code rollback lands on unreadable schema | Expand/contract; migrate ahead, on staging-scale data first |
| No documented env diff | Nobody can say how staging differs from prod | Maintain a parity ledger; un-ledgered diff = defect |
| `if env == "prod"` inside IaC modules | Per-env branches breed divergence | Same module, different inputs (see `infrastructure-as-code`) |

## Red flags — STOP

- "It works in staging, ship it." → Staging is only evidence if the diff is ledgered. What does staging *not* have that prod does?
- "I'll just set that value in prod directly." → That key is now missing from the contract elsewhere. Add it to the matrix, promote it.
- "Just pull a prod dump into staging so it's realistic." → That's a data-protection incident. Anonymize at the boundary or use synthetic.
- "Staging is busy, I'll test on my laptop." → Laptop parity is the worst parity. Spin a preview env.
- "We'll rebuild it for prod from the same commit." → Same commit ≠ same bytes. Promote the digest you tested.
- "The migration was instant on staging." → On how many rows? Run it against prod-scale data first.
- "There's only one difference between staging and prod." → Then it's on the ledger and you can name it. If you can't, there are more than one.
- "Preview environments are too expensive." → Cheaper than the prod incident a stale shared staging let through. Scale them to one replica and tear them down on merge.

## Rationalizations and their counters

- **"Full parity is impossible, so why bother."** Parity is *enumerated difference*, not identity. You can't make staging identical to prod, but you can make every difference deliberate and listed — and that is the whole game.
- **"Per-PR environments are overkill."** They are the single most effective cure for "works in staging," because every change gets a faithful environment instead of fighting for a stale shared one. The IaC that builds prod already builds them.
- **"We need real data to test properly."** You need *representative* data — shape, scale, edge cases. Real data adds risk and compliance exposure without adding test signal you can't synthesize.
- **"The config drift was a one-off."** It is never a one-off; it is the *mechanism*. One forgotten key today is the prod-only failure next quarter. The contract + matrix check removes the mechanism, not the instance.
- **"Promoting the same image is what we basically do."** "Basically" is where artifact parity dies. Either you verify the *digest* at the prod gate or you are trusting a tag — and tags lie.

## The bottom line

Make every difference between environments deliberate, documented, and provably irrelevant to what you're verifying — and "it works in staging" stops being a hope and becomes a guarantee. One image digest promoted unchanged up the ladder. One config contract with a value for every key in every environment, enforced by a build-breaking check and a fail-fast boot. A faithful, throwaway environment per pull request instead of a contended shared staging. Representative synthetic data, never a prod copy below prod. Migrations promoted ahead of the code that needs them, rehearsed at prod scale. A parity ledger that names every intentional difference. Hold those, and the gap between "verified" and "verified where it runs" closes to exactly the differences you chose. Let any of them slide, and staging goes back to being a place that's green right up until prod isn't.

## Cross-references

- `infrastructure-as-code` (this pack) — the same OpenTofu/Terraform modules build every environment from different inputs; this is platform parity and drift detection.
- `deployment-strategies` (this pack) — build-once digests, expand/contract migrations, and the staging→prod promotion gate where these rungs plug in.
- `ci-cd-pipeline-architecture` (this pack) — where the per-rung promotion gates and the config-matrix check live in the pipeline.
- `ordis-security-architect` — secret references vs inlined values, anonymization at the data boundary, OIDC for the promotion pipeline.
