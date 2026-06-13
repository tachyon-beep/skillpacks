---
name: infrastructure-as-code
description: Use when provisioning cloud infrastructure by hand or clicking through a console, when nobody can say what a server's config actually is, when prod and staging have silently diverged, when a "quick fix" was made live and never written down, when terraform plan shows changes nobody made, when state is locked/corrupt/shared from a laptop, when copy-pasted Terraform modules drift apart, or when choosing between Terraform, OpenTofu, and Pulumi — covers declarative desired-state, idempotency, remote state and locking, drift detection, plan/apply review gates, and module design.
---

# Infrastructure as Code

## The production stake

A snowflake server is a server nobody can rebuild. When it dies — disk failure, region outage, a fat-fingered `rm` — the recovery time is not "restore from IaC," it is "the one person who set it up reconstructs it from memory and Slack history, under incident pressure, at 3am." Every manual change to production is a deduction from your ability to recover. Click-ops feels fast because the cost is deferred: it is paid in full, with interest, the first time you need to reproduce, audit, or roll back the environment and cannot.

IaC is not a tool you adopt; it is a **discipline you hold**: the running infrastructure is *defined entirely by version-controlled code*, and the only legitimate way to change it is to change the code and apply it. The console is read-only. The moment that invariant breaks, you no longer have infrastructure as code — you have infrastructure with some code near it.

This sheet is about holding that invariant. It is not a tour of Terraform syntax.

## The three non-negotiable invariants

1. **Declarative desired-state.** You describe *what the infrastructure should be*, not the steps to get there. The tool computes the diff between current and desired and converges. You never write "create then maybe update if exists."
2. **Idempotency.** Applying the same code N times produces the same result as applying it once. If running your pipeline twice creates two load balancers, you have a procedural script masquerading as IaC.
3. **Single source of truth = the repo.** The state of production is whatever the code in `main` says, applied. Anything that exists in the cloud but not in code is drift, and drift is a defect.

If any of these is false, the rest of the discipline cannot stand.

## Tool selection (June 2026 reality)

| Option | License | When to reach for it |
|--------|---------|---------------------|
| **OpenTofu** | MPL-2.0 (open source), CNCF-hosted | **Default open choice.** HCL, ~3,900+ providers, ~1.11.x (1.12 in beta). Drop-in for most existing Terraform. |
| **Terraform** | **BSL 1.1 (source-available, non-OSI)** since 1.6 | Existing investment + you accept the BSL terms (no competing-product clause) and HCP Terraform's managed backend. |
| **Pulumi** | Apache-2.0 | You want **real programming languages** (Python / TypeScript / Go) — loops, types, unit tests, abstractions HCL can't express. |

**Stop calling Terraform "open source."** Since v1.6 it ships under the Business Source License 1.1, which is source-available but not OSI-approved and carries a competing-use restriction. OpenTofu is the MPL-2.0 fork (CNCF sandbox since Apr 2025), and its feature set has now **diverged** from HashiCorp's — state encryption, early variable evaluation, and provider-iteration features landed in OpenTofu independently. If you want an open license, name BSL as the reason you chose OpenTofu, and write it down in an ADR so the next engineer doesn't "upgrade" back to Terraform without realizing it's a license change.

The discipline below is **tool-agnostic**. All three obey the same invariants; only the syntax differs.

## State management and locking — where teams actually get hurt

State is the file that maps your code to real cloud resources. Get it wrong and you get the two worst IaC failure modes: **two engineers applying simultaneously and corrupting state**, and **secrets sitting in plaintext on someone's laptop**.

Rules that are not optional:

- **State lives in a remote, shared, versioned backend.** Never `terraform.tfstate` committed to git. Never on one laptop.
- **State locking is mandatory.** The backend must take a lock for the duration of an apply so a concurrent apply blocks instead of racing.
- **State is encrypted at rest** (S3 SSE / DynamoDB / OpenTofu native state encryption) — it contains resource attributes and frequently secrets.
- **Per-environment state isolation.** Prod state and dev state are separate files/keys. Blowing away dev must be physically incapable of touching prod.

### Example 1 — remote state with locking (OpenTofu / Terraform, S3 + native lockfile)

```hcl
# backend.tf — the lock is what prevents two applies corrupting state
terraform {
  required_version = ">= 1.10"

  backend "s3" {
    bucket       = "acme-tfstate-prod"
    key          = "platform/network/terraform.tfstate"
    region       = "eu-west-1"
    encrypt      = true            # state encrypted at rest — non-negotiable
    use_lockfile = true            # S3-native conditional-write locking (no DynamoDB needed)
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"   # or registry.opentofu.org/... for the OpenTofu registry
      version = "~> 5.60"          # pin: never float the provider major
    }
  }
}
```

Why `use_lockfile` and not the old DynamoDB table: S3 conditional writes give you native locking with one fewer resource to manage. The point is *there is a lock*, not which backend implements it. The failure you are preventing: engineer A and engineer B both run `apply`, both read the same state, both write — and one set of changes silently vanishes while the state file describes a world that no longer exists.

## Idempotency and desired-state in practice

The test for whether you have written IaC or a script: **run apply twice in a row.** The second run must report "No changes." If it wants to recreate or modify things, you have non-deterministic config — usually a `name` that includes a timestamp, an unpinned data source, or an imperative `null_resource`/`local-exec` doing work the provider should own.

```hcl
# WRONG — not idempotent: every apply produces a new name, forcing replacement
resource "aws_s3_bucket" "logs" {
  bucket = "logs-${timestamp()}"   # second apply != first apply
}

# RIGHT — stable, declarative; second apply is a no-op
resource "aws_s3_bucket" "logs" {
  bucket = "acme-app-logs-prod"
}
```

`local-exec` provisioners are the most common idempotency leak: a shell command run "at create time" that is invisible to the desired-state model and re-runs unpredictably. Treat every provisioner as a smell — if the provider can express it as a resource, use the resource.

## Drift detection — the discipline that catches click-ops

Drift is the gap between *what the code says* and *what is actually running*. It appears the instant someone makes a manual change in the console. Undetected drift is how snowflakes are born: the code stops being the source of truth, and the next `apply` either reverts a "critical hotfix" someone made by hand or, worse, refuses to run because reality no longer matches the plan's assumptions.

**You detect drift by running `plan` against the real world on a schedule** — not just at deploy time.

### Example 2 — scheduled drift detection in CI (GitHub Actions, OpenTofu)

```yaml
# .github/workflows/drift.yml — runs nightly, alerts if prod ≠ code
name: drift-detection
on:
  schedule:
    - cron: "0 6 * * *"        # every morning before standup
  workflow_dispatch: {}

permissions:
  id-token: write              # OIDC → no long-lived cloud keys in CI
  contents: read

jobs:
  detect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: opentofu/setup-opentofu@v1
        with: { tofu_version: "1.11.0" }

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::111122223333:role/ci-tofu-readonly
          aws-region: eu-west-1

      - run: tofu init -input=false

      # -detailed-exitcode: 0 = no drift, 2 = drift present, 1 = error
      - id: plan
        run: tofu plan -detailed-exitcode -lock=false -input=false
        continue-on-error: true

      - name: Fail loudly on drift
        if: steps.plan.outputs.exitcode == '2'
        run: |
          echo "::error::Infrastructure has drifted from code. Someone changed prod by hand."
          exit 1
```

`-detailed-exitcode` is the load-bearing flag: exit `2` means "the world no longer matches the code." Wire that to an alert. The day this fires is the day you catch click-ops *before* it becomes a snowflake. Note `-lock=false` is acceptable here only because this is a read-only plan; never disable locking on an apply.

## Plan/apply review — the gate

The `plan` is a code review artifact, not a formality. Three rules:

1. **`apply` only ever runs the exact plan that was reviewed.** Use `tofu plan -out=tfplan` then `tofu apply tfplan`. Never `apply` with no saved plan in CI — that re-plans against possibly-changed state and applies something nobody saw.
2. **A human reviews the plan before prod apply.** The plan output (resources to add/change/**destroy**) goes in the PR. The reviewer is specifically hunting for unexpected `destroy` / `replace` lines — those are how a one-character change deletes a database.
3. **Apply is gated by environment promotion.** Dev → staging → prod, each behind its own state and its own approval. CI uses OIDC short-lived credentials, never stored cloud keys.

```bash
# The only correct apply flow — reviewed plan, then apply that exact plan
tofu plan -out=tfplan -input=false        # produce the artifact
# → attach plan output to PR, human reviews, especially "destroy" / "must be replaced"
tofu apply -input=false tfplan            # apply ONLY what was reviewed
```

## Module design — DRY without coupling disasters

Copy-pasting a working Terraform block into five environments is how five environments silently diverge. Modules are how you write the pattern once. But over-modularizing is its own failure: a module with 40 inputs that abstracts nothing is worse than no module.

- **Module = a meaningful unit of infrastructure** (a service's full footprint, a VPC), not a thin wrapper around one resource.
- **Pin module versions** (`source = "...//modules/vpc?ref=v2.3.1"` or a registry version constraint). An unpinned module ref means a teammate's edit changes your prod on the next apply.
- **Inputs are the contract.** Validate them (`variable` blocks with `validation`), give safe defaults, and document each. Outputs expose only what consumers need.
- **No environment logic inside modules.** The module doesn't know it's "prod." The *caller* passes prod values. This is what keeps one module reusable across environments instead of growing `if env == "prod"` branches.

```hcl
# Calling a pinned, contract-validated module per environment — same code, different inputs
module "network" {
  source  = "git::https://github.com/acme/tf-modules.git//vpc?ref=v2.3.1"  # pinned
  cidr    = var.vpc_cidr          # prod passes 10.0.0.0/16, dev passes 10.10.0.0/16
  azs     = var.availability_zones
  env     = var.environment       # tag/label only — module has no per-env branches
}
```

## Common mistakes

| Mistake | Why it bites | Fix |
|---------|-------------|-----|
| State file committed to git or on a laptop | No locking → corruption; secrets leak in history | Remote encrypted backend with locking |
| No state locking | Concurrent applies silently lose changes | `use_lockfile` / locking backend, always |
| Manual console "hotfix" never codified | Drift; next apply reverts it or breaks | Codify every change; nightly drift detection |
| `apply` re-plans instead of applying saved plan | Applies something nobody reviewed | `plan -out` → `apply tfplan` |
| Unpinned provider/module versions | A remote edit changes your prod unprompted | Pin every provider and module ref |
| `local-exec` doing the provider's job | Non-idempotent, invisible to state | Use the real resource, delete the provisioner |
| Names with `timestamp()`/random per apply | Forces replacement every run | Stable deterministic names |
| One state for all environments | Dev mistake can destroy prod | Per-environment state isolation |
| Long-lived cloud keys in CI | Standing credential to leak | OIDC short-lived federated credentials |
| Calling Terraform "open source" | It's BSL 1.1 since 1.6 — license/legal exposure | Say "source-available (BSL)"; use OpenTofu for open |
| 40-input wrapper "modules" | Abstracts nothing, harder than raw resources | Module = meaningful unit, or no module |

## Red flags — STOP

If you catch yourself (or a teammate) saying any of these, stop and fix the discipline before proceeding:

- "I'll just change it in the console real quick." → That is the first snowflake. Change the code.
- "I'll add it to Terraform later." → Later never comes; the drift compounds. Codify it now.
- "Just run apply, skip the plan review." → Unreviewed `destroy` lines delete databases.
- "The state file is fine in the repo for now." → Plaintext secrets + no locking. Move it today.
- "We don't need locking, it's just me." → "Just me" becomes "me and the CI job" silently.
- "Everyone clicks in the console here, that's how it's done." → That's the diagnosis, not a defense.
- "Pin versions later, I want the latest." → Unpinned = your prod changes when someone else edits a module.
- "It applied once, ship it." → Run it twice. If the second run isn't a no-op, it isn't IaC.

## Rationalizations and their counters

- **"Console is faster for a one-off."** A one-off in prod is never one-off — it must be reproducible, auditable, and survivable. The code path is the fast path measured at recovery time.
- **"IaC is overkill for our size."** The smaller the team, the fewer people hold the knowledge, and the more catastrophic a snowflake's loss. Small teams need reproducibility *most*.
- **"We'll codify the existing infra later."** Use `import` blocks / `tofu import` and bring it under code now. Every day of delay adds drift you'll have to reconcile.
- **"Drift detection is noise."** Drift detection firing *is the signal you wanted* — it caught a manual change before it became permanent and invisible.
- **"Reviewing plans slows us down."** It slows down exactly the applies that would have destroyed something. That is the feature.
- **"Modules are premature abstraction."** Copy-paste across environments is the abstraction debt; you've just deferred it to the day the copies diverge in production.

## The bottom line

The infrastructure is the code, the code is in `main`, and `main` applied *is* production. The console is read-only. The plan is reviewed. The state is remote, locked, and encrypted. Drift detection runs nightly and is allowed to fail the build. Hold those, and a dead server is a `tofu apply` away from recovery instead of a postmortem. Break any of them, and you are growing snowflakes — you just won't find out until the night you need them not to be.

## Cross-references

- `ci-cd-pipeline-architecture` (this pack) — where the `plan`/`apply` gates live in the deployment pipeline.
- `/axiom-solution-architect` — recording the OpenTofu-vs-Terraform license decision as an ADR.
- `ordis-security-architect` — OIDC federation, secret management, state encryption posture.
