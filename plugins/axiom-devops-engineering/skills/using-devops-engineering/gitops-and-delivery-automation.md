---
name: gitops-and-delivery-automation
description: Use when CI runs kubectl apply or helm upgrade straight at the cluster, when nobody can say what version is actually running in prod without SSHing in, when someone hotfixed the cluster by hand and it never made it back to git, when the cluster and the repo have silently diverged, when deploys need long-lived cluster credentials sitting in the CI runner, when rolling back means "re-run the old pipeline and hope", when there is no audit trail of who deployed what when, when manifests live in the same repo and pipeline as application source so a code merge mutates the cluster, or when choosing between Argo CD and Flux or deciding how to separate build from deploy — covers Git as source of truth for desired state, pull-based reconciliation controllers, drift correction, and the CI/CD split.
---

# GitOps and Delivery Automation

## The production stake

When a CI job runs `kubectl apply` against your cluster, the cluster's state is whatever the *last pipeline that happened to run* left behind. Ask "what is deployed in prod right now?" and the honest answer is "I don't know — let me go look." Ask "who changed it and when?" and there is no answer at all, because an imperative push leaves no durable record of intent: the pipeline log scrolls away, the credentials it used are still sitting in the runner, and a manual `kubectl edit` an hour later overwrote half of it with nothing to show the difference.

This is the failure GitOps closes. The cluster's desired state is a **declarative artifact in git**, and an in-cluster controller **continuously pulls that artifact and reconciles reality to match it**. The git history *is* the deploy audit trail — every change is a reviewed, signed, attributable commit. A manual change to the cluster is not a deploy; it is drift, and the controller reverts it. Rollback is `git revert`, not "re-run the old pipeline and pray the artifacts still exist."

GitOps is not "we keep our YAML in git." Plenty of teams keep YAML in git and then `kubectl apply` it from CI — that is push-based imperative delivery with version control bolted on, and it has none of the guarantees. GitOps is the **invariant that the running cluster equals what a controller reconciled from git, with no other write path**. The moment a human or a CI job can write directly to the cluster, the invariant is broken and you are back to "I don't know what's running."

This sheet is about holding that invariant. It is not a tour of Argo CD's UI.

## The four OpenGitOps principles (the contract)

Anchor everything to the CNCF **OpenGitOps** principles. A system is GitOps only if all four hold:

1. **Declarative.** The entire desired state is expressed declaratively — manifests, not scripts. No `kubectl create` steps, no imperative ordering you maintain by hand.
2. **Versioned and immutable.** Desired state is stored in git: versioned, immutable history, the canonical source of truth. Every state the cluster has ever been in is a commit you can return to.
3. **Pulled automatically.** Software agents *pull* the desired state from git. Nothing pushes into the cluster from outside. The cluster reaches out; the outside world never reaches in.
4. **Continuously reconciled.** Agents continuously observe actual state and reconcile it toward desired. Drift is detected and corrected without a human triggering it.

If any one is false, you do not have GitOps — you have a partial imitation that will surprise you on the day it matters. Most real-world breakage is principle 3 (someone added a push path) or principle 4 (reconciliation set to manual-only "to be safe," which just re-introduces drift).

## The split that makes it work: CI builds, CD deploys

The single most important architectural move is **separating CI from CD**. They are different jobs with different blast radii and different credentials:

| | CI (build) | CD (deploy) |
|---|------------|-------------|
| **Job** | Compile, test, build image, scan, sign, push artifact | Reconcile cluster to the desired state in git |
| **Writes to** | Artifact registry only | The cluster only |
| **Cluster credentials** | **None** | Held *inside* the cluster, never exported |
| **Trigger** | Code push / PR | A commit to the config repo |
| **Lives where** | Your CI system (GitHub Actions, etc.) | An in-cluster controller (Argo CD / Flux) |

CI's job ends at "a signed, scanned image is in the registry, and a commit updating the image digest is in the config repo." It never touches the cluster. CD's job is everything after: the in-cluster controller notices the new commit and reconciles.

Why this matters concretely: the catastrophic anti-pattern is CI holding a long-lived `kubeconfig` with cluster-admin and running `kubectl apply`. That credential is a standing skeleton key — leak the runner, leak the cluster. In pull-based GitOps, **no credential ever leaves the cluster.** The controller runs *in* the cluster and reads git (and the registry) with a read-only token. There is nothing for an attacker to steal from CI that grants cluster write. This is the security argument for pull over push, and it is decisive.

**Use two repositories (or at least two clearly separated paths):** the **app repo** (source code, CI) and the **config/deploy repo** (manifests, what the controller watches). Keeping manifests in the app repo means every application code merge risks mutating the cluster, you cannot give different review rules to "change the code" vs "change what's deployed," and the deploy audit trail is tangled with feature commits. Separation lets a code change and a deploy change be two reviewable events.

## Push vs pull — and why pull wins

| | Push (imperative CI deploy) | Pull (GitOps controller) |
|---|------------------------------|---------------------------|
| Who writes to cluster | External CI runner | In-cluster controller |
| Cluster creds | Long-lived, in CI | Never leave the cluster |
| Drift handling | None — last push wins | Detected and auto-corrected |
| Audit trail | Pipeline logs (ephemeral) | Git history (permanent, signed) |
| "What's running?" | Unknown until you inspect | = the commit the controller synced |
| Rollback | Re-run old pipeline | `git revert` → controller reconciles |
| Multi-cluster | N pipelines, N credentials | N controllers, 0 exported creds |

Push-based delivery can *look* like GitOps (manifests are in git!) while violating principles 3 and 4. The tell: **the cluster only changes when a pipeline runs.** In real GitOps the cluster converges toward git continuously, whether or not anything ran — that is what makes drift self-healing.

## Tool selection (June 2026 reality)

| Tool | Shape | Reach for it when |
|------|-------|-------------------|
| **Argo CD** | App-centric controller with UI, RBAC, multi-cluster, ApplicationSets | The default. ~60% of GitOps clusters; the overwhelming majority of its users run it in production. You want a visible app dashboard, sync/health status, and a strong RBAC story. Pair with **Argo Rollouts** for progressive delivery. |
| **Flux** | Fully CRD-native, composable controllers (source / kustomize / helm / image-automation) | You want GitOps as pure Kubernetes API objects with no extra UI surface, tight Helm/Kustomize integration, and built-in image-update automation. Lighter footprint. Pair with **Flagger** for progressive delivery. |

Pick by **ecosystem fit, not hype.** Both fully satisfy the OpenGitOps principles. Argo CD's advantage is operability and the dashboard; Flux's advantage is its everything-is-a-CRD composability and native image automation. Whichever you choose, write the choice and its reasons down — the failure mode is two teams adopting different controllers and nobody being able to reason across clusters.

Progressive delivery (canary/blue-green traffic-shifting, metric-gated automated rollback) is the *deployment-strategy* layer that rides on top of GitOps — see the `deployment-strategies` sheet for Argo Rollouts / Flagger mechanics and Gateway API traffic-shifting. This sheet owns the control plane (reconciliation, drift, the CI/CD split); that sheet owns how a single release rolls out safely.

## Example 1 — Argo CD Application: declarative, self-healing, auto-pruning

The whole GitOps contract for one app, expressed as a Kubernetes object the controller reconciles. The load-bearing lines are in `syncPolicy.automated`.

```yaml
# argocd/applications/payments.yaml — committed to git; Argo CD watches THIS too (App-of-Apps)
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: payments
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io   # cascade-delete managed resources on app removal
spec:
  project: production
  source:
    repoURL: https://github.com/acme/deploy-config.git   # the CONFIG repo, not the app repo
    targetRevision: main                                  # track main; commits here are deploys
    path: apps/payments/overlays/prod                     # Kustomize overlay for this env
  destination:
    server: https://kubernetes.default.svc                # in-cluster; no exported kubeconfig
    namespace: payments
  syncPolicy:
    automated:
      prune: true        # resource removed from git → removed from cluster (no orphans)
      selfHeal: true     # manual kubectl change → reverted to git within the sync interval
      allowEmpty: false  # refuse to sync an empty desired state (guards against a bad render)
    syncOptions:
      - CreateNamespace=true
      - ServerSideApply=true   # field-managed apply; avoids last-write-wins clobbering
    retry:
      limit: 5
      backoff: { duration: 10s, factor: 2, maxDuration: 3m }
  # Surface drift instead of hiding it: anything in cluster but not git shows as OutOfSync.
  ignoreDifferences: []   # keep empty unless a controller legitimately mutates a field
```

`selfHeal: true` is the principle-4 enforcer: someone runs `kubectl edit deployment/payments` to "quickly bump a replica count," and Argo CD reverts it on the next reconcile because git did not say that. `prune: true` is the principle-2 enforcer: delete a manifest from git and the resource leaves the cluster — no zombie Services accumulating from deploys past. Leaving these off (manual sync, no prune) is the most common way teams "do GitOps" while quietly re-creating drift; the controller becomes a fancy `kubectl apply` button a human still has to press, and reality wanders away from git between presses.

**The deploy is a git operation.** To ship payments v2.4.0 you commit a one-line image-digest bump to `deploy-config`; the controller does the rest. To roll back you `git revert` that commit. The audit trail — who, what, when, reviewed by whom — is the git log, permanently, signed.

## Example 2 — CI ends at the registry and a config-repo commit; it never touches the cluster

This is the CI side of the split. Note what is absent: **no `kubectl`, no `helm upgrade`, no kubeconfig, no cluster credential of any kind.** CI builds, signs, pushes, and writes one commit to the config repo. The cluster controller takes it from there.

```yaml
# .github/workflows/release.yml  (in the APP repo)
name: build-and-promote
on:
  push:
    tags: ["v*"]

permissions:
  id-token: write   # OIDC to the registry + cosign keyless — NO long-lived secrets
  contents: read

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      digest: ${{ steps.push.outputs.digest }}
    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-buildx-action@v3   # BuildKit: cache mounts, multi-arch, concurrency

      - name: Build and push (by digest)
        id: push
        uses: docker/build-push-action@v6
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ghcr.io/acme/payments:${{ github.ref_name }}
          provenance: true   # SLSA provenance attestation per image
          sbom: true         # attach SBOM (CycloneDX) at build time

      # Sign image + SBOM, keyless. Verified later by the cluster's admission policy.
      - name: Sign with cosign (keyless via Fulcio + Rekor)
        run: |
          cosign sign --yes ghcr.io/acme/payments@${{ steps.push.outputs.digest }}
        env:
          COSIGN_EXPERIMENTAL: "1"

  promote:
    needs: build
    runs-on: ubuntu-latest
    steps:
      # Check out the SEPARATE config repo and write the new digest there.
      - uses: actions/checkout@v4
        with:
          repository: acme/deploy-config
          token: ${{ secrets.CONFIG_REPO_WRITE_TOKEN }}   # write to git only — NOT to the cluster
          path: deploy-config

      # Pin to the immutable digest, never a mutable tag — the digest is the source of truth.
      - name: Bump prod image digest
        working-directory: deploy-config
        run: |
          cd apps/payments/overlays/prod
          kustomize edit set image \
            ghcr.io/acme/payments=ghcr.io/acme/payments@${{ needs.build.outputs.digest }}

      # The deploy is a commit. Optionally open a PR for prod so the promotion is reviewed.
      - name: Open promotion PR
        working-directory: deploy-config
        run: |
          git config user.name  "ci-bot"
          git config user.email "ci@acme.dev"
          git checkout -b promote/payments-${{ github.ref_name }}
          git commit -am "deploy(payments): promote ${{ github.ref_name }} to prod"
          git push origin HEAD
          gh pr create --fill --base main
        env:
          GH_TOKEN: ${{ secrets.CONFIG_REPO_WRITE_TOKEN }}
```

What this buys you: the worst thing a compromised CI runner can do is push a bad image and open a PR — both reviewable, neither granting cluster write. There is no kubeconfig to exfiltrate. The promotion to prod is a *reviewed git change* (the PR), so the deploy gate is a code review, and merging it is what triggers the cluster controller. **Pin the digest, never a floating tag:** `:latest` or even `:v2.4.0` can be re-pushed to point at different bytes, which silently breaks principle 2 (immutable desired state) and defeats your signature verification. The digest is the only thing that is truly the source of truth.

## Drift correction — the discipline, not the dashboard

Drift is the gap between what git says and what the cluster is actually running. In push-based delivery, drift is invisible until the next deploy stomps it (or fails because reality moved). In GitOps the controller **observes** it continuously and you decide the policy:

- **`selfHeal: true` (auto-correct):** manual changes are reverted to git automatically. This is the strong form — the cluster cannot be hand-edited into a state git doesn't know about for longer than one reconcile interval. Use it for everything you can.
- **Detect-and-alert (no auto-heal):** the controller marks the app `OutOfSync` and you alert on it, but a human decides. Use this only where a controller you don't own legitimately mutates fields (then encode those in `ignoreDifferences` rather than turning off self-heal wholesale).

The anti-pattern is treating an `OutOfSync` status as normal background noise. `OutOfSync` means *someone changed prod outside git* — that is exactly the click-ops signal GitOps exists to catch. Wire it to an alert and treat a sustained `OutOfSync` as an incident, the same way IaC treats a drifted `plan`.

## The audit trail is the git history — protect it

Because the deploy *is* the commit, your deploy audit trail inherits everything git gives you, and you must protect it accordingly:

- **Require PRs and reviews on the config repo's `main`.** A merge to `main` is a production deploy; gate it like one (branch protection, required reviewers, no force-push).
- **Sign commits / require signed commits** so "who deployed this" is cryptographically attributable, not just "whoever set that git author string."
- **Verify image signatures at admission** (cosign policy / Kyverno / OPA Gatekeeper) so even a controller can't deploy an unsigned or untrusted image — closing the loop on the supply-chain attestations CI produced.
- **Rollback is `git revert`**, which is itself an audited, reviewed event — not an out-of-band emergency action that leaves no trace.

This is the answer to "no audit trail of what is deployed": there is now a complete, immutable, signed history of every desired-state change, and the cluster provably matches the latest of them.

## Common mistakes

| Mistake | Why it bites | Fix |
|---------|-------------|-----|
| CI runs `kubectl apply` / `helm upgrade` at the cluster | Push-based; long-lived creds in CI; no drift handling; no real audit trail | Pull-based controller (Argo CD / Flux); CI stops at registry + config commit |
| Long-lived kubeconfig / cluster-admin token in CI | Standing skeleton key; leak the runner, leak the cluster | Controller runs *in* cluster; no credential ever leaves it |
| Manifests in the same repo as app source | Code merge mutates the cluster; can't separate review rules; tangled audit trail | Separate config/deploy repo (or strictly separated path) |
| Manual sync only, `selfHeal: false` | Re-introduces drift between syncs; controller is a button a human presses | `automated: { selfHeal: true, prune: true }` |
| No `prune` | Deleted-from-git resources linger as zombies | `prune: true` |
| Deploying a mutable tag (`:latest`, re-pushable `:v2.4.0`) | Desired state isn't immutable; breaks signature verification; "what's running" is ambiguous | Pin the image **digest** in the manifest |
| Treating `OutOfSync` as normal | The one signal that catches click-ops gets ignored | Alert on sustained `OutOfSync`; treat as an incident |
| "We keep YAML in git, so it's GitOps" | Violates pull + continuous-reconcile; surprises you under load | Hold all four OpenGitOps principles, not just versioning |
| No admission-time signature verification | A bad/unsigned image can still land | cosign/Kyverno/Gatekeeper policy verifies signatures + provenance |
| Unprotected config-repo `main` | A force-push or unreviewed merge *is* an unreviewed prod deploy | Branch protection, required reviews, signed commits |
| One controller per team, no convention | Can't reason across clusters; tribal knowledge | Standardize the controller; record it in an ADR |

## Red flags — STOP

- "I'll just `kubectl apply` it from the pipeline for now." → That is the imperative push you're trying to kill. The cluster's state is now whatever ran last, with no audit trail.
- "Put the kubeconfig in CI secrets." → You just minted a standing skeleton key to the cluster. Use a pull-based controller; nothing leaves the cluster.
- "Let's keep the manifests in the app repo, simpler." → Now every code merge can mutate prod and your deploy history is tangled with feature commits. Split the repos.
- "Turn off self-heal, it's safer." → Self-heal *is* the safety. Off, you've re-invented drift.
- "We'll just `kubectl edit` the replica count quickly." → That's drift. Change git; let the controller reconcile.
- "Pin to `:latest`, it's easier." → Mutable tag = non-immutable desired state = your signatures and rollbacks both lie. Pin the digest.
- "OutOfSync is always red here, ignore it." → That's the diagnosis (chronic click-ops), not a defense.
- "Rollback? I'll re-run the old build." → If the artifacts are gone you can't. `git revert` the config commit; the controller does the rest.

## Rationalizations and their counters

- **"Push is simpler, we already have the pipeline."** Simpler today, opaque tomorrow: you've kept the credential-in-CI risk and the "what's running?" problem. The controller is a one-time setup that removes both permanently.
- **"GitOps is overkill for our size."** The smaller the team, the fewer people hold the deploy knowledge and the more a standing cluster credential in CI hurts when it leaks. Small teams benefit most from "the deploy is a reviewed commit and rollback is a revert."
- **"Self-heal will revert my emergency hotfix."** A hotfix the cluster will silently lose is exactly the change that must go through git — so it survives, is reviewed, and is in the history. Self-heal reverting it is the system telling you to commit it.
- **"Two repos is more overhead."** The overhead is the point: it makes "change the code" and "change what's deployed" two separately-reviewable, separately-permissioned events. One repo collapses that distinction at your expense.
- **"We don't need signed commits/images, we trust our team."** The audit trail and admission policy aren't about distrust; they're about being able to *prove* what shipped after an incident, and stopping a compromised pipeline from deploying arbitrary images.

## The bottom line

Git holds the desired state; an in-cluster controller pulls it and reconciles continuously; nothing else writes to the cluster. CI builds and signs an artifact and writes one commit to the config repo — it never holds a cluster credential. Drift is detected and self-healed. The deploy is a reviewed, signed commit, so the git history *is* the audit trail and rollback is `git revert`. Hold the four OpenGitOps principles and "what is running in prod, who changed it, and how do we undo it" all have one answer: the latest commit the controller synced. Break the pull boundary — one `kubectl apply` from CI, one kubeconfig in a runner — and you're back to a cluster whose state is a rumor.

## Cross-references

- `deployment-strategies` (this pack) — progressive delivery (Argo Rollouts / Flagger, canary/blue-green, Gateway API traffic-shifting, metric-gated automated rollback) that rides on top of the GitOps control plane.
- `ci-cd-pipeline-architecture` (this pack) — the build/test/scan/sign stages that produce the artifact CI commits to the config repo; where the CI/CD split is enforced.
- `infrastructure-as-code` (this pack) — the same desired-state-in-git discipline for cloud infrastructure; drift detection and plan/apply gates are the IaC analog of reconciliation.
- `ordis-security-architect` — supply-chain attestation (SLSA, cosign keyless via Fulcio/Rekor), admission-time signature verification, and OIDC federation for credential-free CI.
- `/axiom-solution-architect` — recording the Argo CD vs Flux choice and the one-repo-vs-two decision as ADRs.
