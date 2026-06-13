---
name: orchestration-and-scheduling
description: Use when pods get traffic before the app is ready, when a deadlocked process keeps serving 500s because nothing restarts it, when one greedy container OOM-kills its neighbors on a node, when a deploy with a broken image takes the whole service down at once, when "scale up" means SSHing in to edit a replica count, when nodes drain during maintenance and take a quorum with them, when secrets are baked into images or env vars in plaintext, when a slow-booting service is killed mid-startup by an impatient probe, or when you cannot answer "what happens to in-flight requests when this pod dies" — covers Kubernetes liveness/readiness/startup probes, resource requests and limits, HPA/VPA autoscaling, controlled rollouts and rollbacks, PodDisruptionBudgets, graceful shutdown, and config/secret mounting.
---

# Orchestration and Scheduling

## The production stake

A Kubernetes cluster is a scheduler that will do *exactly* what your manifests tell it to — including the things you forgot to tell it. A pod with no readiness probe gets added to the Service the instant the container process starts, so the load balancer routes real user traffic to an app that is still loading config, warming a cache, or opening a DB pool. Those requests 500. A pod with no liveness probe that deadlocks stays in the rotation forever, a black hole that the scheduler is convinced is healthy. A container with no memory limit that leaks will consume the whole node, and the kernel OOM-killer — which does not respect your business priorities — reaps whatever it reaps, frequently the *other* tenants on that node. None of these are exotic failures. They are the default behavior of a cluster you have under-specified.

Orchestration discipline is not "we run on Kubernetes." It is the set of declarations that make the scheduler's automatic behavior *match what you actually want under load and under failure*. The control loop is always running. The only question is whether you have told it the truth about your workload.

This sheet is about telling it the truth. It is not a `kubectl` tutorial.

## June 2026 baseline — what's current

- **Runtime is containerd.** Dockershim has been gone for years; the node runtime is containerd (or CRI-O). Nothing here assumes a Docker daemon on nodes.
- **Native sidecar containers are first-class.** Sidecars are now *restartable init containers* (`initContainers` with `restartPolicy: Always`), which start before app containers, run for the pod's life, and are torn down last. Use this for proxies/agents/log shippers instead of the old "extra container in `containers[]`" pattern — it fixes startup ordering and the job-never-completes problem.
- **Gateway API has replaced Ingress for traffic.** Ingress-NGINX retired March 2026. North-south traffic uses Gateway API v1.5 (Feb 2026) via an implementation like Envoy Gateway, Istio, Cilium, or Kong. Probes and PDBs below are independent of the ingress layer, but do not author new `Ingress` objects.
- **Observability is OpenTelemetry/OTLP**, vendor-neutral — probes tell the scheduler about health; OTel tells *you*. Don't conflate the two.

## The invariants

1. **Every container declares its health.** Readiness gates traffic; liveness gates restarts; startup protects slow boots. A container with none of these is lying to the scheduler about its state.
2. **Every container declares its appetite.** Requests drive scheduling and protect you from noisy neighbors; limits cap blast radius. Unbounded containers are a node-wide outage waiting for a leak.
3. **Disruption is bounded.** Voluntary disruption (drains, rollouts, scale-down) is governed by a PodDisruptionBudget so maintenance can't take your quorum below the line.
4. **Config and secrets are mounted, not baked.** Images are environment-agnostic; configuration arrives at runtime from ConfigMaps and Secrets (or an external store), never compiled into the image or pasted in plaintext.

If any of these is unspecified, the scheduler fills the gap with a default that is convenient for *it*, not safe for *you*.

## Probes — the three are not interchangeable

This is the single most misconfigured area in Kubernetes, because all three probes share a syntax and have completely different jobs. Confusing them causes either cascading restart storms or traffic-to-dead-pods.

| Probe | Question it answers | Failure action | Get it wrong and… |
|-------|--------------------|----------------|-------------------|
| **readiness** | "Can this pod take traffic *right now*?" | Removed from Service endpoints (no restart) | Traffic hits a pod that's loading → 500s on every deploy |
| **liveness** | "Is this process wedged and unrecoverable?" | Container is **killed and restarted** | Too aggressive → healthy-but-slow pods get restart-looped under load |
| **startup** | "Has this slow-booting app finished starting?" | Disables liveness/readiness until it passes | Missing → a slow boot is killed by liveness before it ever comes up |

**The cardinal rules:**

- **Liveness must be cheap and local.** It checks "is *this process* deadlocked," nothing else. **Never** put a database call, a downstream HTTP call, or any dependency in a liveness probe. If your DB has a blip and your liveness probe checks the DB, Kubernetes restarts *every pod at once* — you turned a dependency hiccup into a self-inflicted total outage.
- **Readiness may check dependencies** — that's the point. If the pod can't serve (DB pool not open, cache cold), readiness should fail so traffic routes elsewhere, *without* killing the pod.
- **Startup probes protect slow boots.** A JVM or a service loading a large model can take 60–120s. Without a startup probe you either set `liveness.initialDelaySeconds` so high it's useless in steady state, or liveness kills the pod mid-boot. The startup probe holds liveness/readiness off until the app is up, then steps aside.

### Example 1 — a correctly-probed Deployment with all three probes, resources, and graceful shutdown

```yaml
# deployment.yaml — every field here is load-bearing under failure
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orders-api
  labels: { app: orders-api }
spec:
  replicas: 6
  selector: { matchLabels: { app: orders-api } }
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0        # never drop below desired capacity mid-rollout
      maxSurge: 2              # add 2 new pods at a time, then retire old
  template:
    metadata:
      labels: { app: orders-api }
    spec:
      terminationGracePeriodSeconds: 45   # must exceed preStop sleep + longest in-flight request
      containers:
        - name: orders-api
          image: registry.example.com/orders-api@sha256:abc123...  # digest-pinned, never :latest
          ports: [{ containerPort: 8080, name: http }]

          # --- resources: scheduling + noisy-neighbor protection ---
          resources:
            requests: { cpu: "250m", memory: "256Mi" }   # what the scheduler reserves
            limits:   { cpu: "1",    memory: "512Mi" }   # hard cap; memory limit => OOM-kill on breach

          # --- startup: protect a ~30-60s boot; nothing else fires until this passes ---
          startupProbe:
            httpGet: { path: /healthz/start, port: http }
            periodSeconds: 5
            failureThreshold: 24          # 24 * 5s = up to 120s to start before we give up

          # --- readiness: gates TRAFFIC; may check dependencies; never restarts ---
          readinessProbe:
            httpGet: { path: /healthz/ready, port: http }   # this handler verifies DB pool + cache
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 3            # 3 strikes => pulled from Service, pod left alive

          # --- liveness: gates RESTART; cheap + local ONLY; no dependencies ---
          livenessProbe:
            httpGet: { path: /healthz/live, port: http }    # returns 200 iff the event loop is responsive
            periodSeconds: 10
            timeoutSeconds: 2
            failureThreshold: 3            # only a truly wedged process gets restarted

          # --- graceful shutdown: stop taking traffic, then drain in-flight ---
          lifecycle:
            preStop:
              exec:
                # SIGTERM races with endpoint removal; this sleep lets the proxy
                # observe NotReady and stop routing BEFORE the app exits.
                command: ["sh", "-c", "sleep 10"]
```

The `preStop` + `terminationGracePeriodSeconds` pairing is the part everyone skips and then wonders why every rollout drops a handful of requests: when a pod is deleted, Kubernetes *simultaneously* sends SIGTERM and starts removing the endpoint, but endpoint propagation to every proxy is not instant. The `preStop` sleep keeps the old pod serving for the few seconds it takes for routers to stop sending it new work, and the grace period gives in-flight requests time to finish before the kill.

## Resource requests and limits — the noisy-neighbor firewall

This is where shared-cluster outages come from. The scheduler places pods using **requests**, not actual usage; **limits** cap what a pod can consume.

- **Requests = what's reserved.** Set them to realistic steady-state usage. Too low → the scheduler overpacks the node and everything thrashes under load. Too high → you waste capacity and pods go Pending.
- **Memory limit = a hard wall enforced by OOM-kill.** Memory is *incompressible* — you cannot throttle it. A container that exceeds its memory limit is killed (OOMKilled). A container with **no** memory limit that leaks will eat the node and the kernel OOM-killer takes out whatever it decides, often the neighbor, not the culprit. **Always set a memory limit.**
- **CPU limit = throttling, not killing.** CPU is *compressible*: over-limit means the container is throttled (latency), not killed. A widely-held position is to **set CPU requests but omit CPU limits** for latency-sensitive services, so a pod can burst into idle node capacity instead of being throttled while CPU sits free. Set CPU limits when you need hard multi-tenant isolation or predictable cost; otherwise let requests do the protecting.
- **Quality of Service follows from this.** `requests == limits` for both CPU and memory ⇒ **Guaranteed** (last to be evicted under node pressure). Requests set, limits unset/higher ⇒ **Burstable**. Nothing set ⇒ **BestEffort** (first to be evicted, and the cause of most "why did my pod vanish" tickets). Production workloads should be Guaranteed or deliberately Burstable — never BestEffort by accident.

The noisy-neighbor outage in one sentence: a BestEffort pod with no memory limit leaks, fills the node, and the OOM-killer evicts your *Guaranteed* database replica because the leaker had no bound to enforce.

## Autoscaling — HPA and VPA do different jobs and fight if combined naively

- **HPA (Horizontal Pod Autoscaler)** changes the **replica count** based on a metric (CPU, memory, or custom/external metrics like queue depth or p95 latency via the metrics adapter). This is your front-line scaler for stateless request-serving workloads.
- **VPA (Vertical Pod Autoscaler)** changes the **requests/limits** of pods. Useful for workloads you can't shard horizontally, or to *right-size* requests you guessed wrong. Historically VPA evicts pods to resize them; in-place pod resize (resizing without restart) is the direction the ecosystem is moving and removes that disruption.
- **Do not run HPA and VPA on the same metric.** If both scale on CPU they oscillate — HPA adds pods, VPA shrinks each pod's request, HPA sees high utilization again, repeat. Standard pattern: **HPA on a custom/throughput metric, VPA on memory only**, or use VPA in *recommendation mode* to size requests and let HPA own the live scaling.
- **Cluster Autoscaler / Karpenter** scale the *nodes* so HPA's new pods have somewhere to land. HPA scaling pods onto a full cluster just produces Pending pods — node autoscaling is the other half.

### Example 2 — HPA scaling on a custom queue-depth metric with stabilized scale-down

```yaml
# hpa.yaml — scale on actual work backlog, not just CPU, and don't flap on scale-down
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: orders-worker
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: orders-worker
  minReplicas: 3
  maxReplicas: 40
  metrics:
    # CPU as a floor so we don't melt pods even if the queue looks shallow
    - type: Resource
      resource:
        name: cpu
        target: { type: Utilization, averageUtilization: 70 }
    # The real signal: pending messages per replica (via Prometheus/KEDA external metric)
    - type: External
      external:
        metric:
          name: rabbitmq_queue_messages_ready
          selector: { matchLabels: { queue: orders } }
        target:
          type: AverageValue
          averageValue: "30"        # aim for ~30 backlog items per worker
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0     # react fast to a backlog spike
      policies:
        - { type: Percent, value: 100, periodSeconds: 30 }   # can double quickly
    scaleDown:
      stabilizationWindowSeconds: 300   # wait 5 min of low load before shrinking
      policies:
        - { type: Pod, value: 1, periodSeconds: 60 }         # remove at most 1 pod/min
```

The asymmetric `behavior` block is the lesson: scale **up** aggressively (a backlog hurts users now) and scale **down** slowly with a stabilization window (so a momentary lull doesn't tear down capacity you'll need 90 seconds later, which produces the classic flap-and-thrash pattern). Scaling on **queue depth** rather than CPU means you add workers when work piles up, not after the pods are already saturated.

## Rollouts and rollback — the deploy is a control loop, not an event

A naive Deployment update with no surge control can replace all pods at once and take the service down if the new image is broken. The Deployment controller gives you safe primitives; use them.

- **`maxUnavailable: 0` + `maxSurge`** keeps full capacity during a rolling update (add new, verify ready via the readiness probe, *then* retire old). Without this, a RollingUpdate can dip below your needed replica count mid-deploy.
- **Readiness gates the rollout.** A new pod that never becomes Ready stalls the rollout instead of replacing healthy pods with broken ones — *this is why a real readiness probe is non-optional*. With no readiness probe, "container started" counts as "ready," and you roll a crashing image across the whole fleet.
- **Rollback is a first-class operation:** `kubectl rollout undo deployment/orders-api` reverts to the previous ReplicaSet. Know it cold and put it in the runbook. `kubectl rollout status` is how CI waits for a deploy to actually succeed before declaring victory.
- **For traffic-shifted canary/blue-green**, this Deployment-level rolling update is the floor, not the ceiling — graduate to **Argo Rollouts (with Argo CD) or Flagger (with Flux)** driving Gateway API traffic weights with metric-based automated rollback. That progressive-delivery layer is covered in `deployment-strategies` (this pack); don't reinvent it here.

```bash
# Watch a rollout to completion; if it's wedged, roll back deterministically.
kubectl rollout status deployment/orders-api --timeout=120s \
  || kubectl rollout undo deployment/orders-api
```

## PodDisruptionBudgets — protecting quorum from voluntary disruption

A PDB bounds **voluntary** disruption: node drains for maintenance/upgrades, cluster autoscaler scale-down, `kubectl drain`. It does **not** protect against involuntary disruption (node hardware death, OOM-kill) — that's what replica count and anti-affinity are for.

Without a PDB, a routine node drain (a cluster upgrade rolling through the fleet) can evict every replica of a service at once because the eviction API has no instruction to keep any alive. For anything with a quorum or a minimum-capacity requirement, a PDB is mandatory.

```yaml
# pdb.yaml — the upgrade can take pods, but never below 4 of our 6
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: orders-api
spec:
  minAvailable: 4                 # OR maxUnavailable: 2 — express one, not both
  selector:
    matchLabels: { app: orders-api }
```

Pick `minAvailable` for "I must always have at least N serving" and `maxUnavailable` for "you may take at most N at a time." A common trap: setting `minAvailable` equal to `replicas` makes the PDB *block all drains forever* and stalls cluster upgrades — leave headroom. Pair the PDB with `topologySpreadConstraints` so your N replicas aren't all on one node (a PDB can't save you from a single-node failure if everything's co-located).

## Config and secret mounting — images are environment-agnostic

A correct image runs unchanged in dev, staging, and prod; only the mounted configuration differs. Baking config or secrets into the image is how a staging credential ends up in a prod container, and how a leaked image leaks your secrets.

- **ConfigMaps for non-sensitive config**, mounted as files or env. **Prefer file mounts over env vars** for anything that may change: a mounted ConfigMap *projects updates into the pod* (env vars are frozen at start), and env vars leak into crash dumps, child processes, and `/proc`.
- **Secrets are base64, not encrypted, by default.** Enable **encryption at rest** for Secrets in etcd. Better: keep secrets in an external store (Vault, cloud secrets manager) and pull them via the **Secrets Store CSI Driver** or **External Secrets Operator**, so the source of truth is the vault and rotation is real. Never put a secret in a ConfigMap, an image layer, or a plaintext env in the manifest.
- **Mounted secrets should be read-only**, and the pod should run as non-root with a restrictive `securityContext`.

```yaml
# Mount config as a file (live-updatable) and a secret read-only from an external store.
spec:
  containers:
    - name: orders-api
      envFrom:
        - configMapRef: { name: orders-config }       # non-sensitive only
      volumeMounts:
        - { name: app-config, mountPath: /etc/orders, readOnly: true }
        - { name: db-creds,   mountPath: /etc/secrets, readOnly: true }
      securityContext:
        runAsNonRoot: true
        readOnlyRootFilesystem: true
        allowPrivilegeEscalation: false
  volumes:
    - name: app-config
      configMap: { name: orders-config }
    - name: db-creds
      csi:                                              # External Secrets / CSI driver
        driver: secrets-store.csi.k8s.io
        readOnly: true
        volumeAttributes: { secretProviderClass: orders-db-creds }
```

## Common mistakes

| Mistake | Why it bites | Fix |
|---------|-------------|-----|
| No readiness probe | Pod gets traffic while still booting → 500s on every deploy | Readiness probe that verifies the pod can actually serve |
| Dependency check in liveness probe | A DB blip restarts *every* pod at once → self-inflicted outage | Liveness checks only local process health; deps go in readiness |
| No startup probe on a slow-booting app | Liveness kills the pod mid-boot, crash-loops forever | Startup probe with a generous `failureThreshold` |
| No memory limit | A leak eats the node; OOM-killer reaps neighbors | Always set a memory limit (hard wall) |
| No resource requests (BestEffort) | First to be evicted under node pressure; pods vanish | Set requests; aim for Guaranteed or deliberate Burstable |
| HPA and VPA on the same metric | They oscillate, replicas/sizes flap endlessly | Split metrics, or run VPA in recommendation mode |
| No graceful-shutdown `preStop`/grace period | Rollouts drop in-flight requests | `preStop` sleep + `terminationGracePeriodSeconds` > drain time |
| `maxUnavailable` default during rollout | Capacity dips mid-deploy | `maxUnavailable: 0` + `maxSurge` |
| No PDB | A node drain/upgrade evicts all replicas at once | PDB with `minAvailable` leaving headroom |
| `minAvailable == replicas` | Blocks every drain; cluster upgrades stall forever | Leave headroom below replica count |
| Secrets in image/env/ConfigMap | Leaks; no rotation; staging creds reach prod | External store via CSI/ESO, encrypted etcd, read-only mount |
| `image: app:latest` | Non-reproducible rollouts; can't pin a rollback | Digest- or version-pinned images |
| Old bare-sidecar in `containers[]` | Startup ordering races; Jobs never complete | Native sidecar (`initContainers` + `restartPolicy: Always`) |
| Authoring new `Ingress` objects | Ingress-NGINX retired Mar 2026 | Gateway API v1.5 + an implementation |

## Red flags — STOP

If you hear yourself or a teammate say any of these, fix the spec before shipping:

- "The app's fine, it doesn't need a readiness probe." → Then it gets traffic while booting. Add it.
- "I'll just have liveness ping the database." → That makes a DB blip restart the whole fleet. Never.
- "We don't set memory limits, they get in the way." → A single leak then takes the node and your neighbors with it.
- "Just let it use whatever CPU/memory it needs." → BestEffort pods are evicted first. That's a future 2am page.
- "Set replicas to 1, HPA will scale it." → HPA can't sustain availability from a single replica during disruption; min 2–3.
- "We don't need a PDB, drains are rare." → They're rare until the cluster upgrade rolls through and takes everything.
- "Put the secret in an env var, it's faster." → Env leaks into dumps/children; no rotation. Mount from a store.
- "Pin the digest later, `latest` is fine." → Then you can't reproduce or roll back to a known-good image.
- "We'll add graceful shutdown when we see dropped requests." → You're already dropping them; you just haven't measured it.

## Rationalizations and their counters

- **"Probes are overhead, the app self-heals."** Self-heals *how*, if nothing restarts it? The probe is the mechanism, not a redundancy.
- **"Limits cause throttling, so we skip them."** Skip the *CPU* limit if you must — but the *memory* limit is the only thing standing between one leak and a node-wide outage. They are not the same decision.
- **"Autoscaling is for big shops."** A backlog spike hurts a 3-pod service exactly as much. HPA on a queue metric is small-team table stakes.
- **"PDBs are for stateful systems."** Any service with a minimum capacity has a quorum in practice — a PDB is how you tell the upgrade not to cross it.
- **"Config in the image is simpler."** Until you need the same image in two environments, or the leaked image leaks your secrets. One image, mounted config, always.
- **"Graceful shutdown is gold-plating."** It's the difference between a zero-downtime deploy and a deploy that drops a slice of requests every single time, invisibly.

## The bottom line

The scheduler does exactly what you declare and defaults the rest in its own favor. Declare health honestly (readiness gates traffic, liveness gates restart and never touches a dependency, startup protects the boot). Declare appetite honestly (requests for scheduling, a memory limit always, CPU limits deliberately). Bound disruption with a PDB that leaves headroom, roll out with surge and readiness gating, keep `rollout undo` in the runbook, and mount config and secrets at runtime from a real store. Do that and a wedged process restarts itself, a leaky neighbor can't take the node, a node drain can't take quorum, and a bad image stalls instead of breaking the service. Skip any of it and you've handed the control loop a half-truth — and it will act on the half-truth with perfect, automatic consistency.

## Cross-references

- `deployment-strategies` (this pack) — progressive delivery (Argo Rollouts/Flagger), canary/blue-green, Gateway API traffic-shifting, and expand/contract migrations that sit on top of these rollout primitives.
- `infrastructure-as-code` (this pack) — provisioning the cluster, node pools, and autoscaler config as code.
- `ci-cd-pipeline-architecture` (this pack) — where `rollout status`/`undo` gates live in the deploy pipeline.
- `ordis-security-architect` — Secret management, etcd encryption, Pod Security Standards, and `securityContext` hardening.
- `ordis-quality-engineering` — chaos/disruption testing to validate that probes, PDBs, and graceful shutdown actually hold under failure.
