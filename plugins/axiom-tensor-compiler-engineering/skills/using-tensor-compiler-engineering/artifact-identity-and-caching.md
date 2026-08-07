---
name: artifact-identity-and-caching
description: Use when designing content-addressed compiled artifacts and their cache — keying on canonical semantic hash plus device, dtype and compiler version, invalidation rules, and reusing artifacts across runs or trials without semantic risk. Read when a compilation cache returns a stale or wrong artifact.
---

# Artifact Identity and Caching

## When to Use

- Designing the cache key for compiled artifacts.
- A cache returns something that runs but is wrong, or misses when it should hit.
- Reusing artifacts across experiments, runs, or machines.
- Deciding what invalidates a cached artifact.

---

## Core Principle

**A cache key is a claim: "any two compilations with this key are interchangeable." Every field you leave out is a way for that claim to be false, and every field you add unnecessarily is a cache miss you pay for forever.**

Correctness sets the floor and performance sets the ceiling. Get the floor wrong and you serve wrong answers; get the ceiling wrong and you have a slow but correct system. **Only one of those is recoverable**, so when in doubt, add the field.

### The failure this prevents

Keying on source syntax is the classic error, and it is wrong in both directions at once:

- **False hits (dangerous).** Two different programs share a key. The same source text compiled for `float32` and `bfloat16`, or for CPU and CUDA, or under different backend flags, produce different artifacts. Key on source alone and the second compilation gets the first's artifact — running a bfloat16 model with float32 kernels, or worse.
- **False misses (merely expensive).** One program has many source spellings. Reformatting, renaming a variable, or a comment change invalidates every artifact in the cache. Teams respond to a 0% hit rate by disabling the cache, which is the correct response to a cache keyed on the wrong thing.

Both failures have the same root cause: **syntax is not identity.** Semantics is.

---

## Key Composition

```python
import hashlib

def artifact_key(spec, target, compiler) -> str:
    """Every component is here because omitting it produces a WRONG artifact,
    not merely a slow one."""
    parts = [
        spec.semantic_hash,                    # what the program means
        spec.numerical_contract_version,       # tolerances, tf32, determinism, accumulation
        target.device_capability,              # "sm_90" — NOT "cuda:0"
        target.dtype,
        compiler.version,                      # git sha of the compiler itself
        compiler.pass_pipeline_hash,           # which passes, in which order
        stable_hash(target.backend_flags),     # tf32, cudnn.benchmark, determinism
        spec.ir_version,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]
```

Why each field, stated as the failure its absence causes:

| Field | Omitting it means |
|-------|------------------|
| `semantic_hash` | Different programs collide. The core failure |
| `numerical_contract_version` | An artifact built under looser tolerances serves a stricter contract |
| `device_capability` | An sm_80 artifact runs on sm_90 with different kernels available |
| `dtype` | A float32 artifact serves a bfloat16 request |
| `compiler.version` | Yesterday's compiler bug is served forever, surviving the fix |
| `pass_pipeline_hash` | Reordering or disabling a pass produces a stale hit |
| `backend_flags` | TF32-on and TF32-off artifacts are interchangeable. They are not |
| `ir_version` | An IR semantics change silently reuses artifacts built under the old meaning |

Two subtleties that decide whether the key is right:

**`device_capability`, not device index.** `cuda:0` and `cuda:1` on identical hardware must share artifacts — keying on the index is a 2× cache miss for nothing. But sm_80 and sm_90 must not, because kernel availability differs. Capability is the correct granularity.

**`compiler.version` must be the compiler's own SHA**, not the torch version. When you fix a miscompile, every artifact built by the broken compiler must become unreachable. If the key does not include your compiler's version, the fix ships and the bug persists in cache — and the bug reports continue, which is a uniquely demoralising way to spend a week.

### What must NOT be in the key

| Never key on | Because |
|--------------|---------|
| Source text, file path, mtime | Syntax is not identity — false hits and false misses |
| Parameter values | Every optimiser step invalidates everything |
| Wall-clock time, build id | Guarantees a 0% hit rate |
| Hostname, user | Prevents cross-machine sharing, which is the point of a cache |
| Batch size (unless shape-specialised) | Only if the artifact is genuinely specialised — then it is a target property, and say so |

---

## Gate Before Cache

The ordering rule from `compiler-architecture-for-tensor-programs.md`, restated because this is where it is enforced:

```python
def compile_and_publish(spec, target, cache) -> "CompiledArtifact":
    key = artifact_key(spec, target, COMPILER)
    if (hit := cache.get(key)) is not None:
        if hit.conformance_report_id is None:      # defensive: pre-gate entries
            cache.evict(key)
        else:
            return hit

    artifact = compile_artifact(spec, target)
    report = run_conformance(spec, artifact, build_artifact)   # independent gate
    if not report.passed:
        quarantine(artifact, report)     # NOT cached — unreachable, retained for triage
        raise ConformanceFailure(report)

    artifact.conformance_report_id = report.id
    cache.put(key, artifact)
    return artifact
```

Two properties this gives you, both load-bearing:

- **Everything in the cache passed conformance.** A cache with a `verified` flag is a cache where any code path that forgets to check the flag serves unverified artifacts. Making unverified entries unreachable removes the possibility rather than documenting it.
- **A cache hit is as trustworthy as a fresh compile.** Otherwise the cache is a correctness risk and someone will eventually — correctly — disable it in production, at which point you have paid for a cache you cannot use.

---

## Invalidation

Content-addressed caches do not need explicit invalidation for the things in the key: a change produces a different key, and the old entry is garbage rather than wrong. That is the main argument for content addressing over mutable keys.

What still needs handling:

| Event | Action | Why |
|-------|--------|-----|
| Compiler bug fixed | Bump `compiler.version` | Old artifacts become unreachable, not merely stale |
| Numerical contract tightened | Bump `numerical_contract_version` | Old artifacts were verified against a weaker contract |
| Conformance gate strengthened | Re-verify or evict | Old artifacts passed a weaker gate — see below |
| Storage pressure | Evict by LRU + compile cost | Evicting a 40-minute artifact to keep a 2-second one is backwards |
| Toolchain upgrade | Bump via `backend_flags`/toolchain in the key | Different codegen |

The **conformance gate strengthened** row is the one people miss. If you add gradient conformance to a gate that previously did forward-only, every cached artifact was verified by the old, weaker gate. Two honest options: version the gate and include it in the key, or re-verify the cache in the background and evict failures. Doing neither means the cache silently preserves exactly the artifacts your new check was written to catch.

```python
GATE_VERSION = 3   # bump when checks are ADDED; include in the key or re-verify

def revalidate_cache(cache, gate_version=GATE_VERSION):
    """Background sweep. Log evictions loudly — a spike means the new check works."""
    for key, artifact in cache.items():
        if artifact.gate_version >= gate_version:
            continue
        report = run_conformance(load_spec(artifact.spec_id), artifact, build_artifact)
        if report.passed:
            artifact.gate_version = gate_version
        else:
            cache.evict(key)
            alert(f"cached artifact {key} failed the strengthened gate: {report.summary()}")
```

---

## Reuse Across Runs

Reusing artifacts across experiments, trials, or training runs is where caching pays for itself, and where the semantic risk is highest — because the reuse is invisible at the call site.

The rule: **artifacts are reusable exactly when the key is complete.** If the key contains everything that affects behaviour, reuse is safe by construction. If it does not, reuse spreads one artifact's wrongness across every run that hit it — which also destroys the independence of runs you may be comparing.

Two guards worth having in an experimental setting:

```python
@dataclass
class CacheHitRecord:
    """Recorded PER RUN, so any result can be traced to the artifacts that produced it."""
    key: str
    artifact_id: str
    compiled_at: str          # when the artifact was BUILT, not when it was used
    compiler_version: str
    conformance_report_id: str
```

- **Record every hit in the run's provenance.** When run A and run B disagree, "they used the same artifact, built three weeks ago by compiler v1.2" is the first thing you want to know. Without it, a shared artifact is an invisible coupling between supposedly independent runs.
- **Support `--no-cache` and run it periodically in CI.** If cached and fresh compilation ever disagree, the key is incomplete. This is a direct test of the cache's central claim, it is cheap, and nothing else tests it.

---

## Executable Decision Procedure: Auditing a Cache Key

```python
def audit_cache_key(key_fields: set, compiler_facts: dict) -> list[str]:
    """compiler_facts: {"reads_flags": [...], "device_conditional": bool,
                        "dtype_conditional": bool, "has_pass_config": bool}"""
    findings = []
    REQUIRED = {"semantic_hash", "device_capability", "dtype",
                "compiler_version", "numerical_contract_version"}
    for f in REQUIRED - key_fields:
        findings.append(f"CRITICAL: key omits {f!r} — two non-interchangeable artifacts "
                        "can collide")

    FORBIDDEN = {"source_text", "file_path", "mtime", "timestamp", "hostname",
                 "parameter_values", "build_id"}
    for f in FORBIDDEN & key_fields:
        findings.append(f"HIGH: key includes {f!r} — syntax/environment is not identity; "
                        "causes false misses and (for source_text) false hits too")

    if compiler_facts["reads_flags"] and "backend_flags" not in key_fields:
        findings.append("CRITICAL: compiler reads backend flags that are not in the key — "
                        f"{compiler_facts['reads_flags']}")
    if compiler_facts["has_pass_config"] and "pass_pipeline_hash" not in key_fields:
        findings.append("HIGH: pass pipeline is configurable but not keyed — disabling a "
                        "pass yields a stale hit")
    if "device_index" in key_fields:
        findings.append("MEDIUM: keyed on device INDEX rather than capability — "
                        "identical GPUs will not share artifacts")
    return findings
```

The `--no-cache` differential is the empirical companion to this static audit: the audit finds fields you forgot to think about, the differential finds fields you thought about and got wrong.

---

## RED → GREEN Scenario

**RED.** A pipeline caches compiled artifacts keyed on `sha256(model_source + str(input_shape))`. It works for months. Then mixed precision is introduced: the same model is compiled for `float32` on some runs and `bfloat16` on others.

The bfloat16 runs get the float32 artifact. They run — shapes match, nothing raises — and produce float32-precision results while reporting bfloat16 memory savings that are not real. The bfloat16 experiments look surprisingly accurate. Someone writes up "bfloat16 costs us nothing on this architecture", and the finding is a cache bug.

Then someone reformats the codebase. Hit rate goes to zero, compile time triples, and the cache is disabled as "not worth it". Both failures, from one key.

**GREEN.**

```python
key = artifact_key(spec, target, COMPILER)   # semantic_hash + dtype + capability
                                             # + contract version + compiler version
                                             # + pass pipeline + backend flags + ir version
```

- dtype in the key: bfloat16 gets its own artifact. The false hit is impossible.
- `semantic_hash` instead of source text: reformatting does not invalidate anything. Hit rate survives.
- `compiler.version`: when the miscompile is found and fixed, every affected artifact becomes unreachable at once.

And a nightly CI job compiles with `--no-cache` and compares against the cached path. That job is what would have caught the original bug in a day rather than a quarter.

Generalisable: **the two cache failure modes have opposite symptoms and one cause.** False hits are silent and dangerous; false misses are loud and merely expensive. A team that fixes only the loud one — by loosening the key — makes the silent one more likely. Key on semantics, and both go away together.

---

## Anti-Patterns

| Pattern | Why it fails | Fix |
|---------|--------------|-----|
| Key on source text / path / mtime | False hits *and* false misses | Key on `semantic_hash` |
| dtype or device capability omitted | Serves an artifact for the wrong target | Include both |
| Compiler version omitted | Fixed bugs live on in cache | Include the compiler's own SHA |
| Backend flags omitted | TF32-on and TF32-off artifacts interchange | Include a stable hash of flags |
| Keyed on device index | Identical GPUs do not share | Capability, not index |
| Parameter values in the key | Every optimiser step invalidates everything | Semantics only |
| Cache insert before the gate | Unverified artifacts reachable | Gate, then cache |
| `verified` flag instead of exclusion | Any path that forgets the flag serves them | Do not insert failures |
| Gate strengthened, cache untouched | Preserves exactly what the new check targets | Version the gate; re-verify |
| Evicting by size alone | Discards expensive artifacts | LRU weighted by compile cost |
| No `--no-cache` differential | The cache's central claim is untested | Nightly comparison |
| Cache hits not recorded per run | Invisible coupling between "independent" runs | Provenance record per run |

---

## Checklist

- [ ] Key includes semantic hash, contract version, device capability, dtype, compiler version, pass-pipeline hash, backend flags, IR version
- [ ] Key excludes source text, paths, mtimes, hostnames, parameter values, timestamps
- [ ] Device *capability*, not device index
- [ ] Conformance runs before cache insertion; failures quarantined, not flagged
- [ ] Every cached artifact carries its `conformance_report_id`
- [ ] Gate version tracked; strengthening triggers re-verification or eviction
- [ ] Eviction weighted by compile cost, not size alone
- [ ] `--no-cache` differential runs in CI
- [ ] Cache hits recorded in each run's provenance
- [ ] Compiler version bumped on every miscompile fix

---

## Related Sheets

- [ir-contracts-and-semantic-identity.md](ir-contracts-and-semantic-identity.md) — the hash the key is built on
- [compilation-manifests-and-reproducibility.md](compilation-manifests-and-reproducibility.md) — flags and pipeline hash
- [conformance-testing.md](conformance-testing.md) — the gate that precedes insertion
- [cost-estimation-and-compilation-budgets.md](cost-estimation-and-compilation-budgets.md) — cost-weighted eviction and staleness
- [compiler-architecture-for-tensor-programs.md](compiler-architecture-for-tensor-programs.md) — gate-before-cache ordering
