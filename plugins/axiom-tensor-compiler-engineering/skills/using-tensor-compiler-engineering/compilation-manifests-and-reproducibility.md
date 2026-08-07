---
name: compilation-manifests-and-reproducibility
description: Use when recording what a tensor compiler did — every pass, kernel choice, flag, version and environment pin — so that recompilation is deterministic and any behavioural difference between two artifacts can be traced to a specific decision. Read when "why does this artifact behave differently?" has no answer.
---

# Compilation Manifests and Reproducibility

## When to Use

- Two artifacts with the same source behave differently and nobody can say why.
- You need model builds that reproduce: same inputs, same toolchain, same artifact.
- Setting up the manifest schema at the start of a pipeline (do this; retrofitting is much worse).
- Auditing a compiler for manifest completeness.

---

## Core Principle

**The manifest is the answer to exactly one question: "why does this artifact behave differently from that one?" Design it backwards from that question, and it will contain the right fields. Design it as "logging", and it will contain timestamps and log levels and nothing you need.**

### The failure this prevents

Without a complete manifest, a behavioural difference between two artifacts is investigated by bisection over the entire space of possible causes: torch version, CUDA version, driver, backend flags, which passes fired, which kernels were chosen, whether a fallback was taken, what shapes were seen at compile time. That is a multi-day investigation with a low success rate, and its usual terminus is "probably nondeterminism, let's rebuild both" — which resolves nothing and destroys the evidence.

With a complete manifest it is a diff. The whole value proposition is turning a search into a comparison.

---

## Schema

Six sections. Each exists because a real class of difference traces to it.

```python
from dataclasses import dataclass, field

@dataclass
class CompilationManifest:
    # 1. WHAT was compiled — provenance
    spec_id: str
    semantic_hash: str            # carried from the spec, never recomputed
    ir_version: str

    # 2. WHAT IT WAS COMPILED FOR — target
    device_target: str            # "cuda:0" is not a target; "sm_90" is
    device_capability: str
    dtype: str
    numerical_contract_version: str

    # 3. WHAT DID IT — toolchain, pinned
    toolchain: dict = field(default_factory=dict)
    # {"torch": "2.9.1+cu128", "cuda": "12.8", "cudnn": "9.x", "triton": "...",
    #  "compiler": "ours@<git-sha>", "python": "3.12.x"}

    # 4. HOW it was configured — flags that change numerics
    backend_flags: dict = field(default_factory=dict)
    # {"matmul.allow_tf32": False, "cudnn.allow_tf32": True,
    #  "use_deterministic_algorithms": True, "cudnn.benchmark": False,
    #  "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}

    # 5. WHAT WAS DONE — the decision log
    passes: list = field(default_factory=list)      # ordered; includes no-ops
    kernels: dict = field(default_factory=dict)     # op -> {chosen, considered, rejected}
    fallbacks: list = field(default_factory=list)
    optimisations: list = field(default_factory=list)  # fusion/layout/fold/reuse entries

    # 6. WHAT IT COST — and whether anyone checked it
    compile_spend: dict = field(default_factory=dict)   # wall_s, peak_mem, cache_hit
    cost_estimate: dict = field(default_factory=dict)
    conformance_report_id: str | None = None
```

Section 4 is the one most often missing, and it is the one that explains the largest number of mysteries. On torch 2.9 the TF32 defaults are asymmetric — `torch.backends.cuda.matmul.allow_tf32` is `False` while `torch.backends.cudnn.allow_tf32` is `True` — so two machines with different environment setup produce genuinely different numerics with identical code. Without those flags recorded, that difference is invisible and gets investigated as a compiler bug.

Section 6's `conformance_report_id` is what makes "was this artifact ever checked, and against which contract version?" answerable from the artifact alone, six months later, when the CI logs have rotated.

### Capturing the environment

```python
import os, torch

def capture_environment() -> dict:
    return {
        "toolchain": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "python": __import__("platform").python_version(),
            "compiler": os.environ.get("COMPILER_GIT_SHA", "UNPINNED"),
        },
        "backend_flags": {
            "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
        },
        "device": {
            "name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "capability": (str(torch.cuda.get_device_capability())
                           if torch.cuda.is_available() else "n/a"),
        },
    }
```

`"UNPINNED"` as the default for the compiler SHA is deliberate. A manifest that says `UNPINNED` is honest and diffable; one that omits the field entirely looks complete and is not. **Prefer a recorded unknown to a missing field** — the recorded unknown shows up in a diff, the missing field does not.

`cudnn.benchmark` belongs in the list because it makes kernel selection depend on runtime autotuning, which depends on machine load. It is a legitimate setting and a direct cause of "same code, different kernels, different numerics".

---

## Deterministic Recompilation

The property worth having: **same spec + same toolchain + same flags → byte-identical manifest.** Not necessarily a byte-identical binary (codegen may embed paths or timestamps), but an identical *decision log*.

```python
import json, hashlib

MANIFEST_VOLATILE = {"compile_spend", "conformance_report_id"}   # timings vary; decisions must not

def manifest_fingerprint(m: dict) -> str:
    stable = {k: v for k, v in m.items() if k not in MANIFEST_VOLATILE}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]

def test_deterministic_compilation(spec):
    a = compile_artifact(spec, device="cuda", dtype="float32")
    b = compile_artifact(spec, device="cuda", dtype="float32")
    assert manifest_fingerprint(a.manifest) == manifest_fingerprint(b.manifest), \
        diff_manifests(a.manifest, b.manifest)
```

This test is cheap and finds a specific, common bug: **compilation decisions that depend on something not in the manifest.** Autotuning that picks whichever kernel benchmarked fastest under this run's memory pressure; a pass ordering derived from a `set` iteration; a heuristic reading wall-clock time. Each makes the fingerprint unstable, and each means "reproducible build" is false.

Two artifacts differing only in `compile_spend` are the same build. Differing in `kernels` are different builds, and the fingerprint says so.

---

## Diffing Manifests

The payoff. Structure the diff so the numerics-relevant differences surface first:

```python
PRIORITY = ["backend_flags", "toolchain", "numerical_contract_version",
            "kernels", "passes", "optimisations", "fallbacks", "device_capability"]

def diff_manifests(a: dict, b: dict) -> list[str]:
    """Ordered so the most likely explanation appears first."""
    out = []
    for section in PRIORITY:
        va, vb = a.get(section), b.get(section)
        if va == vb:
            continue
        if isinstance(va, dict) and isinstance(vb, dict):
            for k in sorted(set(va) | set(vb)):
                if va.get(k) != vb.get(k):
                    out.append(f"[{section}] {k}: {va.get(k)!r} -> {vb.get(k)!r}")
        else:
            out.append(f"[{section}] {va!r} -> {vb!r}")
    return out
```

`backend_flags` first is not cosmetic. In practice most "mysterious artifact differences" are a flag, and putting flags at the top of the diff turns a multi-day investigation into reading line one:

```
[backend_flags] cudnn.allow_tf32: True -> False
[backend_flags] cudnn.benchmark: False -> True
[toolchain] torch: '2.9.1+cu128' -> '2.9.0+cu124'
```

---

## Executable Decision Procedure: Manifest Completeness Audit

Auditing completeness means checking the manifest against the *code*, not against itself. A manifest with no gaps relative to itself is trivially complete.

```python
def audit_manifest_completeness(manifest, compiler_source_facts) -> list[str]:
    """compiler_source_facts: what the compiler CAN do, from reading its source.
       {"passes": [...], "kernel_selectable_ops": [...], "fallback_paths": [...],
        "reads_env_vars": [...], "device_conditional_passes": [...]}"""
    findings = []

    recorded = {p["pass"] for p in manifest.passes}
    for p in compiler_source_facts["passes"]:
        if p not in recorded:
            findings.append(f"CRITICAL: pass {p!r} exists in the compiler but can never "
                            "appear in a manifest — an unrecordable pass is an "
                            "unexplainable behavioural difference")

    for op in compiler_source_facts["kernel_selectable_ops"]:
        if op not in manifest.kernels:
            findings.append(f"HIGH: op {op!r} has selectable kernels but no manifest entry")
        elif "rejected" not in manifest.kernels.get(op, {}):
            findings.append(f"MEDIUM: op {op!r} records the chosen kernel but not the "
                            "rejected ones — cross-machine differences stay unexplainable")

    for fb in compiler_source_facts["fallback_paths"]:
        if fb not in {f["fallback"] for f in manifest.fallbacks} and \
           fb not in getattr(manifest, "fallbacks_not_taken", []):
            findings.append(f"HIGH: fallback {fb!r} leaves no trace whether taken or not")

    for var in compiler_source_facts["reads_env_vars"]:
        if var not in manifest.backend_flags:
            findings.append(f"CRITICAL: compiler reads env var {var!r} which is not "
                            "recorded — the build depends on unrecorded state")

    for p in compiler_source_facts["device_conditional_passes"]:
        entry = next((e for e in manifest.optimisations if e.get("kind") == p), None)
        if entry and "condition" not in entry:
            findings.append(f"HIGH: {p!r} is device-conditional but records no condition")

    if manifest.toolchain.get("compiler") == "UNPINNED":
        findings.append("CRITICAL: compiler version unpinned — the build is not reproducible")
    if manifest.conformance_report_id is None:
        findings.append("CRITICAL: no conformance report linked — cannot establish this "
                        "artifact was ever verified")
    return findings
```

The first check is the sharpest tool in this sheet: **enumerate the passes in the source, and confirm each one can appear in a manifest.** A pass with no manifest write is a permanent hole in every future investigation, and it is trivially findable by grep.

---

## RED → GREEN Scenario

**RED.** A model is validated on a research cluster and deployed to production. Production accuracy is 0.4% lower — small enough to be seed variance, large enough to matter. The manifest records the model architecture, the git SHA of the training code, and the timestamp.

The investigation runs three weeks. Weights are compared (identical). Preprocessing is compared (identical). Eventually someone notices the production image has a different base CUDA and, with it, `torch.backends.cudnn.benchmark = True` from an unrelated performance change. cuDNN autotuning selected different convolution algorithms with different accumulation orders. The 0.4% is real, explainable, and was recorded nowhere.

**GREEN.** With `backend_flags` in the manifest, the same investigation is:

```text
>>> diff_manifests(research.manifest, production.manifest)
[backend_flags] cudnn.benchmark: False -> True
[backend_flags] cudnn.allow_tf32: False -> True
[toolchain] cuda: '12.8' -> '12.4'
```

Three lines, five minutes. And the second line is the bigger finding: TF32 convolutions in production against float32 in validation. The deployed model was not the validated model.

Generalisable: **the fields you never think to record are the ones that differ between environments, precisely because nobody controls them.** Record everything the compiler reads — flags, env vars, device capability — not everything it decides. Decisions are downstream of inputs; if the inputs are recorded, the decisions are reproducible.

---

## Anti-Patterns

| Pattern | Why it fails | Fix |
|---------|--------------|-----|
| Manifest as "logging" | Contains timestamps, not decisions | Design backwards from "why do these differ?" |
| Backend flags unrecorded | The most common cause of cross-env differences | Section 4, asserted at conformance too |
| Env vars read but unrecorded | Build depends on unrecorded state | Record every var the compiler reads |
| Only the chosen kernel recorded | Cross-machine differences unexplainable | Record considered and rejected |
| Passes that write nothing | Permanent hole in every investigation | Audit source passes against manifest writes |
| No-op passes omitted | "Did not run" and "did nothing" indistinguishable | Append unconditionally |
| Missing field instead of `"UNPINNED"` | Looks complete; does not diff | Record unknowns explicitly |
| No conformance report link | Cannot establish the artifact was verified | `conformance_report_id` |
| Manifest fingerprint unstable | Autotuning/`set` ordering leaks into decisions | Deterministic-recompilation test in CI |
| Unordered diff | Numerics-relevant differences buried | Priority-ordered diff |

---

## Checklist

- [ ] All six manifest sections present
- [ ] `semantic_hash` carried from the spec
- [ ] Toolchain pinned and recorded; unknowns recorded as `"UNPINNED"`
- [ ] Both TF32 flags, `cudnn.benchmark`, determinism, and `CUBLAS_WORKSPACE_CONFIG` recorded
- [ ] Every env var the compiler reads is recorded
- [ ] Every pass in the source can appear in a manifest, including no-ops
- [ ] Kernel entries record chosen, considered, and rejected
- [ ] Fallbacks recorded whether taken or not
- [ ] Device-conditional optimisations record their condition
- [ ] `conformance_report_id` links the artifact to its verification
- [ ] Deterministic-recompilation test in CI comparing manifest fingerprints
- [ ] `diff_manifests` orders backend flags and toolchain first

---

## Related Sheets

- [artifact-identity-and-caching.md](artifact-identity-and-caching.md) — the manifest fingerprint as part of cache identity
- [fusion-and-memory-planning.md](fusion-and-memory-planning.md) — the optimisation entries
- [operator-lowering-and-kernel-selection.md](operator-lowering-and-kernel-selection.md) — kernel and fallback entries
- [conformance-testing.md](conformance-testing.md) — the report this links to
- [miscompile-taxonomy-and-debugging.md](miscompile-taxonomy-and-debugging.md) — the manifest as first stop in triage
