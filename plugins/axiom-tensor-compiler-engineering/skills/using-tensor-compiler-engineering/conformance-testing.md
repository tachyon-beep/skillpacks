---
name: conformance-testing
description: Use when proving a compiled tensor artifact still implements the semantics of its source IR — reference-versus-compiled execution on declared inputs, gradient conformance (gradcheck and cotangent comparison), identity and zero-influence preservation, cross-device and cross-layout agreement — and when placing that gate independently of the compiler that produced the artifact.
---

# Conformance Testing

This is the core sheet of the pack. Everything else produces an artifact; this is what makes the artifact believable.

## When to Use

- Building the gate for a lowering pipeline. Build it **before** the pipeline — a gate written afterwards is written against the compiler's behaviour and will ratify its bugs.
- "Compiled output differs from eager" and you need to know whether the test or the artifact is wrong.
- Gradients disagree after compilation.
- Adding `torch.compile` to something that matters.
- Auditing a compiler someone else built.

---

## Core Principle

**Conformance is five independent questions, asked by something that is not the compiler. Answering four of them is a pass rate of zero, because the one you skipped is the one that fails in production.**

The five:

1. **Forward agreement** — same declared inputs, reference and compiled outputs agree inside the declared budget.
2. **Gradient conformance** — the *backward* agrees. A separate program, separately capable of being wrong.
3. **Structural preservation** — identity/zero-influence subgraphs still behave as identities; no semantic node invented or lost.
4. **Cross-device agreement** — every device the artifact claims to support produces the same answer inside budget.
5. **Cross-layout agreement** — contiguous, channels-last, non-contiguous and strided inputs all agree.

### The failure this prevents

Forward-only conformance is the single most common hole in this domain, and it is invisible by construction. Here is it failing, verified on torch 2.9:

```python
import torch, torch.nn as nn
torch.manual_seed(0)

class BadGelu(torch.autograd.Function):
    """Correct forward. Wrong backward. Ships silently."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.nn.functional.gelu(x)      # exactly right
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.sigmoid(x)             # wrong derivative

ref_lin = nn.Linear(8, 8)
reference = lambda x: torch.nn.functional.gelu(ref_lin(x))
compiled  = lambda x: BadGelu.apply(ref_lin(x))

x = torch.randn(4, 8)
print(torch.allclose(reference(x), compiled(x), rtol=1e-6, atol=1e-7))   # True
```

Forward agreement is *exact*. Every forward test passes at any tolerance. The artifact trains to a different model than the one that was approved, and the divergence is attributed to seed variance for as long as anyone is willing to believe that.

This is not a contrived shape. It is what a hand-written decomposition, a fused kernel with a manually-derived backward, or a custom autograd function looks like when the backward derivation has an error — and backward derivations are where the errors are, because forwards get eyeballed and backwards get trusted.

---

## Check 1 & 2: Forward and Gradient Conformance

Gradient conformance means comparing the *backward pass of the reference* against the *backward pass of the artifact*, driven by the same upstream cotangent. This is distinct from `gradcheck`, which compares an analytic backward against finite differences. You need both, and they catch different things.

```python
import torch
from dataclasses import dataclass, field

@dataclass
class ConformanceReport:
    findings: list = field(default_factory=list)   # (severity, check, detail)
    @property
    def passed(self) -> bool:
        return not any(s == "CRITICAL" for s, _, _ in self.findings)
    def add(self, severity, check, detail):
        self.findings.append((severity, check, detail))

def forward_and_gradient_conformance(ref_fn, cmp_fn, inputs, ref_params, cmp_params,
                                     rtol, atol, report, seed=0):
    """Checks 1 and 2. ref_fn/cmp_fn must be callables over the SAME inputs."""
    out_r, out_c = ref_fn(*inputs), cmp_fn(*inputs)

    if out_r.shape != out_c.shape:
        report.add("CRITICAL", "forward", f"shape {out_r.shape} vs {out_c.shape}")
        return report
    if not torch.allclose(out_r, out_c, rtol=rtol, atol=atol):
        report.add("CRITICAL", "forward",
                   f"max abs diff {(out_r - out_c).abs().max().item():.3e}")

    # One fixed cotangent, shared by both — a random one per side compares nothing.
    torch.manual_seed(seed)
    cotangent = torch.randn_like(out_r)
    g_r = torch.autograd.grad(out_r, ref_params, grad_outputs=cotangent, allow_unused=True)
    g_c = torch.autograd.grad(out_c, cmp_params, grad_outputs=cotangent, allow_unused=True)

    for i, (a, b) in enumerate(zip(g_r, g_c)):
        if (a is None) != (b is None):
            report.add("CRITICAL", "gradient",
                       f"param[{i}]: gradient flows on one side only — "
                       "a pass changed which parameters are trainable")
        elif a is not None and not torch.allclose(a, b, rtol=rtol, atol=atol):
            report.add("CRITICAL", "gradient",
                       f"param[{i}]: max abs diff {(a - b).abs().max().item():.3e}")
    return report
```

Run against the `BadGelu` artifact above (with the seed shown), this reports `param[0]: max abs diff 2.078e+00` and `param[1]: 9.546e-01` while the forward check passes clean. That gap — exact forward, order-1.0 backward error — is the signature of a wrong decomposition backward.

Three details that decide whether this harness works:

- **One cotangent, shared.** Sampling `randn_like` separately for each side compares two different derivatives and always fails. This is the most common reason teams conclude "gradient conformance is too noisy to use".
- **`allow_unused=True` plus the `None`-asymmetry check.** A pass that accidentally detaches a parameter produces `None` on one side. Without the explicit check, `zip` silently compares nothing and the test passes.
- **Same parameter objects where possible.** If the artifact holds copies, compare by name, and verify the copies are bit-identical to the originals first — otherwise you are measuring initialisation noise.

### `gradcheck` — and why it must be float64

`gradcheck` compares the analytic gradient against a finite-difference estimate. Finite differences need the perturbation to survive rounding, and in float32 it does not:

```python
f = lambda x: torch.nn.functional.gelu(x).sum()

x32 = torch.randn(6, dtype=torch.float32, requires_grad=True)
torch.autograd.gradcheck(f, (x32,))    # raises GradcheckError — essentially always

x64 = torch.randn(6, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(f, (x64,))    # True
```

torch even warns you: *"Input #0 requires gradient and is not a double precision floating point... This check will likely fail."*

The operational consequence matters more than the fact. A team that runs `gradcheck` in float32 sees constant failures, concludes the tool is flaky, and disables it. **"gradcheck is flaky" is not an observation about gradcheck; it is a diagnosis of the harness.** Cast a float64 copy of the graph for gradcheck, and use the cotangent comparison above for the real dtype. They answer different questions:

| Tool | Question | dtype |
|------|----------|-------|
| `gradcheck` | Is the analytic backward *mathematically* right? | float64, always |
| Cotangent comparison | Does the artifact's backward match the reference's? | the deployed dtype |
| `gradgradcheck` | Is the double-backward right? | float64; only if you use it |

---

## Check 3: Structural Preservation

Numerical agreement is necessary and not sufficient — `ir-contracts-and-semantic-identity.md` shows an inserted `relu` that is bit-identical and still a contract violation. Two structural checks belong in the gate:

```python
def structural_conformance(spec, artifact, report):
    # (a) identity carried, not recomputed
    if artifact.semantic_hash != spec.semantic_hash:
        report.add("CRITICAL", "identity",
                   "artifact.semantic_hash != spec.semantic_hash — the artifact does not "
                   "claim to implement this spec")

    # (b) topology diff reconciles against the manifest
    for violation in contract_violations(spec.graph_ir, artifact.source_graph):
        if not manifest_explains(artifact.pass_manifest, violation):
            report.add("CRITICAL", "structure",
                       f"unexplained topology change: {violation}")
```

**Zero-influence preservation** is the specific case worth calling out. Many pipelines introduce subgraphs that are exact identities at some parameter setting — a residual branch scaled by α=0, a gated path that is closed at initialisation, a LoRA-style adapter starting at zero. The compiler is entitled to notice these are dead *right now*, and is not entitled to remove them:

```python
def zero_influence_preserved(artifact_module, inputs, gate_params, atol) -> bool:
    """With every influence gate at zero, the artifact must be an exact identity
    on the residual path — and the gated subgraph must still EXIST."""
    with torch.no_grad():
        for p in gate_params:
            p.zero_()
        out = artifact_module(*inputs)
    identity_ok = torch.allclose(out, inputs[0], rtol=0, atol=atol)
    # Existence, not just behaviour: behaviour at zero is identical either way.
    still_present = any("adapter" in n for n, _ in artifact_module.named_modules())
    return identity_ok and still_present
```

The `still_present` clause is the point. At α=0 a removed branch and a present branch behave identically, so a behavioural check alone cannot distinguish them. You must assert the structure.

---

## Check 4 & 5: Cross-Device and Cross-Layout Agreement

Both exist because passes are frequently conditional on device or memory format, and a conditional pass tested on one branch is untested.

```python
def cross_device_conformance(build_artifact, spec, devices, rtol, atol, report):
    inputs_cpu = spec.make_declared_inputs(device="cpu")
    baseline = build_artifact(spec, device="cpu")(*inputs_cpu)
    for dev in devices:
        if dev == "cpu":
            continue
        out = build_artifact(spec, device=dev)(*[i.to(dev) for i in inputs_cpu]).cpu()
        if not torch.allclose(baseline, out, rtol=rtol, atol=atol):
            report.add("CRITICAL", "cross-device",
                       f"cpu vs {dev}: {(baseline - out).abs().max().item():.3e}")

def cross_layout_conformance(module, x, rtol, atol, report):
    """Four layouts. The non-contiguous case is the one that finds stride bugs."""
    with torch.no_grad():
        base = module(x.contiguous())
        variants = {
            "channels_last": x.to(memory_format=torch.channels_last),
            "non_contiguous": torch.zeros(*x.shape[:-1], x.shape[-1] * 2,
                                          dtype=x.dtype, device=x.device
                                          )[..., :x.shape[-1]].copy_(x),
            # `.contiguous()` between the transposes is load-bearing. Without it,
            # x.transpose(-1,-2).transpose(-1,-2) returns the same storage, the
            # same strides and is_contiguous()==True — it IS x, and the check is
            # vacuous. Materialising in transposed order gives a genuinely
            # stride-permuted view (inner-dim stride != 1) of the same values.
            "transposed_strided": x.transpose(-1, -2).contiguous().transpose(-1, -2),
        }
        for name, xv in variants.items():
            out = module(xv)
            out = out.to(memory_format=torch.contiguous_format) if out.dim() == 4 else out
            if not torch.allclose(base, out, rtol=rtol, atol=atol):
                report.add("CRITICAL", "cross-layout",
                           f"{name}: {(base - out).abs().max().item():.3e}")
```

The layouts split into two classes, and knowing which is which is what makes the check usable (all figures measured on torch 2.9.1):

- **Same-kernel layouts** — strided views of an unchanged memory format: row-padded slices, and transposed-then-materialised views whose inner-dimension stride is not 1 — must agree **bit-exactly**. Measured: `0.000e+00` against the contiguous baseline in every configuration tried (CPU and CUDA, narrow and wide convs). Any nonzero difference here is a real finding with no noise floor to hide in. Check that each variant you list is actually a distinct layout: a bare transpose round-trip (`x.transpose(-1,-2).transpose(-1,-2)`) is not — it returns `x`'s own storage, strides and contiguity, so it reports `0.000e+00` no matter how broken the compiler is.
- **Kernel-changing layouts** — `channels_last`, which legitimately dispatches different convolution algorithms — are exact only when the backend happens to pick the same kernels. Measured on the same conv stack: `0.000e+00` on CPU at 8 channels, but `9.5e-07` on CPU at 64 channels (different oneDNN path), `4.1e-05` on CUDA with the default `cudnn.allow_tf32=True`, and `0.000e+00` on CUDA with it off. A channels-last difference inside the reassociation budget is a kernel-selection fact — record which kernel ran in the manifest and hold the diff to the budget; anything beyond budget, or any nonzero *same-kernel* difference, is a bug. If your pipeline pins kernel choice per layout, demand exactness everywhere and say so in the contract.

Cross-device agreement is never exact — different devices run different kernels by construction. Expect order `1e-07` for a small float32 conv stack CPU-versus-CUDA (measured `3.6e-07`), and orders more when TF32 flags differ (`4.5e-04` at 64 channels with `cudnn.allow_tf32=True` against `4.8e-07` with it off) — which is why the flags are asserted before any of this runs.

One configuration trap that will otherwise be misread as a miscompile. On torch 2.9, TF32 defaults are *asymmetric*:

```python
torch.backends.cuda.matmul.allow_tf32   # False
torch.backends.cudnn.allow_tf32         # True   ← convolutions use TF32 by default
```

So a CPU-versus-CUDA comparison of a conv model compares float32 against a 10-bit-mantissa format. Assert both flags against `NumericalContract.allow_tf32` at the top of the gate; otherwise you will spend a day bisecting a pass that is innocent.

---

## Input Selection: The Part Everyone Gets Wrong

Conformance inputs must come from the IR's declared input/output contract, **never** from a helper owned by the compiler. A compiler-owned helper encodes the compiler author's assumptions — usually `torch.randn`, contiguous, moderate magnitude, at initialisation — and those are exactly the conditions under which the interesting bugs hide.

The declared input set should include, at minimum:

| Case | Finds |
|------|-------|
| `randn` at declared shapes | Baseline sanity |
| Cancellation-heavy (values near ±equal, sums near zero) | Reassociation error from fusion |
| Large magnitude (near dtype max) | Overflow in a decomposition (`exp` in a hand-rolled sigmoid) |
| Small magnitude / denormal | Underflow, flush-to-zero differences |
| Exact zeros and ones | Branchy kernels, division guards |
| Non-contiguous / transposed | Stride bugs in layout passes |
| Post-training parameter state, not just init | Subgraphs that are zero-influence at init |
| Batch size 1 and the declared maximum | Shape specialisation, dynamic-shape guards |

The last two are the ones most often missing. Testing only at initialisation means only the subgraph active at initialisation is tested — see the RED scenario in `ir-contracts-and-semantic-identity.md` for what that costs.

---

## Executable Decision Procedure: The Gate

```python
def run_conformance(spec, artifact, build_artifact) -> ConformanceReport:
    """The gate. Imports nothing from the compiler.

    Sequenced cheapest-and-most-diagnostic first: a configuration mismatch or an
    identity mismatch makes every downstream number meaningless, so stop there
    rather than reporting fifty derived failures.
    """
    report = ConformanceReport()
    nc = spec.numerical_contract

    # 0. Configuration — before anything numeric
    assert_backend_flags_match(nc, report)          # tf32 (matmul AND cudnn), determinism
    if not report.passed:
        return report

    # 1. Identity and structure — cheap, and gates the rest
    structural_conformance(spec, artifact, report)
    if not report.passed:
        return report

    # 2. Forward + gradient, over the DECLARED input set
    for case in spec.declared_inputs():                     # not a compiler helper
        rtol, atol = nc.budget_for(case.op_profile)         # derived, per-op
        forward_and_gradient_conformance(
            spec.reference_fn, artifact.module, case.inputs,
            spec.reference_params, artifact.params, rtol, atol, report)

    # 3. Mathematical correctness of the backward, in float64
    gradcheck_float64(spec, report)

    # 4. Structural invariants that behaviour cannot reveal
    zero_influence_conformance(spec, artifact, report)

    # 5. Every claimed device, every declared layout
    cross_device_conformance(build_artifact, spec, nc.devices, *nc.cross_device_budget, report)
    cross_layout_conformance(artifact.module, spec.canonical_input(), *nc.layout_budget, report)

    return report
```

And the rule that makes the gate load-bearing rather than advisory:

```python
def publish(spec, artifact):
    report = run_conformance(spec, artifact, build_artifact)
    if not report.passed:
        quarantine(artifact, report)      # NOT cached, NOT reachable, retained for triage
        raise ConformanceFailure(report)
    artifact_cache.put(artifact, report)  # the report is stored WITH the artifact
```

Storing the report with the artifact matters: six months later, "was this artifact ever conformance-checked, and against which contract version?" must be answerable from the artifact alone.

---

## RED → GREEN Scenario

**RED.** A team adds `torch.compile` to a training pipeline. Their conformance test:

```python
def test_compiled_matches_eager():
    m = build_model().eval()
    c = torch.compile(m)
    x = torch.randn(8, 128)
    assert torch.allclose(m(x), c(x), atol=1e-4)
```

It passes, and stays passing for four months. Then a validation run diverges from the eager baseline after ~2000 steps. The investigation blames data ordering, then the optimiser, then the seed. It is none of those: a decomposition inside the compiled region has a backward that is correct only for positive inputs, and the model's activations become negative once it is trained past its initial regime.

Every hole in the pack's list is present in five lines:

- `.eval()` — the artifact is never checked in the mode it is used in
- no `requires_grad` anywhere — the backward is literally never executed
- `randn` only — no cancellation, no large magnitude, no negatives at scale
- untrained parameters — only the initialisation-time subgraph is exercised
- one device, one layout, and `atol=1e-4` with no derivation

**GREEN.**

```python
def test_conformance():
    spec = load_canonical_spec("model.ir")           # carries semantic hash + numerics
    artifact = compile_artifact(spec, device="cuda", dtype="float32")
    report = run_conformance(spec, artifact, build_artifact)
    assert report.passed, format_findings(report.findings)
```

with `spec.declared_inputs()` supplying train-mode and eval-mode cases, `requires_grad=True` inputs, post-training parameter state loaded from a checkpoint, cancellation-heavy and large-magnitude cases, and both devices. The wrong backward is caught by the cotangent comparison on the first run after it is introduced — it is an order-1.0 error, not a subtle one. It was only subtle because nothing looked.

The generalisable lesson: **the bug was never hard to find. Every hole in that five-line test was a decision to not look somewhere.** Conformance design is mostly the discipline of enumerating where you are not looking.

---

## Anti-Patterns

| Pattern | Why it fails | Fix |
|---------|--------------|-----|
| Conformance inside `compile()` | Self-certification | Sibling gate — `compiler-architecture-for-tensor-programs` |
| Forward-only | Backward is a different program | Cotangent comparison, always |
| `gradcheck` in float32 | Fails always; gets disabled as "flaky" | float64 copy for gradcheck |
| Fresh `randn_like` cotangent per side | Compares two different derivatives | One shared cotangent |
| `allow_unused` without a `None`-asymmetry check | A detached parameter passes silently | Explicit asymmetry check |
| Inputs from a compiler-owned helper | Tests the compiler author's assumptions | Inputs from the IR contract |
| Only `.eval()`, only at init | Misses train-mode and post-training subgraphs | Both modes, trained state |
| One device when several are claimed | Device-conditional passes untested | Every claimed device |
| Contiguous inputs only | Stride bugs in layout passes survive | Four layouts, incl. non-contiguous |
| TF32 flags unasserted | Conv uses TF32 by default; misread as miscompile | Assert both flags first |
| Behavioural zero-influence check only | Removed and present branches behave identically at zero | Assert structure exists |
| Failed artifacts cached with a flag | Reachable by anything that ignores the flag | Quarantine; never inserted |

---

## Checklist

- [ ] The gate imports nothing from the compiler and could judge another compiler's artifact
- [ ] Backend flags (`matmul.allow_tf32`, `cudnn.allow_tf32`, determinism) asserted before any numeric check
- [ ] `semantic_hash` equality checked before numerics
- [ ] Topology diff reconciled against the manifest
- [ ] Forward compared on the declared input set, per-op budgets
- [ ] Gradient compared with a single shared cotangent
- [ ] `None`-gradient asymmetry treated as CRITICAL
- [ ] `gradcheck` runs on a float64 copy
- [ ] Zero-influence subgraphs checked for *existence*, not just behaviour
- [ ] Every claimed device exercised
- [ ] Contiguous, channels-last, non-contiguous, transposed-strided all exercised — and each verified to be a genuinely distinct layout, not a view that is secretly the baseline
- [ ] Inputs include cancellation, large magnitude, zeros, batch=1, batch=max
- [ ] Both train and eval mode
- [ ] Post-training parameter state, not only initialisation
- [ ] Failing artifacts quarantined, never cached
- [ ] The conformance report is stored with the artifact

---

## Related Sheets

- [compiler-architecture-for-tensor-programs.md](compiler-architecture-for-tensor-programs.md) — where the gate sits
- [numerical-contracts-and-tolerances.md](numerical-contracts-and-tolerances.md) — where the budgets come from
- [ir-contracts-and-semantic-identity.md](ir-contracts-and-semantic-identity.md) — the structural checks
- [miscompile-taxonomy-and-debugging.md](miscompile-taxonomy-and-debugging.md) — what to do when the gate fails
- [artifact-identity-and-caching.md](artifact-identity-and-caching.md) — gate before cache
