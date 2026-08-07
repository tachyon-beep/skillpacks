---
description: Run or author the full conformance suite for one compiled artifact - reference execution, gradient conformance, cross-device and cross-layout agreement, manifest completeness - and return a severity-rated report
allowed-tools: ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Skill", "Task"]
argument-hint: "<artifact-path-or-module> [--spec=<canonical-spec>] [--devices=cpu,cuda] [--author-missing]"
---

# Verify Artifact Conformance

Prove — or fail to prove — that one compiled artifact still implements the semantics of its source IR.

```
Load skill: axiom-tensor-compiler-engineering:using-tensor-compiler-engineering
```

Read `conformance-testing.md` in full before running. The checks below are its executable form; the sheet explains why each one exists and what it catches.

## Mode Selection

- **Checks exist** → run them, then audit them for the holes below. An existing suite that passes is not evidence until you have checked *what it does not look at*.
- **Checks are missing** (or `--author-missing`) → author them, run them, report.

Either way the output is the same report.

## The Five Checks

Run in this order. Each earlier check makes the later ones meaningful; running them out of order produces cascades of derived failures with the real cause buried.

### 0. Configuration (before any numeric check)

```python
assert torch.backends.cuda.matmul.allow_tf32 == contract.allow_tf32
assert torch.backends.cudnn.allow_tf32 == contract.allow_tf32     # defaults DIFFER
assert torch.are_deterministic_algorithms_enabled() == contract.deterministic_required
```

On torch 2.9, `cudnn.allow_tf32` defaults to `True` while `matmul.allow_tf32` defaults to `False`. A CPU-vs-CUDA comparison of a conv model therefore compares float32 against a 10-bit mantissa unless this is asserted. Severity **CRITICAL** if unasserted anywhere in the suite — it produces false miscompiles that consume days.

### 1. Structural

- `artifact.semantic_hash == spec.semantic_hash` (carried, not recomputed)
- Topology diff reconciles line-for-line against the manifest
- Zero-influence subgraphs still **exist**, not merely behave correctly at zero

### 2. Forward, on the declared input set

Inputs must come from the IO contract. If they come from a compiler-owned helper, that is a **HIGH** finding regardless of whether the test passes: it tests the compiler author's assumptions.

Required cases: `randn`; cancellation-heavy; large magnitude; small/denormal; exact zeros and ones; non-contiguous; batch=1 and batch=max; train mode and eval mode; post-training parameter state as well as initialisation.

### 3. Gradient conformance — the check most suites lack

Two distinct things, both required:

```python
# (a) cotangent comparison at the deployed dtype — ONE shared cotangent
g = torch.randn_like(out_ref)
gr = torch.autograd.grad(out_ref, ref_params, grad_outputs=g, allow_unused=True)
gc = torch.autograd.grad(out_cmp, cmp_params, grad_outputs=g, allow_unused=True)

# (b) gradcheck on a float64 copy — NEVER float32
torch.autograd.gradcheck(f64_fn, (x64,))
```

If the suite has no gradient check at all: **CRITICAL**. The backward is the larger program — for a two-layer model AOTAutograd captures 7 forward nodes and 18 backward nodes — and a decomposition with a correct forward and a wrong backward passes every forward test at any tolerance.

If gradcheck runs in float32: **HIGH**, and note that the team has almost certainly concluded "gradcheck is flaky" and reduced its role.

### 4. Cross-device and cross-layout

Every device the artifact *claims* to support, and four layouts: contiguous, channels-last, non-contiguous, transposed round-trip.

Layout agreement is two checks, not one. Same-kernel layouts (non-contiguous views, transposed round-trips — memory format unchanged) must be **exact**: any nonzero difference is a finding with no noise floor, the highest-signal check in the suite. Memory-format changes (`channels_last`) legally re-select kernels; hold them to the reassociation budget and require the kernel choice to appear in the manifest. On torch 2.9.1, channels-last conv differs from contiguous by up to `4.1e-05` with the default `cudnn.allow_tf32=True` — a kernel/config fact, not a miscompile.

### 5. Manifest completeness

Audit the manifest against the compiler *source*, not against itself:

- Enumerate passes in the source; confirm each can appear in a manifest
- Every env var and backend flag the compiler reads is recorded
- Kernel entries record considered and rejected, not only chosen
- Device-conditional optimisations record their condition
- `conformance_report_id` present; toolchain not `"UNPINNED"`

## Severity Rubric

| Severity | Meaning |
|----------|---------|
| **CRITICAL** | The artifact may not implement the spec, or nothing would notice if it did not — no gradient check, hash not carried, unexplained topology change, failed artifacts cached, config unasserted |
| **HIGH** | A real class of miscompile is undetectable — forward-only inputs, single device when several claimed, gradcheck in float32, compiler-owned test inputs, unpinned toolchain |
| **MEDIUM** | Detection is weaker than it should be — tolerance without a derivation, rejected kernels unrecorded, no non-contiguous case |
| **LOW** | Hygiene — missing manifest field with no numerical consequence |

## Output Format

```markdown
## Artifact Conformance Report

**Artifact**: <id> | **Spec**: <id> | **Semantic hash**: <carried? yes/no>
**Verdict**: CONFORMANT / NON-CONFORMANT / UNPROVEN

(UNPROVEN is a distinct verdict and often the right one: the checks that exist
pass, but the ones that would catch the likely failures were never written.)

### Check Results
| # | Check | Result | Severity | Evidence |
|---|-------|--------|----------|----------|
| 0 | Configuration | PASS/FAIL/ABSENT | | |
| 1 | Structural | | | |
| 2 | Forward | | | |
| 3 | Gradient | | | |
| 4 | Cross-device / layout | | | |
| 5 | Manifest | | | |

### Findings
[Severity] <what> — <evidence: file:line or measured value> — <sheet that closes it>

### What This Suite Cannot Detect
[The most valuable section. Enumerate the miscompile classes that would pass
 unnoticed given the checks that exist.]

### Critical Path
[Single highest-priority fix, with the sheet.]

### Confidence Assessment
### Risk Assessment
### Information Gaps
### Caveats
```

## Rules

- **A passing suite is not a conformant artifact.** Report what the suite cannot see, always. `UNPROVEN` is the honest verdict when the checks are thin.
- **Never widen a tolerance to make a check pass.** If a check fails, classify with `/diagnose-miscompile` first. Error more than 100× the derived budget is semantic drift, not a tolerance question.
- **Absent checks are findings**, at the same severity as failing ones. An unwritten gradient check and a failing gradient check leave you equally uninformed.
- For a broader adversarial audit of the compiler itself rather than one artifact, dispatch `agent: compiler-conformance-reviewer`.
