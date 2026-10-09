---
name: using-tensor-compiler-engineering
description: "Use when tensor lowering/transforms, torch.fx/compile capture, numerical tolerances, artifact identity or cache behavior need reference-versus-compiled conformance checks."
---

# Tensor compiler conformance

Preserve the declared computation while changing its execution. Distinguish approved semantic IR from the target graph: decompositions, fusion and folding are legal when the contract permits them.

## Work from the affected contract

1. State the input domain, observable outputs/state, dtype/device/layout and numerical budget. Require a bespoke canonical IR only when approved graph identity is an actual system requirement.
2. Trace capture specialization, guards, graph breaks and passes. Preserve source identity/provenance while recording transformations that explain the produced artifact.
3. Build independent reference checks from the declared domain, including boundaries, noncontiguous inputs and aliasing where supported. Test gradients for differentiable/training obligations, and only promised devices/layouts.
4. Derive tolerances from the numerical contract and measurements. A justified contract revision is explicit; widening a tolerance solely to hide a failure is not a repair.
5. Gate artifact reuse on complete cache identity and conformance evidence. Distinguish compile failure, semantic drift, conformance failure and performance regression; bisect pass/node divergence.

## Scope and completion

Deliver the affected transformation/contract, manifest or reproduction and observed conformance/cost evidence. Adopting torch.compile for an existing model does not by itself require a new compiler or IR. A review may be clean with explicit sweep coverage and unavailable checks.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| Artifact Identity and Caching | [artifact-identity-and-caching.md](artifact-identity-and-caching.md) |
| Compilation Manifests and Reproducibility | [compilation-manifests-and-reproducibility.md](compilation-manifests-and-reproducibility.md) |
| Compiler Architecture for Tensor Programs | [compiler-architecture-for-tensor-programs.md](compiler-architecture-for-tensor-programs.md) |
| Conformance Testing | [conformance-testing.md](conformance-testing.md) |
| Cost Estimation and Compilation Budgets | [cost-estimation-and-compilation-budgets.md](cost-estimation-and-compilation-budgets.md) |
| Fusion and Memory Planning | [fusion-and-memory-planning.md](fusion-and-memory-planning.md) |
| IR Contracts and Semantic Identity | [ir-contracts-and-semantic-identity.md](ir-contracts-and-semantic-identity.md) |
| Miscompile Taxonomy and Debugging | [miscompile-taxonomy-and-debugging.md](miscompile-taxonomy-and-debugging.md) |
| Numerical Contracts and Tolerances | [numerical-contracts-and-tolerances.md](numerical-contracts-and-tolerances.md) |
| Operator Lowering and Kernel Selection | [operator-lowering-and-kernel-selection.md](operator-lowering-and-kernel-selection.md) |
| torch.compile and AOTAutograd | [torch-compile-and-aotautograd.md](torch-compile-and-aotautograd.md) |
| torch.fx Capture and Transformation | [torch-fx-capture-and-transformation.md](torch-fx-capture-and-transformation.md) |

## Optional task entry points

- [diagnose-miscompile](../../commands/diagnose-miscompile.md): Diagnose miscompile for the affected tensor compiler engineering contract, with scoped source and verification evidence.
- [scaffold-tensor-compiler](../../commands/scaffold-tensor-compiler.md): Scaffold tensor compiler for the affected tensor compiler engineering contract, with scoped source and verification evidence.
- [verify-artifact-conformance](../../commands/verify-artifact-conformance.md): Verify artifact conformance for the affected tensor compiler engineering contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [compiler-conformance-reviewer](../../agents/compiler-conformance-reviewer.md), [tensor-compiler-architect](../../agents/tensor-compiler-architect.md). No fixed reviewer count is required.
