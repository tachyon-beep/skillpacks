# Tensor compilation boundaries

Separate the approved program, its implementation strategy, the resulting artifact and the evidence that the artifact meets the promised behavior. These can be records/modules in an existing compiler; they do not require a new IR or five services.

## Responsibility map

| Boundary | Useful contract |
|---|---|
| Ingest | Validate source/version, retain identity and expose any normalization/derivation |
| Lower | Map operators to supported target behavior with declared dtype/effect/gradient obligations |
| Optimize | Preserve observables while transforming target topology, scheduling, layout and storage |
| Code generation | Record actual device/dtype/backend choices; expose unsupported behavior rather than silently changing the claim |
| Artifact | Bind executable, source/parameter identities and behavior-affecting build configuration |
| Conformance | Compare against an oracle independent of the transformation's failure mechanism |

Approved source identity remains immutable; target nodes can be fused, decomposed, folded or removed when the logical/effect/numerical contract permits it. Check all uses, aliasing and training lifetimes rather than banning topology changes.

## Independent evidence

Do not ask a transformation's own legality predicate to certify itself. Use a separately specified reference, independent fixtures and distinguishing cases where common-mode errors matter. Shared schemas, libraries or authorship are not automatically defects; identify what a shared mistake could conceal.

Declare inference/training, supported devices/layouts, determinism and tolerance. Training needs applicable forward/backward checks; inference-only scope does not. Performance and correctness failures should remain distinguishable because consumers take different actions.

## Build and diagnosis

Preserve a debuggable reference path and localize failures by comparing stages/intermediates. An eager target implementation can help before opaque fusion/codegen, but an existing compiler may provide equivalent diagnostics. Record transformations/build inputs when a reproducibility or investigation consumer needs them.

Cache identity must cover inputs that change behavior; gate changes may require re-verification of old artifacts. A cached verification flag alone is not evidence under a changed contract. Keep invalid source, unsupported backend, semantic drift and performance regression distinct where their downstream consequences differ.

Deliver the smallest architecture/change that satisfies the actual consumer, with assumptions, tested evidence and gaps. See [source identity](ir-contracts-and-semantic-identity.md), [conformance](conformance-testing.md) and [miscompile diagnosis](miscompile-taxonomy-and-debugging.md).
