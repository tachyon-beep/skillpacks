# Source identity and legal target transformations

Define the approved program and the equivalence promised by compilation. Preserve an immutable source identity; a target graph need not have the same topology. Fusion, decomposition, constant folding and dead-code elimination can change target nodes/edges while preserving the required semantics.

## Identity contract

Record source representation/version, operator semantics, input/output shape/dtype/layout, parameter identity or binding, effects/aliasing, inference versus training and numerical equivalence. Existing FX/exported representations may suffice; a bespoke IR is justified only by a real consumer need.

Canonicalization must preserve distinctions relevant to the claim, including order where observable and parameter sharing. State whether parameter values are part of source identity or a separately identified binding. A structural hash alone is not a complete executable/cache identity; include behavior-affecting toolchain, device and policy inputs where the cache contract requires them.

Do not silently repair an approved source and keep its old identity. Either canonicalize under a declared source contract before approval, or record the derived target and its transformation relationship. Compiler ingest may normalize a target representation when that derivation remains explicit and semantics are checked.

## Decide whether a rewrite is legal

1. Identify whether the change edits approved source meaning or only its target implementation. Source meaning changes require the source approval/version process; target changes require an equivalence argument.
2. Check all consumers and observable effects, including aliasing, exposed intermediates, exceptions, RNG consumption and ordering. Extra/reordered target nodes are not inherently illegal; lost uses/effects are.
3. Check the numerical contract. Wider accumulation can change rounded/bitwise outputs; narrower precision can be legal only when explicitly permitted. Neither direction is automatically valid.
4. For training/autograd promises, check gradients, trainable parameters and saved-for-backward lifetimes. For inference-only artifacts, mark those obligations not applicable rather than inventing a backward requirement.
5. Validate independent conformance on distinguishing inputs and boundary states. Passing sampled cases supports the tested scope; it does not prove all-input equivalence.
6. Record transformation/build evidence at the granularity required for reproducibility or diagnosis. A manifest is useful when it has that consumer; do not require an entry per internal edit for unrelated local work.

## Worked failure: a zero-at-initialization branch

A branch multiplied by a trainable coefficient starts at zero. Folding it away may match every initial forward test yet change the training program when the coefficient becomes nonzero. Inspect trainability and effects before folding; test perturbed parameters and applicable gradients. A truly frozen, unobservable pure subgraph may be eliminated legally.

## Deliverable

A source/equivalence contract, affected target rewrite, evidence and remaining assumptions. Keep source identity distinct from build/artifact identity and explain cache invalidation. Use [conformance checks](conformance-testing.md), [numerical budgets](numerical-contracts-and-tolerances.md) and [artifact caching](artifact-identity-and-caching.md) only as applicable.
