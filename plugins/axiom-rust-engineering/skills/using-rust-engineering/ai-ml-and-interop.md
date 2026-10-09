# Rust ML integration boundaries

Rust library selection and Python interoperability are separate decisions. Inspect existing dependencies, hardware targets and deployment consumers before proposing a new runtime.

## Integration checks

- Establish supported dtype/device/operator coverage and who owns preprocessing, model state and artifact formats.
- Check shape/layout/contiguity and transfer/copy behavior at each tensor boundary. Shared storage needs an explicit lifetime and mutation contract.
- Preserve numerical tolerance and error behavior across eager, compiled and serialized representations.
- A Rust implementation does not establish faster end-to-end execution; measure representative workloads, crossing overhead and data movement.
- Verify the feature/toolchain/native-library matrix used to build and deploy. A local GPU build does not prove a portable artifact.
- For Python bindings, establish owned versus interpreter-bound objects, attach/detach requirements, exceptions, cancellation and interpreter shutdown. Use the dedicated PyO3 pack for production boundary design.
- Check whether concurrent/native work continues after a caller times out. Retrying may repeat effects.

Use existing ML frameworks when they meet the contract; a new framework or Rust rewrite needs a concrete benefit. Validate artifact reload and the real downstream consumer, not only a direct Rust unit call.
