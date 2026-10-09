# torch.compile and AOTAutograd integration checks

Use for graph breaks, recompilation, custom backends, forward/backward capture or artifact provenance in a real PyTorch project. Verify the installed version's public APIs and current [compiler documentation](https://pytorch.org/docs/stable/torch.compiler.html); compiler internals and debugging interfaces change.

## Choose the integration boundary

An existing model, FX/exported representation or compiler artifact may provide sufficient identity and inspection. A bespoke canonical IR is needed only when the application's approval/comparison contract requires capabilities unavailable from that representation. Compiled artifacts can sometimes be inspected, serialized or hashed; portability and stable identity depend on the selected API/backend/version.

Record what is actually identified: source/model, parameter state, guards/input constraints, training mode, dtype/device, backend/toolchain/options and relevant runtime flags. Hashing an incidental cache file is not automatically a semantic identity, and a model hash alone may omit build behavior.

## Diagnose execution

- Establish an eager/reference baseline and representative inputs before attributing a result to compilation.
- Inspect graph breaks, specialization guards and recompilation under the actual input distribution. Dynamic shapes, Python side effects, scalar extraction and mutable state can change capture behavior.
- Separate compilation/startup latency from steady-state runtime. Count graph breaks/recompiles only as evidence for the actual performance or behavior problem.
- AOTAutograd captures/partitions forward and backward in training-oriented flows. Check saved tensors, parameter mutation, aliasing and gradients when the artifact promises autograd; inference scope does not require a backward graph.
- Functionalization/decomposition must preserve the selected effect and numerical contract. A supported operator name alone does not prove the chosen backend handles its shapes/dtypes.
- Keep a debuggable comparison path or intermediate capture where useful for bisection. An eager target graph can help; it is not a mandatory implementation phase for every existing system.
- Check cancellation, fallback and unsupported-op behavior when consumers rely on it. Do not silently represent a fallback execution as the claimed compiled path.

## Validate and report

Compare reference and compiled behavior on declared inputs and relevant shape/layout/dtype/device boundaries. Include backward tests only for training/autograd promises, using appropriate shared cotangents and numerical budgets. Record exact configuration, observed graph/runtime behavior, executed checks and unavailable coverage. Measure end-to-end benefit rather than assuming compilation is faster.

Use [conformance](conformance-testing.md), [identity](ir-contracts-and-semantic-identity.md) and [bisection](miscompile-taxonomy-and-debugging.md) for unresolved questions. Backend implementation and cache APIs must be checked against the supported toolchain.
