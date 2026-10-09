# Python compatibility and type contracts

## Establish compatibility first

Read `requires-python`, CI/runtime versions, dependency pins and type-checker configuration. Syntax supported on the developer machine can still break an older supported runtime. Check deferred annotation evaluation and runtime introspection before changing annotation syntax.

## Repair the contract

- Narrow optional/union values with an actual runtime condition. A cast asserts a belief; it does not validate input.
- Keep invariance and mutability in view when choosing collection types. A read-only interface often expresses a broader valid contract than a mutable concrete container.
- Distinguish protocols for structural capabilities from inheritance used for runtime behavior. Validate overload implementations against all advertised call shapes.
- Model missing, null and present values separately when the domain distinguishes them. Do not satisfy a type checker by inventing a default.
- Check callable variance, keyword-only parameters, async return types and decorators that erase signatures.
- Treat `Any` at untyped dependencies as a boundary to validate or stub; propagating it can suppress errors across the program.
- `TYPE_CHECKING` imports and forward references may behave differently under runtime annotation consumers. Exercise those consumers.
- Generated types, stubs and runtime packages can drift. Verify the version pair before weakening annotations.

## Validate

Run the configured checker on affected code and callers, then relevant runtime checks. Preserve the supported checker/runtime matrix; do not migrate tools just to clear a diagnostic. Record intentional narrow suppressions with the invariant the tool cannot express.
