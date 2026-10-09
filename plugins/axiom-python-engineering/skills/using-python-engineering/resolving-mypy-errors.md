# Resolve type errors without hiding behavior

## Read the whole diagnostic

Capture the command, checker version/configuration, import resolution and relevant error code. Read the reported line with callers and definitions. Fixing a downstream annotation can conceal an upstream value/contract error.

## Repair sequence

1. Reproduce the error in the intended environment; check stub/runtime version agreement.
2. Decide whether implementation or declared contract is wrong. Trace actual values, nullability, mutability and exception paths.
3. Express the invariant with a narrower API, validation, protocol, overload or local narrowing as appropriate.
4. Run affected callers and runtime behavior checks, then rerun the configured checker.

## Dangerous shortcuts

- `cast` changes the checker's view, not the value. Validate untrusted data before assigning a trusted type.
- `assert` is not a durable external-input validator; optimized execution can remove it.
- Broad `Any`, ignored imports and file-wide disables erase coverage beyond the error.
- Changing `Optional[T]` to `T` needs evidence that all reachable paths provide a value.
- Type-ignore comments should name the narrow error code where supported and explain the tool limitation or external incompatibility. Remove stale ignores when their cause disappears.
- A local stub can falsely promise behavior. Keep it scoped, versioned and checked against the runtime dependency.

## Completion

Report remaining errors and intentional suppressions accurately. Do not require a project-wide type migration to complete a local fix; broaden only when the defect's real boundary requires it.
