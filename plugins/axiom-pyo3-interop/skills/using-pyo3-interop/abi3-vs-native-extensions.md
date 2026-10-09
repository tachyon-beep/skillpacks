# ABI and wheel compatibility

Choose from the consumers actually supported: Python implementations/versions/build modes, operating systems, architectures and native libraries. Stable-ABI and interpreter-specific builds trade artifact coverage against available APIs; neither is a universal default.

## Compatibility contract

- Verify the stable/limited API supported by the installed Python and PyO3 versions before selecting an `abi3` floor. A wheel tag is a compatibility claim, not proof of runtime support.
- Free-threaded compatibility and stable-ABI compatibility are distinct, version-dependent questions. Do not infer support from a predicted release or an invented feature name. Check current [PyO3 build/distribution guidance](https://pyo3.rs/main/building-and-distribution.html) and the actual toolchain.
- Audit the module's thread-safety declaration and mutable/unsafe state for the selected build; see [attachment discipline](gil-release-patterns.md).
- Native dependencies, linked symbols and platform policy can constrain portability even when the Python ABI is compatible.
- Changing ABI strategy affects wheel selection and distribution coverage. Treat it as a tested release/migration decision, not an inherently irreversible one-way door.

## Evidence

Build the declared matrix, inspect tags/linkage and install/import/use the artifacts in clean target environments. Exercise public calls and teardown, not only import. Document minimum Python, supported build modes, native prerequisites and untested combinations. A local development interpreter does not validate a published wheel matrix.

Use [wheel distribution](packaging-and-wheels.md) when implementing the packaging pipeline; verify that reference's tool APIs against the project versions.
