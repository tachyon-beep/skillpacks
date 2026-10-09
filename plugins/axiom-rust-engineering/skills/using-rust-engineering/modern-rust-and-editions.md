# Rust edition and toolchain compatibility

Read the manifest edition, rust-version/MSRV, lock policy, target list and CI toolchains before changing syntax or dependencies. Edition and compiler version are related but distinct compatibility decisions.

## Check a proposed change

- Compile on the supported minimum toolchain when claiming MSRV compatibility. A successful latest-toolchain build does not establish it.
- Edition migration can affect name resolution, macro expansion, captures and unsafe requirements. Inspect automated edits and exercise exported macros/downstream callers.
- A dependency's newest compatible release may raise its own MSRV or change enabled features. Check the resolved graph, not just direct manifest constraints.
- Async traits, return-position opaque types and lifetime capture rules may constrain object safety, Send bounds and public API evolution.
- Nightly features need an explicit project requirement and pinned/toolchain policy. Do not introduce nightly to avoid an ordinary stable design choice.
- Cross-compilation can select different cfg branches, native dependencies and linkage. Test the actual advertised target boundary.

Use current compiler diagnostics and the official documentation matching the supported toolchain for uncertain semantics. Do not upgrade an edition or toolchain merely to clear a local warning. Record compatibility checks and unavailable targets.
