---
name: miri-on-workspace-subset
description: Use when running Miri (the MIR interpreter for undefined-behaviour detection) against part of a Rust workspace — typically the unsafe-bearing crates while leaving safe-only crates on the regular toolchain. Covers the arena-crate isolation pattern, the nightly-toolchain split, the CI integration that runs Miri only where it can run (and skips where it can't, like FFI / network / disk I/O), and how to prevent Miri-incompatible code from creeping into Miri-blessed crates. Produces `07-miri-on-subset.md`.
---

# Miri on a Workspace Subset

## Why Miri Doesn't Run on the Whole Workspace

Miri is a UB detector — it interprets MIR (the IR after borrow-checking but before codegen) and catches use-after-free, data races, alignment errors, uninitialised reads, invalid-tag enum constructions, and a long tail of subtle pointer-provenance violations that LLVM happily compiles into "works on my machine" code that crashes in production.

The cost: Miri is *slow* (10–100× slower than native), runs on a *nightly* Rust toolchain (the MIR interpreter API is unstable), and **cannot interpret most of the things real programs do** — system calls, FFI, network I/O, file system, threading via `pthread`, and any code that reaches the kernel. Miri stubs some of these (`std::io` against an in-memory filesystem, simple `std::thread`), but a workspace's HTTP server crate or SQL driver crate cannot run under Miri at all.

Two conclusions follow:

1. **Miri runs on a *subset* of workspace crates** — typically the ones that contain `unsafe` or that implement core data structures (arenas, lock-free containers, bit-packing, FFI shim layers). The rest of the workspace runs on the regular toolchain.
2. **The Miri-blessed subset must be isolatable** — its crates must not transitively depend on Miri-incompatible crates, or `cargo miri test` fails before the test even runs.

`07-miri-on-subset.md` records which crates are in the Miri set, why, what makes them isolatable, and how CI runs Miri against them without slowing down the rest of the workspace.

## The Arena-Crate Pattern

The canonical Miri target is a **leaf crate that owns dangerous code** — typically an arena allocator, a lock-free queue, a slab, a custom `Vec`-like container, or an FFI shim that produces safe abstractions over unsafe primitives.

Properties of a Miri-blessed crate:

- Contains `unsafe` blocks that need verification.
- Has *no* deps that reach the OS — no `std::fs`, no `std::net`, no `tokio`, no `mio`, no FFI to system libraries, ideally no `std::process`.
- Has *minimal* deps overall — every dep is also Miri-compatible (transitively).
- Exposes a *safe* API to its consumers, so the rest of the workspace can use it without inheriting the unsafe surface.

Example shape:

```
crates/
  myapp-arena/                  Miri-blessed: pure unsafe data structure
  myapp-types/                  Miri-blessed: pure data; depends on -arena
  myapp-core/                   Miri-blessed: traits + algorithms; depends on -arena, -types
  myapp-runtime/                NOT Miri-blessed: I/O, async, system effects
  myapp-cli/                    NOT Miri-blessed: binary
```

The workspace's layered structure (per `01-`) makes this work. The lower layers contain the unsafe code and run under Miri; the upper layers add I/O and don't. Cycles between Miri-blessed and Miri-excluded crates would force the lower layers to depend on the upper, dragging Miri-incompatible deps into the Miri set, and then nothing runs.

If the workspace is feature-grouped or domain-grouped (per `01-`), the Miri-blessed set is whichever crates contain the unsafe code and have no I/O. The structure may not align with layers; the inventory in `07-` lists them explicitly.

## The Nightly Toolchain

Miri requires a nightly Rust toolchain (the MIR interpreter API is unstable). The workspace's `rust-toolchain.toml` typically pins a stable channel; a separate, opt-in nightly is needed for Miri.

Two patterns:

### Pattern A: Workspace pins stable; CI installs nightly for the Miri job only

```toml
# rust-toolchain.toml at workspace root
[toolchain]
channel = "1.83"
components = ["rustfmt", "clippy"]
```

The workspace's everyday compilation uses the pinned stable. A CI job installs nightly via `rustup` and runs `cargo +nightly miri test --workspace -p myapp-arena`:

```yaml
# .github/workflows/miri.yml (sketch)
- run: rustup toolchain install nightly --component miri
- run: cargo +nightly miri test -p myapp-arena -p myapp-types -p myapp-core
```

This is the recommended pattern. Developers run Miri locally only when they want to (`rustup toolchain install nightly --component miri` once, then `cargo +nightly miri test`), and the workspace's everyday velocity is unaffected.

### Pattern B: Workspace pins nightly throughout

A workspace whose entire codebase uses nightly features (rare; usually a low-level library project) pins nightly:

```toml
[toolchain]
channel = "nightly-2025-12-01"
components = ["miri", "rust-src"]
```

This makes `cargo miri` work without `+nightly`, but commits the entire workspace to a nightly cadence. Most workspaces should not do this; nightly Rust changes more frequently than stable, and the pin date becomes maintenance load.

Pattern A is the default. Pattern B is for projects whose unsafe code or compiler-internals dependence makes nightly mandatory anyway (e.g., custom allocators that use unstable allocator APIs).

## Stubs, Isolation, and the Things Miri Can't Do

Miri provides stubs for some std functionality:

- `std::io` reads/writes against an in-memory FS (`MIRIFLAGS=-Zmiri-disable-isolation` is required to enable real FS access; usually you don't want this).
- `std::thread::spawn` works; threads are interpreted serially with deterministic interleaving.
- `std::sync::Mutex`, `Arc`, `RwLock` work.
- `std::time::Instant::now()` returns deterministic values (per Miri's clock model).

Miri does *not* provide:

- FFI to anything — `extern "C"` calls fail.
- Real syscalls — `mmap`, `epoll`, `kqueue`, `io_uring`.
- Network I/O — `std::net::TcpStream::connect` fails.
- Real filesystems — without `-Zmiri-disable-isolation`, file ops fail.
- GPU / compute APIs.
- Custom allocators backed by `mmap`.

A Miri-blessed crate either avoids these things entirely or uses an Effects-style abstraction (per `axiom-determinism-and-replay:external-effects-substitution`) where the real implementation is swapped for an in-memory mock under test.

If the Miri-blessed crate has a transitive dep on `tokio`, `std::net`, or any FFI crate, `cargo miri test` fails at link time. The fix is structural: either the dep is genuinely needed (move the crate out of the Miri set) or it leaked in (find the dep edge and break it).

**Diagnostic for unwanted FFI deps in the Miri set:**

```bash
cargo tree -p myapp-arena | grep -E '(sys|ffi|libc|nix|winapi|tokio|mio|reqwest)'
```

Any hit is a candidate for "this should not be in the Miri set." Investigate per-edge.

## CI Integration

A workspace's Miri job runs only on the blessed subset, and only on the nightly toolchain. The shape:

```yaml
# .github/workflows/ci.yml (sketch)
jobs:
  test:
    # Regular tests on stable: every crate
    steps:
      - run: cargo test --workspace --all-targets

  miri:
    # Miri tests on nightly: blessed subset only
    steps:
      - run: rustup toolchain install nightly --component miri --component rust-src
      - run: cargo +nightly miri setup
      - run: cargo +nightly miri test -p myapp-arena -p myapp-types -p myapp-core
        env:
          MIRIFLAGS: "-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check"
```

The `MIRIFLAGS` worth defaulting on:

- `-Zmiri-strict-provenance` — strictest provenance model; catches int-to-ptr casts that are technically allowed but conceptually dubious. Recommended for new code; can be relaxed for legacy.
- `-Zmiri-symbolic-alignment-check` — symbolic (rather than concrete) alignment validation; catches alignment bugs that happen to work on the host CPU.
- `-Zmiri-tree-borrows` — alternative aliasing model (Tree Borrows) instead of the default Stacked Borrows. Some valid code is Stacked-Borrows-rejected and Tree-Borrows-accepted; if you have a soundness disagreement with Stacked Borrows, try Tree Borrows.

The Miri job is allowed to be **slow** (minutes per crate is normal) and runs on a less-frequent schedule than the regular test job — typically on every PR for the affected crates, and nightly for the full Miri set. The cadence trade-off is recorded in `07-`.

## Tests Designed for Miri

Miri runs `#[test]` functions like the regular test runner. Three patterns make tests Miri-friendly:

1. **Use `cfg(miri)` to skip Miri-incompatible tests.** A test that opens a real socket isn't going to work; mark it:

   ```rust
   #[test]
   #[cfg_attr(miri, ignore)]
   fn integration_test_with_real_socket() { /* ... */ }
   ```

   `cargo +nightly miri test` skips the ignored test; regular `cargo test` runs it.

2. **Reduce iteration counts under `cfg(miri)`.** A loop that stresses 10M items takes seconds natively and forever under Miri. Reduce:

   ```rust
   const N: usize = if cfg!(miri) { 1_000 } else { 10_000_000 };
   ```

   The reduced iteration is enough to catch UB; the full iteration runs under regular tests for performance characterisation.

3. **Write tests that exercise the unsafe code paths specifically.** Miri's value is detecting UB in `unsafe` blocks. A test that never enters an `unsafe` path adds no Miri signal. The blessed crates' tests should cover every `unsafe` block with a property test or a focused unit test.

## What `07-miri-on-subset.md` Must Contain

A complete `07-` artifact:

1. **Miri set inventory.** Every crate in the Miri-blessed set, with one row each: name, role (arena / lock-free container / FFI shim / etc.), unsafe-block count (or "audited unsafe-bearing" if zero), justification for inclusion.
2. **Exclusion list.** Every crate explicitly NOT in the Miri set, with the reason (FFI, network I/O, filesystem, async runtime, etc.).
3. **Toolchain pattern.** A or B from § "The Nightly Toolchain"; if A, where the nightly is installed (CI only, or developers also). If B, the pinned nightly date and the rotation policy.
4. **MIRIFLAGS policy.** Which flags are default; which are opt-in for specific crates.
5. **CI invocation.** The exact `cargo miri` command, the cadence (every PR / nightly / weekly), the timeout.
6. **`cfg(miri)` gating policy.** Which test patterns use `#[cfg_attr(miri, ignore)]`; the rationale for each ignore.
7. **Re-evaluation triggers.** What change forces a re-emit of `07-`. Default set: a crate added to or removed from the Miri set; a new dep on a Miri-blessed crate that brings in FFI/I/O; a Miri toolchain bump; a `MIRIFLAGS` change; a pattern change A → B or vice versa.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Running `cargo miri test --workspace` | Fails immediately because some workspace crate has FFI / network deps | Use `-p` flags to limit to the Miri set |
| Miri set crate accidentally adds `tokio` | Next CI run, Miri job fails to link | Either move the crate out of the Miri set, or remove the dep |
| `cfg(miri)` ignored tests not re-evaluated | A bug in unsafe code is silently never tested under Miri | Annual review of `#[cfg_attr(miri, ignore)]` annotations; either un-ignore (it now works under Miri) or document why permanent |
| Stable toolchain pinned to nightly "for Miri" | The whole workspace is on nightly, with the maintenance cost | Use Pattern A; pin stable for everyday work, install nightly in CI for Miri only |
| Miri test loops with large N | Miri job times out | Use `if cfg!(miri) { small } else { large }` for iteration counts |
| `MIRIFLAGS` set in `~/.cargo/config.toml` (developer machine) but not CI | Local Miri passes, CI Miri fails (or vice versa) | Pin MIRIFLAGS in CI workflow file; record in `07-` |

## Cross-References

- `01-workspace-structure.md` — the layered structure makes Miri isolation natural; feature-grouped and domain-grouped structures need explicit Miri-set inventory.
- `02-workspace-dependencies-and-resolver.md` — feature unification can pull FFI deps into the Miri set transitively; record the Miri-set guarantees in `02-`.
- `08-test-organisation-at-workspace-scope.md` — the test-organisation sheet covers test placement; this sheet covers which tests run under Miri.
- `13-workspace-anti-patterns.md` — a Miri-blessed crate that grew an FFI dep and was never noticed is a drift case.
- *Cross-pack:* `axiom-rust-engineering:unsafe-ffi-and-low-level` — the per-crate sheet on unsafe code; this sheet operationalises Miri verification of that unsafe code at workspace scope.
- *Cross-pack:* `axiom-determinism-and-replay:external-effects-substitution` — the Effects pattern that lets a normally-I/O-bound crate run under Miri via in-memory substitutes.

## The Bottom Line

**Miri runs on the subset of crates that own unsafe code and don't touch the OS. Identify the set, enforce its dep isolation, run it on a nightly toolchain in CI on its own schedule, and gate Miri-incompatible tests with `cfg(miri)`. Without this, Miri either runs on nothing (no signal), runs on everything and fails (no signal), or runs on the wrong crates (false confidence). The Miri set is a workspace-scope decision, recorded explicitly, re-evaluated at every dep change.**
