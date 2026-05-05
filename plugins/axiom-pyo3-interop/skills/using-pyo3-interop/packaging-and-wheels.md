---
name: packaging-and-wheels
description: Use when packaging PyO3 wheels for distribution — cibuildwheel, abi3 wheels, manylinux / musllinux / macosx universal2, ARM64 cross-builds, `auditwheel show`, glibc symbol versioning. Wheel packaging is a matrix problem (Python × OS × architecture × libc); cibuildwheel + abi3 collapses it. Produces `11-packaging-and-wheels.md`.
---

# Packaging and Wheels: cibuildwheel, abi3 Wheels, and the Distribution Matrix

## Overview

**Core principle: a PyO3 wheel must be reproducible (any machine with the same config builds an identical wheel), portable (works on the user's machine, not just the developer's), and supply-chain-clean (signed, SBOM'd, with a verifiable origin). The default tools — `maturin build` for the wheel, cibuildwheel for the matrix, abi3 for cross-version compatibility, manylinux / musllinux / macosx_*_universal2 for platform breadth — assemble into a CI pipeline that does this with no developer-machine state. Get this pipeline right once and wheel publishing becomes a non-event; get it wrong and every release is a forensic exercise.**

This sheet is the recipe. It assumes the project has decided abi3 vs native ([`abi3-vs-native-extensions.md`](abi3-vs-native-extensions.md)) and has a working hybrid maturin layout ([`maturin-in-cargo-workspace.md`](maturin-in-cargo-workspace.md)).

## The Wheel Matrix

Even with abi3, wheels are *platform*-specific because they include compiled native code. The matrix you ship:

| Platform              | Architecture | Tag                              |
|-----------------------|--------------|----------------------------------|
| Linux (glibc)         | x86_64       | `manylinux_2_17_x86_64`          |
| Linux (glibc)         | aarch64      | `manylinux_2_17_aarch64`         |
| Linux (musl)          | x86_64       | `musllinux_1_2_x86_64`           |
| Linux (musl)          | aarch64      | `musllinux_1_2_aarch64`          |
| macOS                 | x86_64       | `macosx_11_0_x86_64`             |
| macOS                 | arm64        | `macosx_11_0_arm64`              |
| Windows               | AMD64        | `win_amd64`                      |
| Windows               | ARM64        | `win_arm64` (when needed)        |

For abi3 wheels, **one wheel per row** covers all CPython minors. For native, multiply by N CPython versions.

Common shipping subset for a small project: Linux x86_64 manylinux + macOS x86_64 + macOS arm64 + Windows AMD64. (4 abi3 wheels, or 4×N native wheels.)

## cibuildwheel: The Standard Tool

`cibuildwheel` is the canonical CI tool for building Python wheels across the matrix. It runs in CI (GitHub Actions, GitLab CI, etc.), spins up Docker containers (Linux) or runners (macOS, Windows), builds wheels for each cell, optionally tests them, and emits artifacts.

```yaml
# .github/workflows/wheels.yml
name: Build wheels

on:
  push:
    tags: ['v*']
  pull_request:
  workflow_dispatch:

jobs:
  build_wheels:
    name: Wheels on ${{ matrix.os }} (${{ matrix.arch }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            arch: x86_64
          - os: ubuntu-latest
            arch: aarch64
          - os: macos-13       # Intel
            arch: x86_64
          - os: macos-14       # Apple Silicon
            arch: arm64
          - os: windows-latest
            arch: AMD64

    steps:
      - uses: actions/checkout@v4

      - name: Set up QEMU (for aarch64 cross-build on Linux)
        if: matrix.arch == 'aarch64' && matrix.os == 'ubuntu-latest'
        uses: docker/setup-qemu-action@v3
        with:
          platforms: arm64

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable

      - name: Build wheels
        uses: pypa/cibuildwheel@v2.21
        env:
          CIBW_ARCHS: ${{ matrix.arch }}
          # abi3: one wheel per platform covers cp39+
          CIBW_BUILD: cp39-*
          CIBW_SKIP: '*-musllinux_i686 pp* *_i686'
          # Run pytest after each wheel build
          CIBW_TEST_COMMAND: 'pytest -x {project}/tests'
          CIBW_TEST_REQUIRES: pytest numpy

      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-${{ matrix.arch }}
          path: ./wheelhouse/*.whl
```

Notes:

- **`CIBW_BUILD: cp39-*`** — for an abi3 wheel built against `abi3-py39`, only build once on Python 3.9; the produced `cp39-abi3-*` wheel is forward-compatible.
- **`CIBW_TEST_COMMAND`** — cibuildwheel installs each built wheel into a clean venv and runs the tests. This catches "wheel imports but doesn't actually work" bugs.
- **`CIBW_SKIP`** — drop combinations you don't ship (32-bit, PyPy, etc.). Trim aggressively to keep CI time down.
- **QEMU** — for cross-builds (aarch64 on x86_64 host); QEMU emulates the target architecture inside Docker. Slower but works.

## `pyproject.toml` for cibuildwheel

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "mymod"
version = "0.1.0"
requires-python = ">=3.9"
description = "..."
authors = [{name = "Author", email = "author@example.com"}]
license = {text = "MIT OR Apache-2.0"}
classifiers = [
  "Programming Language :: Rust",
  "Programming Language :: Python :: Implementation :: CPython",
  "Operating System :: OS Independent",
]

[tool.maturin]
manifest-path = "crates/mymod-py/Cargo.toml"
python-source = "python"
module-name = "mymod._native"
features = ["pyo3/extension-module"]
strip = true

[tool.cibuildwheel]
build-frontend = "build"
test-requires = ["pytest", "numpy"]
test-command = "pytest {project}/tests"
# Build dependencies inside the manylinux container
before-build = "rustup default stable"

[tool.cibuildwheel.linux]
# Use the manylinux_2_17 image (PEP 600); supports glibc 2.17+
manylinux-x86_64-image = "manylinux_2_17"
manylinux-aarch64-image = "manylinux_2_17"
# musl image
musllinux-x86_64-image = "musllinux_1_2"
musllinux-aarch64-image = "musllinux_1_2"

[tool.cibuildwheel.macos]
# Universal2 builds (one wheel for x86_64 + arm64)
archs = ["universal2"]

[tool.cibuildwheel.windows]
archs = ["AMD64"]
```

## manylinux: Why You Need It

A wheel built directly on Ubuntu 22.04 will link against glibc 2.35. A user on Ubuntu 18.04 (glibc 2.27) cannot install it — the wheel's symbols are too new.

manylinux solves this:
- It's a Docker image with old glibc (manylinux_2_17 = glibc 2.17 = CentOS 7-ish).
- Building inside the manylinux container produces a wheel that uses only old enough symbols.
- The wheel is then **portable** to any Linux with glibc ≥ 2.17.
- After the build, `auditwheel` rewrites the wheel to copy in any required dynamic libraries and rename the wheel to declare its compatibility (`manylinux_2_17_x86_64`).

cibuildwheel handles all of this automatically when `manylinux-*-image` is set.

For musl-based Linux distributions (Alpine), build under musllinux. cibuildwheel handles this too with `musllinux-*-image`.

## macOS: Universal2 vs Single-Arch

`archs = ["universal2"]` builds a single wheel containing both x86_64 and arm64 binaries (a "fat" wheel). The Mach-O loader picks the right one at import. This simplifies distribution but doubles wheel size.

Alternative: build separate wheels for x86_64 and arm64. Smaller per-wheel, but two artifacts to manage. For a small `.so`, universal2 is fine; for a 50 MB `.so`, separate may be better.

## Windows: Linker and ABI

Windows wheels need:
- **MSVC toolchain** — Visual Studio Build Tools, `cl.exe`. cibuildwheel sets up the toolchain automatically.
- **Static CRT** — for `.so` portability, statically link the C runtime. Set `CARGO_BUILD_RUSTFLAGS=-Ctarget-feature=+crt-static` in the wheel CI environment.

ARM64 Windows wheels are increasingly common; build with `archs = ["AMD64", "ARM64"]` if your users need them.

## Source Distributions (sdist)

A source distribution is a `.tar.gz` containing the project source. Pip falls back to it when no wheel matches the user's platform. For a Rust project, the user must have a Rust toolchain to install from sdist — which most users don't.

```yaml
- name: Build sdist
  run: |
    pip install build
    python -m build --sdist
- uses: actions/upload-artifact@v4
  with:
    name: sdist
    path: dist/*.tar.gz
```

Ship sdist *in addition to* wheels for users on unsupported platforms. The sdist also contains the source for license / audit purposes — uploading to PyPI usually requires it.

## Publishing to PyPI

```yaml
publish:
  needs: [build_wheels, build_sdist]
  runs-on: ubuntu-latest
  if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
  permissions:
    id-token: write          # for PyPI trusted publishing
  steps:
    - uses: actions/download-artifact@v4
      with:
        path: dist
        merge-multiple: true
    - uses: pypa/gh-action-pypi-publish@release/v1
      # Trusted publishing — no API token needed.
```

PyPI's trusted publishing (OIDC) is the recommended pattern. The GitHub Actions workflow declares an identity; PyPI verifies and publishes; no secrets to manage.

## SBOM and Attestations

For supply-chain hygiene, generate an SBOM (Software Bill of Materials) per release:

```yaml
- name: Generate SBOM
  run: |
    pip install cyclonedx-bom
    cyclonedx-py environment > sbom.json

- uses: actions/attest-build-provenance@v1
  with:
    subject-path: 'dist/*.whl'
```

Cross-link to `axiom-audit-pipelines` for the broader supply-chain discipline (signed SBOMs, retention, verifiable build provenance).

## Reproducibility

Two builds on different machines should produce byte-identical wheels (or as close as feasible). The reproducibility checklist:

- **Pin the Rust toolchain** (`rust-toolchain.toml`).
- **Lock cargo deps** (`Cargo.lock` committed; check it in even though clippy lint says don't for libs).
- **Pin build deps** (maturin version, cibuildwheel version, build action versions).
- **Strip debug symbols** (`strip = true` in `[tool.maturin]`).
- **Disable timestamps in artifacts** (Rust does this by default in release builds).
- **Use a fixed `MACOSX_DEPLOYMENT_TARGET`** (`MACOSX_DEPLOYMENT_TARGET=11.0` in env).

Verify with `diffoscope`:

```bash
diffoscope wheelhouse-build1/mymod-*.whl wheelhouse-build2/mymod-*.whl
```

For a fully reproducible build, the diff is empty. In practice, small differences (timestamps in wheel metadata) are normal; the binary `.so` should match.

## Pitfalls

| Pitfall                                                  | Symptom                                                                           | Fix                                                                            |
|----------------------------------------------------------|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Wheel built on Ubuntu 22.04, user on 18.04                | `ImportError: GLIBC_2.32 not found`                                              | Use manylinux container in cibuildwheel                                        |
| auditwheel skipped                                        | Wheel filename wrong; PyPI rejects                                                | cibuildwheel runs auditwheel automatically; don't skip it                      |
| macOS wheel built without `MACOSX_DEPLOYMENT_TARGET`      | Wheel works on builder Mac, not on user's older Mac                              | Set `MACOSX_DEPLOYMENT_TARGET=11.0` in env                                     |
| Windows wheel needs MSVC runtime                          | "VCRUNTIME140.dll not found" on user machine                                     | Static CRT (`-Ctarget-feature=+crt-static`)                                     |
| abi3 not detected by cibuildwheel                         | Builds N wheels per platform when 1 should suffice                              | `CIBW_BUILD: cp39-*` (only the lowest version); abi3 wheel covers the rest    |
| Forgot sdist                                              | Users on unsupported platforms cannot install                                    | Build and publish sdist alongside wheels                                       |
| Universal2 wheel too large                                | Doubled `.so` size                                                                 | Switch to separate `x86_64` / `arm64` wheels                                   |
| Cargo.lock not committed                                  | Wheels build with different versions across CI runs                              | Commit `Cargo.lock` (yes, even for libs in this case)                          |
| Long CI build times                                        | Each PR runs full matrix; expensive                                                | Use `Swatinem/rust-cache@v2`; restrict matrix on PRs; full only on tags        |
| Random PyPI failures                                       | Token-based publishing brittle                                                     | Use trusted publishing (OIDC)                                                   |
| Wheel passes import test but crashes at first use         | Test command too lenient                                                           | Run actual pytest suite, not just `python -c "import mymod"`                   |

## Quick Reference

| Concern                          | Answer                                                                       |
|----------------------------------|------------------------------------------------------------------------------|
| Per-platform wheel               | manylinux (Linux), macosx (macOS), win_amd64 (Windows)                       |
| Cross-CPython wheel               | abi3 wheel: one wheel covers `>= floor` versions                              |
| Build matrix                      | cibuildwheel; one config in `pyproject.toml` plus a CI workflow              |
| Linux portability                  | manylinux Docker image; auditwheel rewrites the wheel                          |
| macOS targets                      | universal2 fat wheel, or separate x86_64 / arm64                              |
| Windows portability                | Static CRT (`-Ctarget-feature=+crt-static`)                                    |
| Source distribution                | `python -m build --sdist`; ship alongside wheels                              |
| PyPI publishing                    | Trusted publishing (OIDC) via `pypa/gh-action-pypi-publish`                  |
| SBOM                               | cyclonedx-bom; build attestations via GitHub Actions                          |
| Reproducibility                    | Pin toolchain, lock deps, strip symbols, fixed deployment target             |

## Cross-References

- [`abi3-vs-native-extensions.md`](abi3-vs-native-extensions.md) — what wheel matrix each option produces
- [`maturin-in-cargo-workspace.md`](maturin-in-cargo-workspace.md) — `maturin build` (what cibuildwheel calls under the hood)
- [`debugging-pyo3.md`](debugging-pyo3.md) — diagnosing wheel-install failures
- [`lifecycle-and-teardown.md`](lifecycle-and-teardown.md) — wheel-level resources (e.g., bundled binaries)
- `axiom-audit-pipelines` — wheel signing, SBOM, and supply-chain evidence
