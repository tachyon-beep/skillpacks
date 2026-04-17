# Performance and Profiling

## Overview

**Core Principle:** Measure before you optimize. Every time. Without exception.

Rust gives you the tools to write fast code, but it does not automatically tell you *where* fast matters. The compiler generates excellent machine code for idiomatic Rust, which means the bottleneck in most programs is not what you think it is. Humans are consistently wrong about where programs spend time — even experienced ones, even in Rust. The discipline of profiling is the discipline of accepting that intuition is unreliable and instruments are authoritative.

This sheet covers the full measurement-to-optimization loop: establishing benchmarks with Criterion, finding CPU hot paths with cargo-flamegraph, perf, and samply, measuring allocator pressure with heaptrack and DHAT, tuning the allocator itself, and wiring up LTO, PGO, and codegen flags. Every section opens with measurement, not with the trick. The trick comes after the measurement confirms the trick is warranted.

Baseline: Rust stable 1.87, 2024 edition.

For async performance (executor tuning, task overhead, `Waker` allocation): see [async-and-concurrency.md](async-and-concurrency.md). For unsafe optimizations (manual SIMD, intrinsics, raw allocator API): see [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md).

## When to Use

Use this sheet when:

- "My Rust binary is slower than expected."
- "How do I profile a Rust program?"
- "I need to generate a flamegraph."
- "I want to benchmark this function before and after my change."
- "Memory usage is higher than I expected."
- "Should I swap to jemalloc / mimalloc?"
- "How do I enable LTO? PGO?"
- "Why is my release binary so large?"
- "Is my hot loop actually getting inlined?"

**Trigger keywords**: `criterion`, `flamegraph`, `perf`, `heaptrack`, `dhat`, `jemallocator`, `mimalloc`, `LTO`, `PGO`, `#[inline]`, `-C target-cpu=native`, `opt-level`, `cargo-bloat`, SIMD, `std::simd`, benchmark, bottleneck, allocator, profiler.

## When NOT to Use

- **Finding logical bugs**: see [testing-and-quality.md](testing-and-quality.md). Profilers tell you where time goes; they do not tell you whether the behavior is correct.
- **Async runtime overhead**: see [async-and-concurrency.md](async-and-concurrency.md) for executor selection, task size, and back-pressure.
- **Unsafe raw allocation or SIMD intrinsics**: see [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md).
- **Build time performance**: see [project-structure-and-tooling.md](project-structure-and-tooling.md) for incremental compilation, sccache, and workspace splits.
- You suspect the algorithm is wrong: fix the algorithm first, then profile.

---

## Measurement First

### The cost of premature optimization

```
Premature optimization is the root of all evil.
    — Donald Knuth

Measure. Don't tune for speed until you've measured, and even then
don't unless one part of the code overwhelms the rest.
    — Rob Pike
```

These aphorisms are not platitudes — they describe a failure mode that wastes real engineering time. The pattern is:

1. Engineer "knows" the hot path is function A.
2. Engineer rewrites A with clever tricks, making it 3× faster.
3. Profiler reveals A was 2 % of runtime; the bottleneck was always B.
4. Net improvement: negligible. Maintainability cost: real.

**Always establish a benchmark before touching performance-sensitive code.** The benchmark is your contract: it tells you the current state, it tells you whether your change helped, and it tells you how much it helped.

### Criterion: statistically rigorous microbenchmarks

`cargo bench` with `libtest` is not suitable for performance work. It produces a single timing number with no variance estimate, no outlier rejection, and no regression detection. Use Criterion.

```toml
# Cargo.toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name       = "my_bench"
harness    = false          # disable libtest; Criterion brings its own
```

```rust
// benches/my_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_process(c: &mut Criterion) {
    // ✅ CORRECT: establish a baseline before any optimization
    let data: Vec<u64> = (0..1000).collect();

    c.bench_function("process_1k", |b| {
        b.iter(|| {
            // black_box prevents the optimizer from eliding the work
            process(black_box(&data))
        })
    });
}

fn bench_process_sizes(c: &mut Criterion) {
    // ✅ CORRECT: parametric benchmark reveals scaling behavior
    let mut group = c.benchmark_group("process");
    for size in [100usize, 1_000, 10_000, 100_000] {
        let data: Vec<u64> = (0..size as u64).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, d| {
            b.iter(|| process(black_box(d)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_process, bench_process_sizes);
criterion_main!(benches);
```

```bash
# Run benchmarks
cargo bench

# Compare against a saved baseline (regression detection)
cargo bench -- --save-baseline before
# ... make your change ...
cargo bench -- --baseline before

# Open the HTML report (requires html_reports feature)
open target/criterion/report/index.html
```

Criterion runs each benchmark for a configurable warm-up duration (default 3 s), then collects samples for the measurement period (default 5 s). It fits a linear model, reports mean ± confidence interval, and flags regressions statistically. A "2% faster" claim from libtest is noise; a "2% faster" claim from Criterion with a tight confidence interval is evidence.

### Sampling vs tracing profilers

| Mode | Examples | Overhead | Fidelity | When to use |
|------|----------|----------|----------|-------------|
| Sampling | perf, cargo-flamegraph, samply | < 1 % | Statistical (misses very fast functions) | Production, long-running programs |
| Tracing / instrumentation | DHAT, heaptrack, Valgrind Massif | 5 – 100× | Exact | Development, targeted allocation analysis |

Start with a sampling profiler to find the hot path. If the hot path involves allocations, switch to a tracing profiler to count exactly how many bytes are allocated and by whom.

---

## CPU Profiling — cargo-flamegraph

cargo-flamegraph is the lowest-friction path from "I want a flamegraph" to "I have a flamegraph." It wraps `perf` (Linux) or DTrace (macOS) and produces a Brendan Gregg-style SVG in one command.

### Install

```bash
cargo install cargo-flamegraph

# Linux: perf must be available
sudo apt install linux-perf      # Debian/Ubuntu
sudo dnf install perf            # Fedora/RHEL

# macOS: flamegraph uses DTrace (no extra install, but needs SIP partially off)
```

### Enable debug symbols in release builds

The single most common flamegraph mistake is profiling without debug symbols. Without symbols, you get addresses, not function names.

```toml
# Cargo.toml — add a dedicated profiling profile
[profile.profiling]
inherits    = "release"
debug       = "line-tables-only"   # function names + line numbers, minimal size impact
# Alternatively:
# debug = true                     # full DWARF info; larger binary, slower link

[profile.release]
# ❌ WRONG: leaving debug = false (default) while generating flamegraphs
# ✅ Use the profiling profile instead of polluting release
```

```bash
# Build with the profiling profile
cargo flamegraph --profile profiling --bin my-binary -- --my-args

# If your binary reads from stdin:
cargo flamegraph --profile profiling --bin my-binary -- < input.txt

# For a bench target (produces a flamegraph of the benchmark itself):
cargo flamegraph --bench my_bench --profile profiling -- --bench process_1k
```

### Reading a flamegraph

The SVG that opens in your browser encodes the full call stack at every sample:

- **Horizontal axis**: proportion of samples (width = fraction of total CPU time). A wide bar means that function (or everything it calls) consumed a lot of CPU.
- **Vertical axis**: call depth. The bottom frame is the entry point; frames above it are callees.
- **What to look for**: wide bars near the top of the stack are the real work. Wide bars near the bottom are framework/runtime overhead you usually cannot avoid. A very flat, wide bar with nothing above it means a function is a leaf — it is doing computational work, not delegating.
- **Inline frames**: with `debug = "line-tables-only"`, inlined functions appear as distinct frames labelled with `[inlined]`. Without debug info they are merged into the caller, hiding which inlined call is hot.

```
Example flamegraph interpretation:

process_request         ████████████████████████████  (70 % of samples)
  deserialize_json      ████████████████              (40 %)
    serde_json::...     ████████████                  (30 %)
  validate_fields       ████████                      (20 %)
    regex::exec         ███████                       (18 %)   ← ACTUAL bottleneck
  write_response        █████                         (10 %)

Action: profile regex::exec. Is the Regex being compiled on every request?
```

### Cargo-flamegraph tips

```bash
# Profile for a longer duration (more samples = more statistical confidence)
CARGO_PROFILE_PROFILING_DEBUG=line-tables-only \
  cargo flamegraph --profile profiling --bin my-binary --output flame.svg -- --duration 30

# Exclude noise from the flamegraph (common in async code)
cargo flamegraph --profile profiling -- 2>&1 | head   # stderr shows perf output

# Reverse flamegraph (shows callers of a function, useful for "who allocates?")
# Use inferno-flamegraph --reverse on the perf.data file directly
```

---

## CPU Profiling — perf (Linux)

`perf` is the Linux kernel's built-in performance counter subsystem. cargo-flamegraph wraps it, but going direct gives you more control: hardware counters, cache miss analysis, branch misprediction counts.

### Basic sampling

```bash
# Build with debug symbols first
cargo build --profile profiling

# Record at 99 Hz with call graphs (-g uses frame pointers or DWARF unwinding)
sudo perf record -F 99 -g --call-graph dwarf -- ./target/profiling/my-binary

# ✅ CORRECT: use dwarf for Rust (frame-pointer-based unwinding is unreliable
# without -C force-frame-pointers=yes, which adds overhead)

# Display results interactively
sudo perf report

# Annotate specific function with source
sudo perf annotate my_hot_function

# Export to text
sudo perf report --stdio | head -100
```

### Symbol requirements

```bash
# ✅ CORRECT: ensure debug info is present
readelf -S target/profiling/my-binary | grep debug  # should show .debug_info etc.

# ❌ WRONG: profiling a stripped binary
cargo build --release                          # strip = true in release by default (1.77+)
sudo perf record ./target/release/my-binary  # → function names are '??'

# ✅ CORRECT: profile with the profiling profile
cargo build --profile profiling
sudo perf record ./target/profiling/my-binary
```

### Hardware counters

```bash
# Cache misses — often the real story for memory-bound code
sudo perf stat -e cache-misses,cache-references,instructions,cycles \
  ./target/profiling/my-binary

# Branch mispredictions — relevant if you have complex predicate code
sudo perf stat -e branch-misses,branch-instructions \
  ./target/profiling/my-binary

# Memory access patterns (Linux 4.x+)
sudo perf mem record ./target/profiling/my-binary
sudo perf mem report
```

### Docker / container considerations

```bash
# perf inside Docker requires:
# 1. --privileged flag (or at minimum --cap-add SYS_ADMIN)
# 2. Host kernel's perf_event_paranoid set permissively

# On host:
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

# Docker run:
docker run --privileged --pid=host my-image perf record ...

# ✅ Alternative without privileges: use samply (see next section)
# samply uses /proc/maps and frame-pointer unwinding in user space — no kernel caps needed
```

---

## pprof and samply

### samply — the friction-free alternative

`samply` is a sampling profiler that requires no elevated privileges and exports to Firefox Profiler. It is the right tool when:

- You cannot `sudo perf` (CI, sandboxed environments, macOS without DTrace access).
- You want an interactive timeline with zoom and call-tree navigation.
- You are debugging async code and want to see `tokio` task boundaries.

```bash
cargo install samply

# Record
samply record ./target/profiling/my-binary

# samply automatically opens Firefox Profiler in your browser
# The UI shows:
#   - Timeline of CPU usage
#   - Flame graph (click to zoom)
#   - Call tree sorted by self/total time
#   - Per-thread breakdown
```

```bash
# Profile a Criterion benchmark with samply
samply record cargo bench --bench my_bench -- --profile-time 10
# The --profile-time flag runs the bench for N seconds without Criterion's
# own timing, which samply can then profile freely
```

### pprof — protobuf export for toolchain integration

`pprof-rs` embeds a profiler in your binary and exports the `pprof` protobuf format, which is compatible with `go tool pprof`, Grafana Pyroscope, and Datadog Continuous Profiler. Use it when:

- You need continuous production profiling (attach to a running server).
- Your observability stack speaks pprof.
- You want on-demand profiling via an HTTP endpoint.

```toml
[dependencies]
pprof = { version = "0.13", features = ["flamegraph", "protobuf-codec"] }
```

```rust
use pprof::ProfilerGuard;

// ✅ CORRECT: targeted profiling of a known-slow section
let guard = ProfilerGuard::new(100).unwrap();   // 100 Hz sampling

// ... run the code you want to profile ...

if let Ok(report) = guard.report().build() {
    // Write protobuf for upload to Pyroscope / go tool pprof
    let mut file = std::fs::File::create("profile.pb").unwrap();
    report.pprof().unwrap().encode(&mut file).unwrap();

    // Or write a local flamegraph SVG
    let file = std::fs::File::create("flamegraph.svg").unwrap();
    report.flamegraph(file).unwrap();
}
```

```bash
# Analyze with go tool pprof (if installed)
go tool pprof -http=:8080 profile.pb
```

---

## Memory Profiling

**Measure before tuning the allocator.** Swapping to jemalloc does nothing if the bottleneck is allocation *rate* (too many small allocations), not allocator *efficiency*. The tools below identify *what* is allocating and *how much* before you reach for an allocator switch.

### heaptrack — system-level heap tracking (Linux)

heaptrack intercepts every `malloc`/`free` at the dynamic linker level. It records every allocation with a full stack trace and lets you replay the allocation history.

```bash
sudo apt install heaptrack heaptrack-gui    # Debian/Ubuntu

# Run your binary under heaptrack (no recompilation required)
heaptrack ./target/release/my-binary

# Analyze
heaptrack --analyze heaptrack.my-binary.*.gz

# GUI (if available)
heaptrack_gui heaptrack.my-binary.*.gz
```

heaptrack output answers:
- **Peak heap usage**: the maximum live bytes at any point.
- **Total allocations**: cumulative count and bytes — high count with low peak = churn.
- **Allocation hot paths**: which call stacks account for most bytes.
- **Temporary allocations**: allocations that live less than one GC cycle — pure overhead.

### dhat — allocation profiling in-process (Linux/macOS/Windows)

DHAT (`dhat-rs` crate) is a lightweight heap profiler that instruments the global allocator. It produces a JSON file readable by the DHAT viewer at `valgrind.org/dhat-viewer`.

```toml
[dependencies]
dhat = "0.3"
```

```rust
// ✅ CORRECT: enable dhat only in a "dhat" cargo feature to avoid overhead in release
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    run_application();

    // Profiler drops here; dhat.json written automatically
}
```

```toml
# Cargo.toml
[features]
dhat-heap = ["dhat"]
```

```bash
cargo run --features dhat-heap
# Open dhat.json in https://nnethercote.github.io/dh_view/dh_view.html
```

DHAT's distinguishing output is the **"at peak"** view: which allocations were live at the moment of maximum heap usage, ranked by bytes. This is the right starting point for reducing memory footprint.

### Valgrind Massif

Massif records heap usage over time and produces a graph you can visualize with `ms_print` or `massif-visualizer`.

```bash
# Requires Valgrind (Linux only; macOS support is incomplete)
sudo apt install valgrind

# Run (very slow — 10–50× overhead is normal)
valgrind --tool=massif --pages-as-heap=no ./target/debug/my-binary

# Visualize
ms_print massif.out.*

# Or with the Qt GUI
massif-visualizer massif.out.*
```

Use Massif when you need a *time series* of heap usage — for example, to understand whether memory grows without bound during a long-running workload. For a snapshot at peak, DHAT is faster and friendlier.

### Platform notes

| Tool | Linux | macOS | Windows |
|------|-------|-------|---------|
| heaptrack | ✅ | ❌ | ❌ |
| DHAT (dhat-rs) | ✅ | ✅ | ✅ |
| Valgrind Massif | ✅ | Partial | ❌ |
| Instruments (Allocations) | ❌ | ✅ | ❌ |
| ETW/WPA heap trace | ❌ | ❌ | ✅ |

---

## Allocator Tuning

**Only reach for a custom allocator after confirming allocation is the bottleneck.** The system allocator (glibc `ptmalloc2` on Linux) performs well for most workloads. The cases where a drop-in replacement wins are narrow but real:

- **Multi-threaded, allocation-heavy workloads**: `ptmalloc2`'s per-arena locking becomes a serialization point. `jemalloc` and `mimalloc` have per-thread magazines that reduce contention.
- **Latency-sensitive paths**: `mimalloc`'s design favors low-latency first free over raw throughput.
- **Fragmentation under load**: `jemalloc`'s size-class design reduces fragmentation for servers with long-running allocation patterns.

### jemallocator

```toml
[dependencies]
jemallocator = "0.5"
```

```rust
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() { /* ... */ }
```

### mimalloc

```toml
[dependencies]
mimalloc = { version = "0.1", default-features = false }
```

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

### Benchmarking the swap

**Do not assume the replacement is faster.** Write a criterion benchmark under realistic workload conditions (realistic thread count, realistic allocation size distribution) and compare wall time and throughput with and without the custom allocator.

```bash
# Baseline with system allocator
cargo bench --bench allocation_bench > before.txt

# Switch global_allocator, recompile, re-benchmark
cargo bench --bench allocation_bench > after.txt

diff before.txt after.txt
```

If the benchmark does not show a measurable difference, the bottleneck is not the allocator. Do not ship the dependency.

---

## Inlining and Codegen

### `#[inline]` — a hint, not a command

The Rust compiler inlines functions based on estimated code size, call depth, and optimization level. `#[inline]` marks a function as a candidate for cross-crate inlining (without it, functions in library crates may not be inlined even if the optimizer would otherwise choose to). It is a hint; LLVM may still decline.

```rust
// ✅ CORRECT: mark small, hot utility functions as inline candidates
// Only after profiling shows they are on the hot path
#[inline]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

// ✅ CORRECT: cross-crate inlining is the main use case
// Without #[inline], LLVM cannot inline this into a downstream crate
// even with LTO, unless LTO is fat (see LTO section)
#[inline]
pub fn inner_hot_path(val: u64) -> u64 {
    val.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}
```

### `#[inline(always)]` — almost always wrong

```rust
// ❌ WRONG: using #[inline(always)] as "make this fast"
#[inline(always)]
fn large_function_with_50_lines() { /* ... */ }
// Forces inlining even when it bloats call sites.
// Large inline expansions defeat the instruction cache.
// Can make hot code *slower* due to I-cache pressure.

// ✅ CORRECT: use #[inline(always)] only for 1–3 line wrappers
// where inlining is provably always the right choice
// and you have a benchmark that confirms the win
#[inline(always)]
fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}
```

### Codegen unit count

Rust by default compiles in multiple codegen units (CGUs) to parallelize compilation. More CGUs = faster build, worse optimization (fewer inlining opportunities across units).

```toml
[profile.release]
codegen-units = 1    # All code in one unit — best optimization, slower link
                     # Measure the difference; not every binary benefits

[profile.profiling]
inherits          = "release"
debug             = "line-tables-only"
codegen-units     = 1               # consistent with what you'll ship
```

### Target CPU

```bash
# ✅ CORRECT: for binaries you compile and run yourself
RUSTFLAGS="-C target-cpu=native" cargo build --release
# Uses AVX2, BMI2, etc. on your machine. Measurably faster for
# numeric/SIMD-heavy code.

# ❌ WRONG: in distributed binaries or Docker images intended for
#           heterogeneous infrastructure
# RUSTFLAGS="-C target-cpu=native" in CI that builds shipping artifacts
# → SIGILL (illegal instruction) on older CPUs that lack the extension.

# ✅ CORRECT for distributed binaries: use explicit target features
RUSTFLAGS="-C target-feature=+sse4.2,+aes" cargo build --release
# Cherry-pick only features that your minimum-supported CPU guarantees.
```

---

## LTO and PGO

Both techniques require measurement to justify. LTO has a real compile-time cost; PGO has a real workflow cost. Both improve real-world throughput for the right workload; neither is a free lunch.

### LTO — Link-Time Optimization

LTO allows the linker to inline and eliminate dead code across crate boundaries.

**Thin LTO**: fast, parallelizable, recovers most of the benefit.
**Fat LTO**: full whole-program optimization, slower link, more aggressive.

```toml
[profile.release]
lto = "thin"    # ✅ Start here. Measure. Upgrade to "fat" only if benchmark warrants.

# "fat" is equivalent to lto = true
# lto = false (default) means no cross-crate optimization
```

```bash
# Measure the impact
cargo build --release --timings    # see where link time is going

# Benchmark before and after
cargo bench -- --save-baseline without-lto
# enable lto = "thin" in Cargo.toml
cargo bench -- --baseline without-lto
```

Thin LTO is almost always the right choice for binaries distributed to customers. Fat LTO shaves a few more percent off tight numeric kernels but doubles or triples link time.

### PGO — Profile-Guided Optimization

PGO uses an instrumented run on representative input to teach LLVM which branches are hot and which functions should be inlined. The payoff is 10–20% throughput improvement for CPU-bound code with non-trivial branch structure (parsers, interpreters, servers).

```bash
cargo install cargo-pgo
```

```bash
# Step 1: build an instrumented binary
cargo pgo instrument build

# Step 2: run under representative workload to collect profiles
./target/release/my-binary --run-representative-workload
# This writes *.profraw files to the current directory

# Step 3: merge profiles
cargo pgo optimize merge

# Step 4: build the optimized binary
cargo pgo optimize build

# Benchmark the PGO binary vs the non-PGO binary
```

```toml
# cargo-pgo handles the LLVM flags, but the release profile still applies
[profile.release]
lto = "thin"     # PGO + thin LTO compose well
codegen-units = 1
```

PGO is high-value for: parsers, compilers, servers with hot request paths. It is low-value for: short-lived CLI tools, code dominated by I/O wait, code with uniform branch probabilities.

---

## Common Hot Spots

Profiler found the hot path. Here is what tends to live there in Rust and what to do about it. **Always verify with a benchmark that the fix actually helps.**

### Excessive cloning

```rust
// ❌ WRONG: cloning a large struct to satisfy the borrow checker
fn process_all(items: &[Item], config: Config) {
    for item in items {
        process_one(item, config.clone());   // Config cloned N times
    }
}

// ✅ CORRECT: pass a reference
fn process_all(items: &[Item], config: &Config) {
    for item in items {
        process_one(item, config);
    }
}
```

### Unbounded allocations in loops

```rust
// ❌ WRONG: allocating inside a hot loop
fn process_batch(records: &[Record]) -> Vec<String> {
    let mut results = Vec::new();
    for record in records {
        let mut buf = String::new();       // allocation per record
        write!(&mut buf, "{}: {}", record.id, record.value).unwrap();
        results.push(buf);
    }
    results
}

// ✅ CORRECT: reuse the buffer
fn process_batch(records: &[Record]) -> Vec<String> {
    let mut results = Vec::with_capacity(records.len());
    let mut buf = String::new();           // one allocation, reused
    for record in records {
        buf.clear();
        write!(&mut buf, "{}: {}", record.id, record.value).unwrap();
        results.push(buf.clone());
    }
    results
}
```

### `Vec<Box<T>>` vs `Vec<T>`

```rust
// ❌ WRONG: boxing unnecessarily in a hot data structure
struct Pool {
    items: Vec<Box<Item>>,   // each Item is a heap allocation; pointer chasing on access
}

// ✅ CORRECT: store values inline
struct Pool {
    items: Vec<Item>,        // items are contiguous in memory; cache-friendly
}

// ❌ Exception: when Item is very large (> ~512 bytes) or when you need
// stable addresses across insertions. Even then, consider a slab allocator.
```

### String allocation in loops

```rust
// ❌ WRONG: allocating a new String for every format operation
for event in events {
    let key = format!("{}:{}", event.namespace, event.name);  // alloc per event
    map.insert(key, event.payload);
}

// ✅ CORRECT: use a stack-allocated alternative or reuse capacity
// Option A: build the key only once per unique pair
// Option B: use a fixed-size key type
#[derive(Hash, Eq, PartialEq)]
struct EventKey {
    namespace: SmolStr,   // smolstr: stack-inline for short strings
    name: SmolStr,
}

// Option C: intern strings (useful when keys repeat frequently)
// see the `string_interner` crate
```

### SIMD opportunities and `std::simd`

Rust's portable SIMD API (`std::simd`) is available on nightly as of 2024 but has not stabilized in 1.87. Use it on nightly or via the `wide` / `packed_simd2` crates for stable.

```rust
// ✅ CORRECT: let the compiler auto-vectorize first
// Criterion will tell you if you need to go further

// Example: summing f32 slice — LLVM auto-vectorizes with -C target-feature=+avx2
fn sum_f32(data: &[f32]) -> f32 {
    data.iter().copied().sum()
}

// Only hand-write SIMD after:
// 1. Benchmark shows this function is hot
// 2. Auto-vectorization is confirmed to NOT be happening (via godbolt.org)
// 3. The SIMD version benchmarks measurably faster
```

---

## Binary Size

Binary size affects startup time, cold cache behavior, and distribution cost. Measure with `cargo-bloat` before choosing a strategy.

```bash
cargo install cargo-bloat

# Show largest functions in the binary
cargo bloat --release --crates         # by crate
cargo bloat --release -n 30            # top 30 functions

# Typical output:
#  File  .text   Size    Crate Name
#  0.5%   1.2%  88.0KiB serde_json <serde_json::...>
#  0.3%   0.6%  43.0KiB regex      regex::exec::...
```

### Size-reduction techniques

```toml
[profile.release]
# Optimize for size instead of speed
opt-level = "s"        # moderate size reduction, slight speed cost
# opt-level = "z"      # aggressive size reduction, more speed cost

# Abort on panic instead of unwinding — removes unwinder tables
panic = "abort"        # saves 50–200 KiB in many binaries

# Strip debug info (default true in release since Rust 1.77)
strip = "debuginfo"    # removes DWARF; "symbols" also strips symbol table
```

```bash
# Strip after build (Linux)
strip target/release/my-binary

# Check sizes
ls -lh target/release/my-binary
size target/release/my-binary       # shows text/data/bss breakdown
```

```toml
# For embedded / minimal targets: combine all three
[profile.release]
opt-level     = "z"
panic         = "abort"
strip         = "symbols"
codegen-units = 1
lto           = true
```

**Measure the binary size before and after.** `opt-level = "z"` can be slower than `opt-level = 3` for CPU-bound workloads. Benchmark throughput alongside size.

---

## Anti-Patterns

### 1. Optimizing without a baseline benchmark

**Wrong:** "I rewrote the hash map lookup to use an open-addressing scheme — it must be faster."

**Why wrong:** Without a before/after Criterion benchmark under realistic load, you cannot know if the change helped, hurt, or was neutral. The optimizer may have already solved what you are manually solving. You may have introduced a regression.

**The fix:** `cargo bench -- --save-baseline before`, make the change, `cargo bench -- --baseline before`. Require measured improvement before merging performance changes.

---

### 2. `#[inline(always)]` as a "make it fast" incantation

**Wrong:**

```rust
#[inline(always)]
fn do_lots_of_work(data: &[u8]) -> Vec<u8> {
    // 80 lines of code
}
```

**Why wrong:** Forcing inline of a large function bloats every call site. If the function appears in multiple hot paths, the increased binary size defeats the instruction cache. LLVM's inlining heuristics are calibrated; overriding them without measurement almost always hurts.

**The fix:** Remove `#[inline(always)]`. Use `#[inline]` only for small, cross-crate utility functions where the cross-crate inlining semantics actually matter. Measure before and after with a flamegraph to confirm whether inlining helped.

---

### 3. Swapping allocators before confirming allocation is the bottleneck

**Wrong:** "Our server is slow; jemalloc should help."

**Why wrong:** The allocator is one component. If the bottleneck is CPU computation, regex evaluation, or I/O wait, jemalloc changes nothing. Swapping the allocator adds a C dependency, a `#[global_allocator]` that applies globally (including to tests and benchmarks), and a maintenance burden.

**The fix:** Profile with heaptrack or DHAT. Confirm that allocator contention or fragmentation shows up as the bottleneck (heaptrack's "temporary allocations" count is the tell). Then benchmark the swap.

---

### 4. Using `cargo bench` (libtest) without Criterion for performance decisions

**Wrong:**

```rust
#[bench]
fn bench_process(b: &mut Bencher) {
    b.iter(|| process(&data));
}
// cargo bench reports: 1,234 ns/iter (+/- 456)
```

**Why wrong:** libtest produces a single point estimate with a noise figure that has no statistical meaning. The `+/- 456` is not a confidence interval; it is the range of measured samples with no outlier rejection. Two runs can differ by 20% due to OS scheduling noise. You cannot reliably detect a 5% regression.

**The fix:** Use Criterion. It runs proper warm-up, collects enough samples for a Student's t-test, reports a 95% confidence interval, and detects regressions against a saved baseline. `harness = false` in `[[bench]]` and a 3-line Criterion setup replaces the entire libtest bench infrastructure.

---

### 5. Leaving `debug = true` in the release profile

**Wrong:**

```toml
[profile.release]
debug = true        # "I'll remove this after I'm done debugging..."
```

**Why wrong:** Full DWARF debug info (`debug = true`) can double or triple binary size. It bloats the binary that ships to customers and slows down the linker in CI. The `debug = true` setting is also easy to forget; it has caused production binaries to ship with 80 MB of debug info.

**The fix:** Use a separate `[profile.profiling]` profile (inheriting from release, with `debug = "line-tables-only"`) for profiling work. The release profile stays clean.

---

### 6. `-C target-cpu=native` in distributed binaries

**Wrong:**

```bash
# CI build script
export RUSTFLAGS="-C target-cpu=native"
cargo build --release
# Ships the binary to customers
```

**Why wrong:** `-C target-cpu=native` compiles for the exact CPU in the CI machine (e.g., an AWS instance with AVX-512). The binary will crash with `SIGILL` (illegal instruction) on any CPU that does not support those extensions — including older developer laptops and many cloud instance types.

**The fix:** Use `-C target-cpu=native` only in binaries you build and run yourself (local benchmarking). For distribution, either target the architecture baseline (`x86-64-v2`, `x86-64-v3`) or explicitly list the feature flags you know your minimum supported CPU provides.

```bash
# ✅ CORRECT: explicit known-safe features
RUSTFLAGS="-C target-feature=+sse4.2" cargo build --release

# ✅ CORRECT: x86-64-v3 baseline (Haswell+, 2013+)
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
```

---

## Checklist

Before claiming a performance improvement is ready to ship:

- [ ] Criterion benchmark exists for the code path being optimized.
- [ ] Baseline saved before the change (`cargo bench -- --save-baseline before`).
- [ ] After the change, `cargo bench -- --baseline before` shows a statistically significant improvement.
- [ ] Flamegraph or perf report confirms the hot path is what you think it is.
- [ ] If memory was the concern: heaptrack or DHAT report was reviewed before and after.
- [ ] If allocator was swapped: benchmark confirms the swap helps under realistic thread count.
- [ ] `#[inline(always)]` is not used without a measured reason.
- [ ] Release profile does not have `debug = true`.
- [ ] CI/distribution build does not use `-C target-cpu=native`.
- [ ] If LTO was enabled: link time increase is acceptable in CI.
- [ ] If PGO was used: the profiling workload is representative of production.
- [ ] Binary size checked with `cargo-bloat` if size is a concern.
- [ ] Correctness tests still pass at all optimization levels.
- [ ] Change is not the "obvious" optimization — it is the *profiled* optimization.

---

## Related Skills

- [modern-rust-and-editions.md](modern-rust-and-editions.md) — edition and compiler version context for codegen flags
- [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — eliminating unnecessary clones at the type-system level
- [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — static vs dynamic dispatch trade-offs in hot paths
- [error-handling-patterns.md](error-handling-patterns.md) — zero-cost error handling; avoiding allocations in error paths
- [project-structure-and-tooling.md](project-structure-and-tooling.md) — release profile configuration, workspace layout, CI setup
- [testing-and-quality.md](testing-and-quality.md) — Criterion benchmark organization, property-based testing for performance invariants
- [systematic-delinting.md](systematic-delinting.md) — clippy lints that surface allocation and performance anti-patterns
- [async-and-concurrency.md](async-and-concurrency.md) — async profiling, executor overhead, task sizing
- [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — manual SIMD, raw allocator API, intrinsics
- [ai-ml-and-interop.md](ai-ml-and-interop.md) — tensor computation performance, BLAS/LAPACK integration, GPU dispatch
