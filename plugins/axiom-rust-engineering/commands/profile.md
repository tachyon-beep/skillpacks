---
description: Profile a Rust binary with cargo-flamegraph / perf / samply and interpret results
allowed-tools: ["Read", "Edit", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[binary or bench target] - what to profile"
---

# Profile Command

Profile Rust binaries to find actual performance bottlenecks. Never optimize without profiling first.

## Core Principle

Humans are terrible at guessing where code is slow, even in Rust. Profile first, then optimize the actual bottleneck.

## Prerequisites

1. **Debug symbols in the profiling profile** (do not pollute release):

   ```toml
   [profile.profiling]
   inherits = "release"
   debug = "line-tables-only"   # function names + line numbers, minimal size impact
   ```

2. **Install a profiler**:

   ```bash
   cargo install flamegraph   # Option 1
   cargo install samply       # Option 2
   # perf is the Linux kernel profiler; install via your package manager
   ```

## Process

### Option 1 — cargo-flamegraph (recommended)

```bash
cargo flamegraph --profile profiling --bin ${ARGUMENTS} -- <binary args>
```

Opens `flamegraph.svg` in your browser.

- **Wide bars** = CPU time spent in that function (including callees)
- **Vertical axis** = call depth (bottom = entry, top = leaves)
- **Color** = distinguishes functions, not significance

Read bottom-to-top; look for wide bars near the top of the stack — those are the real work.

### Option 2 — samply (Firefox Profiler UI, no elevated privileges)

```bash
samply record ./target/profiling/${ARGUMENTS}
```

Opens Firefox Profiler. Provides timeline view, per-thread breakdown, and interactive zoom. Right tool when `sudo perf` is unavailable (CI, containers, macOS without DTrace) or when you want an interactive UI.

### Option 3 — perf (Linux only)

```bash
# Build with the profiling profile first
cargo build --profile profiling

# DWARF unwinding — Rust builds often omit frame pointers, so default -g
# produces broken stacks. Requires debug info.
sudo perf record -F 99 --call-graph dwarf ./target/profiling/${ARGUMENTS}
sudo perf report
```

Go direct to perf when you need hardware counters (cache misses, branch mispredictions). Otherwise prefer Option 1 or 2.

## Analysis

Produce a written analysis naming:

1. **Top 3–5 hot paths** by cumulative time, with call counts where visible.
2. **CPU-bound vs. I/O-bound** character of each (I/O wait shows as blank/idle samples).
3. **Red flags spotted**: repeated allocations in loops, deep call stacks in hot code, unexpected vtable lookups, lock contention.

## Success Criteria

The profile step is complete when:

- A flamegraph SVG or samply recording has been generated.
- The written analysis names the top bottleneck(s) and their approximate time share.
- One concrete next action has been recommended from the routing table below (or "no action needed — bottleneck is outside the code under optimization").

## Route to the Next Step

| Bottleneck Type | Next Step |
|-----------------|-----------|
| Hot CPU loop without allocation | Verify `-C target-cpu=native` / SIMD / `#[inline]`; load `performance-and-profiling.md` |
| Allocator pressure (many small allocations) | Switch to heaptrack or DHAT; load `performance-and-profiling.md` → Memory Profiling |
| Lock contention in async | Load `async-and-concurrency.md` — audit for coarse-grained locking and held guards across `.await` |
| Deep call stacks in tight loops | Refactor to flatten; check `#[inline]` attributes |
| I/O-bound (network, file) | Verify async I/O, parallelism, buffering; load `async-and-concurrency.md` |

## Load Detailed Guidance

For comprehensive performance patterns and optimization techniques:

```
Load skill: axiom-rust-engineering:using-rust-engineering
Then read: performance-and-profiling.md
```

## Example Session

```
User: /rust-engineering:profile my-app

Claude:
1. Verified [profile.profiling] debug = "line-tables-only"
2. cargo flamegraph --profile profiling --bin my-app — generated flamegraph.svg
3. Analysis:
   - process_data(): 45% of time, ~10000 calls
   - allocate_buffer(): 25% of time (inside process_data loop)
   - Mutex::lock contention: 15% in parallel section
4. Recommended next action: allocator pressure — move buffer allocation
   outside the loop, re-profile. Load performance-and-profiling.md for
   heaptrack/DHAT if the allocation pattern needs deeper analysis.
```
