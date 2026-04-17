---
description: Profile a Rust binary with cargo-flamegraph / perf / samply and interpret results
allowed-tools: ["Read", "Edit", "Bash", "Glob", "Grep"]
argument-hint: "[binary or bench target] - what to profile"
---

# Profile Command

Profile Rust binaries to find actual performance bottlenecks. Never optimize without profiling first.

## Core Principle

Humans are terrible at guessing where code is slow, even in Rust. Profile first, then optimize the actual bottleneck.

## Prerequisites

Before profiling, ensure:

1. **Debug symbols in release profile**
   ```toml
   [profile.release]
   debug = true
   # Or lighter-weight line tables only:
   # debug = "line-tables-only"
   ```

2. **Install flamegraph**
   ```bash
   cargo install flamegraph
   ```

## Process

### Option 1: Cargo Flamegraph (Recommended)

```bash
cargo flamegraph --bin ${ARGUMENTS} -o flame.svg
```

Then open `flame.svg` in a browser:

- **Wide bars** = CPU time spent in that function
- **Tall stacks** = deep call chains
- **Inlined frames** = may collapse into parent function
- **Color** = different function, not significance

Read from **left to right** (time axis), **bottom to top** (call stack).

### Option 2: Samply (Firefox Profiler UI)

For interactive profiling:

```bash
# Install once
cargo install samply

# Profile a release binary
samply record ./target/release/${ARGUMENTS}
```

Opens Firefox Profiler UI automatically. Timeline view shows exactly which functions consume time per thread.

### Option 3: Perf (Linux-native)

Lower-level but powerful:

```bash
# Record with call stacks (99Hz sampling rate)
perf record -F 99 -g ./target/release/${ARGUMENTS}

# View results interactively
perf report
```

Navigate with arrow keys, `e` to expand, `+` to view full call tree.

## Interpreting Results

1. **Identify hot paths**
   - Top 5 functions by cumulative time
   - Note call frequency (1000 calls × 1ms each = 1s total)
   - Check for unexpected allocations in hot loops

2. **Distinguish CPU-bound vs I/O-bound**
   - Flamegraph shows CPU time only (idle I/O is blank)
   - Long flat sections with no growth = waiting for I/O
   - Steep growth = CPU-intensive

3. **Red flags**
   - Repeated allocations in hot loops (use arenas, reuse buffers)
   - Unexpected function calls (check for inline opportunities)
   - Deep call stacks in tight loops (flatten with refactoring)
   - Lock contention (check Mutex/RwLock usage)

## After Profiling - Route to Appropriate Skill

Based on findings, recommend:

| Bottleneck Type | Next Step |
|-----------------|-----------|
| Hot CPU loop without allocation | Check for `-C target-cpu=native`, SIMD, `#[inline]` hints |
| Allocator pressure | Load performance-and-profiling.md for heaptrack/DHAT analysis |
| Lock contention in async | Check for coarse-grained locking; see async-and-concurrency.md |
| Deep call stacks | Refactor for flatter call graph or increase inlining |
| I/O-bound (network, file) | Check for parallelism or buffering; see async-and-concurrency.md |

## Load Detailed Guidance

For comprehensive performance patterns and optimization techniques:

```
Load skill: axiom-rust-engineering:using-rust-engineering
Then read: performance-and-profiling.md
```

## Example Session

```
User: /rust-engineering:profile --bin my-app --duration 10

Claude:
1. Checking debug symbols... [profile.release] debug = true ✓
2. Running cargo flamegraph --bin my-app...
3. Generated flame.svg

Analysis:
- process_data(): 45% of time, called 10000 times
- allocate_buffer(): 25% of time (in hot path!)
- thread synchronization: 15% (Mutex::lock in parallel section)

Top recommendations:
- Move buffer allocation outside loop (reuse allocation)
- Consider lock-free queue or channel for data passing
- Profile with samply to see exact thread contention pattern
```

## Iterate

1. Profile baseline
2. Make targeted change (one variable at a time)
3. Re-profile and compare SVG visually
4. Keep flamegraphs from each iteration for comparison
5. Commit performance-related changes with measurement data in commit message

## Alternative: In-Process Instrumentation

For continuous monitoring (not one-shot profiling), consider:

- `pprof-rs` - in-process profiler, can export flamegraph
- `perf_event_open` - low-overhead sampling
- Criterion benchmarks with baseline comparisons
