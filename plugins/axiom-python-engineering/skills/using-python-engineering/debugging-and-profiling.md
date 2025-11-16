
# Debugging and Profiling

## Overview

**Core Principle:** Profile before optimizing. Humans are terrible at guessing where code is slow. Always measure before making changes.

Python debugging and profiling enables systematic problem diagnosis and performance optimization. Use debugpy/pdb for step-through debugging, cProfile for CPU profiling, memory_profiler for memory analysis. The biggest mistake: optimizing code without profiling first—you'll likely optimize the wrong thing.

## When to Use

**Use this skill when:**
- "Code is slow"
- "How to profile Python?"
- "Memory leak"
- "Debugging not working"
- "Find bottleneck"
- "Optimize performance"
- "Step through code"
- "Where is my code spending time?"

**Don't use when:**
- Setting up project (use project-structure-and-tooling)
- Already know what to optimize (but still profile to verify!)
- Algorithm selection (different skill domain)

**Symptoms triggering this skill:**
- Code runs slower than expected
- Memory usage growing over time
- Need to understand execution flow
- Performance degraded after changes


## Debugging Fundamentals

### Using debugpy with VS Code

```python
# ✅ CORRECT: debugpy for remote debugging
import debugpy

# Allow VS Code to attach
debugpy.listen(5678)
print("Waiting for debugger to attach...")
debugpy.wait_for_client()

# Your code here
def process_data(data):
    result = []
    for item in data:
        # Set breakpoint in VS Code on this line
        transformed = transform(item)
        result.append(transformed)
    return result

# VS Code launch.json configuration:
"""
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            }
        }
    ]
}
"""
```

### Using pdb (Python Debugger)

```python
# ✅ CORRECT: pdb for interactive debugging
import pdb

def buggy_function(data):
    result = []
    for i, item in enumerate(data):
        # Drop into debugger
        pdb.set_trace()  # Or: breakpoint() in Python 3.7+

        processed = item * 2
        result.append(processed)
    return result

# pdb commands:
# n (next): Execute next line
# s (step): Step into function
# c (continue): Continue execution
# p variable: Print variable
# pp variable: Pretty print variable
# l (list): Show current location in code
# w (where): Show stack trace
# q (quit): Quit debugger
```

### Conditional Breakpoints

```python
# ❌ WRONG: Breaking on every iteration
def process_items(items):
    for item in items:
        pdb.set_trace()  # Breaks 10000 times!
        process(item)

# ✅ CORRECT: Conditional breakpoint
def process_items(items):
    for i, item in enumerate(items):
        if i == 5000:  # Only break on specific iteration
            breakpoint()
        process(item)

# ✅ BETTER: Use pdb.set_trace with condition
def process_items(items):
    for item in items:
        if item.value < 0:  # Break only when problematic
            breakpoint()
        process(item)
```

### Post-Mortem Debugging

```python
# ✅ CORRECT: Debug after exception
import pdb

def main():
    try:
        # Code that might raise exception
        result = risky_operation()
    except Exception:
        # Drop into debugger at exception point
        pdb.post_mortem()

# ✅ CORRECT: Auto post-mortem for unhandled exceptions
import sys

def custom_excepthook(type, value, traceback):
    pdb.post_mortem(traceback)

sys.excepthook = custom_excepthook

# Now unhandled exceptions drop into pdb automatically
```

**Why this matters**: Breakpoints let you inspect state at exact point of failure. Conditional breakpoints avoid noise. Post-mortem debugging examines crashes.


## CPU Profiling

### cProfile for Function-Level Profiling

```python
import cProfile
import pstats

# ❌ WRONG: Guessing which function is slow
def slow_program():
    # "I think this loop is the problem..."
    for i in range(1000):
        process_data(i)

# ✅ CORRECT: Profile to find actual bottleneck
def slow_program():
    for i in range(1000):
        process_data(i)

# Profile the function
cProfile.run('slow_program()', 'profile_stats')

# Analyze results
stats = pstats.Stats('profile_stats')
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions by cumulative time

# ✅ CORRECT: Profile with context manager
from contextlib import contextmanager
import cProfile

@contextmanager
def profiled():
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()

    stats = pstats.Stats(pr)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)

# Usage
with profiled():
    slow_program()
```

### Profiling Specific Code Blocks

```python
# ✅ CORRECT: Profile specific section
import cProfile

pr = cProfile.Profile()

# Normal code
setup_data()

# Profile this section
pr.enable()
expensive_operation()
pr.disable()

# More normal code
cleanup()

# View results
pr.print_stats(sort='cumulative')
```

### Line-Level Profiling with line_profiler

```python
# Install: pip install line_profiler

# ✅ CORRECT: Line-by-line profiling
from line_profiler import LineProfiler

@profile  # Use @profile decorator
def slow_function():
    total = 0
    for i in range(10000):
        total += i ** 2
    return total

# Run with kernprof:
# kernprof -l -v script.py

# Or programmatically:
lp = LineProfiler()
lp.add_function(slow_function)
lp.enable()
slow_function()
lp.disable()
lp.print_stats()

# Output shows time spent per line:
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#     1                                           def slow_function():
#     2         1          2.0      2.0      0.0      total = 0
#     3     10001      15234.0      1.5     20.0      for i in range(10000):
#     4     10000      60123.0      6.0     80.0          total += i ** 2
#     5         1          1.0      1.0      0.0      return total
```

**Why this matters**: cProfile shows which functions are slow. line_profiler shows which lines within functions. Both essential for optimization.

### Visualizing Profiles with SnakeViz

```bash
# Install: pip install snakeviz

# Profile code
python -m cProfile -o program.prof script.py

# Visualize
snakeviz program.prof

# Opens browser with interactive visualization:
# - Sunburst chart showing call hierarchy
# - Icicle chart showing time distribution
# - Click functions to zoom in
```


## Memory Profiling

### Memory Usage with memory_profiler

```python
# Install: pip install memory_profiler

from memory_profiler import profile

# ✅ CORRECT: Track memory usage per line
@profile
def memory_hungry_function():
    # Line-by-line memory usage shown
    big_list = [i for i in range(1000000)]  # Allocates ~40MB
    big_dict = {i: i**2 for i in range(1000000)}  # Another ~40MB
    return len(big_list), len(big_dict)

# Run with:
# python -m memory_profiler script.py

# Output:
# Line #    Mem usage    Increment   Line Contents
# ================================================
#      3   38.3 MiB     38.3 MiB   @profile
#      4                             def memory_hungry_function():
#      5   45.2 MiB      6.9 MiB       big_list = [i for i in range(1000000)]
#      6   83.1 MiB     37.9 MiB       big_dict = {i: i**2 for i in range(1000000)}
#      7   83.1 MiB      0.0 MiB       return len(big_list), len(big_dict)
```

### Finding Memory Leaks

```python
# ✅ CORRECT: Detect memory leaks with tracemalloc
import tracemalloc

# Start tracing
tracemalloc.start()

# Take snapshot before
snapshot1 = tracemalloc.take_snapshot()

# Run code that might leak
problematic_function()

# Take snapshot after
snapshot2 = tracemalloc.take_snapshot()

# Compare snapshots
top_stats = snapshot2.compare_to(snapshot1, 'lineno')

print("Top 10 memory increases:")
for stat in top_stats[:10]:
    print(stat)

tracemalloc.stop()

# ✅ CORRECT: Track specific objects
import gc
import sys

def find_memory_leak():
    # Force garbage collection
    gc.collect()

    # Track objects before
    before = len(gc.get_objects())

    # Run potentially leaky code
    for _ in range(100):
        leaky_operation()

    # Force GC again
    gc.collect()

    # Track objects after
    after = len(gc.get_objects())

    if after > before:
        print(f"Potential leak: {after - before} objects not collected")

        # Find what's keeping objects alive
        for obj in gc.get_objects():
            if isinstance(obj, MyClass):  # Suspect class
                print(f"Found {type(obj)}: {sys.getrefcount(obj)} references")
                print(gc.get_referrers(obj))
```

### Profiling Memory with objgraph

```python
# Install: pip install objgraph

import objgraph

# ✅ CORRECT: Find most common objects
def analyze_memory():
    objgraph.show_most_common_types()
    # Output:
    # dict                   12453
    # function               8234
    # list                   6789
    # ...

# ✅ CORRECT: Track object growth
objgraph.show_growth()
potentially_leaky_function()
objgraph.show_growth()  # Shows objects that increased

# ✅ CORRECT: Visualize object references
import objgraph
objgraph.show_refs([my_object], filename='refs.png')
# Creates graph showing what references my_object
```

**Why this matters**: Memory leaks cause gradual performance degradation. tracemalloc and memory_profiler help find exactly where memory is allocated.


## Profiling Async Code

### Profiling Async Functions

```python
import asyncio
import cProfile
import pstats

# ❌ WRONG: cProfile doesn't work well with async
async def slow_async():
    await asyncio.sleep(1)
    await process_data()

cProfile.run('asyncio.run(slow_async())')  # Misleading results

# ✅ CORRECT: Use yappi for async profiling
# Install: pip install yappi
import yappi

async def slow_async():
    await asyncio.sleep(1)
    await process_data()

yappi.set_clock_type("wall")  # Use wall time, not CPU time
yappi.start()

asyncio.run(slow_async())

yappi.stop()

# Print stats
stats = yappi.get_func_stats()
stats.sort("totaltime", "desc")
stats.print_all()

# ✅ CORRECT: Profile coroutines specifically
stats = yappi.get_func_stats(filter_callback=lambda x: 'coroutine' in x.name)
stats.print_all()
```

### Detecting Blocking Code in Async

```python
# ✅ CORRECT: Detect event loop blocking
import asyncio
import time

class LoopMonitor:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    async def monitor(self):
        while True:
            start = time.monotonic()
            await asyncio.sleep(0.01)  # Very short sleep
            elapsed = time.monotonic() - start

            if elapsed > self.threshold:
                print(f"WARNING: Event loop blocked for {elapsed:.3f}s")

async def main():
    # Start monitor
    monitor = LoopMonitor(threshold=0.1)
    monitor_task = asyncio.create_task(monitor.monitor())

    # Run your async code
    await your_async_function()

    monitor_task.cancel()

# ✅ CORRECT: Use asyncio debug mode
asyncio.run(main(), debug=True)
# Warns about slow callbacks (>100ms)
```


## Performance Optimization Strategies

### Optimization Workflow

```python
# ✅ CORRECT: Systematic optimization approach

# 1. Profile to find bottleneck
import cProfile
cProfile.run('main()', 'profile_stats')

# 2. Analyze results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(10)  # Focus on top 10

# 3. Identify specific slow function
def slow_function(data):
    # Original implementation
    result = []
    for item in data:
        if is_valid(item):
            result.append(transform(item))
    return result

# 4. Create benchmark
import timeit

def benchmark():
    data = create_test_data(10000)
    time_taken = timeit.timeit(
        lambda: slow_function(data),
        number=100
    )
    print(f"Average time: {time_taken / 100:.4f}s")

benchmark()  # Baseline: 0.1234s

# 5. Optimize
def optimized_function(data):
    # Use list comprehension (faster)
    return [transform(item) for item in data if is_valid(item)]

# 6. Benchmark again
time_taken = timeit.timeit(
    lambda: optimized_function(data),
    number=100
)
print(f"Average time: {time_taken / 100:.4f}s")  # 0.0789s - 36% faster!

# 7. Verify correctness
assert slow_function(data) == optimized_function(data)

# 8. Re-profile entire program to verify improvement
cProfile.run('main()', 'profile_stats_optimized')
```

**Why this matters**: Without profiling, you might optimize code that takes 1% of runtime, ignoring the 90% bottleneck. Always measure.

### Common Optimizations

```python
# ❌ WRONG: Repeated expensive operations
def process_items(items):
    for item in items:
        # Regex compiled every iteration!
        pattern = re.compile(r'\d+')
        match = pattern.search(item)

# ✅ CORRECT: Move expensive operations outside loop
def process_items(items):
    pattern = re.compile(r'\d+')  # Compile once
    for item in items:
        match = pattern.search(item)

# ❌ WRONG: Growing list with repeated concatenation
def build_large_list():
    result = []
    for i in range(100000):
        result = result + [i]  # Creates new list each time! O(n²)

# ✅ CORRECT: Use append
def build_large_list():
    result = []
    for i in range(100000):
        result.append(i)  # O(n)

# ❌ WRONG: Checking membership in list
def filter_items(items, blacklist):
    return [item for item in items if item not in blacklist]
    # O(n * m) if blacklist is list

# ✅ CORRECT: Use set for membership checks
def filter_items(items, blacklist):
    blacklist_set = set(blacklist)  # O(m)
    return [item for item in items if item not in blacklist_set]
    # O(n) for iteration + O(1) per lookup = O(n)
```

### Caching Results

```python
from functools import lru_cache

# ❌ WRONG: Recomputing expensive results
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
# O(2^n) - recalculates same values repeatedly

# ✅ CORRECT: Cache results
@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
# O(n) - each value computed once

# ✅ CORRECT: Custom caching for unhashable arguments
from functools import wraps

def cache_dataframe_results(func):
    cache = {}

    @wraps(func)
    def wrapper(df):
        # Use hash of dataframe content as key
        key = hashlib.md5(df.to_csv(index=False).encode()).hexdigest()

        if key not in cache:
            cache[key] = func(df)

        return cache[key]

    return wrapper

@cache_dataframe_results
def expensive_dataframe_operation(df):
    # Complex computation
    return df.groupby('category').agg({'value': 'sum'})
```


## Systematic Diagnosis

### Performance Degradation Diagnosis

```python
# ✅ CORRECT: Diagnose performance regression
import cProfile
import pstats

def diagnose_slowdown():
    """Compare current vs baseline performance."""

    # Profile current code
    cProfile.run('main()', 'current_profile.prof')

    # Load baseline profile (from git history or previous run)
    # git show main:profile.prof > baseline_profile.prof

    current = pstats.Stats('current_profile.prof')
    baseline = pstats.Stats('baseline_profile.prof')

    print("=== CURRENT ===")
    current.sort_stats('cumulative')
    current.print_stats(10)

    print("\n=== BASELINE ===")
    baseline.sort_stats('cumulative')
    baseline.print_stats(10)

    # Look for functions that got slower
    # Compare cumulative times
```

### Memory Leak Diagnosis

```python
# ✅ CORRECT: Systematic memory leak detection
import tracemalloc
import gc

def diagnose_memory_leak():
    """Run function multiple times and check memory growth."""

    gc.collect()
    tracemalloc.start()

    # Baseline
    snapshot1 = tracemalloc.take_snapshot()

    # Run 100 times
    for _ in range(100):
        potentially_leaky_function()
        gc.collect()

    # Check memory
    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(f"{stat.traceback}: +{stat.size_diff / 1024:.1f} KB")

    tracemalloc.stop()
```

### I/O vs CPU Bound Diagnosis

```python
# ✅ CORRECT: Determine if I/O or CPU bound
import time
import cProfile

def diagnose_bottleneck():
    """Determine if program is I/O or CPU bound."""

    # Time wall clock
    start_wall = time.time()
    main()
    wall_time = time.time() - start_wall

    # Profile CPU time
    pr = cProfile.Profile()
    pr.enable()
    start_cpu = time.process_time()
    main()
    cpu_time = time.process_time() - start_cpu
    pr.disable()

    print(f"Wall time: {wall_time:.2f}s")
    print(f"CPU time: {cpu_time:.2f}s")

    if cpu_time / wall_time > 0.9:
        print("CPU bound - optimize computation")
        # Consider: Cython, NumPy, multiprocessing
    else:
        print("I/O bound - optimize I/O")
        # Consider: async/await, caching, batching
```


## Common Bottlenecks and Solutions

### String Concatenation

```python
# ❌ WRONG: String concatenation in loop
def build_string(items):
    result = ""
    for item in items:
        result += str(item) + "\n"  # Creates new string each time
    return result
# O(n²) time complexity

# ✅ CORRECT: Use join
def build_string(items):
    return "\n".join(str(item) for item in items)
# O(n) time complexity

# Benchmark:
# 1000 items: 0.0015s (join) vs 0.0234s (concatenation) - 15x faster
# 10000 items: 0.015s (join) vs 2.341s (concatenation) - 156x faster
```

### List Comprehension vs Map/Filter

```python
import timeit

# ✅ CORRECT: List comprehension (usually fastest)
def with_list_comp(data):
    return [x * 2 for x in data if x > 0]

# ✅ CORRECT: Generator (memory efficient for large data)
def with_generator(data):
    return (x * 2 for x in data if x > 0)

# Map/filter (sometimes faster for simple operations)
def with_map_filter(data):
    return map(lambda x: x * 2, filter(lambda x: x > 0, data))

# Benchmark
data = list(range(1000000))
print(timeit.timeit(lambda: list(with_list_comp(data)), number=10))
print(timeit.timeit(lambda: list(with_generator(data)), number=10))
print(timeit.timeit(lambda: list(with_map_filter(data)), number=10))

# Results: List comprehension usually fastest for complex logic
# Generator best when you don't need all results at once
```

### Dictionary Lookups vs List Searches

```python
# ❌ WRONG: Searching in list
def find_users_list(user_ids, all_users_list):
    results = []
    for user_id in user_ids:
        for user in all_users_list:  # O(n) per lookup
            if user['id'] == user_id:
                results.append(user)
                break
    return results
# O(n * m) time complexity

# ✅ CORRECT: Use dictionary
def find_users_dict(user_ids, all_users_dict):
    return [all_users_dict[uid] for uid in user_ids if uid in all_users_dict]
# O(n) time complexity

# Benchmark:
# 1000 lookups in 10000 items:
# List: 1.234s
# Dict: 0.001s - 1234x faster!
```

### DataFrame Iteration Anti-Pattern

```python
import pandas as pd
import numpy as np

# ❌ WRONG: Iterating over DataFrame rows
def process_rows_iterrows(df):
    results = []
    for idx, row in df.iterrows():  # VERY SLOW
        if row['value'] > 0:
            results.append(row['value'] * 2)
    return results

# ✅ CORRECT: Vectorized operations
def process_rows_vectorized(df):
    mask = df['value'] > 0
    return (df.loc[mask, 'value'] * 2).tolist()

# Benchmark with 100,000 rows:
# iterrows: 15.234s
# vectorized: 0.015s - 1000x faster!
```


## Profiling Tools Comparison

### When to Use Which Tool

| Tool | Use Case | Output |
|------|----------|--------|
| cProfile | Function-level CPU profiling | Which functions take most time |
| line_profiler | Line-level CPU profiling | Which lines within function slow |
| memory_profiler | Line-level memory profiling | Memory usage per line |
| tracemalloc | Memory allocation tracking | Where memory allocated |
| yappi | Async/multithreaded profiling | Profile concurrent code |
| py-spy | Sampling profiler (no code changes) | Profile running processes |
| scalene | CPU+GPU+memory profiling | Comprehensive profiling |

### py-spy for Production Profiling

```bash
# Install: pip install py-spy

# Profile running process (no code changes needed!)
py-spy record -o profile.svg --pid 12345

# Profile for 60 seconds
py-spy record -o profile.svg --duration 60 -- python script.py

# Top-like view of running process
py-spy top --pid 12345

# Why use py-spy:
# - No code changes needed
# - Minimal overhead
# - Can attach to running process
# - Great for production debugging
```


## Anti-Patterns

### Premature Optimization

```python
# ❌ WRONG: Optimizing before measuring
def process_data(data):
    # "Let me make this fast with complex caching..."
    # Spend hours optimizing function that takes 0.1% of runtime

# ✅ CORRECT: Profile first
cProfile.run('main()', 'profile.prof')
# Oh, process_data only takes 0.1% of time
# The real bottleneck is database queries (90% of time)
# Optimize database queries instead!
```

### Micro-Optimizations

```python
# ❌ WRONG: Micro-optimizing at expense of readability
def calculate(x, y):
    # "Using bit shift instead of multiply by 2 for speed!"
    return (x << 1) + (y << 1)
# Saved: ~0.0000001 seconds per call
# Cost: Unreadable code

# ✅ CORRECT: Clear code first
def calculate(x, y):
    return 2 * x + 2 * y
# Modern Python JIT optimizes this anyway
# Only optimize if profiler shows this is bottleneck
```

### Not Benchmarking Changes

```python
# ❌ WRONG: Assuming optimization worked
def slow_function():
    # Original code
    pass

def optimized_function():
    # "Optimized" code
    pass

# Assume optimized_function is faster without measuring

# ✅ CORRECT: Benchmark before and after
import timeit

before = timeit.timeit(slow_function, number=1000)
after = timeit.timeit(optimized_function, number=1000)

print(f"Before: {before:.4f}s")
print(f"After: {after:.4f}s")
print(f"Speedup: {before/after:.2f}x")

# Verify correctness
assert slow_function() == optimized_function()
```


## Decision Trees

### What Tool to Use for Profiling?

```
What do I need to profile?
├─ CPU time
│   ├─ Function-level → cProfile
│   ├─ Line-level → line_profiler
│   └─ Async code → yappi
├─ Memory usage
│   ├─ Line-level → memory_profiler
│   ├─ Allocation tracking → tracemalloc
│   └─ Object types → objgraph
└─ Running process (no code changes) → py-spy
```

### Optimization Strategy

```
Is code slow?
├─ Yes → Profile to find bottleneck
│   ├─ CPU bound → Profile with cProfile
│   │   └─ Optimize hot functions (vectorize, cache, algorithms)
│   └─ I/O bound → Profile with timing
│       └─ Use async/await, caching, batching
└─ No → Don't optimize (focus on features/correctness)
```

### Memory Issue Diagnosis

```
Is memory usage high?
├─ Yes → Profile with memory_profiler
│   ├─ Growing over time → Memory leak
│   │   └─ Use tracemalloc to find leak
│   └─ High but stable → Large data structures
│       └─ Optimize data structures (generators, efficient types)
└─ No → Monitor but don't optimize yet
```


## Integration with Other Skills

**After using this skill:**
- If I/O bound → See @async-patterns-and-concurrency for async optimization
- If data processing slow → See @scientific-computing-foundations for vectorization
- If need to track improvements → See @ml-engineering-workflows for metrics

**Before using this skill:**
- If unsure code is slow → Use this skill to profile and confirm!
- If setting up profiling → See @project-structure-and-tooling for dependencies


## Quick Reference

### Essential Profiling Commands

```python
# CPU profiling
import cProfile
cProfile.run('main()', 'profile.prof')

# View results
import pstats
stats = pstats.Stats('profile.prof')
stats.sort_stats('cumulative')
stats.print_stats(20)

# Memory profiling
import tracemalloc
tracemalloc.start()
# ... code ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

### Debugging Commands

```python
# Set breakpoint
breakpoint()  # Python 3.7+
# or
import pdb; pdb.set_trace()

# pdb commands:
# n - next line
# s - step into
# c - continue
# p var - print variable
# l - list code
# w - where am I
# q - quit
```

### Optimization Checklist

- [ ] Profile before optimizing (use cProfile)
- [ ] Identify bottleneck (top 20% of time)
- [ ] Create benchmark for bottleneck
- [ ] Optimize bottleneck
- [ ] Benchmark again to verify improvement
- [ ] Re-profile entire program
- [ ] Verify correctness (tests still pass)

### Common Optimizations

| Problem | Solution | Speedup |
|---------|----------|---------|
| String concatenation in loop | Use str.join() | 10-100x |
| List membership checks | Use set | 100-1000x |
| DataFrame iteration | Vectorize with NumPy/pandas | 100-1000x |
| Repeated expensive computation | Cache with @lru_cache | ∞ (depends on cache hits) |
| I/O bound | Use async/await | 10-100x |
| CPU bound with parallelizable work | Use multiprocessing | ~number of cores |

### Red Flags

If you find yourself:
- Optimizing before profiling → STOP, profile first
- Spending hours on micro-optimizations → Check if it's bottleneck
- Making code unreadable for speed → Benchmark the benefit
- Assuming what's slow → Profile to verify

**Always measure. Never assume.**
