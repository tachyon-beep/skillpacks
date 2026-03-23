---
description: Profile Python code to find actual bottlenecks before optimizing
allowed-tools: ["Read", "Bash", "Write", "Skill"]
argument-hint: "<file.py> [function_or_script_args]"
---

# Profile Command

Profile Python code to identify real bottlenecks. Never optimize without profiling first.

## Core Principle

Humans are terrible at guessing where code is slow. Profile first, then optimize the actual bottleneck.

## Process

1. **Run CPU profiling**
   ```bash
   python -m cProfile -s cumulative ${ARGUMENTS} 2>&1 | head -50
   ```

   Or for more detail:
   ```bash
   python -m cProfile -o profile.stats ${ARGUMENTS}
   python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
   ```

2. **Identify top bottlenecks**
   - Look at cumulative time (tottime vs cumtime)
   - Identify which functions consume the most time
   - Note call counts for frequently-called functions

3. **For memory profiling** (if memory is the concern)
   ```bash
   python -m memory_profiler ${ARGUMENTS}
   ```

   Or line-by-line:
   ```python
   # Add @profile decorator to functions of interest
   # Then run: python -m memory_profiler script.py
   ```

4. **Report findings**
   - Top 5 time-consuming functions
   - Whether bottleneck is CPU-bound or I/O-bound
   - Recommended next steps

## After Profiling - Route to Appropriate Skill

Based on findings, recommend:

| Bottleneck Type | Next Step |
|-----------------|-----------|
| I/O-bound (network, file) | Load async-patterns-and-concurrency.md |
| Array/DataFrame operations | Load scientific-computing-foundations.md |
| CPU-bound computation | Consider algorithmic optimization |
| Memory issues | Load debugging-and-profiling.md for memory profiling |

## Load Detailed Guidance

For comprehensive profiling patterns:
```
Load skill: axiom-python-engineering:using-python-engineering
Then read: debugging-and-profiling.md
```

## Example Session

```
User: /python-engineering:profile src/data_processor.py

Claude:
1. Running cProfile on src/data_processor.py...

2. Top bottlenecks:
   - process_dataframe(): 45% of time (cumtime: 12.3s)
   - load_data(): 30% of time (cumtime: 8.1s)
   - validate_rows(): 15% of time (cumtime: 4.0s)

3. Analysis:
   - load_data() appears I/O-bound (file reading)
   - process_dataframe() has high call count (10000 calls)

4. Recommendations:
   - For process_dataframe(): Check for vectorization opportunities
     → Load scientific-computing-foundations.md
   - For load_data(): Consider async I/O or chunked reading
     → Load async-patterns-and-concurrency.md
```
