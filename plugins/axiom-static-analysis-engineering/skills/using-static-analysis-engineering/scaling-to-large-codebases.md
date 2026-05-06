---
name: scaling-to-large-codebases
description: Use when whole-program analysis stops being affordable — codebase exceeds ~100k LoC, analyzer takes minutes per run, CI blocks for too long, IDE integration becomes impossible — and you need incremental analysis, caching of intermediate artifacts, parallelism within phases, watch-mode operation, and partition strategies that preserve soundness. Covers the cache key composition that survives lattice-version bumps, reverse-edge indexes for invalidation, parallel worklist algorithms, fan-out across modules, and the "incremental analysis silently lies" failure mode. Produces `12-scaling-and-incrementality.md`.
---

# Scaling to Large Codebases

## When You Reach for This Sheet

The whole-program design from `three-phase-inference.md` works fine until it doesn't. The signs:

- A clean run takes more than ~30 seconds; CI waits.
- IDE integration is impractical (the analyzer cannot keep up with edits).
- Memory grows superlinearly with codebase size.
- "Just analyse the changed files" turns out to silently miss findings, because analysis is inter-procedural and changed-file analysis loses context.

`12-scaling-and-incrementality.md` is where the engine becomes incremental without becoming dishonest. The hard part is the *without becoming dishonest*. Most incremental analyzers skip work they should not skip; the result is silent unsoundness — clean PRs that introduce bugs because the changed code's flow into unchanged code was never re-analysed.

## The Soundness Floor for Incrementality

An incremental analyzer is sound iff, for every finding the whole-program analyzer would have produced on the current source state, the incremental analyzer also produces it. Equivalently: every finding the whole-program would emit, the incremental either emits or proves absent through a cached prior verdict that is still valid.

Three things must be true:

1. **Cache validity is decidable.** Given a cached result, the engine can decide whether the result is still valid for the current state. "Valid" here means the cached result is what the whole-program would have produced.
2. **Invalidation is complete.** When a change invalidates one cached result, every transitively dependent cached result is also invalidated. No silent stragglers.
3. **The fall-back is soundness, not silence.** If validity cannot be decided, the engine re-analyses; it does not skip.

Lose any of these and the incremental analyzer is a "fast" analyzer that doesn't catch the bug.

## The Cache Key

Every cached artifact (Phase 1 env at a function, Phase 2 summary, Phase 3 callsite env, finding emission) is keyed by a **cache key**. The key is the disagreement-resistant, complete description of every input that determined the artifact.

```
cache_key = sha256(
   analyzer_version            # 99- semver
   || lattice_version           # 02- semver (bumps reset everything)
   || ruleset_version           # 04- semver (rule changes reset rule outputs)
   || stub_library_version      # 08- semver (boundary changes reset cross-module flow)
   || manifest_hash             # 10- effective configuration
   || function_body_canonical   # the AST canonicalised to be format-invariant
   || transitive_input_hashes   # the cache keys of every artifact this one depends on
)
```

**Each component matters:**

- **`analyzer_version`** — a new analyzer release may compute differently; old caches must not be reused.
- **`lattice_version`** — a tier renamed in `02-` invalidates every lattice value computed with the old name.
- **`ruleset_version`** — a rule's predicate changed; cached findings from the rule are stale.
- **`stub_library_version`** — a stub for `requests.get` changed; every cross-module flow through that stub is stale.
- **`manifest_hash`** — the effective manifest changed (different stubs, different rules enabled); cached results may not apply.
- **`function_body_canonical`** — the function's content (canonicalised: whitespace-stripped, comments-removed, AST-normalised). Cosmetic edits don't invalidate.
- **`transitive_input_hashes`** — Phase 2's summary depends on Phase 1's env; Phase 3's callsite env depends on Phase 2's summaries of every callee. Build the dependency graph; hash it.

A cache key that omits any component is a soundness bug waiting for the right release. A cache key that includes them all may produce more cache misses than necessary; that's the right side of the tradeoff for analysis correctness.

## Three Caches, Three Invalidation Rules

Each phase has its own cache.

### Phase 1 cache: per-function env

`function_body_hash → env_at_each_program_point`

**Invalidation:**

- Function body changes (canonical hash differs) → invalidate.
- Lattice version bumps → invalidate (transfer functions may compute differently).
- Stub library updates affecting any callee from Phase 3 callgraph → potentially invalidate (Phase 1 doesn't see callees, but Phase 1 *does* see the stub if a function calls a stubbed external — the env at the call propagation depends on the stub).

In practice, Phase 1 caches survive most edits. They are the cheapest to recompute when invalidated.

### Phase 2 cache: per-function summary

`(function_body_hash, lattice_version, ruleset_version) → FunctionSummary`

**Invalidation:**

- Function body changes → invalidate.
- Lattice version bumps → invalidate.
- Phase 1 env at any program point in the function changes → invalidate (the summary is derived from Phase 1).

Phase 2 cache hits are the highest-value: a summary represents inter-procedural reasoning about the function. Hitting the cache saves rebuilding flow at every callsite.

### Phase 3 cache: per-callsite env

`(callsite_id, callee_summary_hash, caller_env_hash, callgraph_version) → callsite_post_env`

**Invalidation:**

- Callee summary changes → invalidate this callsite's cached env.
- Caller env at the call point changes → invalidate.
- Callgraph version changes (resolution rung changed, or a refinement step added/removed an edge) → invalidate.

Phase 3 invalidation is **transitive**: a callsite's invalidation may cause its caller's summary to change (the post-call env feeds back to the summary), which invalidates *that* function's callsites, and so on. The transitive closure of invalidation is the same shape as Phase 3's iterated worklist.

## The Reverse Edge Index

Incremental invalidation requires answering: "what cached artifacts depend on this one?" Without an index, the engine must scan every cache entry and check transitive_input_hashes. With an index, it's a constant-time lookup.

```
reverse_index : artifact_id → set[artifact_id]
   "for each artifact, the set of artifacts that include it in their cache key"
```

Built incrementally: when an artifact is computed, the engine records its dependencies; when an artifact is invalidated, the engine looks up its dependents and queues them for invalidation.

This is the standard build-system structure (Bazel, Buck2, Pants), retargeted to analyzer artifacts. The index lives in the cache directory; it survives across runs.

**Failure mode:** the reverse index drifts from reality (caches deleted manually, or from a crash mid-run). The engine must validate the index on startup (cheap consistency check) and rebuild from scratch if drift is detected.

## Parallelism Within Phases

Phases are sequential (Phase 2 needs Phase 1; Phase 3 needs Phase 2). Within each phase, parallelism is available.

### Phase 1 — embarrassingly parallel

Each function body is analysed independently. Distribute across cores trivially.

```
with thread_pool() as pool:
    pool.map(analyse_function_body, functions)
```

Caveats:

- Some Phase 1 results depend on stubs (which depend on Phase 3 outcomes if iterated refinement is enabled; see `callgraph-construction.md`). In the iterated case, parallelism is per-iteration.
- Parser and AST construction may not be thread-safe in some toolchains (CPython's `ast` module is, with caveats; some C-extension parsers are not). Confirm.

### Phase 2 — parallel after Phase 1 completes

Each function summary is built from its Phase 1 result. Independent across functions.

### Phase 3 — partial parallelism

The callgraph-level worklist has a partial order: SCCs (strongly connected components) of the callgraph can be processed in topological order. Within an SCC, propagation is sequential (the SCC may converge through internal iteration); across SCCs, parallel.

For most codebases, the callgraph has many small SCCs and a long topological tail. Parallelising across SCCs gives a substantial speedup.

```
sccs = strongly_connected_components(callgraph)
ordered = topological_sort(sccs)
for level in ordered_by_level(ordered):
    pool.map(analyse_scc, level)        # each level processed in parallel
```

### Cross-phase parallelism (advanced)

If iterated refinement is on, Phase 1 and Phase 3 alternate; Phase 2 can run in parallel with Phase 3's stable parts. The complexity may not pay for itself; profile first.

### Determinism under parallelism

Parallel analysis must produce the same findings as sequential analysis. The hard cases:

- **Iteration order** in worklist algorithms — when multiple work items are available, the order they're popped affects the *steps*, not the *fixed point* (if monotonic). But it affects the *fingerprints* if fingerprints encode iteration order. Don't.
- **Hash-map iteration** in any phase — Python `dict`, Go `map` random iteration. Always sort or use ordered structures when iteration order is observable in output.
- **Floating-point lattices** — if the abstract domain involves floats (rare), join order can produce different fixed points across runs. Avoid float lattices; use rationals or symbolic intervals.

Cross-link to `axiom-determinism-and-replay`: the analyzer is a deterministic system in their sense. Same disciplines apply.

## Watch Mode and IDE Integration

For IDE use, the analyzer must keep up with edits — sub-second response to a change is the threshold below which IDE integration feels broken.

The discipline:

- **Watch the file system** — `inotify` / `FSEvents` / equivalent. Each change is a triggered re-analysis.
- **Debounce** — collect changes for ~50ms before re-analysis to absorb burst saves.
- **Re-analyse only the changed function and its dependents** — Phase 1 for the changed body, Phase 2 invalidation, Phase 3 propagation only into the affected callees.
- **Emit findings via LSP** — the engine speaks Language Server Protocol or a SARIF-stream variant; the IDE consumes.
- **Snapshot consistency** — the IDE may have unsaved changes; the engine analyses what the IDE supplies (LSP `textDocument/didChange`) rather than the on-disk content.

For a codebase that takes 30 seconds whole-program, sub-second incremental requires Phase 1 cache hit rates above 99% on a single-function edit. Achievable with the cache key + reverse index discipline above; not achievable with naïve "re-analyse just the file" heuristics that miss flow.

## Partition Strategies (When Even Incremental Isn't Enough)

For codebases beyond ~1M LoC, even incremental whole-program is slow. The escape:

### By module / package

The codebase is partitioned into modules; each module is analysed against summaries of others. Cross-module callsites use the boundary's cached summaries (`08-cross-module-flow.md`).

**Soundness:** sound iff the boundary summaries are accurate. If a module changes its public API in a way the boundary doesn't capture, callers analysed against the old summary are stale.

**Discipline:** module summaries have their own cache keys (versioned), and changing a module's public API invalidates every dependent module's analysis.

### By tier (security-bearing first)

Run only the tier-relevant rules on every PR (e.g., taint rules); run the full set on nightly builds. Cheap; sound for the tier-relevant rules; doesn't catch tier-irrelevant findings on PRs.

### By owner (sharded analysis)

Different teams own different modules; each team runs the analyzer over their part; results aggregated.

**Risk:** boundary findings (a flow from team A's code to team B's sink) may fall in the seams. Ensure the boundary is analysed by *one* team consistently or by a meta-pass.

State the partition strategy explicitly in `12-`. Partition is a soundness compromise; the compromise must be described, not glossed.

## Failure Modes

| Failure | Manifestation | Diagnosis |
|---------|---------------|-----------|
| Stale cache after lattice change | New rule fires on test corpus but not in CI | Lattice version not in cache key |
| Stale cache after analyzer upgrade | Findings disappear that were present before; no rule changes | Analyzer version not in cache key |
| Manifest change not reflected | Disabling a rule still produces its findings | Manifest hash not in cache key, or only a portion of the manifest hashed |
| Incremental misses cross-function flow | Single-file edit doesn't re-analyse callers | Reverse index missing the dependency, or invalidation didn't propagate |
| Parallel run yields different findings than sequential | Determinism violated | Iteration order sensitivity in some phase; sort/canonicalise |
| Watch mode silently drops findings on burst saves | High edit rate; engine misses an intermediate state | Debounce missed an event, or watcher dropped events under load — re-validate cache against latest state on idle |
| Cache grows without bound | Disk full | Eviction policy: LRU + max-age; cache size cap; periodic compaction |
| Reverse index corrupted | Random invalidation misses | Validate on startup; rebuild from cache directory if drift detected |

## Soundness Self-Test

A first-class CI metric: the **incremental-vs-whole-program self-test**.

Periodically (nightly, or on a configurable schedule), run a fresh whole-program analysis on the same source state and diff its findings against the incremental run. Any disagreement is a bug:

```
nightly:
   - cargo run --no-cache --whole-program → findings_full.sarif
   - diff findings_full.sarif findings_latest_incremental.sarif
   - if non-empty: alert; do not silently accept
```

This is the only honest way to know the cache is correct. Without it, "the incremental cache works" is an assertion. Run the self-test; it costs one nightly whole-program run; it earns confidence.

## The Cache as Audit Artifact

For tier L/XL analyzers under audit obligations: the cache is reproducible evidence. A finding's emission was determined by a specific cache state; if the audit needs to reconstruct *why* a finding was emitted on a specific commit, the cache (or the cache's fingerprint chain) is the evidence.

Implications:

- **Cache hashes are recorded** in the analyzer's run output (SARIF properties).
- **Cache content is reproducible** from source + manifest + analyzer version (deterministic computation).
- **Cache eviction is logged** — what was evicted, when, why.

For tier S/M, this is overkill. For tier L/XL, the cache is part of the evidence regime alongside the suppression set and the manifest history. Cross-link to `axiom-audit-pipelines`.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Cache key omits lattice version | Stale findings after `02-` change | Cache key includes every version that determines computation |
| No reverse index | Invalidation requires full cache scan; performance regression on the second-or-later run | Build reverse index incrementally; persist; validate on startup |
| Incremental on changed-files-only | Inter-procedural flow across module boundaries missed | Invalidate transitively via reverse index |
| Parallel iteration order leaks into output | Findings differ across runs | Sort all observable iteration orders; same hashes always |
| Watch mode without debounce | Spurious re-analyses on multi-file save | Debounce ~50ms; coalesce events |
| Watch mode trusts file system | Engine analyses on-disk content; IDE has unsaved edits | LSP is authoritative; engine takes content from LSP |
| Cache size unbounded | Disk full; mysterious slow CI | Eviction policy: LRU + size cap |
| No incremental-vs-whole-program self-test | Cache lies silently | Nightly self-test; alert on diff |
| Module partition without boundary discipline | Cross-module findings missed | Boundary summaries are first-class; cross-link to `08-` |
| Stub-library version not in cache key | Library upgrade silently invalidated nothing | Stub library version in cache key for every Phase 2 + Phase 3 entry |

## The Decision Output (`12-scaling-and-incrementality.md`)

A complete `12-` answers:

1. **Operating modes** — whole-program, incremental, watch; when each applies.
2. **Cache key composition** — every component; rationale; cross-link to versioning of `02-`, `04-`, `08-`, `10-`, `99-`.
3. **Cache structure** — three caches, what each holds, where they live.
4. **Reverse index** — schema; build/maintain protocol; corruption recovery.
5. **Invalidation rules** — per phase; transitive closure.
6. **Parallelism** — within each phase; topological structure of Phase 3; determinism preservation.
7. **Watch-mode protocol** — file watch / LSP; debounce; snapshot consistency.
8. **Partition strategy** — if any; soundness compromise statement.
9. **Eviction and size cap** — policy; when triggered.
10. **Self-test** — incremental-vs-whole-program comparison; cadence; alert behaviour.
11. **Audit considerations** — for tier L/XL, cache as evidence; reproducibility claim.

## Cross-References

- `taint-lattice-design.md` — the lattice version that gates cache validity
- `three-phase-inference.md` — the three caches correspond to the three phases; their dependencies; their invalidation rules
- `plugin-architecture-for-analyzer-rules.md` — the ruleset version that gates Phase 3 cache validity
- `cross-module-flow-analysis.md` — stub library version; module-boundary summaries
- `manifest-driven-configuration-with-coherence-validation.md` — manifest hash that gates cache validity
- `callgraph-construction.md` — callgraph version; reverse index for callgraph edges
- `sarif-emission-and-ci-integration.md` — incremental SARIF emission; baseline comparison interaction
- Cross-pack: `axiom-determinism-and-replay:reproducibility` — incremental analysis is a deterministic computation; same disciplines
- Cross-pack: `axiom-audit-pipelines:fingerprint-chains` — cache fingerprint as audit evidence in tier L/XL
- Cross-pack: `axiom-rust-engineering:performance-profiling` — when the bottleneck is the engine itself, profile before partitioning
