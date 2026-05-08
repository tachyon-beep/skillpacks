# Findings Schema (Ultralarge Track)

## Purpose

When module-by-module archaeology runs across many modules and many reviewers, the **only thing that makes synthesis tractable is a strict schema**. Free-form prose returns from N reviewers across M modules cannot be merged mechanically; the scribe role degenerates into another reviewer.

This sheet defines the canonical schema for module-level findings. It is consumed by:

- `module-reviewer` agent — fills one **partial** per focus (interface / internals / deps / quality)
- `subsystem-scribe` agent — merges 4 partials into one **canonical** module entry
- Synthesis pass — rolls up canonical module entries into the subsystem catalog

**If the schema is wrong, every downstream artifact is wrong.** Treat it as load-bearing.

---

## Core Principle

**Reviewers fill cells in a known table, not write essays.** A reviewer who writes prose has misunderstood the contract. The scribe's job is mechanical merge — that only works if every cell has a known shape.

---

## File Layout

For each subsystem `S` under analysis, the workspace contains:

```
docs/arch-analysis-YYYY-MM-DD-HHMM/
└── ultralarge/
    └── subsystem-S/
        ├── 00-partition.md           # Subsystem boundary, modules in scope, ordering
        ├── modules/
        │   ├── <module-id>.interface.partial.yaml
        │   ├── <module-id>.internals.partial.yaml
        │   ├── <module-id>.deps.partial.yaml
        │   ├── <module-id>.quality.partial.yaml
        │   └── <module-id>.canonical.yaml      # written by scribe after merge
        └── 99-subsystem-catalog-entry.md       # final synthesis (existing format)
```

`<module-id>` = filesystem-safe slug derived from module path (e.g. `src_pkg_auth_handler` for `src/pkg/auth/handler.py`).

**Why YAML for partials and canonical:** machine-mergeable, field validation possible, diffs are reviewable. Markdown only at the synthesis layer (catalog entry), where humans read.

---

## Reviewer Focus Definitions

Each `module-reviewer` invocation MUST be parameterized with exactly one focus:

| Focus | Reads | Produces partial |
|---|---|---|
| `interface` | Public symbols, signatures, type hints, docstrings, exports | `*.interface.partial.yaml` |
| `internals` | Control flow, key algorithms, state, invariants in implementation | `*.internals.partial.yaml` |
| `deps` | Imports, calls out, framework hooks, IO surfaces | `*.deps.partial.yaml` |
| `quality` | Smells, dead code, TODOs, test coverage, debt | `*.quality.partial.yaml` |

A reviewer focused on `interface` does NOT report smells. A reviewer focused on `quality` does NOT enumerate the public API. **Cross-focus reporting is a contract violation** — the scribe cannot merge if reviewers overlap arbitrarily.

---

## Partial Schema (per-focus output)

### Common header (all four focuses)

```yaml
schema_version: 1
module_id: <slug>
module_path: <repo-relative path>
module_loc: <integer line count>
focus: interface | internals | deps | quality
reviewer_run_id: <uuid or timestamp>
confidence: high | medium | low
confidence_evidence: <one sentence citing what was read>
```

### `interface.partial.yaml` body

```yaml
classes:
  - name: <ClassName>
    kind: class | dataclass | protocol | abstract | enum | namedtuple
    public: true | false
    bases: [<BaseName>, ...]
    summary: <one line>
public_functions:
  - name: <function_name>
    signature: <full signature including types>
    summary: <one line>
    raises: [<ExceptionType>, ...]   # if documented or obvious from code
public_constants:
  - name: <NAME>
    type: <type if inferrable>
    summary: <one line>
exports: [<symbol>, ...]   # __all__ or equivalent; empty list if none
contracts:
  - <natural-language contract observed in docstring or asserted invariant>
```

If a section has no entries, write an empty list `[]` — never omit the key.

### `internals.partial.yaml` body

```yaml
key_algorithms:
  - name: <descriptive name>
    location: <function or method>
    purpose: <one line>
    complexity: <Big-O if obvious, else "not analyzed">
state:
  - kind: module-level | class-level | per-instance
    name: <variable or attribute>
    mutability: immutable | mutated | rebound
    purpose: <one line>
invariants:
  - <invariant the code appears to maintain, with location>
control_flow_notes:
  - <non-obvious control-flow observation: early returns, exceptions as control, retries, etc>
side_effects:
  - kind: io | network | filesystem | global-state | logging | other
    location: <function or block>
    summary: <one line>
```

### `deps.partial.yaml` body

```yaml
imports_internal:
  - module: <repo-relative module path>
    symbols: [<symbol>, ...]
    usage: <one line: what this module does with it>
imports_external:
  - package: <distribution name, e.g. numpy>
    symbols: [<symbol>, ...]
    optional: true | false
calls_out:
  - target: <function / method / endpoint>
    via: import | dynamic-dispatch | reflection | subprocess | network | message-bus
    summary: <one line>
framework_hooks:
  - framework: <name, e.g. fastapi, pytest, django>
    hook: <decorator or registration mechanism>
    purpose: <one line>
io_surfaces:
  - kind: cli | http | grpc | file | database | queue | env-var | other
    direction: in | out | both
    summary: <one line>
```

### `quality.partial.yaml` body

```yaml
smells:
  - kind: long-function | god-class | duplicated-logic | magic-number | unclear-naming | tight-coupling | unsafe-default | other
    location: <function/class/line range>
    severity: low | medium | high
    summary: <one line>
todos:
  - location: <line>
    text: <verbatim TODO/FIXME/XXX content>
dead_code_suspects:
  - location: <function or block>
    reason: <why suspected dead>
test_refs:
  - test_file: <repo-relative path>
    covers: [<symbol>, ...]
    kind: unit | integration | property | e2e | smoke
coverage_gaps:
  - symbol: <function or class>
    reason: <why coverage is insufficient — no tests / only happy path / etc>
debt_observations:
  - <one-line observation that doesn't fit a smell category>
```

---

## Canonical Schema (after scribe merge)

The scribe merges the four partials into ONE canonical entry per module. The canonical schema is the **union** of partial bodies plus a `provenance` block.

```yaml
schema_version: 1
module_id: <slug>
module_path: <repo-relative path>
module_loc: <integer>
provenance:
  partials_merged: [interface, internals, deps, quality]
  reviewer_run_ids:
    interface: <id>
    internals: <id>
    deps: <id>
    quality: <id>
  scribe_run_id: <id>
  conflicts_resolved: [<one line per conflict, citing both partials>]
confidence:
  interface: high | medium | low
  internals: high | medium | low
  deps: high | medium | low
  quality: high | medium | low
  overall: high | medium | low   # MIN of the four unless scribe documents reasoning
classes: [...]              # from interface
public_functions: [...]     # from interface
public_constants: [...]     # from interface
exports: [...]              # from interface
contracts: [...]            # from interface
key_algorithms: [...]       # from internals
state: [...]                # from internals
invariants: [...]           # from internals
control_flow_notes: [...]   # from internals
side_effects: [...]         # from internals
imports_internal: [...]     # from deps
imports_external: [...]     # from deps
calls_out: [...]            # from deps
framework_hooks: [...]      # from deps
io_surfaces: [...]          # from deps
smells: [...]               # from quality
todos: [...]                # from quality
dead_code_suspects: [...]   # from quality
test_refs: [...]            # from quality
coverage_gaps: [...]        # from quality
debt_observations: [...]    # from quality
```

The scribe is **not authorized to add new findings**. It may only:

- Copy entries from partials to canonical
- Resolve duplicates (same class listed by interface and internals → keep interface's entry)
- Resolve conflicts (interface says `public: true`, internals says private use only → log in `provenance.conflicts_resolved`, keep interface's)
- Compute `confidence.overall`

If a partial is missing, the canonical entry is **incomplete** — the scribe MUST mark `provenance.partials_merged` accordingly and `confidence.overall: low`.

---

## Worked Example (Imagined Module)

Consider an imagined module `pkg/cache/lru.py` (~180 LOC) implementing an LRU cache.

### `interface.partial.yaml`

```yaml
schema_version: 1
module_id: pkg_cache_lru
module_path: pkg/cache/lru.py
module_loc: 178
focus: interface
reviewer_run_id: 2026-05-08T14:02:11Z
confidence: high
confidence_evidence: Read full file (178 LOC) and verified __all__ matches public symbols.

classes:
  - name: LRUCache
    kind: class
    public: true
    bases: [MutableMapping]
    summary: Bounded LRU cache with TTL eviction.
public_functions:
  - name: lru_cached
    signature: "def lru_cached(maxsize: int = 128, ttl: float | None = None) -> Callable"
    summary: Decorator factory binding an LRUCache to a function.
    raises: [ValueError]
public_constants:
  - name: DEFAULT_MAXSIZE
    type: int
    summary: Default cache capacity (128) used by lru_cached.
exports: [LRUCache, lru_cached, DEFAULT_MAXSIZE]
contracts:
  - LRUCache.get returns sentinel _MISSING (not None) so None can be a valid cached value.
  - lru_cached raises ValueError if maxsize < 1.
```

### `internals.partial.yaml`

```yaml
schema_version: 1
module_id: pkg_cache_lru
module_path: pkg/cache/lru.py
module_loc: 178
focus: internals
reviewer_run_id: 2026-05-08T14:02:14Z
confidence: medium
confidence_evidence: Read all methods. TTL eviction path not exercised by visible code paths; behavior inferred.

key_algorithms:
  - name: LRU eviction via OrderedDict.move_to_end
    location: LRUCache.__getitem__
    purpose: Maintain access order for O(1) eviction.
    complexity: O(1) per access
state:
  - kind: per-instance
    name: _store
    mutability: mutated
    purpose: OrderedDict holding (value, inserted_at) tuples.
  - kind: per-instance
    name: _maxsize
    mutability: immutable
    purpose: Capacity bound, set at construction.
invariants:
  - len(_store) <= _maxsize at all post-method states (asserted in __setitem__ end).
  - Insertion order in _store reflects LRU order (most-recent at end).
control_flow_notes:
  - __getitem__ raises KeyError, but get() returns _MISSING — divergent error model on purpose.
  - TTL check happens lazily on access, not on a timer.
side_effects:
  - kind: logging
    location: LRUCache._evict
    summary: DEBUG-level log on each eviction.
```

### `deps.partial.yaml`

```yaml
schema_version: 1
module_id: pkg_cache_lru
module_path: pkg/cache/lru.py
module_loc: 178
focus: deps
reviewer_run_id: 2026-05-08T14:02:09Z
confidence: high
confidence_evidence: Read all imports; verified no dynamic imports or reflection.

imports_internal:
  - module: pkg/_logging.py
    symbols: [get_logger]
    usage: Module-level logger for eviction debug logs.
imports_external:
  - package: stdlib (collections)
    symbols: [OrderedDict]
    optional: false
  - package: stdlib (time)
    symbols: [monotonic]
    optional: false
calls_out: []
framework_hooks: []
io_surfaces:
  - kind: other
    direction: out
    summary: Logging only; no other IO.
```

### `quality.partial.yaml`

```yaml
schema_version: 1
module_id: pkg_cache_lru
module_path: pkg/cache/lru.py
module_loc: 178
focus: quality
reviewer_run_id: 2026-05-08T14:02:18Z
confidence: high
confidence_evidence: Read full module + tests/test_lru.py (412 LOC).

smells:
  - kind: magic-number
    location: LRUCache.__init__:43
    severity: low
    summary: Default ttl_check_interval=60 hardcoded; not exposed as kwarg.
todos:
  - location: pkg/cache/lru.py:131
    text: "TODO: support per-key TTL override"
dead_code_suspects: []
test_refs:
  - test_file: tests/test_lru.py
    covers: [LRUCache, lru_cached]
    kind: unit
  - test_file: tests/test_lru_property.py
    covers: [LRUCache]
    kind: property
coverage_gaps:
  - symbol: LRUCache._evict
    reason: Eviction path tested only via fill-then-overflow; concurrent access not tested.
debt_observations:
  - lru_cached decorator does not propagate __wrapped__ attribute.
```

The scribe merges these four into `pkg_cache_lru.canonical.yaml`, no new content added.

---

## What a Bad Partial Looks Like

```yaml
focus: interface
classes:
  - name: LRUCache
    summary: |
      LRUCache is a really useful class that lets you cache things.
      It has a get method, a set method, and an evict method. It uses
      an OrderedDict internally, which is a Python data structure that
      maintains insertion order. The TTL feature is interesting because...
```

This is prose, not structured data. Scribe cannot merge. **Reject and re-spawn the reviewer with the schema link.**

---

## Validation Checklist (Reviewer Self-Check)

Before returning, every reviewer MUST verify:

- [ ] **The output file parses as YAML.** Run `python3 -c "import yaml; yaml.safe_load(open('<path>'))"` (or equivalent) AFTER writing. If the parse raises any exception, fix the file and re-validate. **Do not skip this — self-validation that does not actually parse the YAML is silent failure.** Common trap: an unquoted scalar that begins with `"` (e.g., `reason: "Failed to ..." log path requires...`) — YAML treats the leading quote as opening a double-quoted string and chokes on the trailing text. Fix by either quoting the entire value with single quotes (`reason: '"Failed to ..." log path requires...'`) or rewording to avoid leading punctuation.
- [ ] `schema_version: 1` is present
- [ ] `focus` matches the focus the reviewer was assigned
- [ ] All required keys for that focus are present (use `[]` for empty, never omit)
- [ ] `confidence_evidence` cites what was actually read
- [ ] No findings outside this focus's scope (interface reviewer reports zero smells, etc.)
- [ ] Each list entry's `summary` is exactly one line
- [ ] No prose paragraphs — only schema-shaped data

A partial that fails this checklist is invalid and the reviewer is re-spawned.

---

## Validation Checklist (Scribe Self-Check)

Before producing canonical:

- [ ] **All 4 partials parse as YAML.** Run `yaml.safe_load` on each partial BEFORE attempting merge. If any partial fails to parse, STOP — report the failure to the orchestrator with the parser error message. Do NOT attempt to fix the partial yourself; the failing reviewer must be re-spawned. Do NOT produce a canonical from invalid input.
- [ ] All 4 partials present (or `partials_merged` reflects which are missing)
- [ ] `module_id`, `module_path`, `module_loc` agree across all partials (mismatch = reject, re-spawn)
- [ ] No new findings added (every line in canonical traces to a partial)
- [ ] Conflicts logged in `provenance.conflicts_resolved`
- [ ] `confidence.overall` defaults to MIN; deviation requires reasoning in provenance
- [ ] **The canonical file parses as YAML.** Same `yaml.safe_load` check as above, AFTER writing the canonical. If it fails, the fix is on the scribe (likely an escaping issue introduced during merge); fix and re-validate.

---

## Integration with Existing Catalog

After all modules in a subsystem are canonical, a synthesis pass produces the existing
`02-subsystem-catalog.md` entry (contract defined in `analyzing-unknown-codebases.md`)
**by aggregation**:

- **Key Components** ← top-N modules by inbound `imports_internal` references
- **Dependencies (Outbound)** ← union of `imports_internal` minus self-references
- **Patterns Observed** ← recurring `key_algorithms` / `framework_hooks` across modules
- **Concerns** ← roll-up of `smells` (high-severity) + `coverage_gaps` + cross-module duplications
- **Confidence** ← MIN of canonical `confidence.overall` across modules

The ultralarge track does not replace the existing catalog format — it generates it from
denser machine-readable data, with full traceability back to per-module partials.

---

## Anti-Patterns

**❌ Reviewer writes prose under "summary" fields.** One line, schema-shaped.

**❌ Reviewer reports findings outside their focus.** Stay in lane.

**❌ Scribe adds new findings during merge.** Scribe only copies + dedupes + resolves conflicts.

**❌ Skip validation checklists "because the file looks fine".** Schema violations cascade.

**❌ Use markdown for partials.** Markdown is for human-readable synthesis only.

**❌ Aggregate `confidence.overall` as MAX or AVG.** Default is MIN; deviation requires reasoning.

---

## Why This Schema, Not A Different One

Choices that look arbitrary but aren't:

- **YAML over JSON**: comments, multi-line strings, easier human scan during validation.
- **Separate partials per focus rather than one merged document with sections**: enables parallel reviewer dispatch with no shared-state coordination; merge is a single mechanical step.
- **`confidence` per focus, not just overall**: lets the synthesis layer say "interface is high-confidence but quality review was rushed" — important for prioritization downstream.
- **`provenance` block on canonical**: every claim is traceable to a specific partial and reviewer run. Required for the audit-trail expectation the pack already establishes.
- **No "notes" or "other" free-text field at top level**: any free-text field becomes a dumping ground; force findings into typed lists.
