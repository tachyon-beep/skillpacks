---
description: Scaffold a static-analyzer implementation aligned to a declared analyzer tier and design specs. Drops in AST visitor / walker shell, lattice-aware rule registry, plugin loader, finding emitter (SARIF + native), manifest reader, and CI integration hooks consistent with the `axiom-static-analysis-engineering` discipline. Optionally runs a gap-analysis pass via the rule-designer agent before scaffolding. Cross-validates layout against the `using-static-analysis-engineering` artifact set (`01–06`, plus `07–13` where the tier requires).
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[analyzer_name_or_path]"
---

# Scaffold Analyzer Command

You are scaffolding a static analyzer aligned to the `axiom-static-analysis-engineering` discipline. The output is *implementation scaffolding* (Python modules, registry plumbing, manifest reader, finding emitter, CI hooks) consistent with a previously-produced `analyzer-engineering/` specification. This command does NOT replace the design discipline; it implements it.

## Invocation Path

`/scaffold-analyzer` is a Claude Code slash command. It assumes (or detects) an `analyzer-engineering/` workspace from the `using-static-analysis-engineering` skill, scaffolds the engine code that satisfies its specs, and wires the analyzer into the CI gate the manifest declares.

Use `using-static-analysis-engineering` directly for the design pass without code. Use `/design-tier-model` to settle the lattice's tier set before scaffolding. Use `/design-rule-set` to bootstrap an initial rule set against the scaffolded engine.

## Preconditions

The command takes a single optional argument: an analyzer name (string) or a path to a target directory.

### Resolve the argument

```bash
INPUT="${ARGUMENTS}"

if [ -z "${INPUT}" ]; then
  # Use AskUserQuestion to collect:
  # "What is the analyzer name? (Will be created at ./<name>/.)
  #  Or provide a path to an existing directory to scaffold into."
  :
fi

if [ -d "${INPUT}" ]; then
  TARGET_DIR="${INPUT}"
elif [ -f "${INPUT}" ]; then
  echo "ERROR: ${INPUT} is a file. Provide a directory or an analyzer name."
  exit 1
else
  TARGET_DIR="./${INPUT}"
fi
```

### Check for the design artifact set

```bash
ls analyzer-engineering/ 2>/dev/null
```

The required artifact set depends on the analyzer's declared tier (in `00-scope-and-targets.md`):

```
00-scope-and-targets.md                              (always)
01-visitation-strategy.md                            (always)
02-abstract-domain-spec.md                           (always)
03-inference-pipeline-spec.md                        (always)
04-rule-plugin-spec.md                               (always)
05-false-positive-economics.md                       (always)
06-static-runtime-boundary.md                        (always)

07-callgraph-construction.md                         (M+ when dataflow involved)
08-cross-module-flow.md                              (L+ when stubs needed)
09-decorator-as-assertion-spec.md                    (when decorators-as-assertion in scope)
10-manifest-and-coherence.md                         (M+ for non-trivial configuration)
11-sarif-and-ci.md                                   (always when CI-integrated)
12-scaling-and-incrementality.md                     (L+ for codebases > ~100k LoC)
13-llm-assisted-explanation.md                       (optional; when enrichment is in scope)

99-analyzer-engineering-specification.md             (always — consolidated; gate-passed)
```

If the artifact set is absent or incomplete, the command halts and instructs the user to run the design pass via `using-static-analysis-engineering` first. Do not scaffold against a missing or stale spec set — the engine will not match what the spec promises and the consistency gate will fail.

If `99-analyzer-engineering-specification.md` is older than the latest numbered artifact, the spec set is **stale**. Halt and instruct re-gate before scaffolding.

### Check for existing scaffolding

```bash
ls "${TARGET_DIR}/src" "${TARGET_DIR}/pyproject.toml" 2>/dev/null
```

If either exists, this is a **brownfield** scaffold. Use AskUserQuestion to choose:

1. **Augment** — fill in missing pieces; leave existing files (with `.scaffold-suggested` siblings where new content is proposed).
2. **Replace** — archive existing files to `.backup-<timestamp>/`; scaffold fresh.
3. **Validate only** — skip scaffolding; instead spot-check the existing layout against the spec set and emit a gap report.

## Workflow

### Step 1 — Confirm the design specs

Read the artifact set. Extract:

- Tier (from `00-scope-and-targets.md`).
- Visitation strategy (from `01-`): visitor, walker, or transformer.
- Lattice shape (from `02-`): tier set, height, monotonicity claims.
- Inference phasing (from `03-`): which phases are implemented; whole-program vs incremental.
- Plugin discovery (from `04-`): decorator registry, entry points, manifest, or hybrid.
- Suppression mechanism (from `05-`): inline, manifest, both.
- Callgraph rung (from `07-` if present): 0–4.
- Stub library (from `08-` if present): in-tree or external.
- Decorator-as-assertion entries (from `09-` if present): registry of recognised decorators.
- Manifest structure (from `10-` if present): schema version, layered or flat.
- SARIF emission (from `11-`): always; consumer set; gate matrix.
- Incrementality (from `12-` if present): cache layout, reverse index.
- LLM enrichment (from `13-` if present): which rule classes get enrichment.

If any required field is missing, halt and request clarification rather than guessing.

### Step 2 — Optional gap-analysis pre-pass

For brownfield scaffolds, dispatch the `rule-designer` agent (or the `false-positive-analyst` agent for an existing analyzer with a suppression set) for a gap report:

```
Task(subagent_type="rule-designer",
     description="Pre-scaffold gap analysis",
     prompt="Read analyzer-engineering/ specs and the existing source at ${TARGET_DIR}. Report deviations from the spec set with severity and the artifact that closes each gap. Under 600 words.")
```

Use the report to inform whether to Augment, Replace, or Validate.

### Step 3 — Scaffold the directory layout

Standard layout (Python; adjust per language):

```
<analyzer>/
  pyproject.toml
  src/
    <analyzer>/
      __init__.py
      __main__.py                       # CLI entrypoint
      ast_visitor.py                    # 01- visitation
      lattice.py                        # 02- abstract domain
      inference/
        phase1_variable.py              # 03- intra-procedural
        phase2_summary.py               # 03- function summaries
        phase3_callgraph.py             # 03- inter-procedural
        callgraph.py                    # 07- callgraph construction
      stubs/                            # 08- cross-module
        __init__.py
        stdlib/
        thirdparty/
      decorators/                       # 09- recognition registry
        registry.py
      rules/                            # 04- plugin registry + built-in rules
        __init__.py
        registry.py
        builtin/
      manifest/                         # 10- manifest reader + validator
        __init__.py
        loader.py
        coherence.py
        drift.py
      emit/                             # 11- output formats
        sarif.py
        console.py
        lsp.py                          # for IDE integration
      cache/                            # 12- caches + reverse index
        __init__.py
        keys.py
        store.py
        reverse_index.py
      enrich/                           # 13- LLM enrichment (if scoped)
        __init__.py
        prompt.py
        review_gate.py
  tests/
    fixtures/
      tp/                               # rules' test corpus
      tn/
    test_phase1.py
    test_phase2.py
    test_phase3.py
    test_rules/
    test_manifest.py
    test_sarif.py
  ci/
    github-codescanning.yml             # SARIF upload action
  manifest-default.yaml                 # default manifest
  README.md
```

Files that are required (hard scaffold):

- `__main__.py` — CLI entrypoint with subcommands `analyse`, `rules`, `config show --effective`, `config diff`, `config dry-run`.
- `lattice.py` — encodes the tier set from `02-`; the partial order, join, monotonicity assertions; finite-height assertion.
- `ast_visitor.py` — the visitation strategy from `01-`.
- `inference/phase1_variable.py`, `phase2_summary.py`, `phase3_callgraph.py` — phase scaffolds with worklist algorithms; termination assertions reference `02-` and `03-`.
- `rules/registry.py` — the plugin lifecycle from `04-` (load → validate → enable → fire → emit → unload).
- `manifest/loader.py` — reads layered manifests; `coherence.py` runs the coherence pass; `drift.py` runs the drift pass.
- `emit/sarif.py` — SARIF 2.1.0 emission from `11-`; rule descriptors; partialFingerprints.
- `cache/keys.py` — cache key composition from `12-` (analyzer + lattice + ruleset + stub-lib + manifest + body-canonical + transitive).
- A skeleton `pyproject.toml` with `[project.scripts]` registering the CLI.

Files that are tier-conditional:

- `inference/callgraph.py` — for tier M+ with `07-`.
- `stubs/` — for tier L+ with `08-`.
- `decorators/registry.py` — when `09-` is present.
- `cache/store.py` and `cache/reverse_index.py` — for tier L+ with `12-`.
- `enrich/` — when `13-` is present.

### Step 4 — Wire the consistency gate

After scaffolding, generate `analyzer-engineering/consistency-gate.md` (or update if present) with the gate's checks (1–11 from the router's Consistency Gate table) and which scaffolded files satisfy each check. Files without a satisfying scaffold are noted explicitly — these become the work items for the user.

### Step 5 — Wire CI

Drop in `ci/github-codescanning.yml` (or platform-specific equivalent based on consumer-set in `11-`). The CI workflow:

1. Installs the analyzer.
2. Runs `<analyzer> analyse --format=sarif --output=findings.sarif`.
3. Uploads SARIF via `github/codeql-action/upload-sarif@v3` (or platform analogue).
4. Gates the build per `11-`'s exit-code matrix.

Include the resolution-rate / stub-coverage / suppression-rate threshold gates as separate steps using the analyzer's `--check` subcommands.

### Step 6 — Smoke test

After scaffolding, run:

```bash
cd "${TARGET_DIR}" && python -m <analyzer> --version && python -m <analyzer> analyse tests/fixtures/tp/STA001.py --format=sarif | jq '.runs[0].results | length'
```

If this fails, the scaffold has a structural problem; do not declare success.

## Output

Report (terse):

- **Tier detected** and **artifact set status** (complete / incomplete / stale).
- **Scaffold mode** (greenfield / augment / replace / validate-only).
- **Files created / modified** (count by category).
- **Gate file generated** (`consistency-gate.md`) and its current pass/fail summary.
- **Next steps** — typically `/design-rule-set` to bootstrap a starting rule set; or run the smoke-test fixture and iterate.

If any precondition failed, the report is the precondition failure and instructions to remediate. Do not partial-scaffold with known gaps.

## Cross-References

- Router: `using-static-analysis-engineering` — design pass; produces the artifact set this command consumes
- `/design-tier-model` — settle the lattice tier set before scaffolding
- `/design-rule-set` — bootstrap initial rules after scaffolding
- Agent: `rule-designer` — pre-pass gap analysis on brownfield scaffolds
- Agent: `false-positive-analyst` — pre-pass review of an existing analyzer's suppression set before retrofit
- Cross-pack: `axiom-rust-engineering` / `axiom-python-engineering` — implementation discipline for the chosen language
