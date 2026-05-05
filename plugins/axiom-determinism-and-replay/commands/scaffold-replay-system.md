---
description: Scaffold a replay-system implementation aligned to a declared determinism tier. Drops in record/replay loop boilerplate, an Effects substitution layer, snapshot envelope, divergence-detection compare-points, and CI hooks consistent with `axiom-determinism-and-replay` specs. Optionally runs a gap-analysis pass via the determinism-reviewer agent before scaffolding.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[component_name_or_path]"
---

# Scaffold Replay System Command

You are scaffolding a replay-system implementation for a component. The output is *implementation scaffolding* (code files, configuration, schema definitions) that align with a previously-produced or about-to-be-produced `determinism-and-replay/` specification. This command does NOT replace the design specs in the `using-determinism-and-replay` skill; it implements them.

## Invocation Path

`/scaffold-replay-system` is a Claude Code slash command. It dispatches the specialist sheets in `axiom-determinism-and-replay` to determine the right scaffolding shape, optionally calls the `determinism-reviewer` agent to find gaps before scaffolding, and emits skeleton code + configuration consistent with the spec set.

For a clean design pass without code, use the `using-determinism-and-replay` skill directly. For verification of a recorded run against a replay, use `/verify-replay`. For localising a live divergence, use `/diagnose-divergence`.

## Preconditions

The command takes a single argument: a component name (string) or a path to a component directory.

### Resolve the argument

```bash
INPUT="${ARGUMENTS}"

if [ -z "${INPUT}" ]; then
  # Use AskUserQuestion to collect:
  # "Which component or service is this replay system for?
  #  Provide a name (e.g., 'rl-substrate') or a path (e.g., 'services/training/')."
  :
fi

# Path → verify directory
if [ -d "${INPUT}" ]; then
  echo "Scaffolding into: ${INPUT}"
elif [ -f "${INPUT}" ]; then
  echo "ERROR: ${INPUT} is a file. Provide a directory or a component name."
  exit 1
else
  echo "Treating as component name: ${INPUT}"
  echo "Will create determinism-and-replay/${INPUT}/ for design specs."
fi
```

### Check for existing determinism-and-replay workspace

```bash
ls determinism-and-replay/ 2>/dev/null
```

If `determinism-and-replay/` exists with a tier-appropriate artifact set (00–13, 99) for this component, this command consumes those specs and emits scaffolding consistent with them. If specs are absent or incomplete, the command runs the design pass first via the `using-determinism-and-replay` skill.

**Resume vs fresh protocol:**

If `determinism-and-replay/<component>/` already has scaffolded files (source code, schema definitions, replay loop), ask the user via AskUserQuestion:

1. **Augment** — fill in missing scaffolding pieces, leave existing alone.
2. **Replace** — archive existing `determinism-and-replay/<component>/` to `determinism-and-replay/<component>.bak.YYYY-MM-DD/`, scaffold fresh.
3. **Targeted** — scaffold a single specific layer (Effects, snapshot, divergence detector, replay loop), leave others.

If no scaffolding exists, proceed without asking.

## Workflow

### Step 1 — Confirm or run the design pass

Check for the artifact set in `determinism-and-replay/<component>/`. The required set depends on tier (declared in `00-scope-and-goals.md` and `01-determinism-class.md`):

```
00-scope-and-goals.md                               (always)
01-determinism-class.md                             (always)
02-seed-governance-spec.md                          (always)
03-rng-isolation-spec.md                            (always)
04-snapshot-strategy.md                             (S+)
05-divergence-protocol.md                           (S+)
06-replay-infrastructure-spec.md                    (S+)
07-concurrency-determinism-spec.md                  (M+ if concurrent)
08-floating-point-policy.md                         (L+ if FP-heavy)
09-gpu-determinism-config.md                        (L+ if GPU)
10-external-effects-substitution.md                 (L+ if external IO)
11-canonical-state-encoding.md                      (XL, also L if cross-pack with audit)
12-property-test-suite.md                           (M+)
13-cost-of-determinism.md                           (always)
99-determinism-and-replay-specification.md
```

If any *required* artifact is missing for the declared tier:

- For tier XS or S, some artifacts are not required — check the Determinism Tier table in `using-determinism-and-replay/SKILL.md`.
- For required-but-missing artifacts, run the corresponding sheet from the catalog before scaffolding.
- Do NOT scaffold against an incomplete spec. The scaffold's choices come from the spec; absent specs make those choices arbitrarily.

### Step 2 — Optional gap analysis

Before scaffolding, dispatch the `determinism-reviewer` agent against the existing system design (HLD, code map, partial specs) to find determinism gaps:

```
Use the Task tool with subagent_type: "determinism-reviewer"
Provide: the component's design artifacts, the partial spec set, the declared tier
Receive: gap report — leaks per channel, severity-rated
```

Incorporate gap-report findings into the relevant numbered artifacts before scaffolding. Critical findings block scaffolding; high findings block per-channel scaffolding for that channel; medium and low findings inform but do not block.

### Step 3 — Scaffold per spec

Generate scaffolding consistent with the spec. Per-section scaffolding map:

| Spec section | Scaffolds into |
|--------------|----------------|
| `02-seed-governance-spec.md` | Seed-loading from config, `derive_seed(master, name)` helper, seed-audit script, fail-fast on missing seed |
| `03-rng-isolation-spec.md` | Per-component RNG factory, `WorkerRNGs` dataclass, hierarchical-seed derivation, RNG-audit grep script |
| `04-snapshot-strategy.md` | `Snapshot` dataclass, snapshot encoder/decoder, restore lifecycle hooks, lazy-field re-derivation hook |
| `05-divergence-protocol.md` | Compare-point hooks at each granularity, state-hash function, bisection script for offline runs |
| `06-replay-infrastructure-spec.md` | Read-only replay loop, branching replay primitives (if in scope), replay-mode flag plumbing, lifecycle state machine |
| `07-concurrency-determinism-spec.md` | Strategy A: lockstep barrier; Strategy B: schedule-trace recorder/replayer wiring; Strategy C: sorted-iteration helpers and idempotency wrappers |
| `08-floating-point-policy.md` | Process-start FP config: `OMP_NUM_THREADS`, `set_flush_denormal`, BLAS pinning; CI assertion script |
| `09-gpu-determinism-config.md` | Process-start GPU config: `use_deterministic_algorithms`, `cudnn.deterministic`, `CUBLAS_WORKSPACE_CONFIG`, NCCL env vars; assertion script |
| `10-external-effects-substitution.md` | `Effects` dataclass with `for_record`/`for_replay` constructors, per-effect recorders/replayers, closed-world manifest writer |
| `11-canonical-state-encoding.md` | Canonical-encoding wrapper for the snapshot envelope, canonical-tensor-bytes helper, JCS library binding (cross-link audit pack) |
| `12-property-test-suite.md` | Hypothesis test stubs for each required property (replay equivalence, seed isolation, snapshot round-trip, restore idempotence, fork-and-converge, schedule independence) |
| `13-cost-of-determinism.md` | Read-only — informs choices; no code emitted |

### Step 4 — Wire integrations

Wire the scaffolding into the component:

- The system's main loop reads `Effects.for_record` (record mode) or `Effects.for_replay` (replay mode) at startup; the mode is in the run config.
- Every component receives its RNG via `WorkerRNGs` rather than constructing its own.
- The compare-point hook is called at the granularity declared in `05-` (typically per-tick or per-decision); the hash is appended to the run record.
- Snapshot writes use the canonical encoder; reads use the decoder; lazy-field re-derivation runs after the structural restore.
- For concurrent systems: the chosen strategy's primitives wrap the scheduler boundaries.

### Step 5 — CI hooks

Generate CI configuration:

- **FP / GPU assertions**: a process-start script reads the FP and GPU env vars and asserts they match `08-`/`09-`. Failure aborts the test run.
- **Seed audit**: grep-based audit script (per `02-`) runs on every PR; orphan RNGs fail the build.
- **External-effects audit**: grep-based audit script (per `10-`) runs on every PR; external reads outside the Effects layer fail the build.
- **Property tests**: `12-`'s test suite runs on every PR; failing seeds are persisted to a checked-in `.hypothesis` directory.
- **Test vectors**: the recorded `(seed, code_version) → state_hash` triples from each numbered spec run in CI; mismatches abort.
- **Canonical-encoding test vectors**: RFC 8785 vectors from `11-` run on every PR.

### Step 6 — Run the consistency gate

Invoke the consistency-gate procedure from `using-determinism-and-replay/SKILL.md`. Each check produces a pass/fail in the gate report. Failures are addressed before declaring scaffolding complete; this command does NOT bypass the gate.

If `99-determinism-and-replay-specification.md` does not exist yet, generate a draft from the numbered artifacts; the user reviews and signs off before the gate is final.

## Output Location

Specs land in `determinism-and-replay/<component>/`. Code scaffolding lands in the component's existing source tree at the appropriate path (`src/replay/`, `lib/determinism/`, or per the component's convention). Configuration lands wherever the component holds configuration (a `replay-config.toml`, environment-variable schema, etc.).

## Downstream Handoffs (suggest after completion)

- Verification — `/verify-replay` against the scaffolded system once it has produced a recorded run.
- Divergence diagnosis — `/diagnose-divergence` if a recorded run and a replay disagree.
- Class promotion — when the system needs cross-machine bit-exactness, re-run the design pass at L tier and re-scaffold the new sections (`08-`, `09-`, `10-`, `11-`).
- Cross-pack — if the system also has audit-trail obligations, run `/scaffold-audit-trail` (axiom-audit-pipelines); the canonical-encoding rule is shared via `11-`.

## Scope Boundaries

Covered: design pass through the spec, optional gap analysis, code/config scaffolding, CI hooks, consistency gate.

Not covered: production deployment, GPU driver / CUDA installation provisioning (typically owned by ops), choice of cloud provider beyond what the spec declares, performance tuning beyond the determinism-required pinning.

## Common Mistakes (in scaffolding)

| Mistake | Fix |
|---------|-----|
| Scaffolding generated before specs are complete | Halt; complete the spec; come back |
| Scaffolding hard-codes choices not in spec | Choices come from spec; if undocumented, the spec is incomplete |
| Effects layer's `for_live_with_logging` constructor introduced | Forbidden; record + replay are the only modes |
| Skipping `13-cost-of-determinism.md` because "it's just docs" | Cost record is a deliverable; future relaxations require it |
| Choosing a tier higher than the system needs | Tier is a cost-calibration; oversize tiering pays L+ costs for XS+ value |
| Property tests scaffolded but database not checked in | `.hypothesis` directory committed; failing seeds reproducible |
| FP / GPU env-var assertion missing from CI | Without assertion, env-var drift is silent; assert at process start |
| Replay loop and record loop are different code paths | Single code path with `Effects` mode flag; divergence between modes is a bug |
| Compare-point hash function chosen ad hoc per subsystem | One hash function per `05-`; subsystems may contribute substructure but the top-level function is one |
| `13-` written but not consulted when relaxing rules | Relaxations re-emit `13-` and re-gate |

## Cross-Pack Notes

- `axiom-audit-pipelines:scaffold-audit-trail` — if the system needs both replay and audit, run both commands; they share canonical encoding via `11-` and `02-canonical-encoding-spec.md` (audit).
- `axiom-static-analysis-engineering` — many of the audit scripts emitted in Step 5 can be encoded as static-analysis rules in that pack; the spec's rules become enforceable in CI.
- `yzmir-simulation-foundations:check-determinism` — the static-pattern scanner; emits a regression check the scaffolded CI can run against the new system.
