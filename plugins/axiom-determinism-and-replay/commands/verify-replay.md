---
description: Verify a recorded run reproduces under replay. Re-runs the recorded inputs, walks the compare-points, asserts state equivalence per the declared determinism class, and emits a replay verification statement. On FAIL, dispatches the replay-debugger agent for first-differing-op localisation.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[run_record_path]"
---

# Verify Replay Command

You are verifying that a recorded run reproduces under replay. The output is a *replay verification statement* — what was re-run, which compare-points agreed, which diverged, and whether the run satisfies its declared determinism class.

This command does NOT design or scaffold. For design, use the `using-determinism-and-replay` skill. For scaffolding, use `/scaffold-replay-system`. For localising a divergence found by this command, this command can chain into `/diagnose-divergence` or dispatch the `replay-debugger` agent directly.

## Invocation Path

`/verify-replay` re-runs the recorded inputs of a run, computes the same compare-points, and compares against the recorded state hashes. On PASS, emits a verification statement. On FAIL or PARTIAL, may dispatch `replay-debugger` to localise the first-differing operation.

## Preconditions

The command takes a single argument: a path to a run record directory (containing the recorded inputs, snapshots, schedule trace, external-effects log, and the run's `99-` reference).

### Resolve the argument

```bash
INPUT="${ARGUMENTS}"

if [ -z "${INPUT}" ]; then
  # Use AskUserQuestion to collect:
  # "What is the path to the run record? It should contain:
  #    - the recorded run config (with seed),
  #    - snapshots at compare-points,
  #    - external-effects log (if `10-` is in scope),
  #    - schedule trace (if Strategy B from `07-` is in scope),
  #    - state-hash record at each compare-point."
  :
fi

if [ -d "${INPUT}" ]; then
  echo "Verifying run record: ${INPUT}"
else
  echo "ERROR: ${INPUT} not found or is not a directory."
  exit 1
fi
```

### Identify the corresponding spec

The verification rules come from the system's `99-determinism-and-replay-specification.md`. Locate it:

```bash
# Adjacent to the run record
ls "${INPUT}/../determinism-and-replay/" 2>/dev/null
ls determinism-and-replay/ 2>/dev/null

# Or asked from the user
```

If no spec is available, the verification proceeds in *spec-inferred* mode — the equivalence rule is inferred from the run record's state-hash format (bit-exact if hashes are byte-stable; logical-equivalence if there is a stored ε), and confidence is reduced. Unspecced verification is weaker than specced verification; flag this in the output.

## Workflow

### Step 1 — Frame the scope

Determine:

- **What is being re-run.** The full run, or a contiguous segment from compare-point M to compare-point N.
- **The class** declared in `01-` (or inferred). This sets the equivalence test (`==` for bit-exact; `np.allclose(atol=ε)` for logical).
- **The compare-point granularity** from `05-`. Per-tick, per-decision, per-snapshot.
- **External-effects mode**: replay must use `Effects.for_replay`, which reads from the recorded log (cross-link `10-`).
- **Schedule mode**: if Strategy B from `07-`, replay enforces the recorded schedule trace.
- **Code version**: the recorded run's `code_version` must match the current checkout. If not, abort or run in *cross-version* mode (which is informational, not class-checking).

### Step 2 — Reconstruct the replay environment

Build the replay configuration from the run record:

```bash
# Pseudocode
RUN_CONFIG=$(cat "${INPUT}/run-config.json")
SEED=$(jq .seed "${RUN_CONFIG}")
CODE_VERSION=$(jq .code_version "${RUN_CONFIG}")
CONFIG_HASH=$(jq .config_hash "${RUN_CONFIG}")

# Verify the current checkout matches
if [ "$(git rev-parse HEAD)" != "${CODE_VERSION}" ]; then
  echo "WARN: current code is not the recorded code_version."
  # Option: abort, or proceed in cross-version mode (informational only)
fi

# FP / GPU assertions per `08-`/`09-`
./scripts/assert_determinism_env.sh
```

If the FP/GPU assertion fails, the replay environment is not configured for the declared class; abort with a clear error.

### Step 3 — Run the replay

Invoke the system in replay mode. The system reads its inputs from `Effects.for_replay`, advances through compare-points, and emits a state hash at each. The replay produces a parallel run record (the *replay run*) of the same shape as the original (the *recorded run*).

Three terminating conditions:

1. The replay reaches the end of the recorded run.
2. The replay's compare-point hash differs from the recorded hash → divergence detected.
3. The replay's external-effects cursor advances out of sync (`10-`'s closed-world assertion fires) → divergence detected.

### Step 4 — Compare the two runs

For each compare-point:

```python
def compare_at(t: int, recorded: RunRecord, replay: RunRecord, class_: ClassSpec) -> CompareResult:
    a, b = recorded.state_hash[t], replay.state_hash[t]
    if class_.is_bit_exact:
        return CompareResult.PASS if a == b else CompareResult.FAIL
    elif class_.is_logical_equivalence:
        # Hashes encode quantised state; bit-equal → in-class
        if a == b:
            return CompareResult.PASS
        # If hashes differ, restore both states and compare with ε
        s_a = recorded.snapshot[t]
        s_b = replay.snapshot[t]
        return compare_with_tolerance(s_a, s_b, eps=class_.eps)
    elif class_.is_statistical:
        # Per-tick comparison is too tight; fall through to summary stats
        return CompareResult.PASS  # individual ticks not in-scope
```

Aggregate per-compare-point results into a verification result:

- **PASS**: every compare-point in scope satisfies the class.
- **PARTIAL**: most compare-points pass; some are within-class but flagged for review.
- **FAIL**: at least one compare-point violates the class.

### Step 5 — On FAIL, optionally dispatch replay-debugger

If the result is FAIL, the verification statement reports the divergence at the highest level. The user may want first-differing-op localisation. Two paths:

1. **Direct chain**: this command dispatches `replay-debugger` automatically (if a `--diagnose-on-fail` flag is set in the user's invocation).
2. **Chain via `/diagnose-divergence`**: the user runs `/diagnose-divergence ${INPUT}` after seeing the FAIL.

If dispatching `replay-debugger`:

```
Use the Task tool with subagent_type: "replay-debugger"
Provide: the recorded run record, the replay run record, the spec
Receive: divergence report — first-differing-op T₀, channel attribution, suggested next investigation
```

The replay-debugger's report is appended to the verification statement.

### Step 6 — Sign and emit the verification statement

Write the verification statement to `${INPUT}/replay-verification-YYYY-MM-DD.md`:

```markdown
# Replay Verification Statement

- **Verified**: <run record path>
- **Spec**: <99-spec reference, or "spec-inferred">
- **Verifier**: <this command + library versions + GPU SKU + driver>
- **Recorded code_version**: <git SHA>
- **Replay code_version**: <git SHA — same or different>
- **Class**: <01- citation: bit-exact / logical-equivalence with ε / statistical>
- **Scope**: compare-points from <first> to <last>, count = N
- **Result**: PASS | FAIL | PARTIAL

## Compare-Point Summary
- Compare-points evaluated: N
- Compare-points PASS: P
- Compare-points FAIL: F
- Compare-points within-tolerance (logical-equivalence only): W

## First Failure (if any)
- Compare-point T₀: <id>
- Recorded hash: <hex>
- Replay hash: <hex>
- (If snapshots available): field-level differences listed in `replay-debugger` report (appended)

## Environment
- FP config (`08-`): asserted
- GPU config (`09-`): asserted
- External-effects mode: replay (`10-`)
- Schedule mode (`07-`): <Strategy A | Strategy B with trace | Strategy C>

## Caveats
- <e.g., "spec-inferred mode; equivalence rule inferred from hash format">
- <e.g., "code_version differs from recorded; cross-version verification is informational only">
- <e.g., "compare-points coarser than per-tick; divergence may have appeared earlier than reported">

## Next Actions (if FAIL)
- Run `/diagnose-divergence ${INPUT}` to localise the first-differing operation.
- See the appended replay-debugger report (if `--diagnose-on-fail` was set).

## Result Statement
<plain-language summary suitable for the consumer of the verification — designer, on-call, CI bot>
```

### Step 7 — CI integration

`/verify-replay` is appropriate as a CI gate. The recorded run lives in the repo (or in a CI artifact store); the verifier replays on every PR (or on a schedule) and asserts PASS. A FAIL fails the build.

For multi-tier verification: the L+ tier may run `/verify-replay` on multiple GPU SKUs; the verification passes only if all replays pass at their respective SKUs (cross-link `09-`).

## Failure-Mode Handling

| Failure | Action |
|---------|--------|
| Code version mismatch (current ≠ recorded) | Abort with clear error, OR proceed in cross-version mode (informational only — does not verify the class) |
| FP/GPU assertion fails | Abort; the replay environment is not configured for the declared class |
| External-effects cursor desync | The system in replay made a different number of external reads than the recorded run; this is a divergence; report and offer `/diagnose-divergence` |
| Schedule trace incomplete (Strategy B) | Replay cannot enforce the recorded order; treat as a class violation; report |
| Snapshot at T₀ unreadable | Cannot do field-level comparison; report state-hash divergence only; lower confidence |
| Replay timeout | The replay is slower than recorded (possibly due to determinism mode); extend timeout or split scope |
| `99-` missing | Spec-inferred mode; lower confidence in result |
| Replay produces different output but recorded hashes "look fine" | The system's hash function is not stable; investigate `11-` (canonical encoding) |

## Output Location

Verification statements are written to the run record directory, with a date-stamped filename. They are not modified after writing; subsequent verifications produce new statements.

## Downstream Handoffs

- A FAIL or PARTIAL result triggers `/diagnose-divergence` (or, if `--diagnose-on-fail` was set, the chained `replay-debugger` report is already attached).
- A repeated FAIL across consecutive verifications is itself a finding — the system's claim of replay equivalence is not stable. Re-emit the affected numbered artifacts and re-gate.
- For routine CI verifications, the cadence (per `13-cost-of-determinism.md`) determines the next run.

## Scope Boundaries

Covered: replay execution, compare-point comparison, environment assertion, statement output, optional chain to `replay-debugger`.

Not covered: design changes (use the skill); scaffolding new replay infrastructure (`/scaffold-replay-system`); investigation of *why* the divergence occurred beyond first-differing-op localisation (downstream of the report).

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Verifying without asserting FP/GPU env vars | Always assert at Step 2; abort on mismatch |
| Verifying with code_version mismatch silently | Either abort or flag; never claim PASS for cross-version |
| Treating in-class drift as FAIL | Use `01-`'s ε; differences within ε are PASS |
| FAIL retried until PASS (rerun until lucky) | The FAIL is the result; retries are themselves an investigation |
| Verification statement unsigned | Sign with the verifier's identity; statements without provenance are noise |
| Spec-inferred mode treated as full verification | Lower confidence; explicit caveat in statement |
| Skipping the divergence localisation when FAIL | The first FAIL is rarely the only one; localise to find the channel |
| Verification run on a different GPU SKU than recorded | Cross-SKU is class-breaking unless `01-` declares logical-equivalence with appropriate ε; flag |
| Not propagating the verification statement to the run record | Statement IS the verification artefact; without it, the verification did not happen |
