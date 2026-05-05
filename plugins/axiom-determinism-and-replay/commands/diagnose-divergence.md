---
description: Localise a divergence between two runs (or between a recorded run and a replay) to the first differing operation. Walks the divergence protocol from `05-divergence-detection-and-localisation.md`, performs binary-search bisection, attributes the divergence to a channel, and emits a diagnosis report. Dispatches the replay-debugger agent for the actual localisation work.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[run_a_path] [run_b_path]"
---

# Diagnose Divergence Command

You are diagnosing a divergence between two runs. The output is a *divergence diagnosis report* — the first-differing compare-point T₀, the field-level differences at T₀, the channel attribution, and the next investigation step.

This command does NOT design or scaffold or implement fixes. For design, use the `using-determinism-and-replay` skill. For scaffolding, use `/scaffold-replay-system`. For verifying that a recorded run reproduces, use `/verify-replay` (which can chain into this command on FAIL). For implementing the fix, the report's "Next Actions" section names the responsible artifact.

## Invocation Path

`/diagnose-divergence` resolves the two run paths, frames the bisection, dispatches the `replay-debugger` agent for localisation and attribution, and emits a diagnosis report.

## Preconditions

The command takes one or two arguments: paths to two run records, OR a single path with mode determined from context.

### Resolve the arguments

```bash
RUN_A="${1:-}"
RUN_B="${2:-}"

if [ -z "${RUN_A}" ]; then
  # Use AskUserQuestion to collect:
  # "Which two runs are diverging? Provide:
  #    - two run record paths (parallel runs), OR
  #    - one run record path and the replay log (for record-vs-replay)."
  :
fi

if [ -z "${RUN_B}" ]; then
  # Treat single argument as a record-and-replay pair if the directory contains both.
  if [ -f "${RUN_A}/recorded-state-hashes.json" ] && [ -f "${RUN_A}/replay-state-hashes.json" ]; then
    echo "Mode: record-vs-replay"
    MODE="record-vs-replay"
  else
    echo "ERROR: Single argument provided but does not contain both recorded and replay logs."
    exit 1
  fi
else
  echo "Mode: parallel-runs"
  MODE="parallel-runs"
fi

if [ ! -d "${RUN_A}" ]; then
  echo "ERROR: ${RUN_A} not found."
  exit 1
fi
if [ -n "${RUN_B}" ] && [ ! -d "${RUN_B}" ]; then
  echo "ERROR: ${RUN_B} not found."
  exit 1
fi
```

### Identify the corresponding spec

The diagnosis rules come from the system's `99-determinism-and-replay-specification.md`. Locate it:

```bash
# Adjacent to the runs
ls "${RUN_A}/../determinism-and-replay/" 2>/dev/null
ls determinism-and-replay/ 2>/dev/null
```

If no spec is available, the agent operates in *spec-inferred* mode (lower confidence; class is inferred from the run records' state-hash formats).

## Workflow

### Step 1 — Frame the scope

Determine:

- **Mode** — parallel-runs (two live runs) or record-vs-replay.
- **Class** declared in `01-` (or inferred). Sets equivalence test (bit-exact uses `==`; logical uses `np.allclose(atol=ε)`).
- **Compare-point granularity** from `05-`. The bisection unit (tick / decision / snapshot).
- **What is missing**: snapshots at every compare-point (required for field-level comparison at T₀), schedule trace (required for Strategy B), external-effects log (required if `10-` is in scope).

If the runs are at different `code_version`s, flag prominently: a divergence between two code versions is not (necessarily) a determinism bug — it may be a behaviour change. The agent reports it as a finding but classification differs.

### Step 2 — Confirm the divergence is real

Before bisecting, sanity-check:

```bash
# Are the runs the same code, same seed, same config?
diff <(jq .seed,.code_version,.config_hash "${RUN_A}/run-config.json") \
     <(jq .seed,.code_version,.config_hash "${RUN_B}/run-config.json")

# Are their final state hashes actually different?
RUN_A_FINAL=$(jq -r '.[-1]' "${RUN_A}/state-hashes.json")
RUN_B_FINAL=$(jq -r '.[-1]' "${RUN_B}/state-hashes.json")
[ "${RUN_A_FINAL}" = "${RUN_B_FINAL}" ] && echo "Final hashes match — runs do not diverge."
```

If the seeds, code versions, or config hashes differ, the runs are *not the same input*; report this and stop. Replay equivalence is a property of same-input runs.

### Step 3 — Dispatch the replay-debugger agent

```
Use the Task tool with subagent_type: "replay-debugger"
Provide: 
  - the two run record paths (or recorded + replay)
  - the spec (`99-`) or "spec-inferred mode"
  - the declared class and ε
  - the compare-point granularity
  - the schedule trace and external-effects log if available
Receive:
  - divergence report — T₀, field-level differences, channel attribution, confidence, next investigation
```

The agent does the bisection, structured comparison, and channel attribution. This command resolves inputs, frames the scope, and consolidates output.

### Step 4 — Consolidate the diagnosis report

Write the diagnosis report to `${RUN_A}/divergence-diagnosis-YYYY-MM-DD.md` (or to a working directory if both runs are read-only):

```markdown
# Divergence Diagnosis Report

- **Diagnosed**: <run A path> vs <run B path>
- **Mode**: parallel-runs | record-vs-replay
- **Spec**: <99-spec reference, or "spec-inferred">
- **Class**: <01- citation: bit-exact / logical-equivalence with ε / statistical>
- **Result**: LOCALISED at T₀=<id> | UNLOCALISED | NON-REPRODUCING | NOT-A-DIVERGENCE

## Localisation
[Output from the replay-debugger agent's First-Differing Compare-Point section]

## Field-Level Differences at T₀
[Output from the replay-debugger agent's Field-Level Differences section]

## Channel Attribution
[Output from the replay-debugger agent's Channel Attribution section]

## Pre-T₀ Audit
[Output from the replay-debugger agent's Pre-T₀ Audit section]

## Next Actions
- **Most likely fix location**: <numbered artifact + sheet section>
- **Validation**: <how to confirm the fix closes T₀'s divergence>
- **Regression**: <which property test from `12-` should now catch this if it recurs>
- **Re-gate**: <which consistency-gate checks must re-run after the fix>

## Confidence Assessment
[Output from the replay-debugger agent's Confidence Assessment section]

## Risk Assessment
[Output from the replay-debugger agent's Risk Assessment section]

## Information Gaps
[Output from the replay-debugger agent's Information Gaps section]

## Caveats
- This diagnosis localises and attributes; it does not patch.
- Channel attribution is a hypothesis ranked by evidence; investigation is required to confirm.
- A non-reproducing divergence is itself a finding — the system's claim of replay equivalence is flaky.
- This report covers the bytes provided. Other runs of the same system may diverge through other channels.

## Result Statement
<plain-language summary, 1-3 sentences, suitable for designer / on-call / project lead>
```

### Step 5 — Suggest follow-ups

Based on the result:

- **LOCALISED**: the report names the channel; suggest reading the resolving sheet, updating the relevant numbered artifact, then re-running `/verify-replay` to confirm the fix.
- **UNLOCALISED**: the bisection bottomed out at a snapshot interval rather than a single compare-point. Suggest re-running with finer-grained logging within the interval, or accepting the granularity as a known limitation of `05-`'s compare-point cadence.
- **NON-REPRODUCING**: the divergence appeared once and not again. This is itself a class-breaking finding (the class promises *every* run at the same input agrees). Suggest property testing (`12-`) to surface the seed at which it reappears.
- **NOT-A-DIVERGENCE**: the differences are within `01-`'s ε. The original detection threshold is too tight; suggest revising `05-`'s comparison rule.

## Failure-Mode Handling

| Failure | Action |
|---------|--------|
| `99-` missing | Spec-inferred mode; lower confidence in attribution |
| Snapshots missing at compare-points | Bisection localises to interval, not point; coarser report |
| Schedule trace missing for Strategy B | Cannot confirm or rule out concurrency channel; flag |
| External-effects log missing | Cannot confirm or rule out external channel; flag |
| Code versions differ | Not a same-input divergence; report and stop, OR proceed in cross-version mode (informational only) |
| Both runs at different `code_version` (dirty tree marker) | Local diff is itself a candidate cause; flag prominently |
| Bisection cost prohibitive (massive runs) | Approximate by sampling compare-points; document the loss of precision |
| The claimed divergence is within ε | Not a divergence; revise `05-`'s detection threshold; report as NOT-A-DIVERGENCE |
| Multiple plausible channels at T₀ | Report all in likelihood order; suggest cheap experiments to discriminate |

## Output Location

Diagnosis reports are written to the working directory (often the run record's parent), with a date-stamped filename. They are not modified after writing; subsequent diagnoses produce new reports.

## Downstream Handoffs

- A LOCALISED report's "Next Actions" names the resolving sheet; the designer reads it, updates the affected numbered artifact, and re-emits `99-`.
- After a fix, re-run `/verify-replay` to confirm the divergence no longer reproduces.
- If the fix changes the class or breaks the consistency gate, re-emit `99-` and re-gate.
- If the divergence falsifies a class promise that the system depends on, the cost-budget-breach response from `13-cost-of-determinism.md` applies (relax the class, scope the system down, or scope the perf budget down).

## Scope Boundaries

Covered: bisection, structured comparison, channel attribution, diagnosis report, suggested next actions.

Not covered: implementing the fix (downstream); reviewing the design for *other* potential leaks (use `determinism-reviewer`); investigating non-reproducing divergences beyond a single property-test sweep (separate workflow).

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Diagnosing without confirming the divergence is real (same-input) | Step 2 first; if seeds/code/config differ, the divergence is not a class violation |
| Reporting symptom tick (T_max) as the bug | The bug is at T₀; everything after is propagation |
| Single channel attribution when multiple are plausible | Report all in likelihood order with reasoning |
| Treating in-class drift as FAIL | Use `01-`'s ε; differences within ε are NOT-A-DIVERGENCE |
| Bisection without snapshots → assuming arbitrary precision | The bisection can only resolve to compare-point granularity; coarser cadence = coarser report |
| Diagnosis run on a different machine than the runs were produced on | If the recorded runs are from machine X and diagnosis runs on machine Y, machine Y's load and state may interfere; run diagnosis on the same machine when possible |
| Treating non-reproducing divergence as a flake | Non-reproducing = the class promise is itself flaky; it is the worst-case finding |
| Skipping the field-level comparison | Field-level is what enables channel attribution; without it the report is "they differ" |
| Re-running `/verify-replay` immediately after a fix without re-gating | Some fixes are class-breaking (re-emit affected sheets and re-gate before re-verifying) |
| Diagnosis report not signed | Provenance matters; future re-investigations want to know who diagnosed and when |
