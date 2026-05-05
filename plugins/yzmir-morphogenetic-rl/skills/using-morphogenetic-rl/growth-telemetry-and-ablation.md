---
name: growth-telemetry-and-ablation
description: Use when designing the logging schema that survives shape changes — additivity over topology change, ablation-friendly fields (so a grown vs static comparison is meaningful), and the metrics that distinguish controller signal from network signal.
---

# Growth Telemetry and Ablation

## When to Use

- Designing the logging layer for a morphogenetic experiment so ablations remain valid after the network's shape changes
- Existing logs break when growth events fire (column count changes, schemas drift, dashboards crash)
- Trying to attribute a result to a specific controller decision and the data does not support the question
- Setting up dashboards that must survive arbitrarily many topology changes
- Preparing a research substrate where ablations across reward modes / gate policies / slot counts must be comparable

For general ML telemetry patterns, see `yzmir-ml-production`. This sheet covers the *additional* discipline morphogenesis imposes: schemas that do not break when shape changes, decision logs that allow ablation, and the difference between event-grain and step-grain telemetry.

---

## Core Principle

**A morphogenetic system has two telemetry grains, and conflating them destroys ablation.**

- **Step-grain**: per-host-step quantities — loss, gradient norm, per-batch accuracy. Aggregable across runs in the normal way.
- **Event-grain**: per-controller-decision quantities — proposed action, governor verdict, pre-event window, post-event watch outcome. **Not** aggregable across runs without alignment.

If you log event-grain data into the same table as step-grain data, joining the two later requires reconstructing event boundaries from log timestamps. That reconstruction is the bug. Keep them separate, anchor them with `event_id`, and join only when you ablate.

The other principle:

**Schemas must be additive across topology change. They must never widen step-row width when the network grows.**

A naive design logs one column per parameter group, so growing the network adds columns. Now your downstream tooling — dashboards, ablation scripts, comparison plots — silently breaks because runs have incompatible row widths. Don't do this.

---

## The Two Tables

A morphogenetic run produces two structured streams:

### Step Table

One row per host-trainer step. Width is **constant for the run's lifetime**.

| Column | Type | Notes |
|--------|------|-------|
| `run_id` | str | UUID per experiment |
| `step` | int | Host-trainer step counter |
| `loss` | float | Aggregated loss for the step (global if distributed) |
| `grad_norm` | float | Pre-clip global gradient norm |
| `grad_norm_clipped` | float | Post-clip |
| `lr` | float | Current learning rate (host trainer) |
| `param_count` | int | Total parameter count *as of this step* |
| `flops_per_step` | int | Approx FLOPs (recompute when topology changes) |
| `active_module_count` | int | Number of currently-active modules |
| `pending_event_count` | int | Governor watch windows currently open |

Topology changes update `param_count`, `flops_per_step`, `active_module_count` — but the *schema* does not change.

### Event Table

One row per controller decision **and** one row per governor verdict. Width is **constant**.

| Column | Type | Notes |
|--------|------|-------|
| `run_id` | str | Same as step table |
| `event_id` | int | Monotonic per run |
| `step` | int | Step at which the event happened |
| `kind` | enum | `proposal`, `approve`, `veto`, `commit`, `rollback` |
| `slot_id` | str | Action target |
| `action_type` | enum | `grow`, `graft`, `retire`, `noop`, etc. |
| `action_params` | json | Action-specific structured payload |
| `controller_log_prob` | float | For off-policy correction; null for governor verdicts |
| `governor_reason` | str | Structured reason code; null for proposals |
| `pre_event_window_hash` | str | sha256 of frozen pre-event window (for replay) |
| `post_event_window_summary` | json | Median/p99 loss + grad_norm during watch; null until window closes |
| `delta_param_count` | int | Change in parameter count caused by this event (±) |

The `action_params` JSON is **not** the place to widen the schema. Keep its keys structured per `action_type` and document them. Downstream code parses the JSON; the table itself remains additive.

### Join Pattern

```sql
-- Loss trajectory around event 17, ±200 steps
SELECT s.step, s.loss, e.kind, e.action_type, e.governor_reason
FROM steps s
LEFT JOIN events e USING (run_id, step)
WHERE s.run_id = ?
  AND s.step BETWEEN (SELECT step FROM events WHERE event_id = 17) - 200
                 AND (SELECT step FROM events WHERE event_id = 17) + 200
ORDER BY s.step;
```

The event table never widens. The step table never widens. Joins are by `(run_id, step)`. Ablation queries become tractable.

---

## What to Log per Event

Beyond the schema columns, an event row's `action_params` and `post_event_window_summary` should record:

For a **proposal/approve/commit**:

```json
{
  "slot_id": "trunk.layer3.slot_2",
  "action_type": "grow",
  "module_kind": "linear_block",
  "init_strategy": "near_zero",
  "expected_param_delta": 8192,
  "controller_features_hash": "sha256:..."
}
```

For a **veto**:

```json
{
  "slot_id": "trunk.layer3.slot_2",
  "action_type": "grow",
  "veto_reason": "cooldown",
  "cooldown_remaining_steps": 423
}
```

For a **rollback**:

```json
{
  "slot_id": "trunk.layer3.slot_2",
  "action_type": "grow",
  "rollback_reason": "loss_spike",
  "panic_rule": "loss_spike",
  "k_threshold": 8.0,
  "observed_value": 12.7,
  "checkpoint_id": "ckpt_step_4000"
}
```

The `controller_features_hash` in the proposal row is essential for ablation. It lets you ask "for two policies trained with different reward functions, are their action features actually different on the same observations?" — without storing the full features per event.

---

## Per-Module Telemetry: Sidecar, Not Schema Widening

You will want per-module statistics — gradient norm per module, activation magnitude, dead-neuron rate. These cannot live in the step table because the module count changes.

The right pattern is a **sidecar table** keyed by module:

| Column | Type |
|--------|------|
| `run_id` | str |
| `step` | int |
| `module_id` | str |
| `module_kind` | enum |
| `created_at_step` | int |
| `grad_norm` | float |
| `output_l2` | float |
| `dead_unit_fraction` | float |

This table is **long-format**. It has a constant width but variable row count per step. Modules that exist only briefly produce few rows; long-lived modules produce many. Aggregate across runs by `module_kind`, not `module_id`.

A common temptation is to put per-module stats as columns in the step table — `grad_norm.module_3`, `grad_norm.module_4`, etc. Resist. The first growth event invalidates every downstream join, every dashboard, and every ablation script.

---

## Ablation-Friendly Schemas

An ablation answers questions like "does removing event #17 change the result?" or "across these five reward modes, which growth events were unique to mode B?"

Three schema decisions make this tractable:

### 1. Reward Mode Is a First-Class Column

Every step row and every event row carries a `reward_mode` (or whatever ablation axis is varied). Two runs with different reward modes can be compared directly — the schemas align, and `WHERE reward_mode = 'X'` works.

### 2. Action Features Are Hashed, Not Inlined

When the controller's feature vector enters the table only as a hash, you can group "events where the controller saw the same input" across runs. If you need to introspect the features for a specific event, use the replay log — that is its job.

### 3. Watch Windows Resolve Before Aggregation

A pending event has no `post_event_window_summary` yet. Aggregation queries should filter `WHERE post_event_window_summary IS NOT NULL` or risk averaging in a `null`. A common bug: a still-pending event at the end of a run silently distorts the final ablation summary. Either close the window at run-end (commit or rollback explicitly) or filter it out.

### Counterfactual Joins

The most useful ablation query is "the same controller, same data, same seed, but with action X replaced by no-op." This is a counterfactual replay, logged into a sibling run. The schemas align trivially because they were designed to.

```python
# Compare a real run with a counterfactual that skips event 17
real = load_run("run_001")
cf   = load_run("run_001_cf_skip_17")

# Loss divergence after the skip point
divergence = (cf.steps.loss - real.steps.loss).rolling(window=50).mean()
```

If you can write this query in two lines, your telemetry is healthy. If it takes 50 lines of column reconciliation, your schemas widened.

---

## Aggregating Across Runs

Once schemas are additive, common aggregations work:

```sql
-- Median loss trajectory across 10 seeds, with growth-event markers
SELECT step, percentile_cont(0.5) WITHIN GROUP (ORDER BY loss) AS median_loss
FROM steps
WHERE run_id IN (SELECT run_id FROM runs WHERE reward_mode = 'utility_minus_cost')
GROUP BY step
ORDER BY step;
```

```sql
-- Rollback rate by reward mode
SELECT
  r.reward_mode,
  COUNT(*) FILTER (WHERE e.kind = 'rollback')::float
    / NULLIF(COUNT(*) FILTER (WHERE e.kind = 'commit'), 0) AS rollback_ratio
FROM events e JOIN runs r USING (run_id)
GROUP BY r.reward_mode;
```

```sql
-- Where did the controller spend its growth budget?
SELECT slot_id, COUNT(*) AS commits
FROM events
WHERE run_id = ? AND kind = 'commit'
GROUP BY slot_id
ORDER BY commits DESC;
```

The point is not the SQL. The point is that these queries are *short* because the schemas were chosen to keep them short.

---

## Streaming Storage

For a multi-day run, do not buffer everything in memory. The stream looks like:

```
host trainer step → step row → append to step table
controller decision → event row → append to event table
governor verdict → event row → append to event table
watch-window close → UPDATE event row's post_event_window_summary
```

In practice:

- Append-only Parquet partitioned by `run_id` and day, with periodic flushes
- Postgres if you need online queries during the run
- One log file per run if you are doing single-machine work

The watch-window close is an UPDATE, which Parquet does not love. Either:

1. Hold the in-memory event row until the window closes, then write it once
2. Write the row twice — with and without the summary — and dedupe on read by `event_id`

(1) is cleaner if your watch windows are short. (2) is cleaner if you might crash mid-window and need to recover.

---

## What Not to Log

Logs that look useful but are not, and will hurt:

| Don't log | Why |
|-----------|-----|
| Full model weights per event | Disk explodes; replay log is enough |
| Per-step per-module activation tensors | Replace with a summary statistic |
| Controller observation tensors raw | Hash them; refer to replay log if you need the actual tensor |
| Wall-clock time as an analytic feature | Use `step`; wall-clock is only for cost accounting |
| Whether the dashboard is "pretty" | Ablation needs structured data, not pre-rendered plots |

A morphogenetic run can easily produce hundreds of thousands of events. Every byte multiplies.

---

## Common Mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| Wide step table with one column per module | First grow event breaks all downstream code | Sidecar table keyed by `module_id` |
| `action_params` as unstructured text | Cannot query; ablations require parsing prose | Structured JSON with documented keys per `action_type` |
| No `event_id` in step table | Cannot align step trajectory with events except by timestamp | Either join on `step` or carry `last_event_id` as a step column |
| Logging tensors as JSON-stringified blobs | Disk bloat; opaque to query | Hash them; full tensor in replay log only |
| `pending` events never closed | Aggregations include null `post_event_window_summary` | Close at run-end (commit or rollback explicitly) |
| Reward mode encoded only in `run_id` string | Cannot filter across an experiment grid | First-class column |
| One log file per growth event | Filesystem dies; queries become file-walks | One stream per table, partitioned by run |

---

## Rationalization Resistance

| Rationalization | Reality |
|-----------------|---------|
| "I'll add columns when modules appear; my dashboard handles it" | The dashboard handles it for that run. The next ablation across runs breaks. |
| "Only the final loss matters; I don't need per-event data" | You will, the first time you try to attribute the loss change to a specific decision. |
| "Replay logs and telemetry are redundant" | They are not. Replay logs reproduce; telemetry analyzes. Different jobs. |
| "JSON in the events table is fine for analysis" | Until you have 10k events and want to filter on a JSON key in production. Structured columns where you query, JSON for opaque payloads. |
| "I'll figure out the schema after I see what the controller does" | The first ablation request is when. By then it is too late. |
| "I'll store one row per (step, module) — that's flexible" | You just made the step table grow with topology. See: previous mistake. |
| "Per-module stats can live in a separate file per module" | The cross-module aggregation is the point of the sidecar; per-file storage destroys it. |
| "Event-grain and step-grain in the same table is convenient" | It is, until you join across runs and the row counts are run-dependent. |

---

## Red Flags Checklist

- [ ] **Step row width depends on topology** — number of columns changes when network grows
- [ ] **No event table** — only step-grain logs
- [ ] **No sidecar for per-module stats** — module stats are columns in step table
- [ ] **Action payload is unstructured prose** — cannot be queried structurally
- [ ] **Pending events linger past run end** — null post-event summaries skew aggregates
- [ ] **No `event_id`** — events identified only by timestamp
- [ ] **No reward-mode column** — ablation axis lives in run-id string
- [ ] **Tensors stored inline** — disk usage scales with event count × tensor size
- [ ] **Sidecar uses `module_id` as the aggregation key** — cross-run aggregation is impossible (module IDs are run-local); aggregate by `module_kind` instead
- [ ] **No deterministic event ordering** — events from concurrent ranks interleave non-deterministically (see `deterministic-morphogenesis.md`)

---

## Diagnostic Questions

1. **Can you compute "rollback rate by reward mode" in one SQL query?** If no, your schemas are not ablation-friendly.
2. **What happens to your step table when growth fires?** If width changes, fix it now.
3. **How do you store per-module statistics?** If they are columns in step, you have the wrong shape.
4. **What is the schema of `action_params`?** If "anything goes," you have unstructured data masquerading as structured.
5. **How do you align two runs for comparison?** If by timestamp, your event_id discipline is missing.
6. **What does the watch-window close as?** If pending events linger, your aggregates are wrong.
7. **If I asked you to compare 5 reward modes × 3 gate configurations × 10 seeds = 150 runs, can your dashboard do it?** If "after some adapter code," what does that adapter code paper over?

---

## Cross-References

- **Replay-log details (per-event seed, observation hash)**: `deterministic-morphogenesis.md`
- **Fair across-run comparison given the schemas above**: `evaluation-under-topology-change.md`
- **Governor verdicts as event-table content**: `governor-and-safety-gates.md`
- **Controller decision payload schema**: `rl-controller-for-morphogenesis.md`
- **General ML production telemetry**: `yzmir-ml-production`
- **Gradient norm monitoring (which feeds the step table)**: `yzmir-training-optimization/check-gradients.md`
