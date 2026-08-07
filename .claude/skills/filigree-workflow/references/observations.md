# Observations — Ambient Note-Taking

Observations are a scratchpad for things you notice *while doing other work*. They
are not issues — they're lightweight, expiring notes that let you capture a thought
without breaking flow.

## When to Observe

Observations are for **incidental** defects — things you notice *in passing*
while working on something else, that fall *outside the scope of your current
task*. The core use case is: "I don't have time to investigate this right now,
but I want to come back to it."

Examples of good observations:

- A code smell in a neighbouring file you happened to read
- A missing test for an edge case unrelated to what you're changing
- A potential bug in a module you're not touching
- A TODO or FIXME that looks stale
- A dependency that might be outdated

**Always include `file_path` and `line`** when the observation is about specific code.
This anchors it for whoever triages it later.

## When NOT to Observe

**You fix bugs in your currently defined scope. You do NOT use observations to
finish work prematurely.**

If you're working on task X and you notice that your implementation of X has a
gap, a missed edge case, an untested branch, a known shortcoming, or a piece of
follow-up that "should really be done too" — that is **task scope, not an
observation**. You own it. Handle it one of these ways instead:

- **Fix it now** as part of the current task. (Default.)
- **Expand the task** (or split a sub-task) and address it in this work stream.
- **File a proper issue** with a dependency on the current task, so the gap is
  visible in the work record before you close.
- **Surface it to the user** if it changes the shape of what you're delivering.

Filing your own task's deficiencies as observations and closing the task is
**not** completing the task. It is shipping known-broken work and hiding the
debt in a 14-day expiring scratchpad — where it will quietly rot, get
auto-dismissed, and never be addressed. The work record must reflect what is
actually outstanding.

**The test:** *"Would I have noticed this even if I weren't working on this
task?"* If yes → observation. If no → it's part of the work, fix it.

**Don't observe things that are clearly issues either.** If you're confident
something is a bug or a needed feature, create an issue directly. Observations
are for "hmm, this might be worth looking at" — the uncertain middle ground.

## Triage Workflow

Observations expire after 14 days. Triage them before they rot:

1. **At session end:** run `observation_list` and quickly scan what's accumulated
2. **For each observation, decide:**
   - **Dismiss** — not actionable, already fixed, or not worth tracking. Use
     `observation_dismiss` with a brief reason for the audit trail.
   - **Promote** — deserves to be tracked as an issue. Use `observation_promote`
     which atomically creates an issue and labels it `from-observation`. Choose
     the right issue type:
     - `type='bug'` — something is broken or produces wrong results
     - `type='task'` (default) — cleanup, improvement, or "this works but is shitty"
     - `type='feature'` — a missing capability that should exist
     - `type='requirement'` — a formal requirement to be reviewed, approved, and verified, when the requirements pack is enabled
   - **Leave it** — still uncertain. Let it age. If it survives a few sessions
     without being promoted, it's probably a dismiss.

3. **Batch cleanup:** use the MCP tool `observation_batch_dismiss` when several observations
   have gone stale together.

## Promote vs Dismiss

| Signal | Action |
|--------|--------|
| You noticed it twice in separate sessions | Promote |
| It's in a hot code path or critical module | Promote |
| It has a clear fix or next step | Promote |
| It was about code that's since been refactored | Dismiss |
| It's a style/taste preference, not a defect | Dismiss |
| You can't articulate what the fix would be | Leave it (or dismiss if > 7 days old) |

## Tracking the Pipeline

Promoted observations get the `from-observation` label. To see the pipeline output:

```bash
filigree list --label=from-observation     # All promoted observations
filigree search "from-observation"         # Search with context
```
