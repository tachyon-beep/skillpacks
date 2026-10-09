# Product State and Continuity

**Standing ownership is a stateful job run by a stateless agent — so the state needs a durable, inspectable home the next session can resume cold. If continuity depends on what a previous session "remembers," ownership has already failed; the only durable memory is the workspace, and the only proof a decision was made deliberately is its written provenance.** This sheet defines that workspace: five example artifact schemas (combine/adapt them where continuity remains clear), the Product Decision Record, the resume and checkpoint protocols that load and persist it, and the tracker adapter that lets the workspace reference a backlog it never duplicates.

This sheet owns the *state I/O* — the file shapes and the load/persist protocols at their ends. It does **not** own the operating loop's judgment steps or the authority boundary's escalation taxonomy; those are `product-ownership-operating-model.md`. RESUME and CHECKPOINT appear here because they are the loop's read and write ends — continuity is this sheet's job — but their *meaning* inside the loop lives in the operating-model sheet.

## The workspace: durable state in the existing system

Use the product's existing durable system with inspectable history, stable references and appropriate access controls. The Markdown layout below is one example; a versioned wiki/document system can satisfy the same continuity contract. Git commits are a publication/checkpoint mechanism only where the selected workflow and current authorization call for them. Persist a reviewable draft without committing when that is the authorized scope.

```
docs/product/
├── vision.md          # purpose + who-it-serves + anti-goals + the authority grant
├── roadmap.md         # Now/Next/Later as intent; sequencing handed out, not computed here
├── current-state.md   # the resume brief — the continuity hub
├── metrics.md         # north-star + guardrails, each falsifiable, with current readings
└── decisions/         # append-only Product Decision Records
    ├── 0001-adopt-usage-based-pricing-stub.md
    └── 0002-defer-sso-to-later.md
```

Two rules govern everything below. **One:** the workspace holds *strategy and provenance*; the tactical backlog lives in the tracker and is referenced by ID, never copied in. **Two:** files are append-friendly where history matters (`decisions/` is append-only; `metrics.md` accretes readings) and refresh-in-place where only the present matters (`current-state.md`).

## `vision.md` — purpose, who-it-serves, anti-goals, authority grant

Vision states what the product is *for*, who it serves, what it deliberately refuses to be, and — the load-bearing part for an autonomous owner — the explicit grant that says what the agent may do alone and what it must escalate. The grant records actual delegated authority and current user instructions. The examples below are configurable, not a default prohibition on already authorized work.

```markdown
# Vision — <product name>

## Purpose
One paragraph: the change in the world this product exists to make.

## Who it serves
- Primary: <the user/segment whose problem is the reason to exist>
- Secondary: <served, but not at the primary's expense>
- Explicitly not: <who this product is not for — a real exclusion, not a hedge>

## Anti-goals (what it refuses to be)
- <a tempting adjacent product we will NOT become>
- <a capability we will decline even under pressure>

## Authority grant
Granted by: <human owner name/handle>     Last reviewed: <YYYY-MM-DD>
Review cadence: <e.g. monthly, or on any vision change>

Authorized actions and conditions:
  <list the actual delegated actions, including any release, external or
  irreversible actions explicitly authorized; reference session instructions>.

Decisions reserved to the owner:
  <list actual reserved decisions and conditions; do not add categories merely
  because this template mentions them>.

Earlier authorization persists; request only a genuinely missing decision.
```

The grant is product-specific and inspectable by design — not hardcoded in the pack — so a human can read exactly what they delegated and tighten or widen it in one diff.

## `roadmap.md` — Now/Next/Later as intent (sequencing routed out)

This is the *file schema*, not the roadmapping discipline. Use current delivery evidence for Now/Next/Later, uncertainty and any prioritization arithmetic; `/program-management` is an optional reference, alongside `vision-strategy-and-roadmap.md`. The workspace file records *intent and sequence as the owner currently believes it*; the dated commitment and the WSJF computation are produced downstream.

```markdown
# Roadmap — <product name>            Updated: <YYYY-MM-DD> (PDR-0007)

> This example records intent separately from dated delivery commitments.
> Link the current delivery evidence; no mandatory specialist handoff applies.

## Now  (committed, in-flight)
- **<theme / outcome bet>** — why it's the bet · tracker: #142 · metric: north-star
- **<theme>** — ... · tracker: #240

## Next (shaped, decreasing certainty)
- **<theme>** — ... (not yet sequenced)

## Later (directional bets, no order, no dates)
- **<theme>** — on the map; not shaped or sized
```

Each Now/Next item points at its tracker reference, not at a copied task list. When a Now bet is committed for delivery, the seam fires: coordinate sequencing and forecast using current delivery evidence (program-management is optional) — the roadmap cell never becomes a date.

## `decisions/` — the Product Decision Record (PDR)

PDRs are append-only and numbered. A decision without a written record did not happen deliberately — it happened by drift, and the next session cannot tell a deliberate bet from an accident. The distinctive field is **reversal-trigger**: the pre-committed condition under which this decision should be revisited. It is what keeps a bet falsifiable and a commitment from calcifying — the inverse of sunk-cost. Tie it to a metric where you can (`metrics.md`), so the trigger fires on data, not on mood.

```markdown
# PDR-0007 — Adopt usage-based pricing for the metered tier

Date: <YYYY-MM-DD>   Status: accepted   Author: <agent>   Owner sign-off: <yes/n/a>
Supersedes: PDR-0003   Related: roadmap.md (Now), metrics.md (north-star)

## Context
What forced a decision now; the constraint and the stakes. (2–4 sentences.)

## Options considered
1. <option A> — pro / con
2. <option B> — pro / con
3. <option C — including "do nothing"> — pro / con

## The call
The chosen option, stated plainly. If it required escalation, note the grant
clause and the owner's sign-off.

## Rationale
Why this option beat the others given the context — the reasoning, not a restatement.

## Reversal trigger
The condition that says "reopen this": e.g. "if metered-tier activation stays
below TARGET for two readings, revisit pricing." Pre-committed, ideally
metric-bound. (Kill/keep logic: product-metrics-and-experimentation.md.)
```

Append-only is a discipline, not a convenience: superseding PDR-0003 means writing PDR-0007 that says `Supersedes: PDR-0003` and marking 0003 `Status: superseded` — never editing 0003's call. The history *is* the value.

## `current-state.md` — the resume brief (the continuity hub)

This is the single file RESUME reads first, CHECKPOINT rewrites last, and `/own-product` emits. It is the hand-off note from the last session to the next: what is in flight, what is undecided, where the pointers go. It is refreshed in place — only the present matters — and it never duplicates the backlog; it *points* at it.

```markdown
# Current State — <product name>        Checkpoint: <YYYY-MM-DD HH:MM> · revision <id if applicable>

## The bet right now
One line: the Now theme and the metric it's meant to move.

## In flight
- <workstream> — status — tracker: #142 (3 open, 1 in-review) — dispatched to /axiom-planning
- <workstream> — awaiting acceptance against PRD-0006 criteria

## Open questions / blocked-on-owner
- <decision needing owner sign-off> — escalated <date>, awaiting
- <unknown that blocks a bet> — how we plan to resolve it

## Last checkpoint did
2–4 bullets: what changed since last session (PDRs appended, bets moved, metrics read).

## Next session, start here
The first thing to pick up — so ORIENT has a running start.
```

Keep it short enough to read in one screen. If it grows past that, the detail belongs in a PDR, the tracker, or metrics — not here.

## `metrics.md` — north-star + guardrails, falsifiable, with readings

Every metric carries a *falsifiable* target and a dated current reading. "Improve engagement" is not a metric; "weekly active rate ≥ TARGET by <date>, currently BASELINE" is. Guardrails are the metrics that must *not* move the wrong way while you chase the north-star — they make "did it work" honest by naming the harm a win could hide. The *experiment design, instrumentation, and when-to-kill* logic lives in `product-metrics-and-experimentation.md`; this file is the durable scoreboard.

```markdown
# Metrics — <product name>             Last read: <YYYY-MM-DD>

## North-star
| Metric | Target (falsifiable) | Current | Read on | Trend |
|--------|----------------------|---------|---------|-------|
| Weekly active rate | ≥ TARGET by <date> | BASELINE | <date> | ↗ |

## Input metrics (the levers that move the north-star)
| Metric | Target | Current | Read on |
|--------|--------|---------|---------|
| Activation within 24h | ≥ TARGET | ... | <date> |

## Guardrails (must NOT degrade)
| Metric | Floor / ceiling | Current | Read on |
|--------|-----------------|---------|---------|
| p95 latency | ≤ ceiling | ... | <date> |
| Support contact rate | ≤ ceiling | ... | <date> |
```

A target with no date or no number is not falsifiable and cannot be the basis for accepting a bet or firing a PDR reversal trigger. Reject it.

## The tracker adapter contract

The workspace references the backlog; it never holds it. The adapter is a thin, generic interface over whatever tracker the product already uses — five operations plus a stable ID format. The point is that every workspace file cites IDs in that format, and the agent translates an operation to the concrete tracker via the mapping below.

```
read(id)                  → full detail of one item (requirements, deps, status)
list(filter)              → items matching a filter (status, label, milestone)
create(title, type, ...)  → new item; returns its ID
update(id, fields)        → change status/fields/dependencies
close(id, reason)         → resolve with a reason string
```

The five operations are deliberately tracker-agnostic — virtually every issue or case-management system exposes them, whether through a CLI, a REST API, or an MCP server. Bind the adapter once to whatever the product already uses; the workspace never needs to know *which* system it is, only the stable ID format that system issues.

| Operation | Generic case-management call | Must return / do |
|-----------|------------------------------|------------------|
| read   | get one item by its ID            | full detail: title, status, dependencies |
| list   | query items by a filter           | items matching status / label / milestone |
| create | open a new item                   | the new item's stable ID |
| update | edit fields, status, dependencies | confirmation that the item moved |
| close  | resolve with a reason             | the item closed, the reason recorded |

Bind these to the product's tracker through its own CLI or MCP — confirm exact flags and verbs against that tool's help, never trust a string memorised here. Whatever the system, workspace files cite items by its stable ID format verbatim (e.g. `#142` or `TRK-142`). The rule never bends: if you find yourself pasting a task list into `roadmap.md` or `current-state.md`, stop — link the IDs and let the tracker own the detail. Two sources of truth for one backlog is how the workspace and the tracker silently diverge.

## The RESUME protocol

RESUME loads the workspace cold and reconstructs the running picture. It is read-only — it never writes — and it is the first thing any session does. Step by step:

1. **Locate** the workspace (`docs/product/` or the configured path). If absent, this is a bootstrap, not a resume — `/own-product` scaffolds the five artifacts; do not fabricate state.
2. **Read `current-state.md`** first — it is the hand-off note and the fastest path to the running picture.
3. **Read `vision.md`** — reload purpose, anti-goals, and the authority grant *before* deciding anything, so the session knows its own boundaries.
4. **Read `roadmap.md` and `metrics.md`** — the current bet and the scoreboard against which it will be judged.
5. **Reconcile tracker IDs** — for each in-flight pointer in `current-state.md`, `read(id)` / `list(filter)` the tracker to confirm reality matches the brief (an item the brief calls in-review may already be closed). The tracker is authoritative for tactical status; the brief is authoritative for intent.
6. **Skim recent PDRs** — at minimum the ones `current-state.md` references, to recover *why* the current bet is the bet.
7. **Hand off to ORIENT** (`product-ownership-operating-model.md`) with the reconciled picture. RESUME ends where judgment begins.

If the workspace and the tracker disagree, RESUME surfaces the discrepancy — it does not silently pick one. Reconciling it is an ORIENT/DECIDE act.

## The CHECKPOINT protocol

CHECKPOINT persists material state changes so the next session can resume. Use the selected storage/revision mechanism and current authorization. A saved file or document draft can be durable without a Git commit. Adapt the sequence below to the changed state:

1. **Append PDRs** for material decisions that need continuity/provenance — new numbered files in `decisions/`, with reversal triggers. Unrecorded decisions cannot be reliably reconstructed.
2. **Update `roadmap.md`** if a bet moved horizon (Later→Next→Now) — and only the *intent*; never inject a date or a WSJF score the workspace doesn't own.
3. **Refresh `metrics.md`** with any new readings, dated. Note any reading that crosses a PDR reversal trigger.
4. **Reconcile the tracker** — `create` / `update` / `close` items so tracker reality matches what the session actually did; fix any pointer in `current-state.md` that drifted.
5. **Rewrite `current-state.md`** last — it summarizes the new present: in-flight, open questions, what this checkpoint did, where to start next. This is the file that makes the next resume cheap.
6. **Persist and identify** the changed records. Use a Git commit only when that is the authorized workflow; otherwise report the saved draft/revision and any pending publication separately.
7. **Emit the status summary** for the human owner — the present bet, what changed, anything awaiting their sign-off.

Checkpoint material changes at a useful handoff or recovery boundary. Do not commit unrelated work or delay a needed escalation to manufacture a checkpoint.

## Anti-Patterns

1. **State in the agent's head, not on disk.** The session "remembers" the bet, the open questions, the last decision — so nothing is written down, and the next cold session inherits nothing. It feels efficient because writing state is overhead the current session doesn't need. It is fatal because continuity is the entire job. *Fix: persist material state in the selected durable system — see the CHECKPOINT protocol above.*

2. **The backlog copied into the workspace.** Tasks get pasted into `roadmap.md` or `current-state.md` "so it's all in one place." Seductive because a single file feels tidier than a link. It guarantees the workspace and the tracker diverge — two sources of truth for one backlog, and neither can be trusted. *Fix: reference tracker IDs in the verified format; the tracker adapter owns the detail.*

3. **Decisions without provenance.** A bet is chosen and acted on, but no PDR is written, or the PDR has no reversal trigger. Tempting under time pressure — the decision feels obvious now. Later no one (including the agent) can tell a deliberate call from drift, and a calcified bet has no condition that reopens it. *Fix: append a PDR with options, rationale, and a metric-bound reversal trigger; product-ownership-operating-model.md gates the act, this sheet records it.*

4. **Editing an append-only record.** A superseded decision gets edited in place "to keep it current." Seductive because it looks cleaner than a chain of PDRs. It destroys the audit trail that makes ownership reversible and defensible. *Fix: write a new PDR that `Supersedes:` the old one and mark the old `superseded`; never edit the original call.*

5. **Unfalsifiable targets in `metrics.md`.** Metrics read "improve engagement," "better performance" — directional words with no number and no date. Seductive because vague targets are never wrong. They make acceptance and PDR reversal triggers impossible to fire, so bets never get killed. *Fix: every target gets a number and a date against a BASELINE; reject anything you cannot falsify — kill/keep logic in product-metrics-and-experimentation.md.*

6. **A roadmap that drifts into a delivery schedule.** Dates and WSJF scores creep into `roadmap.md` because a stakeholder wanted precision. Seductive because it looks like control. It duplicates `/program-management`'s job, decays the moment reality moves, and turns honest intent into a broken promise. *Fix: distinguish intent from dated commitments and use current delivery evidence; program-management guidance is optional.*

## Cross-References

- `product-ownership-operating-model.md` — owns the six-step loop's judgment steps (ORIENT/DECIDE/DISPATCH/ACCEPT) and the authority-boundary escalation taxonomy. This sheet owns the load/persist ends (RESUME/CHECKPOINT) and the file shapes they move; the `vision.md` grant slot routes its semantics there.
- `product-metrics-and-experimentation.md` — the kill/keep logic and experiment design behind a PDR's reversal trigger and `metrics.md`'s targets; this sheet stores the durable scoreboard, that sheet decides what it means.
- `vision-strategy-and-roadmap.md` — shapes the strategy and the Now/Next/Later bets this sheet records as a file; the workspace is where that thinking is persisted.
- `prd-and-acceptance-criteria.md` — PRDs are the spec a Now bet hands to delivery; `current-state.md` references them by ID and acceptance is judged against their criteria.
- `/program-management` — optional sequencing, flow and forecast checks when the delivery decision needs them. Keep the underlying evidence linked, without a required handoff or duplicate records.
- `/axiom-planning` — turns the top bet's PRD into an executable, codebase-validated plan; `current-state.md` records what was dispatched to it.
