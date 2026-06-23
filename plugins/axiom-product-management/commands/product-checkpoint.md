---
description: "Write standing product state back to the git-versioned workspace at the end of a working session — the CHECKPOINT step of the ownership loop. Reads the existing workspace (default docs/product/), reconciles it against what actually happened this session (git, tracker, metrics), then appends a Product Decision Record for every decision made, updates roadmap.md only where a bet changed horizon as INTENT (never a date or WSJF score), refreshes metrics.md with dated readings and flags any reading that crossed a PDR reversal trigger, rewrites current-state.md as the next session's resume brief, and commits — because an uncommitted workspace is an unsaved one. Respects the authority boundary absolutely: anything irreversible or outward-facing (vision/strategy change, public release or announcement, feature deprecation, pricing/commercial change, data deletion, external-party contact) is FLAGGED for the human owner as an escalation, never acted on; vision.md is never silently rewritten. Emits a concise status summary that is the human's window into what their autonomous owner did. If no workspace exists, it stops and tells the user to run /own-product first — it never fabricates state."
allowed-tools: ["Read", "Grep", "Glob", "Bash", "AskUserQuestion", "Write"]
argument-hint: "[note]"
---

# Product Checkpoint Command

You are running **CHECKPOINT** — the closing bookend of the product ownership loop `RESUME → ORIENT → DECIDE → DISPATCH → ACCEPT → CHECKPOINT`. The job is to **write this session's reasoning back into the git-versioned product workspace** so the next session resumes truthfully instead of relitigating settled questions. The discipline is simple and absolute: **an un-written decision is one the next session re-makes differently, and an uncommitted workspace is an unsaved one.** This command persists; the next `/own-product` reloads.

This command **writes state, it does not make new product decisions.** It records the decisions you already made this session, refreshes the durable artifacts, and emits a status summary. It does not re-plan, re-prioritize from scratch, or shape new bets — and it never crosses the authority boundary. Anything irreversible or outward-facing that surfaced this session is **flagged for the human owner**, not executed.

The optional `[note]` argument is a free-text reminder of what this session was about (e.g. `accepted ALPHA partial; started BRAVO`). Use it to seed the "what this checkpoint did" summary; if absent, derive the delta from git and the tracker.

## Precondition — the workspace must exist

CHECKPOINT operates on an existing workspace; it does not bootstrap one. Locate it first:

```bash
ls -d "${PRODUCT_DIR:-docs/product}" 2>/dev/null && \
  ls "${PRODUCT_DIR:-docs/product}"/{vision.md,roadmap.md,current-state.md,metrics.md} \
     "${PRODUCT_DIR:-docs/product}"/decisions/ 2>/dev/null
```

If the workspace is absent or missing core artifacts, **stop**: tell the user to run `/own-product` first to bootstrap and resume — that command owns the `RESUME → ORIENT` half of the loop and scaffolds the five artifacts. Do **not** fabricate `vision.md` or invent a strategy here. CHECKPOINT writes back what a prior `RESUME` loaded; with no workspace there is nothing to write back to, and inventing one would mean inventing the authority grant that gates everything else.

The artifact schemas, the PDR template, and the tracker-adapter contract are defined in `product-state-and-continuity.md`. This command runs the **CHECKPOINT protocol** specified there (steps below); it does not redefine the file shapes.

## Step 1 — Reconcile what actually happened this session

Before writing anything, establish the delta between the workspace as last checkpointed and reality now. Do not trust recollection — read the signal.

**From git (what changed in the workspace and the product):**
```bash
git -C "${PRODUCT_DIR:-.}" log --oneline -20
git -C "${PRODUCT_DIR:-.}" status --short
```
The last `product:` commit is the prior checkpoint; everything since is this session's delta. Uncommitted changes in the workspace are this session's un-saved work — exactly what CHECKPOINT must persist.

**From the tracker (tactical status is authoritative there, not in the brief):**
For each in-flight pointer in `current-state.md`, reconcile against the tracker via the adapter (`product-state-and-continuity.md`): an item the brief calls *in-review* may already be closed; a bet you dispatched may have spawned new items. The tracker owns tactical status; the workspace owns intent. Where they disagree, the tracker wins for status and the workspace gets corrected.

**Confirm the decisions made this session.** List, plainly, every product call made: bets decided, work accepted/rejected against criteria, reprioritizations, bets killed. Each one needs a PDR in Step 2. If you are unsure whether something rose to a decision, it did — record it. The `[note]` argument and the conversation are your source; ask the user with `AskUserQuestion` only if a decision's *call* or *rationale* is genuinely ambiguous, never to manufacture decisions that were not made.

## Step 2 — Authority-boundary gate (run BEFORE writing)

This gate is the reason an autonomous owner is trustworthy. **Scan everything this session did or proposes to do against the escalation list. Anything that is irreversible or outward-facing is FLAGGED for the human owner — never executed by this command, and `vision.md` is never silently rewritten.**

| Trips when this session involved | Action — do NOT proceed on it |
|----------------------------------|-------------------------------|
| A change to **vision, strategy, or the authority grant** | Do not edit `vision.md`. Record the *proposal* as a PDR with `Status: proposed`, flag it in the summary as awaiting owner sign-off |
| A **public release, announcement, or external communication** | Flag for owner; record the intent, do not trigger it |
| **Deprecating or removing a feature** users depend on | Flag for owner; record the proposed deprecation, do not enact it |
| A **pricing, commercial, contractual, or licensing** change | Flag for owner; PDR as `proposed`, sign-off required |
| **Deleting data** or any irreversible data operation | Flag for owner; never run it from a checkpoint |
| Anything touching **external parties** (customers, partners, regulators) | Flag for owner |

When an item is ambiguous — does it cross the line or not? — **default to flagging it.** The boundary is a one-way door; escalation is cheap, an irreversible mistake is not. Surface a single specific question to the human rather than interpreting in favor of proceeding. The taxonomy and rationale live in `product-ownership-operating-model.md`; the live grant is in `vision.md` (read it — the grant is product-specific, not the default).

A reversed vision/strategy decision is itself a vision change: it escalates. CHECKPOINT records that you *want* to reverse it (PDR, `proposed`) and flags it — it does not perform the reversal.

## Step 3 — Persist the workspace (the CHECKPOINT protocol)

Write the artifacts in this order. Each step is a write to a specific file with a specific discipline.

**1. Append a PDR for every decision made this session.** New numbered files in `decisions/`, append-only, one per decision (`NNNN-short-slug.md`). Each carries context → options → call → rationale → **reversal trigger** (the template is in `product-state-and-continuity.md`). The reversal trigger is the load-bearing field — pre-commit the metric-bound condition under which the bet reopens; tie it to `metrics.md` so it fires on data, not mood. A decision flagged in Step 2 is recorded with `Status: proposed` and a note that owner sign-off is pending; an autonomous decision within the grant is `Status: accepted`. **Never edit an existing PDR** — superseding one means writing a new PDR that says `Supersedes: NNNN` and marking the old `Status: superseded`. The history is the value.

**2. Update `roadmap.md` only if a bet changed horizon — as INTENT.** If a bet moved Later→Next→Now (or was killed), reflect the new band and point it at its tracker ID. Stamp the update line with the date and the driving PDR (`Updated: <date> (PDR-NNNN)`). **Do not inject a date, a WSJF score, or a delivery sequence** — that is `/axiom-program-management`'s job; the roadmap records intent, the dated forecast is produced downstream. Keep the routing banner intact. If no bet changed horizon this session, leave `roadmap.md` untouched.

**3. Refresh `metrics.md` with dated readings.** Add any new north-star, input, or guardrail readings taken this session, each dated. **Flag any reading that crossed a PDR reversal trigger** — that is a kill/keep signal the next `DECIDE` must act on; note it in `current-state.md` under open questions and in the summary. Reject any target that is not falsifiable (no number, no date); kill/keep logic itself lives in `product-metrics-and-experimentation.md` — this file is the durable scoreboard.

**4. Reconcile the tracker.** `create` / `update` / `close` items via the adapter so tracker reality matches what the session actually did, and fix any `current-state.md` pointer that drifted. Never paste a task list into the workspace — link IDs; the tracker owns the detail.

**5. Rewrite `current-state.md` last** — it is the next session's resume brief and the continuity hub. Refresh in place (only the present matters): the bet right now and the metric it moves, what is in flight (with tracker IDs), open questions and anything blocked-on-owner (including every Step-2 flag), what this checkpoint did (2–4 bullets), and where the next session starts. Keep it to one screen; detail belongs in PDRs, the tracker, or metrics.

**6. Commit the workspace.** Durability *is* the commit:
```bash
git -C "${PRODUCT_DIR:-.}" add "${PRODUCT_DIR:-docs/product}"
git -C "${PRODUCT_DIR:-.}" commit -m "product: <bet/state delta>; PDR-NNNN[, PDR-MMMM]"
```
Name the bet and the PDRs touched in the message so the commit log doubles as a decision audit trail. An uncommitted workspace is an unsaved one — until this runs, nothing is checkpointed. Do **not** push, tag, or release from this command; those are outward-facing and gate to the owner.

## Step 4 — Emit the status summary

After committing, emit a tight summary as clean markdown the user can read at a glance. This is the human owner's window into what their autonomous owner did while running — it is how trust is maintained, not boilerplate. Five parts:

1. **The bet right now** — one line: the Now theme and the metric it is meant to move.
2. **What this checkpoint did** — the decisions recorded (PDR numbers), bets that moved horizon, metric readings taken, tracker reconciliation. The delta, not a changelog.
3. **Metric movement and any tripped reversal trigger** — what moved and whether any reading crossed a kill/keep threshold.
4. **Awaiting your sign-off (escalations)** — every Step-2 flag, stated as a specific question the owner can answer. If none, say "nothing escalated this session." This section is mandatory — its absence on a session that touched anything outward-facing is the failure the gate exists to prevent.
5. **Next session starts here** — the one thing the next `RESUME → ORIENT` picks up first.

Lead with the product name and the checkpoint timestamp + commit sha. Do not dump the raw git log, the full PDR bodies, or the tracker export — those are inputs and the durable files; the summary is the human-readable delta.

## Anti-Patterns

1. **Fabricating a workspace instead of stopping.** No `docs/product/` exists, so the command scaffolds `vision.md` and invents a strategy to have something to write to. Seductive because it feels more helpful than refusing. But it manufactures the authority grant that gates every other action and starts the product from an unowned, un-sanctioned strategy. *Fix: stop and route to `/own-product`, which owns bootstrap and RESUME — see the precondition above.*

2. **Silently rewriting `vision.md` or enacting an outward-facing change.** A strategy shift or a release felt like the obvious close to the session, so the command just does it. Seductive because you own the product and the step seems well inside your remit. But vision/strategy, releases, deprecations, pricing, and data deletion are one-way doors that always gate. *Fix: run the Step-2 authority gate; record the proposal as a `proposed` PDR and flag it for the owner — never act (`product-ownership-operating-model.md`).*

3. **Checkpointing without committing.** The artifacts get refreshed in the working tree and the command stops, "saving the commit for later." Seductive because the writing felt like the real work. But the next `RESUME` reads committed state; uncommitted edits are invisible to it and to the audit trail. *Fix: the commit is the checkpoint — Step 3.6 runs every time, naming the bet and PDRs (`product-state-and-continuity.md`).*

4. **A decision made this session with no PDR.** The call felt obvious, so it goes unrecorded. Seductive under end-of-session fatigue — writing provenance is overhead the current session does not need. Later no one can tell a deliberate bet from drift, and the calcified bet has no reversal trigger. *Fix: append a PDR per decision with options, rationale, and a metric-bound reversal trigger; if unsure whether it was a decision, it was — Step 1.*

5. **Editing a superseded PDR in place.** A past decision changed, so its PDR gets edited "to keep it current." Seductive because one clean file beats a chain. It destroys the append-only audit trail that makes ownership reversible and defensible. *Fix: write a new PDR that `Supersedes:` the old and mark the old `superseded`; never touch the original call (`product-state-and-continuity.md`).*

6. **Letting a date or WSJF score leak into `roadmap.md`.** A bet moved to Now, so a target date or a priority score gets written beside it for "precision." Seductive because it looks like control. It duplicates `/axiom-program-management`'s job, decays the moment reality moves, and turns honest intent into a broken promise. *Fix: keep `roadmap.md` intent-only; hand the committed bet to `/axiom-program-management` for sequencing and the forecast — Step 3.2.*

7. **A comfortable summary that buries the escalation.** The summary leads with progress and omits the outward-facing thing that needs sign-off, because flagging it feels like admitting the session is incomplete. But the escalation section is the whole point of an autonomous owner the human can trust. *Fix: the "awaiting your sign-off" section is mandatory and states each flag as a specific question — Step 4.4.*

## Cross-References

- `product-state-and-continuity.md` — defines the CHECKPOINT protocol this command runs, every artifact schema (`vision.md`, `roadmap.md`, `current-state.md`, `metrics.md`), the append-only PDR template, and the tracker-adapter contract. The authority behind this command's file discipline.
- `product-ownership-operating-model.md` — places CHECKPOINT as the closing bookend of the `RESUME → ORIENT → DECIDE → DISPATCH → ACCEPT → CHECKPOINT` loop, and owns the authority-boundary escalation taxonomy the Step-2 gate enforces.
- `product-metrics-and-experimentation.md` — the kill/keep logic behind a tripped PDR reversal trigger; this command records the reading and flags the trip, that sheet decides what to do about it.
- `/own-product` — the `RESUME → ORIENT` entry point that bootstraps or loads the workspace; run it first if no workspace exists, and at the start of every session this command closes.
- `/axiom-program-management` — owns Now/Next/Later sequencing mechanics, WSJF / cost-of-delay, flow metrics, and the dated forecast; `roadmap.md` records intent only and never computes them. The committed bet is handed there for delivery.
