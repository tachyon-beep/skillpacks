# Product Ownership Operating Model

**Owning a product across sessions is not a state of mind — it is a protocol you run every time you sit down, or continuity is a lie you tell yourself.** A stateless model cannot "remember" it owns anything; what it can do is reload its own prior decisions from a versioned workspace, act only within a written authority grant, and write its reasoning back before it stops. This sheet is that protocol: the six-step loop `RESUME → ORIENT → DECIDE → DISPATCH → ACCEPT → CHECKPOINT`, the session bookends that make ownership survive a fresh context, and the authority boundary that separates what you may do alone from what you must escalate. Skip the loop and you do not have a product owner — you have a chatbot that re-litigates settled questions and occasionally ships something irreversible.

The artifacts this loop reads and writes — `vision.md`, `roadmap.md`, the append-only `decisions/` (Product Decision Records), `current-state.md`, `metrics.md` — are *defined* in `product-state-and-continuity.md`: their schemas, the PDR template, the tracker-adapter contract. This sheet does not restate those structures; it operates on them by name. When you need to know what a field means or what a PDR must contain, that is the sheet. When you need to know *what to do in what order and what breaks if you don't*, you are in the right place.

## The loop is the unit of ownership

A session is one pass through the loop. It always opens with `RESUME` and always closes with `CHECKPOINT`; the four middle steps are where the actual product work happens and may iterate or be partially skipped within a session, but the bookends are non-negotiable. The reason is structural: you have no memory between sessions. Everything you "know" about this product on a fresh context comes from `RESUME` reading the workspace, and everything the *next* session will know comes from this session's `CHECKPOINT` writing it. Drop either bookend and the chain of continuity breaks at exactly that link — silently, because nothing errors; the next session simply starts from a stale or contradictory picture and acts on it.

| Step | What it does | Reads (input) | Writes / emits (output) | Failure if skipped |
|---|---|---|---|---|
| **RESUME** | Reconstruct standing context from the workspace; never re-derive from scratch | `current-state.md`, recent PDRs in `decisions/`, `vision.md` (incl. authority grant), `roadmap.md`, `metrics.md`, tracker state via adapter | An internal, loaded picture: where things stand, what's in flight, open questions, last checkpoint | You re-derive strategy from the prompt and silently contradict a past PDR; continuity dies |
| **ORIENT** | Reconcile the loaded picture against current reality; surface what changed | Loaded context + repo/tracker reality + new human input this session | A short orientation: deltas since last checkpoint, what's now decidable, what's blocked | You act on a stale picture; in-flight work or shipped value goes unaccounted for |
| **DECIDE** | Make the product call — which bet, what falsifiable success criteria — within the agreed strategy, or escalate if it isn't | `roadmap.md` bets, `metrics.md` targets, discovery evidence, the authority grant | A decision recorded as a PDR (context → options → call → rationale → reversal trigger); roadmap update if intent moved | Decisions leave no provenance; the next session can't tell *why* and re-opens settled questions |
| **DISPATCH** | Hand the committed bet outward for sequencing, planning, and build — orchestrate, don't implement | The decided bet + its acceptance criteria, tracker IDs | Work handed to siblings: `/axiom-program-management` (sequence/deliver), `/write-prd` → `/axiom-planning` (plan the top item), eng packs (build); tracker IDs recorded | You reimplement delivery mechanics here — restating WSJF, flow, plans — and own nothing cleanly |
| **ACCEPT** | Validate that value *landed* — test the shipped thing against the falsifiable criteria, not against "it shipped" | Acceptance criteria from the PDR/PRD, `metrics.md` readings, the delivered artifact | An accept/reject verdict recorded; metrics updated; a PDR or follow-up bet if the result falsified the hypothesis | Output-over-outcome: you bank "shipped" as success and never learn whether the bet paid |
| **CHECKPOINT** | Write standing state back so the next session can resume truthfully | The session's decisions, deltas, verdicts, open questions | Updated `current-state.md`, appended PDRs, refreshed `roadmap.md` + `metrics.md`; a status summary | The next `RESUME` reads a stale workspace; the session's reasoning is lost; ownership resets |

### RESUME — load, never re-derive

`RESUME` reads the workspace and reconstructs what you already decided. The hard rule: **the workspace is the source of truth about your own past, and you do not silently override it.** If `current-state.md` says a bet is in flight and three PDRs explain why the strategy is what it is, you start from that — you do not infer a fresh strategy from the user's opening message and quietly proceed as if the PDRs didn't exist. If new information genuinely invalidates a past decision, that is a *new* decision (a new PDR with a reversal-trigger that fired), made deliberately in `DECIDE`, not an amnesiac contradiction in `RESUME`. The runnable entry point is `/own-product`: on a workspace that exists it loads and emits the current-state brief; on a repo with none it bootstraps the workspace first (per `product-state-and-continuity.md`) and then resumes.

### ORIENT — reconcile picture against reality

The loaded picture is as of the last checkpoint; the world moved since. `ORIENT` diffs the two: what shipped, what the tracker now shows, what metrics now read, what the human is asking for this session. The output is a short delta — "since last checkpoint, bet X shipped and is awaiting ACCEPT; metric Y moved; the user wants to add Z." `ORIENT` is what keeps `RESUME` from being a trap: loading stale state and acting on it is worse than not loading at all, because it carries false confidence.

### DECIDE — the product call, with provenance

`DECIDE` is where this pack's authority concentrates: choosing the bet and writing its *falsifiable* success criteria. A decision that isn't recorded as a PDR didn't happen as far as the next session is concerned — provenance is not paperwork, it is the only thing that lets a future you (or the human) understand the call without re-deriving it. Every non-trivial decision produces a PDR; if the decision changes vision or strategy, it is *not yours to make alone* — it escalates (see the authority boundary). The acceptance criteria you write here are the contract `ACCEPT` will later test against, so write them falsifiable or you have built in an acceptance gap from the start.

### DISPATCH — hand it out; this is the seam

`DISPATCH` is the load-bearing seam of the whole pack. **Product decided the bet and the criteria; program-management sequences and delivers it.** You hand the committed bet outward and do *not* reimplement what siblings own:

- **`/axiom-program-management`** sequences and delivers it — flow, forecast, WSJF, dependency coordination. You do not restate the WSJF arithmetic or flow-metric definitions here; you route to it.
- **`/write-prd`** turns the bet into a PRD with falsifiable acceptance criteria, ready to hand down; **`/axiom-planning`** turns the PRD's top item into a codebase-validated implementation plan. You hand it the item; it owns the plan.
- **`/axiom-solution-architect`** shapes the *how* (solution/architecture, ADRs) when the bet needs it; the **engineering packs** build. You orchestrate; you do not write the code or the plan.

What `DISPATCH` records in the workspace is the *linkage*: which tracker IDs and which sibling now own the delivery of this bet — never a duplicate of the backlog or the plan. The backlog lives in the tracker via the adapter (`product-state-and-continuity.md`); the workspace references it.

### ACCEPT — did value land?

`ACCEPT` closes the loop product opened in `DECIDE`. Program-management's job ends at *delivered-predictably*; yours continues to *did-it-work*. You take the falsifiable criteria from the PDR/PRD, read the relevant `metrics.md`, and render a verdict against the *outcome*, not the output. "Shipped on time" is program's success; "the metric moved as the hypothesis predicted" is yours. A bet that shipped clean and falsified its own hypothesis is a *successful* product outcome — you learned the bet was wrong cheaply — recorded as such in a PDR, not buried. Skipping `ACCEPT` is how a feature factory feels productive while the business case never realizes.

### CHECKPOINT — write it back or it didn't happen

`CHECKPOINT` is the mirror of `RESUME`: it serializes this session's reasoning into the workspace so the next session can reload it. It updates `current-state.md` (the resume brief), appends any new PDRs, refreshes `roadmap.md` and `metrics.md`, and emits a status summary. The runnable entry point is `/product-checkpoint`. The discipline is to checkpoint *before* you stop, not "later" — an un-written decision is a decision the next session will re-make differently. If a session is interrupted, a partial checkpoint of what's settled beats none; the open questions go into `current-state.md` so the next `RESUME` knows where the seam is.

## The session protocol: bookends are mandatory

The two ends of every session are not optional steps you do when convenient — they are the contract that makes ownership continuous:

- **A session always begins by RESUMING.** You read `current-state.md` and recent PDRs first, before forming any view. You do not re-derive the strategy, and you do not silently contradict a past decision. The user's opening message is *new input to reconcile in ORIENT*, not a replacement for the standing context. If the user's request conflicts with a recorded decision, you surface the conflict ("PDR-014 decided X for reason Y — are we reversing that?") rather than quietly overriding it.
- **A session always ends by CHECKPOINTING.** Whatever you decided, dispatched, or accepted gets written back. The status summary `/product-checkpoint` emits is also the human's window into what their autonomous owner did while running — it is how trust is maintained.

Between the bookends, `DECIDE → DISPATCH → ACCEPT` run as the work demands and may loop: you might decide and dispatch one bet, accept a previously-shipped one, and reprioritize a third, all in one session. That middle is fluid; the bookends are fixed. The command mapping makes this concrete: `/own-product` runs `RESUME` + `ORIENT` (bootstrap or load, emit the brief and proposed next bets); the middle steps run in-session, where `DISPATCH` may invoke `/write-prd` and route to `/axiom-planning` for the top bet; `/product-checkpoint` runs `CHECKPOINT`.

### A session in motion

A concrete pass, with the artifacts named generically, shows how the steps chain and where each one's output feeds the next:

1. **RESUME** (`/own-product`) — load `current-state.md`: "Bet ALPHA in flight, awaiting acceptance; Bet BRAVO is top of Next; metric NORTH-STAR reading below target." Recent PDRs explain why CHARLIE was deprioritized last session. The authority grant in `vision.md` reserves releases and pricing.
2. **ORIENT** — reconcile: the tracker shows ALPHA shipped to production since last checkpoint; the user opens with "let's start on BRAVO." Delta: ALPHA is now acceptable, not just in flight; BRAVO is decidable.
3. **ACCEPT** (ALPHA first) — ALPHA's PDR criterion was "NORTH-STAR rises by TARGET within two weeks." Metrics show it moved half that. Verdict: partial — hypothesis under-confirmed. Record a PDR; flag a follow-up bet to investigate the gap rather than banking ALPHA as a win.
4. **DECIDE** (BRAVO) — within the agreed strategy, so it's yours. Write BRAVO's falsifiable criteria, record the PDR (context → options → call → rationale → reversal trigger).
5. **DISPATCH** (BRAVO) — `/write-prd` produces the PRD; route the top item to `/axiom-planning`; hand sequencing to `/axiom-program-management`. Record the tracker IDs and owning sibling in the workspace.
6. **CHECKPOINT** (`/product-checkpoint`) — update `current-state.md` (ALPHA accepted-partial, BRAVO dispatched, follow-up bet opened), append the two PDRs, refresh `metrics.md`, emit the status summary for the human.

The one thing in that session that would have *stopped* the loop: if "start on BRAVO" had instead been "announce BRAVO publicly," DISPATCH would not have proceeded — that is outward-facing, escalate first.

## The authority boundary: autonomous within strategy, escalate at the edge

This is pure-differentiation territory and the most important discipline in the pack: **act autonomously *within* the agreed strategy; escalate to the human owner before anything hard-to-reverse or outward-facing.** The boundary is not hardcoded in the pack — it lives in `vision.md` as an explicit, product-specific, inspectable *authority grant* (its schema is in `product-state-and-continuity.md`). The pack defines the *default* shape of the boundary; the grant in `vision.md` is the actual authority for *this* product, and `RESUME` loads it every session so you are always acting under the current, written grant — not a remembered one.

**You act autonomously (within the grant) on reversible, inward-facing product work:**

- Prioritizing and reprioritizing the backlog within the agreed strategy
- Writing PRDs and falsifiable acceptance criteria
- Dispatching decided bets to delivery (program-management, planning, eng)
- Accepting or rejecting delivered work against its criteria
- Recording PDRs for decisions within the strategy
- Running internal experiments and instrumenting metrics

**You escalate to the human owner — and do not proceed until granted — before anything hard-to-reverse or outward-facing:**

- Changing the vision, the strategy, or the authority grant itself
- Any public release, announcement, or external communication
- Deprecating or removing a feature users depend on
- Pricing, commercial, contractual, or licensing changes
- Deleting data, or any irreversible data operation
- Anything that touches external parties (customers, partners, regulators)
- Spending money or committing the org to an obligation

The asymmetry is deliberate: reversible inward work is where autonomy creates value and a mistake is cheap to undo; irreversible or outward-facing action is where a mistake is expensive, public, or permanent, so it gets a human gate even when you are confident. Confidence is not the test — *reversibility and audience* are.

**When an action is ambiguous — does it cross the line or not? — the default is to escalate.** Treat the boundary as a one-way door you do not walk through on a guess. Escalation is cheap; an irreversible mistake made under a misread of the grant is not. The right move when unsure is a single, specific question to the human ("this would change pricing copy users see — that reads as outward-facing under the grant; confirm before I proceed?"), not a unilateral interpretation that happens to favor proceeding. A pattern of ambiguity in the same place is a signal that the grant in `vision.md` needs sharpening — propose the clarification (itself a vision change, hence escalated) rather than resolving it silently in your own favor each time.

## Anti-Patterns

1. **Re-deriving the strategy instead of resuming it.** Seductive because the user's opening message feels like the whole context and re-deriving feels responsive and fast. But it discards every prior decision and produces a "strategy" untethered from the recorded one — the product lurches in a new direction each session. *Fix: RESUME from `current-state.md` and recent PDRs before forming any view; the prompt is input to ORIENT, not a replacement for standing context (`product-state-and-continuity.md`).*

2. **Silently contradicting a past PDR.** Seductive because a fresh look genuinely sometimes disagrees with a past call, and overriding feels like good judgment. But an undocumented reversal destroys provenance: the next session sees two conflicting realities with no record of which won or why. *Fix: surface the conflict and reverse a decision only via a new PDR whose reversal-trigger fired, made deliberately in DECIDE — never as an amnesiac override in RESUME.*

3. **Acting when the authority is ambiguous.** Seductive because escalating feels like indecision and you are usually confident you're inside the grant. But the boundary is a one-way door; a confident misread of an irreversible action is exactly the failure the gate exists to prevent. *Fix: default to escalate on ambiguity with a single specific question; reversibility and audience are the test, not your confidence.*

4. **Autonomy overreach on irreversible or outward-facing action.** Seductive because you own the product and a release or a deprecation can feel like an obvious next step well inside your remit. But public, commercial, data-destructive, or external-party actions are where a mistake is permanent or visible, so they always gate — ownership is not unilateral authority. *Fix: route every item on the escalation list to the human and wait; the grant in `vision.md` is the authority, and it explicitly reserves these.*

5. **DISPATCH as reimplementation.** Seductive because restating the WSJF table, the flow plan, or the implementation steps feels like thorough ownership. But it duplicates what siblings own, drifts out of sync with the real backlog and plan, and blurs the seam that makes the pack coherent. *Fix: hand the committed bet outward — `/axiom-program-management` sequences/delivers, `/axiom-planning` plans the top item, eng builds — and record only the linkage (tracker IDs, owning sibling) in the workspace.*

6. **Stopping without checkpointing.** Seductive because the work feels done and writing state back feels like overhead after the real thinking. But an un-written decision is one the next session re-makes differently, and the session's reasoning evaporates with the context. *Fix: run `/product-checkpoint` before you stop; on interruption, checkpoint what's settled and record open questions so the next RESUME knows the seam (`product-state-and-continuity.md`).*

## Cross-References

- `product-state-and-continuity.md` — defines every artifact this loop reads and writes (`vision.md`, `roadmap.md`, `decisions/` PDR template, `current-state.md`, `metrics.md`), the authority-grant schema, and the tracker-adapter contract; this sheet operates on those structures, that sheet specifies them.
- `prd-and-acceptance-criteria.md` — the falsifiable acceptance criteria that DECIDE writes and ACCEPT later tests against; weak criteria here become an acceptance gap downstream.
- `delivery-orchestration-and-acceptance.md` — the full DISPATCH → verify-it-shipped → ACCEPT mechanics; this sheet states the loop steps, that sheet runs the orchestration in depth.
- `product-metrics-and-experimentation.md` — the metric readings ACCEPT consults to judge whether value landed, and the "when to kill a bet" call when a hypothesis is falsified.
- `product-anti-patterns.md` — the broad product failure catalog (build trap, feature factory, vanity metrics, HiPPO capture); this sheet's anti-patterns are loop- and authority-specific and route the rest there.
- `/axiom-program-management` — the DISPATCH destination for sequencing and delivery (flow, forecast, WSJF, coordination); product owns the bet and *did-it-work*, program owns *delivered-predictably*. Do not restate its mechanics here.
- `/axiom-planning` — turns the dispatched bet's top item into a codebase-validated implementation plan; hand it the PRD, it owns the plan.
