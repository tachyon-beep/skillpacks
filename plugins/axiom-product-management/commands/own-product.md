---
description: Take or resume standing ownership of a software product — runs the RESUME + ORIENT half of the operating loop against a git-versioned product workspace (default docs/product/). Branches on whether the workspace exists: if ABSENT, scans the repo, tracker, git history, and README/docs and CONSTRUCTS the five standing artifacts (vision.md, roadmap.md, metrics.md, current-state.md, decisions/), proposing an authority grant for the human to confirm; if PRESENT, LOADS the workspace and RECONCILES it against current reality (repo state, tracker, metrics), emitting a current-state brief, proposed next bets, and any drift. Always ends by surfacing the authority grant for explicit confirmation. Writes real files — never a copy-paste block.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "AskUserQuestion", "Write"]
argument-hint: "[product_path]"
---

# Own Product Command

You are taking **standing ownership** of a software product, or resuming ownership you held in a prior session. This command runs the `RESUME → ORIENT` half of the operating loop (`product-ownership-operating-model.md`): it builds-or-loads the git-versioned **product workspace** and emits a current-state brief plus proposed next bets. The defining property of this command — the reason it exists rather than a chat about the product — is that it **writes durable files to disk**, because ownership is stateful and a stateless agent's only memory is the workspace. Do not emit a copy-paste block for the workspace files; **Write** them.

This is a `/own-product` slash command. The optional argument is the workspace path; default to `docs/product/` if omitted (configurable per product). The middle of the loop (`DECIDE → DISPATCH → ACCEPT`) runs in-session after this; the write-back (`CHECKPOINT`) is `/product-checkpoint`. This command never commits — commit discipline lives in checkpoint.

## Branch first — bootstrap or resume

The **first action** is an existence check on the workspace path. Everything downstream forks on the answer:

- **ABSENT** (`docs/product/` does not exist) → **BOOTSTRAP**. There is no state to resume; construct the workspace from observed reality. Do not fabricate a remembered history.
- **PRESENT** (the workspace exists) → **RESUME**. Load the five artifacts, reconcile them against current reality, surface drift. Do not re-derive strategy from the prompt; the workspace is the source of truth about the product's own past.

```bash
# Settle the branch
PRODUCT_PATH="${1:-docs/product}"
test -f "$PRODUCT_PATH/vision.md" && echo "PRESENT — resume" || echo "ABSENT — bootstrap"
```

If only *some* artifacts exist (a partial or hand-started workspace), treat it as RESUME for what is present and BOOTSTRAP the missing files — reconcile, do not overwrite, what the human already wrote.

## BOOTSTRAP — construct the workspace from observed reality

There is no prior state, so seed the workspace from what the repo and tracker already reveal about the product's direction. Read before asking — observed direction is cheaper and more honest than interrogation:

```bash
# Stated purpose and direction
ls README* docs/ 2>/dev/null
grep -rl -i "vision\|mission\|purpose\|north.star\|roadmap\|metric\|objective" README* docs/ 2>/dev/null

# Observed direction from history — what has actually been built lately
git log --oneline -30 2>/dev/null

# Tracker reality via the adapter (product-state-and-continuity.md) — detect the tracker, do not assume
command -v filigree >/dev/null && filigree list --status=open 2>/dev/null
command -v gh >/dev/null && gh issue list --state open 2>/dev/null
```

From that evidence, draft the five artifacts to the **exact schemas in `product-state-and-continuity.md`** — do not invent shapes:

1. **`vision.md`** — purpose, who-it-serves, anti-goals, and the authority-grant slot. Draft purpose and audience from README/docs; mark anything you inferred rather than found as an assumption to confirm.
2. **`roadmap.md`** — seed Now/Next/Later as **intent only**, from observed direction (recent commits and open tracker themes suggest the Now bet). Include the routing banner: sequencing, WSJF, and dated forecasts are produced by `/axiom-program-management`, never here. No dates, no WSJF scores.
3. **`metrics.md`** — seed a north-star and at least one guardrail, each with a **falsifiable** target as a `BASELINE → TARGET by <date>` placeholder for the human to set real numbers against. A directional word is not a metric; reject "improve engagement."
4. **`current-state.md`** — the resume brief: the inferred current bet, what the tracker shows in flight (by ID), the open questions bootstrap could not resolve, and where the next session starts.
5. **`decisions/`** — create the directory. Optionally seed `0001-bootstrap-from-observed-state.md` recording that the initial workspace was inferred (context → what was observed → the call → reversal trigger: "revisit once the human confirms vision and grant").

**Propose the authority grant via AskUserQuestion before writing it as authoritative.** The default escalation taxonomy is fixed by `product-ownership-operating-model.md` — do not invent a new list. The grant says the agent acts autonomously *within* strategy (prioritize, write PRDs, dispatch delivery, accept against criteria, reprioritize, kill a failing bet) and **escalates before** anything hard-to-reverse or outward-facing: changing the vision/strategy/grant, a public release or announcement, deprecating a feature users depend on, a pricing/commercial change, data deletion, or anything touching an external party. Ask the human to confirm or adjust that scope, then write the confirmed grant into `vision.md`. If the human is not present to confirm in this run, write the grant marked `Status: DRAFT — unconfirmed` and flag it loudly in the brief; never present an unconfirmed default as authoritative.

Then **Write all five files** (and `decisions/`) to the workspace path. These are the deliverable — real files, not a block. Do **not** commit; the first `/product-checkpoint` owns the commit.

## RESUME — load the workspace and reconcile against reality

The workspace exists; load it cold and reconstruct the running picture, then diff it against the world. Follow the RESUME protocol in `product-state-and-continuity.md`, in order:

1. **Read `current-state.md`** first — the hand-off note and fastest path to the running picture.
2. **Read `vision.md`** — reload purpose, anti-goals, and the **authority grant**, before deciding anything, so the session knows its own boundaries.
3. **Read `roadmap.md` and `metrics.md`** — the current bet and the scoreboard it will be judged against.
4. **Reconcile tracker IDs** — for each in-flight pointer in `current-state.md`, `read(id)` / `list(filter)` the tracker. An item the brief calls in-review may already be closed; the tracker is authoritative for tactical status, the brief for intent.
5. **Skim recent PDRs** — at minimum the ones `current-state.md` references — to recover *why* the current bet is the bet, so you do not relitigate it.

Then **ORIENT**: diff the loaded picture against reality and **flag drift explicitly**. Drift to surface includes: tracker items shipped/closed since the last checkpoint (the bet may now be acceptable, not in flight); a metric reading that crosses a PDR reversal trigger; recent commits that advanced or contradicted the recorded direction; and any new human input this session. Surface discrepancies — do not silently pick a side. RESUME and ORIENT are read-only; reconciling the drift into a decision is a `DECIDE` act that runs after this command.

## Always — surface the authority grant for confirmation

Whether bootstrapping or resuming, **end by surfacing the current authority grant** and asking the human to confirm it still holds. On bootstrap this is the proposal-and-confirm above. On resume, the grant was loaded from `vision.md`; restate it plainly ("you granted autonomy to prioritize, spec, dispatch, accept, and kill failing bets; you reserved releases, deprecations, pricing, data deletion, and any external-facing or vision change — still correct?") so the session acts under a *current, confirmed* grant, not a stale remembered one. If the grant's `Last reviewed` date is past its review cadence, say so. A widened or narrowed grant is itself a vision change — escalate it, do not edit it silently.

## Constraints on what this command produces

- **Writes files, never a copy-paste block.** The workspace is the deliverable; on bootstrap, the five artifacts and `decisions/` land on disk via `Write`. Emitting a block instead loses the continuity property the pack exists for.
- **Roadmap is intent only.** No dates, no WSJF, no sequencing — those are `/axiom-program-management`. Keep the routing banner in `roadmap.md`.
- **Metrics are falsifiable.** Every seeded target carries a number and a date against a BASELINE/TARGET placeholder; reject directional words.
- **The grant is confirmed before it is authoritative.** Use the fixed taxonomy; confirm via AskUserQuestion, or write `DRAFT — unconfirmed` and flag it.
- **Generic illustrations only.** Any example metric, bet, or anti-goal is an obvious placeholder (e.g. "reduce activation time from BASELINE to TARGET") — never a real client, organization, or domain.
- **Never commits.** The commit is the checkpoint boundary; this command stops at written files and an emitted brief.

## After running

1. State the branch taken (bootstrap vs resume) and why — the existence check result.
2. Emit the **current-state brief**: the bet right now, what is in flight (by tracker ID), open questions, and — on resume — the drift ORIENT found.
3. Emit **proposed next bets** as intent (top of Next), with the metric each is meant to move. These are proposals for `DECIDE`, not commitments.
4. Surface the **authority grant** for explicit confirmation.
5. Name the **closing seam**: this command ran `RESUME → ORIENT`; the middle of the loop (`DECIDE → DISPATCH → ACCEPT`) runs next in-session, where `DISPATCH` may invoke `/write-prd` for the top bet and route it to `/axiom-planning`; and **`/product-checkpoint`** writes state back and commits when the session ends. Make this seam visible so the human knows where durability comes from.

## Cross-references

- `using-product-management` — the router; load for the ownership discipline behind this command.
- `product-ownership-operating-model.md` — the `RESUME → ORIENT → DECIDE → DISPATCH → ACCEPT → CHECKPOINT` loop and the authority-boundary escalation taxonomy this command's grant uses.
- `product-state-and-continuity.md` — the exact schemas for `vision.md`, `roadmap.md`, `metrics.md`, `current-state.md`, the PDR template, the RESUME protocol, and the tracker-adapter contract this command builds and loads.
- `vision-strategy-and-roadmap.md` — shapes the vision and the Now/Next/Later bets seeded into the workspace as intent.
- `product-metrics-and-experimentation.md` — the kill/keep logic and target design behind the falsifiable metrics this command seeds.
- `/axiom-program-management` — owns sequencing, WSJF, and the dated forecast; the seeded roadmap is intent only and hands the committed bet there.
- `/product-checkpoint` — the write-back command that closes the loop, refreshes the workspace, and commits; this command never commits.
- `/write-prd` — turns the chosen top bet into a PRD with falsifiable acceptance criteria during `DISPATCH`.
