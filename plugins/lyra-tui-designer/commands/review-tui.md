---
description: Critique an existing TUI design or codebase against the terminal-UX failure modes — dispatches the tui-design-reviewer agent and returns severity-rated findings, each citing the sheet that resolves it
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[tui_design_or_code_to_review]"
---

# Review TUI Command

You are reviewing an existing terminal UI — a design write-up, a codebase, or a single render path — against the failure modes that make a TUI a bad place to *be*. Your role is to locate the artifacts, dispatch the `tui-design-reviewer` agent, and surface its severity-rated findings, **each citing the reference sheet in this pack that resolves it.**

## Core Principle

**A TUI that flickers, lags, eats keystrokes, or leaves the terminal wrecked on crash is a UX failure — not a cosmetic one.**

The user lives inside one window of fixed cells. They cannot zoom, cannot resize a button, cannot escape a half-drawn frame. Every mechanical defect is felt as friction in the user's hands and eyes. So this review treats rendering, latency, input, and terminal restoration as *experience* defects with the same weight as a confusing layout — because to the user they are indistinguishable.

**Accuracy over comfort. Evidence over opinion.** A review's job is to name what's wrong with a `file:line` or a screen state, not to reassure. A rubber-stamp is worse than no review: it launders a brittle TUI as ready.

## Invocation path

`/review-tui` is a Claude Code slash command. The command locates the artifacts, dispatches the `tui-design-reviewer` agent (the persona that walks the failure modes), then presents the findings and writes them to disk. For forward design of a *new* TUI rather than critique of an existing one, route to `/design-tui` and the `tui-architect` agent instead.

## Preconditions

The argument may be a directory, a source file, a design doc, or a description. Locate what is actually there before reviewing.

```bash
TARGET="${ARGUMENTS:-.}"

# Is it a path, and what kind?
ls -ld "${TARGET}" 2>/dev/null

# If a tree: find the TUI surface. Frameworks leave fingerprints.
grep -rlE 'ratatui|crossterm|termion' "${TARGET}" 2>/dev/null   # Rust
grep -rlE 'from textual|import textual|rich\.' "${TARGET}" 2>/dev/null  # Python
grep -rlE 'bubbletea|lipgloss|tea\.Model' "${TARGET}" 2>/dev/null  # Go
grep -rlE "from 'ink'|require\('ink'\)" "${TARGET}" 2>/dev/null  # JS
grep -rlE 'notcurses' "${TARGET}" 2>/dev/null  # notcurses
```

**Stop condition:** if `${TARGET}` resolves to nothing reviewable — no path, no source, no design text — stop and report: `No TUI artifact found at ${TARGET}. Provide a path, a source file, or a design description.`

**Gap, not stop:** if you have only a design doc with no code (or only code with no design rationale), proceed and record the limitation in Information Gaps — some failure modes (redraw discipline, signal handling) are only fully evidenced in code; others (information hierarchy) read fine from a design.

## Protocol

### Step 1 — Establish substrate assumptions

Before judging a single frame, pin down what the TUI assumes about its terminal. Most TUI defects are *unstated substrate assumptions* — color depth, Unicode width, terminal size, TTY-ness — that hold on the author's machine and break everywhere else. Note which of these the artifact addresses and which it silently assumes. (Sheet: `terminal-substrate-and-constraints.md`.) This frames the whole review.

### Step 2 — Dispatch the reviewer

Invoke the `tui-design-reviewer` agent. Pass it the resolved target, the detected framework(s), and any specific concern the user raised. The agent walks the failure modes below against the available artifacts and returns a severity-rated findings list with evidence, each finding naming the resolving sheet.

If the agent is unavailable in this environment, the command performs the walk directly using the same failure-mode checklist and severity bands defined here.

### Step 3 — Present findings

Surface results in this order:

- **Summary** — verdict (Solid / Acceptable / Needs Work / Hostile), counts by severity, substrate scope from Step 1.
- **Findings** grouped Critical / Major / Minor, each with `file:line` or a described screen state, a recommendation, **and the resolving sheet**.
- **What the TUI does well** — genuine strengths only; no rubber-stamping.
- **Information Gaps** and **Caveats**.

Then **write** the review to disk (see Output Location).

## Failure Modes → Resolving Sheets

The reviewer checks these. Each maps to the sheet that fixes it; every finding must cite one. Lead each mechanics finding with its UX stake — *what the user feels* — not just the technical defect.

| # | Failure mode (what the user suffers) | Resolving sheet |
|---|---|---|
| 1 | **Wrecked terminal on exit** — crash/Ctrl-C leaves the cursor hidden, raw mode on, alt-screen stuck, mouse-tracking spewing escape codes. The user's shell is now unusable. | `lifecycle-signals-and-terminal-restoration.md` |
| 2 | **Flicker / tearing / full-screen repaint** — the screen strobes on every keystroke or tick; eyes can't settle. | `rendering-and-redraw-discipline.md` |
| 3 | **UI freezes during work** — a fetch/compute blocks the event loop; keystrokes queue, the spinner stops spinning, the app looks hung. | `feedback-latency-and-async-work.md` |
| 4 | **No feedback** — an action fires and nothing visibly changes; the user re-presses, double-submits, or assumes it broke. | `feedback-latency-and-async-work.md` |
| 5 | **Tangled state / unpredictable updates** — modes leak, focus is ambiguous, the same key does different things for no legible reason. | `event-loop-and-state-architecture.md` |
| 6 | **Eaten or mis-routed keystrokes** — filter vs list vs global bindings collide; focus is invisible; Esc does five things. | `input-keyboard-mouse-and-focus.md` |
| 7 | **Breaks at 80 columns / on resize** — panels overlap, text truncates mid-word, SIGWINCH corrupts the layout. | `layout-and-responsive-composition.md` |
| 8 | **Wall of undifferentiated data** — everything shown at once, no hierarchy, no progressive disclosure; the signal drowns. | `information-density-and-progressive-disclosure.md` |
| 9 | **Hidden features** — keybindings live in the author's head; no footer legend, no `?` help, no command palette; needs a README to operate. | `affordances-and-discoverability.md` |
| 10 | **Color-only meaning** — status conveyed by color alone; collapses under `NO_COLOR`, a pipe, a screenshot, a colorblind user, a mono terminal. | `color-theming-and-monospace-canvas.md` |
| 11 | **Inaccessible** — screen-reader-hostile, color-dependent, no keyboard path; excludes users the terminal otherwise serves well. | `accessibility-in-the-terminal.md` |
| 12 | **Breaks off the author's machine** — assumes truecolor, a Nerd Font, a 120-col window, a real TTY; fails over SSH / in CI / piped / in a thin terminal. | `distribution-and-cross-environment.md` + `terminal-substrate-and-constraints.md` |
| 13 | **Verified only by eye** — "I ran it and it looked right"; no golden-frame / snapshot / headless test, so layout and render regressions ship silently. | `testing-tuis.md` |

A finding that cites no sheet is incomplete — either it maps to one of these or it is out of scope for this pack.

## Severity bands

Surface these with the same definitions. Severity is the reviewer's call, not the user's; the *response* to a finding (fix, waive, defer) is the maintainer's.

| Severity | Meaning | Examples |
|----------|---------|----------|
| **Critical** | Wrecks the user's environment or makes the TUI unusable. Fix before ship. | Terminal not restored on panic/SIGINT/SIGTERM (#1); event loop blocks indefinitely on I/O (#3); layout corrupts on resize so content is unreadable (#7); no keyboard path to a core function (#11). |
| **Major** | Significantly degrades the experience; works but hurts. | Per-keystroke full repaint causing visible flicker (#2); action with no feedback under ~100ms (#4); color-only status that vanishes under `NO_COLOR` (#10); core feature invisible with no hint or help (#9); truecolor/Nerd-Font assumption with no fallback (#12). |
| **Minor** | Polish; functional but suboptimal. | Inconsistent key hints, cosmetic spacing, a help overlay that exists but is hard to find, a snapshot test missing for a stable-but-low-risk panel (#13). |

## Prohibited patterns

- **Don't rubber-stamp.** "Looks great, ship it" is not a review. A review that finds nothing is suspicious — either name specific strengths or look harder. State counts: "Zero Critical, two Major (no SIGWINCH handling; color-only status), four Minor."
- **Don't sandwich.** Lead with the worst finding, not with praise. "Critical: terminal left in raw mode on panic" before "the layout is tidy."
- **Don't relabel under pressure.** "The flicker is minor, don't block" is not an input. If it strobes on every keystroke it is Major; the maintainer may *choose* to waive it, but the review states it as it is.
- **Don't review product intent.** The review checks whether the TUI is *robust, legible, and restorable* — not whether the tool should exist or whether the feature set is right.
- **Don't excuse "it works on my machine."** That is failure mode #12, not a defense against it. The user's terminal is not the author's.

## Handling pressure

- *"It always restores fine for me."* Then it is tested under SIGINT, SIGTERM, panic, and a killed parent — show that, or it is unverified (#1, #13). Manual happy-path runs don't cover the crash path, which is exactly where terminals get wrecked.
- *"TUIs can't really be tested."* That rationalization is what `testing-tuis.md` exists to refute — golden-frame, headless, and in-memory-backend testing exist for every major framework. The finding stands.
- *"Just review the layout, not the code."* A design-only review is valid but limited; redraw discipline and signal handling are only fully evidenced in code. Proceed and flag the limit in Information Gaps.

## Output Location

```bash
# Directory target → write inside it; file/description target → write alongside or to cwd
REVIEW_FILE="${TARGET%/}/tui-review-$(date +%Y-%m-%d).md"
```

If a review for today already exists, append `-v2`, `-v3` rather than overwriting — prior reviews are the change history.

## Output Format

```markdown
# TUI Review: [name]

## Summary
**Verdict:** Solid / Acceptable / Needs Work / Hostile
**Framework(s):** [detected]
**Substrate scope:** [what's assumed vs handled — from Step 1]
**Findings:** N Critical, N Major, N Minor

## Findings

### Critical
| Finding (UX stake first) | Evidence | Recommendation | Sheet |
|---|---|---|---|
| [Terminal left in raw mode on panic — user's shell is broken after a crash] | `src/app.rs:142` (no panic hook / no `Drop` restore) | Install a panic hook + RAII guard that disables raw mode, leaves alt-screen, shows cursor | lifecycle-signals-and-terminal-restoration.md |

### Major
[same table]

### Minor
[same table]

## What the TUI does well
- [Genuine strength with evidence]

## Information Gaps
- [What couldn't be assessed and why]

## Caveats
- [Scope of this review]
```

## Cross-Pack Discovery

```python
import glob
if glob.glob("plugins/lyra-ux-designer/.claude-plugin/plugin.json"):
    print("Available: lyra-ux-designer for general UX/IA/interaction critique beyond the terminal")
if glob.glob("plugins/muna-technical-writer/.claude-plugin/plugin.json"):
    print("Available: muna-technical-writer for status-bar / help-overlay microcopy review")
if glob.glob("plugins/ordis-quality-engineering/.claude-plugin/plugin.json"):
    print("Available: ordis-quality-engineering for building out the snapshot/CI test suite (#13)")
```

## Scope Boundaries

**Covered:** severity-rated critique of an existing TUI design or codebase against the 13 terminal-UX failure modes, each finding citing its resolving sheet; substrate-assumption framing; written review on disk.

**Not covered:** forward design of a new TUI (use `/design-tui` + `tui-architect`); rewriting the code; general non-terminal UX critique (use `lyra-ux-designer`); business-fit assessment.
