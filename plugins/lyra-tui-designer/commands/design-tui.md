---
description: Design a new terminal UI (TUI) — interaction model, layout, keymap, state-architecture sketch, and degradation plan, framework-agnostic
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[app_or_feature_to_design]"
---

# Design TUI Command

You are doing **forward design** for a terminal user interface — taking an app or feature idea and producing a concrete, buildable design: an interaction model, a layout that survives resize, a keymap a touch-typist can trust, a state-architecture sketch that won't freeze, and a degradation plan for the terminals you don't control.

The deliverable is a **design package**, not code. Someone should be able to hand it to an implementer using ratatui, Textual, Bubble Tea, Ink, or notcurses and have them build the same app.

## Core Principle

**A TUI lives inside a substrate you do not own.** The user's terminal emulator, their `$TERM`, their colour theme, their font, their SSH latency, their window size, and their habit of hitting `Ctrl-C` when bored are all givens you design *around*. A TUI that flickers, lags behind keystrokes, garbles on a narrow pane, assumes truecolor, or leaves the shell in raw mode after a crash is a **UX failure** — no matter how clever the feature is.

So design for the hostile case first, then enhance for the comfortable one. Mechanics (redraw, event loop, signals) are not separate from UX — they *are* the UX in a terminal.

## Framework Stance

Stay **framework-agnostic**. Reach for whichever framework best illustrates a point — ratatui (Rust) for explicit immediate-mode redraw control, Textual + Rich (Python) for a retained widget tree and CSS-like layout, Bubble Tea (Go) for clean Elm/MVU message passing, Ink (JS) for React-style composition, notcurses for high-FPS/pixel-blitting. Name a concrete framework only to make a mechanic vivid; never assume the implementer is using one stack. If the user has already committed to a framework, design within its grain.

## Design Process

### Step 1: Establish Context

Use `AskUserQuestion` to fill gaps the brief leaves open. Ask only what you genuinely need:

1. **What is it?** Long-running dashboard/monitor, a one-shot picker/wizard, an editor/REPL, a full-screen app, or an inline (non-alt-screen) widget?
2. **Where does it run?** Local fast terminal, over SSH (latency!), inside tmux/screen, in CI/non-interactive, in a constrained terminal (8/16-colour, no mouse, narrow)?
3. **Who uses it and how often?** A daily-driver tool earns a dense expert keymap; a once-a-quarter tool must be discoverable cold.
4. **Input reality:** keyboard-only, mouse optional, or mouse-expected? Any users on screen readers (see `accessibility-in-the-terminal.md`)?
5. **What's the heavy work?** Network, disk, subprocesses, big buffers, tailing — anything that could block the loop or grow without bound.

### Step 2: Decide the Interaction Model

Pick the spine before pixels. Reference `affordances-and-discoverability.md` and `input-keyboard-mouse-and-focus.md`.

- **Modality:** modal (vim-like: normal/insert/command) vs modeless (every key always means one thing). State the trade: modal buys density and costs discoverability — pay the cost back with a persistent mode indicator.
- **Navigation:** focus order, focus ring visibility, how the user moves between panes/regions, and what "selected" looks like when the terminal can't draw a focus glow (reverse video, a `▌` gutter bar, a bracket `[ ]`).
- **Primary loop:** what the user does 80% of the time, expressed as the shortest possible key sequence.
- **Discoverability:** how a cold user learns the app — a footer hint bar, a `?` help overlay, command palette. A TUI with no visible affordances is a blank wall (see `affordances-and-discoverability.md`).

### Step 3: Lay Out the Screen

Design the layout as a **responsive composition**, not a fixed grid. Reference `layout-and-responsive-composition.md` and `information-density-and-progressive-disclosure.md`.

- Sketch the layout in ASCII at a realistic default size **and** at a hostile narrow size (e.g. 80×24 and 40×20). Show what collapses, stacks, hides, or scrolls.
- Define the cell budget: which regions are fixed, which flex, which have min-widths below which they hide entirely rather than garble.
- Apply progressive disclosure — what's always visible, what's one keystroke away, what lives behind a detail pane. Don't cram; an overflowing pane that truncates mid-word reads as broken.
- Account for wide/emoji/CJK glyphs (they occupy two cells) and never assume a monospace cell == one Unicode scalar (see `color-theming-and-monospace-canvas.md`).

### Step 4: Specify the Keymap

Produce a real keymap table, not "vim-like." Reference `input-keyboard-mouse-and-focus.md`.

- List every binding: key, context/mode, action, and whether it's discoverable from the UI.
- Honour platform reflexes: `Ctrl-C` should mean interrupt/quit unless you have a *very* good reason and a visible alternative; `q`/`Esc` to back out; arrows AND `hjkl` if your audience is mixed.
- Flag conflicts and terminal-stolen keys (many emulators eat `Ctrl-S`/`Ctrl-Q` for flow control, `Ctrl-Z` suspends, some keys are indistinguishable in legacy mode — `Tab`/`Ctrl-I`, `Enter`/`Ctrl-M`, `Esc` vs Alt-prefix).
- Decide the mouse contract: optional enhancement or required? If you grab the mouse, you may break the terminal's own copy/paste selection — say how the user still copies text.

### Step 5: Sketch the State Architecture

This is where responsiveness is won or lost. Reference `event-loop-and-state-architecture.md`, `feedback-latency-and-async-work.md`, and `rendering-and-redraw-discipline.md`.

- **Spine:** define the event loop as read-input → update-state → render. Prefer an MVU/Elm shape (Bubble Tea, Textual, redux-ish Ink): *render is a pure function of state; state changes only via messages; side effects run off the loop.*
- **Model sketch:** the core state shape, what's derived vs stored, and where "current selection / scroll offset / focus" live (one source of truth, not two that drift).
- **Off-loop work:** name every blocking thing (fetch, query, file read, subprocess) and how it gets off the loop (task/goroutine/async/worker thread) so input never lags. A frozen TUI is indistinguishable from a crashed one.
- **Feedback for in-flight work:** spinners, progress, optimistic updates, and debounced input — what the user sees in the gap (see `feedback-latency-and-async-work.md`).
- **Redraw discipline:** what triggers a repaint, how to avoid full-screen flicker (diffed/double-buffered draw, damage regions, frame-rate cap for tailing views) — see `rendering-and-redraw-discipline.md`.
- **Bounded buffers:** any growing log/output buffer gets a cap and an eviction rule, or the process bloats until the OOM killer wins.

### Step 6: Write the Degradation & Restoration Plan

The part GUI designers never have to think about, and the part that earns user trust. Reference `terminal-substrate-and-constraints.md`, `distribution-and-cross-environment.md`, and `lifecycle-signals-and-terminal-restoration.md`.

- **Capability tiers:** truecolor → 256 → 16 → 8 → no-color/`NO_COLOR`; mouse vs no mouse; Unicode/powerline glyphs vs ASCII fallback; alt-screen vs inline. Specify what each tier looks like — colour must never be the *only* signal (see `accessibility-in-the-terminal.md`).
- **Size limits:** the minimum viable size, and what the app shows below it (a polite "terminal too small" rather than a garbled grid).
- **Restoration contract:** on quit, crash, panic, `SIGINT`, `SIGTERM`, and `SIGWINCH`, the terminal is returned to a sane state — cooked mode restored, alt-screen exited, cursor shown, mouse reporting off, colours reset. A crash that leaves the next shell prompt invisible is the worst first impression a TUI can make (see `lifecycle-signals-and-terminal-restoration.md`).
- **Environment portability:** behaviour under tmux/screen, over SSH, in CI/non-TTY (detect and degrade to plain output), across emulators (see `distribution-and-cross-environment.md`).

### Step 7: Note How It'll Be Tested

Briefly point the implementer at a testing approach so the design is verifiable — snapshot/golden-frame tests of rendered output, simulated input/event injection, resize and capability-tier matrices. Reference `testing-tuis.md`.

## Using the SME Agent

For a deeper or higher-stakes design, or to pressure-test your own draft, invoke the **tui-architect** agent. Hand it the context from Step 1 plus any draft you've produced; it returns an architecture-level critique and recommendations across the event loop, layout, and degradation plan. Treat its output as expert review to fold back into the package, not as a replacement for the steps above.

## Design Output Format

Write the design package to a file (`Write`) when the user wants an artifact; otherwise present it inline. Structure:

```markdown
# TUI Design: [App / Feature Name]

## Context
- **Type:** [dashboard / picker / editor / wizard / inline widget]
- **Runs in:** [local / SSH / tmux / CI / constrained terminals]
- **Users & frequency:** [...]
- **Input model:** [keyboard-only / mouse-optional / mouse-expected]
- **Heavy work:** [network / disk / subprocess / tailing / large buffers]
- **Framework (if chosen):** [ratatui / Textual / Bubble Tea / Ink / notcurses / open]

## Interaction Model
- **Modality:** [modal / modeless] — trade-off and how it's mitigated
- **Primary loop:** [the 80% action, shortest path]
- **Focus & navigation:** [focus order, how "selected" is shown]
- **Discoverability:** [hint bar / `?` overlay / command palette]

## Layout

### Default (e.g. 100×30)
```
┌───────────────┬──────────────────────┐
│ [region A]    │ [region B]           │
├───────────────┴──────────────────────┤
│ [status / hint bar]                   │
└───────────────────────────────────────┘
```

### Narrow (e.g. 40×20)
```
┌───────────────────┐
│ [region A]        │   ← B collapses to a toggle
│ [hint bar]        │
└───────────────────┘
```

- **Fixed vs flex regions:** [...]
- **Min sizes / hide thresholds:** [...]
- **Progressive disclosure:** always-visible | one-key | detail pane

## Keymap

| Key | Context/Mode | Action | Discoverable? |
|-----|--------------|--------|---------------|
| `j` / `↓` | list | move down | hint bar |
| `Enter` | list | open selection | hint bar |
| `/` | any | filter | hint bar |
| `?` | any | help overlay | always |
| `q` / `Ctrl-C` | any | quit | hint bar |

- **Conflicts / terminal-stolen keys:** [...]
- **Mouse contract:** [optional / required; how copy/paste still works]

## State Architecture
- **Spine:** [event loop shape; MVU / retained tree / immediate-mode]
- **Model:** [core state shape; single source of truth for selection/scroll/focus]
- **Off-loop work:** [each blocking op → how it leaves the loop]
- **In-flight feedback:** [spinners / progress / optimistic / debounce]
- **Redraw:** [what triggers repaint; flicker/frame-cap strategy]
- **Bounded buffers:** [caps and eviction]

## Degradation & Restoration Plan
- **Colour tiers:** truecolor → 256 → 16 → 8 → `NO_COLOR` (non-colour signals: [...])
- **Mouse / Unicode / alt-screen fallbacks:** [...]
- **Minimum size & under-size message:** [...]
- **Restoration on quit/crash/SIGINT/SIGTERM/SIGWINCH:** [...]
- **Portability:** tmux/screen, SSH latency, CI/non-TTY, emulator variance

## Testing Notes
- [snapshot/golden-frame | input injection | resize & tier matrix]

## Open Questions / Risks
- [...]
```

## Reference Sheets

This command draws on the `using-tui-designer` sheets — load the relevant one when a step needs depth:

- `terminal-substrate-and-constraints.md` — what a terminal actually is and the limits it imposes
- `event-loop-and-state-architecture.md` — the spine: MVU, off-loop side effects
- `rendering-and-redraw-discipline.md` — flicker, diffed draws, frame caps
- `feedback-latency-and-async-work.md` — making the wait feel good
- `layout-and-responsive-composition.md` — responsive composition under resize
- `information-density-and-progressive-disclosure.md` — how much to show
- `input-keyboard-mouse-and-focus.md` — keymaps, focus, mouse contract
- `affordances-and-discoverability.md` — making a character grid teach itself
- `color-theming-and-monospace-canvas.md` — colour, themes, glyph-width reality
- `accessibility-in-the-terminal.md` — screen readers, colour-independence
- `lifecycle-signals-and-terminal-restoration.md` — never wreck the user's shell
- `distribution-and-cross-environment.md` — tmux/SSH/CI/emulator portability
- `testing-tuis.md` — snapshot, input injection, resize/tier matrices

## Scope Boundaries

**This command covers:**
- Forward design of a TUI: interaction model, layout, keymap, state-architecture sketch, degradation plan
- Framework-agnostic guidance with concrete framework examples

**Not covered:**
- Writing the implementation code
- Critiquing an existing finished TUI (that's a review task)
- GUI / web / mobile interface design (use `/create-interface`)
