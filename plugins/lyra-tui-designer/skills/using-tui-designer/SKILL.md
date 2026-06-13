---
name: using-tui-designer
description: Use when designing, building, or reviewing a terminal user interface (TUI) — a full-screen or interactive terminal app — across any framework (ratatui/Rust, Textual+Rich/Python, Bubble Tea/Go, Ink/JS, ncurses/notcurses). Covers terminal substrate and capability detection, event-loop and state architecture, responsive layout, flicker-free rendering, keyboard/mouse/focus input, semantic color and theming, affordances and discoverability, feedback and async work, information density, terminal accessibility, signal-safe lifecycle and terminal restoration, testing, and cross-environment distribution. Routes to 13 specialist sheets.
---

# Using TUI Designer

## Overview

This meta-skill routes you to the right TUI design skills based on your situation. Load this skill when you need terminal-UI expertise but aren't sure which specific sheet to use.

**Core Principle**: A TUI is a *user interface*, not a CLI with cursor tricks. The terminal is a hostile, capability-variable canvas shared with the user's shell — and a TUI that flickers, lags on every keystroke, eats input, or leaves a wrecked terminal behind on crash is a UX failure, not just a bug. Match your situation to the appropriate sheet, load only what you need.

**Framework-agnostic**: These sheets teach principles, then illustrate each with whichever framework shows it best — ratatui (Rust), Textual + Rich (Python), Bubble Tea (Go), Ink (JS/React), ncurses/notcurses. The discipline transfers; the API does not. Bring your own stack.

## When to Use

Load this skill when:
- Starting any TUI design, build, or review task
- User mentions: "TUI", "terminal app", "terminal UI", "full-screen terminal", "ratatui", "Textual", "Rich", "Bubble Tea", "Ink", "ncurses", "notcurses", "curses", "alternate screen", "raw mode"
- User describes terminal-app symptoms: "screen flickers", "it's laggy on keypress", "my terminal is broken after it crashes", "colors look wrong over SSH", "mouse doesn't work", "resize breaks the layout", "how do I test a TUI"
- You need to critique or review an existing terminal interface
- You need to design or build a new terminal interface or widget

**Don't use for**: Graphical desktop apps, web apps, or mobile apps. For application GUI (web app, desktop GUI, mobile, game, AI/chat surfaces) route to the sibling `lyra-ux-designer` pack (`/ux-designer`). For static documentation or marketing sites, route to `lyra-site-designer` (`/site-designer`). This pack is specifically for **interactive terminal applications** — the constrained, monospace, capability-variable surface that runs inside a terminal emulator. A plain non-interactive command-line tool (flags in, text out, exit) is a CLI, not a TUI; it generally needs none of this — reach here only once the program takes over the screen or reacts to live input.

---

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-tui-designer/SKILL.md`

Reference sheets like `rendering-and-redraw-discipline.md` are at:
  `skills/using-tui-designer/rendering-and-redraw-discipline.md`

NOT at:
  `skills/rendering-and-redraw-discipline.md` ← WRONG PATH

When you see a link like `[rendering-and-redraw-discipline.md](rendering-and-redraw-discipline.md)`, read the file from the same directory as this SKILL.md.

---

## Routing by Symptom / Task

Find your situation in the left column; load the sheet on the right. Most real tasks pull two or three sheets — the rightmost sheets in this table compose with the leftmost.

| Your situation (symptom or task) | Route to |
| --- | --- |
| "What can I even rely on?" — terminal size, truecolor vs 256, Unicode width, SSH/CI/no-TTY, capability detection, the alternate screen | [terminal-substrate-and-constraints.md](terminal-substrate-and-constraints.md) |
| "How should the app be structured?" — the event loop, where state lives, Model-View-Update, immutable view-from-state, avoiding spaghetti redraws | [event-loop-and-state-architecture.md](event-loop-and-state-architecture.md) |
| "The layout breaks on resize / different sizes" — splitting space, constraints vs fixed sizes, responsive reflow, min-size and overflow behavior | [layout-and-responsive-composition.md](layout-and-responsive-composition.md) |
| "The screen flickers / tears / redraws everything" — diffed/double-buffered rendering, draw-on-change not on-loop, frame budgets, partial redraw | [rendering-and-redraw-discipline.md](rendering-and-redraw-discipline.md) |
| "Keys do the wrong thing / mouse doesn't work / focus is lost" — key parsing, modifiers, paste, mouse events, focus model and tab order, keymaps | [input-keyboard-mouse-and-focus.md](input-keyboard-mouse-and-focus.md) |
| "Colors look wrong / unreadable over SSH / in light terminals" — semantic color roles, the truecolor→256→16→mono fallback ladder, NO_COLOR, theming | [color-theming-and-monospace-canvas.md](color-theming-and-monospace-canvas.md) |
| "Nobody can tell what this app does or how to use it" — discoverability without a manual, key hints, status/command bars, empty states, onboarding | [affordances-and-discoverability.md](affordances-and-discoverability.md) |
| "It freezes while it works / no progress shown" — keeping the loop responsive, async/background work, spinners and progress, cancellation, debounce | [feedback-latency-and-async-work.md](feedback-latency-and-async-work.md) |
| "Too much/too little on screen at once" — information density in a tiny grid, progressive disclosure, panes vs modals, scrolling, truncation strategy | [information-density-and-progressive-disclosure.md](information-density-and-progressive-disclosure.md) |
| "Is this usable with a screen reader / low vision / no color?" — honest terminal a11y, screen-reader reality, contrast, motion, never color-only signals | [accessibility-in-the-terminal.md](accessibility-in-the-terminal.md) |
| "It leaves my terminal broken after Ctrl-C / crash / SSH drop" — raw-mode and alternate-screen restoration, SIGINT/SIGTERM/SIGWINCH/SIGTSTP, panic hooks | [lifecycle-signals-and-terminal-restoration.md](lifecycle-signals-and-terminal-restoration.md) |
| "How do I test a TUI?" — golden-frame snapshots, headless drivers, simulating input and resize, asserting on rendered output, CI without a TTY | [testing-tuis.md](testing-tuis.md) |
| "Works on my machine, breaks on theirs" — cross-terminal/cross-OS robustness, tmux/SSH/CI, Windows, packaging, env detection, graceful degradation | [distribution-and-cross-environment.md](distribution-and-cross-environment.md) |

---

## Routing by Task Shape

### "Explain / teach me" (learning)

Start at the sheet that owns the concept (table above). For a from-scratch mental model of *why the terminal is hard*, begin with [terminal-substrate-and-constraints.md](terminal-substrate-and-constraints.md), then [event-loop-and-state-architecture.md](event-loop-and-state-architecture.md) — substrate and spine are the foundation everything else stands on.

### "Build a new TUI" (greenfield)

Load roughly in this order — this is the dependency spine:

1. [terminal-substrate-and-constraints.md](terminal-substrate-and-constraints.md) — what the canvas can and can't do; detect capability up front.
2. [event-loop-and-state-architecture.md](event-loop-and-state-architecture.md) — pick the loop/state shape before writing widgets.
3. [layout-and-responsive-composition.md](layout-and-responsive-composition.md) — compose space with constraints, not hard-coded coordinates.
4. [rendering-and-redraw-discipline.md](rendering-and-redraw-discipline.md) — render flicker-free, on-change.
5. [input-keyboard-mouse-and-focus.md](input-keyboard-mouse-and-focus.md) — wire input and focus.
6. **Always add** [lifecycle-signals-and-terminal-restoration.md](lifecycle-signals-and-terminal-restoration.md) — set up restore-on-exit and a panic hook *before* you first enter raw mode, not after the first time you nuke your own terminal.

Then layer in color/theming, affordances, feedback, density, and accessibility as the app grows; add testing and distribution before you ship.

### "Review / critique this TUI"

Default review set (load all three):
- [rendering-and-redraw-discipline.md](rendering-and-redraw-discipline.md) — flicker, full-redraws, frame budget.
- [lifecycle-signals-and-terminal-restoration.md](lifecycle-signals-and-terminal-restoration.md) — does it restore the terminal on every exit path, including panic/SIGTERM/SSH drop? (Highest-severity failure mode — a wrecked terminal is the one bug users never forgive.)
- [accessibility-in-the-terminal.md](accessibility-in-the-terminal.md) — color-only signals, contrast, screen-reader reality.

Then add the sheet matching the specific complaint from the symptom table.

### "Fix a specific bug"

Go straight to the symptom row in the table above. Common pairings:
- Flicker/tearing → rendering + (often) event-loop architecture.
- Laggy keystrokes / UI freezes → feedback-latency-and-async-work + event-loop.
- Broken terminal after exit → lifecycle-signals-and-terminal-restoration (alone is usually enough).
- Wrong colors over SSH/CI → color-theming + terminal-substrate (capability detection).
- Layout breaks on resize → layout + the SIGWINCH part of lifecycle-signals.

---

## Multi-Sheet Scenarios

### Build a scrollable, filterable list/table view
1. [layout-and-responsive-composition.md](layout-and-responsive-composition.md) (pane split, list region)
2. [input-keyboard-mouse-and-focus.md](input-keyboard-mouse-and-focus.md) (nav keys, focus, filter input)
3. [information-density-and-progressive-disclosure.md](information-density-and-progressive-disclosure.md) (truncation, scroll, detail-on-demand)
4. [rendering-and-redraw-discipline.md](rendering-and-redraw-discipline.md) (only redraw the changed rows)

### Add a long-running operation (fetch / index / build) without freezing the UI
1. [feedback-latency-and-async-work.md](feedback-latency-and-async-work.md) (background work, progress, cancellation)
2. [event-loop-and-state-architecture.md](event-loop-and-state-architecture.md) (message back into the loop, never block it)
3. [affordances-and-discoverability.md](affordances-and-discoverability.md) (show it's working and how to cancel)

### Make it robust enough to ship
1. [distribution-and-cross-environment.md](distribution-and-cross-environment.md) (tmux/SSH/CI/Windows, degradation)
2. [terminal-substrate-and-constraints.md](terminal-substrate-and-constraints.md) (capability detection drives degradation)
3. [color-theming-and-monospace-canvas.md](color-theming-and-monospace-canvas.md) (NO_COLOR, fallback ladder)
4. [testing-tuis.md](testing-tuis.md) (golden frames + headless CI runs)
5. [lifecycle-signals-and-terminal-restoration.md](lifecycle-signals-and-terminal-restoration.md) (no broken-terminal regressions)

---

## Cross-Pack Integration

### Lyra TUI + Lyra UX (sibling pack)
General UX principles — visual hierarchy, interaction feedback, the meaning of accessibility — live in `lyra-ux-designer` (`/ux-designer`). This pack is the *terminal specialization* of that discipline: the same principles re-derived for a monospace, capability-variable, signal-exposed canvas. Reach for `/ux-designer` for the cross-platform principle; reach here for how it actually plays out in a terminal.

### Lyra TUI + Axiom (Rust / Python / Go engineering)
For deep implementation in a specific stack, pair with the relevant engineering pack: `/rust-engineering` (ratatui/crossterm), `/python-engineering` (Textual/Rich). These sheets own *the UX and the terminal mechanics*; the engineering packs own *idiomatic code, error handling, and async runtime* in the host language.

### Lyra TUI + Ordis (Quality Engineering)
For the testing strategy *around* the golden-frame/headless tactics in [testing-tuis.md](testing-tuis.md) — test pyramid, flakiness, CI gates — pair with `/quality-engineering`.

---

## Specialist Sheet Catalog

The 13 sheets, in spine order (foundations first, delivery last):

1. [terminal-substrate-and-constraints.md](terminal-substrate-and-constraints.md) — The terminal as a canvas + capability detection (size, color depth, Unicode width, TTY presence, alternate screen).
2. [event-loop-and-state-architecture.md](event-loop-and-state-architecture.md) — The event-loop spine + Model-View-Update; where state lives and how the view derives from it.
3. [layout-and-responsive-composition.md](layout-and-responsive-composition.md) — Responsive constraint-based layout; splitting space, min-sizes, overflow, reflow on resize.
4. [rendering-and-redraw-discipline.md](rendering-and-redraw-discipline.md) — Flicker-free, diffed, event-driven rendering; draw-on-change, double buffering, frame budgets.
5. [input-keyboard-mouse-and-focus.md](input-keyboard-mouse-and-focus.md) — Keyboard, mouse, modifiers, paste, and a coherent focus/tab model with discoverable keymaps.
6. [color-theming-and-monospace-canvas.md](color-theming-and-monospace-canvas.md) — Semantic color roles, the truecolor→256→16→mono fallback ladder, NO_COLOR, light/dark, theming.
7. [affordances-and-discoverability.md](affordances-and-discoverability.md) — Making a TUI learnable without a manual: key hints, status/command bars, empty states, onboarding.
8. [feedback-latency-and-async-work.md](feedback-latency-and-async-work.md) — Responsive feedback and async work; never block the loop, progress/spinners, cancellation, debounce.
9. [information-density-and-progressive-disclosure.md](information-density-and-progressive-disclosure.md) — Designing for constrained space: density, panes vs modals, scrolling, truncation, detail-on-demand.
10. [accessibility-in-the-terminal.md](accessibility-in-the-terminal.md) — Honest terminal accessibility: screen-reader reality, contrast, motion, never relying on color alone.
11. [lifecycle-signals-and-terminal-restoration.md](lifecycle-signals-and-terminal-restoration.md) — Signal-safe terminal restoration (**highest severity**): raw mode + alternate screen cleanup across exit, panic, SIGINT/SIGTERM/SIGWINCH/SIGTSTP, SSH drop.
12. [testing-tuis.md](testing-tuis.md) — Golden-frame and headless TUI testing; simulating input/resize, asserting rendered output, CI without a TTY.
13. [distribution-and-cross-environment.md](distribution-and-cross-environment.md) — Cross-environment robustness and delivery; tmux/SSH/CI/Windows, packaging, env detection, graceful degradation.

**Cross-pack**:
- `lyra-ux-designer/*` (`/ux-designer`) — Cross-platform UX principles this pack specializes for the terminal.
- `axiom-rust-engineering/*`, `axiom-python-engineering/*` — Idiomatic implementation in the host language.
- `ordis-quality-engineering/*` (`/quality-engineering`) — Test strategy around TUI test tactics.
