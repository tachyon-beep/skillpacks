---
description: Use when designing, building, or reviewing a terminal user interface (TUI) — a full-screen or interactive terminal app — across any framework (ratatui/Rust, Textual+Rich/Python, Bubble Tea/Go, Ink/JS, ncurses/notcurses). Covers terminal substrate and capability detection, event-loop and state architecture, responsive layout, flicker-free rendering, keyboard/mouse/focus input, semantic color and theming, affordances and discoverability, feedback and async work, information density, terminal accessibility, signal-safe lifecycle and terminal restoration, testing, and cross-environment distribution.
---

# TUI Designer Routing

**A TUI is a *user interface*, not a CLI with cursor tricks.** The terminal is a hostile, capability-variable canvas shared with the user's shell — and a TUI that flickers, lags on every keystroke, eats input, or leaves a wrecked terminal behind on crash is a UX failure, not just a bug. When designing, building, or reviewing a full-screen or interactive terminal app, load this pack.

Use the `using-tui-designer` skill from the `lyra-tui-designer` plugin. Content authority lives in `plugins/lyra-tui-designer/skills/using-tui-designer/SKILL.md` — this wrapper is a thin pointer.

## When to Use

- Designing or architecting a terminal UI's event loop, state, and layout
- Achieving flicker-free, low-latency rendering and redraw discipline
- Handling keyboard, mouse, and focus input; semantic color, theming, NO_COLOR and capability fallback
- Affordances, discoverability, feedback, and async work in a terminal
- Terminal accessibility and information density
- Signal-safe lifecycle so a crash or Ctrl-C restores the terminal cleanly
- Golden-frame testing and cross-environment distribution
- Working in any TUI framework — ratatui (Rust), Textual + Rich (Python), Bubble Tea (Go), Ink (JS), notcurses/ncurses
- User mentions: "TUI", "terminal app", "ratatui", "Textual", "Bubble Tea", "ncurses", "curses"

**Don't use for**: general UI/UX design across web/mobile/desktop (use `/ux-designer`), static website or docs-site design (use `/site-designer`), document/PDF visual design (use `/document-designer`), or CLI argument/flag design without a full-screen interface (that is plain CLI ergonomics, not TUI design).

## Commands

- `/lyra-tui-designer:design-tui` — design or implement a terminal UI (event loop, layout, rendering, input, theming)
- `/lyra-tui-designer:review-tui` — critique an existing TUI against rendering, input, lifecycle, and accessibility failure modes
- `/lyra-tui-designer:audit-tui-accessibility` — audit a TUI for terminal accessibility (color, contrast, screen readers, NO_COLOR, capability fallback)

## Agents

- `tui-architect` — specialist subagent for terminal-UI architecture (event loop, state, layout, rendering, lifecycle) across frameworks. Follows SME Agent Protocol with confidence/risk assessment.
- `tui-design-reviewer` — specialist subagent that reviews a TUI design or implementation against the pack's failure-mode catalog. Follows SME Agent Protocol.

## Cross-references

- General UI/UX design across web, mobile, desktop, game surfaces → `/ux-designer`
- Static website / docs-site design → `/site-designer`
- Document/PDF visual design (Typst, Pandoc, branded templates) → `/document-designer`
