---
name: using-tui-designer
description: "Use when building or reviewing interactive terminal applications, especially capability detection, rendering, input/focus, terminal restoration, accessibility, and cross-environment behavior."
---

# Terminal UI Engineering and Design

A TUI borrows a capability-variable terminal shared with the user's shell.
Protect terminal state and responsiveness as part of the user experience.
Use ordinary CLI output for noninteractive tools; this pack concerns live input
or screen ownership.

## Workflow

1. Inspect the framework/version, event loop, current terminal capability probes,
   input handling, cleanup paths, and complaint or requested interaction.
2. Treat state as the source for rendering. Keep blocking work off the input loop;
   distinguish worker results, cancellation, errors and stale responses.
3. Layout against actual dimensions and Unicode cell width; define minimum-size,
   overflow, scrolling and resize behavior. Redraw from changed state and measure
   flicker/latency rather than assuming a buffer solves every issue.
4. Before mutating the terminal, establish cleanup ownership, including partial
   setup failure. Restore changed modes on supported normal/error/signal paths.
   Document limits such as SIGKILL, aborts and terminal disconnects; do not promise
   impossible cleanup guarantees.
5. Preserve keyboard exit/focus, discoverable actions, non-color status and a plain
   output path where appropriate. Validate with assistive technology before
   claiming screen-reader accessibility.
6. Test touched behavior through the existing headless/PTY harness and actual
   terminals when capability, signal or rendering behavior is in scope. Include
   relevant resize, crash/suspend, non-TTY, SSH/tmux or platform conditions.

## Retrieve by the actual uncertainty

References are beside this file; load only relevant sections.

| Need | Reference |
|---|---|
| TTY, Unicode, color and alternate screen | [terminal-substrate-and-constraints.md](terminal-substrate-and-constraints.md) |
| State/event loop and background work | [event-loop-and-state-architecture.md](event-loop-and-state-architecture.md), [feedback-latency-and-async-work.md](feedback-latency-and-async-work.md) |
| Geometry and rendering | [layout-and-responsive-composition.md](layout-and-responsive-composition.md), [rendering-and-redraw-discipline.md](rendering-and-redraw-discipline.md) |
| Key/paste/mouse/focus | [input-keyboard-mouse-and-focus.md](input-keyboard-mouse-and-focus.md) |
| Cleanup and signals | [lifecycle-signals-and-terminal-restoration.md](lifecycle-signals-and-terminal-restoration.md) |
| Color fallback, discoverability and density | [color-theming-and-monospace-canvas.md](color-theming-and-monospace-canvas.md), [affordances-and-discoverability.md](affordances-and-discoverability.md), [information-density-and-progressive-disclosure.md](information-density-and-progressive-disclosure.md) |
| Accessibility | [accessibility-in-the-terminal.md](accessibility-in-the-terminal.md) |
| Testing and distribution | [testing-tuis.md](testing-tuis.md), [distribution-and-cross-environment.md](distribution-and-cross-environment.md) |

Framework examples are illustrative; verify their APIs and failure paths against
the project's version. Pair language engineering or UX guidance only when needed.
`tui-architect` and `tui-design-reviewer` are optional bounded roles.

## Deliver

Return the implementation/design or prioritized findings with the failing state,
evidence and correction. Report tested terminals/platforms and unresolved cases.
A golden frame does not establish signal cleanup or accessibility; a clean normal
exit does not establish panic/partial-initialization restoration.
