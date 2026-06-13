---
description: Audit a TUI for terminal accessibility and graceful degradation — NO_COLOR, color-blind safety, no-TTY/CI fallback, reduced motion, visible focus, and honest screen-reader claims
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[tui_to_audit]"
---

# Audit TUI Accessibility & Degradation

You are auditing a terminal user interface for **accessibility and graceful degradation**. The target is `$1` (a repo path, a binary/command name, or a description of the TUI).

A TUI is a visual medium painted on top of a stream of bytes, and that gap is where terminal accessibility lives — and where most TUIs quietly exclude people. This audit is **not** a WCAG-for-the-web checklist transplant. The terminal has its own failure modes: state signaled by color the user can't receive, output that becomes garbage the instant it isn't a live TTY, motion that floods an SSH link, and accessibility claims no one ever tested.

**Core stance (carry it through every finding): in the terminal, accessibility is mostly _restraint and honesty_.** Don't lean on channels (color, motion, spatial position) some users can't receive; don't claim assistive-technology support you haven't earned; always leave an escape hatch (`NO_COLOR`, a `--plain`/line mode, a non-visual status path). The accessible choice is almost always the more robust choice for _everyone_ — a status that carries an icon _and_ a word survives a screenshot, a `grep`, a colorblind reviewer, a black-and-white console, and a 2-bit serial link. Color-only survives none of them.

This command draws its rules from two reference sheets — read them if you need depth on any finding:
- **`accessibility-in-the-terminal.md`** — honest screen-reader reality, color-blind-safe palettes, 16/256-color contrast under an unknown theme, reduced-motion, visible/redundant focus, and the `NO_COLOR` convention.
- **`color-theming-and-monospace-canvas.md`** — semantic-role colors, the truecolor→256→16→monochrome fallback ladder, `NO_COLOR` mechanics, light/dark polarity, box-drawing ASCII fallback, and Unicode display-width math.

---

## Step 0: Scope the target

Determine what you can actually inspect, then ask only for what's missing.

1. If `$1` is a path, `Glob`/`Grep` the source for the framework and the load-bearing primitives (see Step 1's evidence list). If it's a command name, try to locate its source or at least run capability probes against it (Step 2).
2. If you cannot tell what you're auditing, or can't determine whether a `--plain`/line mode exists, ask:

Use **AskUserQuestion** to resolve blocking gaps. Good questions:
- "What is the TUI built with?" (ratatui/Rust · Textual+Rich/Python · Bubble Tea/Go · Ink/JS · notcurses · other) — frameworks differ in what they give you for free (Rich auto-honors `NO_COLOR`; raw ratatui does not).
- "Can I run the binary in this environment, or is this a static source audit?" (run probes vs. read-only).
- "Has anyone ever opened this in a screen reader, or is the goal to find out whether you can honestly claim support?"

Do **not** invent a framework or assume a `--plain` mode exists. Absence of an escape hatch is itself a top finding.

---

## Step 1: Static evidence pass (source available)

`Grep` for the tells. Each pattern below maps to a specific finding dimension; cite file:line in the report.

| Look for | Pattern (adapt per language) | Smells of |
|----------|------------------------------|-----------|
| Color-as-meaning | `Color::Red`, `"red"`, `\x1b[31m`, `fg=red`, `.foreground(...)` near error/ok/warn logic | Color-only state signaling |
| Hardcoded literals | hex `#`, RGB triples, ANSI indices, color *names* outside a theme/role module | No fallback ladder; can't honor `NO_COLOR`/light themes |
| `NO_COLOR` handling | `NO_COLOR`, and crucially `== "1"` / `== "true"` / boolean parse of it | Missing, or spec-violating (must be *presence*, any value) |
| TTY detection | `isatty`, `is_terminal`, `IsTerminal`, `isTTY` | Whether output degrades when piped/redirected/CI |
| Plain/line mode | `--plain`, `--no-tui`, `--no-color`, `--ascii`, `--reduce-motion` flags | The genuinely accessible path — present or absent |
| Motion | spinner frames, `tea.Tick`, `setInterval`, timers driving redraw; `\x1b[5m` / `blink` | Reduced-motion path; fps cap; seizure hazard |
| Focus | border *color* swapped on focus with no border-*style*/marker change | Focus shown by color alone |
| Width math | `len(`, `.length`, `.chars().count()`, `s[:n]` slicing on display text | Wide/zero-width glyph smear (CJK/emoji) corrupting layout |
| Box-drawing | `┌─┐│╭╮`, rounded/heavy glyphs without a UTF-8 probe + ASCII fallback | Mojibake / sheared frames over Latin-1/`LANG=C` |
| AT claims | "accessible", "screen-reader" in README/docs/comments | Possibly an untested (false) claim |

---

## Step 2: Dynamic degradation probes (binary runnable)

These are the terminal's equivalent of "unplug the mouse." Run the target under each stripped condition and observe whether state survives. Use `Bash`; capture output for the report.

- **`NO_COLOR` strip** — run with `NO_COLOR=1`. Every state (error/warn/ok/selected/focused) must remain distinguishable by glyph/label/shape. If error and success become identical, that's a Critical color-only finding.
- **Not-a-TTY / pipe** — pipe stdout to a file or `cat` (`<cmd> | cat`, `<cmd> > out.txt`). It must **not** emit a flickering alt-screen, raw escapes, or garbage; it should degrade to clean lines (or refuse interactivity gracefully). This is the path a screen reader, a `grep`, and CI all take.
- **`TERM=dumb` / low color depth** — run with `TERM=dumb` and with a 16-color `TERM`. Truecolor escapes must down-convert, not print garbage.
- **`LANG=C` / non-UTF-8 locale** — run with `LANG=C LC_ALL=C`. Box-drawing must fall back to ASCII (`+ - |`), not mojibake; rows must stay aligned.
- **Reduced motion** — check for `--reduce-motion`/`--no-animations` and `REDUCE_MOTION`/`TUI_REDUCE_MOTION` env handling; confirm non-TTY runs disable animation automatically.
- **Wide-glyph stress** — feed a value containing CJK and/or emoji (e.g. `日本語ファイル`, a ZWJ emoji) into any column/table/title and check the frame doesn't shear.

If you cannot run the binary, mark these as "unverified — recommend running" rather than guessing pass/fail.

---

## Step 3: Score the audit dimensions

Assess each dimension, rate severity (Critical / Major / Minor), and cite evidence (file:line or probe output).

### Dimension 1 — Color is never the only channel
Every meaningful state must be distinguishable through **at least two of**: a word/label, a symbol/shape, position/structure — color is decoration, never the carrier. The target shape is unified diff: `+`/`-` carries the meaning, color is gloss. Check error/warn/ok status, selected rows (need a `▸`/`>` gutter marker, not just background color), required fields, and any categorical encoding (charts, multi-series sparklines). **The meaning inside the color is the bug.**

### Dimension 2 — Color-blind safety (when color IS used)
Red/green as the *only* distinction fails the most common CVD types (deuteranopia/protanopia) — they see two muddy tones. Fixes in order: carry meaning in glyph (solves it outright); differ pairs by *lightness* not just hue; use a CVD-safe categorical palette (Okabe–Ito) for charts/tags; add a non-color channel (line style, markers) to categorical data anyway.

### Dimension 3 — `NO_COLOR` & the fallback ladder
`NO_COLOR` is a cross-tool convention: **presence (any non-empty value, even `0`) disables color** — never a boolean parse. It disables color but **NOT** glyphs/labels — `✖ ERROR:` must still render and still mean error. Color depth detected once at startup, resolved truecolor→256→16→monochrome with explicit down-conversion (the terminal does *not* gracefully degrade a truecolor escape). Detection order: `NO_COLOR`→off; not-a-TTY→off; `FORCE_COLOR`→on; else probe. A `--color=auto|always|never` flag mirrors `ls`/`grep`/`git`.

### Dimension 4 — No-TTY / CI degradation
Piped, redirected, or CI output must become clean lines — no alt-screen thrash, no raw escapes, no spinner timers. The single most effective accessibility feature is a first-class `--plain`/line mode (auto-detected when not a TTY). Follow `git`/`kubectl`/`gh`/`cargo`: page interactively, pipe cleanly.

### Dimension 5 — Reduced motion & no flicker
Terminal motion is *uncapped* — no OS "reduce motion" toggle reaches it, and it floods CPU/SSH bandwidth. Honor `--reduce-motion`/`REDUCE_MOTION`; cap spinners to ~8–10fps (60fps gains nothing, costs 10×); **never `\x1b[5m` blink** (seizure hazard); animate *state*, not decoration; auto-disable animation when not a TTY. Prefer discrete "Step 3 of 7" updates over perpetual animation (also screen-reader-friendlier).

### Dimension 6 — Visible, redundant focus & keyboard operability
Everything reachable by keyboard (mouse is enhancement; many SSH sessions don't forward it). Focus indicated *redundantly* — border *style* (double vs. plain) and/or a `▸`/title marker, not border color alone. Standard keys do standard things (Tab/arrows/`Esc`/`Enter`/`q`/`Ctrl-C`/`?`). Don't trap focus; the user must always reach quit/help and leave — and leaving must restore the terminal.

### Dimension 7 — Honest screen-reader claims (read this hardest)
A full-screen, alt-screen, raw-mode TUI is, by default, **close to unusable with a screen reader**: no accessibility tree, in-place cursor moves instead of new lines, app-invented focus the OS can't follow, box-drawing announced as garbage. **Never claim "accessible" or "screen-reader friendly" for a full-screen TUI no one has opened in a screen reader.** The honest, useful posture: ship a `--plain` line mode (the path AT users *can* use), optionally surface critical status via OSC 9/777 desktop notifications, and state plainly: *"primarily a sighted, visual interface; for non-visual use, run `--plain`."* Flag any unverified AT claim in source/docs as a finding.

---

## Step 4: Report

Write the audit to `<repo>/docs/tui-accessibility-audit.md` (or print it if no writable path). Use this structure:

```markdown
# TUI Accessibility & Degradation Audit

**Target:** [path / command / description]
**Framework:** [ratatui / Textual+Rich / Bubble Tea / Ink / notcurses / …]
**Date:** [date]
**Method:** [static / dynamic probes / both]

## Verdict
**Overall:** [Robust / Partially degraded / Excludes users]
**Critical findings:** [count]   **`--plain` mode:** [present / ABSENT]   **`NO_COLOR` honored:** [yes / no / spec-violating]

| Dimension | Critical | Major | Minor | Verified by |
|-----------|----------|-------|-------|-------------|
| 1 Color-not-alone | | | | static/probe |
| 2 Color-blind safety | | | | |
| 3 NO_COLOR + ladder | | | | |
| 4 No-TTY / CI degradation | | | | |
| 5 Reduced motion / flicker | | | | |
| 6 Focus + keyboard | | | | |
| 7 Honest SR claims | | | | |

## Findings
| # | Dim | Severity | Evidence (file:line / probe) | Why it excludes someone | Fix |
|---|-----|----------|------------------------------|--------------------------|-----|
| | | | | | |

## Probe results
| Condition | Command run | Observed | Pass/Fail/Unverified |
|-----------|-------------|----------|----------------------|
| NO_COLOR=1 | | | |
| piped (\| cat) | | | |
| TERM=dumb / 16-color | | | |
| LANG=C | | | |
| reduced motion | | | |
| wide-glyph (CJK/emoji) | | | |

## Remediation, prioritized
1. **Critical** — fixes that make a state perceivable at all (color-only error; no `--plain`; alt-screen garbage when piped).
2. **Major** — degraded-but-usable (red/green-only categories; uncapped spinner; focus-by-color-only; spec-violating `NO_COLOR`).
3. **Minor** — polish (Okabe–Ito for charts; OSC notification for long jobs; ASCII border toggle).

## Honest accessibility statement (proposed)
[The one true sentence the project can ship — e.g. "This is primarily a sighted, visual TUI. For non-visual use, screen readers, and piping, run `--plain`." Only upgrade this if a screen-reader test was actually performed.]
```

---

## Red flags to call out explicitly

Surface these as Critical wherever found — they are the recurring terminal-accessibility failures:
- A color set to mean something with **no glyph/label** carrying the same meaning.
- "accessible"/"screen-reader friendly" written for a full-screen TUI with **no `--plain` mode** and **no screen-reader test**.
- `NO_COLOR` parsed as a boolean (`== "1"`/`"true"`) instead of presence-checked.
- Disabling color *also* disabling the ability to signal state.
- Only error/success distinction is **red vs. green hue**.
- A spinner/animation with **no reduced-motion path and no fps cap**, or any `\x1b[5m`/`blink`.
- Focus differing between panes **only by color**.
- An action that **requires the mouse**.
- Alt-screen escapes or raw output emitted when stdout **is not a TTY**.
- Width/pad/truncate done with `len()`/`.length`/index slicing on display text (CJK/emoji shear).

## Scope boundaries

**Covers:** terminal accessibility (color-not-alone, CVD safety, focus, keyboard), degradation under `NO_COLOR`/no-TTY/`TERM=dumb`/`LANG=C`, reduced motion, and honest screen-reader posture.
**Does not cover:** general TUI design critique (layout, density, affordances), performance/redraw tuning beyond its motion-cost angle, or lifecycle/terminal-restoration correctness — route those to the matching `using-tui-designer` reference sheets.

When finished, return a one-line confirmation naming the verdict, the critical-finding count, and the report path (e.g. `Audited <target>: Partially degraded, 3 critical findings — report at <path>`).
