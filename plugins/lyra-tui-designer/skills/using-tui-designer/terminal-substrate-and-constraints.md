---
name: terminal-substrate-and-constraints
description: Use when a TUI assumes truecolor/24-bit color or emoji/unicode glyphs render everywhere, when raw mode or the alternate screen is entered without checking the output is a real TTY, when the same code runs identically whether stdout is a terminal or a pipe/file/CI runner, when boxes/borders/tables misalign because string length was used as on-screen width, when CJK or emoji or combining characters truncate mid-glyph and smear the layout, when output is garbled under `tmux`/`screen`/SSH/`dumb`/Windows consoles, or when "it works in my terminal" is the only test that has been run — covers the terminal cell grid, isatty detection, $TERM/$COLORTERM/terminfo capability probing, color-depth fallback (truecolor → 256 → 16 → mono), grapheme-cluster display width, and the not-a-TTY contract.
---

# Terminal Substrate and Constraints

## The UX stake

A TUI does not render onto a clean, known canvas the way a browser or a native window does. It renders onto **someone else's terminal** — an emulator you did not choose, configured by a user you will never meet, possibly forwarded over SSH, possibly multiplexed under tmux, possibly not a terminal at all but a pipe into `grep` or a CI log collector. Every assumption you bake in about that surface is a bet against a real user's setup.

When the bet loses, the user does not see "a capability mismatch." They see a broken program: a table whose right border zig-zags, error text they can't read because their theme remaps red, a wall of `\x1b[38;2;...m` escape garbage in their CI log, or — worst — a shell with echo turned off after your app entered raw mode on a pipe and never recovered. **A TUI that assumes the substrate is a UX failure before a single widget is drawn.** This sheet is about meeting the terminal where it actually is.

Everything downstream — color theming, layout, rendering discipline, lifecycle restoration — assumes you have *correctly characterized the substrate first*. Get this wrong and the other sheets are building on sand.

---

## 1. What a terminal actually is: a cell grid

Drop the mental model of "a window I draw pixels into." A terminal is:

- A **grid of character cells**, `columns × rows` (e.g. 120 × 40). You address cells, not pixels.
- Each cell holds **one user-perceived character** plus styling (foreground/background color, bold, underline, etc.). "One character" is doing a lot of work here — see §5.
- **Monospace by contract.** Every cell is nominally the same width. Your layout math depends on this — and it is exactly what wide characters and emoji violate.
- Driven by **in-band control sequences** (ANSI/VT escape codes): `\x1b[2J` clears the screen, `\x1b[1;1H` moves the cursor home, `\x1b[31m` sets red foreground. Text and commands share one byte stream. There is no separate "API channel."
- **Stateful and global.** Cursor position, current style, raw vs. cooked mode, alternate-screen flag — these are properties of the terminal, not your process. If you change them and exit without restoring, the user inherits your mess. (That restoration discipline is `lifecycle-signals-and-terminal-restoration.md`; this sheet is about not making bad assumptions about that state in the first place.)

Two consequences that drive the rest of this sheet:

1. **Capabilities vary wildly.** Whether `\x1b[38;2;r;g;bm` (24-bit truecolor) produces a precise color, an approximation, or literal garbage depends on the emulator. You must *detect*, not assume.
2. **The grid only exists if there's a terminal at all.** Pipe your output into a file and there is no cursor, no grid, no resize — just bytes. Code that assumes the grid will misbehave catastrophically there (§3).

A frameworks' job (ratatui, Textual, Bubble Tea, Ink, notcurses) is to abstract the escape-sequence layer and give you a cell buffer. It does **not** absolve you of understanding the substrate — most of the failures in this sheet are things the framework *cannot* decide for you because they depend on the user's environment at runtime.

---

## 2. TTY vs. pipe vs. file vs. CI: the `isatty` gate

The single most-skipped check in TUI code: **is my output actually connected to a terminal?**

```
$ myapp                 # stdout → terminal      → interactive TUI is appropriate
$ myapp | less          # stdout → pipe          → NOT a terminal
$ myapp > out.txt        # stdout → regular file  → NOT a terminal
$ myapp                 # under CI (GitHub Actions, Jenkins) → usually NOT a terminal
$ ssh host myapp         # may or may not allocate a PTY (ssh -t forces one)
```

`isatty(fd)` answers "is this file descriptor a terminal device?" It is the gate that decides whether the whole interactive machine should even start.

**The failure (RED #27, #29):** entering raw mode and the alternate screen unconditionally, then discovering output is a pipe. Raw mode on a non-TTY may error, may silently no-op, or may corrupt the stream. The alternate-screen enter/leave sequences become literal escape junk in the captured file. And on a crash there's no terminal to restore — so the cleanup you wrote does nothing, or worse, writes restoration escapes into someone's log.

**The contract:** check `isatty` *before* touching terminal state. If stdout is not a TTY, do not enter raw mode, do not enter the alternate screen, do not draw a UI. Fall back to plain, line-oriented output (the depth of that fallback is `distribution-and-cross-environment.md`; the *decision to branch at all* is here).

Check the **right** descriptor:
- Gate **interactive UI** on `stdout` being a TTY (that's where the grid lives).
- Read **keyboard input** only if `stdin` is a TTY.
- They can differ independently: `myapp < script.txt` has a non-TTY stdin but a TTY stdout; `myapp | tee log` has the reverse. Don't probe one and assume the other.

Also honor explicit overrides so users and CI can force behavior:
- `NO_COLOR` (any non-empty value) → disable color even on a TTY. (`color-theming-and-monospace-canvas.md` owns the color side; the detection lives here.)
- `CI` env var set → assume non-interactive even if a PTY happens to be allocated.
- `TERM=dumb` → no cursor addressing, no colors; treat as effectively non-interactive.
- A `--no-tty` / `--plain` flag, and ideally a `--force-tty` escape hatch for the rare user who *does* want UI through a pipe.

```python
# Python — Textual / Rich. Decide the mode before constructing the app.
import sys, os

def output_mode():
    if "--plain" in sys.argv:                 return "plain"
    if os.environ.get("CI"):                  return "plain"   # CI logs, not a screen
    if not sys.stdout.isatty():               return "plain"   # piped / redirected
    if os.environ.get("TERM", "") == "dumb":  return "plain"
    return "interactive"

if output_mode() == "interactive":
    MyApp().run()                 # alt screen, raw mode, event loop
else:
    for row in compute_rows():    # plain line output; no escapes, honors NO_COLOR downstream
        print(format_plain(row))
```

```go
// Go — Bubble Tea. golang.org/x/term provides IsTerminal.
package main

import (
	"os"
	tea "github.com/charmbracelet/bubbletea"
	"golang.org/x/term"
)

func main() {
	interactive := term.IsTerminal(int(os.Stdout.Fd())) &&
		os.Getenv("CI") == "" &&
		os.Getenv("TERM") != "dumb"

	if !interactive {
		renderPlain(os.Stdout) // line-oriented fallback; no alt screen, no raw mode
		return
	}
	// Bubble Tea defaults to the alt screen; only opt in once we know it's a TTY.
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		os.Exit(1)
	}
}
```

> Rule of thumb: **`isatty` is not optional and it is not an edge case.** Roughly a third of the ways your program gets invoked in practice are non-interactive (pipes, redirects, CI, cron, `xargs`). Branch on it at the top, once.

---

## 3. Capability detection: color depth, terminfo, $TERM / $COLORTERM

Once you know you *have* a terminal, you still don't know what it can do. Terminals advertise capability through environment variables and the terminfo database — imperfectly, and with decades of historical baggage.

### Color depth — the tiered model

There is no single "the terminal supports color" boolean. There is a ladder, and you pick the **highest tier the terminal actually supports**, then degrade:

| Tier | Escape form | Colors | How it's advertised |
|------|-------------|--------|---------------------|
| Truecolor / 24-bit | `\x1b[38;2;R;G;Bm` | 16.7M | `$COLORTERM` = `truecolor` or `24bit` |
| 256-color | `\x1b[38;5;Nm` | 256 | `$TERM` contains `256color` (e.g. `xterm-256color`) |
| 16-color (ANSI) | `\x1b[3Nm` / `\x1b[9Nm` | 16 | basic `$TERM` (`xterm`, `vt100`, `linux`) |
| Monochrome | no color | 1 | `$TERM=dumb`, `NO_COLOR` set, or not a TTY |

**The failure (RED #28):** emitting `\x1b[38;2;...m` truecolor unconditionally. On a terminal that only understands 256 or 16 colors, the result ranges from "wrong color" to "the rest of the line is interpreted as text" — visible escape garbage. Old `screen`, the Linux VT console, many CI shells, and locked-down corporate emulators do not do truecolor.

**Detection, in priority order:**

1. **Honor `NO_COLOR`** (any value) → monochrome, full stop. User intent overrides everything.
2. **`$COLORTERM` is `truecolor` / `24bit`** → 24-bit is safe.
3. **`$TERM` contains `256color`** → 256-color.
4. **`$TERM` is a known color terminal** (`xterm`, `screen`, `tmux`, `vt100`, …) → 16-color.
5. **`$TERM` is `dumb` / unset / a pipe** → monochrome.

Never *assume up* the ladder; only assume down. When unsure, the safe default is the lower tier — a slightly duller palette is invisible to the user; escape garbage is not.

```rust
// Rust — capability tier independent of any TUI framework.
// crossterm/ratatui can also report this; here's the substrate logic itself.
use std::env;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ColorDepth { TrueColor, Ansi256, Ansi16, Mono }

fn detect_color_depth() -> ColorDepth {
    // 1. User intent (https://no-color.org/) wins outright.
    if env::var_os("NO_COLOR").is_some() {
        return ColorDepth::Mono;
    }
    // 2. Explicit truecolor advertisement.
    if let Ok(ct) = env::var("COLORTERM") {
        if ct.eq_ignore_ascii_case("truecolor") || ct.eq_ignore_ascii_case("24bit") {
            return ColorDepth::TrueColor;
        }
    }
    let term = env::var("TERM").unwrap_or_default();
    // 3 / 5. dumb or unset → no color.
    if term.is_empty() || term == "dumb" {
        return ColorDepth::Mono;
    }
    // 3. 256-color advertisement.
    if term.contains("256color") {
        return ColorDepth::Ansi256;
    }
    // 4. Known color terminal → assume 16.
    ColorDepth::Ansi16
}

/// Map a desired RGB to whatever the terminal can actually render.
fn fg_escape(r: u8, g: u8, b: u8, depth: ColorDepth) -> String {
    match depth {
        ColorDepth::TrueColor => format!("\x1b[38;2;{r};{g};{b}m"),
        ColorDepth::Ansi256   => format!("\x1b[38;5;{}m", rgb_to_256(r, g, b)),
        ColorDepth::Ansi16    => format!("\x1b[3{}m", rgb_to_ansi16(r, g, b)),
        ColorDepth::Mono      => String::new(), // no color; rely on text/icon signal
    }
}
# fn rgb_to_256(_r:u8,_g:u8,_b:u8)->u8{0} fn rgb_to_ansi16(_r:u8,_g:u8,_b:u8)->u8{0}
```

> Most mature frameworks ship a detector (ratatui via crossterm/`supports-color`, Rich/Textual via its `ColorSystem`, Bubble Tea via `termenv`, Ink via `supports-color`). **Use the framework's detector rather than rolling your own** — but understand the ladder above so you know what it's deciding and can debug it when a user reports "colors are wrong on my box." The mistake this section closes is *bypassing* detection and emitting one fixed tier.

### Terminfo and other capabilities

Color is the loudest capability but not the only one you can't assume:

- **Alternate screen** (`smcup`/`rmcup`, `\x1b[?1049h`/`l`) — the off-screen buffer you switch to so the user's scrollback is preserved. Almost universal on modern emulators, but absent on `dumb` and some embedded consoles. Frameworks gate this for you; the point is *don't hand-emit the sequence unconditionally*.
- **Mouse reporting**, bracketed paste, focus events, synchronized output (`\x1b[?2026h`), the kitty keyboard protocol, hyperlinks (OSC 8), Sixel/iTerm/kitty inline images — all optional, all emulator-specific.
- **terminfo** (`terminfo(5)`, queried via `tput`/`tigetstr` or libraries like `ncurses`) is the canonical capability database keyed by `$TERM`. It maps capability names (`colors`, `smcup`, `cup`, `cuf1`) to the right escape strings *for that terminal*. It is more authoritative than sniffing `$TERM` substrings — but it is also frequently stale or wrong over SSH (the remote box may lack a terminfo entry for your local `$TERM`, which is why `TERM=xterm-256color` sometimes breaks on a minimal server). Treat terminfo as the source of truth where available, with the `$TERM`/`$COLORTERM` heuristics as a fallback.
- **Runtime queries**: some capabilities can be *asked* via Device Attributes / DECRQSS escape queries (write the query, read the reply with a short timeout). Powerful but fiddly — the reply lands in your input stream and you must not hang forever waiting for one. Reserve for cases where env detection genuinely isn't enough.

**The asymmetry that governs all of this:** an *unused* capability costs nothing (you simply don't get a fancier rendering). A *wrongly-assumed* capability costs the user a corrupted screen. So the safe direction is always to under-assume and let detection upgrade you.

---

## 4. The multiplexer / SSH / Windows reality

`$TERM` lies more often than you'd like, and the substrate has more variants than "xterm":

- **tmux / screen** rewrite `$TERM` to `screen` or `tmux-256color` and *gate* what passes through to the real terminal. Truecolor under tmux needs tmux's own passthrough enabled; otherwise your 24-bit escapes get downsampled or dropped regardless of the outer emulator. Detect the multiplexer (`$TMUX`, `$STY`) and don't be surprised when capabilities narrow.
- **SSH** can run with or without a PTY. `ssh host cmd` often has **no** TTY (so your `isatty` gate correctly drops to plain mode); `ssh -t host cmd` forces one. Latency also makes full-screen redraws painful — another reason redraw discipline (`rendering-and-redraw-discipline.md`) matters, but the *detection* that you're on a constrained link starts here.
- **Windows** is its own world: the legacy `cmd.exe`/conhost did not process ANSI by default until VT mode was enabled (`ENABLE_VIRTUAL_TERMINAL_PROCESSING`); Windows Terminal and the ConPTY layer changed this, but you cannot assume which one a user has. `$TERM` is often unset on native Windows. Cross-platform frameworks (crossterm, Textual, Bubble Tea) abstract this — but if you bypass them you must enable VT mode explicitly or emit nothing useful.

The takeaway is not "memorize every emulator." It is: **`$TERM` is a hint, not a guarantee; detect defensively, degrade gracefully, and never assume the loudest capability tier.**

---

## 5. Unicode display width and grapheme clusters: `len != width`

This is the bug that smears every border, and it is purely a substrate fact: **the number of bytes, code points, or `String.length` units in a string is not the number of terminal cells it occupies.** Monospace layout is computed in *cells*. If you measure anything else, your columns drift.

Three distinct lengths, all different:

| String | Bytes (UTF-8) | Code points | Grapheme clusters | **Display cells** |
|--------|---------------|-------------|-------------------|-------------------|
| `"hello"` | 5 | 5 | 5 | **5** |
| `"café"` (combining ´) | 6 | 5 | 4 | **4** |
| `"日本語"` (CJK) | 9 | 3 | 3 | **6** (wide!) |
| `"👍"` | 4 | 1 | 1 | **2** (wide) |
| `"👨‍👩‍👧"` (ZWJ family) | 18 | 5 | 1 | **2** |
| `"é"` (e + combining acute) | 3 | 2 | 1 | **1** |

The phenomena, each its own trap:

- **East Asian Wide / Fullwidth** characters (CJK ideographs, fullwidth forms, many emoji) occupy **2 cells**. Unicode Standard Annex #11 (East Asian Width) classifies these. If you slice a line to "40 characters" and it contains wide glyphs, it overshoots the column into the next pane — the classic right-border zig-zag (RED #8, and the cross-reference target in `color-theming-and-monospace-canvas.md`).
- **Combining marks / grapheme clusters** (UAX #29): a base character plus zero or more combining code points render as **one** cell. `e` + U+0301 is one grapheme, one cell, two code points. Iterating by code point splits the accent off; iterating by byte splits a code point and emits a mojibake `�`.
- **Zero-width** characters (ZWJ U+200D, variation selectors, ZWSP) occupy **0 cells** but are real code points. An emoji family sequence is 7 code points fused by ZWJ into **one** grapheme that renders in **2** cells — and even *that* width depends on whether the terminal supports the ligature (some show it as separate glyphs, widening unpredictably).
- **Ambiguous-width** characters (some box-drawing, some symbols) are 1 or 2 cells *depending on the terminal's CJK setting*. This is genuinely terminal-dependent; the conservative move is to avoid ambiguous-width glyphs in load-bearing layout.

**The failures this closes (RED #8 and the substrate root of the layout smears):**
1. Truncating by byte index → splits a UTF-8 sequence → `�` and a corrupted line.
2. Truncating/padding by code-point or `.length` count → ignores wide chars → columns overflow.
3. Slicing in the middle of a grapheme cluster → orphaned combining marks, broken emoji.

**The discipline:** measure and slice by **display width over grapheme clusters**, never by bytes, never by code-point count, never by your language's default string length. Use a width-aware library — they exist for every ecosystem:

| Ecosystem | Display-width / grapheme tooling |
|-----------|----------------------------------|
| Rust | `unicode-width` (cell width), `unicode-segmentation` (grapheme clusters) |
| Python | `wcwidth` / `wcswidth`; Rich computes cell length internally |
| Go | `mattn/go-runewidth`, `rivo/uniseg` (grapheme + width) |
| JS/TS | `string-width`, `grapheme-splitter` / `Intl.Segmenter` |
| C / curses | `wcwidth(3)` / `wcswidth(3)` (locale-dependent) |

```rust
// Rust — truncate a string to a column budget, measured in TERMINAL CELLS,
// never cutting inside a grapheme cluster.
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Fit `s` into `max_cols` cells. Appends an ellipsis (1 cell) if truncated.
fn truncate_to_width(s: &str, max_cols: usize) -> String {
    if s.width() <= max_cols {
        return s.to_string();
    }
    let budget = max_cols.saturating_sub(1); // reserve a cell for '…'
    let mut out = String::new();
    let mut used = 0;
    for g in s.graphemes(true) {            // iterate whole grapheme clusters
        let w = g.width();                  // 0, 1, or 2 cells
        if used + w > budget {
            break;
        }
        out.push_str(g);
        used += w;
    }
    out.push('…');
    out
}

// truncate_to_width("日本語のテキスト", 6) -> "日本…"  (6 cells, not 6 chars)
// truncate_to_width("hello world", 8)      -> "hello w…"
```

```javascript
// JS / Ink (React for the terminal). Pad a cell to an exact column width
// using DISPLAY width, so a table column stays aligned across CJK/emoji rows.
import stringWidth from 'string-width';

/** Right-pad `text` with spaces to exactly `cols` terminal cells (truncate if wider). */
function padToWidth(text, cols) {
  let width = stringWidth(text);
  if (width === cols) return text;
  if (width < cols) return text + ' '.repeat(cols - width);

  // Too wide: walk grapheme segments, accumulating display width.
  const seg = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
  let out = '';
  let used = 0;
  for (const { segment } of seg.segment(text)) {
    const w = stringWidth(segment);
    if (used + w > cols) break;
    out += segment;
    used += w;
  }
  return out + ' '.repeat(cols - used); // re-pad to the exact budget
}

// padToWidth('👍 done', 10) keeps the column aligned even though '👍' is 2 cells.
```

> The cardinal rule: **layout math is cell math.** Any place you compute a column count, a truncation point, padding, a centered title, a border length, or a cursor position, you are working in display cells over grapheme clusters. The moment you reach for `s.len()`, `len(s)`, or `s.length` to mean "how wide is this on screen," you have a latent layout bug that will surface the first time a user has a Japanese filename, an accented name, or a thumbs-up in their data.

---

## 6. "It works in my terminal != done" checklist

The author's terminal — a modern truecolor emulator, UTF-8 locale, generous size, no multiplexer, local not SSH — is the *most* forgiving substrate that exists. Shipping after testing only there is shipping untested. Before you call substrate handling done, walk this:

**Is-it-even-a-terminal**
- [ ] Output piped (`myapp | cat`) produces clean, escape-free text — no alt-screen junk, no cursor codes. (RED #29; depth in `distribution-and-cross-environment.md`.)
- [ ] Output redirected to a file (`myapp > out.txt`) is readable plain text.
- [ ] `isatty` is checked on the *correct* descriptor (`stdout` for UI, `stdin` for input) before raw mode / alt screen. (RED #27.)
- [ ] Runs non-interactively under CI (`CI=1`) without trying to draw a UI.
- [ ] `NO_COLOR=1` produces fully monochrome output, with error/status still distinguishable by text or icon, not color.

**Color tiers**
- [ ] Truecolor escapes are gated behind detection, not emitted unconditionally. (RED #28.)
- [ ] Forced 256-color (`TERM=xterm-256color`, no `COLORTERM`) renders without garbage.
- [ ] Forced 16-color (`TERM=xterm`) and `TERM=dumb` both degrade cleanly, not into escape soup.

**Unicode width**
- [ ] A row containing CJK (`日本語`) keeps borders aligned. (RED #8.)
- [ ] A row containing emoji, including a ZWJ sequence (`👨‍👩‍👧`), doesn't overflow its column.
- [ ] A combining-mark string (`café` with U+0301) isn't split mid-grapheme.
- [ ] Truncation of a wide-char line never emits `�` (no mid-codepoint cuts).

**Environments beyond your desktop**
- [ ] Inside tmux/screen — truecolor narrows gracefully; layout intact.
- [ ] Over SSH, both with a PTY (`ssh -t`) and without (`ssh host myapp` → plain fallback).
- [ ] A non-UTF-8 / `C` locale (`LANG=C`) degrades to ASCII box-drawing instead of garbage.
- [ ] Windows Terminal and (if you support it) legacy conhost, or the framework's documented Windows support.
- [ ] A genuinely small terminal (e.g. 40×10) — capability detection still runs; minimum-size handling is `layout-and-responsive-composition.md`'s job, but detection must not crash.

If you can't tick a box, you have an *assumption*, not a tested behavior. Quick ways to force conditions without owning ten terminals: `NO_COLOR=1 myapp`, `TERM=dumb myapp`, `TERM=xterm myapp`, `myapp | cat`, `myapp > /tmp/out`, `ssh localhost myapp`, run under `tmux new`, and feed CJK/emoji fixtures into your data.

---

## Common mistakes

- **Treating "supports color" as a boolean.** It's a four-tier ladder (truecolor → 256 → 16 → mono). Pick the highest *detected* tier and degrade; never emit a fixed tier. (RED #28)
- **Emitting truecolor escapes unconditionally.** `\x1b[38;2;...m` on a 16-color or `dumb` terminal becomes literal garbage on screen. Gate it behind `$COLORTERM`/terminfo detection.
- **Entering raw mode / the alternate screen before checking `isatty`.** On a pipe, file, or CI runner there is no terminal to put into raw mode; you corrupt the stream and your crash-cleanup escapes leak into the captured output. Gate the whole interactive machine on `stdout.isatty()`. (RED #27)
- **Probing the wrong descriptor.** UI-readiness is about `stdout`; keyboard input is about `stdin`. They differ under redirects; don't check one and assume the other.
- **Ignoring `NO_COLOR` / `CI` / `TERM=dumb`.** These are explicit user/environment intent and override your detection. Honoring them is cheap and expected.
- **Measuring on-screen width with `len()` / `.length` / byte count.** This is the root cause of every misaligned border. Wide chars are 2 cells, combining marks are 0, ZWJ sequences fuse. Use a display-width library over grapheme clusters. (RED #8)
- **Slicing strings by byte or code-point index.** Splits UTF-8 sequences (→ `�`) or grapheme clusters (→ orphaned accents, broken emoji). Slice by grapheme cluster, budgeting by display width.
- **Trusting `$TERM` as ground truth.** It's rewritten by tmux/screen, often stale over SSH, frequently unset on Windows. Prefer terminfo where available; treat `$TERM` substrings as a fallback heuristic; under-assume when unsure.
- **Hand-emitting the alternate-screen / cursor / mode sequences instead of using the framework's gated API.** The framework already conditions these on capability and platform; bypassing it reintroduces every bug above.
- **Testing only in the author's terminal.** The dev box is the most capable, most forgiving substrate there is. Walk the §6 checklist; an untested condition is an assumption, not a feature.

---

## Where this sits in the pack

- This sheet decides **what the substrate is and can do.** Restoring it safely on exit/crash is `lifecycle-signals-and-terminal-restoration.md`.
- **Color tiers** detected here feed `color-theming-and-monospace-canvas.md` (which owns NO_COLOR fallback, non-color error signaling, and the monospace canvas); the *wide-char layout smear* is co-owned — width measurement is defined here, its layout consequences there.
- The **not-a-TTY branch** opens here (the `isatty` decision) and is fully realized in `distribution-and-cross-environment.md` (depth of the plain-output fallback and cross-emulator/SSH/CI/Windows coverage).
- **Redraw cost** under constrained links (SSH, slow terminals) is why detection matters, but the discipline lives in `rendering-and-redraw-discipline.md`.
- Forcing these conditions in automated tests (piped output, `TERM=dumb`, CJK fixtures, snapshot frames) is `testing-tuis.md`.
