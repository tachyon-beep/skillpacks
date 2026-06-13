---
name: color-theming-and-monospace-canvas
description: Use when picking colors for a TUI, hardcoding ANSI escape codes or hex/RGB SGR sequences, defining a theme or color palette, when output looks wrong over SSH or in tmux/screen or on a 16-color terminal, when NO_COLOR / COLORTERM / TERM is set and ignored, when an error is shown as red text only, when box-drawing characters or borders render as garbage (mojibake) or misaligned, when CJK / emoji / combining / wide glyphs truncate or smear a row, when light-terminal users see unreadable low-contrast text, or when "it looked fine on my machine" but breaks on someone else's terminal.
---

# Color, Theming, and the Monospace Canvas

## The UX stake

A terminal is not a blank canvas you own — it is a *guest room* in software the user already configured to their eyes, their hardware, and their accessibility needs. The user picked their colorscheme. The user set `NO_COLOR`. The user is on a 16-color serial console, or a high-contrast light theme, or a screen reader piping your output to braille. When your TUI hardcodes `\x1b[31m` for "error," paints unreadable gray-on-gray because you only tested dark mode, or smears an entire row because a CJK name was one column wider than you counted — that is not a cosmetic bug. **It is a UX failure**: the user cannot read the state of their own system in their own terminal.

The discipline of this sheet is a single inversion of instinct: **you do not choose colors and characters. You choose *roles*, and you let the environment resolve them.** Semantic role → capability-aware resolution → safe fallback. Every shortcut around that ladder is a future bug report titled "looks broken on my terminal."

Three things this sheet refuses to let you ship:
- **Color as the only channel.** Red text is invisible to a CVD user, a monochrome terminal, a `NO_COLOR` user, and a screen reader. Color is an *accent on* meaning, never the carrier *of* meaning.
- **Assumed capability.** Truecolor, 256-color, even box-drawing glyphs are *probed*, not assumed. The fallback ladder is mandatory, not optional polish.
- **Naive width math.** A "character" is not a column. The grid is measured in *display cells*, and getting that wrong corrupts every layout downstream.

## When to use

Load this sheet when you are:
- Choosing or specifying colors, defining a palette, or building a theme system
- Writing or reviewing any code that emits SGR / ANSI escapes (`\x1b[…m`), sets foreground/background, or styles text
- Debugging output that looks wrong over SSH, in tmux/screen, in CI logs, on a Linux VT, or on Windows
- Handling `NO_COLOR`, `COLORTERM`, `TERM`, `FORCE_COLOR`, or a `--no-color` flag
- Drawing borders, boxes, rules, separators, tables, or tree connectors
- Placing any text that may contain CJK, emoji, combining marks, or other non-ASCII

**Cross-links:** column/width math is *defined* in `terminal-substrate-and-constraints.md` (the grid model) — this sheet applies it to color and box-drawing. For "error must reach a non-visual user" see `accessibility-in-the-terminal.md`. For capability *detection* mechanics see `terminal-substrate-and-constraints.md`; for the non-TTY (piped) path see `distribution-and-cross-environment.md`.

---

## 1. Semantic color — roles, not literals

The cardinal rule: **never name a color, name what it means.** Your code refers to `theme.error`, never `red`. `theme.muted`, never `bright_black`. `theme.accent`, never `#7aa2f7`.

Why this is non-negotiable:
- A literal color is a decision frozen at the wrong layer. `red` cannot become "the user's configured danger color," cannot dim to a 16-color approximation, cannot vanish under `NO_COLOR`, cannot flip for a light theme. A *role* can do all four because resolution happens later, where capability is known.
- Literals scatter the same decision across the codebase. When you discover your "success" green is unreadable on a light terminal, a role is one edit; literals are a grep-and-pray.

### The role vocabulary

Define a small, closed set of roles. Resist the urge to add a role per widget — roles are *meanings*, and there are surprisingly few:

| Role | Meaning | Typical accent |
|------|---------|----------------|
| `text` / `fg` | Default foreground | terminal default |
| `bg` | Default background | terminal default |
| `muted` / `dim` | De-emphasized, metadata, timestamps | dim / gray |
| `accent` / `primary` | Focus, selection, the one thing that matters now | brand hue |
| `success` | Completed, healthy, passing | green family |
| `warning` | Degraded, attention, recoverable | yellow/amber family |
| `error` / `danger` | Failed, destructive, blocking | red family |
| `info` | Neutral notice | blue/cyan family |
| `border` | Structural lines, separators | muted |
| `selection_bg` | Background of the focused row | accent-tinted |

Everything else is a *combination* of a role plus an attribute (bold, dim, reverse, underline). A "selected error row" is `error` foreground + `selection_bg` background + bold — not a new color literal.

### The rule that color is never alone

Pair every semantic color with a **redundant non-color channel**. This is the single most-violated rule in TUIs and the root of RED finding #6 (error signaled by color alone).

| State | Color (accent) | Redundant channel (the actual signal) |
|-------|----------------|----------------------------------------|
| Error | red | prefix `ERROR` / `✗` / `[!]`, bold, log level word |
| Warning | yellow | prefix `WARN` / `⚠` / `[~]` |
| Success | green | prefix `OK` / `✓` / `[+]` |
| Selected row | accent bg | `▸` gutter marker / reverse video / `>` |
| Disabled | dim | `(disabled)` text / parenthesization |

If you remove all color from your TUI (which `NO_COLOR` and monochrome terminals literally do), **the meaning must survive completely**. Test this: run your app with color stripped and confirm every state is still distinguishable. If error and success look identical in monochrome, you have a color-only design and it excludes CVD users, monochrome terminals, `NO_COLOR` users, and screen-reader users in one stroke.

---

## 2. The fallback ladder: truecolor → 256 → 16 → monochrome

A role resolves to a concrete escape sequence *only at render time*, against the terminal's detected color depth. There are four rungs, and a correct TUI handles all of them:

```
24-bit truecolor   COLORTERM=truecolor|24bit   → exact RGB  (\x1b[38;2;R;G;Bm)
256-color (8-bit)  TERM=*-256color             → nearest indexed (\x1b[38;5;Nm)
16-color (ANSI)    baseline TERM=xterm/vt100    → nearest of 16 (\x1b[3Nm/\x1b[9Nm)
monochrome         NO_COLOR / dumb / not a TTY  → no color; attrs + glyphs only
```

**Detect once, at startup. Cache the result.** Probing per-frame is wasteful and racy.

Detection inputs, in priority order:
1. `NO_COLOR` present (any value, even empty) → force monochrome. Highest priority. (See §3.)
2. Output is not a TTY (piped/redirected) → monochrome unless `FORCE_COLOR` overrides. (See `distribution-and-cross-environment.md`.)
3. `COLORTERM` ∈ {`truecolor`, `24bit`} → truecolor.
4. `TERM` contains `256color` → 256.
5. `TERM` ∈ {`dumb`, ``} → monochrome.
6. Otherwise → 16-color baseline.

### Down-conversion: define it, don't hope for it

The terminal does **not** gracefully degrade a truecolor escape on a 16-color terminal — it may show a wrong color, ignore it, or print garbage. *You* down-convert. Two correct strategies:

- **Quantize**: map the truecolor RGB to the nearest entry in the 256-cube, then (for 16-color) to the nearest ANSI base. Good general default; libraries (`anstyle`, `termcolor`, Rich, `lipgloss`) ship this.
- **Curated tiers** (better for brand-critical themes): hand-author each role at each rung so your "accent" is intentional at 16 colors, not an algorithmic surprise.

A theme entry therefore looks like this (Python / Textual-style theme dict, abbreviated):

```python
THEME = {
    "error": Color(
        truecolor="#f7768e",   # exact
        ansi256=210,           # nearest xterm-256 index
        ansi16="bright_red",   # one of the 16
        attrs=("bold",),       # survives even in monochrome
        glyph="✗",             # redundant channel
        ascii_glyph="x",       # when even Unicode is unsafe (see §5)
    ),
    ...
}
```

The key insight: `attrs` and `glyph` are present at *every* rung, including monochrome. Color is the thing that falls away; meaning is not.

---

## 3. Honoring `NO_COLOR`

`NO_COLOR` (https://no-color.org) is a cross-tool informal standard with one rule: **if the environment variable `NO_COLOR` is present — regardless of its value, including empty string — the program must not emit color.** This closes RED finding #7.

Correct handling:

```rust
// Rust — the check is presence, NOT truthiness.
let no_color = std::env::var_os("NO_COLOR").is_some();   // empty string still counts
let force    = std::env::var_os("FORCE_COLOR").is_some();
let is_tty   = std::io::stdout().is_terminal();          // std::io::IsTerminal

let use_color = !no_color && (force || is_tty);
```

Common ways teams get this wrong:
- Checking `NO_COLOR == "1"` or `if no_color == "true"` — **wrong**, presence is the signal, not the value. `NO_COLOR=` (empty) must still disable color.
- Honoring `NO_COLOR` for some output paths but not others (the help text obeys, the error banner doesn't). It is all-or-nothing.
- Treating `NO_COLOR` as "remove foreground colors" but leaving background fills, reverse video as decoration, or 256-color spinners. Monochrome means *no color* — keep only attributes (bold/dim/underline/reverse) and glyphs.

Precedence, codified: `NO_COLOR` (off) beats `FORCE_COLOR` (on) — when both are set, **respect the user's choice to disable**. Offer a `--color=auto|always|never` flag whose `never` is equivalent to `NO_COLOR` and whose `always` equals `FORCE_COLOR`, with the env vars taking precedence over `auto` and the explicit flag taking precedence over env vars when the user passes it directly. Document the precedence; do not make users guess.

---

## 4. Light/dark and theme structure

"It looked fine on my machine" is almost always **"I only tested my dark theme."** A huge fraction of terminal users run light backgrounds (Solarized Light, default macOS Terminal, many SSH-from-corporate setups). A palette tuned for dark backgrounds produces washed-out, low-contrast, sometimes invisible text on light — a direct accessibility and readability failure.

### Prefer terminal defaults; tint, don't paint

The safest theme is the *thinnest* one: use the terminal's default foreground and background (`text`/`bg` = "default") and only assign accent colors to roles that genuinely need them. Defaults are guaranteed legible because the user chose them. Every absolute color you set is a bet that it contrasts against *their* background — a bet you lose for half your users if you only test one mode.

When you must set backgrounds (selection bars, panels), do not assume the polarity. Detect or adapt:
- Query background luminance when possible (OSC 11 reply on capable terminals) and pick the variant.
- Where querying is unavailable, offer a `--theme light|dark|auto` and a config file; default to `auto` driven by `COLORFGBG` if present.
- Never hardcode `black text on white panel` — on a dark terminal that white panel is a blinding rectangle; on a light terminal a black-on-default panel may vanish.

### Theme structure

A theme is data, not code. Make it:
- **A named set of role → tiered-color mappings** (as in §2), with `light` and `dark` variants.
- **User-overridable** via a config file (TOML/YAML/JSON). Users *will* want their colorscheme to match the rest of their terminal. Letting them remap roles is the single most-appreciated feature of mature TUIs.
- **Validated for contrast** at author time. Even in a terminal you should sanity-check that `muted` is still readable and that `error` foreground clears its background — aim for the spirit of WCAG (large/mono text), not pixel-perfect ratios you can't compute against an unknown emulator palette. The practical test: read every role on both a light and a dark terminal; if you squint, raise the contrast.

---

## 5. Box-drawing, borders, and the ASCII fallback

Borders make a TUI legible — and break it spectacularly when the glyphs don't render. Box-drawing characters (`┌─┐│└┘├┤┬┴┼`, the rounded `╭╮╰╯`, heavy and double variants) live in Unicode and depend on (a) the output being UTF-8 and (b) the font having the glyphs. Over a Latin-1 SSH session, a misconfigured locale (`LANG=C`), an old Windows console, or a font without box-drawing coverage, they render as `mojibake` (`â”Œâ”€`), as missing-glyph boxes (`□`), or — worse — as the *wrong width*, which corrupts alignment for the whole frame. This is RED finding #8's cousin and a top cause of "garbage borders."

### The border style ladder

Choose a border *style* (a role), resolve to glyphs by capability:

```
Unicode rounded   ╭───╮   (nicest, needs UTF-8 + good font)
                  │   │
                  ╰───╯
Unicode light     ┌───┐   (broadest Unicode support)
                  └───┘
ASCII             +---+   (always works: + - | )
                  |   |
                  +---+
none / rule       ─────   or a blank line / indentation only
```

Detect UTF-8 capability (`LANG`/`LC_ALL`/`LC_CTYPE` contains `UTF-8`/`utf8`) and fall back to ASCII (`+`, `-`, `|`) when it is absent or when the terminal is `dumb`. Offer an explicit `--ascii` / config switch — some users on flaky links *prefer* ASCII even when UTF-8 is technically available.

```go
// Go / Bubble Tea + lipgloss — pick a border set by capability, not by hope.
func borderFor(caps Caps) lipgloss.Border {
    switch {
    case caps.Unicode && caps.RoundedOK:
        return lipgloss.RoundedBorder()        // ╭ ╮ ╰ ╯
    case caps.Unicode:
        return lipgloss.NormalBorder()         // ┌ ┐ └ ┘
    default:
        return lipgloss.Border{                // pure ASCII, never mojibake
            Top: "-", Bottom: "-", Left: "|", Right: "|",
            TopLeft: "+", TopRight: "+", BottomLeft: "+", BottomRight: "+",
        }
    }
}
```

Rules that keep borders aligned:
- **Every border glyph must be exactly one display cell wide.** The standard light/heavy/double box-drawing chars are width-1; verify any decorative substitute (some "fancy" dividers are width-2 and will shear the frame).
- **Account for the border in width math.** A box of inner width `w` consumes `w + 2` columns (left + right glyph). Off-by-one here clips the last column or wraps the frame — RED finding "wrong width math."
- **Don't mix styles in one frame** unless intentional; a heavy outer + light inner is a design choice, a *random* mix is a bug.

---

## 6. Unicode width on the grid (the substrate connection)

This is where color and borders meet the hard floor of the monospace canvas, and where RED finding #8 (CJK/wide-glyph truncation smears layout) lives.

**The grid model — defined in `terminal-substrate-and-constraints.md` — is: the terminal is a grid of fixed-size *cells*, and what you place is measured in *display columns*, not bytes and not Unicode scalar values (code points).** This sheet's job is to make you apply that model every time you cut, pad, or align text that carries content.

The three traps:

1. **Bytes ≠ columns.** `len("café")` in UTF-8 is 5 bytes for 4 columns. Truncating by bytes splits a multi-byte sequence into garbage and miscounts width.
2. **Code points ≠ columns.** A "character count" still lies, because:
   - **Wide (East-Asian) glyphs** — most CJK ideographs, many emoji — occupy **2 columns** each. `"日本語"` is 3 code points but **6 columns**. Pad or truncate it as if it were 3 and every following column is shifted left by 3.
   - **Zero-width** code points — combining marks (`e` + `◌́` = `é`), variation selectors, ZWJ — occupy **0 columns**. `"é"` (decomposed) is 2 code points, 1 column.
   - **ZWJ emoji sequences** (`👨‍👩‍👧`) are several code points rendering as 1–2 columns *if the terminal supports them* — and inconsistently across emulators, which is why you keep emoji out of load-bearing layout.
3. **Truncation mid-cluster.** Cutting a wide glyph or a grapheme cluster in half emits a half-character the terminal cannot render — it smears, drops, or shifts, and your column count diverges from the terminal's for the rest of the line.

### The discipline

- **Measure with a real width function**, never `len()` / `.length` / `.chars().count()`. Use a Unicode-width-aware library: `unicode-width` (Rust), `wcswidth`/Rich's `cell_len` (Python), `runewidth` (Go), `string-width` (JS). These implement the East-Asian Width tables.
- **Truncate and pad in column space**, snapping to grapheme-cluster boundaries — never split a wide glyph or a combining sequence. When a wide glyph won't fit the last column, drop it and pad one space rather than emit half of it.
- **Add an ellipsis in column space.** A `…` is one column; budget for it (`width - 1`) before truncating.

```python
# Python (Textual/Rich-flavored) — fit a value into N display columns, wide-glyph safe.
from rich.cells import cell_len, set_cell_size

def fit_cell(text: str, width: int) -> str:
    if cell_len(text) <= width:
        # pad in COLUMN space, not character count
        return text + " " * (width - cell_len(text))
    # truncate to width-1 columns, append a 1-column ellipsis, never split a wide glyph
    return set_cell_size(text, max(0, width - 1)) + "…"

# fit_cell("日本語ファイル", 8) -> "日本…   "  (correct 8 columns)
# A naive text[:8] would slice by code points and shear the table.
```

```rust
// Rust (ratatui-flavored) — width-correct truncation with ellipsis.
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

fn fit_cell(s: &str, width: usize) -> String {
    if s.width() <= width {
        let mut out = s.to_string();
        out.push_str(&" ".repeat(width - s.width())); // pad in columns
        return out;
    }
    let budget = width.saturating_sub(1); // reserve one column for the ellipsis
    let mut used = 0;
    let mut out = String::new();
    for ch in s.chars() {
        let w = ch.width().unwrap_or(0); // 2 for wide, 0 for combining
        if used + w > budget { break; }  // never split a wide glyph
        out.push(ch);
        used += w;
    }
    out.push('…');
    out.push_str(&" ".repeat(width.saturating_sub(used + 1)));
    out
}
```

Borders and color sit on top of this: a width-1 border glyph plus a width-correct interior is the only way `w + 2` columns actually consume `w + 2` cells. Get the width math wrong and no amount of correct color saves the frame.

---

## Common mistakes

| Mistake | Why it breaks | Fix |
|---------|---------------|-----|
| Hardcoding `\x1b[31m` / `red` / `#hex` at the call site | Can't fall back, can't theme, can't honor `NO_COLOR`, can't flip for light mode | Reference a role; resolve at render time |
| Error = red text only | Invisible to CVD, monochrome, `NO_COLOR`, screen-reader users | Add a redundant channel (prefix/glyph/bold); color is an accent |
| Checking `NO_COLOR == "1"` | Spec says *presence* (any value, even empty) disables color | Test for presence: `is_some()` / `in os.environ` |
| Emitting truecolor escapes on every terminal | 16-color terminals show wrong color or garbage | Detect depth once; down-convert via the ladder |
| Testing only dark mode | Light-terminal users get washed-out / invisible text | Use terminal defaults; ship light+dark variants; test both |
| Box-drawing without UTF-8 detection | Mojibake / missing-glyph boxes / sheared frames | Border-style ladder with ASCII (`+ - |`) fallback |
| Measuring text with `len()` / `.length` | Bytes or code points ≠ columns; wide/zero-width glyphs miscount | Unicode-width-aware measure + grapheme-safe truncate |
| Truncating with `s[:n]` | Splits wide glyphs / combining sequences → smear, shift | Truncate in column space, snap to cluster boundary, reserve ellipsis column |
| Painting `black on white` panels | Inverts disastrously on the opposite-polarity terminal | Tint relative to default bg, or detect polarity |
| Per-role color literals scattered across files | One readability fix becomes a grep hunt | Centralize in a theme; roles are the only public color API |

---

## Red flags — STOP

If you catch yourself doing any of these, stop and apply the ladder:

- Typing a hex code, an RGB triple, an ANSI index, or a color *name* anywhere outside the theme definition.
- Writing an SGR escape (`\x1b[…m`) by hand in widget/render code.
- Signaling *any* state — error, selection, success, focus — with color and nothing else.
- Reading `NO_COLOR` and comparing it to a value instead of checking presence.
- Emitting a truecolor or 256-color escape without having detected that the terminal supports it.
- Choosing colors having only looked at your own (probably dark) terminal.
- Placing a box-drawing or emoji character without a non-Unicode fallback path.
- Computing a width, an offset, a pad, or a truncation with `len()` / `.length` / character count.
- Slicing a string by index to fit a width.

## Counters to the rationalizations

- *"Everyone has truecolor in 2026."* No — SSH to a serial console, a Linux VT, CI log capture, `screen`, locked-down corporate terminals, and braille displays do not. The ladder is the cost of running in *someone else's* terminal, which is the entire job.
- *"`NO_COLOR` is niche."* It is a deliberate accessibility and preference signal, and honoring it is two lines. Ignoring it tells the user their explicit configuration doesn't matter — the opposite of design-led.
- *"Red obviously means error."* Not to ~8% of men (CVD), not in monochrome, not under `NO_COLOR`, not to a screen reader. "Obvious" is "obvious to me on my terminal" — the exact assumption this sheet exists to break.
- *"It rendered fine when I ran it."* You ran it on one emulator, one font, one locale, one color depth, one background polarity. "Fine on my machine" is the failure mode, not the test.
- *"`len()` is close enough; who uses CJK/emoji?"* Your users' filenames, branch names, commit authors, and process titles do. One wide glyph shears the rest of the row; "close enough" corrupts every column after it.
- *"ASCII borders look ugly."* A correct ASCII frame is legible everywhere; a Unicode frame that mojibakes is unreadable. Legible-everywhere beats pretty-sometimes — and you ship both via the ladder, so it's a false choice.
- *"I'll add theming later."* Roles cost nothing to adopt now and are a painful retrofit later (every literal is a separate edit). The cheap moment is before the literals multiply.

---

## Acceptance checklist

A TUI passes this sheet when:

- [ ] No color literal (hex/RGB/index/name) appears outside the theme definition; widgets reference roles.
- [ ] Every semantic state carries a redundant non-color channel (prefix, glyph, or attribute).
- [ ] Color depth is detected once at startup and resolved through truecolor → 256 → 16 → monochrome.
- [ ] `NO_COLOR` (presence, any value) forces monochrome on *all* output paths; `--color=auto|always|never` and `FORCE_COLOR` precedence are documented.
- [ ] The full UI is legible with color stripped (monochrome run distinguishes every state).
- [ ] Light and dark variants exist, default to terminal defaults where possible, and have both been viewed on opposite-polarity terminals.
- [ ] Borders resolve through a style ladder with a pure-ASCII (`+ - |`) fallback gated on UTF-8 detection.
- [ ] All width/pad/truncate uses a Unicode-width-aware function and snaps to grapheme boundaries (wide and zero-width glyphs handled).
- [ ] The theme is user-overridable via config.
