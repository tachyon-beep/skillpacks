---
name: accessibility-in-the-terminal
description: Use when a TUI signals state with color alone (red = error, green = ok), when you are about to claim a terminal app is "accessible" or "screen-reader friendly" without testing it, when reviewing color choices for color-vision deficiency / contrast / NO_COLOR, when adding animations/spinners/transitions, when state changes off-screen with no non-visual notification, or when keyboard focus is invisible — covers honest screen-reader reality in the terminal, color-blind-safe palettes, 16/256-color contrast, reduced-motion, visible focus, and the NO_COLOR convention.
---

# Accessibility in the Terminal

## The UX stake

A TUI is a visual medium running on top of a stream of bytes. That gap is where terminal accessibility lives — and where most TUIs quietly exclude people.

Two failures dominate, and they are the ones this sheet exists to kill:

1. **Color-only state signaling.** Red text for "error," green for "ok," and nothing else. This is invisible to roughly 1 in 12 men and 1 in 200 women (color-vision deficiency, CVD), invisible on monochrome and e-ink terminals, invisible under `NO_COLOR`, invisible to anyone who themed their terminal so "red" isn't red, and invisible to a screen reader, which announces the *characters* and knows nothing about the SGR escape that colored them. An error the user cannot perceive is an error that did not happen — until it bites them.

2. **Dishonesty about assistive technology.** Developers say "my TUI is screen-reader friendly" because they added some labels. They never opened a screen reader. The truth — which this sheet insists you state plainly — is that **the overwhelming majority of full-screen TUIs are effectively opaque to screen readers.** Pretending otherwise is worse than admitting it, because it sends a blind user into a wall while telling them the door is open.

Accessibility is not a compliance checkbox bolted on at the end. In the terminal it is mostly *restraint and honesty*: don't lean on channels (color, motion, spatial position) that some users can't receive, don't claim AT support you haven't earned, and always leave an escape hatch (`NO_COLOR`, a `--plain`/line mode, a non-visual status path).

> **Lyra principle:** the accessible choice is almost always the more robust choice for *everyone*. A status that carries an icon *and* a word survives a screenshot, a grep, a colorblind reviewer, a black-and-white printout, and a 2-bit serial console. Color-only survives none of them.

---

## HONEST screen-reader reality (read this first)

This is the section most TUI guides skip. Don't skip it.

### Why most TUIs are opaque to assistive technology

Screen readers (NVDA, JAWS, VoiceOver, Orca) were built for two worlds: the GUI accessibility tree (AX/UIA/AT-SPI) and the **line-oriented** terminal. A traditional command-line program emits lines; the terminal's accessibility integration reads new lines aloud as they appear. That works.

A full-screen TUI breaks that model on purpose. It enters the **alternate screen**, switches to **raw mode**, and paints a 2-D grid by moving the cursor with escape sequences and overwriting cells. From the screen reader's perspective:

- There is no DOM, no accessibility tree, no role/name/value for any "widget." A ratatui `List` or a Textual `DataTable` is just *cells the program drew*. The screen reader sees a wall of characters with no structure.
- Updates are **in-place cursor moves**, not new lines. The screen reader has no reliable event saying "this region changed." It may read the whole screen, read nothing, or read stale content.
- Focus is a concept *your app invented*. The OS doesn't know which pane has focus, so the screen reader can't follow it.
- Box-drawing characters (`│ ├ ─ ╮`), spinners, and progress bars get spelled out or announced as garbage.

**Say this out loud in your docs and your head:** *a full-screen, alt-screen TUI is, by default, close to unusable with a screen reader.* No amount of internal "labels" changes that, because there is no channel to carry those labels to the AT.

### Partial mitigations (honest about being partial)

You cannot make an alt-screen TUI a great screen-reader experience. You *can* do better than nothing. In rough order of impact:

1. **Offer a non-TUI mode — this is the real accessibility feature.** The single most effective thing you can ship is a `--plain` / `--no-tui` / line mode (or auto-detect: see the not-a-TTY rule below) that renders the same information as a normal stream of lines, no alt screen, no raw mode. This is the path a screen-reader user (and a `grep`, and a pipe) can actually use. Treat it as a first-class output target, not a degraded fallback.
   - `git` does this implicitly: `git log` pages interactively but pipes cleanly. `kubectl`, `gh`, `cargo` all degrade to line output. Follow them.
2. **Don't trap users in the alt screen with no exit.** Always honor `Ctrl-C`/`q` and restore the terminal (see `lifecycle-signals-and-terminal-restoration.md`). A user who can't read the screen must still be able to *leave* it cleanly.
3. **Carry critical status on a channel that survives line mode.** If your app's important output (results, errors, the final answer) can be emitted as plain lines on exit or in `--plain`, a blind user gets the payload even if the live TUI was opaque.
4. **Mind the OSC 777 / terminal-notification path.** Some terminals support desktop notifications via escape sequences (e.g. OSC 9 / OSC 777). These reach the OS notification system, which *is* accessible. Use them for genuinely important, infrequent events (job done, build failed) — never for chatter.
5. **Keep a textual, copyable status line.** Even sighted-but-AT-using and low-vision users benefit when state is expressible as words they can route through their own tooling.

### The rule

**Never claim "accessible" or "screen-reader friendly" for a full-screen TUI you have not tested with a screen reader.** If you haven't tested, the honest claim is: *"This TUI is primarily a sighted, visual interface. For non-visual use, run `--plain`."* Ship the `--plain` mode and that sentence is true and useful. Skip it and any accessibility claim is a lie.

---

## Never signal by color alone

This is the rule the RED baseline names directly: *"Error signaled by color alone (red text only)."* Close it.

### The redundancy principle

Every meaningful state must be distinguishable through **at least two of**: a **word/label**, a **symbol/icon/shape**, **position/structure**, and (optionally, as a *bonus* not a *substitute*) color.

| State | Color only (WRONG) | Redundant (RIGHT) |
|-------|--------------------|--------------------|
| Error | red text | `✖ ERROR:` + red (and `[ERROR]` / `E` in ASCII mode) |
| Warning | yellow text | `⚠ WARN:` + yellow (`[WARN]` / `!`) |
| Success | green text | `✔ OK:` + green (`[OK]` / `+`) |
| Info | dim/blue | `ℹ` / `·` + label |
| Selected row | reverse-video colour | leading `▸`/`>` marker **and** reverse video |
| Required field | red border | `*` glyph + label "(required)" |
| Diff add/remove | green/red lines | leading `+`/`-` (this is *why* diffs use sigils — they work in black and white) |

Note the diff example: unified diff has been color-blind-safe since before "color-blind-safe" was a phrase, because the `+`/`-` carries the meaning and color is decoration. That is the target shape for *all* TUI state.

### Don't forget shape for selection and focus

Highlighting the selected list row purely by background color fails the same way. Add a **gutter marker** (`▸`, `>`, `*`) so selection is legible when color is stripped or inverted. Reverse video is more robust than a color background because it survives most palettes — but pair it with a marker anyway.

### Example: redundant status, Rust + ratatui

Status semantics decided once, rendered with glyph + label + (optional) color. The glyph and label always render; color is the only thing gated on capability.

```rust
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

#[derive(Clone, Copy)]
enum Level { Error, Warn, Ok, Info }

struct Theme {
    use_color: bool,   // false under NO_COLOR / not-a-tty / --plain
    use_unicode: bool, // false on ASCII-only / capability probe miss
}

impl Level {
    /// Glyph FIRST. Even with zero color and zero unicode, the prefix
    /// carries the state. Color is the last, optional layer.
    fn marker(self, t: &Theme) -> &'static str {
        match (self, t.use_unicode) {
            (Level::Error, true) => "✖ ERROR",
            (Level::Error, false) => "[ERROR]",
            (Level::Warn,  true) => "⚠ WARN",
            (Level::Warn,  false) => "[WARN]",
            (Level::Ok,    true) => "✔ OK",
            (Level::Ok,    false) => "[OK]",
            (Level::Info,  true) => "ℹ INFO",
            (Level::Info,  false) => "[INFO]",
        }
    }
    fn color(self) -> Color {
        match self {
            Level::Error => Color::Red,
            Level::Warn => Color::Yellow,
            Level::Ok => Color::Green,
            Level::Info => Color::Cyan,
        }
    }
}

fn status_line<'a>(level: Level, msg: &'a str, t: &Theme) -> Line<'a> {
    let label = format!("{}: ", level.marker(t));
    let mut style = Style::default();
    if t.use_color {
        style = style.fg(level.color());
    }
    Line::from(vec![Span::styled(label, style), Span::raw(msg)])
}
```

The load-bearing detail: `marker()` runs *before* and *independently of* `use_color`. If you ever find yourself writing the meaning *inside* the color, stop — that is the bug.

### Example: redundant status, Python + Textual/Rich

Rich already honors `NO_COLOR` and detects non-terminals; lean on it, but still carry the glyph.

```python
from rich.console import Console
from rich.text import Text

# Rich auto-disables color when NO_COLOR is set or stdout isn't a TTY.
console = Console()

LEVELS = {
    "error": ("✖", "ERROR", "bold red"),
    "warn":  ("⚠", "WARN",  "yellow"),
    "ok":    ("✔", "OK",    "green"),
    "info":  ("ℹ", "INFO",  "cyan"),
}

def status(level: str, msg: str) -> Text:
    glyph, label, style = LEVELS[level]
    # ASCII fallback when the terminal can't encode the glyph.
    if not console.options.encoding.lower().startswith("utf"):
        glyph = {"✖": "x", "⚠": "!", "✔": "+", "ℹ": "i"}[glyph]
    # Glyph + LABEL are plain text — they survive even when `style`
    # is dropped (NO_COLOR / piped / dumb terminal).
    t = Text(f"{glyph} {label}: ", style=style)
    t.append(msg, style="")
    return t

console.print(status("error", "failed to connect to broker"))
# Renders "✖ ERROR: failed to connect to broker" with or without color.
```

---

## Color-blind-safe palettes

When you *do* use color (as the optional reinforcing layer), don't make it actively hostile to the ~8% of users with CVD.

### The cardinal sin: red/green as the only distinction

The most common CVD types (deuteranopia, protanopia) confuse red and green — exactly the pair most TUIs reach for to mean bad/good. If red and green are *only* told apart by hue, those users see two similar muddy tones.

Fixes, in order of preference:
1. **Carry the meaning in the glyph/label** (see above) so hue is decoration. This alone solves it.
2. **Shift the palette so the pair differs in lightness, not just hue.** A dark red vs. a light/bright green differ in brightness, which CVD users *can* see. Don't pair equal-luminance red and green.
3. **Use a CVD-safe palette** for any categorical encoding (charts, multi-series sparklines, tag colors). Okabe–Ito is the standard 8-color set designed for this; map its colors to the nearest 256-color cube indices for the terminal.
4. **Add a non-color channel to categorical data anyway** — line style (`─ ┈ ╌`), markers (`● ▲ ■`), or labels. A multi-line sparkline distinguished only by color is unreadable in mono and hard for CVD users regardless.

### Okabe–Ito → 256-color, as a usable starting palette

These are perceptually distinct for the common CVD types. Truecolor terminals can use the exact hex; 256-color terminals use the nearest cube index; 16-color and below should fall back to glyph/label only.

| Name | Hex (truecolor) | ~256 index |
|------|-----------------|-----------|
| Orange | `#E69F00` | 214 |
| Sky blue | `#56B4E9` | 75 |
| Bluish green | `#009E73` | 36 |
| Yellow | `#F0E442` | 227 |
| Blue | `#0072B2` | 31 |
| Vermillion | `#D55E00` | 166 |
| Reddish purple | `#CC79A7` | 175 |
| Black/grey | `#000000` | 232+ |

Pick from *adjacent rows* when you need two contrasting categories — avoid using vermillion and bluish-green as your only error/ok pair without glyphs.

### Don't hardcode against the user's theme

You don't know the user's background. Light terminals, dark terminals, solarized, custom — all exist. A "subtle grey" you chose on your dark theme can be invisible on someone's light theme. Two defenses: prefer the terminal's *default* foreground/background for body text (don't set it), and gate any color you *do* set behind contrast that holds on both light and dark (next section). Better still, ship a couple of named themes and a `--theme` flag (see `color-theming-and-monospace-canvas.md`).

---

## Contrast in 16/256-color

Truecolor lets you pick exact, contrast-tested values. **16-color and 256-color do not** — the palette is fixed and, for the base 16, *redefinable by the user's terminal theme*. You must reason about contrast under that uncertainty.

Principles:

- **Don't fight the default.** For primary text, *don't set a foreground or background at all* — inherit the user's terminal default, which they have already tuned to be readable for themselves. The most contrast-safe text is the text you didn't recolor.
- **The base-16 colors are not fixed pixels.** `Color::Red` (index 1) is whatever the user's theme maps it to. You cannot guarantee its contrast ratio. So in 16-color land, color is *advisory only* — never the sole carrier (back to the glyph rule).
- **Use bright variants for emphasis, dim for de-emphasis** rather than picking arbitrary hues. Bold/bright and dim are relative to the user's own palette and therefore more likely to preserve a contrast *direction* even if you can't guarantee a ratio.
- **In 256-color, target the greyscale ramp (232–255) for backgrounds/borders** and pick foregrounds several steps away on the ramp; perceived contrast on the greyscale ramp is far more predictable than across the color cube.
- **Aim for the WCAG-equivalent direction even though you can't measure pixels.** You can't compute a 4.5:1 ratio against an unknown theme, but you *can* avoid known-bad pairings: never put a 256-cube color on a same-lightness 256-cube color; never rely on `dim` as the only difference between two states (some terminals render `dim` identically to normal).
- **`reverse`/inverse video is the most portable emphasis** because it swaps whatever the user's fg/bg are — it preserves their contrast by construction. Prefer it for selection highlight over a chosen background color.

> The honest summary: in 16/256-color you *cannot prove* contrast, so you must *design so contrast doesn't carry meaning*. Color reinforces; glyph + label + structure + reverse-video carry. That is the same conclusion as the color-only rule, arrived at from the contrast direction.

---

## Reduced-motion restraint

Motion in a TUI — spinners, marquees, animated transitions, blinking, rapidly updating counters — is far more hostile than the same motion in a GUI, because:

- It re-renders cells constantly, which is **CPU burn and SSH-bandwidth flooding** (a real cost on remote/slow links) — itself a UX failure (see `rendering-and-redraw-discipline.md`).
- For users with vestibular disorders, ADHD, or photosensitivity, terminal motion is *uncapped and unstandardized* — there is no OS "reduce motion" toggle the terminal forwards. Constant motion in a focal CLI window is genuinely harmful, and blinking text can be a seizure trigger.
- Screen magnifier users lose their place when regions animate.

Discipline:

- **Honor a reduced-motion signal.** There is no universal terminal API, so adopt conventions: respect `NO_COLOR`-adjacent norms, read a `--no-animations`/`--reduce-motion` flag and a `REDUCE_MOTION`/`TUI_REDUCE_MOTION` env var, and **always disable animation when not a TTY**. When reduced motion is on, replace spinners with a *static* state word that updates on real change ("Loading…" → "Loaded 42 items"), not on a timer.
- **Never `blink`.** SGR blink (`\e[5m`) is the canonical accessibility hazard. Don't use it. Ever.
- **Cap update rate.** A spinner that ticks at 60fps gains nothing over ~8–10fps and costs ten times the bandwidth. Throttle. A progress display that updates more than a few times a second is just flicker.
- **Animate only state, not decoration.** A progress bar advancing because real work advanced is informative. A pulsing border because it looks cool is pure cost. Cut decorative motion entirely.
- **Prefer discrete updates to continuous animation.** "Step 3 of 7: compiling" that changes when the step changes beats a perpetual animation, and it doubles as a screen-reader-friendlier, line-mode-friendlier status.

### Example: reduced-motion-aware spinner, Go + Bubble Tea

The model picks a *static* representation when motion is suppressed, so the same code path serves animated, reduced-motion, and non-TTY runs.

```go
package main

import (
	"os"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"golang.org/x/term"
)

type model struct {
	frame      int
	reduceMot  bool
	label      string
}

func reduceMotion() bool {
	if os.Getenv("REDUCE_MOTION") != "" || os.Getenv("TUI_REDUCE_MOTION") != "" {
		return true
	}
	// Not a real terminal? No animation — and no alt screen either.
	if !term.IsTerminal(int(os.Stdout.Fd())) {
		return true
	}
	return false
}

type tickMsg time.Time

func tick() tea.Cmd {
	// ~8fps, not 60. Enough to read as "working," cheap on SSH.
	return tea.Tick(125*time.Millisecond, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m model) Init() tea.Cmd {
	if m.reduceMot {
		return nil // no timer at all
	}
	return tick()
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if _, ok := msg.(tickMsg); ok && !m.reduceMot {
		m.frame++
		return m, tick()
	}
	return m, nil
}

func (m model) View() string {
	if m.reduceMot {
		// Static, screen-reader / line-mode friendly. Updates on real
		// state change (m.label), never on a timer.
		return m.label + "…\n"
	}
	frames := []rune{'⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'}
	return string(frames[m.frame%len(frames)]) + " " + m.label + "\n"
}
```

---

## Keyboard operability + visible focus

A TUI is keyboard-first by nature — but "keyboard-first" silently assumes the user can *see where the keys go*. Invisible focus (RED baseline #21) breaks that for everyone, and disproportionately for low-vision and cognitive-load users.

Rules:

- **Everything reachable by keyboard.** No action may require the mouse (mouse is an enhancement, not a requirement; many terminals/SSH sessions don't forward it). This is just baseline TUI competence.
- **Focus must be *visibly* indicated, redundantly.** The focused pane/widget needs a non-color cue: a highlighted/double border vs. plain border, a `▸`/`>` title marker, a `[*]` tab indicator — *plus* optional color. If the only difference between "filter box focused" and "list focused" is a border color, CVD and mono users are lost. (See `input-keyboard-mouse-and-focus.md` for the focus *model*; this sheet owns its *accessibility*.)
- **Standard keys do standard things.** Tab/Shift-Tab to move focus, arrows within a widget, `Esc` to back out, `Enter` to confirm, `q`/`Ctrl-C` to quit, `?` for help. Surprising bindings are an accessibility and discoverability failure.
- **Make bindings discoverable** (a hint bar, `?` help) — owned by `affordances-and-discoverability.md`, but accessibility-relevant because hidden keys are unusable keys.
- **Don't trap focus.** The user must always be able to reach quit/help and leave. Combined with the lifecycle sheet: leaving must *restore the terminal*.
- **Respect the OS, not just your app.** Don't capture keys the terminal/OS needs (be careful overriding `Ctrl-C`, `Ctrl-Z`, `Ctrl-D` without giving an equivalent).

Visible-focus pattern: render border *style* by focus, not just border *color*.

```rust
use ratatui::widgets::{Block, Borders, BorderType};
use ratatui::style::{Style, Modifier};

fn pane_block(title: &str, focused: bool, use_color: bool) -> Block<'_> {
    // Shape difference (border type + title marker) carries focus even
    // with no color. Color is additive.
    let (btype, marker) = if focused {
        (BorderType::Double, "▸ ")
    } else {
        (BorderType::Plain, "  ")
    };
    let mut style = Style::default();
    if focused {
        style = style.add_modifier(Modifier::BOLD);
        if use_color { /* optionally add an accent fg here */ }
    }
    Block::default()
        .borders(Borders::ALL)
        .border_type(btype)
        .title(format!("{marker}{title}"))
        .border_style(style)
}
```

---

## NO_COLOR as an accessibility signal

`NO_COLOR` is a cross-tool convention (no-color.org): **if the `NO_COLOR` environment variable is present and non-empty, your program must not emit color.** It is an explicit, user-set accessibility/preference signal, and honoring it is not optional.

Who sets it and why: users on monochrome/limited terminals, users with CVD or contrast needs who find your palette harmful, users piping output into tools that choke on escapes, CI systems, and people who simply don't want color. They are telling you, machine-readably, *"the color channel does not work for me."* If your error state was color-only, `NO_COLOR` just made it invisible — which is exactly why the color-only rule and `NO_COLOR` are the same lesson twice.

Implementation rules:

- **Check presence, not value.** Per spec, `NO_COLOR` set to *anything* non-empty (even `0`) disables color. Don't parse it as a boolean.
- **`NO_COLOR` disables color but NOT the glyphs/labels.** This is the whole point: under `NO_COLOR`, `✖ ERROR: ...` still renders and still means error. If turning off color also turns off your ability to signal state, you never had accessible state.
- **Also auto-disable color when stdout is not a TTY** (piped/redirected) — see `terminal-substrate-and-constraints.md` and `distribution-and-cross-environment.md`. Detection order: `NO_COLOR` set → off; not a TTY → off; `FORCE_COLOR` set → on (escape hatch for users who *want* color through a pipe); else probe the terminal.
- **Pair it with `--color=auto|always|never`** for explicit override, mirroring `ls`/`grep`/`git`.

```python
import os
import sys

def should_use_color() -> bool:
    # 1. NO_COLOR: presence (non-empty) wins, regardless of value.
    if os.environ.get("NO_COLOR"):
        return False
    # 2. FORCE_COLOR escape hatch (e.g. user pipes to a pager that renders ANSI).
    if os.environ.get("FORCE_COLOR"):
        return True
    # 3. Only colorize a real terminal; piped/redirected output stays clean.
    return sys.stdout.isatty()
```

Wire the boolean into the *one* place that decides `use_color` (the `Theme` in the Rust example). Everything downstream renders glyph + label unconditionally and color conditionally. One switch, no scattered `if no_color` checks.

---

## Common mistakes

| Mistake | Why it fails | Fix |
|---------|--------------|-----|
| "Red text = error" and nothing else | Invisible to CVD, mono, themed, `NO_COLOR`, and screen-reader users | Glyph + label first; color last |
| Claiming "screen-reader friendly" without testing one | It is almost certainly opaque; the claim sends blind users into a wall | Test, or claim only "visual TUI; use `--plain`" — and ship `--plain` |
| No `--plain` / line mode | The one genuinely accessible path doesn't exist | Make line output a first-class target |
| Honoring `NO_COLOR` by dropping color *and* glyphs | State becomes unsignalable | `NO_COLOR` drops color only; glyphs/labels always render |
| Parsing `NO_COLOR=0` as "color on" | Violates the spec; user gets color they disabled | Presence (non-empty) check, not boolean parse |
| Red/green as the sole categorical distinction | Confused by the most common CVD types | Differ by lightness + glyph; use Okabe–Ito for charts |
| Hardcoding fg/bg for body text | Breaks on the user's light/dark/custom theme; unknown contrast | Inherit terminal default; use `reverse` for emphasis |
| 60fps spinners / decorative animation | CPU burn, SSH flood, motion-sensitivity harm | Throttle to ~8fps, honor reduce-motion, cut decorative motion |
| `\e[5m` blink for emphasis | Seizure/motion hazard, often ignored anyway | Never blink; use bold/reverse |
| Focus shown only by border *color* | Invisible to CVD/mono; user can't tell what's focused | Border *style* + title marker + color |
| "256-color so I picked a nice grey" | Contrast unknowable against user's theme; may be invisible | Greyscale ramp / reverse video; never rely on a chosen hue for legibility |
| Mouse-only actions | Excludes keyboard-only users and mouseless terminals | Every action keyboard-reachable |

---

## Red flags — STOP

If you catch yourself about to do any of these, stop and fix it before writing more code:

- You're about to set a color to mean something with **no glyph/label carrying the same meaning**.
- You're about to write "accessible" or "screen-reader friendly" in a README/PR/comment for a full-screen TUI you have **never opened in a screen reader** and that has **no `--plain` mode**.
- You wrote `if os.environ["NO_COLOR"] == "true"` or any boolean parse of `NO_COLOR`.
- Disabling color in your code path *also* disabled the ability to signal state.
- Your only error/success distinction is **red vs. green hue**.
- You added a spinner/animation with **no reduced-motion path and no fps cap**, or you typed `\e[5m` / `blink`.
- Focus differs between panes **only by color**.
- An action in your TUI **requires the mouse**.
- You set an explicit foreground/background for primary body text "to make it look right on my terminal."

---

## Counters to the rationalizations

Agents (and humans) skip terminal accessibility with a small set of excuses. Each is wrong:

- *"It's a TUI, accessibility doesn't apply — terminal users are technical."* CVD, low vision, vestibular sensitivity, and screen-reader use are independent of technical skill. Blind developers exist and use the terminal heavily *because* line mode is accessible — which is exactly the mode you're about to skip.
- *"Color makes it readable; that IS the UX."* Color is an *enhancement* layer. If it's load-bearing, your UX collapses under `NO_COLOR`, a pipe, a screenshot, a colorblind reviewer, and a mono terminal. Glyph + label + color is *more* readable, for everyone, in more contexts.
- *"Screen readers can't do TUIs anyway, so why bother."* Correct premise, wrong conclusion. *Because* they can't, you ship the `--plain`/line mode that they *can* use, and you stop falsely claiming AT support. Doing nothing AND lying is the failure; doing the honest minimum is the fix.
- *"I'll add accessibility later."* The glyph-first status helper, the single `use_color` switch, and the `--plain` flag are all *cheaper to design in than to retrofit*, because retrofitting means hunting every scattered color-as-meaning site. Later is more expensive, not less.
- *"Reduced motion is niche."* It's also *free performance and SSH-friendliness* — the reduced-motion path is the same path that stops you flooding a remote terminal at 60fps. You want it anyway.
- *"NO_COLOR is obscure."* It's honored by `git`, `ls`, `grep`, `cargo`, `npm`, `gh`, ripgrep, fd, bat, and most modern CLIs. Your tool not honoring it is the anomaly, and it's a three-line check.
- *"Testing with a screen reader is too much work for this project."* Then don't claim screen-reader support. The honesty (and the `--plain` mode) costs nothing and is the actually-useful deliverable. Unverified accessibility claims are worse than none.

---

## Cross-references

- `color-theming-and-monospace-canvas.md` — owns palettes/theming/`NO_COLOR` mechanics and UTF-8/CJK width; this sheet owns their *accessibility consequences*.
- `lifecycle-signals-and-terminal-restoration.md` — a user who can't read the screen must still exit cleanly; restoration is an accessibility guarantee.
- `terminal-substrate-and-constraints.md` / `distribution-and-cross-environment.md` — TTY detection and the not-a-TTY fallback that backs `--plain`.
- `rendering-and-redraw-discipline.md` — the redraw discipline that makes reduced-motion cheap.
- `input-keyboard-mouse-and-focus.md` — the focus *model*; this sheet owns *visible, redundant* focus indication.
- `affordances-and-discoverability.md` — discoverable keybindings, which are unusable if hidden.
