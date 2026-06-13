---
name: input-keyboard-mouse-and-focus
description: Use when keystrokes land in the wrong widget, Ctrl-C is swallowed or kills the app uncleanly, a multi-line paste arrives as a storm of individual keypresses (running commands or mangling input), there is no visible focus indicator between a filter box and a list, Tab/Shift-Tab focus order is wrong or traps the user, Esc lags or triggers the wrong action, Alt/Ctrl/Shift modifier combos or function keys are indistinguishable or unrecognized, mouse clicks/scroll/drag do nothing or leak escape garbage onto the screen, or you are wiring key bindings, chords, kitty keyboard protocol, bracketed paste, or mouse capture in a ratatui/Textual/Bubble Tea/Ink/notcurses TUI.
---

# Input: Keyboard, Mouse, and Focus

## The UX stake

Input is the entire contract between the user and a TUI. There is no cursor hovering a button to reassure them, no ripple animation, no browser autofill. The only evidence that the program is listening is that *the right thing happens when they press the right key* — and that they can see where their keystrokes will go before they press anything.

Four input failures destroy that contract, and all four are invisible in a quick demo and brutal in real use:

- **Swallowed Ctrl-C.** The user hits the one key combination every Unix user reaches for to escape, and nothing happens — or worse, it kills the process so hard the terminal is left in raw mode. Trust is gone in one keypress.
- **Paste treated as keystrokes.** The user pastes a 40-line config or a URL with a `?` in it, and your per-keystroke handler fires forty times — running shortcuts, triggering searches, corrupting the field. This is a data-loss bug wearing a UX costume.
- **Lost focus.** Two panes, no visible indication of which one is "live." The user types and watches their characters disappear into the wrong widget. They now distrust *every* keystroke and slow down to a crawl.
- **Ambiguous Esc.** Esc is both "the user pressed the Escape key" and "the leading byte of an arrow key, a function key, or an Alt-chord." Get the disambiguation wrong and either Esc feels broken (200ms of lag) or arrow keys randomly close dialogs.

This sheet is about closing those four, plus the substrate facts (kitty keyboard protocol, bracketed paste, mouse opt-in cost) that determine *which* failures are even avoidable on a given terminal.

> Scope note: terminal-restoration-on-crash and SIGINT/SIGTERM/SIGWINCH plumbing live in `lifecycle-signals-and-terminal-restoration.md`. This sheet covers the *input semantics* — including treating Ctrl-C as a routable key event rather than a process signal, which only works if lifecycle has already armed a clean shutdown path.

---

## Key events: the model under every framework

Every TUI framework, regardless of language, normalizes raw terminal bytes into a **key event** with roughly this shape:

```
KeyEvent {
  code:      Char('a') | Enter | Esc | Tab | BackTab | Up | Down |
             Left | Right | Home | End | PageUp | PageDown |
             Backspace | Delete | Insert | F(1..=12) | ...
  modifiers: SHIFT | CTRL | ALT | SUPER  (a bitset; can combine)
  kind:      Press | Repeat | Release   (Release/Repeat only on capable terminals)
}
```

Design rules that hold across stacks:

1. **Route on the semantic event, never on raw bytes.** Match `KeyCode::Char('q')` or `key.name == "q"`, not byte `0x71`. The framework already handled UTF-8 decoding, escape-sequence parsing, and the legacy ambiguities — re-doing it by hand is how you reintroduce bugs the framework already fixed.

2. **Modifiers are a bitset, not an enum.** `Ctrl+Shift+Up` is one event with two modifier bits set. Test with bitwise contains (`mods.contains(CTRL)`), not equality (`mods == CTRL`), or Shift-held combos silently fail.

3. **Don't assume Release/Repeat exist.** Most terminals send only Press. Key-up events and auto-repeat distinction require the kitty keyboard protocol (below). Build the *default* interaction on Press-only; treat Release as a progressive enhancement (e.g. push-to-talk, vim-style key-held panning).

4. **Decide the binding table once, centrally.** A scattered `if key == ...` mess across every widget is how routing becomes ambiguous (RED #22). Keep a single keymap that maps `(focus_context, key_event) → action`, so "what does `j` do here?" has exactly one answer.

---

## Modifiers and chords — and what plain terminals literally cannot send

This is the most misunderstood part of TUI input, and getting it wrong produces the "my shortcut doesn't work and I don't know why" bug that no amount of code review finds.

**The hard fact: the legacy terminal input protocol cannot represent many key combinations at all.** It is not a framework limitation; the bytes do not exist on the wire.

| Combo | What the terminal actually sends (legacy) | Distinguishable? |
|---|---|---|
| `Ctrl+I` | `0x09` — identical to `Tab` | **No.** Ctrl-I *is* Tab. |
| `Ctrl+M` | `0x0D` — identical to `Enter`/Return | **No.** Ctrl-M *is* Enter. |
| `Ctrl+H` | `0x08` — often identical to `Backspace` | Usually no. |
| `Ctrl+[` | `0x1B` — identical to `Esc` | **No.** Ctrl-[ *is* Escape. |
| `Ctrl+Space` | `0x00` (NUL), or nothing on some terminals | Flaky. |
| `Ctrl+1`, `Ctrl+2` … | usually no distinct byte at all | **No** on most terminals. |
| `Shift+Enter`, `Shift+Tab` | `Tab`→`BackTab` works; `Shift+Enter` usually doesn't | Partial. |
| plain letters with `Ctrl` (`Ctrl+A`..`Ctrl+Z`) | control bytes `0x01`..`0x1A` | Yes (but collides as above). |

**Design consequence:** never make `Ctrl+I`, `Ctrl+M`, or `Ctrl+[` mean something *different* from `Tab`, `Enter`, and `Esc`. If you bind "indent" to Ctrl-I, you have also bound it to Tab, and the user pressing Tab to move focus will indent instead. Choose chords from the combinations terminals can actually send, or require the kitty protocol and provide a legacy fallback binding.

---

## Kitty keyboard protocol: why it exists and when to use it

The **kitty keyboard protocol** (a.k.a. the "comprehensive keyboard handling" / CSI-u progressive enhancement, now supported by kitty, foot, WezTerm, Ghostty, recent Konsole/iTerm2 builds, and others) fixes the legacy ambiguities by encoding every key as an unambiguous `CSI unicode-key-code ; modifiers u` sequence. With it enabled you get:

- **Disambiguated combos** — `Ctrl+I` ≠ `Tab`, `Ctrl+[` ≠ `Esc`, `Ctrl+Enter`, `Ctrl+Shift+letter`, `Ctrl+digit`.
- **Key release and repeat events** — real key-up, real auto-repeat distinction.
- **All modifiers reported**, including Super/Hyper/Meta and lock states.
- **No more Esc-timing guesswork** — Esc arrives as a single tagged event, killing the ambiguity problem at the source (see the Esc section).

The cost and the discipline:

1. **It is opt-in and must be negotiated, then released.** You push the protocol flags on startup and *must* pop them on exit — including on panic/crash. A leaked kitty-keyboard mode leaves the user's shell receiving CSI-u garbage. The push/pop belongs in the same restoration guard as raw mode and the alt screen (see `lifecycle-signals-and-terminal-restoration.md`).
2. **Detect, never assume** (RED #28). Query support and degrade. On a terminal without it, your `Ctrl+I` binding *will* collide with Tab — so every kitty-only chord needs a legacy-safe alternative or a clearly-communicated "requires a modern terminal" note.
3. **Request only the flags you use.** Enabling release events when you only handle presses just adds event volume you must filter.

Enabling it in **ratatui / crossterm (Rust)**:

```rust
use crossterm::event::{
    KeyboardEnhancementFlags, PushKeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    KeyCode, KeyModifiers, KeyEventKind, Event,
};
use crossterm::{execute, terminal::supports_keyboard_enhancement};
use std::io::stdout;

// On startup, AFTER entering raw mode + alt screen, only if supported:
if supports_keyboard_enhancement().unwrap_or(false) {
    execute!(
        stdout(),
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
        )
    )?;
}

// On EVERY exit path (normal, error, panic hook): pop before leaving raw mode.
let _ = execute!(stdout(), PopKeyboardEnhancementFlags);

// Now Ctrl+Enter and key-release are real events:
if let Event::Key(k) = event {
    match (k.code, k.modifiers, k.kind) {
        // Ctrl+Enter — IMPOSSIBLE to receive without the protocol enabled.
        (KeyCode::Enter, KeyModifiers::CONTROL, KeyEventKind::Press) => submit_and_stay(),
        // Plain Enter still works on legacy terminals.
        (KeyCode::Enter, KeyModifiers::NONE,    KeyEventKind::Press) => submit_and_close(),
        // Push-to-reveal: only meaningful because release events are reported.
        (KeyCode::Char(' '), _, KeyEventKind::Press)   => start_preview(),
        (KeyCode::Char(' '), _, KeyEventKind::Release) => stop_preview(),
        _ => {}
    }
}
```

> Guardrail: filter `KeyEventKind::Release` early in code paths that assume Press. After enabling `REPORT_EVENT_TYPES`, crossterm delivers Release events too — un-filtered, a single `q` keypress fires your quit handler twice (once on press, once on release). This is a classic post-enablement regression.

---

## Bracketed paste: stop treating a paste as a keystroke storm

When a user pastes into a normal terminal program, the terminal sends the clipboard contents **byte-for-byte as if typed**. A pasted `git push --force\n` is indistinguishable from the user typing it — including the trailing newline that *submits the form*. In a TUI with per-keystroke bindings this is catastrophic: pasting text that contains `j`, `/`, `q`, `:`, or a newline triggers navigation, search, quit, command-mode, and submit. This is RED #2's keyboard cousin and a genuine safety bug, not a polish issue.

**Bracketed paste mode** is the fix. When enabled, the terminal wraps pasted content in `ESC[200~ … ESC[201~`, so the program can treat the whole blob as one atomic *paste event* instead of N key events.

Rules:

1. **Enable it, and handle the paste event distinctly from key events.** Insert the payload into the focused text field literally — do **not** route it through the keymap, and do **not** auto-submit on an embedded newline.
2. **Sanitize the payload.** Strip or visibly escape control characters and embedded escape sequences before display; a pasted ANSI sequence must never reach the screen unfiltered (terminal-injection risk). Decide a policy for embedded newlines in single-line fields (usually: replace with spaces or reject).
3. **Like every mode, it must be disabled on exit/panic** alongside raw mode (lifecycle sheet).

**Textual / Rich (Python)** handles bracketed paste for you and surfaces it as a first-class `Paste` message — the right shape to copy in any stack:

```python
from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog
from textual import events

class PasteAwareApp(App):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="type or paste — newlines won't submit")
        yield RichLog()

    def on_key(self, event: events.Key) -> None:
        # Single keypresses route through bindings as normal.
        if event.key == "enter":
            self.query_one(RichLog).write("submit")

    def on_paste(self, event: events.Paste) -> None:
        # A paste arrives as ONE event with the full blob — never as N key events.
        # It does NOT pass through on_key, so embedded 'q', '/', '\n' are inert.
        cleaned = event.text.replace("\n", " ").replace("\r", "")
        self.query_one(Input).insert_text_at_cursor(cleaned)
        event.stop()
```

The load-bearing point: `on_paste` and `on_key` are separate channels. The day you discover a paste is leaking into `on_key`, bracketed paste is off or your terminal doesn't support it — degrade by detecting unusually fast key bursts, but the real fix is enabling the mode.

---

## Mouse events and their opt-in cost

Mouse support in a terminal is **opt-in and expensive in side effects**, which is why "the mouse does nothing" and "clicking spews `35;40M` garbage" are both common (RED-adjacent). You enable a mouse-tracking mode and the terminal starts sending `ESC[<…M` / `…m` sequences for clicks, drags, and wheel scroll, parsed by the framework into mouse events with column/row, button, and modifiers.

The cost you are signing up for:

- **You break the terminal's own selection and copy.** Once you capture mouse events, the user's normal click-drag-to-select-and-copy stops working — your app is eating those events. Power users *hate* this. The mitigation everyone expects: hold **Shift** to bypass app mouse capture and get native selection back (most terminals honor this automatically *if* you don't also grab Shift-modified mouse events).
- **SGR mouse mode is mandatory for wide terminals.** The legacy X10 encoding caps coordinates at column 223. Enable **SGR extended mouse mode** (`1006`) or clicks past column 223 are wrong or dropped.
- **It must be disabled on exit/panic** like every other mode, or the user's shell fills with mouse-report garbage on every click.

**Design guidance:** keyboard-first, mouse-optional. Treat mouse as an accelerator over a fully keyboard-navigable UI, never the only path. Scroll-wheel-to-scroll-a-list is the highest-value, lowest-cost mouse feature; full drag-to-resize panes is high-cost and should be earned. And whatever the mouse can do, a key must also do — for SSH-through-three-hops users, screen-reader users, and anyone whose terminal mangles mouse reporting.

**Bubble Tea (Go)** — opt in explicitly, handle wheel scroll, leave selection alone:

```go
package main

import tea "github.com/charmbracelet/bubbletea"

type model struct{ offset int }

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.MouseMsg:
		switch msg.Button {
		case tea.MouseButtonWheelUp:
			if m.offset > 0 {
				m.offset--
			}
		case tea.MouseButtonWheelDown:
			m.offset++
		case tea.MouseButtonLeft:
			if msg.Action == tea.MouseActionPress {
				m.selectRowAt(msg.Y) // click-to-select is an accelerator...
			}
		}
	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k": // ...the keyboard path always exists.
			if m.offset > 0 {
				m.offset--
			}
		case "down", "j":
			m.offset++
		case "ctrl+c", "q":
			return m, tea.Quit
		}
	}
	return m, nil
}

func main() {
	// WithMouseCellMotion enables SGR mouse + cell-level motion reporting.
	// Bubble Tea restores mouse mode on exit as part of its own teardown.
	p := tea.NewProgram(model{}, tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		panic(err)
	}
}
```

---

## The focus model: the answer to "where do my keystrokes go?"

Lost focus (RED #22) and no-visible-focus-indicator (RED #21) are the same disease seen from the inside and the outside. The cure is an explicit focus model with three parts: **a single source of truth, a visible indicator, and a defined focus order.**

### 1. Exactly one focused widget, owned by central state

There is one `focused: FocusId` in your app state. Input dispatch reads it first:

```
on key event:
  if it is a paste event        -> deliver to focused text widget (literal insert)
  else if it is a GLOBAL key    -> handle (quit, help, focus-cycle) — works regardless of focus
  else                          -> deliver to the focused widget's keymap
```

Global keys (quit, help `?`, `Tab` to cycle focus) must work from *any* focus context, or the user gets trapped. Everything else is scoped to the focused widget. This single dispatch order is what makes "does `/` start a search or type a slash?" have one deterministic answer: it depends on whether focus is on the list (search) or the input (slash).

### 2. The focus indicator is non-negotiable and must not rely on color alone

The focused widget must be *unmistakable* without staring:

- **Border emphasis** — focused pane gets a heavy/double/colored border; unfocused panes get a dim single border. (Color *plus* weight, so it survives NO_COLOR / monochrome / CVD — see `color-theming-and-monospace-canvas.md` and `accessibility-in-the-terminal.md`.)
- **Title/caret cues** — `[ Filter ]` focused vs ` Filter ` unfocused; a visible block caret in the focused field only.
- **Dim the rest** — reduce the contrast of unfocused content so the eye is pulled to the live pane.

Never signal focus by a single subtle color change. A blue vs gray border with identical weight is invisible to a meaningful fraction of users and to anyone on a monochrome theme.

### 3. Focus order: predictable, reversible, and never a trap

- `Tab` advances, `Shift+Tab` (`BackTab`) reverses, through a **defined, stable order** — usually top-to-bottom, left-to-right reading order.
- Focus **cycles** (wraps), and **escape from a sub-region is always possible** — no widget may capture Tab so the user can't leave it.
- Skip non-interactive widgets. Disabled widgets are not in the order.
- When the focused widget is destroyed (e.g. a closed pane), move focus to a deterministic neighbor — never to "nothing," which silently drops all subsequent keystrokes.

### Worked example: the filter-box-plus-list pattern (the RED #21/#22 scenario)

This two-pane "type to filter, arrow to select" layout is the single most common TUI shape and the one the RED baseline keeps failing. The contract:

- **Tab** toggles focus between the filter input and the result list; the focused one gets a bold/colored border and the other dims.
- While the **filter** is focused: printable keys edit the query; `Down`/`Enter` move focus into the list (so you can keep your hands moving).
- While the **list** is focused: `j/k`/arrows move selection, `/` jumps back to the filter, `Enter` activates.
- **Global, focus-independent:** `Ctrl+C` and `q`-when-not-in-a-text-field quit; `?` shows help.
- **Selection survives filtering** by anchoring on item identity, not absolute index (that state rule lives in `event-loop-and-state-architecture.md`, but it is *triggered* here by focus-aware keystroke routing).

A character typed while the list is focused must never silently edit the hidden filter, and an arrow press while the filter is focused must never silently move a hidden selection. That is exactly what the single-focus dispatch order guarantees.

---

## The Esc-ambiguity / escape-timing problem

On a legacy terminal, pressing **Esc** sends one byte: `0x1B`. But `0x1B` is *also* the first byte of nearly every special key:

```
Esc           ->  1B
Up arrow      ->  1B 5B 41        ("ESC [ A")
F1            ->  1B 4F 50        ("ESC O P")
Alt+f         ->  1B 66           ("ESC f")   <- Alt is "ESC then the key"
```

So when the parser sees `0x1B`, it cannot yet know whether the user pressed Escape *or* is one byte into an arrow key. The legacy resolution is a **timeout**: wait a few milliseconds for a follow-up byte; if none arrives, it was a lone Esc.

This forces a genuinely bad tradeoff on legacy terminals:

- **Timeout too short** → arrow keys and Alt-chords over a slow/laggy SSH link get split: the `ESC` is read as Escape and the `[A` leaks as stray characters. Pressing Up closes your dialog and types `A`.
- **Timeout too long** → Esc feels sluggish; the user hits Escape to cancel and waits 200ms watching nothing happen. In a vim-modal TUI where Esc is pressed constantly, this is maddening.

Design rules:

1. **Prefer the kitty keyboard protocol where available** — with `DISAMBIGUATE_ESCAPE_CODES`, Esc arrives as its own tagged event with *zero* timeout, and arrow/Alt sequences are unambiguous. This dissolves the problem rather than tuning it. Detect support and use it.
2. **On legacy terminals, tune the timeout, don't ignore it.** Most frameworks expose it (crossterm's `poll` granularity, Textual's `ESCDELAY`-style handling, libtermkey/notcurses escape timers). A common sweet spot is ~25–50ms locally; allow a longer value for high-latency SSH and make it configurable.
3. **Give Esc an unambiguous escape hatch.** Provide a second, non-`Esc` binding for the same action (`q`, `Ctrl+G`, `Ctrl+[` is *not* a substitute since it equals Esc) so users on bad links aren't blocked when Esc is being absorbed by the timeout.
4. **Make Esc's meaning layered and predictable.** When Esc *does* fire, it should do the least-destructive contextual thing: close the topmost modal, else clear the active filter/selection, else (with confirmation) prompt to quit. Esc should never *immediately* quit the whole app — too easy to hit, too ambiguous to trust.

---

## Common mistakes

| Mistake | Why it bites | Fix |
|---|---|---|
| Binding an action to Ctrl-C without honoring quit | Users' universal "get me out" reflex is dead; combined with no restoration, leaves a wrecked terminal | Treat Ctrl-C as quit by default; if you must repurpose it, require explicit confirm AND keep a guaranteed clean-shutdown path (lifecycle sheet) |
| Per-keystroke handler with no bracketed paste | A pasted blob fires N bindings — runs commands, submits forms, corrupts fields; a real safety/data-loss bug | Enable bracketed paste; handle the `Paste` event as one atomic literal insert; sanitize control chars |
| `mods == CTRL` equality check | Silently fails the moment Shift is also held (`Ctrl+Shift+…`) | Use bitwise `contains` on the modifier bitset |
| Binding "indent" to Ctrl-I (or any to Ctrl-M / Ctrl-[) | On legacy terminals Ctrl-I *is* Tab, Ctrl-M *is* Enter, Ctrl-[ *is* Esc — pressing Tab now indents | Pick chords the wire can represent, or require kitty protocol with a legacy fallback binding |
| Enabling `REPORT_EVENT_TYPES` then not filtering Release | Every keypress fires its handler twice (press + release) | Filter to `KeyEventKind::Press` in handlers that assume single-fire |
| No visible focus indicator between panes | Users type into the wrong widget and distrust every keystroke | Border weight + dim-the-rest + caret; color *plus* a non-color cue |
| Focus signaled by color alone | Invisible on NO_COLOR/monochrome and to CVD users | Always pair color with weight/border/title change |
| A widget that captures Tab | User is trapped, can't reach other panes | Tab/Shift-Tab are global focus-cycle keys; reserve them; provide in-widget alternatives |
| Mouse capture with no keyboard equivalent | SSH/screen-reader/mangled-mouse users are locked out; native copy-paste breaks | Keyboard-first; mouse is an accelerator only; honor Shift-bypass for native selection |
| X10 mouse encoding (no SGR `1006`) | Clicks past column 223 are wrong/dropped on wide terminals | Always enable SGR extended mouse mode |
| Esc immediately quits the app | Too easy to hit, ambiguous with arrow/Alt bytes — accidental data loss | Esc does the least-destructive contextual thing; quit needs confirm and a non-Esc binding too |
| Esc timeout untuned | Too short = arrow keys leak `[A` over SSH; too long = Esc feels broken | Prefer kitty disambiguation; else tune (~25–50ms local), make it configurable, add a non-Esc escape key |
| Routing on raw bytes / re-parsing escapes by hand | Reintroduces every legacy ambiguity the framework already solved | Match the framework's semantic `KeyCode`/key name; keep one central keymap |
| Enabling a keyboard/paste/mouse mode without popping it on exit | Leaks CSI-u / `200~` / mouse-report garbage into the user's shell on crash | Push/pop every mode in the same restoration guard as raw mode and alt screen |

---

## Checklist before you ship input handling

- [ ] Ctrl-C quits (or is confirmed-repurposed) and always reaches a clean shutdown.
- [ ] Bracketed paste enabled; paste is one atomic literal insert, sanitized, never routed through bindings, never auto-submits.
- [ ] Modifier checks use bitwise contains, not equality.
- [ ] No binding collides with the Ctrl-I=Tab / Ctrl-M=Enter / Ctrl-[=Esc legacy aliases.
- [ ] Kitty keyboard protocol detected (not assumed), enabled when present, popped on every exit incl. panic; Release events filtered where Press is assumed.
- [ ] Exactly one focused widget in central state; global keys (quit/help/focus-cycle) work from any focus.
- [ ] Focus indicator uses color *plus* a non-color cue; unfocused panes dim.
- [ ] Tab/Shift-Tab cycle a stable focus order, wrap, skip disabled widgets, and never trap.
- [ ] Mouse is opt-in, SGR `1006` enabled, every mouse action has a keyboard equivalent, Shift-bypass for native selection preserved, mode popped on exit.
- [ ] Esc disambiguation handled: kitty protocol where available; tuned, configurable timeout otherwise; a non-Esc alternative binding; Esc does the least-destructive contextual action, never an instant quit.
