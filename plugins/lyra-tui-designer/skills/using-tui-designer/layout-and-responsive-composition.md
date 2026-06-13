---
name: layout-and-responsive-composition
description: Use when a TUI hardcodes 80x24 or any fixed dimensions, when content gets clipped or smeared after a terminal resize, when rows go stale or overlap after dragging the window, when the layout breaks in a narrow split pane or a maximized 4K terminal, when long lines neither wrap nor truncate cleanly, when there is no SIGWINCH/resize reflow, when panes overlap or vanish, or when the app crashes / renders garbage in a tiny terminal — covers constraint and flex layout, min-size and "terminal too small" states, overflow (scroll regions, ellipsis truncation), and reflow on resize.
---

# Layout and Responsive Composition

## The UX stake

A terminal is not a fixed canvas. It is a rectangle the user resizes constantly — they drag the window edge, split a tmux pane, SSH from a phone, maximize on a 4K monitor, or run you inside a 40-column sidebar. Every one of those is a viewport you did not choose.

A TUI that assumes 80x24 is the terminal equivalent of a website that only renders at 1280px. The symptoms are visceral and immediate:

- A pane runs off the right edge and the user never learns it exists.
- The window shrinks and the old frame's rows stay frozen on screen — ghosts that lie about state.
- A 3-column dashboard collapses into overlapping garbage at 50 columns.
- A long path silently loses its last segment — the part that mattered.
- The user resizes to read something better, and the app renders nothing at all.

Layout is the load-bearing wall of TUI UX. If it does not reflow, nothing built on top of it survives contact with a real user's terminal. This sheet is framework-agnostic on purpose: the constraint model below shows up in ratatui (Rust), Textual (Python), Bubble Tea / Lip Gloss (Go), Ink (JS), and notcurses, because the underlying problem — "I have W columns and H rows, distribute them" — is universal.

## The mental model: never reason in absolute cells

The single most important habit: **stop thinking in pixels-that-happen-to-be-cells, start thinking in constraints.** You almost never want "the sidebar is 30 wide." You want "the sidebar is *at most* 30 wide, *at least* 20, and the body takes the rest." The framework resolves those constraints against whatever `(width, height)` it currently has.

Three constraint primitives cover ~90% of real layouts. Every mature TUI framework exposes some spelling of them:

| Intent | ratatui | Textual (CSS) | Lip Gloss / Bubble Tea | Ink (Flexbox) |
|---|---|---|---|---|
| Take a fixed N cells | `Constraint::Length(n)` | `width: 30;` | `lipgloss.Width(n)` | `width={30}` |
| Take a share of leftover space | `Constraint::Fill(weight)` / `Ratio` | `fr` units / `1fr` | manual flex math | `flexGrow={1}` |
| Take what content needs, clamped | `Constraint::Min/Max` | `min-width` / `max-width` | clamp + truncate | `minWidth` / `maxWidth` |

Rules of thumb that keep layouts honest:

1. **At most one "flex/fill" region per axis** absorbs the slack. Everything else is fixed or clamped. If two regions both say "give me the rest," resize behavior becomes ambiguous and jittery.
2. **Fixed regions get a `Max` as well as a `Length`.** A status bar that is `Length(1)` is fine; a help panel that is `Length(40)` should be `Min(0)` + `Max(40)` so it yields gracefully when the terminal is narrower than 40.
3. **Every leaf that holds text declares an overflow policy** — wrap or truncate — *before* it can be clipped. Clipping is what happens when you forgot to decide.

## Designing for ANY size

Treat the terminal like a fluid web layout, not a print page. Borrow the responsive playbook directly:

- **Define breakpoints, not magic numbers.** Pick a small set of width thresholds and name them by what they *enable*, not by raw columns:

  ```text
  XS   < 60 cols   single pane, stacked, minimal chrome
  SM   60–99 cols  body + collapsible sidebar
  MD   100–159     sidebar + body + status
  LG   >= 160      sidebar + body + detail/preview pane
  ```

  Drive layout off the band, so the same code serves a phone-SSH session and a 4K terminal.

- **Demote, don't delete.** As width shrinks, progressively drop the *lowest-priority* region rather than crushing everything proportionally. A 3-pane dashboard at LG becomes 2-pane at MD (hide the preview), 1-pane at SM (sidebar becomes a toggleable overlay), and a bare list at XS. The user always keeps the primary content.
- **Stack on the height axis too.** Vertical space is scarcer than people think — a tmux pane can be 8 rows tall. If you have a header, body, and footer, the body is your flex region and the chrome must collapse first (e.g. merge a two-line header into one) when rows get tight.
- **Anchor proportions in `fr`/ratio, finish with `clamp`.** "Sidebar is 25% but never below 20 and never above 40" survives both extremes far better than any fixed width.

The test you run in your head for every screen: *does this read at 40 columns? does it not look absurdly empty at 200?* If a layout only looks right in a narrow band of sizes, it is not responsive — it is lucky.

## Minimum-size handling and the graceful "too small" state

There is always a size below which your real UI cannot honestly render. The failure mode is rendering it anyway: panes overlap, borders collide, text smears, or the framework panics on a zero-or-negative computed width.

The fix is a deliberate **floor state**. Pick a minimum `(min_w, min_h)` for your primary layout. Below it, render a single, centered, calm message instead — never a broken version of the real thing.

```text
┌────────────────────┐
│                    │
│  Terminal too small │
│  Need 60x20         │
│  Now 44x12          │
│                    │
└────────────────────┘
```

Design notes that make this state good rather than insulting:

- **State the requirement and the current size.** "Need 60x20, now 44x12" tells the user exactly how much to drag. "Too small" alone is a dead end.
- **Recover automatically.** This is a transient state tied to size, not an error — the moment SIGWINCH reports enough room, render the real UI. Never require a keypress to escape it.
- **Center and clamp the message itself**, because the floor state must also survive being shown at 10x3. Truncate hard rather than wrap into oblivion.
- **Pick the floor from your *tightest* essential region**, not an arbitrary number. If your list rows are unreadable below 40 columns and you need 3 rows of chrome plus 5 of content, your floor is ~40x8.

## Overflow: decide wrap vs. truncate, then make truncation legible

Content will exceed its box. The only question is what you do about it, and silence (clip) is the wrong answer.

**Per-region policy.** For each text region, choose explicitly:

- **Wrap** — for prose, descriptions, log lines you want fully readable. Costs vertical space; pairs with a scroll region.
- **Truncate** — for single-line cells, titles, paths, table columns where vertical space is fixed. Costs information; pairs with an ellipsis and ideally a way to see the full value (detail pane, tooltip-on-focus, status line).

**Truncate with a real ellipsis, width-aware.** Naive `s[:n]` is a bug magnet: it counts bytes or code points, not display columns, so it smears any CJK/emoji/combining-character content (covered in depth in `color-theming-and-monospace-canvas.md`). Measure in *display width* and reserve a column for the `…`:

```python
# Python — width-aware middle-ellipsis for paths (preserve the meaningful tail)
from wcwidth import wcswidth

def truncate_mid(s: str, max_cols: int, ell: str = "…") -> str:
    if wcswidth(s) <= max_cols:
        return s
    if max_cols <= 1:
        return ell[:max_cols]
    budget = max_cols - wcswidth(ell)
    head_budget = budget // 2
    tail_budget = budget - head_budget
    # walk from each end accumulating display width
    head, w = [], 0
    for ch in s:
        cw = wcswidth(ch)
        if w + cw > head_budget: break
        head.append(ch); w += cw
    tail, w = [], 0
    for ch in reversed(s):
        cw = wcswidth(ch)
        if w + cw > tail_budget: break
        tail.append(ch); w += cw
    return "".join(head) + ell + "".join(reversed(tail))

# truncate_mid("/home/user/projects/very/deep/path/config.toml", 28)
# -> "/home/user/proj…/config.toml"   (keeps the filename, the part you care about)
```

For columns, prefer *middle* truncation for paths/identifiers (the tail carries identity) and *end* truncation for prose. Either way, the ellipsis is a contract: it tells the user "there is more here," which silent clipping never does.

**Scroll regions for the overflow you can't truncate away.** When content is intrinsically longer than its box (a log, a file, a long list), the region becomes a viewport over a larger buffer. The layout's job is to size the viewport; the scroll/virtualization mechanics (render only visible rows, track offset, stick-to-bottom) live in `rendering-and-redraw-discipline.md` and `event-loop-and-state-architecture.md`. The layout contract is simply: **the scroll region's height is whatever the flex solver gives it, and it must recompute that height on every resize.** A scrollbar or `[12–34 / 220]` position indicator turns invisible overflow into a visible affordance.

## Reflow on resize: SIGWINCH → recompute, don't patch

The crash you must not ship: the user drags the window and the previous frame's content stays on screen, half-overwritten, rows from the old size colliding with the new. That is what happens when you compute layout once at startup and never again.

The correct model is **stateless layout recomputation**: layout is a pure function of the *current* `(width, height)`, recomputed from scratch whenever the size changes. You never incrementally edit a frozen layout; you regenerate it.

How the size change reaches you depends on the substrate:

- On Unix, the kernel sends **`SIGWINCH`** when the terminal size changes. Most TUI frameworks already trap it and surface it as an event (ratatui via the `crossterm`/`termion` event stream, Bubble Tea as `tea.WindowSizeMsg`, Textual as an `on_resize` / `Resize` event, Ink via `process.stdout.on('resize')`). Prefer the framework's event over installing your own raw signal handler — raw async signal handling that mutates render state is its own footgun (see `lifecycle-signals-and-terminal-restoration.md`).
- On Windows there is no SIGWINCH; size changes arrive through console input events. Again, lean on the framework, which abstracts this.

The reflow sequence on every size event:

1. Read the new `(cols, rows)` from the event (do not re-query lazily — the event already carries the truth).
2. Re-evaluate the breakpoint band — maybe you cross from MD to SM and drop a pane entirely.
3. Re-run the constraint solver against the new dimensions to get fresh rectangles.
4. **Clamp dependent state** so it stays valid: a scroll offset that was fine at 50 rows may now point past the end at 20 rows — clamp it; a selected index that scrolled off-screen should be brought back into view. (Selection/scroll *identity* survival under filtering is `event-loop-and-state-architecture.md`'s job; here you only re-clamp against the new viewport.)
5. Repaint from the new layout. With an alternate screen + a diffing renderer this is flicker-free; without diffing you'll need a full clear, which is exactly the flicker problem `rendering-and-redraw-discipline.md` solves.

```go
// Go / Bubble Tea — layout is recomputed from WindowSizeMsg, never cached as absolutes.
func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case tea.WindowSizeMsg:
        m.w, m.h = msg.Width, msg.Height
        m.layout = computeLayout(m.w, m.h)   // pure: size in, rects out
        // re-clamp viewport-dependent state against the new body height
        m.list.SetHeight(m.layout.body.h)
        if m.scrollOff > m.maxScroll() {
            m.scrollOff = m.maxScroll()
        }
        return m, nil
    }
    return m, nil
}
```

Debounce only if resize events flood (some terminals emit a burst during a drag) — coalesce to the last size and reflow once on settle. But never *drop* the final event: the last size is the one the user is looking at.

## Worked example: a 3-pane layout that survives 40 and 200 columns

The canonical TUI shape — sidebar (navigation), body (primary content), detail/preview (context) — is also the canonical place fixed-width assumptions go to die. Here is a responsive version in two frameworks. The behavior contract is identical:

- **LG (>= 160 cols):** all three panes. Sidebar clamped 24–32, detail clamped 30–48, body takes the rest.
- **MD (100–159):** drop the detail pane; sidebar + body.
- **SM (60–99):** sidebar collapses to a narrow rail (or a toggleable overlay); body dominates.
- **XS (< 60 wide) or < 8 rows:** floor state ("terminal too small").

### ratatui (Rust)

```rust
use ratatui::layout::{Constraint, Direction, Layout, Rect};

enum Band { Floor, Sm, Md, Lg }

fn band(area: Rect) -> Band {
    if area.width < 60 || area.height < 8 { Band::Floor }
    else if area.width < 100 { Band::Sm }
    else if area.width < 160 { Band::Md }
    else { Band::Lg }
}

/// Pure function: terminal Rect in, pane Rects out. Re-run on every resize.
fn compute_panes(area: Rect) -> Panes {
    // Reserve one row for a status/help line on the height axis first.
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(area);
    let (content, status) = (rows[0], rows[1]);

    match band(area) {
        Band::Floor => Panes::TooSmall { area, status },

        Band::Sm => {
            // narrow rail + body; detail demoted away entirely
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(8), Constraint::Min(20)])
                .split(content);
            Panes::Two { sidebar: cols[0], body: cols[1], status }
        }

        Band::Md => {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                // clamp sidebar 24..=32, body absorbs the rest
                .constraints([Constraint::Max(32), Constraint::Min(40)])
                .split(content);
            Panes::Two { sidebar: cols[0], body: cols[1], status }
        }

        Band::Lg => {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Max(32),  // sidebar, clamped
                    Constraint::Min(60),  // body, the flex region
                    Constraint::Max(48),  // detail, clamped
                ])
                .split(content);
            Panes::Three { sidebar: cols[0], body: cols[1], detail: cols[2], status }
        }
    }
}
```

`compute_panes` is called fresh inside the draw closure on every frame, so a `crossterm::event::Event::Resize(w, h)` automatically produces a correct relayout — there is no cached geometry to go stale. The floor branch renders the centered "too small" message and ignores the rest.

### Textual (Python) — same contract, CSS-driven

Textual leans on CSS-like layout and `on_resize`, so the band logic toggles classes and the stylesheet does the constraint math:

```python
from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.events import Resize

class Dashboard(App):
    CSS = """
    #sidebar { width: 32; min-width: 24; }
    #detail  { width: 48; min-width: 30; }
    #body    { width: 1fr; }            /* the single flex region */

    /* Responsive demotion via classes set in on_resize */
    .band-md #detail { display: none; }                 /* drop detail */
    .band-sm #detail { display: none; }
    .band-sm #sidebar { width: 8; min-width: 8; }       /* collapse to a rail */
    .floor #sidebar, .floor #body, .floor #detail { display: none; }
    .floor #toosmall { display: block; content-align: center middle; }
    """

    def compose(self) -> ComposeResult:
        yield Static("nav",     id="sidebar")
        yield Static("content", id="body")
        yield Static("context", id="detail")
        yield Static("",        id="toosmall")

    def on_resize(self, event: Resize) -> None:
        w, h = event.size.width, event.size.height
        band = ("floor" if (w < 60 or h < 8)
                else "band-sm" if w < 100
                else "band-md" if w < 160
                else "band-lg")
        self.screen.set_classes(band)   # stylesheet resolves the rest
        toosmall = self.query_one("#toosmall", Static)
        toosmall.update(f"Terminal too small\nNeed 60x8 — now {w}x{h}")
```

Two frameworks, two idioms (imperative constraint solver vs. declarative CSS), one identical responsive contract. That portability is the proof the model is right: you are designing *constraints and bands*, not pixel positions.

## Common mistakes

- **Hardcoding 80x24 (or any fixed size).** The classic. Query the real size at startup *and* reflow on resize. 80x24 is a historical default, not a guarantee — modern terminals are routinely 200+ wide and as small as 40.
- **Computing layout once and caching absolute rectangles.** The frozen-frame bug. Make layout a pure function of current size, recomputed every resize/frame. Never incrementally patch a stale layout.
- **Ignoring SIGWINCH / the resize event entirely.** The app keeps drawing at the old size; rows go stale, content clips, ghosts pile up. Subscribe to the framework's resize event and reflow.
- **No minimum-size / "too small" state.** At small sizes the constraint solver hands you tiny or zero-width rects; you render overlapping garbage or divide-by-zero panic. Define a floor and show a calm, auto-recovering message below it.
- **Silent clipping instead of a wrap-vs-truncate decision.** If you didn't choose, you chose clip — the worst option, because it hides that information was lost. Decide per region; truncate with a visible width-aware ellipsis.
- **Byte/codepoint-based truncation.** `s[:n]` smears CJK, emoji, and combining characters because they aren't one column each. Measure in display width (`wcwidth`/`unicode-width`). See `color-theming-and-monospace-canvas.md`.
- **Two flex regions fighting for the slack.** Ambiguous, jittery resize. Exactly one fill region per axis; everything else fixed or clamped.
- **Demoting by crushing instead of dropping.** Proportionally shrinking every pane gives you three useless panes. Drop the lowest-priority region at narrow widths so the primary content stays usable.
- **Forgetting the height axis.** Wide-but-short terminals (8-row tmux panes) break layouts that only reflow horizontally. Constrain and collapse on rows too.
- **Not re-clamping viewport state after resize.** A scroll offset or selection index valid at 50 rows points off the end at 20. Clamp dependent state inside the reflow path.
- **No automated test that the layout holds at extremes.** Render to a virtual buffer at 40x10, 80x24, and 200x60 and snapshot it. Golden-frame layout tests catch the regression you'd otherwise ship. See `testing-tuis.md`.

## Where this connects

- **`rendering-and-redraw-discipline.md`** — once layout is correct, repaint it without flicker (diffing, damage tracking, viewport virtualization).
- **`lifecycle-signals-and-terminal-restoration.md`** — SIGWINCH lives alongside SIGINT/SIGTERM; how to handle signals without corrupting render state or leaking the terminal on exit.
- **`color-theming-and-monospace-canvas.md`** — display-width measurement that makes truncation and column alignment correct for CJK/emoji.
- **`event-loop-and-state-architecture.md`** — selection/scroll identity that must *survive* filtering, distinct from the resize re-clamp described here.
- **`testing-tuis.md`** — snapshot/golden-frame regression tests across multiple terminal sizes.
