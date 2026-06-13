---
name: rendering-and-redraw-discipline
description: Use when a TUI flickers, tears, or jitters; when the screen visibly repaints, blinks, or "blanks" on every keypress or tick; when CPU sits high while the UI is idle; when a fan spins or an SSH/remote session lags because the app floods the terminal with bytes; when scrolling a long list, log tail, table, or tree feels janky; when you're writing a render/draw/paint loop, calling `clear()` then redrawing everything, doing raw `print`/`stdout.write`/ANSI escapes by hand, or rendering all N rows instead of the visible viewport; when frames update faster than the eye or terminal can keep up; or when you're choosing between immediate-mode and retained-mode rendering.
---

# Rendering & Redraw Discipline

**The UX stake:** a TUI that flickers, tears, or stutters reads as *broken* before the user has judged a single feature. Flicker is the terminal equivalent of a flashing, mis-aligned web page that reflows under the cursor — it signals "amateur" and "untrustworthy" instantly, and on a remote (SSH) session it also makes the app *physically slow*, because every wasted frame is bytes pushed down a high-latency pipe. Rendering discipline is not a performance micro-optimization you bolt on later. It is the difference between an interface that feels solid and one that feels like it's coming apart. **Redraw is something you earn with a change, not something you do on a clock.**

This sheet closes three failure modes that travel together:

- **Flicker** — visible blink/blank as the screen is cleared and rewritten.
- **Full-screen repaint every frame** — rewriting every cell whether or not it changed.
- **Tearing** — a half-updated frame on screen because writes weren't presented atomically.

They share one root cause: *the program decides what to put on the glass by brute force on a timer, instead of by computing the difference caused by an event.*

---

## The mental model: immediate-mode vs retained-mode

Every TUI you'll build sits on one of two rendering philosophies. Knowing which one you're in tells you where flicker comes from and who is responsible for preventing it.

### Immediate-mode

You describe the **entire UI from scratch every frame** as a pure function of state. There are no persistent widget objects; you call `paragraph(...)`, `list(...)`, `block(...)` afresh each draw. ratatui (Rust) and Dear ImGui-style libraries work this way.

The trap: "I rebuild the whole UI each frame" is *not* the same as "I repaint every cell each frame." Immediate-mode libraries reconcile your freshly-described frame against the previously-presented one and emit **only the differences**. The cost you must avoid is calling that draw on a free-running loop when nothing changed — and, separately, *clearing the terminal yourself* before letting the library draw, which throws away the diff baseline and forces a full repaint (the #1 cause of flicker in ratatui apps).

### Retained-mode

You build a **tree of persistent widgets once**, then mutate their properties. The framework owns the tree, tracks which nodes are dirty, and repaints only the damaged regions on the next tick. Textual + Rich (Python), Ink (React-for-the-terminal, JS), and notcurses' plane model work this way.

The trap: bypassing the tree. The moment you `print()`, `console.print()`, or write raw ANSI *around* a retained framework, you scribble on cells the framework believes it owns. On the next reconcile it either fights you (flicker as it "corrects" your output) or leaves stale glyphs behind (tearing). In retained-mode, **the framework's diff is the only writer of truth — do not write outside it.**

| | Immediate-mode (ratatui) | Retained-mode (Textual, Ink, notcurses) |
|---|---|---|
| You provide | A full frame description, every draw | A widget tree, mutated over time |
| Diffing owner | Library (frame-vs-frame buffer diff) | Framework (dirty-node / damage tracking) |
| Flicker comes from | Drawing on a busy loop; `clear()` before draw; raw writes | Writing outside the tree; forcing full refresh |
| Your job | *Call draw only on change*; never hand-clear | *Mutate state, not the screen*; never print raw |

In **both** models the discipline is identical: **let the library compute the diff, and only trigger it when something actually changed.**

---

## Rule 1 — The framework diffs. Do not fight it.

Mature TUI frameworks maintain a back buffer (the frame you're composing) and a front buffer (what's currently on the glass), compute the cell-level delta, and emit the minimal cursor-move + write sequence to reconcile them. This is the same double-buffering idea games use, applied to a grid of character cells. **It already solves flicker and tearing for you — if you don't sabotage it.**

You sabotage it three ways:

1. **Clearing the screen yourself before the draw.** `terminal.clear()`, `ESC[2J`, `console.clear()` — these blank the front buffer, so the next diff sees "everything changed" and repaints the whole screen. That repaint *is* the flicker. Let the framework's first draw populate the screen and every subsequent draw diff against it. The only legitimate full clear is on a confirmed resize, and even then the framework usually handles it.
2. **Writing raw bytes around the framework.** A stray `println!`, `print()`, or `process.stdout.write()` lands on cells the framework's front buffer doesn't know about. Now the framework's model of the screen is wrong, and its next diff produces garbage or tears. Route *all* output through the framework's draw path.
3. **Forcing a full refresh "to be safe."** Calling `refresh(full=True)` / `screen.refresh()` on every tick discards damage tracking and repaints everything. Reserve full refresh for the genuine edge cases (resize, return from a shelled-out subprocess, recovery from corruption) — never as a habit.

> **The principle:** your job is to give the diff engine an *accurate, minimal* description of the desired frame and let it do the work. Every time you reach past it to the raw terminal, you're taking on the diffing yourself — and you will do it worse.

---

## Rule 2 — Redraw on change/event, not on a busy frame loop

The single most common naive TUI shape is a `loop { draw(); }` that repaints as fast as the CPU allows, or a `setInterval(render, 16)` that fires 60 times a second regardless of whether anything happened. **Both burn a core, both flood remote terminals, and both are pure waste because nothing changed between most frames.**

A TUI is an **event-driven** program, not a game with continuous physics. The screen should change exactly when one of these occurs:

- the user pressed a key, moved/clicked the mouse, or resized the terminal;
- async work completed (a request returned, a file-watch event fired, a child process emitted output);
- a timer you *deliberately* armed elapsed (a spinner frame, a clock tick, a debounce flush).

The loop's job is to **block until one of those happens**, mutate state, mark the UI dirty, and draw *once*. If nothing happened, the loop sleeps and the screen is untouched — zero bytes, zero CPU.

```rust
// ratatui (Rust) — event-driven loop. Draw only after a real event mutated state.
// crossterm provides poll(); the loop blocks instead of spinning.
fn run(terminal: &mut Terminal<impl Backend>, app: &mut App) -> io::Result<()> {
    terminal.draw(|f| ui(f, app))?;        // initial paint, once

    loop {
        // Block until an event OR the next deliberate deadline (spinner/clock).
        // No event => no wakeup => no redraw => no CPU. This is the whole point.
        let timeout = app.next_deadline().saturating_duration_since(Instant::now());
        if event::poll(timeout)? {
            match event::read()? {
                Event::Key(k) if k.code == KeyCode::Char('q') => return Ok(()),
                Event::Key(k)    => app.on_key(k),
                Event::Resize(w, h) => app.on_resize(w, h),
                Event::Mouse(m)  => app.on_mouse(m),
                _ => {}
            }
            app.dirty = true;              // a real change happened
        } else {
            // poll timed out: only the armed deadline (e.g. spinner) is due
            app.tick();                    // advance spinner/clock state
            app.dirty = app.has_animation();
        }

        if app.dirty {
            terminal.draw(|f| ui(f, app))?; // ratatui diffs back-vs-front buffer here
            app.dirty = false;
        }
        // NB: no terminal.clear() anywhere. Clearing kills the diff and causes flicker.
    }
}
```

The retained-mode frameworks make this automatic — *if you let them*. In Textual you never call draw at all; you mutate reactive state and the framework schedules a repaint of only the affected widgets:

```python
# Textual (Python) — retained-mode. You mutate reactive state; the framework
# decides what to repaint. There is no render loop you own, and no clear() to call.
from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import Label, Input
from textual.containers import Vertical

class Counter(App):
    # reactive => assigning to it marks the dependent widget dirty and schedules
    # a *partial* repaint of just that widget. No busy loop, no full-screen paint.
    count: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(id="display")
            yield Input(placeholder="press +/- (or type to filter)…")

    def watch_count(self, value: int) -> None:
        # Runs only when count actually changes -> updates one Label, not the screen.
        self.query_one("#display", Label).update(f"Count: {value}")

    def on_key(self, event) -> None:
        if event.key == "plus":
            self.count += 1     # triggers watch_count -> one widget repainted
        elif event.key == "minus":
            self.count -= 1

# Anti-pattern to NOT write here: a `while True: self.refresh()` background task.
# That defeats Textual's damage tracking and reintroduces flicker + CPU burn.
```

> **The principle:** the terminal is updated as a *consequence* of a change, never as a *schedule*. If you can't name the event that justifies a redraw, don't redraw.

---

## Rule 3 — Render the viewport, not the whole dataset (virtualization)

Damage-level diffing fixes *how much* the framework writes to the screen. It does **not** save you from *building* a frame you can't display. If you have 200,000 log lines and a 40-row pane, formatting, styling, and laying out all 200,000 rows every frame is wasted work even if only 40 cells change on the wire — you've spent CPU producing 199,960 rows the user will never see, and on a keystroke that latency is felt.

**Render only the slice that fits, plus a small overscan margin.** This is virtualization (a.k.a. windowing), and it is the difference between a log viewer that stays smooth at millions of lines and one that pegs a core on every arrow-key.

```go
// Bubble Tea (Go) — viewport windowing. Build & style ONLY the visible rows.
// `items` may hold millions of entries; we touch at most height+overscan of them.
func (m model) View() string {
    h := m.height          // visible rows in the list pane
    if h <= 0 { return "" }

    const overscan = 2     // tiny margin so fast scroll doesn't show blank edges
    start := m.offset - overscan
    if start < 0 { start = 0 }
    end := m.offset + h + overscan
    if end > len(m.items) { end = len(m.items) }

    var b strings.Builder
    // Loop bound is the WINDOW, never len(m.items). This is the load-bearing line:
    // formatting cost is O(viewport), not O(dataset).
    for i := start; i < end; i++ {
        line := m.items[i]
        if i == m.selected {
            b.WriteString(selectedStyle.Render(truncateToWidth(line, m.width)))
        } else {
            b.WriteString(truncateToWidth(line, m.width))
        }
        b.WriteByte('\n')
    }
    return b.String()
}
```

The same rule holds in every framework: ratatui's `List`/`Table` with `ListState` only renders rows in view; Textual's `DataTable`/`ListView` and `RichLog` window internally; Ink should map over a sliced array, not the full one. **Never iterate the entire collection to build a frame.** If your draw function's loop bound is the dataset length rather than the viewport height, that's a bug, full stop.

This also bounds *memory* pressure: a log tail should sit behind a **ring buffer** with a hard cap, so the backing store can't grow without limit. (Buffer sizing and the scroll/selection-identity problems that virtualization exposes are owned by `event-loop-and-state-architecture.md`; cross-reference it whenever you windowed a list.)

---

## Rule 4 — Frame budget + throttling high-frequency updates

Some sources push updates faster than a human can perceive or a terminal can paint: a log tail at 10,000 lines/sec, a progress callback firing per byte, a file-watcher in a storm, a metrics stream. **Painting once per source-event is both pointless (the eye resolves ~30–60 changes/sec) and ruinous (you flood the terminal and starve the input loop).** Decouple the *data rate* from the *frame rate*.

Set a frame budget and **coalesce**: accept every event into state immediately, but redraw at most once per budget window (target a frame cap of ~30–60 fps; on remote/SSH sessions, lower it — every frame is bytes over the wire).

```javascript
// Ink (JS) — coalesce a high-frequency stream into a capped frame rate.
// Data arrives at any rate; we re-render at most ~30fps so the terminal isn't flooded.
import { useState, useEffect, useRef } from 'react';
import { Text } from 'ink';

function LogTail({ stream }) {
  const [view, setView] = useState([]);
  const buffer = useRef([]);          // accept ALL events here, instantly
  const MAX = 5000;                   // ring-buffer cap: never grows unbounded

  useEffect(() => {
    const onLine = (line) => {
      buffer.current.push(line);
      if (buffer.current.length > MAX) buffer.current.shift();
    };
    stream.on('line', onLine);

    // The ONLY thing tied to a clock: a ~33ms (≈30fps) flush. The stream can
    // emit 10k lines/sec; React still re-renders at most ~30 times/sec.
    const id = setInterval(() => {
      // Snapshot only the visible tail; never hand React the whole buffer.
      setView(buffer.current.slice(-40));
    }, 33);

    return () => { stream.off('line', onLine); clearInterval(id); };
  }, [stream]);

  return view.map((l, i) => <Text key={i}>{l}</Text>);
}
```

Two more budgeting moves that belong with this rule:

- **Debounce derived work, not just paints.** A filter box that re-runs a substring scan over a huge list on *every keystroke* will block the loop even with perfect diffing — because the expensive part is the scan, not the paint. Debounce the scan (e.g. 80–120 ms after the last keystroke) and run it off the input thread. (Debounce/async is owned by `feedback-latency-and-async-work.md`; redraw throttling here is its rendering-side sibling.)
- **Animate on a timer you *armed*, and disarm it.** A spinner needs a periodic redraw — that's a *deliberate* deadline (see Rule 2's `next_deadline`), not a busy loop. When the spinner stops, disarm the timer so the loop goes back to fully event-driven and idle CPU returns to zero.

> **The principle:** state absorbs the firehose; the frame budget meters what reaches the glass. Never let an external data rate dictate your paint rate.

---

## Rule 5 — Double buffering: why raw `stdout` writes tear

**Tearing** is what you see when a frame reaches the screen *half-updated*: the top of a table is the new data, the bottom is the old, because the writes that should have arrived as one atomic frame were dribbled out and the terminal rendered them as they landed.

The defense is **double buffering**: compose the full next frame in an off-screen back buffer, compute the delta against the on-screen front buffer, then emit the reconciling write sequence so the visible frame transitions in one coherent batch. This is exactly what ratatui's `Buffer`, Textual/Rich's `Compositor`, and notcurses' planes-to-rasterized-frame pipeline do. **The atomic-frame guarantee is the entire reason these libraries exist.**

Raw `stdout`/`println!`/`console.print()` writes have **none** of this:

- there is no back buffer, so partial output is visible as it's flushed — that's the tear;
- there is no diff, so you either over-write (flicker) or under-write (stale glyphs);
- the framework's front-buffer model is now wrong, so its *next* legitimate frame is corrupted too;
- on a line-buffered or slow stream, your frame can be split across flushes mid-escape-sequence, producing visible garbage.

```text
        Tearing (raw writes, no back buffer)        Atomic frame (double-buffered)
        ───────────────────────────────────        ──────────────────────────────
        write row 0 ──► visible immediately         compose ALL rows in back buffer
        write row 1 ──► visible immediately         diff back vs front buffer
        …user's eye catches the half-drawn frame    emit one reconciling write batch
        write row N ──► visible immediately         ──► whole frame appears coherent
```

**Practical rules:**
- Do every paint through the framework's draw/compose API. There is no "just this once" raw write — the one stray `print()` is the bug you'll spend an afternoon chasing.
- Need a hardware/terminal feature the framework supports (synchronized output, `ESC[?2026h/l`)? Use the framework's switch for it, don't hand-roll the escape.
- Must shell out to a program that writes to the terminal directly (`$EDITOR`, `less`, `fzf`)? Suspend the framework cleanly first (leave the alt screen / its raw-mode guard), run the child, then restore and force *one* full refresh on return. That return-from-subprocess case is the **only** time a deliberate full repaint is correct. (Suspend/restore handshake is owned by `lifecycle-signals-and-terminal-restoration.md`.)

> **The principle:** a frame must reach the screen all-at-once or not at all. Anything that writes outside the double-buffered compose path forfeits that guarantee.

---

## Common mistakes

- **`clear()` then redraw, every frame.** The textbook flicker generator. Clearing blanks the front buffer so the diff repaints everything. Remove the clear; let the diff work.
- **A free-running `loop { draw() }` or `setInterval(render, 16)`.** Repaints when nothing changed. Burns CPU, floods SSH. Replace with a *blocking* event loop that draws only on a real change or an armed deadline.
- **Loop bound = dataset length, not viewport height.** Building all N rows to show 40. Window the slice; overscan by 1–2 rows; cap the backing buffer.
- **A stray `print!`/`console.print()`/`stdout.write` inside or beside a framework draw.** Corrupts the front-buffer model → flicker or tearing on the *next* legitimate frame. Route everything through the draw path.
- **`refresh(full=True)` / `screen.refresh()` on every tick "to be safe."** Defeats damage tracking; it's a full repaint with extra steps. Reserve full refresh for resize and return-from-subprocess.
- **One paint per source-event on a high-rate stream.** A 10k-line/sec tail painting per line floods the terminal and starves input. Coalesce into a ~30–60 fps budget; state absorbs the rate.
- **Filtering/sorting/searching synchronously inside the keystroke handler.** The diff is fine; the *scan* blocks. Debounce + move off-thread (see `feedback-latency-and-async-work.md`).
- **Spinner timer left armed after the work finishes.** Keeps the loop waking and CPU non-zero forever. Disarm animation timers when animation stops.
- **Assuming "rebuild the whole UI" (immediate-mode) means "repaint the whole screen."** It does not — the library diffs your rebuilt frame. The waste is *drawing when nothing changed*, not *describing the whole frame*.

---

## Red flags — STOP

If any of these is true of code you're writing or reviewing, stop and fix it before moving on:

- [ ] There is a `clear()` / `ESC[2J` / `console.clear()` immediately before a per-frame draw.
- [ ] The render is driven by a free-running loop or a fixed interval rather than by events + armed deadlines.
- [ ] A draw/View function iterates the **full** dataset instead of the visible window.
- [ ] There is any raw `print` / `stdout.write` / hand-written ANSI **outside** the framework's draw path.
- [ ] A high-frequency source triggers one paint per event, with no frame budget / coalescing.
- [ ] `refresh(full=True)` or equivalent runs on a normal tick (not just resize / subprocess return).
- [ ] Idle CPU is non-zero — the app paints (or wakes to check) when nothing has changed.
- [ ] An animation timer (spinner/clock) stays armed after the animation has stopped.
- [ ] You cannot name the specific event that justifies each redraw.

---

## Counters to the rationalizations (don't skip the rule)

> "It's a small app, I'll just clear and redraw everything — diffing is premature optimization."
Flicker is visible in a 50-line app the same as a 50,000-line one. The "optimization" *is* the correct draw path; the brute-force version is the one that needs justifying. The fix is not more code — it's *deleting* the `clear()`.

> "Repainting every frame is simpler than tracking what changed."
You're not tracking it — the framework already is. Letting the diff work is *less* code than a clear-and-repaint, and it's the path the library was built for. The "simple" version reimplements rendering badly.

> "60fps feels responsive, so I'll just render on a timer."
Responsiveness comes from redrawing *promptly when something changes*, which an event loop does in well under a frame. A free-running timer adds latency between change and paint (you wait for the next tick), wastes a core, and on SSH makes the app *slower*. Event-driven is both faster-feeling and cheaper.

> "Rendering all the rows is fine, the terminal only shows 40 of them anyway."
The terminal clips them, but *you still built and styled all N* before it clipped — that cost lands on the user's keystroke. The clip is not free; the work before the clip is the latency.

> "One raw `print` for a debug line won't hurt."
It corrupts the framework's front-buffer model, so the *next* real frame tears or flickers — and you'll debug that for an hour without connecting it to the print. Send debug output to a log file or a dedicated pane, never to the live screen.

> "The data really does arrive 10,000 times a second, so I have to render that fast."
No human resolves 10,000 changes/sec and no terminal paints them. You must *absorb* that rate into state and *meter* what reaches the glass. Painting at the data rate is the bug, not the requirement.

> "I'll add a `refresh(full=True)` to fix a glitch I'm seeing."
A full refresh hides the symptom (the framework's model got out of sync) instead of fixing the cause (something wrote outside the draw path, or you cleared the buffer). Find the rogue writer; don't paper over it with a repaint that reintroduces flicker.

---

## See also

- `lifecycle-signals-and-terminal-restoration.md` — suspend/restore handshake for shelling out; the one legitimate full repaint on return; never leaving raw mode / alt screen on crash.
- `event-loop-and-state-architecture.md` — ring-buffer caps, scroll/selection identity under filtering, stick-to-bottom tailing — the state side of what virtualization exposes.
- `feedback-latency-and-async-work.md` — debouncing derived work, moving scans off the render thread, in-flight feedback while async work runs.
- `layout-and-responsive-composition.md` — resize reflow and minimum-size handling, the legitimate triggers for a full recompute.
- `terminal-substrate-and-constraints.md` — capability probes (synchronized output, truecolor, alt screen) before you assume a rendering feature exists.
- `testing-tuis.md` — golden-frame / snapshot regression tests that catch flicker-causing full repaints and layout drift.
