---
name: event-loop-and-state-architecture
description: Use when a TUI freezes or stops repainting while it fetches data, runs a query, or tails a file; when input lags behind keystrokes because a synchronous network/disk/CPU call sits inside the event loop; when state and rendering have tangled into one mutable blob so the screen and the data disagree; when scroll position or the selected row jumps to the wrong item after a filter changes the list; when an auto-scrolling/tailing view fights the user's manual scrolling; when an in-memory log or output buffer grows without bound and the process bloats; or when you are choosing an architecture for a terminal app (ratatui, Textual, Bubble Tea, Ink, notcurses) and need an event-loop spine, Model-View-Update / Elm-style message passing, commands/effects for side effects, and off-loop I/O.
---

# Event Loop & State Architecture

## Overview

Every TUI is a loop: read input, update state, draw. Get that spine right and the rest of the design has a foundation to stand on. Get it wrong and nothing else can save the app — the prettiest layout in the world is worthless if the screen freezes the moment the user does anything interesting.

This sheet is about the **spine**: the event loop, the **Model-View-Update (MVU / Elm)** discipline that keeps state and rendering separate, and the rule that side effects (network, disk, subprocesses, timers) must live *off* the loop so they never stall input or rendering.

**Core principle**: *Render is a pure function of state. State changes only in response to messages. Side effects never run inside the loop.* When all three hold, the UI stays responsive no matter what the world does, and you can reason about — and test — exactly what the screen shows.

## When to Use

Load this sheet when:

- The TUI **goes unresponsive** during a fetch, query, build, or file load — the spinner doesn't spin, keys don't register, resize doesn't reflow.
- **Input lags** behind the keyboard — you type `hello` and the screen catches up a second later.
- **State and view are tangled** — render code mutates data, or two places hold the "current selection" and they drift apart.
- **Scroll/selection breaks after filtering** — you select row 7, type a filter, and now you're on the wrong item (or the cursor vanishes).
- A **tailing/follow view** (logs, chat, `ps`-style monitors) jumps to the bottom while the user is trying to scroll up.
- A **buffer grows forever** — every log line is appended to a `Vec`/list/array with no cap, and memory climbs until the OOM killer arrives.
- You're **picking an architecture** for a new terminal app and want a spine that scales past a toy.

**Don't use for**: terminal restoration on crash (see `lifecycle-signals-and-terminal-restoration.md`); flicker/redraw cost (see `rendering-and-redraw-discipline.md`); how to *show* in-flight work and debounce keystrokes (see `feedback-latency-and-async-work.md` — this sheet gets the work *off the loop*; that sheet makes the waiting *feel* good).

---

## The UX Stake: A Frozen TUI Is a Broken TUI

A GUI that hangs at least keeps its window, its title bar, its OS-drawn close button. A TUI that hangs is **indistinguishable from a crashed process**. The cursor stops blinking where the user expects it, `Ctrl-C` may or may not be heard (if you swallowed it and then blocked, it won't be), and the user is left staring at a static grid of characters with no way to tell whether the app is working hard or dead.

So the user does what users do: they hit `Ctrl-C` harder, then `Ctrl-\`, then close the terminal tab — and now your alt-screen/raw-mode cleanup never runs and their *next* shell prompt is invisible. One blocked event loop cascades into a wrecked terminal.

This is why architecture is a UX concern, not just an engineering one:

- **Responsiveness is trust.** Sub-100ms acknowledgement of every keystroke is the difference between "this tool is solid" and "this tool is janky." The event loop is where that promise is kept or broken.
- **Freezes destroy the mental model.** Users build a model of "I press, it responds." A single multi-second freeze teaches them the tool is unreliable, and they hedge — saving constantly, avoiding features, switching away.
- **A frozen loop can't clean up.** A loop blocked on a `recv()` can't process `SIGWINCH` (stale layout), can't honor a quit key, can't even *render* "loading…". The freeze and the broken-exit are the same bug.

The architecture's first job is to **guarantee the loop keeps turning** — always reading input, always able to draw — regardless of how slow the world is.

---

## Model-View-Update: The Spine

Borrowed from the Elm architecture and now the dominant TUI pattern (Bubble Tea is explicitly Elm; ratatui apps converge on it; Textual is a message-passing variant; Ink is React-with-`useState`/`useReducer`), MVU gives you three named pieces and one rule connecting them:

```
        ┌─────────────────────────────────────────────┐
        │                                             │
        ▼                                             │
   ┌─────────┐   message    ┌──────────┐   new state  │
   │  EVENT  │ ───────────► │  UPDATE  │ ────────────►│
   │  LOOP   │              │ (pure-ish)│              │
   └─────────┘              └──────────┘              │
        ▲                        │                    │
        │                        │ optional command   │
        │ input + command        ▼                    │
        │ results            ┌──────────┐    state     │
        └─────────────────── │ EFFECTS  │ ◄───────────┘
                             │ (off-loop)│
                             └──────────┘
                                  │
                          ┌──────────┐
   draw every tick ──────►│   VIEW   │  pure: state → frame
                          └──────────┘
```

- **Model (state)** — one owned, serializable description of *everything the screen could show*: the data, the filter text, which pane has focus, the scroll offset, the selected item's identity, whether a request is in flight. Plain data. No I/O handles, no render objects, no callbacks-with-captured-mutable-state.
- **View (render)** — a **pure function** `view(state) -> frame`. It reads the model and produces cells/widgets. It must not mutate the model, not perform I/O, not depend on anything but its arguments. Given the same model, it always draws the same frame.
- **Update** — `update(state, message) -> (state, command)`. The *only* place state changes. It takes the current state and one message (a keypress, a resize, a "data arrived" event, a tick), returns the next state, and optionally a **command** describing a side effect to run.

The rule that makes this work: **the only way to change state is to send a message into `update`.** Input handlers don't mutate state directly. Background tasks don't mutate state directly. Everything funnels through messages, so there is exactly one ordering of state changes and one place to reason about them.

### Why this closes the failure modes

- **Tangled state and view** disappears because the view is *forbidden* from mutating and the model holds no render objects. The screen can never disagree with the data — it *is* a function of the data.
- **Input lag** disappears because `update` is fast (it just transforms data and *describes* effects), and slow work is a command run off-loop.
- **Off-loop work** is structural: commands are the only way to do I/O, and commands run on the runtime/threadpool/async executor, not in `update`.

---

## State ≠ View: Render Is a Pure Function of State

The single most common architectural mistake in TUIs is letting rendering and state-mutation share the same code. It always starts innocently — "I'll just decrement the scroll offset right here in the draw function when I notice it's off the screen" — and ends with a screen that flickers between two truths because the data was edited mid-paint.

Hold the line:

| Belongs in **state (model)** | Belongs in **view (render)** | Belongs **nowhere near either** |
|---|---|---|
| selected item *identity* | how a selected row is styled | the network socket |
| scroll offset | which rows are visible this frame | the file handle being tailed |
| filter text | how matches are highlighted | the spawned subprocess |
| "request in flight" flag | the spinner glyph for that frame | the actual HTTP call |
| terminal size (from resize msg) | wrap vs. truncate of a long line | a timer that fires every 1s |

**Test of purity:** could you call `view(state)` a thousand times in a tight loop and get the identical frame and zero side effects every time? If not, something that should be a message-driven state change is hiding inside render.

A practical consequence: because `view` is pure, you can **snapshot-test** it — render a known model to a string buffer and golden-compare it (see `testing-tuis.md`). You can't golden-test a render function that phones home.

---

## Effects / Commands: The Only Door for Side Effects

`update` must stay fast and (mostly) pure, but real apps must fetch, write, spawn, and wait. The reconciliation is the **command** (Elm/Bubble Tea call it `Cmd`; Textual uses `run_worker` / `call_later`; Ink uses `useEffect`; ratatui apps usually use an `Action`/`Effect` enum dispatched to a runtime).

A command is **data that *describes* an effect** — not the effect itself. `update` returns "go fetch URL X," not the bytes. Something outside the loop performs it and feeds the **result back in as a new message**.

```
update sees   KeyPressed(Enter)
        │
        ▼
returns (state{ in_flight: true }, Some(Cmd::Fetch(url)))
        │                                   │
   loop draws spinner immediately      runtime runs fetch
        │                              off the event loop
        │                                   │
        │                              completes
        │                                   ▼
        └──────────── message ──── Msg::FetchDone(Ok(data))
                                             │
                                             ▼
                                update sets state{ in_flight:false, data }
```

The loop **never waits** for the fetch. It returned, drew the spinner, and went back to reading input. When the result is ready it arrives as just another message in the same queue as keypresses — and gets handled in order, deterministically.

This is also where **cancellation** lives cleanly: pressing `Esc` produces `Msg::Cancel`, `update` flips `in_flight: false` and returns `Cmd::AbortFetch`; a late `FetchDone` for a superseded request is ignored by checking a request id in the model. (The *feel* of in-flight feedback and cancel UX is `feedback-latency-and-async-work.md`'s job; the *plumbing* is here.)

---

## Never Block: Push I/O Off the Loop

The cardinal sin (RED #16): **a synchronous, slow call inside the read-update-draw loop.** Every millisecond `update` spends in a blocking `read()` is a millisecond the loop cannot read input or redraw. A 2-second sync HTTP call = a 2-second frozen UI.

**Diagnostic — is it on the loop?** If a single user keystroke can directly trigger a function that does network/disk/subprocess/long-CPU work *before control returns to the loop*, it is on the loop. Fix it.

The off-loop mechanism depends on the runtime, but the shape is identical everywhere:

| Framework | Off-loop mechanism | Result delivered as |
|---|---|---|
| Bubble Tea (Go) | return a `tea.Cmd` (runs in a goroutine) | a `tea.Msg` on the program's channel |
| ratatui (Rust) | `tokio::spawn` or `std::thread`, send over an `mpsc` channel | an `Action`/event read by the loop |
| Textual (Python) | `@work` / `run_worker` (async or thread worker) | a posted message / direct state set on UI thread |
| Ink (JS) | `useEffect` + `async`/`await`, `setState` on resolve | a React re-render |
| notcurses (C/Rust) | a worker thread + a pipe/channel the poll loop watches | an event the input poll selects on |

Rules that hold across all of them:

1. **`update` does no blocking I/O.** Ever. It computes next state and *describes* commands.
2. **CPU-bound work also counts.** A synchronous sort of 5M rows, a regex over a 200MB buffer, JSON-parsing a huge payload — these freeze the loop exactly like network does. Move heavy CPU off-loop too (a thread / worker pool / `spawn_blocking`).
3. **Results re-enter as messages**, never as direct mutation from the worker. A worker thread writing into shared mutable model state from another thread is how you get races *and* lose the single-ordering guarantee that makes MVU debuggable.
4. **Keep one input source non-blocking or interruptible.** Poll input with a timeout (e.g. crossterm `event::poll(Duration)`) or select across input + result channels so the loop wakes for *either*. Never `read()` with no timeout while also expecting background results.

---

## List-State Discipline: Identity, Stick-to-Bottom, and Bounded Buffers

Three closely-related state bugs sink "list of things that changes over time" TUIs — pickers, log viewers, process monitors, chat. They live in the model, so they belong here.

### 1. Anchor selection to identity, not absolute index (RED #17)

If the selected row is stored as `selected_index: 5` and the user types a filter that drops the list from 200 items to 12, index 5 now points at a *different item* — or is out of bounds. The selection silently teleports.

**Fix:** store the selected item's stable **identity** (an id, a key, a path), and derive the visible index from it at render time.

```rust
struct Model {
    all: Vec<Item>,           // full set, stable order
    filter: String,
    selected_id: Option<ItemId>,  // identity, NOT an index
    scroll_top_id: Option<ItemId>,
}

// view derives the index from identity against the *filtered* view:
fn visible(model: &Model) -> Vec<&Item> {
    model.all.iter().filter(|i| i.matches(&model.filter)).collect()
}
fn selected_row(model: &Model) -> Option<usize> {
    let vis = visible(model);
    model.selected_id
        .and_then(|id| vis.iter().position(|i| i.id == id))
}
```

When the filter changes, `update` keeps `selected_id` if it still matches; otherwise it falls back to the first visible item (a deliberate, visible rule) rather than landing on a random index. The user's *choice* survives the list reshaping under it.

### 2. Stick-to-bottom must be a state flag, not a guess (RED #18)

A tailing view (logs, chat) wants to auto-scroll to newest. But the moment the user scrolls up to read history, auto-scroll fighting them is infuriating — the view yanks back to the bottom every time a line arrives.

**Fix:** model an explicit `follow: bool`. New data only auto-scrolls **when `follow == true`**. Scrolling up sets `follow = false`; scrolling back to (or pressing End at) the bottom sets `follow = true`. Show the state ("FOLLOWING" / "PAUSED — End to resume") so it's never a mystery.

```go
type model struct {
    lines  *ringBuffer
    offset int   // top visible line
    follow bool  // stick-to-bottom?
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case newLineMsg:
        m.lines.push(msg.line)
        if m.follow {
            m.offset = m.lines.len() - m.viewHeight // pin to bottom
        }
    case tea.KeyMsg:
        switch msg.String() {
        case "up", "k":
            m.offset--
            m.follow = false              // user took control
        case "end", "G":
            m.offset = m.lines.len() - m.viewHeight
            m.follow = true               // re-engage tailing
        }
    }
    return m, nil
}
```

### 3. Bound the buffer with a ring (RED #19)

Appending every log line / output chunk to an unbounded list is a memory leak with extra steps. A long-lived monitor will consume gigabytes and then die.

**Fix:** a fixed-capacity **ring buffer** (cap by line count or bytes). Old lines drop off the back. Make the cap configurable and *visible* if data is being discarded ("showing last 50,000 lines"). This is a *state* decision — the model owns the cap — and it pairs with viewport virtualization (render only visible rows; see `rendering-and-redraw-discipline.md`) so a bounded buffer also renders cheaply.

```python
from collections import deque
# O(1) push, automatic eviction, bounded memory:
self.lines: deque[str] = deque(maxlen=50_000)
```

---

## A Minimal MVU Example (Bubble Tea, Go)

A complete, runnable spine: a counter that also kicks off an off-loop fetch and never blocks. Note how `Update` only transforms state and returns commands, `View` is pure, and the slow work runs in a `Cmd`.

```go
package main

import (
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// ---- MODEL: all state, plain data ----
type model struct {
	count    int
	loading  bool
	result   string
	reqID    int // guards against stale results
}

// ---- MESSAGES: the only way state changes ----
type fetchDoneMsg struct {
	reqID int
	data  string
}

// ---- COMMAND: describes off-loop work; runtime runs it in a goroutine ----
func fetchCmd(id int) tea.Cmd {
	return func() tea.Msg {
		time.Sleep(2 * time.Second) // stand-in for a slow network call
		return fetchDoneMsg{reqID: id, data: "payload fetched"}
	}
}

func (m model) Init() tea.Cmd { return nil }

// ---- UPDATE: pure-ish; transforms state, returns commands. NO blocking I/O. ----
func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			return m, tea.Quit
		case "+":
			m.count++
		case "f": // fire the fetch — UI stays live during the 2s
			m.loading = true
			m.reqID++
			return m, fetchCmd(m.reqID)
		}
	case fetchDoneMsg:
		if msg.reqID == m.reqID { // ignore superseded results
			m.loading = false
			m.result = msg.data
		}
	}
	return m, nil
}

// ---- VIEW: pure function of state. No mutation, no I/O. ----
func (m model) View() string {
	status := "idle"
	if m.loading {
		status = "loading… (press + or q — still responsive)"
	} else if m.result != "" {
		status = "result: " + m.result
	}
	return fmt.Sprintf("count: %d\nstatus: %s\n[+] inc  [f] fetch  [q] quit\n", m.count, status)
}

func main() {
	tea.NewProgram(model{}).Run()
}
```

While the 2-second fetch runs, `+` still increments and `q` still quits — because the fetch is a `Cmd` on a goroutine, not a call inside `Update`.

## The Same Spine in ratatui (Rust) with an Off-Loop Channel

ratatui has no built-in runtime, so you wire the spine by hand — which makes the off-loop discipline explicit. The loop polls input *with a timeout* and drains a results channel, so it never blocks on either source.

```rust
use std::sync::mpsc;
use std::time::Duration;
use crossterm::event::{self, Event, KeyCode};

// ---- MODEL ----
struct Model {
    count: i64,
    loading: bool,
    result: Option<String>,
    req_id: u64,
    quit: bool,
}

// ---- MESSAGES ----
enum Msg {
    Key(KeyCode),
    FetchDone { req_id: u64, data: String },
}

// ---- UPDATE: returns an optional command (here: a closure to run off-loop) ----
fn update(m: &mut Model, msg: Msg, tx: &mpsc::Sender<Msg>) {
    match msg {
        Msg::Key(KeyCode::Char('q')) => m.quit = true,
        Msg::Key(KeyCode::Char('+')) => m.count += 1,
        Msg::Key(KeyCode::Char('f')) => {
            m.loading = true;
            m.req_id += 1;
            let id = m.req_id;
            let tx = tx.clone();
            // EFFECT: off the loop. The render/input thread is untouched.
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(2)); // slow work
                let _ = tx.send(Msg::FetchDone { req_id: id, data: "payload".into() });
            });
        }
        Msg::FetchDone { req_id, data } if req_id == m.req_id => {
            m.loading = false;
            m.result = Some(data);
        }
        _ => {}
    }
}

// ---- VIEW: pure function of &Model into a string (real code draws widgets) ----
fn view(m: &Model) -> String {
    let status = if m.loading {
        "loading… (+ and q still work)".to_string()
    } else {
        m.result.clone().unwrap_or_else(|| "idle".into())
    };
    format!("count: {}  status: {}\n[+] inc  [f] fetch  [q] quit", m.count, status)
}

fn run(mut m: Model) -> std::io::Result<()> {
    let (tx, rx) = mpsc::channel::<Msg>();
    loop {
        // 1. DRAW — pure render of current state (terminal.draw(... view(&m) ...))
        println!("{}", view(&m));

        // 2. READ INPUT, non-blocking via a poll timeout — loop never stalls.
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(k) = event::read()? {
                update(&mut m, Msg::Key(k.code), &tx);
            }
        }
        // 3. DRAIN off-loop results that arrived since last tick.
        while let Ok(msg) = rx.try_recv() {
            update(&mut m, msg, &tx);
        }
        if m.quit { break; }
    }
    Ok(())
}
```

The two sources — input and the results channel — are both **non-blocking** (`poll` with a timeout, `try_recv`). The loop wakes ~20×/sec, services whatever is ready, redraws, and goes around again. No keystroke ever waits on a fetch; no fetch ever waits on a keystroke.

> Same architecture, three idioms: **Textual** would make `Model` reactive attributes, do the slow work in an `@work(thread=True)` worker, and post a message back to set state on the UI thread; **Ink** would hold state in `useReducer`, run the fetch in a `useEffect`, and `dispatch` on resolve. The names change; *state-is-data, view-is-pure, effects-off-loop* does not.

---

## Common Mistakes

- **Blocking call inside `update`/the loop (RED #16).** `let data = http::get(url)?;` directly in a key handler. The whole UI freezes for the duration. → Return a command; deliver the result as a message.
- **CPU-bound work treated as "not I/O, so it's fine."** A synchronous parse/sort/regex over a huge buffer freezes the loop just as hard as a network call. → Move heavy CPU off-loop (thread / worker / `spawn_blocking`) too.
- **View mutates state.** Adjusting scroll offset, lazily loading data, or toggling flags inside the render function. The screen flickers between truths and is impossible to test. → Render reads only; all mutation goes through `update` via messages.
- **State held outside the model.** A `selected` variable in a widget, a `loading` boolean in a callback closure, "current page" tracked in two places. They drift. → One model owns all state; there is one source of truth.
- **Worker threads mutating shared state directly.** Background task reaches into the model from another thread. Races, plus you lose the single deterministic ordering. → Workers *send messages*; only the loop applies them.
- **Selection/scroll stored as absolute index (RED #17).** Breaks the instant the underlying list is filtered or reordered. → Anchor to item identity; derive the index at render.
- **Auto-scroll with no `follow` flag (RED #18).** Tailing yanks the view to the bottom while the user reads history. → Explicit `follow` state; scrolling up disengages, End re-engages; show which mode is active.
- **Unbounded append buffer (RED #19).** Every line `push`ed forever → memory climbs until OOM. → Fixed-capacity ring buffer (cap by lines or bytes), and say so on screen if data is being dropped.
- **No request-id guard on async results.** A stale fetch lands after a newer one and overwrites fresh data. → Tag each request with an id; `update` ignores results whose id ≠ current.
- **Blocking input read with no timeout while expecting background events.** `event::read()` with no `poll` means the loop sleeps until a key arrives and never notices the result that just landed. → Poll with a timeout, or `select`/drain across input + result channels.
- **One giant `Msg` handled with deeply nested mutation.** As the app grows, `update` becomes unreadable. → Compose: split state into sub-models with their own `update`, route messages to them, keep each handler small.

## Related Sheets

- `lifecycle-signals-and-terminal-restoration.md` — keeping the loop's exit path clean: SIGINT/SIGTERM/SIGWINCH, raw-mode + alt-screen restoration on panic/quit.
- `feedback-latency-and-async-work.md` — once work is off the loop (this sheet), how to *show* it: spinners, progress, debounced filtering, empty/loading/no-match states, cancel UX.
- `rendering-and-redraw-discipline.md` — the View half: damage tracking, viewport virtualization, avoiding full-repaint flicker. A pure `view(state)` is what makes diffing possible.
- `input-keyboard-mouse-and-focus.md` — turning raw input into the messages this loop consumes, and the focus model that routes a keystroke to the right component.
- `testing-tuis.md` — because `view` is pure and state changes only via messages, you can snapshot-test frames and drive `update` with synthetic message sequences.
