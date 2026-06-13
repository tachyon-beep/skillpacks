---
name: feedback-latency-and-async-work
description: Use when a TUI freezes during a network call, file scan, subprocess, or query; when a long operation runs with no spinner, progress bar, or "working..." state and the screen looks hung; when a filter or search re-runs synchronously on every keystroke and the input lags; when an operation cannot be cancelled (Esc/Ctrl-C does nothing while it runs); when there is no loading / empty / no-match state so an idle screen is indistinguishable from a crash; or when blocking work lives inside the input/render loop instead of off-thread/async — covers perceived performance, spinners vs determinate progress, optimistic feedback, off-loop work streamed back as messages, cancellation, and debouncing across ratatui, Textual, Bubble Tea, Ink, and notcurses.
---

# Feedback, Latency, and Async Work

## The UX stake: perceived performance is the product

A terminal user cannot see a process list. When your TUI calls the network, walks a million files, or shells out to `git log`, the only thing the user can observe is the **screen**. If the screen stops responding, you have not "started a slow operation" — from the user's seat you have **crashed**. They will hammer Ctrl-C, kill the terminal, and lose their work.

This is the failure that separates a toy from a tool:

> **A frozen frame is a bug report being written in the user's head.**

Perceived performance is not the same as actual performance. A 4-second operation with a live progress bar feels *faster and safer* than a 1.5-second operation that locks the keyboard. The research-backed thresholds you are designing against:

| Latency | What the user perceives | What you must show |
|---|---|---|
| **< 100 ms** | Instant. Direct manipulation. | Nothing extra — just the result. |
| **100 ms – 1 s** | A noticeable beat. Still "the system did it." | Keep the keyboard alive; a subtle hint is enough. |
| **1 s – 10 s** | "Is it working?" Attention drifts. | **Spinner or progress, plus a label of what's happening, plus a way to cancel.** |
| **> 10 s** | "Is it broken?" Trust collapses. | **Determinate progress (or staged status), ETA if possible, and prominent cancel.** |

Every guideline below exists to keep you on the right side of those rows. The non-negotiable rule underneath all of them:

> **Never do blocking work on the thread that reads input and paints the screen.** (RED #13, #16)

The event loop must keep spinning — reading keys, handling resize, repainting — *while* the slow thing happens somewhere else. This sheet is the async/feedback half; the architectural half (how the loop, the worker channel, and state fit together) lives in **event-loop-and-state-architecture.md**. Read them together.

---

## The core architecture: work off the loop, results back as messages

The single pattern that closes the freeze (RED #13, #16) is:

1. The event loop owns the screen and the keyboard, and **never blocks**.
2. Slow work runs **somewhere else** — a thread, a goroutine, an `async` task, a worker.
3. The worker does not touch the UI directly. It **streams results back as messages** into the same queue the loop already drains for key/resize/tick events.
4. The loop applies each message to state, marks the affected region dirty, and lets normal redraw discipline repaint (see **rendering-and-redraw-discipline.md**).

```
                    ┌─────────────────────────────────────┐
   key / resize ───▶│            EVENT LOOP                │───▶ render (diffed)
   tick (spinner) ─▶│  drains one queue, never blocks      │
        ▲           └───────────────┬─────────────────────┘
        │                           │ spawn(job, cancel_token)
        │ Msg::Progress(n)          ▼
        │ Msg::Chunk(line)   ┌──────────────┐
        └────────────────────│   WORKER     │  blocking I/O lives HERE
          Msg::Done(result)  │ thread/async │
          Msg::Failed(err)   └──────────────┘
```

Why messages and not shared mutable state with locks? Because the loop is single-consumer: it already has exactly one place where it decides "what changed, repaint it." Funneling worker output through that same place means your spinner tick, your incoming data, and a window resize are all handled by one consistent update path. No torn reads, no "who repaints?" ambiguity, no lock held across a render.

> **Design rule:** the worker's vocabulary is `Started / Progress / Chunk / Done / Failed / Cancelled`. The loop's job is to translate those into state transitions. If you find a worker calling `draw()`, you have a bug waiting to flicker.

### What state each in-flight job needs

Model the operation as an explicit state machine, not a boolean `is_loading`:

```
Idle ──start──▶ Running{started_at, progress?, cancel} ──▶ Done{result}
                      │                                  ╲
                   cancel                              Failed{err}
                      ▼
                  Cancelling ──▶ Cancelled
```

`Cancelling` is a real state, not a fiction — the worker may take a moment to notice the cancel token. Showing "Cancelling…" instead of snapping straight back to idle is honest feedback (RED #14).

---

## Spinners vs determinate progress: pick by what you know

The most common feedback mistake is reaching for a spinner when you actually know the denominator — or showing a progress bar that lies because you *don't*.

| Use a **spinner / indeterminate** when… | Use **determinate progress** when… |
|---|---|
| You don't know how long it'll take (a network request, a query) | You know the total (files to scan, bytes to download, rows to import) |
| The work is a single opaque step | The work is countable units |
| Duration is short and roughly bounded (1–5 s) | Duration is long (>10 s) and the user needs reassurance it's advancing |

**Rules for spinners:**
- A spinner must **animate from a timer tick, not from data arriving.** If your spinner only advances when a chunk comes in, a stall makes it freeze — exactly when the user most needs to see life. Drive it from a ~80–125 ms tick message in the loop.
- Always pair a spinner with a **label**: `⠋ Fetching results…`, not a bare `⠋`. The label is what tells the user *what* is slow and reassures them it's expected.
- Don't show a spinner for sub-100 ms work — the flash is visual noise. **Delay-show** it: only reveal the spinner if the operation hasn't finished within ~150 ms (otherwise fast results "blink" the spinner annoyingly).

**Rules for determinate progress:**
- Show the fraction, not just the bar: `[████████░░░░░░] 412 / 1,024 files`.
- If you can, add a rate/ETA once you have a stable sample: `… 412/1024 · ~3s left`. Never show an ETA you computed from a single data point — it'll jitter wildly and look broken.
- **Never let the bar go backwards** unless the total genuinely changed; a regressing bar reads as a bug.
- If the total is unknown until partway through, start with a spinner and **upgrade** to a bar once you learn the denominator. Don't fake a denominator.

**Glyph note:** spinner braille (`⠋⠙⠹…`) and block elements (`█░`) are not universal. Probe for UTF-8 / capability and fall back to ASCII (`|/-\` spinner, `#`/`-` bar) when the terminal can't render them — see **terminal-substrate-and-constraints.md** and **color-theming-and-monospace-canvas.md**. Double-width glyph handling matters here too: a bar built from wide glyphs will smear your layout.

---

## Optimistic feedback: acknowledge before you confirm

When an action is *very likely* to succeed and reversible, reflect it in the UI **immediately**, then reconcile when the real result lands. This is what makes a TUI feel instant even over a slow link.

- User toggles a checkbox / stars an item / renames a row → flip the visual state **now**, fire the request in the background, and roll back with a visible error marker if it fails.
- User submits a chat message → append it to the transcript greyed/"sending…" instantly, then solidify it (or mark it failed with a retry affordance) on the response.

Two guardrails so optimism doesn't become lying:
1. **Mark the provisional state visibly** (dimmed, a small `…`, an italic note) so the user knows it's not yet confirmed — and so a CVD/monochrome user isn't relying on color alone (cross-link **accessibility-in-the-terminal.md**).
2. **Always have a rollback path with a real error message.** Optimistic UI that silently swallows failures is worse than a spinner — the user believes a thing happened that didn't.

Don't be optimistic about destructive or irreversible operations (deletes, payments, anything you can't take back). There, confirm first, show real progress, and reflect the *actual* outcome.

---

## Cancellation: every long operation must be abortable

If the user can start it, the user must be able to **stop** it. An uncancellable long operation (RED #14) is a trap: the user picked the wrong file, fat-fingered a query, or just changed their mind, and now they're hostage to your I/O.

Design contract:
- **Esc cancels the current operation** in most TUIs; **Ctrl-C** should also abort the in-flight job *before* it's treated as "quit the app." (Be careful: Ctrl-C is also the SIGINT/quit path — coordinate with **lifecycle-signals-and-terminal-restoration.md** so a single Ctrl-C cancels the job and a second one exits, or so cancel and quit are clearly distinguished.)
- Cancellation must be **cooperative and prompt**: the worker checks a cancel token / context between units of work and bails out. A cancel that takes 30 seconds to honor isn't cancellation.
- Show the transition: `Running → Cancelling… → Cancelled`. Don't pretend it stopped instantly if it didn't.
- **Clean up partial work** on cancel — close sockets, delete half-written temp files, release the connection. A cancel that leaks resources is a different bug for tomorrow.
- Make cancel **discoverable**: while the operation runs, show the hint (`Esc cancel`) in the status line (cross-link **affordances-and-discoverability.md**).

---

## Debouncing and throttling: don't re-run on every keystroke

A live filter / search box that re-runs the query **synchronously on each keypress** (RED #15) is the canonical lag source: the user types "error", and the UI runs five filters — over a large or remote dataset — before they finish the word, stuttering the whole time.

Two tools:
- **Debounce** — wait until input *stops* for N ms, then run once. Best for "filter as you type": run the search ~150–250 ms after the last keystroke. Fast typists trigger exactly one query.
- **Throttle** — run at most once per N ms regardless of input rate. Best for continuous streams (a resize drag, a scroll-driven recompute) where you want periodic updates, not one-at-the-end.

Combine with off-loop execution: debounce decides *when* to run; the worker pattern decides *where*. The keystroke updates the filter string and repaints the input box **instantly** (so typing always feels responsive); the actual filtering runs debounced and off the loop, streaming matches back as messages. Stale results from a superseded query must be **dropped** — tag each query with a generation/sequence id and ignore responses whose id is no longer current, or you'll paint results for "err" after the user has typed "error".

> Cross-link: when the filtered set changes, scroll/selection must survive by **identity, not absolute index** — see **event-loop-and-state-architecture.md** (RED #17).

---

## Show the empty, loading, and no-match states — or the app looks dead

A blank region with no explanation is indistinguishable from a freeze (RED #20). Every data surface needs the full set of states, not just "has data":

- **Loading** — `⠙ Searching…` (with cancel hint). The very first paint of a slow surface should be the loading state, never an empty void.
- **Empty (nothing yet)** — `No results. Type to search, or press / to filter.` Tell them what to do.
- **No match (searched, found nothing)** — `No matches for "xyzzy". Esc to clear.` Distinct from "empty" — the user did something and it returned nothing; say so.
- **Error** — a real message with a recovery action (`Failed to fetch: connection refused — r to retry`), never color-only red (cross-link **color-theming-and-monospace-canvas.md**, **accessibility-in-the-terminal.md**).

These states are cheap to build and are the difference between "the tool is thinking" and "the tool is broken."

---

## Worked example 1 — Bubble Tea (Go): off-loop fetch, spinner, cancel, debounced filter

Bubble Tea's `Cmd`/`Msg` model is the off-loop pattern made literal: `Update` never blocks, slow work returns a `tea.Cmd` that runs in a goroutine and yields a `Msg`.

```go
package main

import (
	"context"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
)

// ---- messages streamed back from off-loop work ----
type resultsMsg struct{ gen int; items []string }
type errMsg struct{ gen int; err error }
type debounceMsg struct{ gen int } // fires after the user stops typing

type model struct {
	query    string
	gen      int        // generation id: drops stale results (RED #15)
	loading  bool
	items    []string
	err      error
	spin     spinner.Model
	cancel   context.CancelFunc
}

func initialModel() model {
	s := spinner.New()
	s.Spinner = spinner.Dot
	return model{spin: s}
}

func (m model) Init() tea.Cmd { return m.spin.Tick }

// debounce: schedule a run 200ms after the LAST keystroke, tagged with gen
func debounce(gen int) tea.Cmd {
	return tea.Tick(200*time.Millisecond, func(time.Time) tea.Msg {
		return debounceMsg{gen: gen}
	})
}

// the actual slow work — runs in a goroutine, honors cancellation
func search(ctx context.Context, gen int, q string) tea.Cmd {
	return func() tea.Msg {
		items, err := fetchFromNetwork(ctx, q) // blocking I/O, but NOT on the loop
		if err != nil {
			if ctx.Err() != nil { // cancelled: drop silently
				return nil
			}
			return errMsg{gen: gen, err: err}
		}
		return resultsMsg{gen: gen, items: items}
	}
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "esc":
			if m.loading && m.cancel != nil { // Esc cancels the in-flight job (RED #14)
				m.cancel()
				m.loading = false
				return m, nil
			}
			return m, tea.Quit
		case "backspace":
			if len(m.query) > 0 {
				m.query = m.query[:len(m.query)-1]
			}
		default:
			if len(msg.Runes) == 1 {
				m.query += string(msg.Runes)
			}
		}
		// input box repaints INSTANTLY; the search is debounced + off-loop
		m.gen++
		return m, debounce(m.gen)

	case debounceMsg:
		if msg.gen != m.gen { // superseded by newer keystroke — ignore
			return m, nil
		}
		if m.cancel != nil {
			m.cancel() // abort previous in-flight search
		}
		ctx, cancel := context.WithCancel(context.Background())
		m.cancel, m.loading, m.err = cancel, true, nil
		return m, search(ctx, m.gen, m.query)

	case resultsMsg:
		if msg.gen != m.gen { // stale result for an old query — drop it
			return m, nil
		}
		m.loading, m.items = false, msg.items
		return m, nil

	case errMsg:
		if msg.gen != m.gen {
			return m, nil
		}
		m.loading, m.err = false, msg.err
		return m, nil

	case spinner.TickMsg: // spinner animates from the TICK, not from data (RED #14)
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd
	}
	return m, nil
}

func (m model) View() string {
	header := "Search: " + m.query + "\n\n"
	switch {
	case m.loading:
		return header + m.spin.View() + " Searching…  (Esc to cancel)\n" // loading state
	case m.err != nil:
		return header + "Failed: " + m.err.Error() + "  (type to retry)\n" // not color-only
	case m.query != "" && len(m.items) == 0:
		return header + "No matches for \"" + m.query + "\".\n" // no-match ≠ empty (RED #20)
	case len(m.items) == 0:
		return header + "Type to search.\n" // empty state
	}
	out := header
	for _, it := range m.items {
		out += "  " + it + "\n"
	}
	return out
}
```

What this closes: the loop never blocks (RED #13/#16); the spinner is timer-driven so a stalled network still shows life (RED #14); Esc cancels (RED #14); the filter is debounced and stale results are dropped by generation id (RED #15); loading/empty/no-match/error are all distinct (RED #20).

---

## Worked example 2 — ratatui (Rust): worker thread, determinate progress, cancel token

ratatui is immediate-mode and gives you no built-in async, so *you* own the channel and the tick. The loop drains one `mpsc` receiver carrying both terminal events and worker messages, and uses a small `poll` timeout so the spinner/progress repaints even when nothing is happening.

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

enum Msg {
    Progress { done: usize, total: usize }, // determinate: we know the denominator
    Done(Vec<String>),
    Failed(String),
}

enum JobState {
    Idle,
    Running { started: Instant, done: usize, total: usize, cancel: Arc<AtomicBool> },
    Cancelling,
    Finished(Vec<String>),
    Failed(String),
}

fn spawn_scan(tx: mpsc::Sender<Msg>, total: usize) -> Arc<AtomicBool> {
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    thread::spawn(move || {
        let mut acc = Vec::new();
        for i in 0..total {
            if flag.load(Ordering::Relaxed) {
                return; // cooperative cancel: bail promptly, drop partial work (RED #14)
            }
            // ... do one unit of real (blocking) work off the UI thread ...
            acc.push(format!("item {i}"));
            // stream progress back as a message — NEVER draw from here
            if tx.send(Msg::Progress { done: i + 1, total }).is_err() {
                return; // loop is gone; stop
            }
            thread::sleep(Duration::from_millis(3));
        }
        let _ = tx.send(Msg::Done(acc));
    });
    cancel
}

fn run(/* terminal, key_rx, ... */) {
    let (tx, rx) = mpsc::channel::<Msg>();
    let mut job = JobState::Idle;
    let mut spinner_frame = 0usize;

    loop {
        // 1) drain worker messages without blocking
        while let Ok(msg) = rx.try_recv() {
            job = match (msg, std::mem::replace(&mut job, JobState::Idle)) {
                (Msg::Progress { done, total }, JobState::Running { started, cancel, .. }) =>
                    JobState::Running { started, done, total, cancel },
                (Msg::Done(items), _) => JobState::Finished(items),
                (Msg::Failed(e), _) => JobState::Failed(e),
                (_, other) => other, // ignore progress arriving after cancel/finish
            };
        }

        // 2) handle input WITHOUT blocking — poll with a short timeout so the
        //    spinner/progress keeps animating even with no keypress (RED #14)
        if let Some(key) = poll_key(Duration::from_millis(100)) {
            match key {
                Key::Char('s') if matches!(job, JobState::Idle | JobState::Finished(_)) => {
                    let cancel = spawn_scan(tx.clone(), 1024);
                    job = JobState::Running { started: Instant::now(), done: 0, total: 1024, cancel };
                }
                Key::Esc | Key::CtrlC => {
                    if let JobState::Running { cancel, .. } = &job {
                        cancel.store(true, Ordering::Relaxed); // cooperative cancel
                        job = JobState::Cancelling;            // honest intermediate state
                    } else {
                        break; // nothing running → Esc/Ctrl-C quits (see lifecycle sheet)
                    }
                }
                _ => {}
            }
        }

        spinner_frame = spinner_frame.wrapping_add(1);
        draw(&job, spinner_frame); // diffed redraw; ratatui only repaints changed cells
    }
}

fn draw(job: &JobState, frame: usize) {
    const SPIN: [&str; 4] = ["⠋", "⠙", "⠹", "⠸"]; // fall back to |/-\ if no UTF-8
    match job {
        JobState::Idle => render_line("Press 's' to scan."),
        JobState::Running { started, done, total, .. } => {
            let pct = (*done as f64 / *total as f64 * 100.0) as u16;
            let bars = (pct / 5) as usize; // 20-cell bar
            let bar = format!("[{}{}]", "█".repeat(bars), "░".repeat(20 - bars));
            // determinate progress with count + ETA once we have a stable sample
            let eta = if *done > 0 {
                let per = started.elapsed().as_secs_f64() / *done as f64;
                format!(" · ~{:.0}s left", per * (*total - *done) as f64)
            } else { String::new() };
            render_line(&format!("{} {bar} {done}/{total}{eta}   (Esc cancel)", SPIN[frame % 4]));
        }
        JobState::Cancelling => render_line("Cancelling…"),         // RED #14: don't fake instant stop
        JobState::Finished(items) => render_line(&format!("Done: {} items", items.len())),
        JobState::Failed(e) => render_line(&format!("Failed: {e}  (s to retry)")), // not color-only
    }
}
```

What this closes: blocking I/O lives on a worker thread, the loop polls with a timeout and never blocks (RED #13/#16); progress is **determinate** with a count and a sanely-derived ETA; the spinner advances from the loop frame, not from data; Esc/Ctrl-C sets a cooperative cancel token and the UI shows `Cancelling…` then resolves (RED #14); progress arriving after cancel/finish is ignored. Redraw stays diffed via ratatui (cross-link **rendering-and-redraw-discipline.md**).

---

## Common mistakes

- **Blocking call inside the input/render loop.** `requests.get(...)` / `cmd.Output()` / a synchronous DB query right where you handle keys. The whole UI freezes — keyboard dead, no resize, no repaint. (RED #13, #16) → run it off-loop and stream results back as messages.
- **Spinner driven by data, not by a timer.** The animation only moves when a chunk arrives, so a stall — the exact moment the user needs reassurance — looks like a hang. → tick the spinner from the loop on a fixed interval.
- **No delay-show on the spinner.** Flashing a spinner for 40 ms of work is visual noise that makes fast operations feel *less* polished. → only reveal it after ~150 ms.
- **Fake or backwards progress bars.** Inventing a denominator you don't have, or letting the bar regress, reads as a bug. → use a spinner until you genuinely know the total, then upgrade; never go backwards.
- **No cancellation.** The user is held hostage by a slow or mistaken operation. (RED #14) → Esc/Ctrl-C aborts via a cooperative token; show `Cancelling…`; clean up partial work.
- **Cancel that leaks.** Aborting without closing sockets / deleting temp files / releasing the connection. → wire cleanup into the cancel path.
- **Synchronous per-keystroke filtering.** Every keypress re-runs the query, stuttering on large/remote data. (RED #15) → debounce (~150–250 ms) and run off-loop.
- **Painting stale results.** Showing matches for "err" after the user typed "error" because an old query finished late. → tag queries with a generation id and drop superseded responses.
- **No loading/empty/no-match/error states.** A blank region is indistinguishable from a crash. (RED #20) → render all four explicitly, with distinct copy and a recovery action.
- **Optimistic UI with no rollback.** Reflecting an action immediately, then silently swallowing the failure, so the user believes something happened that didn't. → mark provisional state visibly and always reconcile failures with a real message.
- **Color-only progress/error.** A red bar or red "failed" with no glyph/text excludes CVD, monochrome, and screen-reader users. (cross-link **accessibility-in-the-terminal.md**, **color-theming-and-monospace-canvas.md**) → pair color with a symbol and words.
- **UTF-8 spinner/bar glyphs with no fallback.** Braille and block elements render as `?`/boxes or smear (double-width) on terminals that can't handle them. → probe capability and fall back to ASCII (cross-link **terminal-substrate-and-constraints.md**).
- **Ctrl-C means quit *and* cancel with no coordination.** A single Ctrl-C either kills the app mid-job or does nothing. → define the rule (first cancels, second quits) jointly with **lifecycle-signals-and-terminal-restoration.md**.

## Cross-links

- **event-loop-and-state-architecture.md** — how the loop, worker channel, and state machine fit together; off-thread/async work coupling (RED #16); identity-anchored selection across filtered sets (RED #17); ring-buffered streams (RED #19).
- **rendering-and-redraw-discipline.md** — apply worker messages as dirty regions; don't full-repaint on every tick (RED #3, #4).
- **lifecycle-signals-and-terminal-restoration.md** — Ctrl-C/SIGINT coordination between cancel and quit; clean exit paths (RED #1, #2).
- **color-theming-and-monospace-canvas.md** — non-color-only error/progress; NO_COLOR; double-width glyph safety (RED #6, #7, #8).
- **accessibility-in-the-terminal.md** — non-visual status path for spinners/progress/errors (RED #25, #26).
- **affordances-and-discoverability.md** — surfacing the `Esc cancel` hint (RED #23, #24).
- **terminal-substrate-and-constraints.md** — capability probe before using braille/block glyphs (RED #27, #28).
