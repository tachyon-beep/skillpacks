---
name: testing-tuis
description: Use when a TUI's rendering, layout, input handling, or terminal restoration is verified only by hand ("I ran it and it looked right"), when a layout or color change shipped and silently broke a panel at 80 columns or under SIGWINCH, when there is no test that asserts what the screen actually shows, when reviewers say "TUIs can't really be tested" and skip coverage, when a resize/min-size regression slipped through, when input routing (filter vs list, focus, keybindings) has no automated check, or when you need golden-frame / snapshot / headless / in-memory-backend / CI testing for ratatui, Textual, Bubble Tea, Ink, or notcurses.
---

# Testing TUIs

A TUI's output is a 2D grid of cells — characters, foreground/background colors, attributes. That grid is **data**. Data can be asserted on. The belief that terminal UIs are inherently untestable is the single biggest reason TUIs ship with broken layouts, smeared CJK, color-only errors, and panics that leave the user's terminal in raw mode. Every one of those is a regression an automated test would have caught.

The UX stake is direct: an untested TUI regresses *visually* and *behaviorally* between releases, and the only detector is a human who happens to resize their terminal to the wrong width on the wrong day. "Looked fine on my machine" is not QA — it is a single sample from a space of (terminal size × color depth × TTY-or-pipe × emulator × locale) that has thousands of points. Tests sample that space deterministically so users do not sample it by suffering.

This sheet shows how to test the three things hand-QA reliably misses: **what the screen renders** (golden frames), **how input is handled** (headless event injection), and **that the layout survives sizes and capability levels** (parametrized backends). It is a discipline sheet: the rules are not optional, and the rationalizations for skipping them are predictable and answered below.

Cross-references: this sheet's tests *prove* the behaviors that `rendering-and-redraw-discipline`, `layout-and-responsive-composition`, `lifecycle-signals-and-terminal-restoration`, and `accessibility-in-the-terminal` define. A snapshot is only as good as the property it encodes — write golden frames that assert the *non-color* error signal, the truncation policy, the focus indicator, not just "the bytes match."

---

## The core move: render to a buffer, not a terminal

Every mature TUI framework can render a frame into an **in-memory backend** instead of a real PTY. You never need a terminal to test a TUI. You need the framework's test backend and an assertion on the resulting cell buffer.

| Framework | In-memory backend / test harness | What you assert on |
|---|---|---|
| ratatui (Rust) | `TestBackend` + `Terminal::backend()` → `Buffer` | `Buffer` cells (char, fg/bg, modifiers); `buffer.area`; `assert_buffer_eq!` |
| Textual (Python) | `App.run_test()` → `Pilot`; `app.export_screen_text()` / SVG snapshot | rendered text, widget queries, `pytest-textual-snapshot` SVG goldens |
| Bubble Tea (Go) | `teatest` (`x/exp/teatest`) with `tea.WithoutRenderer` / golden output | `teatest.WaitFor`, `teatest.RequireEqualOutput` against `.golden` files |
| Ink (JS/TS) | `ink-testing-library` `render()` → `lastFrame()` | `lastFrame()` string snapshot via Jest/Vitest `toMatchSnapshot` |
| notcurses (C) | render to a non-TTY `FILE*` / `ncdirect` to a buffer; `ncplane_contents()` | plane contents, EGC at coordinates, cell channels |

The unifying idea: **the render target is swappable.** If your app code calls `terminal.draw(...)` it can call it against `TestBackend`; if it writes through a `Renderer` interface it can write to a `bytes.Buffer`. Code that can *only* render to `/dev/tty` is code you wrote untestable on purpose — that coupling is itself the first thing to fix.

---

## Golden-frame / snapshot testing the render buffer

A **golden frame** (a.k.a. snapshot) is a recorded, reviewed, checked-in representation of exactly what the UI renders for a given state and size. The test re-renders and asserts byte/cell-equality against the golden. When it differs, either you introduced a regression (fix the code) or you changed the layout intentionally (review the diff, re-bless the golden). The diff *is* the layout review.

What a golden frame must capture to be worth keeping:

- **Cell content** — the characters at each position (catches truncation, smearing, off-by-one borders).
- **Attributes** — bold/reverse/underline (catches "the focus indicator disappeared").
- **Color *intent*** — not raw RGB, but the *role* (error/warn/ok). Assert that the error row carries a reverse-video or a `!` prefix, not merely "fg == red." A golden that only checks red is the color-only-error bug with a green checkmark on it.

### Example 1 — ratatui (Rust), golden buffer with explicit cell assertions

```rust
use ratatui::{
    backend::TestBackend,
    Terminal,
    buffer::Buffer,
    layout::Rect,
};

// A render fn that takes a frame is testable; one that owns its own Terminal+stdout is not.
fn render_status(frame: &mut ratatui::Frame, state: &AppState) { /* ... draws widgets ... */ }

#[test]
fn status_bar_shows_noncolor_error_marker_at_80_cols() {
    let backend = TestBackend::new(80, 24);          // in-memory; no real terminal
    let mut terminal = Terminal::new(backend).unwrap();
    let state = AppState::with_error("disk full");

    terminal.draw(|f| render_status(f, &state)).unwrap();

    // Whole-frame golden: expected is built cell-by-cell or from a fixture string grid.
    let expected = Buffer::with_lines(vec![
        // ... 23 lines ...
        "ERR! disk full                                                                  ",
    ]);
    // assert_eq compares chars AND styles. To pin the non-color signal explicitly:
    let buf: &Buffer = terminal.backend().buffer();
    let cell = buf.get(0, 23);                        // first col of status line
    assert_eq!(cell.symbol(), "E");                   // 'ERR!' label — survives NO_COLOR
    assert!(buf.get(3, 23).symbol() == "!");          // bang marker is a non-color channel
    // And the layout itself:
    assert_eq!(buf.area, Rect::new(0, 0, 80, 24));    // size regression guard
    assert_eq!(buf, &expected);                       // full golden
}
```

The full-buffer `assert_eq!` is the layout regression net. The individual `cell.symbol()` checks are the *property* assertions that keep the golden honest — they say "this error is legible without color," which is the actual UX requirement.

### Example 2 — Textual (Python), SVG snapshot + Pilot

Textual's `pytest-textual-snapshot` renders the app to an SVG (a faithful, reviewable image of the terminal frame) and diffs it against a checked-in golden. The diff is shown as a side-by-side image in the test report — the cleanest "review the layout change" workflow of any framework.

```python
# test_log_viewer.py
import pytest
from log_viewer.app import LogViewerApp

@pytest.mark.parametrize("size", [(80, 24), (120, 40), (40, 12)])
def test_log_viewer_layout(snap_compare, size):
    # Renders headlessly at the given size, compares to a blessed SVG golden.
    # Fails (with an image diff) on any layout/style regression at any size.
    assert snap_compare("log_viewer/app.py", terminal_size=size)


@pytest.mark.asyncio
async def test_filter_updates_match_count_and_keeps_focus():
    app = LogViewerApp(seed_lines=BIG_FIXTURE)
    async with app.run_test() as pilot:          # headless Pilot driver; no real TTY
        await pilot.press("/")                    # focus filter (keybinding under test)
        await pilot.press(*"error")               # type into filter
        await pilot.pause()                        # let async filter settle
        count = app.query_one("#match-count").renderable
        assert "matches" in str(count)            # empty/no-match state is shown
        assert app.focused.id == "filter"         # focus did not silently jump
```

`snap_compare` parametrized over sizes is the size-regression guard and the golden-frame test in one. The `run_test()` block is the headless input test. Note the assertions on **match-count** (empty-state affordance) and **focus** — golden images alone would not catch a focus regression that renders identically.

### Keeping goldens maintainable

- **Bless deliberately.** Re-blessing (`cargo insta accept`, `--snapshot-update`, Jest `-u`) must be a reviewed action, never a reflex. A PR that re-blesses 14 goldens with no layout intent is a PR that disabled 14 tests.
- **One state per golden.** A golden of "the whole app mid-flow" diffs noisily. Snapshot discrete states: empty, one match, no-match, error, loading, min-size-too-small.
- **Normalize nondeterminism.** Freeze the clock, the spinner phase, RNG, and any timestamp before snapshotting, or your goldens flake. (See `feedback-latency-and-async-work` for the spinner/clock seams.)
- **Store as text where you can.** Text goldens (ratatui `Buffer::with_lines`, Ink `lastFrame()`) diff in the PR review UI; binary/SVG goldens need a tool. Prefer the reviewable form.

---

## Simulating input and events headlessly

Hand-QA tests input by *being* a human pressing keys. Automated tests inject the same events into the same handler. The seam is your event/message type: if your app reduces over a `Msg`/`Action`/`Event` enum (see `event-loop-and-state-architecture`), you can feed it a scripted sequence and assert the resulting state and frame — no keyboard, no PTY.

- **ratatui:** construct `crossterm::event::Event::Key(KeyEvent::new(KeyCode::Char('/'), ...))` and pass it to your `update(&mut state, event)` directly. Pure function in, new state out.
- **Textual:** `pilot.press("/", "e", "r")`, `pilot.click("#run")`, `pilot.resize_terminal(40, 12)`.
- **Bubble Tea:** `tm.Send(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("error")})`; `tm.Send(tea.WindowSizeMsg{Width: 40, Height: 12})`.
- **Ink:** `stdin.write("error")`; the component re-renders, assert `lastFrame()`.

What to test through injected input:

- **Keybinding routing** — `/` focuses the filter; `j`/`k` move selection; PgUp pages. The failing case (a keystroke going to the wrong widget) is invisible in a static golden and only appears under injected sequences.
- **Focus model** — after `Tab`, the focus indicator moved *and* subsequent keys route to the newly focused widget. Assert both.
- **Resize as an event** — `WindowSizeMsg` / `pilot.resize_terminal` drives reflow without a real SIGWINCH. This is how you test the resize path in CI (real SIGWINCH needs a PTY; the *message* does not).
- **Stick-to-bottom / scroll-vs-tail** — inject "user scrolled up" then "new line arrived" and assert the viewport did *not* jump. This bug is purely behavioral; only injected sequences catch it.

```go
// Bubble Tea (Go) — teatest: drive the program headlessly, then golden the final frame.
func TestResizeReflowsAndTooSmallState(t *testing.T) {
    tm := teatest.NewTestModel(t, newApp(fixtureLines()), teatest.WithInitialTermSize(80, 24))

    tm.Send(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("/")})           // focus filter
    tm.Send(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("err")})
    tm.Send(tea.WindowSizeMsg{Width: 30, Height: 8})                       // shrink hard
    tm.Send(tea.WindowSizeMsg{Width: 10, Height: 3})                       // below minimum

    tm.Send(tea.Quit())
    out, _ := io.ReadAll(tm.FinalOutput(t, teatest.WithFinalTimeout(time.Second)))
    // Golden asserts the "terminal too small" degraded state, not a corrupted frame.
    teatest.RequireEqualOutput(t, out)   // compares against testdata/*.golden
}
```

---

## Rendering to an in-memory backend (no real terminal)

The non-negotiable enabler. In CI there is no TTY (`isatty()` is false; `$TERM` may be unset). Tests that touch `/dev/tty`, call `crossterm::terminal::enable_raw_mode()`, or `SetConsoleMode` will hang, error, or — worse — *pass locally and fail in CI for the wrong reason*. The fix is structural:

- Render through the framework's test backend (`TestBackend`, `run_test()`, `teatest`, `ink-testing-library`).
- Keep terminal setup/teardown (raw mode, alt screen, signal handlers) in a thin **lifecycle** layer that the headless tests do not invoke. Your `update` and `view`/`render` functions take state and a buffer; they know nothing about TTYs. (This is the same seam `lifecycle-signals-and-terminal-restoration` requires for its own reasons — testability and clean restoration come from the same decoupling.)
- Test the not-a-TTY *behavior* too: when `isatty()` is false, the app should fall back to plain line output, not enter raw mode (failure #29, owned by `distribution-and-cross-environment`). Stub the TTY probe and assert the fallback path renders.

If your render code cannot be pointed at an in-memory buffer, that is a design defect to fix *before* writing tests — not a reason the TUI is "untestable."

---

## Testing across sizes and capability levels

Most "looked fine" regressions are size or capability regressions. Hand-QA almost always tests one size (the dev's window) and one capability level (the dev's truecolor terminal). Tests parametrize the axes a human won't.

**Sizes** — at minimum:

- A small standard size (80×24) — the historical floor; still common over SSH and in CI consoles.
- A large size (e.g. 200×50) — catches "fixed-width layout leaves a dead zone" and centering bugs.
- An awkward narrow size (40×12) — catches truncation/wrap policy failures and CJK smear.
- **Below minimum** (e.g. 10×3) — must render the graceful "terminal too small" state, never a corrupt frame or a panic (failure #11).

**Capability levels** — drive these as inputs, not as ambient environment:

- **NO_COLOR / monochrome** — render with color disabled; assert the error/warn/ok states are still distinguishable by glyph/label/attribute (failures #6, #7, #25). This is the test that proves your accessibility claim.
- **Color depth** — 16-color vs 256 vs truecolor; assert graceful degradation, not a crash or invisible text.
- **No alt-screen / no mouse** — assert the app still functions when the capability is absent (failure #28).
- **Locale / wide glyphs** — feed CJK and combining characters; assert width-aware truncation does not split a double-width cell or smear the border (failure #8).

```ts
// Ink (JS/TS) — ink-testing-library + Vitest: parametrize size and NO_COLOR.
import { render } from 'ink-testing-library';
import { describe, it, expect } from 'vitest';
import LogViewer from '../src/LogViewer.js';

describe('LogViewer layout matrix', () => {
  for (const [cols, rows] of [[80, 24], [200, 50], [40, 12], [10, 3]]) {
    it(`renders cleanly at ${cols}x${rows}`, () => {
      const { lastFrame } = render(<LogViewer width={cols} height={rows} lines={FIXTURE} />);
      expect(lastFrame()).toMatchSnapshot();          // golden per size
    });
  }

  it('signals error without relying on color (NO_COLOR)', () => {
    process.env.NO_COLOR = '1';
    const { lastFrame } = render(<LogViewer width={80} height={24} error="disk full" lines={[]} />);
    const frame = lastFrame()!;
    expect(frame).toMatch(/ERR!|✗|\berror\b/i);       // non-color signal present
    expect(frame).toMatchSnapshot();
  });
});
```

A size/capability matrix is a handful of `for` loops. It is the cheapest, highest-yield test a TUI can have, and it closes the entire class of "broke at width N" regressions.

---

## CI wiring

The goal: every PR re-renders the golden matrix headlessly, fails on any unblessed diff, and never depends on a real terminal.

- **No TTY allocation.** Run the suite directly (`cargo test`, `pytest`, `go test`, `vitest run`). Do not wrap in `script`/`unbuffer`/`expect` to fake a PTY — that is a smell that your tests still need a terminal. If a test genuinely needs PTY behavior (rare; usually a real-SIGWINCH integration test), isolate it and gate it, do not make the whole suite depend on it.
- **Pin the environment that affects rendering.** Set `TERM` to a known value, set `COLUMNS`/`LINES` only if your framework reads them, and run color tests with `NO_COLOR` set/unset explicitly. Locale matters for width tests — set `LC_ALL=C.UTF-8` (or your fixture locale) so CJK width math is deterministic across runners.
- **Make golden diffs reviewable artifacts.** On failure, upload the rendered-vs-expected (Textual SVG diff, text diff, `.golden` diff) as a CI artifact so the reviewer sees *what changed on screen* in the PR, not just "snapshot mismatch."
- **Block the reflexive re-bless.** Re-blessing in CI must be impossible (no `-u`/`--snapshot-update` in the CI command) so a green build can never come from a silently-updated golden. Re-blessing is a local, reviewed, committed action.
- **Matrix the platform if you ship cross-platform.** Windows console vs Unix PTY-less render differ; run the headless suite on each target OS (failure #30, owned by `distribution-and-cross-environment`; the tests here are how you execute that coverage).

```yaml
# Illustrative CI step (any framework) — headless, deterministic, no re-bless.
- name: TUI render + behavior tests
  env:
    TERM: xterm-256color
    LC_ALL: C.UTF-8
  run: |
    pytest -q            # or: cargo test --all  | go test ./...  | vitest run
- name: Upload snapshot diffs on failure
  if: failure()
  uses: actions/upload-artifact@v4
  with:
    name: tui-snapshot-diffs
    path: ./**/snapshot_report/   # framework-specific golden/diff output
```

---

## Counter to "TUIs cannot really be tested"

This claim is false, and naming *why* it is reached for is part of the discipline. It usually rests on one of these confusions:

- **"It needs a real terminal."** No — it needs a *render target*, and every framework ships an in-memory one. The terminal is an output device, not a dependency of the layout logic. If your code can't render without a TTY, fix the coupling (it is the same coupling that breaks clean restoration on panic).
- **"The output is just bytes / escape codes, you can't assert on it."** The output is a *cell grid*; the framework gives it to you as a `Buffer` / text / SVG. Asserting on a grid is no harder than asserting on JSON. And escape codes are exactly what you *want* to assert when checking attributes — reverse video for focus is a specific byte sequence.
- **"It changes too often; snapshots are brittle."** Brittleness is a signal you snapshotted the wrong granularity (whole app mid-flow) or left nondeterminism in (live clock, spinner). One-state-per-golden plus frozen time gives stable goldens. A golden that flakes is telling you to fix the seam, not to delete the test.
- **"I'll just look at it."** You will look at it once, at one size, on one terminal, today. The test looks at it every PR, at four sizes, with and without color, on every OS, forever. Manual inspection is a sample of size one from a space of thousands.
- **"Interaction can't be scripted."** Interaction is a sequence of events into your update function. `pilot.press`, `tm.Send`, `stdin.write`, a constructed `KeyEvent` — every framework scripts input headlessly. The focus/routing bugs that hand-QA misses are exactly the ones injected sequences catch.

The honest version of the claim is narrow: *photographic* fidelity (does this glyph anti-alias correctly in iTerm2's GPU renderer?) is out of scope for unit tests — that belongs to a tiny, manual, cross-emulator smoke pass at release time. Everything about *what the app decides to render and how it responds to input* is testable, and is the 95% where regressions actually live.

---

## Common mistakes

- **Snapshotting only that color, not the signal.** A golden that checks `fg == red` for errors passes while the NO_COLOR user sees nothing. Assert the glyph/label/attribute that carries the meaning without color.
- **Goldens with live nondeterminism baked in.** Timestamps, spinner phase, RNG, animation frames make goldens flake; the team learns to ignore failures, and the tests are dead. Freeze every nondeterministic source before snapshotting.
- **One mega-golden of the whole app mid-flow.** Diffs are noisy, intent is unclear, re-blessing is reflexive. Snapshot discrete states.
- **Reflexive re-bless.** `-u` / `accept` run without reading the diff converts a regression into a "passing" test. Re-blessing is a reviewed decision.
- **Testing at one size only.** The dev's window is one point in the size space. Parametrize; always include below-minimum.
- **Letting tests touch a real TTY.** Raw-mode/alt-screen calls in tests hang or pass-for-the-wrong-reason in CI. Render to the in-memory backend; keep lifecycle out of the tested path.
- **Faking a PTY in CI** (`script`, `unbuffer`, `expect` around the whole suite). If the suite needs a fake terminal, it is testing the wrong layer. Headless backends need no terminal.
- **Asserting on raw byte strings the framework already structures.** Compare `Buffer`/text/SVG, not hand-spliced escape sequences — unless the attribute *is* the thing under test.
- **No empty/no-match/loading/too-small golden.** These states are where "looks frozen" bugs live and are trivial to snapshot. Their absence means they are untested by construction.
- **Treating manual inspection as coverage.** "I ran it" is a demo, not a test. It leaves no artifact, runs once, and samples one configuration.

---

## Red flags — STOP

If any of these is true, stop and add the test before the change ships:

- A layout, color, focus, or keybinding change is going out with **no golden-frame test** asserting the new render.
- The TUI has **zero automated tests** of render output, input handling, or terminal restoration (RED failure #31).
- There is **no snapshot/golden regression test for layout** (RED failure #32).
- The only size ever tested is the developer's terminal; **below-minimum and narrow sizes are unverified**.
- Error/warn states are asserted **only by color**, with no non-color-signal check.
- A test **enters raw mode or opens the alt screen** (it is exercising lifecycle, not logic, and will misbehave in CI).
- Goldens were **re-blessed with no diff review**, or the CI command can re-bless (`-u`/`--snapshot-update`/`accept`).
- The render path **cannot be pointed at an in-memory backend** — it only writes to `/dev/tty`.
- A reviewer waves off coverage with **"TUIs can't be tested"** — that is the rationalization this sheet exists to refute.
- "I tested it manually" is offered as the **sole** evidence the change works.

The rule: **what the screen renders and how the app responds to input must be asserted by an automated test that runs headlessly in CI.** A TUI you only verified by looking at it is a TUI whose next layout, size, or color regression your users will find for you.
