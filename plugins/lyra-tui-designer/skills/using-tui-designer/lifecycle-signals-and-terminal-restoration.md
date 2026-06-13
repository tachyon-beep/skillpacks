---
name: lifecycle-signals-and-terminal-restoration
description: Use when a TUI leaves the terminal broken after it exits — invisible cursor, no echo, garbled prompt, stuck in the alternate screen, mouse-tracking escape codes spewing on every click, "my shell is wrecked, I had to run reset"; or when the app dies on a panic/exception/stack-trace and never cleans up; or Ctrl-C / kill / closing the SSH session leaves raw mode on; or it corrupts output when piped, redirected, or run in CI/cron/Docker because it assumed a TTY; or resizing the window (SIGWINCH) garbles the screen; or Ctrl-Z suspend-and-resume comes back broken. Covers alternate-screen and raw-mode lifecycle, RAII/guard/defer/finally restoration on every exit path, SIGINT/SIGTERM/SIGTSTP/SIGWINCH handling, and graceful degradation when stdout is not a terminal.
---

# Lifecycle, Signals, and Terminal Restoration

## The UX stake

A TUI borrows the user's terminal. Raw mode, the alternate screen, a hidden cursor, mouse-tracking, bracketed paste — these are *global, persistent* mutations of a shared resource the user did not give you to keep. The contract is simple and absolute: **whatever state you change, you change back, on the way out, every single time.**

When you break that contract the user does not see a bug report. They see their prompt vanish. They type and nothing echoes. Their shell is "broken." They run `reset`, or close the terminal and open a new one, and they remember your program as the one that *trashed their machine*. There is no recovering that trust with a nice color scheme. **A TUI that leaves the terminal broken on exit is a UX failure of the highest severity** — higher than a flicker, higher than a clipped row — because it persists *after your program is gone* and damages the thing the user lives in all day.

This is also the failure mode that un-guided code gets wrong most reliably. The happy path looks fine: enter alt screen, draw, leave alt screen, exit 0. Then the first panic, the first `Ctrl-C`, the first `kill`, the first `tui | head`, and the terminal is wreckage. Restoration is not a feature you add at the end. It is the *first* thing you wire up and the *last* thing that runs.

This sheet is a **discipline sheet**. The rule is non-negotiable: **terminal state is restored on every exit path — normal return, error, signal, and panic/crash — and you never enter raw mode or the alternate screen without first confirming you are attached to a real TTY.** The rest is how to make that true in real code, in more than one language.

---

## The mental model: a stack of borrowed mutations

Setup is a sequence of terminal mutations. Teardown is that sequence **reversed**:

```
SETUP (in order)                      TEARDOWN (reverse order)
  1. probe: is stdout a TTY?            6. show cursor
  2. enable raw mode                    5. disable bracketed paste
  3. enter alternate screen             4. disable mouse capture
  4. enable mouse capture               3. leave alternate screen
  5. enable bracketed paste             2. disable raw mode
  6. hide cursor                        1. (nothing — probe had no effect)
```

Two rules fall out of this picture immediately:

1. **Teardown order is the reverse of setup order.** Leaving the alternate screen *before* showing the cursor can hide your fix; disabling raw mode *before* leaving the alt screen can leak control codes into the user's shell. Reverse-order is not aesthetic — it is correctness.
2. **The only way to guarantee teardown runs is to bind it to scope exit, not to a line of code you hope is reached.** A `cleanup()` call at the bottom of `main` runs on the happy path and *only* the happy path. Panics skip it. `os.Exit` skips it. An unhandled signal skips it. That is the trap, and the next section is the way out.

---

## ALWAYS restore — bind teardown to scope, not to a hopeful call site

The single most important idea in this sheet: **do not write `cleanup()` and trust that control reaches it.** Bind teardown to a language construct that fires *no matter how the scope is left*. Every language has one. Use it.

| Language / runtime | Construct that always runs | Mechanism |
|---|---|---|
| Rust | `Drop` on a guard struct | RAII — runs on return, `?` early-exit, and panic unwind |
| Go | `defer` | runs on return and on `panic` (recover in the same frame) |
| Python | `try/finally` or a context manager (`__exit__`) | runs on return, exception, and most signals-as-exceptions |
| C / C++ | C++ destructor (RAII) / `atexit` + signal handler in C | scope exit; `atexit` for normal exit only |
| Node / JS (Ink) | `try/finally` + `process.on('exit'|'SIGINT'|...)` | unwind + process-level hooks |
| Bubble Tea (Go) | framework `Program.Run()` deferred restore + own signal layer | library restores; you still own SIGTERM/panic |

### Rust — a `Drop` guard is the canonical answer

In Rust, RAII makes this clean: a guard struct enters terminal mode in its constructor and restores in `Drop`. `Drop` runs on normal return, on `?`-propagated errors, **and during panic unwind** — so a `panic!` anywhere in your draw loop still restores the terminal on the way out.

```rust
use std::io::{self, IsTerminal, Stdout, Write};
use crossterm::{
    cursor, execute,
    event::{DisableMouseCapture, EnableMouseCapture},
    terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    },
};

/// RAII guard. Constructing it puts the terminal into TUI mode.
/// Dropping it — on return, on `?`, or during a panic unwind — restores it.
pub struct TerminalGuard {
    out: Stdout,
}

impl TerminalGuard {
    pub fn enter() -> io::Result<Self> {
        // RULE: never enter raw mode on a non-TTY. Bail to the caller's
        // headless path instead of corrupting a pipe or a CI log.
        if !io::stdout().is_terminal() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "stdout is not a TTY; refusing to enter raw mode",
            ));
        }
        enable_raw_mode()?;
        let mut out = io::stdout();
        execute!(out, EnterAlternateScreen, EnableMouseCapture, cursor::Hide)?;
        Ok(Self { out })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        // Reverse order. Best-effort: a Drop must not panic, so we ignore
        // errors here — there is nothing useful to do if restore itself fails,
        // and panicking in Drop during an unwind aborts the process.
        let _ = execute!(
            self.out,
            cursor::Show,
            DisableMouseCapture,
            LeaveAlternateScreen
        );
        let _ = disable_raw_mode();
        let _ = self.out.flush();
    }
}
```

The crucial extra step in Rust: **a panic by default unwinds and prints the message *while the terminal is still in alt-screen/raw mode*** — so the user sees nothing, or a smeared single-line trace, and then the guard runs after the message is already lost. Install a panic hook that restores *first*, then prints:

```rust
pub fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Restore the terminal BEFORE the default hook prints the trace,
        // so the backtrace lands on a clean, scrolling, cursor-visible screen.
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, cursor::Show);
        default(info);
    }));
}

fn main() -> io::Result<()> {
    install_panic_hook();
    let _guard = TerminalGuard::enter()?; // restored on EVERY exit from main
    run_app()?;                            // panic here? still restored.
    Ok(())
}
```

> Note: with `panic = "abort"` in your release profile, `Drop` does **not** run on panic — the panic hook is then your *only* restoration path on a crash. This is exactly why you install both: the guard for clean and error exits, the hook for crashes. Belt and suspenders, because the cost of a bare terminal is so high.

### Python (Textual / Rich) — context manager, plus the signals the framework misses

Textual and `rich` restore on clean exit and on a caught `KeyboardInterrupt`. They do **not** save you from `SIGTERM` (a plain `kill`, a container stop, a `systemd` shutdown), which by default terminates the process with *no* unwinding and *no* `finally`. You must convert it into an orderly shutdown.

```python
import os
import signal
import sys
from contextlib import contextmanager

from rich.console import Console

console = Console()

@contextmanager
def terminal_session():
    """Enter TUI mode; restore on return, exception, OR SIGTERM."""
    if not sys.stdout.isatty():
        # Not a terminal (pipe, redirect, CI). Refuse raw mode; the caller
        # should fall back to a plain, line-oriented path.
        raise RuntimeError("stdout is not a TTY; refusing to enter alt screen")

    def _on_sigterm(signum, frame):
        # Turn the signal into a normal exception so `finally` runs.
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, _on_sigterm)

    console.show_cursor(False)
    print("\x1b[?1049h", end="", flush=True)  # enter alternate screen
    try:
        yield console
    finally:
        # Reverse order, always — clean exit, exception, KeyboardInterrupt.
        print("\x1b[?1049l", end="", flush=True)  # leave alternate screen
        console.show_cursor(True)
        signal.signal(signal.SIGTERM, previous)


def main() -> int:
    with terminal_session() as con:
        run_app(con)   # any exception here still restores the terminal
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> If you use `curses`, prefer `curses.wrapper(main)` — it sets up and tears down `cbreak`/`noecho`/`keypad` for you and restores them in its own `finally`, even on exception. Do **not** hand-roll `initscr()`/`endwin()` pairs without a `try/finally`; an exception between them is exactly the bug this sheet exists to prevent.

### Go (Bubble Tea / tcell) — `defer` plus your own signal layer

Bubble Tea's `Program.Run()` restores the terminal via its own deferred teardown and traps `SIGINT` by default. But it does **not** trap `SIGTERM`, and a `panic` *inside a `tea.Cmd` goroutine* you spawned will not be recovered by the framework. Own those edges:

```go
func main() {
    if !term.IsTerminal(int(os.Stdout.Fd())) {
        runHeadless() // not a TTY: plain output, no alt screen
        return
    }

    p := tea.NewProgram(initialModel(), tea.WithAltScreen(), tea.WithMouseCellMotion())

    // Convert SIGTERM into an orderly Quit so Bubble Tea's deferred
    // restore runs, instead of a hard kill that leaves raw mode on.
    sigs := make(chan os.Signal, 1)
    signal.Notify(sigs, syscall.SIGTERM)
    go func() {
        <-sigs
        p.Quit() // triggers the framework's clean teardown
    }()

    if _, err := p.Run(); err != nil {
        // p.Run already restored the terminal before returning the error.
        fmt.Fprintln(os.Stderr, "error:", err)
        os.Exit(1)
    }
}
```

> The Go trap to internalize: **`os.Exit()` does not run `defer`s.** If any code path calls `os.Exit` while the terminal is in raw mode and your restore is a `defer`, the terminal is left broken. Restore *before* you exit, or route every exit through the framework's `Quit`.

---

## Signal handling

Signals are the exit paths you did not write. A user pressing `Ctrl-C`, a `kill` from another shell, a window resize, a `Ctrl-Z` — each delivers a signal, and the *default disposition* of most of them is "terminate the process immediately, no cleanup." That default is what leaves the terminal broken. Handling signals is not optional polish; it is half of "restore on every exit path."

| Signal | Trigger | What a TUI must do |
|---|---|---|
| `SIGINT` | `Ctrl-C` | Initiate orderly shutdown → run teardown → exit. (Or intercept as an in-app key if you intentionally own `Ctrl-C`.) |
| `SIGTERM` | `kill`, container stop, `systemd`, supervisor | Same as SIGINT: orderly shutdown + teardown. **The most commonly missed signal.** |
| `SIGHUP` | terminal/SSH session closes | Restore and exit; the TTY is going away. |
| `SIGWINCH` | window resized | Re-query terminal size, mark a full redraw, reflow layout. Do **not** exit. (Reflow detail lives in `layout-and-responsive-composition.md`; *receiving* the signal is here.) |
| `SIGTSTP` | `Ctrl-Z` (suspend) | Restore the terminal to cooked mode, then re-raise the *default* SIGTSTP so the shell actually suspends you. |
| `SIGCONT` | `fg` (resume) | Re-enter raw mode + alt screen, re-query size, force a full redraw. |

### SIGINT / SIGTERM — orderly shutdown, not a hard kill

The pattern is identical for both: **catch the signal, set a shutdown flag (or send a quit message into your event loop), let teardown run, then exit.** Never do real teardown work *inside* the signal handler in languages where handlers run in a restricted async-signal-safe context (C, and to a degree Go). Flip a flag; let the main loop notice.

```rust
// Rust + crossterm: the event loop already surfaces Ctrl-C as a Key event
// when raw mode is on, so you handle it as a keystroke and `break`, which
// drops the guard. For SIGTERM (no key event), install a flag handler:
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

let shutdown = Arc::new(AtomicBool::new(false));
{
    let s = shutdown.clone();
    // `signal-hook` registers an async-signal-safe handler that only
    // flips the flag — the loop checks it and exits, dropping the guard.
    signal_hook::flag::register(signal_hook::consts::SIGTERM, s)?;
}
loop {
    if shutdown.load(Ordering::Relaxed) { break; } // guard drops → restored
    // ... poll events with a timeout, draw ...
}
```

### SIGTSTP / SIGCONT — suspend and resume must round-trip cleanly

This is the subtle one, and it is a real RED-baseline failure: a TUI that does not handle `Ctrl-Z` either ignores it (the user can't suspend) or suspends *with raw mode still on*, so the shell prompt they drop back to is broken until they `fg` again. The correct dance:

1. On `SIGTSTP`: tear the terminal down to its cooked state (show cursor, leave alt screen, disable raw mode), then **re-raise SIGTSTP with the default handler** so the kernel actually stops the process. If you swallow it, you never suspend.
2. On `SIGCONT` (when `fg` resumes you): re-enter raw mode and the alt screen, **re-query the size** (the user may have resized while suspended), and force a full repaint.

```python
import signal

def _on_suspend(signum, frame):
    teardown_terminal()                      # cooked mode, cursor shown
    signal.signal(signal.SIGTSTP, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTSTP)     # actually suspend now
    signal.signal(signal.SIGTSTP, _on_suspend)  # re-arm for next time

def _on_resume(signum, frame):
    setup_terminal()                         # raw mode, alt screen
    request_full_redraw()                    # screen is stale after resume

signal.signal(signal.SIGTSTP, _on_suspend)
signal.signal(signal.SIGCONT, _on_resume)
```

> Many mature frameworks (Textual, ncurses-via-wrapper, Bubble Tea) handle suspend/resume for you. Verify it — actually press `Ctrl-Z`, then `fg` — rather than assuming. If the framework owns it, you are done; if not, you own it.

### SIGWINCH — receive it here, reflow elsewhere

A resize that is not handled leaves stale rows, clipped columns, and a layout sized for the old window. The *receiving* discipline is: register for `SIGWINCH`, and on delivery **invalidate cached size, re-query the real dimensions, and request a full redraw** — never trust a cached width/height across a winch. The actual reflow/min-size/"terminal too small" policy belongs to `layout-and-responsive-composition.md`; this sheet's job is making sure the signal is wired at all. The RED baseline shows code that hardcodes 80×24 and never listens for the resize — both halves are failures.

---

## Graceful degradation when there is no TTY

The probe — "is stdout actually a terminal?" — is not a nicety. It is a precondition for *every* terminal mutation. Enabling raw mode on a pipe, a file redirect, or a CI log does not produce a TUI; it produces corrupted bytes and, frequently, a hung process waiting for input that will never come from a keyboard.

**Rule: probe before you mutate.** `isatty(stdout)` (and often `isatty(stdin)`). If it is not a terminal, take a completely different code path — plain, line-oriented, non-interactive output — and never touch raw mode or the alternate screen.

```
                 ┌─────────────────────────┐
   start ───────▶│ isatty(stdout) == true?  │
                 └───────────┬─────────────┘
                     yes     │      no
              ┌──────────────┘      └──────────────┐
              ▼                                     ▼
   probe capabilities                    headless / plain path:
   (alt screen? colors? size)            - line-buffered stdout
   enter TUI mode                        - no escape codes, no raw mode
   run interactive loop                  - --json / --plain friendly
                                         - exit with a clean status code
```

Decision points for the no-TTY path:

- **Piped / redirected** (`mytui | grep`, `mytui > out.txt`): emit plain text or a structured format (`--json`). The user explicitly asked for capturable output; give it to them.
- **CI / cron / systemd / Docker without a PTY**: no terminal at all. Run non-interactively or refuse with a clear message ("requires an interactive terminal; pass `--plain` for batch mode") and a non-zero exit code — never hang.
- **`dumb` terminal** (`TERM=dumb`, some editors' embedded shells, basic CI): no cursor addressing, no colors. Degrade to scrolling line output.
- **`NO_COLOR` / not a color terminal**: drop color but keep structure. (Full color-fallback policy lives in `color-theming-and-monospace-canvas.md`; the *TTY/capability gate* is here.)

The deeper point: **TTY-ness and capabilities are runtime facts you detect, not assumptions you bake in.** Don't assume the alternate screen exists; don't assume truecolor; don't assume a width of 80. Probe, then choose. (Substrate probing in depth: `terminal-substrate-and-constraints.md`. Cross-environment testing: `distribution-and-cross-environment.md`.)

```go
// Go: branch at the top of main, before any terminal mutation.
interactive := term.IsTerminal(int(os.Stdout.Fd())) &&
    term.IsTerminal(int(os.Stdin.Fd())) &&
    os.Getenv("TERM") != "dumb" &&
    !forcePlain // honor an explicit --plain / CI override

if !interactive {
    renderPlain(os.Stdout) // no escape codes, no raw mode, no alt screen
    return
}
```

---

## Common mistakes

- **Cleanup as a final statement in `main`.** Runs on the happy path only. Panics, `os.Exit`, `sys.exit` from deep code, and unhandled signals all skip it. → Bind teardown to scope (Drop / defer / finally / context manager).
- **`panic = "abort"` (Rust) with restoration only in `Drop`.** Abort skips `Drop` entirely. → Also install a panic hook that restores before the trace prints.
- **`os.Exit()` (Go) / `os._exit()` (Python) while in raw mode.** Skips `defer`/`finally`. → Restore first, or route exits through the framework's quit.
- **Teardown in the wrong order.** Leaving alt screen after re-showing the cursor, or disabling raw mode before leaving alt screen, leaks codes into the shell. → Strict reverse of setup order.
- **Handling SIGINT but not SIGTERM.** `kill`, container stops, and supervisors send SIGTERM; the default disposition is a hard kill with no cleanup. → Handle SIGTERM identically to SIGINT.
- **Doing real work inside a signal handler.** Re-entrancy and async-signal-safety hazards (especially C/Go). → Flip a flag or send a message; let the main loop tear down.
- **Swallowing SIGTSTP.** The user presses `Ctrl-Z` and nothing happens, or they suspend with raw mode on. → Restore, re-raise default SIGTSTP, re-arm; re-setup on SIGCONT.
- **Entering raw mode / alt screen without an `isatty` probe.** Corrupts pipes and CI logs; can hang waiting for keyboard input that never comes. → Probe first; take a plain path otherwise.
- **Caching terminal size across a resize.** Stale width/height after SIGWINCH produces clipped or smeared layout. → Invalidate and re-query on every winch.
- **The panic trace printed inside the alt screen.** The crash message is the most useful output you'll ever produce, and it lands on a screen the user never sees. → Restore *before* printing the trace.
- **Restoration that panics/throws.** A teardown that itself errors during an unwind can abort or mask the original failure. → Best-effort, ignore-error teardown in the cleanup path.

---

## Red flags — STOP

If any of these are true, stop and fix restoration before writing another feature:

- There is no `Drop` / `defer` / `finally` / context manager wrapping terminal setup — teardown is "a function I call at the end."
- You have not tested what happens on a **panic / unhandled exception** mid-draw. (Trigger one on purpose. Look at the terminal afterward.)
- You have not pressed **`Ctrl-C`** and confirmed the prompt comes back clean.
- You have not run **`kill <pid>`** (SIGTERM) from another shell and confirmed clean restoration.
- You have not run the program with output **piped or redirected** (`yourtui | cat`, `yourtui > /tmp/x`) and confirmed it does not emit raw escape codes or hang.
- You have not run it **without a TTY** (in CI, in `sh -c 'yourtui < /dev/null'`, in a container without `-t`).
- You enable raw mode / enter the alt screen **before** any `isatty` check.
- Your panic/crash backtrace prints **inside** the alternate screen.
- You handle `SIGINT` but `grep` your code finds no `SIGTERM`.
- Pressing **`Ctrl-Z`** then `fg` leaves the screen or the shell broken.

The acceptance bar for this sheet: **kill the program in the four ugliest ways you can — Ctrl-C, SIGTERM, a forced panic, and `tui | head` — and in all four the terminal comes back clean.** If you cannot demonstrate that, the work is not done.

---

## Rationalizations — and the counter

These are the things an agent (or a rushing engineer) says to skip the rule. Each is wrong, and here is why.

- *"I'll add cleanup at the end of main — that covers it."*
  It covers the one path that was never the problem. The terminal breaks on panics, signals, and early exits, none of which reach the end of main. Bind teardown to scope.

- *"The framework (ratatui/Textual/Bubble Tea) handles restoration, so I don't need to."*
  It handles *its* paths — usually clean exit and SIGINT. It does not, by default, handle SIGTERM, panics in goroutines/threads you spawned, `os.Exit`, or `panic = "abort"`. Read what it actually covers and own the rest. Verify by killing it the ugly ways.

- *"Nobody runs this without a TTY."*
  Someone pipes it into `grep` on day one. CI runs it with no PTY. A user redirects output to a file to attach to a bug report — about *your terminal corruption*. The non-TTY path is not an edge case; it's a guarantee.

- *"It works on my machine / in my terminal."*
  Your machine never sent SIGTERM, never lost the SSH session (SIGHUP), never ran in `TERM=dumb`, never crashed mid-frame. "Works on my machine" tests exactly the path that already works.

- *"Crashes are rare; I'll handle restoration after I ship the features."*
  Crashes are rare; *the first crash* is certain, and its cost — a wrecked terminal the user blames on you — is the worst single outcome the program can produce. Restoration is cheap to wire up first and expensive to retrofit through every exit path later. It is the foundation, not the finish.

- *"Restoring before printing the panic is over-engineering."*
  The panic message is your most valuable diagnostic, and without this it is invisible or smeared. One panic hook turns "the screen went blank" into a readable backtrace on a clean screen. That is the opposite of over-engineering.

- *"I'll just tell users to run `reset` if it breaks."*
  You are shipping a known defect and outsourcing the cleanup to the victim. The user who has to run `reset` has already decided your tool is unreliable.

---

## Checklist

- [ ] Terminal setup is wrapped in an RAII guard / `defer` / `try-finally` / context manager — teardown is bound to scope exit, not a call site.
- [ ] Teardown runs in the exact reverse order of setup.
- [ ] A panic / unhandled-exception hook restores the terminal **before** printing the trace.
- [ ] (Rust) If `panic = "abort"`, restoration also lives in the panic hook, not only in `Drop`.
- [ ] `SIGINT` triggers orderly shutdown with full teardown.
- [ ] `SIGTERM` is handled identically to `SIGINT` (not left at default kill).
- [ ] `SIGTSTP`/`SIGCONT` round-trip: suspend restores to cooked mode and re-raises default; resume re-enters TUI mode and forces a full redraw.
- [ ] `SIGWINCH` invalidates cached size, re-queries dimensions, and requests a full redraw.
- [ ] `isatty` is checked **before** any raw-mode / alt-screen mutation.
- [ ] A non-TTY (pipe, redirect, CI, `TERM=dumb`) takes a plain, line-oriented path and never hangs.
- [ ] Verified by hand: Ctrl-C, `kill` (SIGTERM), a forced panic, and `tui | head` all leave the terminal clean.

---

## Related sheets

- `terminal-substrate-and-constraints.md` — capability probing in depth (truecolor, alt-screen support, size queries) once you've confirmed a TTY.
- `layout-and-responsive-composition.md` — what to *do* on resize: reflow, min-size, "terminal too small," wrap-vs-truncate. This sheet only delivers the SIGWINCH signal.
- `color-theming-and-monospace-canvas.md` — `NO_COLOR`/monochrome fallback once you know color is available.
- `distribution-and-cross-environment.md` — testing restoration and the non-TTY path across emulators, SSH, CI, and Windows.
- `event-loop-and-state-architecture.md` — where the shutdown flag / quit message is consumed and how the loop exits cleanly.
