# Terminal Lifecycle and Restoration

A TUI temporarily changes shared terminal state: raw mode, alternate screen,
cursor visibility, mouse capture and bracketed paste. Establish cleanup ownership
before the first mutation, including partial initialization failure. Restore on
supported normal/error/unwind/signal paths and document uncatchable failures.
SIGKILL, power loss, hard abort and a disconnected terminal cannot be repaired by
a handler that never runs. Do not promise restoration on literally every exit.

## Borrowed state and partial setup

Probe relevant input/output TTY capabilities before entering interactive modes.
Preserve the user's original state where the framework supports it. Teardown
reverses owned setup, tolerates repeated invocation and must not mask the original
error. An exit statement at the bottom of `main` is insufficient.

Bind cleanup to scope (Rust RAII, Python context manager/finally, Go defer,
C++ destructor) and inspect framework-specific gaps. Scope cleanup does not run
on every process-level termination; signal/abort behavior needs separate handling.

### Rust guard example

Verify APIs against the project's crossterm version. The guard exists before any
fallible terminal mutation. Mark escape-sequence modes before attempting writes
so a partial write still gets best-effort reversal. Test setup failures as well
as a successful exit; this illustration is not a framework integration test.

```rust
use std::io::{self, IsTerminal, Stdout, Write};
use crossterm::{
    cursor, execute,
    event::{DisableMouseCapture, EnableMouseCapture},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

pub struct TerminalGuard {
    out: Stdout,
    raw: bool,
    alternate: bool,
    mouse: bool,
    hidden: bool,
}

impl TerminalGuard {
    pub fn enter() -> io::Result<Self> {
        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "interactive TTY required"));
        }
        let mut guard = Self {
            out: io::stdout(), raw: false, alternate: false, mouse: false, hidden: false,
        };
        enable_raw_mode()?;
        guard.raw = true;
        guard.alternate = true;
        execute!(guard.out, EnterAlternateScreen)?;
        guard.mouse = true;
        execute!(guard.out, EnableMouseCapture)?;
        guard.hidden = true;
        execute!(guard.out, cursor::Hide)?;
        Ok(guard)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.hidden { let _ = execute!(self.out, cursor::Show); }
        if self.mouse { let _ = execute!(self.out, DisableMouseCapture); }
        if self.alternate { let _ = execute!(self.out, LeaveAlternateScreen); }
        if self.raw { let _ = disable_raw_mode(); }
        let _ = self.out.flush();
    }
}
```

If a setup operation fails, the already-constructed guard drops and reverses the
modes attempted so far. A raw-mode API that partially mutates state before failing
may need a framework-specific restoration path; verify its contract. If callers
can start with modes already enabled, capture/restore that state rather than
blindly treating disable/show/leave as the original state.

## Signals, panic and process termination

| Event | Required treatment |
|---|---|
| Normal return/error/unwind | Scope-owned teardown, including partial setup. |
| SIGINT/SIGTERM | Request orderly shutdown through the event loop/framework; test actual coverage. |
| SIGTSTP/SIGCONT | Restore before suspend and reinitialize after resume where supported. |
| SIGWINCH | Invalidate size and reflow from fresh dimensions; no destructive redraw in a raw signal handler. |
| Panic | Restore before printing if the trace would be hidden; inspect unwind versus abort configuration. |
| Explicit exit/abort | Check which cleanup is skipped; restore before an owned explicit exit when feasible. |
| SIGKILL/disconnect | State a recovery limitation; cleanup may never execute or writes may fail. |

Raw signal handlers must use signal-safe mechanisms. Prefer a framework-supported
notification/flag/channel into orderly loop shutdown, not arbitrary logging,
allocation or terminal I/O directly in the handler. Panic hooks require their own
best-effort safety review; they do not guarantee recovery from all crashes.

Python `finally` does not run on default terminating signals or `os._exit`.
Go `defer` does not run on `os.Exit`, and a panic in another goroutine requires
appropriate ownership. Rust `Drop` does not run on hard abort; a panic hook may
help for a Rust panic but cannot handle every process failure. Inspect the exact
framework/version rather than assume any of these are automatically covered.

## Verification

Use existing headless/PTY tests and representative terminals to exercise normal
exit, exceptions/unwind, each supported signal, suspend/resume, partial setup,
non-TTY streams and relevant SSH/tmux behavior. Compare terminal modes, cursor,
mouse/paste behavior and shell usability before/after. Do not send destructive
signals to unrelated processes or the user's shell.

Offer a clean noninteractive/plain path rather than writing escape sequences to a
pipe or CI log. Document recovery such as a shell `reset` for unhandled terminal
failure without presenting it as a substitute for correct lifecycle ownership.

Related: [terminal-substrate-and-constraints.md](terminal-substrate-and-constraints.md),
[testing-tuis.md](testing-tuis.md), [accessibility-in-the-terminal.md](accessibility-in-the-terminal.md).
