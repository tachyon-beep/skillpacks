---
name: distribution-and-cross-environment
description: Use when a TUI "works on my machine" but breaks over SSH, inside tmux/screen, in CI, or on a teammate's terminal; when output is garbled or the app hangs after `your-tui | head` / `your-tui > out.txt` (no-TTY / piped path); when truecolor or alt-screen features assumed locally render as escape-code soup elsewhere; when startup feels slow or the first frame lags; when fast scroll, large output, or a flood of log lines tears, stutters, or pins a CPU core; when Windows Terminal, macOS Terminal.app, iTerm2, Alacritty, kitty, WezTerm, or the VS Code integrated terminal each behave differently; when packaging/distribution (cargo install, pip, npm, Homebrew, a single static binary) gives users a different experience than your dev build; or when you have no pre-release matrix and ship blind across emulators, multiplexers, locales, and remote sessions.
---

# Distribution and Cross-Environment Robustness

## The UX stake

A TUI is not software that runs on *your* terminal. It is software that runs on a stranger's terminal, reached over a flaky SSH link, nested two layers deep inside tmux, with a locale you didn't set, a `$TERM` you didn't anticipate, and a color palette someone customized in 2014. The moment your binary leaves your laptop, every assumption you made about the substrate becomes a bet — and "works only locally" is the single most common way a beautiful TUI becomes an unusable one.

The failure is experiential before it is technical. When a user pipes your tool and gets a screenful of `^[[38;2;128;...` escape codes, they don't think "ah, a TTY-detection gap." They think *this tool is broken* and they reach for the one that isn't. When your app pins a core because someone `tail -f`'d a million-line log into it, they don't think "no throttling." They think *this thing is slow and bloated.* When startup takes 900ms because you probed the terminal synchronously over a 200ms-RTT SSH session, the user feels the lag in their fingertips and quietly stops using it.

Cross-environment robustness is therefore a **design responsibility**, not a packaging afterthought. The deliverable is not "a binary" — it's *the same intended experience, degraded gracefully, on every surface a user can plausibly reach you from.* This sheet is how you get there.

This sheet closes three specific, recurring failure modes:

- **Works-only-locally** — the app assumes your emulator, your `$TERM`, your locale, your truecolor support, and falls apart anywhere else (including when its own output is piped or redirected).
- **tmux / SSH breakage** — multiplexer and remote-session layers that strip, rewrite, or lie about capabilities, corrupt the clipboard/title/alt-screen, or change timing.
- **Slow startup** — synchronous capability probing, heavy work before first paint, or fat binaries that make the tool feel laggy the instant it launches.

> Boundary note: *what* the substrate can do and *how to probe it* live in `terminal-substrate-and-constraints.md`. *Restoring* the terminal cleanly on every exit path lives in `lifecycle-signals-and-terminal-restoration.md`. *Not tearing on fast redraws* lives in `rendering-and-redraw-discipline.md`. This sheet is about **delivery across the matrix** — getting a correct, fast, well-packaged experience to every environment, and proving it before release.

---

## 1. The environment matrix

You do not test "the terminal." You test a **grid**. The two axes that matter most:

**Axis A — the emulator / surface** (what draws the cells):

| Surface | What it's good at | What bites you |
|---|---|---|
| **Windows Terminal** | Truecolor, modern; default on Win 11 | Conhost legacy mode still exists; `cmd.exe`/old PowerShell hosts lack VT until enabled; backslash paths in config |
| **macOS Terminal.app** | Ubiquitous on macOS | **No truecolor** (256-color only as of current builds) — your 24-bit theme silently quantizes or looks wrong; weaker glyph coverage |
| **iTerm2** | Truecolor, images, rich | Proprietary escape sequences (imgcat, badges) you must not depend on |
| **VS Code integrated terminal** | Where many devs actually live | xterm.js — quirks around mouse reporting, `$TERM=xterm-256color` even when more is available, focus/blur events, occasional reflow lag |
| **Alacritty / kitty / WezTerm / foot** | Fast, GPU, truecolor, modern protocols (kitty keyboard, kitty graphics) | Differ on which *extended* protocols they support; don't assume kitty-isms everywhere |
| **Linux console (`/dev/tty`, no X)** | The fallback of fallbacks | 8/16 colors, limited glyphs, no mouse, narrow Unicode |

**Axis B — the transport / wrapper** (what sits between your process and those cells):

| Layer | What it changes | What bites you |
|---|---|---|
| **Local, interactive** | Nothing — your happy path | Lulls you into shipping assumptions |
| **SSH** | Latency (every probe round-trips); `$TERM` propagation; sometimes a dumber pty | Synchronous probing = visible startup lag; clipboard (OSC 52) blocked; resize events delayed |
| **tmux / screen** | `$TERM` becomes `screen`/`tmux-256color`; passthrough rules; clipboard, title, truecolor all mediated | Truecolor needs explicit tmux config or gets quantized; OSC sequences need `Ptmux;` wrapping; SIGWINCH timing differs; alt-screen interactions |
| **mosh** | UDP, aggressive local echo & prediction, screen diffing | Heavy redraws get predicted/clobbered; some escapes dropped |
| **CI (GitHub Actions, GitLab, Jenkins)** | **Not a TTY.** `$TERM` often `dumb` or unset; no resize; output captured to a log | Anything that assumes a TTY hangs or vomits escapes into the build log |
| **Piped / redirected (`| less`, `> file`, `$(...)`)** | **Not a TTY.** Consumer wants plain text | Alt screen + raw mode + color into a pipe = corrupted, unusable output |
| **Container / minimal image** | Missing locale, missing `terminfo`, `$TERM` unknown | `terminfo` lookups fail; UTF-8 not the locale → glyph mojibake |

The matrix is the **product** of these, but you don't test all ~70 cells. You pick **representative, high-risk cells** (Section 6 gives the checklist). The mental model: *every cell is a different user having a different first impression of your tool.*

### The three classes of behavior every cell falls into

1. **Interactive TTY** — full experience: alt screen, raw mode, color, mouse.
2. **Non-TTY (piped/redirected/CI)** — *degrade to plain text.* No alt screen, no raw mode, no cursor games, optionally no color. This is failure mode #29 and it is non-negotiable: **a TUI must detect that nobody is watching and behave like a CLI.**
3. **Capability-reduced TTY** (Terminal.app, tmux without truecolor, Linux console) — interactive, but *narrower*: fewer colors, fewer glyphs, no fancy protocols. Degrade features, keep function.

---

## 2. The non-TTY contract (closing "no fallback when piped")

This is the highest-leverage robustness fix in the whole sheet, because it is the most common and most embarrassing break: a user runs `mytool | grep foo` or `mytool > report.txt` and gets binary garbage.

**Rule:** Before you enter raw mode or the alternate screen, ask *"is stdout actually a terminal someone is looking at?"* If not, run a **plain, line-oriented, streaming** code path instead.

Decision flow:

```
is stdout a TTY?
├── no  → PLAIN MODE: line-buffered text to stdout, no alt screen,
│         no raw mode, no cursor moves; color only if forced (see §4).
│         Exit codes meaningful. This is your "headless" rendering.
└── yes → is stdin a TTY too (can we read keys)?
          ├── no  → READ-ONLY interactive or fall to plain mode
          │         (e.g. `producer | mytool` with no keyboard)
          └── yes → FULL INTERACTIVE MODE
```

Two refinements that separate a robust tool from a fragile one:

- **Check stdout *and* stderr independently.** Progress/spinners belong on stderr; if stderr is a TTY but stdout is piped, you can still show a spinner to the human while streaming clean data to the pipe.
- **Offer an explicit override.** `--no-tui` / `--plain` and an env var (e.g. `MYTOOL_PLAIN=1`) so users and scripts can force plain mode even on a TTY, and `--tui` to force interactive even when detection is conservative. Detection is a heuristic; always give an escape hatch.

### Example A — Rust (crossterm / ratatui): gate the whole TUI on TTY detection

```rust
use std::io::{self, IsTerminal, Write};
use crossterm::tty::IsTty; // crossterm also exposes IsTty on the stream

fn main() -> anyhow::Result<()> {
    let force_plain = std::env::var_os("MYTOOL_PLAIN").is_some()
        || std::env::args().any(|a| a == "--plain" || a == "--no-tui");

    let stdout_is_tty = io::stdout().is_terminal();
    let stdin_is_tty = io::stdin().is_terminal();

    if force_plain || !stdout_is_tty {
        // PLAIN MODE: stream results as text. No alt screen, no raw mode.
        // Honors NO_COLOR (see color-theming sheet); piped output stays clean.
        run_plain(io::stdout().lock())?;
        return Ok(());
    }
    if !stdin_is_tty {
        // We can draw but can't read keys: render a final static frame,
        // or fall back to plain. Choose per app; never block on input.
        run_plain(io::stdout().lock())?;
        return Ok(());
    }

    // FULL INTERACTIVE: only now do we touch raw mode / alt screen.
    // (Entering/leaving + panic-safe restore lives in the lifecycle sheet.)
    run_interactive()
}

fn run_plain<W: Write>(mut out: W) -> io::Result<()> {
    for line in produce_results() {
        writeln!(out, "{line}")?; // line-buffered, pipe-friendly, Ctrl-C clean
    }
    Ok(())
}
# fn produce_results() -> Vec<String> { vec![] }
# fn run_interactive() -> anyhow::Result<()> { Ok(()) }
```

The discipline: **`run_interactive()` is the *only* function allowed to enter raw mode or the alt screen, and it's only reachable through the gate.** There is no path where a pipe gets escape codes.

### Example B — Python (Textual / Rich): detect, and let the framework degrade

Rich already implements much of the non-TTY contract — but you must let it, and you should make the headless path explicit for scripts and CI.

```python
import sys
import os
from rich.console import Console

def main() -> int:
    force_plain = (
        os.environ.get("MYTOOL_PLAIN")
        or "--plain" in sys.argv
        or "--no-tui" in sys.argv
    )

    # Console.is_terminal is False when stdout is piped/redirected or in
    # most CI; force_terminal=None lets Rich auto-detect rather than guess.
    console = Console(force_terminal=None)

    if force_plain or not console.is_terminal:
        # PLAIN MODE: emit clean text. No Live, no alt screen, no spinners
        # on stdout. (A Textual App would simply never be launched here.)
        for line in produce_results():
            print(line)   # respects pipes, CI logs, `> file`
        return 0

    # FULL INTERACTIVE: safe to launch the Textual App / Rich Live now.
    from myapp.tui import MyApp
    MyApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

For **Textual specifically**: it refuses to start a full app when there's no TTY and will tell you so — which is correct, but means *you* must provide the headless path (a `--plain` command, or Textual's `run_async`/headless driver for tests). Don't let "Textual won't start in CI" be a surprise; design the plain path on purpose.

> The same gate applies in **Bubble Tea** (`isatty` on the file descriptor before `tea.NewProgram`; or pass `tea.WithInput`/`tea.WithoutRenderer` for headless), **Ink** (check `process.stdout.isTTY`; Ink renders to a string when not a TTY but you usually want a deliberate plain path), and **notcurses** (it will fail to init without a suitable terminal — catch it and fall back).

---

## 3. Capability differences + detection *in the wild*

`terminal-substrate-and-constraints.md` covers *how* to probe (querying terminfo, `$COLORTERM`, DA1/DA2 responses, env vars). This section is about the **distribution reality**: which environments lie, which strip, and how to detect *safely under latency*.

### The capabilities that actually vary between environments

| Capability | Where it's missing / lies | Detection signal | Fallback |
|---|---|---|---|
| **Truecolor (24-bit)** | macOS Terminal.app (none); tmux (unless configured); mosh; many SSH'd boxes | `$COLORTERM=truecolor\|24bit`; cautious DA query | Quantize to 256, then 16, then mono (NO_COLOR) |
| **256 color** | Linux console, `$TERM=dumb`, ancient | `$TERM` ends in `-256color`; terminfo `colors#256` | 16-color palette |
| **Alt screen** | `$TERM=dumb`, some CI, some old | terminfo `smcup`/`rmcup` present | Inline scrolling output, no alt screen |
| **Mouse** | VS Code (quirky), tmux (needs config), SSH'd minimal | terminfo / DA; or just offer keyboard parity | Keyboard-only navigation (always provide this) |
| **Unicode width / glyphs** | locale not UTF-8; Linux console; old fonts | `LANG`/`LC_*` contains `UTF-8`; runtime width probe | ASCII box-drawing & symbols; see CJK note below |
| **OSC 52 clipboard** | SSH (often blocked), tmux (needs `set-clipboard on`) | No reliable query — *assume it can fail* | Print the value so the user can copy manually |
| **Kitty keyboard / graphics protocol** | Everywhere except kitty (& a few) | Specific query; default OFF | Legacy keys / no inline images |
| **Synchronized output (DECSET 2026)** | Many older emulators | Query / feature-test | Plain redraw (flicker-tolerant) |

### Detection principles that survive contact with reality

1. **Prefer cheap, local signals over round-trips.** `$COLORTERM`, `$TERM`, `$NO_COLOR`, locale env vars cost nothing and don't depend on the terminal answering. Reserve active queries (DA1/DA2, DECRQM, cursor-position probes) for things env can't tell you — because over SSH every query is a round-trip you *feel* (see §5 on startup).

2. **Time-box every active probe.** If you send a query escape and read the reply, set a deadline (e.g. 100–250ms). A terminal — or a multiplexer that swallowed the query — that never answers must not hang your startup. On timeout, assume the *conservative* capability, not the optimistic one.

3. **Detect, then *let the user override*.** `--truecolor` / `--256color` / `--ascii` flags and matching env vars. No detection is perfect; the user knows their terminal better than your heuristic does.

4. **Degrade by *feature*, not by all-or-nothing.** Missing truecolor shouldn't disable color — it should drop to 256. Missing Unicode shouldn't break layout — it should swap glyphs. Each capability fails independently.

### tmux and SSH: the wrappers that lie

These deserve special treatment because they're where "works locally" goes to die.

**tmux specifics:**
- Inside tmux, `$TERM` is usually `screen-256color` or `tmux-256color` — which by default advertises **no truecolor** even on a truecolor outer terminal. Truecolor only works if the user set `set -ga terminal-overrides ",*:Tc"` (or `terminal-features`). **You cannot assume truecolor inside tmux.** Detect via `$COLORTERM`, and quantize gracefully when it's absent.
- OSC escapes (clipboard via OSC 52, window title) must be **wrapped in tmux's passthrough** (`\ePtmux;\e<seq>\e\\`) or they get eaten. If you support clipboard, detect `$TMUX` and wrap; otherwise the feature silently no-ops and the user thinks it's broken.
- SIGWINCH and resize: tmux mediates resize. Your reflow logic (see `layout-and-responsive-composition.md`) must be event-driven, not poll `$COLUMNS` once at startup — inside tmux, panes resize constantly.

**SSH specifics:**
- **Latency is the enemy of synchronous probing.** A startup that does five terminal queries × 200ms RTT = 1 second of dead air before first paint. Batch queries, time-box them, and prefer env signals (§5).
- `$TERM` is propagated from the client; a minimal server may not have matching `terminfo`. Bundle a small fallback or degrade when `terminfo` lookup fails (common in containers reached over SSH).
- OSC 52 clipboard is frequently blocked by the SSH server / tmux. Don't promise clipboard; *offer it and have a manual fallback*.

---

## 4. Color across the matrix (the distribution slice)

Full theming, NO_COLOR, and monochrome fallback live in `color-theming-and-monospace-canvas.md`. The cross-environment slice you must honor at the delivery boundary:

- **Honor `NO_COLOR`** (any non-empty value → no color) — universal convention, and your *piped/CI* default should lean monochrome anyway.
- **Honor `FORCE_COLOR` / a `--color=always` flag** so users can force color through a pipe (e.g. `mytool --color=always | less -R`). Detection says "no TTY → no color," but the user knows `less -R` will render it.
- **Quantization ladder:** truecolor → 256 → 16 → none. Pick the rung from detected capability (§3), and *test that your palette is still legible at each rung* — a theme tuned for 24-bit often turns to mud at 16.
- **CI default = no color** unless `FORCE_COLOR` is set; colored escape codes in a captured build log are noise that hides real output.

(Errors signaled by color *alone* — failure mode #6 — is an accessibility break covered in the color/accessibility sheets, but it has a cross-environment edge too: on a 16-color or monochrome remote box your "red = error" vanishes entirely. Always pair color with a glyph/word: `✗ ERROR` / `[FAIL]`.)

---

## 5. Performance at scale

A TUI that lags is a UX failure as surely as one that flickers. Three distinct performance problems show up across environments, and SSH/mosh amplify all of them because every byte you emit crosses the wire.

### 5.1 Large output / fast scroll / floods (closing CPU-burn-on-flood)

The classic break: a tool that ingests `tail -f` of a busy log, or paginates a 2-million-row result, and either (a) tries to hold it all in memory, (b) redraws everything on every line, or (c) emits more bytes than a remote link can carry.

**Defenses (most are detailed in sibling sheets; here's the cross-environment framing):**

- **Bounded buffers / ring buffers** (failure mode #19, `event-loop-and-state-architecture.md`): never hold unbounded input. Over SSH this is doubly important — a flood you can't keep up with is a flood you must *drop the middle of*, not buffer forever.
- **Coalesce / throttle redraws to a frame budget.** Don't repaint per input line; repaint at most ~30–60fps (every ~16–33ms), or only when something actually changed (damage tracking — `rendering-and-redraw-discipline.md`). When 10,000 lines arrive in one tick, you draw *one* frame showing the latest viewport, not 10,000 frames.
- **Render the viewport, not the corpus** (virtualization, failure mode #5). Scroll position selects a window into the data; you draw the visible rows only. This is what makes "instant" scroll through a huge buffer possible.
- **Bound bytes-on-the-wire.** Synchronized output (DECSET 2026) where available, diff-based redraw everywhere, and avoid full-screen clears on every frame. Over a 200ms link, a full repaint of an 80×50 screen is *kilobytes per frame* — at 60fps that's a flood you created.

### 5.2 Backpressure: never let the producer outrun the renderer

Architecturally (cross-references `event-loop-and-state-architecture.md` and `feedback-latency-and-async-work.md`): ingestion runs **off the render/input thread**. A bounded channel between them provides backpressure — when the UI can't keep up, the producer slows or drops, rather than the UI thread blocking. The user always keeps an interactive, scrollable view even mid-flood. A frozen UI during a flood is failure mode #13/#16 wearing a performance hat.

### Example C — Go (Bubble Tea): throttle a high-rate stream to a frame budget

```go
// Coalesce a firehose of log lines into at most one redraw per ~33ms.
// The producer goroutine never touches the model; it only sends batches.
type batchMsg []string
type tickMsg struct{}

func tickEvery() tea.Cmd {
	return tea.Tick(33*time.Millisecond, func(time.Time) tea.Msg { return tickMsg{} })
}

type model struct {
	ring    *ringBuffer // bounded — drops oldest, never grows unbounded (§5.1)
	pending []string    // accumulated since last frame
	dirty   bool        // damage flag: only redraw when something changed
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case batchMsg: // many of these can arrive between ticks
		for _, line := range msg {
			m.ring.Push(line) // bounded; old lines evicted under flood
		}
		m.dirty = true
		return m, nil // NOTE: no View() forced here — we wait for the tick
	case tickMsg:
		if !m.dirty {
			return m, tickEvery() // nothing changed: skip the repaint entirely
		}
		m.dirty = false
		return m, tickEvery() // one coalesced frame regardless of input rate
	}
	return m, nil
}
```

The point that survives every environment: **input rate and render rate are decoupled.** Ten thousand lines/sec produces ~30 frames/sec, each showing the current viewport from a bounded buffer. CPU stays flat; the wire stays calm; the UI stays responsive — local, over SSH, or inside tmux.

### 5.3 Startup time (closing "slow startup")

Startup latency is the first thing every user feels, and it's *worst* exactly where you can't see it: over SSH, in a cold container, inside tmux.

Where startup time hides:

1. **Synchronous terminal probing** — the big one. Five active queries over a 200ms SSH link = 1s before first paint. **Fix:** prefer env signals; batch the queries you must make; time-box them; do them *after* a fast first paint where possible.
2. **Heavy work before first frame** — loading a huge config, scanning a directory tree, opening a DB, an initial network call on the UI thread. **Fix:** paint a skeleton/loading frame *immediately*, then load asynchronously and fill in (this is also the empty/loading-state fix, failure mode #20).
3. **terminfo / locale lookups failing slowly** in minimal images. **Fix:** detect failure fast and fall back; don't retry into a timeout.
4. **Binary / runtime cold-start.** A 200MB Electron-grade bundle or a cold Python interpreter importing the world. **Fix:** see packaging (§6) — lazy imports, trimmed binaries, AOT where the ecosystem allows.

**Target:** first meaningful frame in well under ~150ms locally, and a *deliberate* loading state so that even when the real work is slow (remote, cold), the user sees life immediately rather than a hung cursor. *Time to first frame is a number you should measure on every release* (§6 checklist).

---

## 6. Packaging and distribution

How users *get* the tool shapes the experience they get. The recurring trap: your dev build behaves differently from the artifact users install (different env, different terminfo, different default flags, a fat binary that cold-starts slowly).

### Packaging by ecosystem

| Stack | Common channels | Distribution-specific watchouts |
|---|---|---|
| **Rust (ratatui/crossterm)** | `cargo install`, prebuilt static binary (musl), Homebrew, cargo-binstall, distro pkg | Static-link (musl) so it runs on minimal/remote boxes without glibc surprises; strip the binary for size/startup; ship per-OS/arch artifacts; test the *installed* binary, not `cargo run` |
| **Python (Textual/Rich)** | `pipx`/`uv tool install`, wheel on PyPI, PyInstaller/Nuitka single-file, Docker | Cold interpreter + heavy imports = slow start → **lazy-import the TUI framework** behind the plain/interactive gate; pin a Python floor; for single-file builds verify terminfo/locale still resolve |
| **Go (Bubble Tea)** | single static binary, Homebrew, `go install`, GoReleaser | Easiest story — one static binary per OS/arch; still test on Windows Terminal + minimal Linux; embed assets, don't read from `$HOME`-relative paths that may not exist |
| **JS (Ink)** | `npm i -g`, `npx`, pkg/bun single-file | Node cold-start + `node_modules` weight; `npx` re-download latency; ensure `isTTY` checks work under the chosen bundler |
| **C (notcurses)** | distro packages, static build | Hard dependency on terminfo/locale at runtime — document and detect; degrade if absent |

### Cross-cutting packaging rules

- **Test the artifact you ship, not your dev loop.** `cargo run` / `python -m` / `go run` use *your* environment. Install the actual package into a clean container and run *that*. Most "works locally" bugs are really "works in my dev environment."
- **Per-OS / per-arch matrix at build time.** At minimum Linux x86_64 + arm64, macOS arm64 (+ x86_64 if you support it), Windows x86_64. Cross-compile or matrix-build in CI.
- **Don't depend on the user's dotfiles.** Your tool must work for a user with an empty `$HOME`, no `~/.config`, default `$TERM`. Ship sane defaults; treat config as *optional enhancement*.
- **Bundle or gracefully miss terminfo/locale.** In minimal containers these are often absent. Detect and fall back rather than crash.
- **Document the support matrix.** Tell users which terminals/transports are tested and supported. A short "Supported terminals" section in the README sets expectations and turns "broken!" bug reports into "unsupported, here's the fallback."
- **Version your protocol assumptions.** If you adopt kitty graphics or extended keyboard, gate it behind detection and a flag, and note it — so a future terminal change doesn't silently break installed users.

---

## 7. Pre-release cross-environment checklist

Run this before *every* release. It is the antidote to "untested across emulators / SSH / CI / Windows" (failure mode #30) and "no fallback when piped" (#29). You don't need all 70 matrix cells — you need these **representative, high-risk cells** plus the automated smoke tests.

**Non-TTY contract (must pass — these are the embarrassing breaks):**
- [ ] `mytool | cat` → clean plain text, no escape codes, no alt-screen residue
- [ ] `mytool > out.txt` then inspect `out.txt` → plain text only
- [ ] `mytool | head` → exits cleanly on SIGPIPE, no panic, no hang
- [ ] Run in CI (`$TERM` unset/`dumb`, no TTY) → produces useful log output, **exits**, never hangs waiting on a terminal query or input
- [ ] `--plain` / `MYTOOL_PLAIN=1` forces plain mode even on a TTY
- [ ] `NO_COLOR=1` → no color anywhere; `FORCE_COLOR=1 mytool | less -R` → color preserved

**Emulators (interactive, spot-check the high-risk set):**
- [ ] **Windows Terminal** (and one legacy host, e.g. classic PowerShell/conhost) — VT enabled, paths, colors
- [ ] **macOS Terminal.app** — confirm truecolor theme degrades to 256 *legibly* (not mud)
- [ ] **iTerm2** — full experience; no dependence on iTerm-only escapes
- [ ] **VS Code integrated terminal** — mouse, focus/blur, resize, `$TERM` quirks
- [ ] One modern GPU emulator (**Alacritty / kitty / WezTerm**) — truecolor + extended protocols if used
- [ ] **Linux console / minimal `$TERM`** — 16-color + ASCII glyph fallback intact

**Transports / wrappers:**
- [ ] **Over SSH** — measure time-to-first-frame; confirm no multi-second probe stall; resize works
- [ ] **Inside tmux** — truecolor quantizes gracefully (not assumed); resize reflows; clipboard either works (passthrough) or fails *visibly with manual fallback*
- [ ] **tmux *over* SSH** (both layers) — the worst case; the experience is still usable
- [ ] **mosh** (if you claim support) — heavy redraws don't get clobbered

**Locale / Unicode / width:**
- [ ] `LANG=C` / non-UTF-8 locale → ASCII fallback, no mojibake, no layout smear
- [ ] CJK / emoji / combining-character content → double-width handled, no column drift (ties to `color-theming-and-monospace-canvas.md`)
- [ ] Narrow terminal (e.g. 40 cols) and "terminal too small" state both behave (ties to layout sheet)

**Performance:**
- [ ] **Time-to-first-frame measured** locally and over SSH; within target; loading state shown if slow
- [ ] **Flood test**: pipe a huge/fast stream (`yes | head -n 5000000 | mytool`, or a busy `tail -f`) → CPU stays bounded, UI stays interactive, memory stays bounded (ring buffer), wire bytes don't explode
- [ ] **Fast-scroll test**: hold PgDn / drag scroll through a large buffer → no tearing, no stutter (virtualized viewport)

**Packaging:**
- [ ] Install the **shipped artifact** into a clean container/VM and run it (not `cargo run`/`python -m`)
- [ ] Works with **empty `$HOME` / no config** — sane defaults
- [ ] Each target OS/arch artifact launches and renders a first frame
- [ ] README documents the **supported terminal/transport matrix** and known fallbacks

**Automated guardrails (so the matrix doesn't rot — see `testing-tuis.md`):**
- [ ] Headless smoke test in CI asserts the plain-mode path produces expected text and exits 0
- [ ] A golden/snapshot frame test runs in CI (headless driver) so layout regressions are caught without a human eyeballing every emulator
- [ ] A "pipe into the tool" and "pipe out of the tool" test exists and runs in CI

---

## Common mistakes

- **Testing only on your daily-driver terminal.** Your iTerm2/WezTerm with a tuned config is the *least* representative environment your users have. The matrix exists because your laptop is a sample size of one.
- **No TTY check before raw mode / alt screen.** The #1 distribution embarrassment: `mytool | grep` spews escape codes. Gate the entire interactive path on `is_terminal()` and provide a plain code path. (Failure mode #29.)
- **Hanging in CI.** Reading from stdin or waiting on a terminal query when there's no TTY → the build job hangs until it times out. Detect non-TTY and exit; time-box every probe.
- **Assuming truecolor.** Terminal.app has none; tmux usually hides it; many SSH'd boxes lack it. Detect via `$COLORTERM`, quantize down a ladder, and verify legibility at each rung.
- **Synchronous capability probing on startup.** Cheap locally, painful over SSH (every query is a round-trip). Prefer env signals, batch queries, time-box them, paint a fast first frame.
- **Unbounded buffers under a flood.** `tail -f` of a busy log into an in-memory list = OOM or swap death. Ring-buffer everything you ingest; drop the middle, not the present.
- **Redrawing per input line instead of per frame.** A firehose of 10k lines/sec should produce ~30 frames/sec, not 10k repaints. Decouple input rate from render rate; over SSH this is the difference between calm and a self-inflicted byte flood.
- **OSC sequences (clipboard/title) that silently no-op in tmux/SSH.** They need passthrough wrapping (`\ePtmux;...`) or get eaten; over SSH clipboard is often blocked outright. Detect, wrap, and provide a manual fallback — never a silent failure the user reads as "broken."
- **Depending on the user's dotfiles, locale, or a populated `$HOME`.** Ship sane defaults; treat config and a rich locale as optional enhancements, not preconditions.
- **Testing `cargo run` / `python -m` instead of the installed artifact.** The thing users run has a different environment, a different startup cost, and different default flags. Test what you ship.
- **Color (or any signal) by itself.** On a 16-color or monochrome remote box, "red = error" disappears. Pair every color with a glyph or word (`✗ ERROR`). Cross-environment robustness and accessibility are the same discipline here.
- **No documented support matrix.** Users can't tell "unsupported here, use `--ascii`" from "this tool is broken." Write down what you test and what the fallbacks are.

---

## Where this connects

- **Probing capabilities & substrate facts** → `terminal-substrate-and-constraints.md`
- **Restoring the terminal on every exit (panic, Ctrl-C, SIGTERM)** → `lifecycle-signals-and-terminal-restoration.md`
- **Damage tracking, diff redraw, no-flicker, virtualization** → `rendering-and-redraw-discipline.md`
- **Bounded buffers, off-thread ingestion, stick-to-bottom, identity-anchored scroll** → `event-loop-and-state-architecture.md`
- **Async work, spinners/progress, debounce, cancel, empty/loading states** → `feedback-latency-and-async-work.md`
- **NO_COLOR, theming, quantization, CJK/double-width glyphs** → `color-theming-and-monospace-canvas.md`
- **Reflow on resize, minimum-size and "too small" states, wrap-vs-truncate** → `layout-and-responsive-composition.md`
- **Headless drivers, golden/snapshot frame tests, CI smoke tests** → `testing-tuis.md`
