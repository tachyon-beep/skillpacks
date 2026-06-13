---
name: affordances-and-discoverability
description: Use when a TUI's keybindings live only in the author's head, when users cannot tell which rows are selectable or where the cursor is, when there is no status bar / footer legend / key hint / help overlay / `?` screen, when features (filter `/`, PgUp/PgDn, Home/End, jump-to, multi-select) are invisible until someone reads the source, when reviewers say "how was I supposed to know that key existed", when the app needs a README just to operate, or when you are adding a command palette, on-screen hints, or first-run guidance to a ratatui / Textual / Bubble Tea / Ink / notcurses interface so it is learnable without a manual.
---

# Affordances and Discoverability

A TUI has no hover state, no tooltips, no right-click menu, no menu bar the user can browse, and no greyed-out buttons that hint at what is possible-but-not-now. It is a grid of monospace cells. Every capability your app has is, by default, *invisible* — the user cannot see that PgDn scrolls, that `/` filters, that `Enter` opens, that `Tab` switches panes, that `:` opens a command palette. They find out by reading your source, your README, or by accident. Most never find out at all. They use 10% of the program and assume the other 90% does not exist.

This is the failure this sheet closes: **hidden keybindings, undiscoverable features, and no help.** The UX stake is direct — an interface whose power is unreachable is, for the user who cannot reach it, exactly as capable as one that never had the power. Discoverability is not a "nice to have" layered on top of features; it is the surface through which features *become* features. A keybinding nobody can find is dead code with a UX cost.

The design goal is a single sentence: **the user should be able to operate the program competently without reading anything outside the program itself.** No man page, no `--help` dump they have to memorise, no wiki. The screen teaches the screen.

## The discoverability ladder

Real interfaces layer affordances so the cost of discovery scales with the obscurity of the feature. Cheap, common actions are advertised constantly; rare, expert actions are one cheap lookup away. Four rungs:

1. **Persistent hints** — an always-visible footer/status bar showing the handful of context-relevant keys. Zero cost to discover; the user sees them without doing anything.
2. **Help overlay** — a `?` (or `F1`) full-screen or modal cheat sheet listing *every* binding, grouped. One keystroke away; the user discovers it because the footer advertises `? help`.
3. **Command palette** — a fuzzy-searchable list of every action by name, with its binding shown next to it. The user types what they *want* ("export", "theme") and finds the action and learns its key. This is the discoverability backstop: anything not on the footer or in muscle memory is reachable by typing a word.
4. **First-run / empty-state guidance** — when there is nothing to show yet, the empty pane is real estate to teach the primary action ("Press `/` to filter, `n` for new").

A well-designed TUI uses all four. The footer handles the 5 keys used constantly; the help overlay handles "I know there's more"; the palette handles "I don't even know what it's called or what key it is"; the empty state handles "I just opened this and don't know where to start." Skip any rung and a class of user falls through.

## Rung 0: signaling interactivity in a text grid

Before you advertise *keys*, the grid itself must signal *state* — what is selected, where the cursor is, what is focused, what is actionable. In a GUI this is done with affordances the OS provides (a button looks raised, a text field has a border and a blinking caret, the focused widget has a glow). In a TUI you must build every one of these out of the only materials you have: **reverse video, color, bold, a cursor glyph, a gutter marker, and a border style.** Use them deliberately and consistently.

**The selected row.** A list where the "current" item is indistinguishable from the others is unusable — the user presses `Enter` and has no idea what they just acted on. The conventional, robust signal is **reverse video (inverted fg/bg) on the full row width**, optionally plus a gutter marker (`▌`, `▶`, `>`). Full-width matters: highlighting only the text leaves the rest of the row dark and the selection looks ragged; padding the highlight to the pane width reads as a solid selection bar, the way every file manager and `fzf` does it.

```
  config.toml        2.1 KB
▌ src/main.rs        14 KB     ← reverse-video bar, full width, gutter ▌
  README.md          3.4 KB
```

Crucially: **do not signal selection with color alone.** A red or blue text color for the selected row fails the same users a color-only error fails — color-vision-deficient users, monochrome terminals, `NO_COLOR`, custom themes where your accent collides with the background. Reverse video survives all of those (it swaps whatever the two colors are), and is the one "highlight" primitive that is universally legible. Layer color on top as enhancement, never as the sole carrier. (See `color-theming-and-monospace-canvas` and `accessibility-in-the-terminal`.)

**The cursor.** In an editable field (a filter box, a text input), the user must see *where their next keystroke lands*. Either show the real terminal cursor at the insertion point (un-hide it and position it; most frameworks expose `set_cursor`) **or** draw a synthetic block/bar caret with reverse video on the character under it. A filter box with no visible caret reads as a label, not an input — users type and watch nothing happen because they did not realise it was focused.

**Focus.** When two regions can both receive keystrokes (filter box + list, sidebar + main pane), the *focused* one must be visibly distinct, because the same key means different things in each (`j` scrolls the list but types a letter in the filter). The standard signal is a **brighter / accented border on the focused pane and a dim border on the unfocused one** — a one-line change that turns "ambiguous keystroke routing" into "obvious." (Focus mechanics and key-routing are owned by `input-keyboard-mouse-and-focus`; this sheet covers making the focus *visible*.)

```
┌─ Filter ─────────┐   ┌─ Files ──────────┐
│ main             │   │ ▌ src/main.rs    │   ← right pane focused:
└──────────────────┘   │   README.md      │     bright border + active bar
   (dim border)        └──────────────────┘     left pane: dim border
```

**Actionability.** If only *some* rows are openable (a tree where leaves open and branches expand), the difference should be visible: a `▸`/`▾` disclosure triangle, a trailing `/` on directories, a dimmed style on inert rows. The user should never have to press a key to discover whether a key does anything here.

The rule across all of these: **state the user must reason about must be visible, and signaled by more than color.** Selection, cursor, focus, and actionability are the four; a TUI that renders all four legibly is already more discoverable than most, before you have added a single hint.

## Rung 1: the always-visible key hint (status bar / footer legend)

The footer is the single highest-leverage discoverability feature and the cheapest to build: **a one-line bar, usually at the bottom, showing the keys relevant *right now*.** It is the TUI's equivalent of a toolbar — constantly present, glanceable, teaching by existing.

Design rules that separate a good footer from a useless one:

- **Show the keys for the current context, not all keys.** When the filter box is focused, show `Esc cancel · Enter apply`. When the list is focused, show `/ filter · ↑↓ move · Enter open · ? help`. A footer that lists 30 global bindings is noise; one that shows the 4–6 that apply *here* is signal. This is what makes it teach — it answers "what can I do *now*."
- **Always reserve a slot for `? help`** (or `F1`). This is the link to Rung 2. As long as the footer is visible, the user can always find the full reference, which means you never have to cram everything into the footer.
- **Use the `key action` pairing format**, separated by a low-contrast `·` or `│`. Dim the separators and brighten the keys so the keys pop. `q quit  / filter  ? help` parses instantly.
- **Modifier notation must be conventional and consistent:** `^C` or `Ctrl-C` (pick one), `M-x`/`Alt-x`, `S-Tab`/`Shift-Tab`, `↑↓←→` for arrows, `⏎`/`Enter`, `⎋`/`Esc`. Decide your house style once.
- **It must reflow on resize and degrade gracefully.** On a narrow terminal you cannot show 8 hints; drop the least important ones and keep `? help` (which is the escape hatch to *all* of them). Never let the footer overflow and corrupt the layout — truncate with an ellipsis or shorten labels. (Reflow is owned by `layout-and-responsive-composition`; the footer is a first-class participant in it.)
- **Keep it to one line.** A two-line footer is eating content space; if you have that many keys to advertise, that is what the help overlay is for.

A footer is not a substitute for a help overlay — it is the advertisement *for* the help overlay, plus the cheat for the handful of keys used every second. The two work as a pair.

## Rung 2: the help overlay

The footer shows ~5 keys; the application has ~30. The help overlay is where the other 25 live. Triggered by `?` (the near-universal convention) and/or `F1`, it shows **every binding, grouped by category, with one-line descriptions**, over (or instead of) the current screen, dismissed by `?` again, `Esc`, or `q`.

What good help overlays do:

- **Group bindings semantically**, not alphabetically: *Navigation*, *Selection*, *Actions*, *View*, *Global*. A flat 30-row list is a wall; grouped, the user scans to the category they want.
- **Show the binding exactly as the user must press it**, next to a verb-first description: `gg   Jump to top`, `/    Filter`, `dd   Delete selected`. Align the keys in a column so the eye runs down them.
- **Be generated from the same binding table that drives the event loop**, never hand-maintained in a separate string. A help overlay that has drifted from the real bindings is worse than none — it actively lies. Make the keymap the single source of truth and render *both* the dispatcher and the overlay from it; then a binding cannot exist without being documented, and cannot be documented without existing. This is the structural fix for "hidden keybindings": it becomes *impossible* to add a binding the help screen omits.
- **Be context-aware if the app is modal.** A modal/vim-like TUI can show the bindings for the *current* mode plus global ones, the way `less` shows different help in different states. At minimum, show everything and label mode-specific keys.
- **Scroll if it overflows**, and say so (`↑↓ scroll · ? close`). On a tiny terminal the full keymap will not fit; treat the overlay like any other scrollable viewport.

```
╭───────────────── Keyboard Shortcuts ─────────────────╮
│  Navigation                Actions                    │
│    ↑/k     Up                Enter  Open              │
│    ↓/j     Down              Space  Select            │
│    PgUp    Page up           d      Delete            │
│    PgDn    Page down         r      Rename            │
│    g g     Top               y      Copy              │
│    G       Bottom                                     │
│  View                      Global                      │
│    /       Filter            :      Command palette   │
│    Tab     Switch pane       q      Quit              │
│                             ?      Toggle this help   │
╰──────────────── ↑↓ scroll · ? / Esc close ───────────╯
```

The help overlay closes "no help" outright. The footer's `? help` slot closes "the help exists but nobody can find it." You need both.

## Rung 3: the command palette pattern

The command palette (popularised by Sublime/VS Code, now standard in `lazygit`, `k9s`, `Helix`, GitHub's own TUI, and Textual apps) is the **discoverability backstop and the power-user accelerator in one widget.** Triggered by `:` (vim/ex tradition), `Ctrl-K`, `Ctrl-P`, or `Ctrl-Shift-P`, it opens a fuzzy-searchable list of **every named action**, each shown *with its keybinding*, and runs the one the user picks.

Why it is the most powerful discoverability tool you can ship:

- **It inverts the lookup.** The footer and help overlay require the user to know what they are looking at and scan for it. The palette lets the user type what they *want* in their own words — "exp" surfaces "Export to CSV", "thm" surfaces "Cycle theme". They find the feature by intent, not by memorising a layout. This is how a user discovers a feature *they did not know existed*: they guess a word, and there it is.
- **It teaches keybindings as a side effect.** Because each palette entry shows its accelerator (`Export to CSV          ^E`), every time the user reaches for the palette to do something, they are shown the faster key for next time. The palette is a built-in tutor that promotes users from palette → muscle memory.
- **It is the home for the long tail.** Actions too rare to deserve a footer slot or a dedicated key still need to be reachable. The palette holds *all* of them, so you can keep the footer minimal without making rare features undiscoverable.

Design rules:

- **Every action in the palette must carry its binding (if it has one)** in a right-aligned column. This is non-negotiable — it is what makes the palette a teaching tool rather than just a fallback.
- **Fuzzy match, not prefix match.** "exp csv" should find "Export selection to CSV". Subsequence matching (the `fzf` algorithm) is the expectation.
- **Same source of truth as the keymap and help overlay.** The action registry — `{name, description, binding, handler}` — drives the dispatcher, the help overlay, *and* the palette. One list, three surfaces. (See "the action registry," below.)
- **Show actions that have no binding too.** The palette is the *only* way to reach an action you chose not to bind. That is fine and intended; not everything deserves a key.
- **Respect availability.** Grey out (or omit, with a reason) actions not valid in the current context, the way a GUI greys a menu item. "Paste" with an empty clipboard should look unavailable, not silently no-op.

## Rung 4: learnability without a manual

The four rungs above are mechanisms. Learnability is the property they exist to produce, and a few habits pull it together so a first-time user is competent in minutes:

- **Teach in the empty state.** Before there is any data, the pane is blank canvas — use it. An empty list should read `No items. Press n to create, / to filter` rather than a void that looks like a hang. (The "looks frozen" failure is owned by `feedback-latency-and-async-work`; the *teaching* opportunity of the empty state is owned here.) This is the cheapest onboarding you will ever build: the user learns the primary action at exactly the moment they need it.
- **Make the first action obvious from the first frame.** A new user's eye should land on one thing to do. The footer's leftmost hint, the empty-state prompt, and the initial focus ring should all point at the same primary action.
- **Follow conventions so prior knowledge transfers.** `q`/`Ctrl-C` quit, `?` help, `/` filter/search, `j/k` and arrows move, `Enter` confirm/open, `Esc` cancel/back, `Tab` next field, `gg`/`G` top/bottom, `:` command palette, `Space` toggle/select. Every convention you honour is a binding the user *already knows* and you never have to teach. Every convention you violate is a binding you must now teach *and* one that will surprise users who reach for the standard one. Deviate only with a strong reason, and advertise the deviation loudly in the footer.
- **Prefer self-evident bindings, and mnemonic ones where you can.** `n` for new, `d` for delete, `r` for rename, `/` for filter, `?` for help are guessable; `x` for "toggle vertical split" is not. Guessable bindings make the help overlay a confirmation rather than a necessity.
- **Show, don't hide, the modal state.** If your app is modal (vim-style normal/insert), the *current mode must be on screen at all times* — a `-- INSERT --` style indicator — or users will be lost about why their keys do unexpected things. A mode you cannot see is the worst kind of hidden state.
- **Never require reading something outside the program.** The test of success: hand the binary to someone who has never seen it, with no README and no `--help`, and watch them. If they can filter, navigate, act, and quit within a minute or two — purely from what the screen tells them — the interface is learnable. If they get stuck and ask "how do I...", that question names a missing affordance: the answer should have been on the footer, in the help overlay, or in the palette.

## The action registry: one source of truth for three surfaces

The structural mistake behind most discoverability failures is **the keymap, the help text, and the palette being three separate, hand-maintained lists.** They drift; the help screen documents a key that was renamed; a new binding never makes it into the help; the palette is missing half the actions. The fix is architectural: define every action **once**, as data, and render every surface from it.

An action record carries everything all three surfaces need:

```
Action {
  id:          "export_csv",
  name:        "Export to CSV",      // shown in palette + help
  description: "Write selection to a .csv file",
  binding:     Some("ctrl-e"),       // None = palette-only
  category:    "Actions",            // groups the help overlay
  available:   |state| !state.selection.is_empty(),
  run:         |state| { ... },
}
```

From this one list you generate: (1) the **key dispatcher** (match the pressed key against `binding`); (2) the **help overlay** (group by `category`, list `binding` + `name`); (3) the **command palette** (fuzzy-match `name`, show `binding`, gate on `available`); and optionally (4) the **footer** (a curated subset, by `id`). Now "hidden keybinding" and "undiscoverable feature" are *structurally impossible*: an action that exists is, by construction, in the help overlay and the palette, and an action you delete vanishes from all three at once. This is the single most important thing this sheet asks you to build.

## Code example 1 — Textual (Python): bindings → footer + help, for free

Textual's `BINDINGS` table is exactly the action registry pattern: declare each binding once with a `show=True`/`description`, and the built-in `Footer` renders the visible ones while the `?`-bound help screen lists them all. The keymap *is* the documentation.

```python
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, ListView, ListItem, Label


class FileBrowser(App):
    CSS = """
    ListView > ListItem.--highlight { background: $accent; }  /* full-width selection bar */
    """

    # One declaration per action. `show=True` puts it on the Footer (Rung 1);
    # every binding here is also picked up by the built-in help panel (Rung 2),
    # so the footer and help can never drift from the real keymap.
    BINDINGS = [
        Binding("slash", "filter", "Filter", show=True),
        Binding("enter", "open", "Open", show=True),
        Binding("d", "delete", "Delete", show=True),
        Binding("question_mark", "help_panel", "Help", show=True),
        Binding("colon", "command_palette", "Commands", show=True),
        Binding("g,g", "top", "Top", show=False),     # power key: in help, not footer
        Binding("G", "bottom", "Bottom", show=False),
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        self.list = ListView()           # highlighted item gets the .--highlight bar
        yield self.list
        yield Footer()                   # renders the show=True bindings, live + reflowing

    def on_mount(self) -> None:
        if not self.list.children:       # Rung 4: teach in the empty state
            self.list.append(ListItem(Label("No files. Press [b]/[/b] to filter, "
                                            "[b]:[/b] for commands.")))

    def action_help_panel(self) -> None:
        self.action_show_help_panel()    # full keymap overlay, grouped, from BINDINGS

    # action_filter / action_open / action_delete / action_top ... handlers


if __name__ == "__main__":
    FileBrowser().run()
```

The discoverability win is that there is *no separate help string to maintain*: `Footer`, the help panel, and Textual's built-in command palette all read `BINDINGS`. Add a binding and it appears on all three; the `show` flag is just the footer/help curation knob.

## Code example 2 — ratatui (Rust): one keymap, three surfaces

ratatui is unopinionated — it gives you cells, not widgets-with-bindings — so you build the registry yourself. The payoff is total: the same `Vec<Action>` drives the footer, the `?` overlay, and the `:` palette, so they cannot diverge.

```rust
use ratatui::{prelude::*, widgets::*};

struct Action {
    name: &'static str,        // shown in palette + help
    key: &'static str,         // human-readable binding, e.g. "PgDn"
    desc: &'static str,
    in_footer: bool,           // Rung 1 curation
}

// THE single source of truth. Dispatcher, footer, help overlay, and palette
// all read this list — a binding cannot exist undocumented, or vice versa.
const ACTIONS: &[Action] = &[
    Action { name: "Filter",   key: "/",     desc: "Fuzzy-filter the list", in_footer: true  },
    Action { name: "Open",     key: "Enter", desc: "Open selected item",    in_footer: true  },
    Action { name: "Page down",key: "PgDn",  desc: "Scroll one page",       in_footer: false },
    Action { name: "Top",      key: "gg",    desc: "Jump to first item",    in_footer: false },
    Action { name: "Help",     key: "?",     desc: "Toggle this help",      in_footer: true  },
    Action { name: "Commands", key: ":",     desc: "Open command palette",  in_footer: true  },
    Action { name: "Quit",     key: "q",     desc: "Exit",                  in_footer: true  },
];

/// Rung 1: always-visible footer of the curated keys, reflow-friendly.
fn footer() -> Line<'static> {
    let mut spans = Vec::new();
    for a in ACTIONS.iter().filter(|a| a.in_footer) {
        spans.push(Span::styled(a.key, Style::new().fg(Color::Yellow).bold()));
        spans.push(Span::raw(format!(" {}", a.name)));
        spans.push(Span::styled("  ·  ", Style::new().fg(Color::DarkGray)));
    }
    Line::from(spans)
}

/// Rung 2: full keymap overlay, generated from the same list.
fn help_overlay() -> Paragraph<'static> {
    let rows: Vec<Line> = ACTIONS.iter().map(|a| {
        Line::from(vec![
            Span::styled(format!("{:<7}", a.key), Style::new().fg(Color::Yellow).bold()),
            Span::raw(a.desc),
        ])
    }).collect();
    Paragraph::new(rows).block(
        Block::bordered().title(" Keyboard Shortcuts  (?/Esc to close) "),
    )
}

/// Rung 0: selected row as a FULL-WIDTH reverse-video bar (not color-only),
/// so selection is legible under NO_COLOR / monochrome / CVD.
fn render_list(f: &mut Frame, area: Rect, items: &[String], selected: usize) {
    let rows: Vec<ListItem> = items.iter().enumerate().map(|(i, s)| {
        let style = if i == selected {
            Style::new().add_modifier(Modifier::REVERSED)   // survives any theme
        } else {
            Style::new()
        };
        ListItem::new(Line::from(Span::styled(format!(" {s}"), style)))
    }).collect();
    f.render_widget(List::new(rows), area);
}
```

The `:` command palette reuses `ACTIONS` directly: fuzzy-match the typed query against `a.name`, render each candidate as `"{name}  {key}"` so the user *learns the accelerator while using the fallback*. Three surfaces, one list — the structural guarantee that nothing your program can do is hidden from the people using it.

## Common mistakes

- **Treating the footer as optional chrome.** The most common omission. Without it, every key is hidden by default and the user has no anchor to even discover the help overlay. The footer is the entry point to the entire discoverability ladder; it is the first thing to build, not the last.
- **Color-only selection / focus / cursor.** A blue "selected" row, an accent-colored focused border, a colored caret — all invisible to CVD users, monochrome terminals, `NO_COLOR`, and clashing themes. Use **reverse video** as the load-bearing signal and treat color as enhancement. (Cross-ref `color-theming-and-monospace-canvas`, `accessibility-in-the-terminal`.)
- **Help text maintained separately from the keymap.** The classic drift bug: the `?` screen documents a key that no longer exists, or omits one that does. Generate help (and the palette) from the same action registry that drives the dispatcher, or it *will* lie.
- **A footer that lists every binding.** A 30-key footer is noise and overflows narrow terminals. Show the ~5 context-relevant keys plus `? help`; let the overlay carry the rest.
- **No `? help` slot in the footer.** Even with a great help overlay, if nothing advertises it, users do not press `?` — they were never told to. The overlay is only as discoverable as its advertisement.
- **No visible focus indicator across panes.** Two panes that both take keystrokes with identical borders make every key ambiguous. A bright-vs-dim border is a one-line fix. (Routing mechanics: `input-keyboard-mouse-and-focus`.)
- **A blank empty state.** An empty list with no prompt reads as a frozen or broken app *and* wastes the best onboarding moment you get. Always teach the primary action where the data would be.
- **Inventing bindings for actions that have conventions.** Rebinding quit off `q`/`Ctrl-C`, help off `?`, or search off `/` discards prior knowledge every user already has and forces you to teach what you could have inherited. Honour conventions; advertise the rare deviation.
- **Hidden modal state.** A vim-style mode with no on-screen `-- INSERT --` indicator means keys silently change meaning with no visible cause — the most disorienting hidden state a TUI can have.
- **A command palette whose entries omit their keybindings.** Then the palette is only a fallback, never a tutor; users never graduate from palette to muscle memory. Always show the accelerator beside the action name.
- **Footer/overlay that overflows instead of reflowing.** On a narrow or resized terminal, an un-truncated footer corrupts the layout — a discoverability feature that breaks rendering. Reflow and truncate gracefully; always keep `? help`. (Cross-ref `layout-and-responsive-composition`.)

## Cross-references

Sheets: `input-keyboard-mouse-and-focus` (focus model and key routing — this sheet makes focus *visible*; that one decides *where keys go*), `color-theming-and-monospace-canvas` (why selection/error signals must not be color-only; NO_COLOR), `accessibility-in-the-terminal` (non-visual discoverability and screen-reader paths), `feedback-latency-and-async-work` (empty / loading / no-match states that this sheet uses to teach), `layout-and-responsive-composition` (footer and overlays as participants in reflow), `information-density-and-progressive-disclosure` (revealing detail on demand, the complement to revealing *actions* on demand).
