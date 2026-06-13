---
name: information-density-and-progressive-disclosure
description: Use when a TUI screen feels cramped, cluttered, or overwhelming; when every row shows every field and the list becomes an undifferentiated wall of text; when users can't tell what's important at a glance in a monospace grid; when detail is buried so deep nobody finds it, or so shallow the list is useless; when the app has become "modal soup" — popups stacked on popups, dialogs that trap the user with no clear way back; when you're choosing between scrolling, paging, or pagination for a long list; when long values get smeared or silently chopped; when a dashboard tries to show eight panels in 80 columns; when deciding what to put in a list row versus a detail pane versus a modal; when there's "too much to fit" and you need to decide what the terminal shows now, what it shows on demand, and what it never shows. Covers visual hierarchy in a fixed-width cell grid, focus+context, and modal discipline.
---

# Information Density and Progressive Disclosure

## The UX stake

A TUI lives inside a fixed grid of monospace cells — often 80×24, sometimes 200×60, never infinite and never under your control. Every glyph you place competes for the same scarce real estate, and unlike a GUI you cannot lean on font size, whitespace gradients, drop shadows, or a scroll wheel that "just works." The terminal gives you rows, columns, a handful of attributes (bold, dim, reverse, color), and the user's attention for roughly two seconds before they decide your tool is a mess.

Density is therefore a *design decision*, not an accident of how much data you happen to have. Three failures recur:

- **Overcrowding** — you pour every field of every record onto the screen because "the user might need it," and the result is a beige wall where nothing is findable. High density without hierarchy is just noise.
- **Buried information** — terrified of crowding, you hide the one number the user came for behind three keystrokes and a panel they never knew existed. Low density without disclosure paths is just an empty room.
- **Modal soup** — you reach for a popup every time you need to show "a bit more," and the user ends up four dialogs deep, unsure which Escape closes what, having lost the list they were working in.

The discipline that resolves all three is **progressive disclosure**: decide what the terminal shows *now*, what it shows *on demand*, and what it *never* shows — and make the path between those tiers obvious. This sheet is about spending constrained space deliberately.

> This is the *spatial* counterpart to the responsive sheet. `layout-and-responsive-composition.md` decides how regions resize and reflow when the terminal changes shape; this sheet decides what *content* goes into a region of a given size and how the user reaches the rest.

---

## Designing for constrained space

Start from a content budget, not a wishlist. Before drawing anything, answer:

1. **What is the one job of this screen?** A log viewer's job is "let me find and read the relevant line." A package manager's job is "let me see what will change and approve it." Everything that does not serve that job is a candidate for a lower disclosure tier.
2. **What is the minimum viewport you must support?** Pick a floor (commonly 80×24, sometimes 60×20 for split panes or constrained SSH). Design the *primary* job to fit that floor. Treat extra space as a bonus that reveals more, not a baseline you depend on.
3. **What is the natural unit of the data?** A row? A card? A tree node? The unit determines your density model. A row of a process list and a node of a config tree have completely different budgets.

### The three disclosure tiers

| Tier | Where it lives | Cost to reach | Put here… |
|------|----------------|---------------|-----------|
| **Glance** | The persistent layout — list rows, status bar, header | Free (always visible) | The 2-4 fields that let a user *scan and decide* |
| **On demand** | Detail pane, expanded row, footer detail, modal | One keystroke / focus change | The full record once a user has chosen it |
| **Never (here)** | Another screen, a log file, `--json` output, the docs | Context switch | Bulk data, audit trails, machine-consumption formats |

The single most common density mistake is collapsing all three tiers into the Glance tier — every field, every row, all the time. The second most common is putting Glance-tier facts (the error count, the current branch, whether you're connected) into a modal nobody opens.

### Budgeting a row

A list row in 80 columns has ~78 usable cells after a 1-cell gutter on each side. Allocate them explicitly:

```
 │ ● │ k8s-prod-api-7f9c      │ Running   │  3/3 │ 4d │ 142m │
 ▲   ▲                        ▲           ▲      ▲    ▲
 │   status glyph (3)         name (24)   state  ↑    ↑
 gutter                                   (11)   ready age  cpu
                                                 (6)  (4)  (6)
```

Fixed-width fields (status, ready, age, cpu) get a hard cell budget and right-align numbers so they form a readable column. The one variable-width field (name) gets the slack — and a truncation policy (see below) for when it overflows. If you find yourself wanting a sixth and seventh column, that's the signal those fields belong in the On-demand tier, not that you should shrink the font you don't have.

---

## Visual hierarchy in monospace

You have no typography. Your hierarchy toolkit is small and you must use all of it deliberately:

| Channel | Use it for | Pitfall |
|---------|-----------|---------|
| **Position** | Most important info top-left; status bar bottom | The eye lands top-left first — don't waste it on chrome |
| **Bold / bright** | The primary field of a row; the focused element | Bold everything = bold nothing |
| **Dim / faint** | Secondary metadata (age, ids, paths) | Some terminals render dim as invisible — never put load-bearing info in dim alone |
| **Reverse video** | The selected row / focused cell | Reverse + color interacts unpredictably; test it |
| **Color** | Categorical state (green/yellow/red) — as *reinforcement* | Color alone fails CVD, NO_COLOR, monochrome, and screen readers — see below |
| **Whitespace / alignment** | Grouping and column structure | Ragged columns destroy scannability faster than anything |
| **Glyph / symbol** | Status (`●`, `✓`, `!`, `▸`) | Width and font coverage vary; have an ASCII fallback |
| **Indentation** | Hierarchy / nesting in trees | Deep nesting eats your width budget fast |

### Redundant encoding is non-negotiable

Hierarchy that relies on a single channel excludes someone. The canonical failure is **error signaled by color alone** — red text and nothing else. A user with deuteranopia, a user who set `NO_COLOR=1`, a user piping through a pager that strips SGR codes, and a screen-reader user all see plain text with no signal at all.

Encode state in at least two channels — typically a **glyph + a word + color**:

```
 ● Running      ✓ green
 ◐ Pending      ~ yellow
 ✗ Failed       ! red, bold
 ⊘ Unknown      ? dim
```

Now the same row is legible as `✗ Failed` in pure monochrome, conveys urgency via bold, and is read aloud as the literal word "Failed." Color is the *fastest* channel for sighted users in a colored terminal, but it is never the *only* one. (Full treatment in `color-theming-and-monospace-canvas.md` and `accessibility-in-the-terminal.md`.)

### Alignment is your strongest free tool

Monospace is the one place where ASCII-art alignment actually works, and it's the cheapest hierarchy you have. Right-align numbers so magnitudes line up. Left-align labels. Pad fixed fields. The difference between scannable and unscannable is almost always alignment, not color:

```
  Bad (ragged)                 Good (aligned)
  cpu 142m mem 38Mi            cpu  142m   mem   38Mi
  cpu 7m mem 1.2Gi             cpu    7m   mem  1.2Gi
  cpu 1003m mem 512Mi          cpu 1003m   mem  512Mi
```

Beware double-width glyphs (CJK, many emoji): a `🚀` occupies two cells. Measure **display width**, not byte length or `char` count, or your columns smear one cell to the right for every wide glyph. Use a grapheme/width-aware function (`unicode-width` in Rust, `wcwidth`/Rich's cell length in Python, `runewidth` in Go) — covered in `color-theming-and-monospace-canvas.md`.

---

## Focus + context (detail without losing place)

The core tension: the user needs *detail* about one item, but if you replace the whole screen with that detail, they lose the *context* of where they were and what's around it. "I drilled into a log line and now I can't tell which of the 4,000 lines it was."

Patterns, from cheapest to richest:

**1. Master–detail split.** The list stays on the left (or top); a detail pane shows the full record for the selected item on the right (or bottom). Selection and context are never lost — moving the cursor updates the detail live. This is the workhorse pattern for any "list of things with depth." Budget: needs ~100+ columns to feel good horizontally; below that, stack vertically (list on top, detail below) or make detail a toggled overlay of the same region.

**2. Expand-in-place (accordion).** The selected row grows to reveal sub-fields beneath it, pushing later rows down; collapsing restores the compact view. Excellent for trees and "a few rows need depth, most don't." The user never leaves the list. Keep the expand/collapse affordance visible (`▸`/`▾`).

**3. Peek footer.** A persistent 1-3 line region at the bottom shows detail for whatever the cursor is on. Cheap (no layout split), always-available context, but limited depth. Good for "one extra line of explanation per item."

**4. Pushed detail view with breadcrumb.** Full-screen detail, but with a breadcrumb header (`logs ▸ pod-7f9c ▸ line 3,412`) and a guaranteed `Esc`/`Backspace` that returns to *the same scroll position and selection*. Acceptable when detail genuinely needs the whole screen — but you must preserve and restore the list's selection identity on return (see below), or you've recreated "lost my place."

### Preserve selection by *identity*, not index

When you drill in and come back — or when a filter, sort, refresh, or live update changes the list — restore the selection by the item's stable identity (id, key, path), never by absolute row index. Anchoring on index is the bug behind "I had the failing pod selected, the list refreshed, and now I'm pointed at a healthy one." (The general rule lives in `event-loop-and-state-architecture.md`; the disclosure consequence is: every drill-in path must round-trip the *identity* of what the user was looking at.)

---

## Scroll vs page vs paginate

Three different mechanisms for "more than fits," with different mental models. Choosing the wrong one is a real usability failure.

| Mechanism | Mental model | Best for | Keys | Watch out |
|-----------|--------------|----------|------|-----------|
| **Continuous scroll** | "One long thing I move a window over" | Logs, single documents, chat history | `↑`/`↓`, `PgUp`/`PgDn`, `Home`/`End`, wheel | Must show a scrollbar/position indicator or the user is lost in space |
| **Paging** | "Jump a screenful at a time through one thing" | Same content as scroll, faster traversal | `PgUp`/`PgDn`, `Space`/`b` (pager idiom) | Disorienting if it scrolls a full page with no row of overlap — keep 1-2 rows of context |
| **Pagination** | "Discrete numbered pages of a set" | Large *result sets*, especially server-backed where you fetch page-at-a-time | `n`/`p`, `[`/`]`, page-number jump | Loses cross-page context; only justified when fetching everything is infeasible |

**Default to continuous scroll with viewport virtualization for any list a single machine holds.** Pagination is a *data-fetching* compromise (the server only gives you 50 at a time), not a UX preference — don't impose page boundaries on data that's already in memory just because it's long.

### Virtualize the viewport, always

Whatever the mechanism, render only the rows in the visible window plus a small over-scan, never all N items. Rendering 50,000 rows to draw 24 of them burns CPU, causes flicker, and makes scrolling lag. The render contract is "given scroll offset and viewport height, produce these rows" — covered in `rendering-and-redraw-discipline.md`. Virtualization is what makes density *affordable*: you can have a million-line buffer because you only ever paint a screenful.

### Always show position

In a scrolling view the user must know *where* they are in the whole. A scrollbar, a `[1,240 / 50,000]` counter, or a percentage indicator turns "lost in an infinite list" into "near the top, lots below." Without it, scroll is a black void. Pair this with the **stick-to-bottom** rule for tailing logs (auto-follow when at the bottom, detach when the user scrolls up, re-attach at `End`) — detailed in `event-loop-and-state-architecture.md`.

---

## Panels vs modals + modal discipline

This is where "too much to fit" most often goes wrong, and the failure has a name: **modal soup**.

### Panels are persistent; modals are interruptions

- A **panel** is a region of the persistent layout — a sidebar, a detail pane, a footer. It coexists with everything else. The user can look at it *and* the list at the same time. Panels are how you raise density without hiding context.
- A **modal** is a temporary overlay that *takes over input* and blocks the underlying UI until dismissed. It is an interruption by design.

### The bias: prefer panels, expansion, and footers over modals

Most "show me more" needs are better served by the focus+context patterns above. Reach for a modal only when the interaction is genuinely **blocking and decision-shaped**:

**Legitimate modal uses**
- A confirmation that gates a destructive, irreversible action ("Delete 12 resources? [y/N]")
- A required input the rest of the UI cannot proceed without (a passphrase, a commit message)
- A transient command surface that *is* the interaction (a command palette, fuzzy-finder, key-help overlay) — and that closes cleanly on one key

**Smells that you're building modal soup**
- A modal to display read-only detail (use a panel or expand-in-place)
- A modal opened from inside another modal (stop — flatten the flow)
- More than one modal "owed" a dismiss at once
- A modal with no visible way to close, or where `Esc` does something other than "close this"

### Modal discipline (the rules that prevent soup)

1. **One modal at a time.** Model modals as a stack, but enforce a max depth of 1 for ordinary flows. If a modal needs to spawn another, that's a sign the task belongs on its own screen.
2. **`Esc` always cancels the topmost modal, and only the topmost.** This is the most violated and most important rule. The user's panic button must be reliable: `Esc` backs out exactly one level, never the app, never two levels.
3. **Show the dismiss affordance.** A modal must state how to leave it: `[Enter] confirm  [Esc] cancel`. Never make the user guess.
4. **Trap focus inside the modal while it's open, restore it on close.** Tab/arrows cycle within the modal, not the background list. On close, focus returns to exactly where it was. (Focus mechanics: `input-keyboard-mouse-and-focus.md`.)
5. **Dim or visibly recede the background.** The user must see that the underlying UI is inert. A modal that floats over an unchanged, still-bright list reads as "is this on top or am I editing the list?"
6. **Default the safe choice.** For destructive confirmations, the default (the one Enter triggers, the capitalized letter) is *cancel*: `[y/N]`, not `[Y/n]`.
7. **Don't lose work behind a modal.** Cancelling a modal returns to the prior state untouched — never a stale or cleared list.

A command palette is the elegant escape hatch from modal soup: instead of a forest of menus and dialogs, one discoverable overlay (`Ctrl-P`, `:`) exposes every action by name with fuzzy search, runs it, and gets out of the way. It's a modal, but a disciplined one — single, self-dismissing, and it *reduces* the number of persistent affordances competing for screen space.

---

## Truncation / summary patterns

Values overflow their cell budget. The wrong responses are (a) let them smear into the next column and (b) silently chop them so the distinguishing part vanishes. Both destroy the data's usefulness — a path truncated to `/usr/local/share/applic…` is identical to ten other paths.

### Choose truncation by where the signal lives

| Pattern | Looks like | Use when the meaningful part is… |
|---------|-----------|-----------------------------------|
| **Tail ellipsis** | `kubernetes-prod-api-de…` | …at the start (most labels, names read left-to-right) |
| **Head ellipsis** | `…prod-api-7f9c-rz4x` | …at the end (filenames, ids, log lines where the tail differs) |
| **Middle ellipsis** | `/usr/local/…/config.toml` | …at both ends (paths, URLs, fully-qualified names) |
| **Wrap** | text continues on next line | The item deserves multiple rows (selected/expanded row, detail pane) |
| **Summary + count** | `app.log, db.log +14 more` | A *list of small values* in one cell |
| **Numeric humanize** | `1.4G`, `3.2k`, `4d 6h` | Long numbers — humanize before truncating |

Always use a real ellipsis character `…` (or `...` in pure-ASCII mode) so truncation is *visible* — a silently chopped string lies to the user. Measure in display cells, not characters, when deciding where to cut, and never cut in the middle of a double-width glyph (you'll leave a half-cell artifact). Wrap-vs-truncate is a policy, not a per-call decision — set it per region (compact list rows truncate; the detail pane wraps); the responsive interaction with width lives in `layout-and-responsive-composition.md`.

### Summary rows are progressive disclosure in miniature

When a cell would hold a collection, show a summary and disclose on demand: `5 errors, 12 warnings` with the full breakdown one keystroke away; `+14 more` that expands; a count badge that becomes a list when focused. This is the same Glance → On-demand contract applied at the cell level.

---

## Two worked examples (different frameworks)

### Example 1 — Master–detail with redundant state encoding (Rust, ratatui)

A process list (Glance tier: glyph + name + state + cpu) with a live detail pane (On-demand tier: full record). State is encoded in glyph *and* word *and* color, so it survives `NO_COLOR` and monochrome. The detail pane never replaces the list, so context is never lost.

```rust
use ratatui::{prelude::*, widgets::*};

struct Proc { name: String, state: State, cpu_m: u32, mem_mi: u32, cmd: String }
#[derive(PartialEq)] enum State { Running, Pending, Failed }

impl State {
    // Redundant encoding: glyph + word + color. Never color alone.
    fn glyph(&self) -> &'static str { match self { Self::Running=>"●", Self::Pending=>"◐", Self::Failed=>"✗" } }
    fn word(&self)  -> &'static str { match self { Self::Running=>"Running", Self::Pending=>"Pending", Self::Failed=>"Failed" } }
    fn style(&self, color_ok: bool) -> Style {
        if !color_ok { return Style::default(); } // honor NO_COLOR / monochrome
        match self { Self::Running=>Style::default().fg(Color::Green),
                     Self::Pending=>Style::default().fg(Color::Yellow),
                     Self::Failed =>Style::default().fg(Color::Red).add_modifier(Modifier::BOLD) }
    }
}

/// Visible-width truncation with a real ellipsis (tail variant).
fn truncate_tail(s: &str, max: usize) -> String {
    use unicode_width::UnicodeWidthStr;       // measure DISPLAY width, not bytes
    if s.width() <= max { return s.to_string(); }
    let mut acc = String::new();
    let mut w = 0;
    for g in s.chars() {
        let gw = unicode_width::UnicodeWidthChar::width(g).unwrap_or(0);
        if w + gw > max.saturating_sub(1) { break; }   // reserve 1 cell for '…'
        acc.push(g); w += gw;
    }
    acc.push('…');
    acc
}

fn render(f: &mut Frame, procs: &[Proc], sel: usize, color_ok: bool) {
    // Master-detail split. Below ~100 cols you'd switch to vertical stacking.
    let cols = Layout::horizontal([Constraint::Min(34), Constraint::Length(40)]).split(f.area());

    // GLANCE tier: only the fields needed to scan and decide. Virtualize: the
    // List widget paints just the visible window, so this scales to huge sets.
    let rows: Vec<ListItem> = procs.iter().map(|p| {
        let name = truncate_tail(&p.name, 22);
        let line = Line::from(vec![
            Span::styled(format!("{} ", p.glyph()), p.style(color_ok)),
            Span::raw(format!("{name:<22} ")),
            Span::styled(format!("{:<8}", p.word()), p.style(color_ok)),
            Span::raw(format!("{:>5}m", p.cpu_m)),   // right-aligned numbers => readable column
        ]);
        ListItem::new(line)
    }).collect();

    let mut st = ListState::default();
    st.select(Some(sel));
    f.render_stateful_widget(
        List::new(rows).highlight_style(Style::default().add_modifier(Modifier::REVERSED)) // selection survives no-color
            .block(Block::bordered().title(" processes [↑↓ select  ↵ detail  / filter] ")),
        cols[0], &mut st);

    // ON-DEMAND tier: full record for the cursor's item. Context (the list) stays put.
    let p = &procs[sel];
    let detail = format!(
        "name   {}\nstate  {} {}\ncpu    {}m\nmem    {}Mi\ncmd    {}",
        p.name, p.glyph(), p.word(), p.cpu_m, p.mem_mi, p.cmd, // detail pane WRAPS; it doesn't truncate
    );
    f.render_widget(
        Paragraph::new(detail).wrap(Wrap { trim: false })
            .block(Block::bordered().title(" detail ")),
        cols[1]);
}
```

Why it closes the failure modes: the row carries four scannable fields (not twenty), the rest lives in the always-visible detail pane (disclosed, not buried), and there's no modal in sight — detail is a *panel*, so the user never loses their place. State reads correctly with color off.

### Example 2 — Expand-in-place + disciplined modal (Python, Textual)

A results list where most rows stay compact (Glance) and the selected row expands in place to reveal detail (On-demand) — no context lost, no modal. A destructive action *does* use a modal, but a disciplined one: single, `Esc`-cancels, safe default, dismiss affordance shown.

```python
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static, Footer
from textual.widgets.option_list import Option
from textual.containers import Vertical

ITEMS = [  # (id, name, status, detail...)
    {"id": "a1", "name": "deploy-api",   "status": ("✓", "Passed",  "green")},
    {"id": "a2", "name": "deploy-worker","status": ("✗", "Failed",  "red")},
    {"id": "a3", "name": "migrate-db",   "status": ("~", "Pending", "yellow")},
]

def row(item, expanded: bool) -> str:
    glyph, word, color = item["status"]
    # Redundant encoding: glyph + word, color via markup; legible if markup stripped.
    head = f"{'▾' if expanded else '▸'} [{color}]{glyph}[/] {item['name']:<16} {word}"
    if not expanded:
        return head
    # Expanded (On-demand) detail indented beneath the row — list context preserved.
    return head + f"\n    id     {item['id']}\n    status {glyph} {word}\n    log    /var/log/{item['name']}.log"

class ConfirmDelete(ModalScreen[bool]):
    """Disciplined modal: single, Esc cancels, safe default, dismiss affordance shown."""
    BINDINGS = [("escape", "dismiss(False)", "Cancel"), ("y", "dismiss(True)", "Delete")]
    def __init__(self, name: str) -> None:
        super().__init__(); self._name = name
    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(f"Delete '{self._name}'? This cannot be undone.")
            yield Static("[b][Esc][/] cancel   [[/]y[]] delete   (default: cancel)")  # affordance always shown
    # Note: no nested modals, no second dialog — if it needed one it'd be its own screen.

class Results(App):
    CSS = "#dialog { width: 50; border: round $warning; padding: 1 2; }"
    BINDINGS = [("d", "delete", "Delete selected")]
    expanded_id: str | None = None

    def compose(self) -> ComposeResult:
        self.olist = OptionList(*self._options())
        yield self.olist
        yield Footer()   # on-screen keybinding discoverability

    def _options(self):
        return [Option(row(it, it["id"] == self.expanded_id), id=it["id"]) for it in ITEMS]

    def on_option_list_option_selected(self, ev: OptionList.OptionSelected) -> None:
        # Toggle expand-in-place. Disclosure without leaving the list = no lost context.
        self.expanded_id = None if self.expanded_id == ev.option_id else ev.option_id
        self.olist.clear_options(); self.olist.add_options(self._options())
        self.olist.highlighted = ev.option_index   # restore by position the user just acted on

    def action_delete(self) -> None:
        idx = self.olist.highlighted
        if idx is None: return
        item = ITEMS[idx]
        # Modal only because this is blocking + irreversible — the legitimate case.
        def done(confirmed: bool | None) -> None:
            if confirmed:
                ITEMS.remove(item)
                self.expanded_id = None
                self.olist.clear_options(); self.olist.add_options(self._options())
        self.push_screen(ConfirmDelete(item["name"]), done)

if __name__ == "__main__":
    Results().run()
```

Why it closes the failure modes: rows are compact until *the user* asks for depth (expand-in-place), so the list is neither overcrowded nor hollow; detail is disclosed inline so it's never buried; and the only modal is the one case that warrants it — a blocking, irreversible confirmation that is single, `Esc`-cancellable, safe-by-default, and tells the user how to leave.

---

## Common mistakes

- **Every field in every row.** Twenty columns of metadata because "someone might want it." Pick the 2-4 that drive a decision; demote the rest to a detail tier. Density without hierarchy is noise.
- **Burying the headline.** The error count, connection status, or current context shoved into a panel nobody opens. Glance-tier facts belong in the persistent layout (header/status bar), always visible.
- **Modal soup.** A popup for read-only detail, modals launched from modals, `Esc` that doesn't reliably back out one level. Prefer panels/expansion; reserve modals for blocking, decision-shaped interactions; enforce one-at-a-time and reliable `Esc`.
- **Color-only hierarchy.** Red-for-error and nothing else. Excludes CVD users, `NO_COLOR`, monochrome terminals, SGR-stripping pipes, and screen readers. Always encode in ≥2 channels (glyph + word + color).
- **Ragged columns.** Numbers left-aligned, fields unpadded, widths drifting row to row. Alignment is your cheapest, strongest hierarchy tool — use it.
- **Silent / mid-glyph truncation.** Chopping a string with no ellipsis (it lies), or cutting where the distinguishing part lives (head vs tail vs middle matters), or measuring in bytes/chars so double-width glyphs smear the grid. Use a visible `…`, choose the variant by where the signal is, and measure display width.
- **Pagination on in-memory data.** Imposing numbered pages on a list the machine already holds. That's a server-fetch compromise, not a UX choice — default to continuous virtualized scroll with a position indicator.
- **Rendering all N rows.** Painting 50,000 rows to show 24. Virtualize the viewport; it's what makes large datasets affordable and keeps scrolling smooth (see `rendering-and-redraw-discipline.md`).
- **Losing the user's place on drill-in/refresh.** Restoring selection by absolute index instead of stable identity, so a refresh or filter points the cursor at the wrong item. Round-trip the *identity* of what the user was looking at.
- **No position indicator while scrolling.** A scrolling view with no scrollbar/counter/percentage is a void. Show where the user is in the whole.
- **Designing for the big terminal.** Building the dashboard at 200×60 and watching it collapse in an 80×24 SSH session. Design the primary job for your minimum viewport; treat extra space as bonus disclosure.

---

## Related sheets

- `layout-and-responsive-composition.md` — how regions resize/reflow with terminal size and the wrap-vs-truncate policy interaction
- `rendering-and-redraw-discipline.md` — viewport virtualization, damage/diff, paint only what changed
- `color-theming-and-monospace-canvas.md` — color fallbacks, NO_COLOR, double-width glyph handling
- `event-loop-and-state-architecture.md` — identity-anchored selection, stick-to-bottom, ring buffers
- `input-keyboard-mouse-and-focus.md` — focus model, modal focus trapping, key routing
- `affordances-and-discoverability.md` — surfacing keybindings and the actions disclosure relies on
- `accessibility-in-the-terminal.md` — non-visual status paths, redundant encoding for screen readers
