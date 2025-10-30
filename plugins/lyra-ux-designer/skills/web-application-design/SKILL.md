---
name: web-application-design
description: Use when designing web applications, SaaS tools, dashboards, or admin panels - responsive patterns (mobile <768px, tablet 768-1024px, desktop >1024px), data visualization, keyboard shortcuts, bulk actions, and progressive enhancement
---

# Web Application Design

## Overview

This skill provides **The Web Application Usability Framework**, a systematic 4-dimension methodology for designing complex web applications, SaaS tools, dashboards, and admin panels. Use this when designing data-heavy interfaces, workflow-driven tools, or multi-platform responsive web applications.

**Core Principle**: Web applications serve power users who need speed, clarity, and flexibility. Success means presenting complex data clearly, enabling efficient workflows through keyboard shortcuts and bulk actions, adapting gracefully across devices, and ensuring core functionality works even in degraded conditions.

**Focus**: Complex data displays, keyboard-driven workflows, responsive design (mobile to desktop), and progressive enhancement for reliability.

## When to Use This Skill

**Use this skill when:**
- Designing web applications, SaaS tools, or admin panels
- Working with complex data displays (tables, charts, dashboards)
- Building tools for power users or enterprise environments
- Implementing responsive web designs across mobile, tablet, and desktop
- User mentions: "web app", "dashboard", "SaaS", "admin panel", "data table", "keyboard shortcuts", "bulk actions"

**Don't use this skill for:**
- Marketing websites or landing pages (simpler content-focused design, not workflow-driven)
- Mobile-native apps (use `lyra/ux-designer/mobile-design-patterns`)
- Desktop-native software (use `lyra/ux-designer/desktop-software-design`)
- Simple forms or single-purpose pages (use `lyra/ux-designer/interaction-design-patterns`)

---

## The Web Application Usability Framework

A systematic 4-dimension evaluation model for web application design:

1. **DATA CLARITY** - Tables, charts, dashboards, real-time updates, export
2. **WORKFLOW EFFICIENCY** - Keyboard shortcuts, bulk actions, inline editing, command palette
3. **RESPONSIVE ADAPTATION** - Mobile <768px, tablet 768-1024px, desktop >1024px
4. **PROGRESSIVE ENHANCEMENT** - Core content works without JS/CSS, graceful degradation

Evaluate designs by examining each dimension systematically, ensuring data is understandable, workflows are fast, interfaces adapt across devices, and functionality degrades gracefully.

---

## Dimension 1: DATA CLARITY

**Purpose:** Present complex data in an understandable, scannable, actionable way

Users come to web applications to work with data—view it, understand it, act on it. Poor data presentation creates cognitive overload and slows decision-making. Clear data visualization, sortable tables, and meaningful empty states are essential.

### Evaluation Questions

1. **Can users understand data at a glance?**
   - Visual hierarchy guides the eye to important metrics
   - Charts match data type (trends → line, comparison → bar)
   - Numbers formatted appropriately (1,234 vs 1234, $1.2M vs $1,234,567)

2. **Are tables sortable and filterable?**
   - Column headers clickable for sorting (ascending/descending)
   - Filter controls above table or in column headers
   - Search functionality for large datasets
   - Pagination or infinite scroll for performance

3. **Do visualizations match data relationships?**
   - Line charts: Trends over time
   - Bar charts: Comparisons between categories
   - Pie charts: Part-to-whole (use sparingly, max 5 slices)
   - Scatter plots: Correlations between variables

4. **Are empty states helpful and actionable?**
   - Explain why empty ("No results found", "You haven't created any projects yet")
   - Provide next action (CTA: "Create your first project")
   - Show illustration or helpful tips

5. **Is real-time data clearly indicated?**
   - Visual indicator (pulsing dot, "Live" badge)
   - Timestamp ("Updated 2 seconds ago")
   - Smooth updates (no jarring content shifts)

### Tables with Sorting and Filtering

**Pattern: Sortable Table Headers**

```html
<!-- Table Structure -->
<table>
  <thead>
    <tr>
      <th>
        <button class="sort-header" data-column="name">
          Name
          <span class="sort-icon">↕</span>
        </button>
      </th>
      <th>
        <button class="sort-header" data-column="date">
          Date Created
          <span class="sort-icon">↓</span> <!-- Active: descending -->
        </button>
      </th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <!-- Data rows -->
  </tbody>
</table>
```

**Sorting States:**
- **Unsorted:** ↕ (double arrow, neutral gray)
- **Ascending:** ↑ (up arrow, brand color)
- **Descending:** ↓ (down arrow, brand color)

**Interaction:**
- Click header: Toggle sort (unsorted → ascending → descending → ascending)
- Keyboard: Tab to header, Enter/Space to sort
- Visual feedback: Icon changes, column highlighted

**Filter Controls:**

```html
<!-- Filters Above Table -->
<div class="table-filters">
  <input type="search" placeholder="Search by name..." />
  <select>
    <option value="">All Statuses</option>
    <option value="active">Active</option>
    <option value="inactive">Inactive</option>
  </select>
  <button>Clear Filters</button>
</div>
```

**Pagination:**
- Bottom of table: "Showing 1-25 of 487 results"
- Page controls: « Previous | 1 2 3 ... 20 | Next »
- Items per page: Dropdown (25, 50, 100)
- Performance: Only load visible rows (virtual scrolling for 1000+ rows)

**Row Selection for Bulk Actions:**

```html
<table>
  <thead>
    <tr>
      <th>
        <input type="checkbox" id="select-all" aria-label="Select all rows" />
      </th>
      <th>Name</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <input type="checkbox" name="row-select" value="123" />
      </td>
      <td>Project Alpha</td>
      <td>Active</td>
    </tr>
  </tbody>
</table>
```

**Selection Feedback:**
- Selected row: Background color change (light blue, 10% opacity)
- Bulk action bar appears: "3 items selected | Delete | Archive | Export"
- "Select all" checkbox states: Unchecked, Checked, Indeterminate (some selected)

### Data Visualization Best Practices

**Line Charts (Trends Over Time):**
- **Use for:** Time-series data, progress tracking, historical trends
- **Best practices:**
  - X-axis: Time (dates, hours)
  - Y-axis: Value (starts at 0 unless all values in narrow range)
  - Max 5 lines per chart (more = unreadable)
  - Legend with colors, or direct labels on lines
  - Tooltips on hover: "Aug 15, 2024: 1,234 users"

**Bar Charts (Comparisons Between Categories):**
- **Use for:** Comparing quantities across categories
- **Best practices:**
  - Horizontal bars: Better for long category names
  - Vertical bars: Better for time-based categories (months)
  - Sort by value (descending) unless chronological
  - Space between bars: 20-40% of bar width
  - Value labels on bars or at end

**Pie Charts (Part-to-Whole Relationships):**
- **Use sparingly:** Only when showing parts of 100%
- **Max 5 slices:** More slices = hard to compare
- **Alternative:** Donut chart (center shows total)
- **Better alternative:** Horizontal bar chart (easier to compare)
- **Labels:** Percentage + category name

**Dashboard Hierarchy (F-Pattern Layout):**

```
┌─────────────────────────────────────────────────┐
│  [Most Important Metric]   [Primary Action]     │  ← Top row: Critical
├──────────────────┬──────────────────┬───────────┤
│  Metric Card     │  Metric Card     │  Chart    │  ← Second row: Important
├──────────────────┴──────────────────┴───────────┤
│                                                  │
│  Detailed Table or Secondary Chart              │  ← Below fold: Details
│                                                  │
└─────────────────────────────────────────────────┘
```

**F-Pattern Scanning:**
- **Top-left:** Most important metric (users scan here first)
- **Top-right:** Primary action (CTA, create, export)
- **Left edge:** Vertical scan (labels, categories)
- **Horizontal scans:** Decreasing attention as user scrolls down

**Metric Cards:**
- Value (large, bold): "1,234"
- Label (smaller, gray): "Active Users"
- Change indicator: "↑ 12% vs last week" (green for positive, red for negative)
- Sparkline (optional): Tiny trend chart (last 7 days)

### Progressive Disclosure for Complex Data

**Pattern: Summary Cards → Detail Views**

```html
<!-- Summary Card (Collapsed) -->
<div class="summary-card">
  <h3>Project Alpha</h3>
  <p>Status: Active | 12 tasks remaining</p>
  <button class="expand-btn">View Details →</button>
</div>

<!-- Detail View (Expanded) -->
<div class="detail-view">
  <h3>Project Alpha</h3>
  <div class="metadata">
    <p>Created: Aug 1, 2024</p>
    <p>Owner: Jane Smith</p>
    <p>Budget: $50,000</p>
  </div>
  <table>
    <!-- Detailed task list -->
  </table>
  <button class="collapse-btn">Hide Details ↑</button>
</div>
```

**When to Use:**
- Large datasets: Show 5-10 items, "Load more" or pagination
- Complex records: Show summary, expand for full details
- Nested data: Accordion or tree view

**Interaction:**
- Click to expand: Smooth height animation (300ms ease-out)
- Keyboard: Tab to button, Enter/Space to toggle
- Icon: ▶ (collapsed) → ▼ (expanded)

### Real-Time Updates

**WebSocket for Live Data:**
- **Use for:** Stock prices, live dashboards, chat, collaborative editing
- **Pattern:** Establish WebSocket connection on page load, update DOM on message
- **Visual indicator:** "Live" badge (pulsing green dot), "Connected" status

**Polling for Updates:**
- **Use for:** Less time-critical updates (notifications, status checks)
- **Interval:** 60 seconds typical (balance freshness vs server load)
- **Pattern:** setInterval(fetchUpdates, 60000)

**Visual Indicators:**
- **Pulsing dot:** Live connection (CSS animation, 2s pulse)
- **Timestamp:** "Updated 2 seconds ago" (relative time, updates every second)
- **Flash on update:** Brief highlight (yellow background, fades after 2s)

**Smooth Content Updates:**
- Don't shift layout abruptly (causes disorientation)
- Animate new items in (slide down or fade in, 200ms)
- If sorting changes, animate reordering (swap positions smoothly)

### Export Functionality

**Export Options:**

**CSV for Data Analysis:**
- All rows or filtered/selected rows
- All columns or visible columns only
- Filename: "projects_export_2024-08-15.csv"

**PDF for Reports:**
- Formatted for printing (headers, footers, page numbers)
- Charts rendered as images
- Logo and branding

**Copy to Clipboard:**
- Selected cells or entire table
- Tab-separated values (paste into Excel/Sheets)
- Confirmation toast: "Copied 10 rows to clipboard"

**Pattern: Export Dropdown**

```html
<div class="export-dropdown">
  <button>Export ▼</button>
  <ul class="dropdown-menu">
    <li><button>Export to CSV</button></li>
    <li><button>Export to PDF</button></li>
    <li><button>Copy to Clipboard</button></li>
  </ul>
</div>
```

---

## Dimension 2: WORKFLOW EFFICIENCY

**Purpose:** Enable power users to work at maximum speed with minimal friction

Web application users are often power users who perform repetitive tasks daily. Keyboard shortcuts, bulk actions, inline editing, and command palettes dramatically improve efficiency for these users.

### Evaluation Questions

1. **Can power users complete tasks without touching the mouse?**
   - Keyboard shortcuts for common actions
   - Tab navigation through all interactive elements
   - Focus indicators visible and high-contrast

2. **Are bulk actions available for repetitive tasks?**
   - Select multiple items (checkboxes)
   - "Select all" option
   - Bulk actions: Delete, Archive, Export, Tag, Assign

3. **Does inline editing reduce modal dialogs?**
   - Click to edit cells in tables
   - Auto-save on blur
   - No modal for simple edits

4. **Is there an autosave mechanism?**
   - Draft saved every 30 seconds
   - Visual indicator: "Saving...", "All changes saved"
   - No "lost work" anxiety

5. **Are contextual actions easily accessible?**
   - Right-click context menus
   - Hover actions on table rows
   - Quick actions on cards

### Keyboard Shortcuts

**Essential Shortcuts (Cross-Platform):**

```
Cmd/Ctrl+S: Save
Cmd/Ctrl+F: Search/Find (open search modal or focus search input)
Cmd/Ctrl+Z: Undo
Cmd/Ctrl+Shift+Z: Redo
Cmd/Ctrl+K: Command palette (universal action search)
Cmd/Ctrl+/: Show keyboard shortcuts panel
```

**Navigation Shortcuts:**

```
Tab: Next field/element
Shift+Tab: Previous field/element
Arrow keys: Navigate lists, tables, calendar cells
Enter: Submit form, activate button, open selected item
Esc: Close modal, cancel edit, clear search
```

**Table Navigation:**

```
Arrow Up/Down: Previous/next row
Arrow Left/Right: Previous/next column (if cell-focused)
Space: Select/deselect row (checkbox)
Enter: Open detail view for selected row
```

**Application-Specific Shortcuts:**

```
G then I: Go to Inbox (Gmail-style navigation)
C: Compose new (email, message, post)
/: Focus search (common in many apps)
?: Show keyboard shortcuts help
```

**Discoverability Strategies:**

1. **Tooltips with Shortcuts:**
   - Button tooltip: "Save (Cmd+S)"
   - Show shortcut in gray, smaller text

2. **Onboarding Hints:**
   - First-time user: "Pro tip: Press Cmd+K to quickly find anything"
   - Dismissable, don't show again after 3 views

3. **Keyboard Shortcuts Panel:**
   - Triggered by Cmd+/ or ? key
   - Modal with categorized shortcuts (Navigation, Editing, Actions)
   - Searchable

**Implementation:**

```javascript
// Global keyboard listener
document.addEventListener('keydown', (e) => {
  // Cmd+S or Ctrl+S: Save
  if ((e.metaKey || e.ctrlKey) && e.key === 's') {
    e.preventDefault();
    saveDocument();
  }

  // Cmd+K or Ctrl+K: Command palette
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault();
    openCommandPalette();
  }
});
```

### Bulk Actions

**Pattern: Select Multiple Items, Act on All**

```html
<!-- Table with Bulk Selection -->
<table>
  <thead>
    <tr>
      <th>
        <input type="checkbox" id="select-all" />
      </th>
      <th>Name</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><input type="checkbox" name="item" value="1" /></td>
      <td>Item 1</td>
      <td>Active</td>
    </tr>
    <tr>
      <td><input type="checkbox" name="item" value="2" /></td>
      <td>Item 2</td>
      <td>Inactive</td>
    </tr>
  </tbody>
</table>

<!-- Bulk Action Bar (appears when items selected) -->
<div class="bulk-action-bar" hidden>
  <span class="selection-count">3 items selected</span>
  <button class="bulk-delete">Delete</button>
  <button class="bulk-archive">Archive</button>
  <button class="bulk-export">Export</button>
  <button class="bulk-tag">Add Tag</button>
  <button class="deselect-all">Clear Selection</button>
</div>
```

**Bulk Action Behavior:**

1. **Select items:** Click checkboxes or Shift+Click for range selection
2. **Bulk action bar appears:** Slides in from top or bottom
3. **Click action:** Confirmation modal for destructive actions (delete)
4. **Process:** Show progress if >10 items ("Deleting 47 items... 23/47")
5. **Feedback:** Toast notification "3 items deleted" or "Export complete"
6. **Undo option:** "3 items deleted. Undo?" (5-second window)

**"Select All" Checkbox States:**
- **Unchecked:** No items selected
- **Checked:** All visible items selected
- **Indeterminate:** Some items selected (shows dash/minus icon)

**Selection Shortcuts:**
- Cmd+A: Select all (focus table first)
- Shift+Click: Select range (click first, Shift+Click last)
- Cmd+Click: Toggle individual selection (add/remove from selection)

### Inline Editing

**Pattern: Click to Edit, Auto-Save on Blur**

```html
<!-- Table Cell (View Mode) -->
<td class="editable-cell" data-field="name" data-id="123">
  <span class="cell-value">Project Alpha</span>
  <button class="edit-icon" aria-label="Edit name">✏️</button>
</td>

<!-- Table Cell (Edit Mode) -->
<td class="editable-cell editing">
  <input type="text" value="Project Alpha" class="cell-input" />
  <button class="save-btn">✓</button>
  <button class="cancel-btn">✕</button>
</td>
```

**Inline Edit Interaction:**

1. **Enter edit mode:**
   - Click cell or edit icon
   - Double-click cell
   - Keyboard: Tab to cell, Enter to edit

2. **Edit value:**
   - Input field focused, text selected
   - Type new value
   - Validation inline (if applicable)

3. **Save:**
   - Click save button
   - Press Enter
   - Blur (click outside) triggers auto-save
   - API request in background

4. **Cancel:**
   - Click cancel button
   - Press Esc
   - Revert to original value

5. **Feedback:**
   - Saving: Spinner or "Saving..." text
   - Success: Brief green border flash, "Saved"
   - Error: Red border, error message, revert to original

**Auto-Save Indicators:**
- Icon: ⏳ (saving), ✓ (saved), ⚠ (error)
- Text: "Saving...", "All changes saved", "Failed to save"
- Position: Top-right corner or inline with edited field

**When NOT to Use Inline Editing:**
- Complex fields (multi-line text, rich text editor → use modal or dedicated page)
- Multiple related fields (edit form in modal or expand row)
- Destructive actions (delete → confirmation modal)

### Command Palette

**Pattern: Cmd+K Opens Search for All Actions**

```html
<!-- Command Palette Modal -->
<div class="command-palette-modal">
  <input
    type="search"
    placeholder="Search for actions, pages, or data..."
    class="command-search"
    autofocus
  />
  <ul class="command-results">
    <li class="result-item">
      <span class="result-icon">📄</span>
      <span class="result-title">Create New Project</span>
      <span class="result-shortcut">Cmd+N</span>
    </li>
    <li class="result-item">
      <span class="result-icon">🔍</span>
      <span class="result-title">Search Projects</span>
      <span class="result-shortcut">Cmd+F</span>
    </li>
    <li class="result-item selected">
      <span class="result-icon">⚙️</span>
      <span class="result-title">Settings</span>
    </li>
  </ul>
</div>
```

**Command Palette Features:**

- **Search:** Type to filter actions, pages, recent items
- **Keyboard navigation:** Arrow keys to select, Enter to execute
- **Shortcuts shown:** Display keyboard shortcut if available
- **Recents:** Show recently used actions at top
- **Categories:** Actions, Pages, Projects, Settings
- **Fuzzy search:** "crnw" matches "Create New Project"

**What to Include:**
- Navigation: All pages/sections
- Actions: Create, Export, Delete, Settings
- Recent items: Recently viewed projects, documents
- Search results: Data from tables, lists

**Examples:**
- Slack: Cmd+K (Quick switcher - channels, DMs, files)
- VS Code: Cmd+Shift+P (Command palette - all editor actions)
- Linear: Cmd+K (Create issue, search, navigate)

### Contextual Actions

**Pattern: Right-Click Context Menu**

```html
<!-- Table Row -->
<tr class="table-row" data-id="123">
  <td>Project Alpha</td>
  <td>Active</td>
  <td>
    <!-- Optional: Three-dot menu for touch/mobile -->
    <button class="row-menu-btn" aria-label="More actions">⋮</button>
  </td>
</tr>

<!-- Context Menu (appears on right-click) -->
<ul class="context-menu" style="top: 200px; left: 300px;">
  <li><button>Open</button></li>
  <li><button>Edit</button></li>
  <li><button>Duplicate</button></li>
  <li class="divider"></li>
  <li><button class="danger">Delete</button></li>
</ul>
```

**Context Menu Behavior:**

- **Trigger:** Right-click on row, card, or item
- **Position:** Appears at mouse cursor position
- **Keyboard:** Shift+F10 or Menu key (if available)
- **Close:** Click outside, press Esc, select action
- **Actions:** Open, Edit, Duplicate, Archive, Delete (dangerous actions at bottom, red)

**Touch/Mobile Alternative:**
- Three-dot menu button (⋮) visible on mobile
- Tap to show action dropdown or bottom sheet
- Right-click not available on touch devices

**Hover Actions (Desktop):**

```html
<!-- Card with Hover Actions -->
<div class="card">
  <h3>Project Alpha</h3>
  <p>12 tasks remaining</p>

  <!-- Actions appear on hover (desktop only) -->
  <div class="hover-actions">
    <button aria-label="Edit">✏️</button>
    <button aria-label="Archive">📦</button>
    <button aria-label="Delete">🗑️</button>
  </div>
</div>
```

**CSS for Hover Actions:**

```css
.hover-actions {
  opacity: 0;
  transition: opacity 150ms ease-out;
}

.card:hover .hover-actions {
  opacity: 1;
}

.card:focus-within .hover-actions {
  opacity: 1; /* Also show on keyboard focus */
}
```

### Undo/Redo

**Implementation Guidance:**

**Keyboard Shortcuts:**
- Cmd+Z / Ctrl+Z: Undo
- Cmd+Shift+Z / Ctrl+Shift+Z: Redo (or Cmd+Y / Ctrl+Y)

**State Management:**
- Maintain history stack of actions
- Max history: 20-50 actions (memory constraint)
- Persist across page refresh (localStorage)

**Visual Feedback:**
- Toast notification: "Undo: Deleted 3 items"
- Menu items: "Undo Delete" (shows last action)
- Disabled state: Gray out if no undo/redo available

**What Actions to Track:**
- Destructive: Delete, archive, remove
- Edits: Text changes, status updates
- Bulk actions: All bulk operations
- Don't track: Navigation, search, filter (non-destructive)

**Undo Toast Pattern:**

```html
<!-- Toast with Undo -->
<div class="toast success">
  <span>3 items deleted</span>
  <button class="undo-btn">Undo</button>
</div>
```

- Duration: 5 seconds (enough time to undo)
- Auto-dismiss: After 5s (or user clicks undo)
- Position: Bottom-center or top-right

### Autosave

**Pattern: Save Draft Every 30 Seconds**

```javascript
let autosaveTimer;
let isDirty = false; // Track if content changed

// Mark as dirty on user input
contentField.addEventListener('input', () => {
  isDirty = true;
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(autosave, 30000); // 30 seconds
});

async function autosave() {
  if (!isDirty) return;

  updateStatus('Saving...');
  try {
    await saveDraft();
    updateStatus('All changes saved');
    isDirty = false;
  } catch (error) {
    updateStatus('Failed to save. Retry in 60s.');
    autosaveTimer = setTimeout(autosave, 60000); // Retry
  }
}
```

**Autosave Indicators:**

- **Saving:** "Saving..." (gray, spinner icon)
- **Saved:** "All changes saved" (green, checkmark, fades after 2s)
- **Error:** "Failed to save. Trying again..." (red, warning icon)
- **Position:** Top-right, below header, or inline with content

**Best Practices:**
- Save after 30 seconds of inactivity (not every keystroke)
- Save on blur (user navigates away)
- Debounce: Wait for user to stop typing
- Conflict resolution: Server timestamp wins, or show merge UI

**Manual Save:**
- Still provide "Save" button (Cmd+S)
- Immediate save, not delayed
- Feedback: "Saved successfully"

---

## Dimension 3: RESPONSIVE ADAPTATION

**Purpose:** Provide optimal experience across all screen sizes (mobile, tablet, desktop)

Web applications must work on phones, tablets, and desktops. Layout, navigation, and interaction patterns must adapt. Mobile users need simplified views and touch targets; desktop users need dense information and keyboard shortcuts.

### Evaluation Questions

1. **Does the layout adapt gracefully from mobile to desktop?**
   - Mobile: Single column, simplified
   - Tablet: Two columns, condensed
   - Desktop: Multi-column, full features

2. **Are touch targets appropriately sized on mobile?**
   - Mobile: 48x48px minimum
   - Desktop: 32x32px acceptable (mouse precision)

3. **Do complex features have mobile alternatives?**
   - Desktop: Hover actions, right-click menus
   - Mobile: Tap menus, swipe actions, bottom sheets

4. **Is the mobile experience functional (not just hidden)?**
   - Don't hide critical features behind mobile breakpoint
   - Simplify, don't remove
   - Provide alternative interactions

### Breakpoints

**Standard Breakpoints:**

```css
/* Mobile: <768px */
@media (max-width: 767px) {
  /* Single column, stacked navigation, simplified tables */
}

/* Tablet: 768-1024px */
@media (min-width: 768px) and (max-width: 1024px) {
  /* Two columns, condensed nav, partial tables */
}

/* Desktop: >1024px */
@media (min-width: 1025px) {
  /* Multi-column, persistent nav, full features */
}
```

### Mobile (<768px)

**Layout:**
- **Single column:** Content stacks vertically
- **Full-width:** Cards, buttons, inputs span full width (minus padding)
- **Padding:** 16px left/right margins

**Navigation:**
- **Hamburger menu:** Three-line icon, opens drawer navigation
- **Bottom tabs:** 3-5 items for primary navigation (if app-like)
- **Sticky header:** Fixed top navigation (collapses on scroll optional)

**Tables:**
- **Option 1: Horizontal scroll** (preserve structure, scroll left/right)
- **Option 2: Hide columns** (show critical columns only, "View details" for rest)
- **Option 3: Card view** (stack data vertically, each row becomes card)
- **Option 4: Accordion rows** (tap to expand full details)

**Touch Targets:**
- **Minimum:** 48x48px (finger-sized)
- **Spacing:** 8px between targets
- **Buttons:** Full-width or large (minimum 48px height)

**Information Density:**
- **Reduced:** Show essential data, hide secondary info
- **Progressive disclosure:** "Show more" buttons
- **Simplified charts:** Single metric cards instead of complex multi-line charts

**Mobile-Specific Patterns:**
- Tap instead of hover (no hover state on mobile)
- Swipe gestures (swipe to delete, pull to refresh)
- Bottom sheets instead of modals (easier thumb reach)

### Tablet (768-1024px)

**Layout:**
- **Two-column:** Sidebar + content, or split view
- **Grid:** 2-3 columns for cards
- **Flexible:** Adapt to portrait (narrower) vs landscape (wider)

**Navigation:**
- **Condensed sidebar:** Icons + labels (collapsed width)
- **Icon nav:** Top nav with icons and text
- **Tabs:** Horizontal tabs for sections

**Tables:**
- **Partial columns:** Show more columns than mobile, hide non-critical
- **Horizontal scroll:** If all columns needed
- **Responsive breakpoints:** Adjust column visibility at 768px and 1024px

**Touch Targets:**
- **Still touch-friendly:** 48x48px (tablets are touch)
- **Hybrid:** Support touch and mouse (some tablets have trackpads)

**Information Density:**
- **Medium:** More than mobile, less than desktop
- **Two-column cards:** Side-by-side instead of stacked

### Desktop (>1024px)

**Layout:**
- **Multi-column:** 3+ columns, sidebars, panels
- **Persistent navigation:** Sidebar always visible, or top nav
- **Wider content:** Max-width 1400-1600px, centered

**Navigation:**
- **Full sidebar:** Expanded with icons + labels
- **Top nav with dropdowns:** Mega-menus for complex hierarchies
- **Breadcrumbs:** Show navigation path

**Tables:**
- **All columns visible:** Full table with all data
- **Sortable:** Column headers clickable
- **Hover states:** Row highlights on hover, actions appear

**Mouse/Keyboard Interactions:**
- **Hover:** Tooltips, action buttons, previews
- **Right-click:** Context menus
- **Keyboard shortcuts:** Full keyboard support
- **Focus indicators:** Visible for keyboard navigation

**Information Density:**
- **High:** Show all data, charts, metrics
- **Multi-pane:** Master-detail views (list + detail side-by-side)
- **Dense tables:** More rows per page (25-50)

**Desktop-Specific Patterns:**
- Hover to reveal actions
- Drag-and-drop reordering
- Resizable panels (drag divider to adjust width)
- Multi-select with Shift+Click, Cmd+Click

### Responsive Patterns

**Mobile-First CSS:**

```css
/* Base styles (mobile <768px) */
.container {
  padding: 16px;
}

.card {
  width: 100%;
  margin-bottom: 16px;
}

.sidebar {
  display: none; /* Hidden on mobile, hamburger menu instead */
}

/* Tablet (768px+) */
@media (min-width: 768px) {
  .container {
    display: grid;
    grid-template-columns: 200px 1fr; /* Sidebar + content */
    padding: 24px;
  }

  .sidebar {
    display: block; /* Show sidebar */
  }

  .card {
    width: calc(50% - 12px); /* Two columns */
  }
}

/* Desktop (1025px+) */
@media (min-width: 1025px) {
  .container {
    grid-template-columns: 250px 1fr 300px; /* Sidebar + content + right panel */
    max-width: 1600px;
    margin: 0 auto;
  }

  .card {
    width: calc(33.33% - 16px); /* Three columns */
  }
}
```

### Responsive Navigation

**Mobile: Hamburger Menu → Drawer**

```html
<!-- Mobile Navigation -->
<header class="mobile-header">
  <button class="hamburger-btn" aria-label="Open menu">☰</button>
  <h1>App Name</h1>
</header>

<nav class="drawer-nav" hidden>
  <ul>
    <li><a href="/dashboard">Dashboard</a></li>
    <li><a href="/projects">Projects</a></li>
    <li><a href="/settings">Settings</a></li>
  </ul>
</nav>
```

**Tablet: Icon Nav with Labels**

```html
<!-- Tablet Navigation (Condensed Sidebar) -->
<nav class="sidebar-nav condensed">
  <ul>
    <li>
      <a href="/dashboard">
        <span class="icon">📊</span>
        <span class="label">Dashboard</span>
      </a>
    </li>
    <li>
      <a href="/projects">
        <span class="icon">📁</span>
        <span class="label">Projects</span>
      </a>
    </li>
  </ul>
</nav>
```

**Desktop: Full Sidebar**

```html
<!-- Desktop Navigation (Full Sidebar) -->
<nav class="sidebar-nav full">
  <ul>
    <li class="nav-item active">
      <a href="/dashboard">
        <span class="icon">📊</span>
        <span class="label">Dashboard</span>
        <span class="badge">3</span>
      </a>
    </li>
    <li class="nav-item">
      <a href="/projects">
        <span class="icon">📁</span>
        <span class="label">Projects</span>
      </a>
    </li>
  </ul>
</nav>
```

### Responsive Tables

**Option 1: Horizontal Scroll (Preserve Structure)**

```css
/* Mobile: Scroll horizontally to see all columns */
.table-container {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch; /* Smooth scrolling on iOS */
}

table {
  min-width: 600px; /* Force horizontal scroll */
}
```

**Option 2: Hide Columns (Show Critical Only)**

```css
/* Mobile: Hide non-critical columns */
@media (max-width: 767px) {
  .column-created-date,
  .column-last-updated {
    display: none;
  }
}

/* Desktop: Show all columns */
@media (min-width: 1025px) {
  .column-created-date,
  .column-last-updated {
    display: table-cell;
  }
}
```

**Option 3: Card View (Stack Data Vertically)**

```html
<!-- Mobile: Cards instead of table rows -->
<div class="card" data-id="123">
  <h3>Project Alpha</h3>
  <p>Status: <span class="badge">Active</span></p>
  <p>Created: Aug 1, 2024</p>
  <button>View Details</button>
</div>

<!-- Desktop: Traditional table -->
<table>
  <tr>
    <td>Project Alpha</td>
    <td><span class="badge">Active</span></td>
    <td>Aug 1, 2024</td>
    <td><button>View Details</button></td>
  </tr>
</table>
```

**Option 4: Accordion Rows (Tap to Expand)**

```html
<!-- Mobile: Tap row to expand details -->
<div class="table-row collapsed">
  <div class="row-summary">
    <span class="name">Project Alpha</span>
    <span class="status">Active</span>
    <button class="expand-btn">▼</button>
  </div>
  <div class="row-details" hidden>
    <p>Created: Aug 1, 2024</p>
    <p>Owner: Jane Smith</p>
    <p>Budget: $50,000</p>
  </div>
</div>
```

### Responsive Charts

**Mobile: Simplified Charts**

- **Single metric:** Large number with sparkline
- **Simple bar chart:** 3-5 bars maximum
- **Avoid:** Multi-line charts (hard to read on small screen)

**Desktop: Full Detail Charts**

- **Multi-line charts:** Up to 5 lines
- **Complex visualizations:** Scatter plots, heat maps
- **Tooltips:** Hover for detailed data

**Responsive Chart Pattern:**

```css
/* Mobile: Smaller height, simplified */
.chart-container {
  height: 200px;
}

/* Desktop: Larger, more detail */
@media (min-width: 1025px) {
  .chart-container {
    height: 400px;
  }
}
```

### Touch vs Hover

**Mobile (Touch):**
- **No hover states:** All affordances must be visible
- **Tap actions:** Single tap to activate
- **Long press:** Context menu (500ms hold)
- **Swipe:** Swipe to delete, swipe to reveal actions
- **No right-click:** Use long-press or visible menu button

**Desktop (Mouse + Hover):**
- **Hover states:** Actions appear on hover, tooltips, previews
- **Click:** Left-click to activate
- **Right-click:** Context menu
- **Cursor changes:** Pointer for links, grab for draggable
- **Keyboard focus:** Visible focus indicators

**Responsive Hover Pattern:**

```css
/* Desktop: Show actions on hover */
@media (min-width: 1025px) {
  .card-actions {
    opacity: 0;
    transition: opacity 150ms;
  }

  .card:hover .card-actions {
    opacity: 1;
  }
}

/* Mobile: Always show actions (no hover) */
@media (max-width: 767px) {
  .card-actions {
    opacity: 1; /* Always visible */
  }
}
```

---

## Dimension 4: PROGRESSIVE ENHANCEMENT

**Purpose:** Ensure core functionality works in all conditions (slow networks, JavaScript disabled, older browsers)

Users access web apps under varying conditions: slow 3G networks, JavaScript blocked by corporate firewalls, older browsers. Progressive enhancement ensures core content and actions work even when conditions are degraded.

### Evaluation Questions

1. **Does content load if JavaScript fails?**
   - Core HTML content visible
   - Server-side rendering for critical content
   - Graceful degradation for interactive features

2. **Are core actions available without JavaScript?**
   - Forms submit to server (full page reload)
   - Links work (not JavaScript-only routing)
   - No "JavaScript required" errors

3. **Does the page render in a usable state without CSS?**
   - Semantic HTML structure
   - Logical source order (header → nav → content → footer)
   - Text-based fallbacks for icons

4. **Is there graceful degradation for older browsers?**
   - Feature detection (not browser detection)
   - Polyfills for missing features
   - Fallbacks for unsupported CSS

### Enhancement Layers

**Layer 1: Core HTML (Works Everywhere)**

Semantic HTML with forms, links, and headings. Content is accessible and functional with zero JavaScript or CSS.

```html
<!-- Core HTML: Form submits to server -->
<form action="/projects/create" method="POST">
  <label for="project-name">Project Name</label>
  <input type="text" id="project-name" name="name" required />

  <label for="project-status">Status</label>
  <select id="project-status" name="status">
    <option value="active">Active</option>
    <option value="inactive">Inactive</option>
  </select>

  <button type="submit">Create Project</button>
</form>

<!-- Core HTML: Links work without JavaScript -->
<nav>
  <a href="/dashboard">Dashboard</a>
  <a href="/projects">Projects</a>
  <a href="/settings">Settings</a>
</nav>
```

**Layer 2: CSS Enhancement (Visual Hierarchy)**

Add visual hierarchy, layout, spacing, and responsive design. Page looks professional and organized.

```css
/* CSS: Visual hierarchy */
.form-field {
  margin-bottom: 16px;
}

label {
  display: block;
  font-weight: 600;
  margin-bottom: 4px;
}

input, select {
  width: 100%;
  padding: 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 16px;
}

button {
  background: #0066cc;
  color: white;
  padding: 12px 24px;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;
}

button:hover {
  background: #0052a3;
}
```

**Layer 3: JavaScript Optimization (Faster Feedback)**

Add client-side validation, AJAX requests (no page reload), smooth animations, and interactive features.

```javascript
// JavaScript: Client-side validation (faster feedback)
form.addEventListener('submit', async (e) => {
  e.preventDefault(); // Prevent full page reload

  // Validate
  if (!nameInput.value) {
    showError('Project name is required');
    return;
  }

  // AJAX submit
  const response = await fetch('/projects/create', {
    method: 'POST',
    body: new FormData(form)
  });

  if (response.ok) {
    showSuccess('Project created!');
    // Smooth redirect or update UI
  } else {
    showError('Failed to create project. Try again.');
  }
});
```

**Progressive Enhancement Strategy:**

1. Build core HTML (works without JS/CSS)
2. Add CSS (visual hierarchy, responsive)
3. Add JavaScript (better UX, faster feedback)
4. Test with JS disabled, CSS disabled, slow network

### Forms Without JavaScript

**Pattern: Server-Side Validation, Full Page Submission**

```html
<!-- Form submits to server, no JavaScript required -->
<form action="/projects/create" method="POST">
  <!-- Server renders error messages if validation fails -->
  <?php if (isset($errors['name'])): ?>
    <div class="error-message"><?= $errors['name'] ?></div>
  <?php endif; ?>

  <label for="name">Project Name</label>
  <input
    type="text"
    id="name"
    name="name"
    value="<?= htmlspecialchars($submitted_name ?? '') ?>"
    required
  />

  <button type="submit">Create Project</button>
</form>
```

**Server-Side Flow:**
1. User submits form
2. Server validates input
3. If errors: Render form with error messages, pre-fill submitted values
4. If success: Redirect to success page or show confirmation

**Enhancement with JavaScript:**
- Client-side validation (instant feedback, no round-trip)
- AJAX submission (no page reload)
- Loading spinner (better perceived performance)

### Content-First Loading

**Priority: Above-Fold Content → Critical Resources → Non-Critical Assets**

```html
<!-- Critical inline CSS for above-fold content -->
<head>
  <style>
    /* Inline critical CSS (above-fold styles) */
    body { font-family: sans-serif; margin: 0; }
    .header { background: #333; color: white; padding: 16px; }
    .main-content { max-width: 1200px; margin: 0 auto; padding: 16px; }
  </style>

  <!-- Defer non-critical CSS -->
  <link rel="preload" href="/styles/main.css" as="style" onload="this.onload=null;this.rel='stylesheet'" />
  <noscript><link rel="stylesheet" href="/styles/main.css" /></noscript>
</head>

<body>
  <!-- Above-fold content loads first -->
  <header class="header">
    <h1>Dashboard</h1>
  </header>

  <main class="main-content">
    <h2>Your Projects</h2>
    <!-- Critical content here -->
  </main>

  <!-- Defer non-critical JavaScript -->
  <script src="/scripts/main.js" defer></script>
</body>
```

**Loading Priority:**

1. **HTML (0ms):** Structure and content
2. **Critical CSS (inline, 0ms):** Above-fold styles
3. **Critical JS (deferred):** Loads after HTML parsed
4. **Non-critical CSS (lazy, 1-2s):** Below-fold styles
5. **Non-critical assets (3-5s):** Analytics, ads, social widgets

### Graceful Degradation

**Pattern: Feature Detection, Fallbacks for Missing Capabilities**

```javascript
// Feature detection (not browser detection)
if ('IntersectionObserver' in window) {
  // Use IntersectionObserver for lazy loading
  const observer = new IntersectionObserver(lazyLoad);
  images.forEach(img => observer.observe(img));
} else {
  // Fallback: Load all images immediately
  images.forEach(img => {
    img.src = img.dataset.src;
  });
}

// CSS Grid fallback
@supports (display: grid) {
  .container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
  }
}

@supports not (display: grid) {
  .container {
    display: flex;
    flex-wrap: wrap;
  }

  .container > * {
    flex: 0 0 calc(33.33% - 16px);
  }
}
```

**Polyfills for Older Browsers:**

```javascript
// Conditionally load polyfills
if (!('fetch' in window)) {
  // Load fetch polyfill for IE11
  loadScript('/polyfills/fetch.js');
}

if (!('Promise' in window)) {
  // Load Promise polyfill
  loadScript('/polyfills/promise.js');
}
```

**No-JavaScript Fallback:**

```html
<noscript>
  <div class="noscript-warning">
    <p>JavaScript is disabled. Some features may not work.</p>
    <p>Core functionality is still available.</p>
  </div>
</noscript>
```

### Performance Budgets

**Target Metrics (Google Lighthouse):**

- **First Contentful Paint (FCP):** <1.8 seconds
- **Largest Contentful Paint (LCP):** <2.5 seconds
- **Time to Interactive (TTI):** <3.8 seconds
- **Total Blocking Time (TBT):** <300ms
- **Cumulative Layout Shift (CLS):** <0.1

**Bundle Size Budgets:**

- **Total JavaScript:** <200KB gzipped
- **Total CSS:** <50KB gzipped
- **Critical CSS (inline):** <14KB (fits in first TCP packet)
- **Images:** WebP format, responsive sizes, lazy loading

**Optimization Strategies:**

- Code splitting (load only what's needed)
- Tree shaking (remove unused code)
- Minification and compression (gzip, Brotli)
- CDN for static assets
- Browser caching (long cache headers for versioned assets)

---

## Web-Specific Design Patterns

### Navigation Patterns

**Persistent Top Nav:**

```html
<nav class="top-nav">
  <div class="logo">App Name</div>
  <ul class="nav-links">
    <li><a href="/dashboard">Dashboard</a></li>
    <li><a href="/projects">Projects</a></li>
    <li><a href="/reports">Reports</a></li>
  </ul>
  <div class="user-menu">
    <button>User Name ▼</button>
  </div>
</nav>
```

**Sidebar Nav:**

```html
<aside class="sidebar">
  <nav>
    <ul>
      <li class="active"><a href="/dashboard">📊 Dashboard</a></li>
      <li><a href="/projects">📁 Projects</a></li>
      <li><a href="/settings">⚙️ Settings</a></li>
    </ul>
  </nav>
</aside>
```

**Breadcrumbs:**

```html
<nav class="breadcrumbs" aria-label="Breadcrumb">
  <ol>
    <li><a href="/">Home</a></li>
    <li><a href="/projects">Projects</a></li>
    <li aria-current="page">Project Alpha</li>
  </ol>
</nav>
```

**Mega-Menu (Complex Hierarchies):**

```html
<nav class="mega-menu">
  <button class="menu-trigger">Products ▼</button>
  <div class="mega-menu-panel">
    <div class="menu-column">
      <h3>Category 1</h3>
      <ul>
        <li><a href="/product-a">Product A</a></li>
        <li><a href="/product-b">Product B</a></li>
      </ul>
    </div>
    <div class="menu-column">
      <h3>Category 2</h3>
      <ul>
        <li><a href="/product-c">Product C</a></li>
        <li><a href="/product-d">Product D</a></li>
      </ul>
    </div>
  </div>
</nav>
```

### Feedback Mechanisms

**Toast Notifications:**

```html
<!-- Toast: Brief, auto-dismiss message -->
<div class="toast success">
  <span class="icon">✓</span>
  <span class="message">Project created successfully</span>
  <button class="close-btn" aria-label="Dismiss">✕</button>
</div>
```

**Position:** Top-right or bottom-center
**Duration:** 3-5 seconds, auto-dismiss
**Types:** Success (green), Error (red), Warning (yellow), Info (blue)

**Inline Validation:**

```html
<!-- Error state -->
<div class="form-field error">
  <label for="email">Email</label>
  <input type="email" id="email" value="invalid-email" aria-invalid="true" />
  <span class="error-message">Please enter a valid email address</span>
</div>

<!-- Success state -->
<div class="form-field success">
  <label for="username">Username</label>
  <input type="text" id="username" value="johndoe" />
  <span class="success-message">✓ Username available</span>
</div>
```

**Loading States:**

```html
<!-- Button loading state -->
<button class="btn loading" disabled>
  <span class="spinner"></span>
  Saving...
</button>

<!-- Skeleton screen (loading placeholder) -->
<div class="skeleton-card">
  <div class="skeleton-header"></div>
  <div class="skeleton-text"></div>
  <div class="skeleton-text short"></div>
</div>
```

**Error Pages:**

```html
<!-- 404 Error Page -->
<div class="error-page">
  <h1>404</h1>
  <h2>Page Not Found</h2>
  <p>The page you're looking for doesn't exist or has been moved.</p>
  <a href="/dashboard" class="btn primary">Go to Dashboard</a>
</div>

<!-- 500 Error Page -->
<div class="error-page">
  <h1>500</h1>
  <h2>Something Went Wrong</h2>
  <p>We're working to fix the issue. Please try again later.</p>
  <button onclick="location.reload()">Retry</button>
</div>
```

### Layout Patterns

**F-Pattern Scanning:**

Users scan web pages in an F-shaped pattern: horizontal scan at top, second horizontal scan lower, then vertical scan on left.

**Design Implications:**
- **Top-left:** Logo, app name
- **Top-right:** Primary action, user menu
- **Left edge:** Navigation, filters
- **Top horizontal:** Important metrics, key actions

**Fixed Headers:**

```css
.header {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  background: white;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.main-content {
  margin-top: 64px; /* Header height */
}
```

**Sticky CTAs (Call-to-Action):**

```css
.sticky-cta {
  position: sticky;
  bottom: 0;
  background: white;
  padding: 16px;
  box-shadow: 0 -2px 8px rgba(0,0,0,0.1);
  z-index: 100;
}
```

**Modals:**

```html
<!-- Modal Dialog -->
<div class="modal-overlay">
  <div class="modal" role="dialog" aria-labelledby="modal-title">
    <div class="modal-header">
      <h2 id="modal-title">Confirm Delete</h2>
      <button class="close-btn" aria-label="Close">✕</button>
    </div>
    <div class="modal-body">
      <p>Are you sure you want to delete this project?</p>
    </div>
    <div class="modal-footer">
      <button class="btn secondary">Cancel</button>
      <button class="btn danger">Delete</button>
    </div>
  </div>
</div>
```

**Modal Best Practices:**
- Backdrop click to close (or not, for critical modals)
- Esc key to close
- Focus trap (Tab cycles within modal)
- Focus first input on open
- Restore focus to trigger element on close

### Search Patterns

**Autocomplete:**

```html
<div class="search-autocomplete">
  <input
    type="search"
    placeholder="Search projects..."
    aria-autocomplete="list"
    aria-controls="search-results"
  />
  <ul id="search-results" role="listbox">
    <li role="option">Project Alpha</li>
    <li role="option">Project Beta</li>
    <li role="option">Project Gamma</li>
  </ul>
</div>
```

**Filters:**

```html
<div class="search-filters">
  <input type="search" placeholder="Search..." />

  <select name="status">
    <option value="">All Statuses</option>
    <option value="active">Active</option>
    <option value="inactive">Inactive</option>
  </select>

  <select name="owner">
    <option value="">All Owners</option>
    <option value="me">My Projects</option>
    <option value="team">Team Projects</option>
  </select>

  <button class="filter-btn">Apply Filters</button>
  <button class="clear-btn">Clear</button>
</div>
```

**Advanced Search:**

```html
<form class="advanced-search">
  <div class="search-row">
    <select name="field">
      <option value="name">Name</option>
      <option value="description">Description</option>
    </select>
    <select name="operator">
      <option value="contains">Contains</option>
      <option value="equals">Equals</option>
      <option value="starts_with">Starts with</option>
    </select>
    <input type="text" name="value" />
  </div>
  <button class="add-criteria">+ Add Criteria</button>
  <button type="submit">Search</button>
</form>
```

**Search Results:**

```html
<div class="search-results">
  <div class="results-header">
    <span class="results-count">487 results for "alpha"</span>
    <select name="sort">
      <option value="relevance">Relevance</option>
      <option value="date">Date</option>
      <option value="name">Name</option>
    </select>
  </div>

  <div class="result-item">
    <h3><a href="/projects/123">Project Alpha</a></h3>
    <p>Description with <mark>alpha</mark> highlighted...</p>
    <span class="meta">Created Aug 1, 2024 • Owner: Jane Smith</span>
  </div>
</div>
```

---

## Anti-Patterns

### Priority 0 (Critical - Never Do)

**1. Hiding essential features behind hamburger menu on desktop:**
- **Problem:** Desktop has plenty of space, hiding nav reduces discoverability
- **Impact:** Users don't explore features, lower engagement
- **Fix:** Persistent sidebar or top nav on desktop (>1024px)

**2. No keyboard navigation support:**
- **Problem:** Excludes keyboard-only users, WCAG fail
- **Impact:** Accessibility violation, legal risk
- **Fix:** All interactive elements accessible via Tab, Enter, Space, Arrow keys

**3. Forms that break without JavaScript:**
- **Problem:** Core functionality requires JS, fails in degraded conditions
- **Impact:** Users can't complete tasks, frustrated
- **Fix:** Forms submit to server, work without JS (progressive enhancement)

**4. Blocking entire interface during one operation:**
- **Problem:** Full-page spinner, can't do anything else
- **Impact:** Poor UX, feels slow and clunky
- **Fix:** Inline loading states, allow parallel tasks

### Priority 1 (High - Avoid)

**5. Tables without sorting or filtering:**
- **Problem:** Users can't find data in large tables
- **Impact:** Inefficient, forces manual scanning
- **Fix:** Sortable column headers, filter controls above table

**6. No mobile-responsive design:**
- **Problem:** Desktop layout shrunk to mobile, tiny text and targets
- **Impact:** Unusable on mobile, high bounce rate
- **Fix:** Mobile-first responsive design, 48px touch targets, simplified layouts

**7. Slow load times (>5 seconds to interactive):**
- **Problem:** Users wait, may leave before page loads
- **Impact:** High bounce rate, poor perceived performance
- **Fix:** Code splitting, lazy loading, performance budgets (TTI <3.8s)

**8. No autosave for long forms:**
- **Problem:** Users lose work if browser crashes or accidental navigation
- **Impact:** Frustration, lost time, negative sentiment
- **Fix:** Autosave draft every 30 seconds, "All changes saved" indicator

### Priority 2 (Medium - Be Cautious)

**9. Over-reliance on hover states:**
- **Problem:** Mobile has no hover, features inaccessible
- **Impact:** Mobile users can't access actions
- **Fix:** Always provide tap alternative, show actions on mobile

**10. Complex keyboard shortcuts without discoverability:**
- **Problem:** Users don't know shortcuts exist
- **Impact:** Features unused, efficiency gains missed
- **Fix:** Keyboard shortcuts panel (Cmd+/), tooltips show shortcuts

**11. Too many modals:**
- **Problem:** Modal fatigue, interrupts flow
- **Impact:** Annoying, slows users down
- **Fix:** Inline editing, slide-out panels, bottom sheets (fewer modals)

**12. Dense dashboards without hierarchy:**
- **Problem:** Too much information, no visual priority
- **Impact:** Cognitive overload, users don't know where to look
- **Fix:** F-pattern layout, clear visual hierarchy, progressive disclosure

---

## Practical Application

### Workflow 1: New Dashboard Design

**Step 1: Research User Needs**
- Interview users: What metrics matter most?
- Identify top 3 actions (create, export, filter)
- Understand workflows (daily check-ins, weekly reports)

**Step 2: Wireframe Layout**
- Sketch F-pattern: Important metrics top-left, actions top-right
- Plan card hierarchy: Primary metrics (large), secondary (smaller)
- Include: Real-time updates, filters, export

**Step 3: Prioritize Data**
- Most important metric: Large, top-left (e.g., "Active Users: 1,234")
- Change indicators: ↑ 12% (green for positive)
- Trend chart: Line chart (last 30 days)
- Detailed table: Below fold, sortable and filterable

**Step 4: Responsive Breakpoints**
- Desktop (>1024px): Multi-column grid (3-4 metric cards per row)
- Tablet (768-1024px): Two columns
- Mobile (<768px): Single column, stacked cards

**Step 5: Test with Real Data**
- Empty state: "No data yet. Connect your account to get started."
- Loading state: Skeleton cards while fetching
- Error state: "Failed to load. Retry?"
- Real data: Verify hierarchy and scannability

### Workflow 2: Enterprise Admin Panel

**Step 1: Understand Workflows**
- Map user tasks: Create users, manage permissions, view logs
- Identify repetitive actions: Bulk user imports, role assignments
- Define power user needs: Keyboard shortcuts, command palette

**Step 2: Design Keyboard Shortcuts**
- Cmd+K: Command palette (quick access to all actions)
- Cmd+N: Create new user
- Cmd+F: Search/filter users
- Document shortcuts in help panel (Cmd+/)

**Step 3: Implement Bulk Actions**
- Table with checkboxes: Select multiple users
- Bulk action bar: "5 users selected | Delete | Deactivate | Export"
- Confirmation modal: "Delete 5 users? This cannot be undone."
- Undo option: Toast "5 users deleted. Undo?"

**Step 4: Responsive Adaptation**
- Desktop: Full table (all columns), persistent sidebar nav
- Tablet: Hide secondary columns, condensed nav
- Mobile: Card view (user cards), hamburger menu
- Test: Ensure critical actions available on all screen sizes

**Step 5: Progressive Enhancement**
- Core: Form submits to server (create user without JS)
- Enhanced: AJAX form submission (no page reload)
- Optimized: Inline validation, auto-complete for roles

### Workflow 3: Data-Heavy Application

**Step 1: Choose Data Visualization**
- Time-series: Line chart (revenue over time)
- Comparisons: Bar chart (sales by region)
- Part-to-whole: Donut chart (market share by product)
- Avoid: Pie charts with >5 slices (use bar chart instead)

**Step 2: Design Sortable Tables**
- Column headers: Clickable, sort icons (↕ ↑ ↓)
- Filter controls: Above table (search, status dropdown)
- Pagination: "Showing 1-50 of 1,234" (50 per page on desktop)
- Row selection: Checkboxes for bulk export

**Step 3: Add Filtering**
- Search: By name, description
- Dropdowns: Status, category, owner
- Date range: Created date, updated date
- Clear filters: "Clear all" button

**Step 4: Export Options**
- CSV: All rows (or filtered/selected)
- PDF: Formatted report with charts
- Copy: Selected cells to clipboard

**Step 5: Real-Time Updates**
- WebSocket connection: Live price updates
- Visual indicator: "Live" badge (pulsing green)
- Smooth updates: Flash yellow on change (fades after 2s)
- Timestamp: "Updated 3 seconds ago"

### Workflow 4: SaaS Tool Redesign

**Step 1: Analyze Pain Points**
- User feedback: "Too many clicks to complete tasks"
- Analytics: High bounce on multi-step forms
- Support tickets: "How do I edit this?"

**Step 2: Simplify Workflows**
- Reduce steps: Combine create + configure into one page
- Inline editing: Click to edit, no modal
- Autosave: No manual save button, auto-save on blur

**Step 3: Add Inline Editing**
- Table cells: Click to edit, Enter to save, Esc to cancel
- Visual feedback: Border highlights, "Saving..." indicator
- Error handling: Red border, error message, revert on fail

**Step 4: Implement Command Palette**
- Trigger: Cmd+K
- Search: All actions, pages, recent items
- Keyboard nav: Arrow keys to select, Enter to execute
- Fuzzy search: "crproj" matches "Create Project"

**Step 5: Test Efficiency Gains**
- Measure: Time to complete tasks (before vs after)
- User testing: Watch power users use keyboard shortcuts
- Analytics: Track command palette usage, inline edit adoption
- Iterate: Add more shortcuts based on user behavior

---

## Related Skills

**Core Lyra Skills:**
- **`lyra/ux-designer/visual-design-foundations`**: Visual hierarchy for dashboards, typography for dense data, color for status indicators
- **`lyra/ux-designer/information-architecture`**: Navigation structure for complex apps, IA for multi-level hierarchies
- **`lyra/ux-designer/interaction-design-patterns`**: Keyboard shortcuts, button states, feedback patterns, loading states
- **`lyra/ux-designer/accessibility-and-inclusive-design`**: Keyboard navigation, screen reader support, WCAG compliance for web apps
- **`lyra/ux-designer/user-research-and-validation`**: Usability testing for workflows, A/B testing for dashboards

**Platform Skills:**
- **`lyra/ux-designer/mobile-design-patterns`**: Mobile-responsive considerations (<768px breakpoint), touch targets, gestures
- **`lyra/ux-designer/desktop-software-design`**: Keyboard-first workflows, power user patterns, dense information displays

**Cross-Faction:**
- **`muna/technical-writer/clarity-and-style`**: Microcopy for UI, error messages, empty states
- **`ordis/security-architect/secure-authentication-patterns`**: Auth flows for SaaS, session management, secure forms

---

## Additional Resources

**Web Application Standards:**
- Material Design for Web: https://m3.material.io/develop/web
- Apple Human Interface Guidelines (Web): https://developer.apple.com/design/human-interface-guidelines/designing-for-the-web
- Microsoft Fluent UI: https://developer.microsoft.com/en-us/fluentui

**Performance Standards:**
- Google Lighthouse: FCP <1.8s, LCP <2.5s, TTI <3.8s
- Web Vitals: https://web.dev/vitals/
- Performance budgets: <200KB JS gzipped, <50KB CSS gzipped

**Responsive Design:**
- Breakpoints: Mobile <768px, Tablet 768-1024px, Desktop >1024px
- Touch targets: 48x48px minimum (mobile)
- Mobile-first CSS: Base styles mobile, enhance with media queries

**Accessibility Standards:**
- WCAG 2.1 AA: Keyboard navigation, focus indicators, 4.5:1 contrast
- ARIA: Proper roles, labels, live regions for dynamic content
- Keyboard shortcuts: Document in accessible help panel

**Data Visualization:**
- Line charts: Trends over time
- Bar charts: Comparisons between categories
- Pie/donut charts: Part-to-whole (max 5 slices)
- Tables: Sortable, filterable, paginated

**Keyboard Shortcuts:**
- Essential: Cmd+S (save), Cmd+F (find), Cmd+K (command palette)
- Navigation: Tab, Arrow keys, Enter, Esc
- Platform conventions: Cmd (Mac), Ctrl (Windows/Linux)

**Progressive Enhancement:**
- Layer 1: Core HTML (semantic, accessible)
- Layer 2: CSS (visual hierarchy, responsive)
- Layer 3: JavaScript (client-side validation, AJAX)
- Test: Disable JS, disable CSS, slow network
