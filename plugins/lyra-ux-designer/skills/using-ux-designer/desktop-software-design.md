
# Desktop Software Design

## Overview

This skill provides **The Desktop Application Workflow Model**, a systematic 4-dimension methodology for designing native desktop applications that serve professional users and power workflows. Use this when designing productivity tools, creative software, development environments, or any application where users expect keyboard-driven efficiency, extensive customization, and multi-tasking capabilities.

**Core Principle**: Desktop applications serve power users who demand speed, precision, and control. Success means enabling keyboard-first workflows, supporting complex multi-window tasks, allowing deep workspace customization, and providing expert paths that accelerate professional workflows without overwhelming beginners.

**Focus**: Keyboard efficiency, multi-window organization, workspace persistence, and progressive disclosure of advanced features.

## When to Use This Skill

**Use this skill when:**
- Designing native desktop applications (Windows, macOS, Linux)
- Building Electron-based desktop apps
- Creating professional tools (IDEs, design software, DAWs, CAD)
- Implementing keyboard-driven interfaces for power users
- Designing multi-window or multi-document applications
- User mentions: "desktop app", "keyboard shortcuts", "menu bar", "workspace", "power user", "panels"

**Don't use this skill for:**
- Web applications (use `lyra/ux-designer/web-application-design`)
- Mobile apps (use `lyra/ux-designer/mobile-design-patterns`)
- Simple utility tools (may not need full desktop complexity)
- Kiosk or public terminal interfaces (different interaction model)


## The Desktop Application Workflow Model

A systematic 4-dimension evaluation model for desktop application design:

1. **WINDOW ORGANIZATION** - Multi-window, panels/palettes, tabbed interfaces, docking
2. **KEYBOARD EFFICIENCY** - Power user shortcuts, menu mnemonics, keyboard-first navigation
3. **WORKSPACE CUSTOMIZATION** - Toolbars, panels, themes, layouts, import/export settings
4. **EXPERT PATHS** - Advanced settings, scripting, batch operations, command palette, extensibility

Evaluate designs by examining each dimension systematically, ensuring window layouts support complex workflows, all actions are keyboard-accessible, users can adapt the interface to their needs, and advanced features are discoverable without overwhelming beginners.


## Dimension 1: WINDOW ORGANIZATION

**Purpose:** Support complex multi-window workflows and task management typical of desktop environments

Desktop applications leverage the spacious screen real estate and multi-tasking capabilities unique to desktop computers. Users expect to arrange windows across multiple monitors, dock panels to their preferred locations, and have their workspace configuration persist between sessions.

### Evaluation Questions

1. **Does the window structure match task complexity?**
   - Simple utilities: Single window, focused purpose
   - Document editors: Multi-window or tabbed (multiple files)
   - Complex tools: Panels, palettes, multi-monitor support

2. **Can users arrange windows to fit their workflow?**
   - Dockable panels (drag to dock/undock/stack)
   - Resizable splits and dividers
   - Remember positions between sessions

3. **Are window states preserved between sessions?**
   - Window position and size
   - Panel visibility and arrangement
   - Tab order and open documents

4. **Does the app support multi-monitor setups?**
   - Windows can move to any monitor
   - Remember monitor-specific positions
   - Full-screen on secondary monitor

### Single-Window Applications

**Use for:** Simple utilities, focused tools, single-purpose apps

**Characteristics:**
- One main window, no child windows
- Simple task (calculator, timer, color picker)
- All functionality visible at once
- Quick launch and close
- Minimal memory footprint

**Examples:**
- macOS Calculator: Single window, always on top option
- Windows Clock: Single window with tabs (Timer, Alarm, Stopwatch)
- Color pickers: Compact single window

**Best Practices:**
- Window size appropriate for content (not unnecessarily large)
- "Always on top" option for reference tools
- Remember last position and size
- Keyboard shortcuts for main actions

### Multi-Window Applications

**Use for:** Document editors, file browsers, multi-document interfaces

**Characteristics:**
- Each document/file in separate window
- Windows can be arranged independently
- System-level window management (Cmd+Tab, Alt+Tab)
- Can view multiple documents side-by-side

**Examples:**
- Text editors: Each file in separate window
- Image editors (some): Multiple images in separate windows
- Finder/Explorer: Multiple folder windows

**Window Management:**
- Cmd+N / Ctrl+N: New window
- Cmd+W / Ctrl+W: Close current window
- Cmd+` / Alt+Tab: Cycle between app windows
- Window menu: List of open windows

**Multi-Monitor Support:**
- Windows can span or move to any monitor
- Full-screen on secondary monitor
- Remember which monitor window was on

### Panels and Palettes

**Use for:** Professional tools, creative software, IDEs

**Characteristics:**
- Floating or docked panels
- Tool palettes, property inspectors, layer panels
- Can be shown/hidden as needed
- Stackable in tabs
- Resizable and rearrangeable

**Examples:**
- Adobe Photoshop: Layers, Tools, Properties panels
- VS Code: Explorer, Search, Git, Extensions sidebar panels
- Blender: Properties, Outliner, Timeline panels

**Panel Types:**

**Tool Palettes:**
- Icons for tools (brush, select, pen)
- Single-column or grid layout
- Keyboard shortcuts shown in tooltips
- Active tool highlighted

**Property Inspectors:**
- Shows properties of selected object
- Context-sensitive (changes based on selection)
- Form fields for editing values
- Organized by category (accordion or tabs)

**Contextual Panels:**
- Color picker, layer manager, timeline
- Specific to workflow (design, video editing)
- Can be hidden when not needed

**Docking Behavior:**

**Floating (Undocked):**
- Free-floating window
- Can position anywhere, even on second monitor
- Has own window controls (close, minimize)
- Always on top of main window

**Docked:**
- Attached to edge of main window (left, right, top, bottom)
- Integrated into window layout
- Resizable via drag divider
- Collapses to icon strip or hides completely

**Tabbed/Stacked:**
- Multiple panels in same dock area
- Tabs at top or bottom
- Switch between tabs
- Drag tab to reorder or undock

**Best Practices:**
- Sensible defaults (common panels visible, others hidden)
- View menu: List all panels with checkboxes (show/hide)
- Drag panel header to move/undock
- Double-click header to dock/undock
- Close button on each panel
- Remember panel arrangement between sessions

### Tabbed Interfaces

**Use for:** Many documents in single window, web browsers, terminal emulators

**Characteristics:**
- Multiple documents in single window
- Tabs at top (or bottom) for switching
- Saves screen space vs multiple windows
- Easier to organize than scattered windows

**Examples:**
- Web browsers (Chrome, Firefox): Multiple pages in tabs
- VS Code: Multiple files in tabs
- Terminal emulators: Multiple shells in tabs

**Tab Management:**

**Creating Tabs:**
- Cmd+T / Ctrl+T: New tab
- Cmd+Click link: Open in new tab
- File > Open in New Tab

**Navigating Tabs:**
- Cmd+Tab / Ctrl+Tab: Next tab
- Cmd+Shift+Tab / Ctrl+Shift+Tab: Previous tab
- Cmd+1-9 / Ctrl+1-9: Jump to tab number
- Cmd+W / Ctrl+W: Close current tab

**Tab Features:**
- Close button on each tab (X)
- Drag tabs to reorder
- Drag tab out to create new window
- Drag tab to different window to merge
- Right-click: Close, Close Others, Close to Right
- Unsaved indicator (dot or asterisk)
- Preview on hover (thumbnail or tooltip)

**Tab Bar Overflow:**
- Scroll buttons when too many tabs
- Or shrink tabs to minimum width
- Or show dropdown menu of all tabs

### Window Snapping and Management

**Purpose:** Quick window arrangement for multi-tasking

**Windows (Snap Assist):**
- Win+Left Arrow: Snap left half
- Win+Right Arrow: Snap right half
- Win+Up Arrow: Maximize
- Win+Down Arrow: Restore/minimize
- Win+Shift+Arrow: Move to another monitor

**macOS (requires third-party or native in recent versions):**
- Drag window to edge (hold for green button options)
- Green button: Full-screen or tile left/right
- Third-party: Rectangle, Magnet (Win-style snapping)

**Linux (varies by window manager):**
- GNOME: Super+Left/Right (half screen)
- KDE: Similar to Windows
- Tiling window managers: i3, bspwm (keyboard-driven)

**Best Practices:**
- Support standard OS window management
- Don't override system shortcuts
- Allow windows to be resized and moved freely
- Remember window position between sessions

### Window State Persistence

**Purpose:** Users return to their exact workspace configuration

**What to Remember:**
- Window position (x, y coordinates)
- Window size (width, height)
- Window state (maximized, minimized, full-screen)
- Monitor (if multi-monitor setup)
- Panel visibility and arrangement
- Panel sizes and docked positions
- Tab order and open documents
- Scroll positions (nice-to-have)

**Storage:**
- Application preferences file
- JSON, XML, or binary format
- Per-user configuration

**Edge Cases:**
- Monitor no longer exists: Fall back to primary monitor
- Resolution changed: Scale or center window
- Off-screen window: Move to visible area

**Reset Option:**
- "Reset Window Layout" in View menu
- Restores default layout
- Clear and immediate

### System Tray / Menu Bar Integration

**Purpose:** Background apps, status indicators, quick actions

**Windows (System Tray):**
- Icon in bottom-right taskbar
- Right-click: Context menu (Settings, Quit)
- Left-click: Open main window or status
- Notifications: Toast pop-ups

**macOS (Menu Bar):**
- Icon in top-right menu bar
- Click: Dropdown menu or panel
- Status text (e.g., time tracker: "2:34")
- Can hide main app window, live in menu bar only

**Linux (System Tray):**
- Varies by desktop environment
- GNOME: App indicator
- KDE: System tray icon

**Use Cases:**
- Background apps (music players, chat)
- System monitors (CPU, network, battery)
- Quick toggles (VPN on/off, do not disturb)
- Status displays (syncing, connected)

**Best Practices:**
- Minimize to tray option (don't quit app)
- "Show main window" menu item
- "Quit" menu item (clear exit)
- Tooltip shows app name and status
- Update icon for status changes (syncing, error)

### Multi-Monitor Support

**Purpose:** Professional users often have 2-3+ monitors

**Features:**
- Windows can move to any monitor
- Full-screen on secondary monitor (not just primary)
- Remember which monitor each window was on
- Handle monitor disconnect gracefully

**Best Practices:**
- Don't assume single monitor
- Test with 2+ monitor setup
- Handle monitor disconnect (move windows to available monitor)
- Allow floating panels on secondary monitor
- Don't lock windows to primary monitor


## Dimension 2: KEYBOARD EFFICIENCY

**Purpose:** Enable power users to work without reaching for the mouse

Desktop power users expect to complete tasks entirely via keyboard. Every action should be keyboard-accessible, shortcuts should follow platform conventions, and users should be able to customize shortcuts to match their muscle memory from other tools.

### Evaluation Questions

1. **Are all actions accessible via keyboard?**
   - Every menu item has keyboard equivalent
   - Modal dialogs have keyboard shortcuts
   - Tab navigation through all controls

2. **Do shortcuts follow platform conventions?**
   - Cmd (macOS) vs Ctrl (Windows/Linux)
   - Standard shortcuts (Cmd+S, Cmd+C, Cmd+V)
   - Platform-specific patterns (Cmd+Q on macOS, Alt+F4 on Windows)

3. **Are shortcuts discoverable?**
   - Shown in menus next to commands
   - Tooltips include shortcuts
   - Help menu: "Keyboard Shortcuts" reference

4. **Can users customize shortcuts?**
   - Preferences: Keyboard Shortcuts panel
   - Conflict detection (warn if duplicate)
   - Reset to defaults option

### Essential File Operation Shortcuts

**Universal Shortcuts (Cross-Platform):**

```
Cmd+N / Ctrl+N: New document/file
Cmd+O / Ctrl+O: Open file
Cmd+S / Ctrl+S: Save
Cmd+Shift+S / Ctrl+Shift+S: Save As
Cmd+W / Ctrl+W: Close window/tab
Cmd+Q / Ctrl+Q: Quit application (macOS uses Cmd+Q, Windows uses Alt+F4)
```

**Platform Differences:**
- **macOS:** Uses Cmd (⌘) key for shortcuts
- **Windows/Linux:** Uses Ctrl key for shortcuts
- **macOS Quit:** Cmd+Q (app-level)
- **Windows Quit:** Alt+F4 (window-level, closes app if last window)

**Best Practices:**
- Detect platform and show correct modifier key
- Use standard shortcuts (don't reinvent)
- Cmd+W closes window/tab, not app (on macOS)
- Cmd+Q quits entire app (macOS only)
- Unsaved changes: Prompt before closing

### Essential Edit Operation Shortcuts

**Standard Edit Shortcuts:**

```
Cmd+Z / Ctrl+Z: Undo
Cmd+Shift+Z / Ctrl+Shift+Z: Redo (or Cmd+Y / Ctrl+Y on Windows)
Cmd+X / Ctrl+X: Cut
Cmd+C / Ctrl+C: Copy
Cmd+V / Ctrl+V: Paste
Cmd+A / Ctrl+A: Select All
```

**Redo Variations:**
- **macOS:** Cmd+Shift+Z (standard)
- **Windows:** Ctrl+Y (more common) or Ctrl+Shift+Z
- Support both on Windows for user preference

**Advanced Edit Shortcuts:**

```
Cmd+D / Ctrl+D: Duplicate
Cmd+G / Ctrl+G: Find Next (after Cmd+F)
Cmd+Shift+G / Ctrl+Shift+G: Find Previous
Cmd+E / Ctrl+E: Use selection for find (place selection in search field)
```

### Essential Navigation Shortcuts

**In-App Navigation:**

```
Cmd+F / Ctrl+F: Find/Search (open search dialog or focus search field)
Cmd+Tab / Ctrl+Tab: Switch tabs (forward)
Cmd+Shift+Tab / Ctrl+Shift+Tab: Switch tabs (backward)
Cmd+1-9 / Ctrl+1-9: Switch to tab number (1st, 2nd, 3rd...)
Cmd+` / Alt+`: Cycle between app windows (macOS: Cmd+`, Windows varies)
```

**System Navigation:**
- **macOS:** Cmd+Tab (switch apps), Cmd+` (switch windows within app)
- **Windows:** Alt+Tab (switch windows), Win+Tab (Task View)
- **Linux:** Alt+Tab (switch windows), varies by WM

**Form Navigation:**

```
Tab: Next field/control
Shift+Tab: Previous field/control
Arrow keys: Navigate lists, trees, menus
Space: Activate checkbox, toggle, button
Enter: Submit form, activate default button, open selected item
ESC: Cancel operation, close dialog
```

**Best Practices:**
- Tab order matches visual order (left-to-right, top-to-bottom)
- Focus indicators visible (2px outline, high contrast)
- Enter activates default button (often "OK" or "Submit")
- ESC always cancels or closes current context

### Power User Shortcuts

**Advanced Shortcuts:**

```
F2: Rename (selected file, item)
Delete / Backspace: Delete selected item
Cmd+Delete / Shift+Delete: Permanently delete (bypass trash)
Cmd+I / Ctrl+I: Get Info / Properties
Cmd+, / Ctrl+,: Preferences/Settings (macOS standard)
F1: Help (Windows), Help not standard on macOS
F11: Full-screen (toggle)
```

**Selection Shortcuts:**

```
Cmd+Click / Ctrl+Click: Add to selection (multi-select)
Shift+Click: Range selection (select all between first and last)
Cmd+A / Ctrl+A: Select all
ESC: Deselect all
```

**Examples:**
- Click first item, Shift+Click last item: Select range
- Cmd+Click multiple items: Add each to selection
- Click empty area or ESC: Deselect

**Window Management:**

```
Cmd+M / Win+Down: Minimize window
Cmd+H / Win+H: Hide window (macOS hides app windows)
Cmd+Option+H: Hide others (macOS)
F11: Full-screen toggle (varies by app)
```

### Shortcut Discoverability

**Show Shortcuts in Menus:**

```
File
  New              Cmd+N
  Open...          Cmd+O
  Save             Cmd+S
  Save As...       Cmd+Shift+S
  Close Window     Cmd+W
```

**Display Format:**
- Right-aligned in menu
- Gray color (less prominent than label)
- Use symbols on macOS: ⌘ Cmd, ⌥ Option, ⇧ Shift, ⌃ Control
- Use text on Windows: Ctrl+S, Alt+F4

**Tooltips with Shortcuts:**

```
Button: Save
Tooltip: "Save document (Cmd+S)"
```

- Brief delay before showing (300ms)
- Include shortcut in parentheses
- Position near cursor

**Keyboard Shortcuts Reference:**

- Help menu: "Keyboard Shortcuts" menu item
- Opens modal or separate window
- Organized by category:
  - File Operations
  - Edit Operations
  - Navigation
  - View
  - Tools
  - Window Management
- Searchable (filter by name or shortcut)
- Printable version

### Shortcut Customization

**Preferences > Keyboard Shortcuts:**

**Interface:**
- List of all actions with current shortcut
- Click to edit (record new shortcut)
- Search/filter actions
- Reset to defaults button
- Import/export shortcuts (share with team)

**Conflict Detection:**
- Warning if shortcut already used
- "Replace existing shortcut?" confirmation
- Show what action will lose the shortcut

**Presets:**
- Default shortcuts
- VS Code preset (for code editors)
- Sublime Text preset
- Vim/Emacs keybindings (for editors)
- Allow users to switch preset

**Best Practices:**
- Don't override system shortcuts (Cmd+Tab, Alt+F4)
- Warn users if remapping common shortcuts
- Provide sensible defaults (most users won't customize)
- Sync shortcuts across devices (if cloud-based app)

### Menu Mnemonics (Windows/Linux)

**Alt Key Access:**

```
Alt (press and release): Highlight menu bar
Alt+F: File menu
Alt+E: Edit menu
Alt+V: View menu
Alt+H: Help menu
```

**In Menu:**
- Underlined letter indicates mnemonic
- Alt+F, then S: File > Save
- Can chain: Alt, F, S (Save)

**Not Standard on macOS:**
- macOS doesn't use mnemonics
- Use Cmd+shortcuts instead

**Best Practices (Windows/Linux):**
- First letter of menu item (F for File)
- Avoid conflicts (different letter if duplicate)
- Show underlined letter in menu
- Support Alt+Letter for quick access


## Dimension 3: WORKSPACE CUSTOMIZATION

**Purpose:** Let users adapt the interface to their workflow and preferences

Professional desktop users work in the same application for hours daily. They expect to customize toolbars, arrange panels to suit their workflow, save layout presets, and choose themes that reduce eye strain. Customization should be deep but not overwhelming, with sensible defaults that work out-of-the-box.

### Evaluation Questions

1. **Can users customize toolbars and panels?**
   - Add/remove tools from toolbar
   - Rearrange toolbar items (drag to reorder)
   - Show/hide panels (View menu checkboxes)
   - Dock panels to preferred locations

2. **Are workspace layouts savable and shareable?**
   - Save current layout as preset
   - Load preset (Editing, Debugging, Design)
   - Export/import layouts (share with team)
   - Quick-switch layouts (dropdown or menu)

3. **Does the app support themes and appearance?**
   - Light mode, dark mode, high contrast
   - Custom themes (user-created or third-party)
   - Accent colors (brand or user preference)
   - Font size adjustment (accessibility)

4. **Are defaults sensible for new users?**
   - Works out-of-box without customization
   - Common panels visible, niche panels hidden
   - Standard toolbar (most-used tools)
   - Easy to reset to defaults

### Customizable Toolbars

**Purpose:** Quick access to frequently-used commands

**Customization:**

**Add/Remove Tools:**
- Right-click toolbar: "Customize Toolbar"
- Drag tools from palette to toolbar
- Drag tools off toolbar to remove
- Reset to default set

**Reorder Tools:**
- Drag tools to reorder
- Visual gap indicator while dragging
- Snap into position on drop

**Display Options:**
- Icons only (compact)
- Icons + labels (clearer)
- Labels only (rare, accessibility)
- Small icons vs large icons

**Separators and Spacing:**
- Drag separator to toolbar (visual divider)
- Flexible space (pushes items to right)
- Fixed space (small gap)

**Examples:**
- **macOS:** Right-click toolbar > "Customize Toolbar" (shows palette sheet)
- **Windows:** Right-click toolbar > "Customize" (dialog or inline editing)
- **VS Code:** View > Appearance > Customize Layout

**Best Practices:**
- Default toolbar has most common actions
- "Customize Toolbar" in View menu
- Visual feedback while dragging (semi-transparent item)
- Preview before committing change
- Undo customization (or reset to defaults)

### Dockable Panels

**Purpose:** Arrange information and tools to match workflow

**Panel Manipulation:**

**Show/Hide:**
- View menu: List of panels with checkboxes
- Keyboard shortcut to toggle (e.g., Cmd+1 for panel 1)
- Close button on panel header (X)

**Dock/Undock:**
- Drag panel header to move
- Drag to edge to dock (visual preview)
- Drag away from edge to undock (floating)
- Double-click header to toggle dock/undock

**Resize:**
- Drag divider between panels
- Double-click divider to collapse/expand
- Minimum and maximum sizes enforced

**Stack in Tabs:**
- Drag panel onto another panel (tabbed group)
- Tabs at top or bottom of panel stack
- Click tab to switch
- Drag tab to reorder or separate

**Examples:**
- **Adobe Photoshop:** Drag panels, dock to edges, stack in tabs, collapse to icons
- **VS Code:** Drag sidebar panels, resize, move to secondary sidebar
- **Blender:** Complex docking system, divide areas, merge areas

**Visual Feedback:**
- Blue outline where panel will dock
- Semi-transparent panel while dragging
- Snap to docking zones

**Best Practices:**
- Clear docking zones (edges, corners)
- Visual preview before dropping
- Remember panel arrangement between sessions
- Reset to default layout option

### Workspace Layouts

**Purpose:** Save and switch between layout presets for different tasks

**Concept:**
- Layout = panel arrangement, sizes, visibility
- Presets for common workflows
- Quick-switch for context changes

**Built-in Presets:**
- **Default:** Balanced layout for general use
- **Editing:** Large editor area, minimal panels
- **Debugging:** Debugger panels prominent, console, watch variables
- **Design:** Canvas large, tool panels on sides

**Custom Layouts:**
- Arrange panels as desired
- View > Save Layout As... (name it)
- Appears in View > Layouts menu
- Switch anytime

**Layout Switcher:**
- Dropdown in toolbar (quick access)
- Or View menu > Layouts submenu
- Keyboard shortcut (e.g., Cmd+Shift+1 for layout 1)

**Examples:**
- **Adobe Photoshop:** Window > Workspace > Essentials, Photography, Painting
- **VS Code:** View > Command Palette > "View: Load Workspace"
- **Eclipse IDE:** Window > Perspective > Open Perspective (Java, Debug, Git)

**Best Practices:**
- Provide 2-4 built-in presets
- Easy to create custom layouts
- Delete custom layouts (manage layouts)
- Export/import layouts (share with team, sync across devices)

### Themes and Appearance

**Purpose:** Reduce eye strain, match user preference, accessibility

**Theme Types:**

**Light Mode:**
- White or light gray backgrounds
- Dark text
- Standard for most users
- Good for bright environments

**Dark Mode:**
- Dark backgrounds (black, charcoal, navy)
- Light text (white, light gray)
- Reduces eye strain in dim environments
- Popular for developers, designers

**High Contrast:**
- Maximum contrast (black and white)
- Accessibility feature
- For users with low vision
- System-level setting (OS theme)

**Custom Themes:**
- User-created or third-party
- JSON or CSS-based theme files
- Color palette (background, text, accent, warning, error)
- Syntax highlighting (for code editors)

**Accent Colors:**
- Brand color or user preference
- Used for highlights, active states, links
- Preferences > Appearance > Accent Color

**Examples:**
- **macOS System:** Light, Dark, Auto (follows system)
- **VS Code:** Many themes (Dracula, Monokai, Solarized)
- **Slack:** Light, Dark, Sync with OS
- **Adobe Apps:** Dark theme standard for creative work

**Implementing Themes:**

**CSS Variables (Web-based or Electron):**
```css
:root {
  --background: #ffffff;
  --text: #000000;
  --accent: #0066cc;
}

[data-theme="dark"] {
  --background: #1e1e1e;
  --text: #d4d4d4;
  --accent: #0099ff;
}
```

**Native Apps:**
- Platform theme APIs (Windows: Light/Dark, macOS: NSAppearance)
- Custom theme engine (load colors from file)

**Respect System Theme:**
- Preferences > Appearance > "Sync with system"
- Automatically switch light/dark based on OS setting
- macOS: Auto (light during day, dark at night)

**Best Practices:**
- Offer light, dark, and system sync
- Don't force dark mode (user preference)
- Test all themes for readability
- Support high contrast mode (accessibility)
- Syntax themes for code editors (many options)

### Preferences and Settings

**Purpose:** Central location for all customization options

**Organization:**

**Tabbed Categories:**
- General, Appearance, Editor, Keyboard, Extensions
- Each tab is focused category
- Searchable across all settings

**Search:**
- Search field at top
- Filters settings by name or keyword
- Highlights matches

**Settings Format:**

**Checkboxes:**
- Enable/disable features
- "Show line numbers", "Auto-save", "Spell check"

**Dropdowns:**
- Choose from options
- "Theme: Light / Dark / Auto"
- "Font: System / Monospace / Custom"

**Sliders/Numbers:**
- Numeric values
- "Font size: 12" (slider or input)
- "Auto-save interval: 30 seconds"

**Text Fields:**
- Custom values
- "Default save location: /Users/name/Documents"

**Color Pickers:**
- Custom colors
- "Accent color: [blue]"

**Best Practices:**
- Searchable settings (filter by keyword)
- Reset to defaults (per-category or global)
- Import/export settings (JSON file)
- Sync settings across devices (cloud-based)
- Clear labels and descriptions
- Group related settings together
- Dangerous settings in "Advanced" tab


## Dimension 4: EXPERT PATHS

**Purpose:** Provide advanced features and automation for power users without overwhelming beginners

Professional desktop users demand advanced capabilities: scripting for automation, batch operations for efficiency, extensibility via plugins, and direct access to all commands. These features should be discoverable (not hidden) but progressively disclosed so beginners aren't overwhelmed.

### Evaluation Questions

1. **Are advanced features discoverable but not intrusive?**
   - Command palette for quick access
   - Advanced preferences in separate tab (not mixed with basics)
   - Help documentation for power features
   - Onboarding hints for expert paths

2. **Can repetitive tasks be automated?**
   - Macro recording (record actions, replay)
   - Scripting API (Python, JavaScript, Lua)
   - Batch operations (process multiple files)
   - Keyboard maestro integration (macOS)

3. **Is there a command palette for quick access?**
   - Cmd+Shift+P / Ctrl+Shift+P
   - Fuzzy search all commands
   - Shows keyboard shortcuts
   - Includes recent commands

4. **Does the app support extensibility?**
   - Plugin/extension system
   - Marketplace or directory
   - API documentation
   - Community contributions

### Command Palette

**Purpose:** Fuzzy search for all actions, keyboard-first power user tool

**Trigger:**
- Cmd+Shift+P / Ctrl+Shift+P (VS Code style)
- Or Cmd+K / Ctrl+K (some apps)
- Appears as modal overlay

**Features:**

**Fuzzy Search:**
- Type partial command: "sv" matches "Save"
- "opfl" matches "Open File"
- Results ranked by match quality

**Categories:**
- All commands (File, Edit, View, Tools)
- Recently used (top of list)
- Frequently used
- Pages/Navigation
- Settings

**Keyboard Navigation:**
- Arrow Up/Down: Select result
- Enter: Execute selected command
- ESC: Close palette
- Type to filter results

**Display Format:**
```
Create New File                    Cmd+N
Open File...                       Cmd+O
Save                               Cmd+S
Save As...                         Cmd+Shift+S
-----------------------------------
Toggle Sidebar                     Cmd+B
Toggle Terminal                    Cmd+`
Command Palette                    Cmd+Shift+P
```

**Examples:**
- **VS Code:** Cmd+Shift+P (all commands, extensions, settings)
- **Sublime Text:** Cmd+Shift+P
- **Spotlight (macOS):** Cmd+Space (system-wide search, not just app)
- **Alfred (macOS):** Similar to Spotlight, extensible

**Best Practices:**
- Fast response (no network delay)
- Include keyboard shortcuts in results
- Show icons for visual scanning
- Recently used at top (learn user patterns)
- Persistent across sessions (remember recent)

### Advanced Preferences

**Purpose:** Expert settings without overwhelming beginners

**Organization:**

**Basic vs Advanced Tabs:**
- Preferences > General (basic settings)
- Preferences > Advanced (expert settings)
- Clear separation, progressive disclosure

**Advanced Settings Examples:**
- Performance tuning (cache size, thread count)
- Experimental features (beta functionality)
- Developer options (debug mode, console)
- Network settings (proxy, timeout)
- File associations (default app for file types)

**Progressive Disclosure:**
- Simple mode by default
- "Show Advanced" button expands expert options
- Collapsible sections (accordion)
- Warning badges for dangerous settings

**Search in Preferences:**
- Search field filters all settings (basic + advanced)
- Finds advanced settings even if hidden
- Ensures discoverability

**Examples:**
- **VS Code:** Settings UI (basic) or settings.json (advanced)
- **Firefox:** Preferences (basic) or about:config (advanced, warning message)
- **Adobe Lightroom:** Preferences (tabs for each category, advanced options within)

**Best Practices:**
- Don't hide essential settings in "Advanced"
- Clear labels and descriptions for expert settings
- Warning for dangerous changes (can break app)
- Reset to defaults option
- Tooltips explain technical terms

### Scripting and Automation

**Purpose:** Automate repetitive tasks, extend functionality

**Macro Recording:**

**Record Actions:**
- Edit > Start Recording Macro
- Perform actions (clicks, typing, shortcuts)
- Edit > Stop Recording
- Save macro with name

**Playback:**
- Edit > Playback Macro
- Or assign keyboard shortcut to macro
- Executes recorded actions

**Examples:**
- **Microsoft Office:** Record Macro (VBA)
- **Photoshop:** Actions panel (record and replay steps)
- **Excel:** Macro recorder (VBA scripts)

**Scripting API:**

**Embed Scripting Language:**
- Python (common for creative tools)
- JavaScript (Electron apps)
- Lua (games, lightweight tools)
- AppleScript (macOS automation)

**API Access:**
- Document manipulation
- UI automation
- File I/O
- Network requests

**Script Editor:**
- Built-in or external
- Syntax highlighting
- Debugging tools
- Script library (saved scripts)

**Examples:**
- **Blender:** Python API (access all objects, run headless)
- **Sublime Text:** Python plugins
- **AutoCAD:** LISP scripting
- **Affinity Designer:** JavaScript macros

**Use Cases:**
- Batch processing (resize 100 images)
- Custom tools (specialized workflow)
- Integration (connect to external APIs)
- Data generation (create test data)

**Best Practices:**
- Clear API documentation
- Example scripts (cookbook)
- Script marketplace (share scripts)
- Sandboxing (limit dangerous operations)
- Error handling (clear error messages)

### Batch Operations

**Purpose:** Process multiple files/items at once

**Common Batch Operations:**

**Batch File Processing:**
- Resize images (all images in folder)
- Convert formats (PSD to PNG)
- Rename files (add prefix/suffix)
- Apply filters (sharpen, color correct)

**Batch Actions:**
- Delete multiple items (select all, delete)
- Move/copy files (drag multiple)
- Tag/categorize (apply tag to selection)
- Export (export selected items)

**UI Patterns:**

**Select Multiple:**
- Cmd+Click / Ctrl+Click: Add to selection
- Shift+Click: Range selection
- Cmd+A / Ctrl+A: Select all

**Batch Action Menu:**
- Right-click selection: Context menu
- "Apply to 12 selected items"
- Actions: Delete, Move, Export, Convert

**Progress Indicator:**
- Progress bar: "Processing 45 of 127 files"
- Cancel button (stop batch)
- Errors: "3 files failed, view log"

**Examples:**
- **Lightroom:** Export multiple photos, apply preset to batch
- **Photoshop:** Batch > Automate (run action on folder)
- **Finder/Explorer:** Multi-select files, rename all (pattern)

**Best Practices:**
- Clear progress feedback
- Cancelable (don't lock UI)
- Error handling (skip failed items, continue)
- Undo support (or backup before batch)
- Summary report (123 succeeded, 3 failed)

### Recently Used

**Purpose:** Quick access to recent files, commands, and actions

**Recent Files:**
- File > Open Recent (list of recent files)
- 10-20 items (configurable)
- Keyboard shortcut to reopen last (Cmd+Shift+T)

**Recent Commands:**
- Command palette: Recent at top
- Edit > Repeat Last Action (Cmd+Shift+Z or F4)

**Recent Folders:**
- File > Open Recent Folder
- Or Open dialog remembers last location

**Jump Lists (Windows):**
- Right-click taskbar icon: Recent files
- Quick access without opening app

**macOS Dock Menu:**
- Right-click dock icon: Recent files, windows

**Best Practices:**
- Persistent across sessions
- Clear recents option (privacy)
- Pin favorites (always show)
- Keyboard access (arrow keys navigate)

### Extensibility and Plugins

**Purpose:** Community contributions, user customization

**Plugin Architecture:**

**Plugin Types:**
- UI extensions (new panels, tools)
- File format support (import/export)
- Scripting integrations (connect to services)
- Themes (appearance customization)

**Plugin Manager:**
- Browse plugins (marketplace)
- Install, update, uninstall
- Enable/disable plugins
- Configure plugin settings

**API for Developers:**
- Clear documentation
- Examples and templates
- Debugging tools
- Submission guidelines (marketplace)

**Examples:**
- **VS Code:** Extensions marketplace (thousands of extensions)
- **Figma:** Plugins (community-created tools)
- **Blender:** Add-ons (Python-based extensions)
- **Sublime Text:** Package Control (plugin manager)

**Security:**
- Sandboxing (limit plugin permissions)
- Code review (for marketplace submissions)
- User warnings (unofficial plugins)

**Best Practices:**
- Easy to install (one-click)
- Auto-update plugins (optional)
- Clear permissions (what plugin can access)
- Disable plugins if causing issues
- Official + community plugins (both valuable)


## Desktop-Specific Design Patterns

### Menu Bar

**Purpose:** Standard navigation and command access in desktop applications

**Standard Menu Structure:**

**File Menu:**
```
File
  New                    Cmd+N
  Open...                Cmd+O
  Open Recent           ▶
  ---------------------
  Close                  Cmd+W
  Close All              Cmd+Option+W
  Save                   Cmd+S
  Save As...             Cmd+Shift+S
  ---------------------
  Page Setup...
  Print...               Cmd+P
  ---------------------
  Quit                   Cmd+Q (macOS) or Exit (Windows)
```

**Edit Menu:**
```
Edit
  Undo                   Cmd+Z
  Redo                   Cmd+Shift+Z
  ---------------------
  Cut                    Cmd+X
  Copy                   Cmd+C
  Paste                  Cmd+V
  Delete                 Delete
  Select All             Cmd+A
  ---------------------
  Find                   Cmd+F
  Replace                Cmd+Option+F
  ---------------------
  Preferences            Cmd+, (macOS) or Options (Windows)
```

**View Menu:**
```
View
  Sidebar                Cmd+B
  Toolbar                Cmd+Option+T
  Status Bar
  ---------------------
  Zoom In                Cmd++
  Zoom Out               Cmd+-
  Actual Size            Cmd+0
  ---------------------
  Full Screen            Cmd+Control+F
  ---------------------
  Customize Layout...
```

**Help Menu:**
```
Help
  [App Name] Help
  Keyboard Shortcuts     Cmd+/
  Release Notes
  ---------------------
  Report Issue...
  ---------------------
  About [App Name]       (macOS: App menu > About)
```

**Platform Differences:**

**macOS:**
- Menu bar at top of screen (global)
- Application menu (first menu): About, Preferences, Hide, Quit
- Menu bar always visible (even without windows)
- System menus (Window, Help) added automatically

**Windows/Linux:**
- Menu bar in window (local to window)
- File menu includes Exit (Quit)
- Edit menu includes Options/Preferences
- Alt key accesses menu (mnemonics)

**Best Practices:**
- Follow platform conventions
- Group related items (dividers between groups)
- Show keyboard shortcuts (right-aligned)
- Disable unavailable items (grayed out)
- Hierarchical menus (submenu arrow: ▶)

### Context Menus (Right-Click)

**Purpose:** Quick access to contextual actions

**Trigger:**
- Right-click (mouse)
- Control+Click (macOS, one-button mouse)
- Shift+F10 or Menu key (keyboard)
- Long-press (touchscreen)

**Content:**
- Context-sensitive (different per item type)
- 5-10 items maximum
- Most common actions
- Dividers separate groups
- Dangerous actions at bottom (red text)

**Examples:**

**File Context Menu:**
```
Open
Open With              ▶
-----------------------
Get Info               Cmd+I
Rename                 F2
Duplicate              Cmd+D
-----------------------
Move to Trash          Cmd+Delete
```

**Text Selection Context Menu:**
```
Cut                    Cmd+X
Copy                   Cmd+C
Paste                  Cmd+V
-----------------------
Look Up "[word]"
Search Google for "[selection]"
```

**Best Practices:**
- Top 5-7 actions (don't overwhelm)
- Match selected object (file, text, canvas object)
- Include keyboard shortcuts (discoverability)
- Close on click outside or ESC
- Keyboard navigation (arrow keys)

### Dialogs

**Purpose:** Modal and modeless windows for focused tasks

**Modal Dialogs:**

**Characteristics:**
- Block interaction with main window
- Must be dismissed before continuing
- Use for critical decisions (save before quit)
- Or complex forms (preferences)

**Examples:**
- Save confirmation: "Save changes before closing?"
- Preferences/Settings dialog
- Print dialog
- Error alerts

**Button Placement:**
- **macOS:** Right-aligned (Cancel left, OK/primary right)
- **Windows:** Right-aligned (OK left, Cancel right) — note: varies
- Primary action visually prominent (filled button)
- Cancel as secondary (outline button) or link

**Keyboard:**
- Enter: Primary action (OK, Save)
- ESC: Cancel or Close
- Tab: Navigate fields

**Modeless Dialogs:**

**Characteristics:**
- Float on top, don't block main window
- Can interact with main window while dialog open
- Use for reference or tools (Find & Replace, Color Picker)

**Examples:**
- Find & Replace (search while editing)
- Color picker (select colors while designing)
- Inspector panel (view properties)

**Best Practices:**
- Use modal for critical decisions only
- Modeless for tools and reference
- Clear primary action (visual weight)
- Keyboard shortcuts (Enter, ESC)
- Don't nest modals (confusing)

### Wizards (Multi-Step Dialogs)

**Purpose:** Guide users through complex setup or process

**Structure:**
- Step 1: Welcome / Choose option
- Step 2: Configure settings
- Step 3: Review choices
- Step 4: Confirmation / Complete

**Navigation:**
- Buttons: "Back", "Next", "Cancel"
- Last step: "Finish" or "Done"
- Progress indicator: "Step 2 of 4"
- Can skip optional steps

**Examples:**
- First-launch setup (choose preferences)
- Export wizard (format, quality, location)
- Import wizard (select source, map fields, import)
- Project setup (name, location, template)

**Best Practices:**
- Clear progress (step 2 of 4)
- Back button works (don't lose data)
- Skip optional steps (offer defaults)
- Review before final step (confirmation)
- Cancelable (ESC or Cancel button)

### File System Integration

**Purpose:** Native file operations and OS integration

**Open/Save Dialogs:**

**Native Dialogs:**
- Use OS-native file picker (NSOpenPanel on macOS, IFileOpenDialog on Windows)
- Users familiar with platform dialog
- Consistent with other apps
- Access to system features (favorites, recent, network)

**Open Dialog:**
- Choose file(s) to open
- File type filter (dropdown: "Images", "All Files")
- Multi-select (Cmd+Click / Ctrl+Click)
- Preview pane (if supported by OS)

**Save Dialog:**
- Choose location and filename
- File format dropdown (if multiple formats)
- Options expander (quality, settings)
- Overwrite confirmation (if file exists)

**Best Practices:**
- Remember last location (per dialog or global)
- Default filename (not "Untitled")
- Clear file format options
- Validate filename (illegal characters)

**Drag-and-Drop:**

**Accept Files:**
- Drag file from Finder/Explorer onto app window
- Visual feedback (border highlight, "Drop to open")
- Open file or import content

**Export via Drag:**
- Drag item from app to Finder/Explorer
- Creates file at drop location
- Or copy data to compatible app

**Best Practices:**
- Visual feedback (highlight drop zone)
- Support multiple files (batch open)
- Validate file types (reject incompatible)
- Handle errors gracefully (unsupported format)

**Quick Look / Preview (macOS):**
- Select file, press Space: Preview
- Support in your file format (implement Quick Look plugin)
- Thumbnail generation

**File Associations:**
- Register app as handler for file types (.psd, .sketch)
- Double-click file: Opens in your app
- Right-click file > Open With: Your app listed
- Set as default handler

**Best Practices:**
- Register relevant file extensions
- Provide app icon for file type
- Don't claim generic types (.txt) as default (without user choice)

### Status Bar

**Purpose:** Display status information and contextual hints

**Location:**
- Bottom of window (standard)
- Full-width bar

**Content:**

**Left Side:**
- Status text (e.g., "Ready", "Saving...", "23 items selected")
- Contextual hints ("Click to select")

**Right Side:**
- Indicators (e.g., "Ln 45, Col 12" in text editor)
- Progress bar (if operation in progress)
- Status icons (connection, sync, notifications)

**Examples:**
- **Text Editor:** Line/column number, character encoding, language mode
- **Web Browser:** URL on hover, page load progress
- **File Manager:** "12 items, 3 selected, 1.2 GB"

**Best Practices:**
- Always visible (not floating)
- Update in real-time (status changes)
- Clear, concise text
- Click to show details (e.g., click errors: show error list)


## Platform-Specific Considerations

### macOS

**Design Language:**
- San Francisco font (system default)
- Translucent sidebars (vibrancy effect)
- Full-height sidebars (no gap at top)
- Traffic light buttons (close, minimize, maximize) in title bar

**Menu Bar:**
- Global menu bar (top of screen)
- Application menu (app name): About, Preferences, Hide, Quit
- Window menu: List of open windows
- Help menu: Searchable help

**Keyboard Shortcuts:**
- Cmd (⌘) key (not Ctrl)
- Cmd+Q: Quit app
- Cmd+H: Hide app
- Cmd+W: Close window
- Cmd+`: Cycle app windows

**Window Management:**
- Green button: Full-screen or tile (hover for options)
- Yellow button: Minimize to Dock
- Red button: Close window (doesn't quit app)

**Integration:**
- Spotlight search (index app content)
- Quick Look (preview files with Space)
- Share menu (system share sheet)
- Services menu (inter-app actions)

**Best Practices:**
- Use NSToolbar for native toolbar
- Vibrancy for sidebars (if appropriate)
- Support full-screen mode
- App continues running with no windows (dock icon)

### Windows

**Design Language:**
- Segoe UI font (system default)
- Fluent Design (transparency, depth, motion)
- Ribbon UI (Office-style) or classic menu bar
- Window controls (minimize, maximize, close) in title bar

**Menu Bar:**
- In window (local to window)
- File menu includes "Exit"
- Edit menu includes "Options" (preferences)
- Mnemonics (underlined letters, Alt+F for File)

**Keyboard Shortcuts:**
- Ctrl key (not Cmd)
- Alt+F4: Close window (quit if last window)
- Ctrl+W: Close tab/document
- Win+Arrow: Snap window (left, right, up, down)

**Window Management:**
- Maximize button: Full-screen or snap options
- Minimize button: To taskbar
- Close button: Close window (quit app if last window)

**Integration:**
- Taskbar integration (pinned apps, jump lists)
- Notifications (Action Center)
- Share target (Windows 10+)
- File Explorer integration (context menu)

**Best Practices:**
- Support snap assist (Win+Arrow)
- Jump lists for recent files (taskbar right-click)
- Native file dialogs (IFileOpenDialog)
- Respect system theme (light/dark mode)

### Linux

**Design Language:**
- Varies by desktop environment (GNOME, KDE, XFCE)
- GTK or Qt toolkit
- Follows desktop environment theme

**GNOME (GTK):**
- Header bar (title bar + toolbar combined)
- Hamburger menu or app menu
- Clean, minimal design

**KDE (Qt):**
- Traditional menu bar
- More customization options
- Similar to Windows layout

**Keyboard Shortcuts:**
- Ctrl key (like Windows)
- Alt+F4: Close window (common)
- Ctrl+Q: Quit app (some apps)

**Window Management:**
- Varies by window manager
- GNOME: Super+Left/Right (snap half)
- KDE: Similar to Windows
- Tiling WMs: i3, bspwm (keyboard-driven)

**Integration:**
- .desktop file (app launcher)
- File manager integration (Nautilus, Dolphin)
- System tray (varies by DE)

**Best Practices:**
- Use GTK or Qt (native look)
- Respect desktop theme
- Standard .desktop file (for app menu)
- Follow freedesktop.org standards

### Cross-Platform (Electron, Qt, etc.)

**Unified Design:**
- Custom UI (not native controls)
- Consistent across platforms
- Brand identity maintained

**Platform Adaptations:**
- Detect platform, adjust shortcuts (Cmd vs Ctrl)
- Menu bar position (global on macOS, local on Windows/Linux)
- Native file dialogs (use platform dialogs)
- Window controls (platform-specific styling)

**Examples:**
- **VS Code (Electron):** Unified UI, platform-specific shortcuts
- **Figma (Web/Electron):** Custom UI, platform shortcuts
- **Spotify (Electron):** Custom design, cross-platform

**Best Practices:**
- Consistent design language
- Platform-specific keyboard shortcuts
- Native file dialogs (better UX)
- Support platform gestures (macOS trackpad)
- Respect system themes (light/dark mode)


## Anti-Patterns

### Priority 0 (Critical - Never Do)

**1. No keyboard access to core features:**
- **Problem:** Mouse-only interface, excludes keyboard users
- **Impact:** Accessibility fail, WCAG violation, frustrates power users
- **Fix:** Tab navigation, keyboard shortcuts for all actions, focus indicators

**2. Non-standard shortcuts (Ctrl+S not saving):**
- **Problem:** Breaks user expectations, muscle memory fails
- **Impact:** Errors, frustration, slow workflow
- **Fix:** Use platform conventions (Cmd+S / Ctrl+S for save)

**3. Lost work (no autosave, no crash recovery):**
- **Problem:** Users lose hours of work on crash or accidental close
- **Impact:** Extreme frustration, negative reviews, lost trust
- **Fix:** Autosave drafts every 2-5 minutes, crash recovery, unsaved changes warning

**4. Unresponsive UI (blocking on long operations):**
- **Problem:** Entire app freezes during save, export, or processing
- **Impact:** Feels broken, users force-quit, lose work
- **Fix:** Background threads, progress indicator, cancelable operations

### Priority 1 (High - Avoid)

**5. Modal dialogs for every action:**
- **Problem:** Constant interruptions, slow workflow
- **Impact:** Annoying, inefficient, breaks concentration
- **Fix:** Inline editing, modeless panels, fewer confirmations

**6. No workspace persistence (lose layout on restart):**
- **Problem:** Users reconfigure panels and windows every session
- **Impact:** Time waste, frustration, inconsistent experience
- **Fix:** Save window positions, panel layout, open tabs between sessions

**7. Hidden essential features (no discoverability):**
- **Problem:** Power features exist but users never find them
- **Impact:** Underutilized features, users leave for competitors
- **Fix:** Command palette, tooltips, onboarding hints, help documentation

**8. Inconsistent UI (different patterns in different areas):**
- **Problem:** Save button top-right in one dialog, bottom-left in another
- **Impact:** Cognitive load, users must relearn each screen
- **Fix:** Design system, consistent button placement, standard patterns

### Priority 2 (Medium - Be Cautious)

**9. Too many customization options (overwhelming):**
- **Problem:** 500 settings in preferences, users lost
- **Impact:** Decision paralysis, never customize, confused
- **Fix:** Sensible defaults, progressive disclosure, basic vs advanced tabs

**10. Non-native file dialogs (platform inconsistency):**
- **Problem:** Custom file picker doesn't match OS
- **Impact:** Unfamiliar, lacks OS features (favorites, recent), confusing
- **Fix:** Use platform-native dialogs (NSOpenPanel, IFileOpenDialog)

**11. Reinventing standard controls:**
- **Problem:** Custom buttons, checkboxes, dropdowns that look/behave differently
- **Impact:** Users must learn new patterns, accessibility issues
- **Fix:** Use platform controls or follow established design systems

**12. Over-reliance on mouse (not keyboard-accessible):**
- **Problem:** Hover-only actions, drag-only reordering, no keyboard alternative
- **Impact:** Excludes keyboard users, slow for power users
- **Fix:** Keyboard shortcuts for all actions, Tab navigation, arrow keys


## Practical Application

### Workflow 1: Document Editor Design

**Step 1: Define Document Model**
- Single-document (one window per file) or multi-tab?
- Decision: Multi-tab (many documents, single window)
- Rationale: Easier to organize than scattered windows

**Step 2: Design Multi-Tab Interface**
- Tabs at top (standard)
- Cmd+T: New tab, Cmd+W: Close tab
- Cmd+1-9: Jump to tab number
- Drag tabs to reorder
- Unsaved indicator (dot on tab)

**Step 3: Implement Keyboard Shortcuts**
- File: Cmd+N, Cmd+O, Cmd+S, Cmd+Shift+S
- Edit: Cmd+Z, Cmd+Shift+Z, Cmd+X/C/V
- Find: Cmd+F
- Show in tooltip and menu

**Step 4: Add Autosave**
- Save draft every 2 minutes
- Status indicator: "All changes saved"
- Crash recovery: Restore last autosave on relaunch
- Unsaved changes warning on quit

**Step 5: Export and Print**
- File > Export: PDF, DOCX, HTML
- File > Print: Native print dialog
- File > Page Setup: Margins, orientation

**Step 6: Preferences**
- Cmd+,: Open preferences
- Tabs: General, Appearance, Editor
- Settings: Font, theme, autosave interval
- Reset to defaults button

### Workflow 2: Design Tool (Creative Professional)

**Step 1: Canvas + Panels Layout**
- Large canvas area (center)
- Tool palette (left sidebar)
- Properties inspector (right sidebar)
- Layers panel (right, tabbed with properties)

**Step 2: Tool Palettes**
- Tools: Select, Pen, Brush, Text, Shape
- Single-click to select tool
- Keyboard shortcuts (V, P, B, T, R)
- Active tool highlighted

**Step 3: Dockable Panels**
- Drag panel header to move
- Dock to left/right/bottom
- Stack panels in tabs
- View menu: Show/hide panels
- Remember arrangement between sessions

**Step 4: Workspace Layouts**
- Built-in: Default, Painting, Photo Editing
- Custom: User saves current layout
- View > Workspace > [name]
- Quick-switch dropdown in toolbar

**Step 5: Plugin Support**
- Plugin menu: List installed plugins
- Plugin manager: Browse, install, disable
- API: Canvas access, tool creation, file export
- Community marketplace

**Step 6: Export and Batch Processing**
- Export: PNG, JPG, SVG, PDF
- Batch export: All artboards or selected
- Resize on export (1x, 2x, 3x)
- Save export preset (for reuse)

### Workflow 3: IDE / Code Editor

**Step 1: File Tree + Editor Layout**
- File tree (left sidebar)
- Editor area (center, tabbed)
- Terminal (bottom panel, toggleable)
- Debug panels (right sidebar, shown when debugging)

**Step 2: Editor Tabs**
- Multiple files open in tabs
- Cmd+P: Quick open file (fuzzy search)
- Cmd+Tab: Switch tabs
- Cmd+W: Close tab
- Split editor (side-by-side, up to 3 columns)

**Step 3: Command Palette**
- Cmd+Shift+P: All commands
- Fuzzy search: "git commit", "format document"
- Shows keyboard shortcuts
- Recent commands at top

**Step 4: Integrated Debugger**
- Breakpoints: Click gutter to toggle
- Debug panel: Variables, call stack, watch
- Keyboard: F5 (start), F10 (step over), F11 (step into)
- Inline variable values during debug

**Step 5: Extensions**
- Extensions sidebar: Browse, install
- Language support (Python, Go, Rust)
- Themes (dark mode, light mode, custom)
- Linters, formatters, git tools
- Auto-update extensions

**Step 6: Terminal Integration**
- Integrated terminal (Cmd+`)
- Split terminals (multiple shells)
- Run tasks (npm, make, cargo)
- Link file paths (click to open)

### Workflow 4: Data Analysis Tool

**Step 1: Data Import**
- File > Import: CSV, Excel, SQL, API
- Import wizard: Choose columns, data types
- Preview data (first 10 rows)
- Confirm import

**Step 2: Data Table View**
- Sortable columns (click header)
- Filter rows (search, column filters)
- Edit cells inline (click to edit)
- Add/remove columns

**Step 3: Charts + Visualizations**
- Drag column to chart area: Create chart
- Chart types: Line, bar, scatter, pie
- Customize: Colors, labels, axes
- Export chart (PNG, SVG)

**Step 4: Batch Processing**
- Select multiple rows (Cmd+Click)
- Batch actions: Delete, tag, export
- Apply formula to column (fill down)
- Merge datasets (join tables)

**Step 5: Export Results**
- Export to CSV (all rows or filtered)
- Export to Excel (multiple sheets)
- Export chart images (PNG, SVG)
- Generate report (PDF with charts + tables)

**Step 6: Scripting**
- Built-in Python console
- Run scripts on data (pandas, numpy)
- Save scripts (reusable workflows)
- Script library (community scripts)


## Related Skills

**Core Lyra Skills:**
- **`lyra/ux-designer/visual-design-foundations`**: Visual hierarchy for complex interfaces, typography for dense data, color for status indicators
- **`lyra/ux-designer/information-architecture`**: Menu structure, panel organization, command categorization
- **`lyra/ux-designer/interaction-design-patterns`**: Keyboard shortcuts, button states, feedback patterns, focus indicators
- **`lyra/ux-designer/accessibility-and-inclusive-design`**: Keyboard navigation, screen reader support, focus management, WCAG compliance

**Platform Skills:**
- **`lyra/ux-designer/web-application-design`**: Shared patterns (command palette, keyboard shortcuts), responsive considerations for Electron apps
- **`lyra/ux-designer/mobile-design-patterns`**: Touch considerations if desktop app has touch support (Surface, touchscreen laptops)

**Cross-Faction:**
- **`muna/technical-writer/clarity-and-style`**: Microcopy for menus, tooltips, error messages, help documentation
- **`ordis/security-architect/secure-authentication-patterns`**: Desktop auth flows, credential storage, session management


## Additional Resources

**Design Guidelines:**
- **macOS Human Interface Guidelines:** https://developer.apple.com/design/human-interface-guidelines/macos
- **Windows Fluent Design System:** https://www.microsoft.com/design/fluent/
- **GNOME Human Interface Guidelines:** https://developer.gnome.org/hig/
- **KDE Human Interface Guidelines:** https://develop.kde.org/hig/

**Desktop Patterns:**
- Menu bars: File, Edit, View, Window, Help (standard structure)
- Keyboard shortcuts: Platform conventions (Cmd vs Ctrl)
- Window management: Multi-monitor, persistence, docking
- Context menus: Right-click for common actions

**Keyboard Standards:**
- **Essential:** Cmd+N, Cmd+O, Cmd+S, Cmd+W, Cmd+Q
- **Edit:** Cmd+Z, Cmd+Shift+Z, Cmd+X/C/V, Cmd+A
- **Navigation:** Cmd+F, Cmd+Tab, Tab/Shift+Tab
- **Platform differences:** Cmd (macOS) vs Ctrl (Windows/Linux)

**Workspace Customization:**
- Dockable panels (drag to dock, stack in tabs)
- Toolbar customization (add/remove tools)
- Layout presets (save and load)
- Themes (light, dark, high contrast)

**Power User Features:**
- Command palette: Cmd+Shift+P (fuzzy search all commands)
- Scripting: Python, JavaScript, Lua APIs
- Batch operations: Process multiple files/items
- Extensions: Plugin system, marketplace

**Accessibility Standards:**
- **WCAG 2.1 AA:** Keyboard navigation, focus indicators, contrast
- **Keyboard-first:** All actions accessible via keyboard
- **Focus indicators:** Visible 2px outline, high contrast
- **Screen readers:** ARIA labels, semantic structure
