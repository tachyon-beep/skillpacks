
# Textual TUI Development

## Overview

**Core Principle:** Textual apps are built with composition, not inheritance. Create apps by composing widgets in `compose()`, styling with Textual CSS, and reacting to state changes through reactive attributes. Think like modern web development, but for the terminal.

Textual is a Python framework for building sophisticated terminal user interfaces (TUIs). It uses an async architecture powered by asyncio, a CSS-like styling system, and a reactive data binding model inspired by modern frontend frameworks. The most common mistakes: forgetting to `await` mount operations, blocking the event loop, and not understanding the reactive lifecycle.

## When to Use

**Use this skill when:**
- Building terminal applications with Textual
- "compose() not showing widgets"
- "reactive not updating"
- "Textual CSS not working"
- "How to test Textual apps?"
- "Widget events not firing"
- "run_test() issues"

**Don't use when:**
- General Python async patterns (use async-patterns-and-concurrency)
- Type errors (use resolving-mypy-errors)
- Project setup without Textual (use project-structure-and-tooling)

**Symptoms triggering this skill:**
- Widgets not appearing after compose
- UI not updating when data changes
- NoMatches errors when querying widgets
- Test failures with Pilot


## App Structure

### Basic App Pattern

```python
# ✅ CORRECT: Standard app structure
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static

class MyApp(App):
    """A well-structured Textual app."""

    CSS_PATH = "my_app.tcss"  # External CSS file
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Dark Mode"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the widget tree."""
        yield Header()
        yield Static("Hello, World!", id="greeting")
        yield Footer()

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.theme = "textual-dark" if self.theme == "textual-light" else "textual-light"

if __name__ == "__main__":
    app = MyApp()
    app.run()
```

**Why this matters**: The `compose()` method defines the widget tree. Widgets are yielded, not returned. External CSS keeps styling separate from logic.

### Compose vs Mount

```python
# ❌ WRONG: Mounting widgets that should be composed
class BadApp(App):
    def on_mount(self) -> None:
        self.mount(Header())  # Should be in compose()
        self.mount(Static("Content"))

# ✅ CORRECT: Use compose() for initial structure
class GoodApp(App):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Content")

# ✅ CORRECT: Use mount() for dynamic additions
class DynamicApp(App):
    def compose(self) -> ComposeResult:
        yield Static("Initial content", id="container")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        # Mount dynamically in response to events
        await self.query_one("#container").mount(Static("New item"))
```

**Why this matters**: `compose()` defines the initial structure. `mount()` adds widgets dynamically at runtime. Await mount when you need to query the mounted widget immediately.

### Container Patterns

```python
from textual.containers import Container, Horizontal, Vertical, Grid

# ✅ CORRECT: Using containers for layout
class LayoutApp(App):
    CSS = """
    #sidebar { width: 20; }
    #main { width: 1fr; }
    """

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Container(Static("Sidebar"), id="sidebar")
            yield Container(Static("Main Content"), id="main")

# ✅ CORRECT: Context manager for nesting
class NestedApp(App):
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Header()
            with Horizontal():
                yield Button("Save")
                yield Button("Cancel")
            yield Footer()
```

**Why this matters**: Containers organize widgets spatially. Use `with` syntax for nesting - cleaner than manual parent assignment.


## Textual CSS

### CSS Basics

```css
/* my_app.tcss */

/* Target by widget type */
Button {
    width: 16;
    height: 3;
}

/* Target by ID */
#sidebar {
    width: 25;
    background: $surface;
}

/* Target by class */
.highlighted {
    background: $accent;
    text-style: bold;
}

/* Descendant selector */
#main Static {
    padding: 1 2;
}

/* Pseudo-classes */
Button:hover {
    background: $primary-lighten-1;
}

Button:focus {
    border: heavy $accent;
}
```

### CSS Variables and Theming

```python
# ✅ CORRECT: Using CSS variables
class ThemedApp(App):
    CSS = """
    Screen {
        background: $background;
    }

    .panel {
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    .error {
        color: $error;
    }
    """
```

**Why this matters**: Textual CSS uses `$` for design tokens that adapt to themes. Use them instead of hardcoded colors for theme compatibility.

### Common CSS Properties

```css
/* Layout */
width: 50%;           /* Percentage */
width: 20;            /* Cells */
width: 1fr;           /* Fraction of remaining space */
width: auto;          /* Fit content */
height: 100%;

/* Box model */
padding: 1 2;         /* Vertical horizontal */
margin: 1;
border: solid green;
border: heavy $primary;

/* Positioning */
dock: top;            /* Dock to edge */
dock: left;
offset: 5 10;         /* X Y offset */

/* Grid layout */
layout: grid;
grid-size: 3 2;       /* Columns rows */
grid-gutter: 1;
column-span: 2;

/* Text */
text-align: center;
text-style: bold italic;
color: $text;
background: $surface;

/* Visibility */
display: none;        /* Remove from layout */
visibility: hidden;   /* Keep space, hide content */
```


## Reactivity

### Reactive Attributes

```python
from textual.reactive import reactive, var

# ❌ WRONG: Regular attributes don't trigger updates
class BadWidget(Widget):
    def __init__(self):
        super().__init__()
        self.count = 0  # Changes won't update UI

    def render(self) -> str:
        return f"Count: {self.count}"

# ✅ CORRECT: Use reactive for automatic refresh
class GoodWidget(Widget):
    count = reactive(0)  # Changes trigger render()

    def render(self) -> str:
        return f"Count: {self.count}"

# ✅ CORRECT: Use var when you don't need refresh
class StateWidget(Widget):
    is_loading = var(False)  # State without refresh
    data = reactive("")      # Data that should refresh
```

**Why this matters**: `reactive` triggers `render()` on change. `var` stores state without refresh. Choose based on whether the UI needs to update.

### Watch Methods

```python
class WatchExample(Widget):
    count = reactive(0)
    name = reactive("default")

    # ✅ CORRECT: Watch with new value only
    def watch_count(self, count: int) -> None:
        self.log(f"Count changed to {count}")

    # ✅ CORRECT: Watch with old and new values
    def watch_name(self, old_name: str, new_name: str) -> None:
        self.log(f"Name changed from {old_name} to {new_name}")
```

### Validation

```python
class ValidatedWidget(Widget):
    percentage = reactive(0)

    def validate_percentage(self, value: int) -> int:
        """Clamp value to valid range."""
        return max(0, min(100, value))

    # Setting self.percentage = 150 becomes self.percentage = 100
```

### Compute Methods

```python
from textual.color import Color

class ComputedWidget(Widget):
    red = reactive(0)
    green = reactive(0)
    blue = reactive(0)
    color = reactive(Color.parse("black"))

    def compute_color(self) -> Color:
        """Automatically recomputed when red/green/blue change."""
        return Color(self.red, self.green, self.blue)
```

**Why this matters**: Compute methods auto-update when dependencies change. Textual tracks which reactives are read during computation.

### Data Binding

```python
class ChildWidget(Widget):
    value = reactive(0)

class ParentApp(App):
    count = reactive(0)

    def compose(self) -> ComposeResult:
        # Bind parent's count to child's value
        yield ChildWidget().data_bind(value=ParentApp.count)

    def action_increment(self) -> None:
        self.count += 1  # Child's value updates automatically
```

### Setting Reactives Without Watchers

```python
class SafeWidget(Widget):
    name = reactive("")

    def __init__(self, name: str) -> None:
        super().__init__()
        # ❌ WRONG: Watch method may fail before mount
        # self.name = name

        # ✅ CORRECT: Set without triggering watch
        self.set_reactive(SafeWidget.name, name)
```

**Why this matters**: In `__init__`, the widget isn't mounted yet. Watch methods that query the DOM will fail. Use `set_reactive` to initialize without triggering watchers.


## Event Handling

### Message Handlers

```python
from textual.widgets import Button, Input

class EventApp(App):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type here", id="my-input")
        yield Button("Submit", id="submit")

    # ✅ CORRECT: Handler by message type
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self.process_input()

    # ✅ CORRECT: Handler by message type and selector
    @on(Input.Submitted, "#my-input")
    def handle_input_submit(self, event: Input.Submitted) -> None:
        self.log(f"Input submitted: {event.value}")

    # ✅ CORRECT: Multiple selectors
    @on(Button.Pressed, "#save, #cancel")
    def handle_dialog_buttons(self, event: Button.Pressed) -> None:
        pass
```

### Custom Messages

```python
from textual.message import Message

class CustomWidget(Widget):
    class DataLoaded(Message):
        """Sent when data finishes loading."""
        def __init__(self, data: list) -> None:
            self.data = data
            super().__init__()

    async def load_data(self) -> None:
        data = await fetch_data()
        self.post_message(self.DataLoaded(data))

class MyApp(App):
    def on_custom_widget_data_loaded(self, message: CustomWidget.DataLoaded) -> None:
        self.log(f"Received {len(message.data)} items")
```

### Actions and Bindings

```python
class ActionApp(App):
    BINDINGS = [
        ("ctrl+s", "save", "Save"),
        ("ctrl+q", "quit", "Quit"),
        Binding("d", "toggle_dark", "Dark Mode", show=False),
    ]

    def action_save(self) -> None:
        """Called when ctrl+s pressed or action invoked."""
        self.save_document()

    def action_toggle_dark(self) -> None:
        self.theme = "textual-dark" if self.theme == "textual-light" else "textual-light"
```


## Testing

### Basic Test Pattern

```python
import pytest
from textual.pilot import Pilot

async def test_app_starts():
    """Test that app composes correctly."""
    app = MyApp()
    async with app.run_test() as pilot:
        # App is now running in headless mode
        assert app.query_one("#greeting")

async def test_button_click():
    """Test clicking a button."""
    app = MyApp()
    async with app.run_test() as pilot:
        await pilot.click("#my-button")
        assert app.state == "clicked"

async def test_key_press():
    """Test keyboard input."""
    app = MyApp()
    async with app.run_test() as pilot:
        await pilot.press("q")
        # Or multiple keys:
        await pilot.press("h", "e", "l", "l", "o")
```

### Testing with Pauses

```python
async def test_async_update():
    """Test that requires message processing."""
    app = MyApp()
    async with app.run_test() as pilot:
        app.trigger_async_update()

        # ❌ WRONG: Assert immediately may fail
        # assert app.updated  # May be False still

        # ✅ CORRECT: Pause to let messages process
        await pilot.pause()
        assert app.updated
```

**Why this matters**: Textual is async. Actions may not complete immediately. `pilot.pause()` waits for pending messages.

### Snapshot Testing

```python
# Install: pip install pytest-textual-snapshot

def test_app_appearance(snap_compare):
    """Test visual appearance with snapshot."""
    assert snap_compare("my_app.py")

def test_with_interaction(snap_compare):
    """Snapshot after some interaction."""
    assert snap_compare("my_app.py", press=["tab", "enter"])

def test_with_setup(snap_compare):
    """Snapshot with custom setup."""
    async def setup(pilot):
        await pilot.click("#load-button")
        await pilot.pause()

    assert snap_compare("my_app.py", run_before=setup)
```


## Custom Widgets

### Basic Custom Widget

```python
from textual.widget import Widget
from textual.reactive import reactive

class Counter(Widget):
    """A simple counter widget."""

    DEFAULT_CSS = """
    Counter {
        height: 3;
        border: solid $primary;
        padding: 0 1;
    }
    """

    count = reactive(0)

    def render(self) -> str:
        return f"Count: {self.count}"

    def increment(self) -> None:
        self.count += 1
```

### Widget with Composition

```python
class SearchBox(Widget):
    """A search box with button."""

    DEFAULT_CSS = """
    SearchBox {
        layout: horizontal;
        height: auto;
    }
    SearchBox Input {
        width: 1fr;
    }
    """

    class Submitted(Message):
        def __init__(self, query: str) -> None:
            self.query = query
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search...")
        yield Button("Go", id="search-btn")

    @on(Button.Pressed, "#search-btn")
    @on(Input.Submitted)
    def handle_submit(self) -> None:
        query = self.query_one(Input).value
        self.post_message(self.Submitted(query))
```


## Common Anti-Patterns

### Blocking the Event Loop

```python
# ❌ WRONG: Blocking I/O in event handler
import requests

class BadApp(App):
    def on_button_pressed(self, event: Button.Pressed) -> None:
        # This blocks the entire UI!
        response = requests.get("https://api.example.com/data")
        self.update(response.json())

# ✅ CORRECT: Use async HTTP client
import httpx

class GoodApp(App):
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.example.com/data")
            self.update(response.json())

# ✅ CORRECT: Use workers for blocking operations
from textual import work

class WorkerApp(App):
    @work(thread=True)
    def fetch_data(self) -> None:
        # Runs in thread, doesn't block UI
        response = requests.get("https://api.example.com/data")
        self.call_from_thread(self.update, response.json())
```

### NoMatches Errors

```python
# ❌ WRONG: Querying before mount completes
class BadApp(App):
    def on_key(self) -> None:
        self.mount(MyWidget())
        self.query_one(MyWidget).do_something()  # NoMatches!

# ✅ CORRECT: Await mount
class GoodApp(App):
    async def on_key(self) -> None:
        await self.mount(MyWidget())
        self.query_one(MyWidget).do_something()  # Works!
```

### Storing Widget References with Recompose

```python
# ❌ WRONG: Storing references with recompose=True
class BadWidget(Widget):
    data = reactive([], recompose=True)

    def __init__(self):
        super().__init__()
        self.items = []  # Will become stale!

    def compose(self) -> ComposeResult:
        for item in self.data:
            widget = ItemWidget(item)
            self.items.append(widget)  # Stale after recompose
            yield widget

# ✅ CORRECT: Query when needed
class GoodWidget(Widget):
    data = reactive([], recompose=True)

    def compose(self) -> ComposeResult:
        for item in self.data:
            yield ItemWidget(item)

    def get_items(self) -> list[ItemWidget]:
        return list(self.query(ItemWidget))
```


## Workers (Background Tasks)

```python
from textual import work
from textual.worker import Worker

class WorkerApp(App):
    # Thread worker for blocking I/O
    @work(thread=True)
    def blocking_fetch(self, url: str) -> None:
        response = requests.get(url)  # Blocking OK in thread
        self.call_from_thread(self.handle_response, response)

    # Async worker for async I/O
    @work
    async def async_fetch(self, url: str) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            self.handle_response(response)

    # Exclusive worker (cancels previous)
    @work(exclusive=True)
    async def search(self, query: str) -> None:
        # Previous search cancelled if new one starts
        results = await self.api.search(query)
        self.show_results(results)
```


## Screens and Navigation

```python
from textual.screen import Screen, ModalScreen

class SettingsScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Static("Settings")
        yield Button("Back", id="back")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()

class ConfirmDialog(ModalScreen[bool]):
    """Modal that returns True/False."""

    def compose(self) -> ComposeResult:
        yield Static("Are you sure?")
        yield Button("Yes", id="yes")
        yield Button("No", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

class MyApp(App):
    SCREENS = {"settings": SettingsScreen}

    def action_settings(self) -> None:
        self.push_screen("settings")

    async def action_confirm(self) -> None:
        result = await self.push_screen_wait(ConfirmDialog())
        if result:
            self.do_action()
```

