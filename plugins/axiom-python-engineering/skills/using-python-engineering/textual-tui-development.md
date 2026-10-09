# Textual lifecycle and interaction checks

Verify Textual APIs against the project's installed version. Work from the actual app, widget tree and event ownership; a TUI needs terminal and lifecycle evidence as well as isolated function tests.

## Lifecycle

- Distinguish composition from mounting. Query child widgets only when they exist, and avoid depending on incidental composition order.
- Reactive updates and watchers can trigger additional changes; guard against recursive updates and inconsistent intermediate state.
- Keep blocking work off the event loop. Workers need explicit cancellation, stale-result handling and a shutdown owner.
- A result for an old selection must not overwrite the current screen. Carry stable request/selection identity through asynchronous work.
- Screen push/pop and modal dismissal must restore focus and release resources. Check repeated open/close, failure and cancellation.
- Preserve terminal state on normal exit, errors and interruption. Cleanup belongs to the owning context.

## Interaction and rendering

Check keyboard-only navigation, focus visibility, resize, narrow terminals, scroll state, input validation and loading/error/empty states. CSS/layout changes can hide controls or create scroll traps even when business logic passes.

Use the project's Pilot/test harness where available; exercise real terminal behavior for claims the harness cannot establish. Document which interactions and dimensions were checked. Introduce a new component or visual redesign only when it serves the requested behavior.
