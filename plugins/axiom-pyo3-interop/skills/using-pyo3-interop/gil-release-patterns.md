# Attachment, detachment and GIL discipline

Establish the Python/PyO3 version and interpreter build. A `Python<'py>` token represents valid interpreter attachment; in a GIL-enabled build it also represents holding that interpreter's GIL. In a free-threaded build it does not make arbitrary shared data exclusive.

## Native work boundary

Extract or validate inputs while attached, perform permitted native work detached, and reattach to construct Python results or call Python. Use the installed API's `Ungil`/closure requirements rather than treating `Send` as a timeless complete safety rule. Never access interpreter-bound values from detached work.

Detach for long native work and blocking waits where the API permits. Measure contention/latency rather than counting source lines or applying a universal microsecond threshold. Detachment remains meaningful on free-threaded Python: runtime-wide synchronization can otherwise hang or deadlock. See [PyO3 free-threading guidance](https://pyo3.rs/main/free-threading.html#detaching-to-avoid-hangs-and-deadlocks).

## Failure checks

- Do not hold a native mutex while waiting for a thread that needs interpreter attachment, or call Python under a lock without checking re-entry and lock order.
- Retaining a Python owner preserves lifetime; it does not prevent another thread from mutating a buffer. See [NumPy borrowing](numpy-buffer-protocol.md).
- Native work may continue after caller cancellation. Define stop signaling, result ownership and shutdown behavior explicitly.
- Periodic callbacks require valid attachment, propagated errors and an interruption policy. Their frequency is a workload decision.
- Free-threaded builds increase concurrent access: audit globals, unsafe GIL-based assumptions, mutable pyclasses and callbacks. Module thread-safety declarations and build cfg are version-specific; do not invent a Cargo feature to enable safety.
- Detachment does not undo effects or grant permission to retain a borrow beyond its owner. Interpreter teardown is a separate ownership boundary.

## Validate

Exercise concurrent calls, re-entry, blocking waits, cancellation and shutdown under supported builds. `/audit-gil-discipline` supplies a scoped investigation contract, not a source-line-count static proof. Record the tested matrix and unresolved native/thread-safety assumptions.
