# Async and Concurrency

## Overview

**Core Principle:** Async Rust is a zero-cost, poll-driven concurrency model built on the `Future` trait. It enables high-throughput I/O without the overhead of OS threads. The runtime is not in `std` — you choose and bring it. Tokio is the production default.

This sheet assumes you already know `async`/`await` syntax. The goal here is the mental model under the syntax: why futures are state machines, why executors exist as crates, how structured concurrency prevents resource leaks, and where Rust's ownership model creates async-specific traps that don't exist in other languages (holding a `std::sync::Mutex` guard across an `.await` is the canonical one).

Async is appropriate for I/O-bound workloads. It is not a substitute for parallelism. CPU-bound work still needs threads or `rayon`. The concurrency here is cooperative, not preemptive.

## When to Use

Use this sheet when:

- "Why is `spawn` requiring `'static` bounds on my closure?"
- "My future is not `Send` — what does that mean and how do I fix it?"
- "`MutexGuard` cannot be held across an `await` point."
- "Which channel do I use: `mpsc`, `oneshot`, `broadcast`, or `watch`?"
- "How do I implement graceful shutdown?"
- "`JoinSet` vs spawning tasks individually — what's the difference?"
- "How do I call blocking code from async?"
- "What runtime should I use for a CLI, a service, a WASM module?"
- "Native async fn in traits — do I still need `async-trait`?"
- "My service is slow under load even with async — where do I start?"

## When NOT to Use

- **CPU-bound parallelism**: Use `rayon`, `std::thread`, or `tokio::task::spawn_blocking`. Async does not speed up CPU work.
- **`Future`/`await` syntax basics**: Covered by the Rust Book. This sheet assumes you know `async fn` compiles.
- **`Pin` and self-referential types**: See [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — the `Pin` section is the prerequisite for understanding why async state machines are `!Unpin`.
- **Send/Sync rules for non-async code**: See [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — the Send and Sync section.
- **Trait object interaction with async** (object safety, `dyn Future`): See [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) before this sheet's async trait section.
- **Performance profiling of async tasks**: See [performance-and-profiling.md](performance-and-profiling.md) for Tokio console and async flamegraph tooling.

## Rust's Async Model

### Futures as State Machines

When the compiler encounters an `async fn`, it rewrites it into a state machine struct. Each `.await` point becomes a state boundary. The struct stores all live local variables at each yield point.

```rust
// What you write:
async fn fetch_and_parse(url: &str) -> Result<Vec<u8>, reqwest::Error> {
    let response = reqwest::get(url).await?;   // state boundary 0→1
    let body = response.bytes().await?;         // state boundary 1→2
    Ok(body.to_vec())
}

// What the compiler conceptually generates (simplified):
enum FetchAndParseState<'a> {
    Start { url: &'a str },
    WaitingForResponse { url: &'a str, /* future */ },
    WaitingForBytes { /* response, future */ },
    Done,
}

// The generated type implements Future:
// impl<'a> Future for FetchAndParseState<'a> {
//     type Output = Result<Vec<u8>, reqwest::Error>;
//     fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> { ... }
// }
```

The `poll` model is explicit: the executor calls `poll` on a future. If the future is ready, it returns `Poll::Ready(output)`. If it is waiting for I/O, it registers a waker with the reactor and returns `Poll::Pending`. The executor will call `poll` again when the waker fires. No thread is blocked in between.

### Pin, `!Unpin`, and Pin Projection in Practice

An `async fn` that holds a reference across `.await` desugars into a state machine that stores the reference AND the referent in the same struct. Moving that struct would invalidate the reference — hence the generated future is `!Unpin`. This is the async-specific *application* of Pin; for the underlying move-semantics mental model, see [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md).

```rust
// The async fn you write:
async fn borrow_across_await() {
    let data = vec![1, 2, 3];
    let first = &data[0];               // borrow into `data`
    tokio::task::yield_now().await;     // suspension point — future is stored somewhere
    println!("{first}");                // reference must still be valid after resume
}

// Conceptual desugaring (not the real generator, but the shape):
// struct BorrowAcrossAwait {
//     data: Vec<i32>,
//     first: *const i32,   // points INTO `data` above — self-referential
//     state: State,
// }
// Moving this struct after `first` is initialized would leave `first` dangling.
// The compiler marks it `!Unpin` so the type system forbids safe moves once pinned.
```

Once a future is pinned (e.g. inside `Box::pin(...)` or on the executor's task slab), you can only reach its fields through `Pin<&mut Self>`. You cannot get an unrestricted `&mut Self`, because that would let you `std::mem::swap` the struct and break the self-reference. A manual `Future` impl therefore looks like:

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

struct MyFuture { /* fields, possibly including a child future */ }

impl Future for MyFuture {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Can't just write `self.inner.poll(cx)` — we'd need a `Pin<&mut Inner>`
        // for a structurally-pinned field. Projection is how you get there.
        Poll::Ready(())
    }
}
```

**`pin-project-lite` is the safe, stable-Rust answer.** It generates a projection method that hands you `Pin<&mut T>` for fields marked `#[pin]` and plain `&mut T` for the rest — and enforces the structural-pin rules at compile time:

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pin_project_lite::pin_project! {
    struct Timed<F> {
        #[pin] inner: F,       // structurally pinned — projection yields Pin<&mut F>
        polls: u64,            // not pinned — freely movable
    }
}

impl<F: Future> Future for Timed<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();       // generated: { inner: Pin<&mut F>, polls: &mut u64 }
        *this.polls += 1;
        this.inner.poll(cx)
    }
}
```

Why not just do the projection by hand with unsafe? You can — but the invariant ("never move out of a structurally-pinned field, never re-pin a non-pinned field as pinned, uphold drop ordering") is easy to violate silently:

- **WRONG — unsafe projection that breaks pin guarantees:**

    ```rust
    // impl<F: Future> Future for Timed<F> {
    //     fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
    //         let this = unsafe { self.get_unchecked_mut() };  // now have &mut Self
    //         this.polls += 1;
    //         // Easy mistake: someone later adds `mem::swap(&mut this.inner, &mut other)`
    //         // and the compiler will not stop them. Self-reference invalidated.
    //         unsafe { Pin::new_unchecked(&mut this.inner) }.poll(cx)
    //     }
    // }
    ```

- **CORRECT:** use `pin_project!` / `pin_project_lite::pin_project!`. The generated `project()` enforces the discipline mechanically; any field access that would violate pinning fails to compile.

**When you actually need this:** nearly always you don't — just write `async fn` and let the compiler generate the state machine. Reach for manual `Future` + pin projection only for custom executor primitives, combinators that wrap other futures (timeouts, rate limiters, instrumented wrappers), or FFI adapters that bridge a C callback to the `Future` trait. For day-to-day service code, `async fn` plus existing combinators (`tokio::time::timeout`, `futures::future::join_all`) is the right tool.

### Why the Runtime is a Crate, Not `std`

`std::future::Future` defines the *interface*. The *execution* of futures (the executor that calls `poll`, the reactor that registers I/O readiness with the OS, the thread pool that runs work) is not in `std`. This is intentional:

- **Embedded targets** need no-alloc executors (Embassy).
- **WASM** needs a browser-event-loop-based executor.
- **High-performance services** need io_uring integration (glommio, monoio).
- **Application code** typically wants tokio.

The tradeoff: you must choose and depend on a runtime. Mixing runtimes in a single binary is possible but fragile — library crates should generally not start their own runtime.

```rust
// Library crate: stay runtime-agnostic, expose async fn
pub async fn compute(input: &str) -> String {
    // works with any executor
    tokio::time::sleep(std::time::Duration::from_millis(10)).await; // if tokio feature enabled
    input.to_uppercase()
}

// Binary crate: owns the runtime decision
#[tokio::main]
async fn main() {
    let result = compute("hello").await;
    println!("{result}");
}
```

## Choosing a Runtime

### Tokio (Default Pragmatic Choice)

Use tokio for: HTTP servers, gRPC services, CLI tools with async I/O, database connection pools, anything in the broad async Rust ecosystem (axum, sqlx, reqwest, tonic all target tokio).

```toml
[dependencies]
tokio = { version = "1.40", features = ["full"] }
```

Features of note: `rt-multi-thread` (the default multi-threaded scheduler), `rt` (current-thread only), `net`, `time`, `sync`, `fs`, `macros`. Use `"full"` in application crates, cherry-pick in library crates.

**Multi-thread vs current-thread:**

```rust
// Multi-thread (default): spawns one thread per CPU core, work-stealing scheduler
// Best for: services, anything with mixed I/O + CPU, most production use
#[tokio::main]
async fn main() {
    // equivalent to:
    // tokio::runtime::Runtime::new().unwrap().block_on(main_inner())
}

// Current-thread: single OS thread, cooperative multitasking
// Best for: CLIs, tests, embedded, WASM, latency-sensitive single-stream work
#[tokio::main(flavor = "current_thread")]
async fn main() {
    // No thread pool; all tasks interleaved on this OS thread
}

// Manual runtime construction for fine-grained control:
fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)                    // override CPU count
        .thread_name("my-service-worker")
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async_main());
}
```

### smol

Lightweight alternative. No proc macros, smaller binary, good for CLIs and tools where you want async without the full tokio surface area. Uses `async-io` for the reactor. Less ecosystem coverage — many crates require tokio features explicitly.

```toml
[dependencies]
smol = "2"
```

### async-std (Deprecated Track)

async-std mirrors `std` APIs with async counterparts. Its development has significantly slowed and it is not recommended for new projects. Existing code using it works but migration to tokio is the community direction.

### glommio / monoio (io_uring Niche)

For Linux-only, io_uring-based I/O with thread-per-core architecture. Latency and throughput improvements for storage-intensive workloads. Incompatible with tokio ecosystem. Worth evaluating for: high-throughput file servers, database storage engines, anything where syscall count matters.

### Decision Guide

```
Target environment?
├── Embedded / no-std / no-alloc (Cortex-M, ESP32, RISC-V) → Embassy
├── Browser WASM → wasm-bindgen-futures (or gloo for UI glue)
├── Linux-only, io_uring, thread-per-core storage engine → glommio or monoio
├── Lightweight CLI or tool, no tokio-ecosystem deps needed → smol
└── Anything else (services, tools, HTTP, gRPC, databases, ML) → tokio
                      │
                      └── Tokio flavor?
                          ├── CLI / test / WASI / single-connection app    → `flavor = "current_thread"`
                          ├── !Send futures must be spawned                → `current_thread` + `LocalSet`
                          └── HTTP/gRPC service, mixed I/O + CPU (default) → `flavor = "multi_thread"`
                                                                             (default; tune `worker_threads`
                                                                              to CPU count, raise
                                                                              `max_blocking_threads` if
                                                                              spawn_blocking load is high)
```

**Do not pick `async-std` for new work.** Its maintainers have stepped back; community migration is toward tokio (or smol for lightweight cases). Existing code on `async-std` still functions, but new dependencies almost always assume tokio.

## Tokio Essentials

### `#[tokio::main]` and Runtime Setup

```rust
use tokio::time::{sleep, Duration};

// Standard multi-thread runtime (most common)
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("on tokio multi-thread runtime");
    sleep(Duration::from_millis(100)).await;
    Ok(())
}

// Current-thread runtime (CLIs, tests, scripts)
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    println!("on single OS thread");
    Ok(())
}

// Runtime with custom configuration
fn main() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .max_blocking_threads(256)    // for spawn_blocking pool
        .thread_stack_size(3 * 1024 * 1024)
        .enable_io()
        .enable_time()
        .build()?
        .block_on(async_main())
}

async fn async_main() -> anyhow::Result<()> {
    Ok(())
}
```

### `tokio::spawn` — Fire Off Async Tasks

`tokio::spawn` schedules a future on the runtime's thread pool. The spawned task is completely independent — it can outlive the spawning context. This is why the future must be `'static` and `Send` (for multi-thread runtime).

```rust
use tokio::task::JoinHandle;

async fn process_item(id: u64) -> String {
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    format!("processed {id}")
}

#[tokio::main]
async fn main() {
    // Spawn and collect handles
    let handles: Vec<JoinHandle<String>> = (0..5)
        .map(|id| tokio::spawn(process_item(id)))
        .collect();

    // Await all handles
    for handle in handles {
        match handle.await {
            Ok(result) => println!("{result}"),
            Err(e) if e.is_panic() => eprintln!("task panicked: {e}"),
            Err(e) => eprintln!("task cancelled: {e}"),
        }
    }
}
```

**The `'static + Send` requirement explained:**

```rust
// This fails: `data` is not 'static (borrowed from main's stack)
async fn main_broken() {
    let data = String::from("hello");
    // compile_fail: `data` does not live long enough
    // tokio::spawn(async { println!("{data}"); }); // E0597 + not 'static
}

// Fix: move owned data into the future
async fn main_fixed() {
    let data = String::from("hello");
    tokio::spawn(async move {      // `move` transfers ownership into the future
        println!("{data}");
    }).await.unwrap();
}

// Fix for shared data: Arc
async fn main_shared() {
    let data = std::sync::Arc::new(String::from("hello"));
    let data2 = std::sync::Arc::clone(&data);
    tokio::spawn(async move {
        println!("{data2}");       // Arc clone is 'static
    }).await.unwrap();
}
```

### `spawn_blocking` — Isolating Blocking Work

Never call blocking code (synchronous file I/O, CPU-intensive computation, legacy sync libraries, `std::thread::sleep`) from an async context without `spawn_blocking`. Doing so starves the executor.

```rust
use tokio::task;

async fn hash_password(password: String) -> String {
    // bcrypt is CPU-intensive and blocks the thread for hundreds of milliseconds.
    // Offload to the blocking thread pool to avoid starving the async executor.
    task::spawn_blocking(move || {
        // Runs on a dedicated blocking thread pool (max_blocking_threads, default 512)
        bcrypt::hash(&password, bcrypt::DEFAULT_COST).unwrap()
    })
    .await
    .expect("blocking task panicked")
}

async fn read_config(path: &str) -> std::io::Result<String> {
    // Prefer tokio::fs for async I/O. If using std::fs, wrap with spawn_blocking.
    let path = path.to_string();
    task::spawn_blocking(move || std::fs::read_to_string(path))
        .await
        .expect("blocking task panicked")
}

// Better: use tokio's async filesystem API
async fn read_config_async(path: &str) -> std::io::Result<String> {
    tokio::fs::read_to_string(path).await
}
```

### Tracing Integration

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

```rust
use tracing::{info, instrument, warn};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())  // RUST_LOG=info
        .with_target(true)
        .compact()
        .init();

    info!("service starting");
    serve().await;
}

#[instrument(skip(request), fields(method = %request.method()))]
// hyper 1.0 replaced `hyper::Body` with `hyper::body::Incoming` on the server
// side and `http_body_util::*` bodies on the client side. Use `Incoming` here.
async fn handle_request(request: hyper::Request<hyper::body::Incoming>) -> &'static str {
    // span created automatically, fields recorded, duration logged
    info!("handling request");
    "ok"
}

async fn serve() {
    warn!("serve not implemented");
}
```

For async-aware distributed tracing (span propagation across `.await` points), tokio's runtime integrates with `tracing` via the `tracing-futures` crate and the Tokio Console (`tokio-console`) for live task inspection.

## Structured Concurrency

### `JoinSet` — Safe Task Collection

`JoinSet` tracks a dynamic set of tasks and cancels remaining tasks when dropped. This prevents the "spawn and forget" anti-pattern where tasks continue running after their logical owner is gone.

```rust
use tokio::task::JoinSet;

async fn process_urls(urls: Vec<String>) -> Vec<Result<String, reqwest::Error>> {
    let mut set = JoinSet::new();

    for url in urls {
        set.spawn(async move {
            reqwest::get(&url).await?.text().await
        });
    }

    let mut results = Vec::new();
    while let Some(res) = set.join_next().await {
        match res {
            Ok(inner) => results.push(inner),
            Err(e) => eprintln!("task panicked or cancelled: {e}"),
        }
    }
    results
}

// JoinSet with bounded concurrency (semaphore + JoinSet).
// CRITICAL: acquire the permit BEFORE spawning. Acquiring inside the spawned
// task does not bound spawn rate — N tasks would still be created immediately
// and simply contend on the semaphore once running. The pattern below blocks
// the spawning loop until a permit is available, providing real backpressure.
async fn process_bounded(urls: Vec<String>, max_concurrent: usize) -> Vec<String> {
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent));
    let mut set = JoinSet::new();

    for url in urls {
        // Acquire BEFORE spawning — this is the backpressure point.
        let permit = sem.clone().acquire_owned().await.unwrap();
        set.spawn(async move {
            let body = reqwest::get(&url).await.unwrap().text().await.unwrap();
            drop(permit); // explicit drop for clarity; would drop at scope end regardless
            body
        });
    }

    let mut results = Vec::new();
    while let Some(Ok(body)) = set.join_next().await {
        results.push(body);
    }
    results
}
```

### `tokio::select!` — Racing Futures

`select!` polls multiple branches concurrently and resolves as soon as one completes. The others are dropped (cancelled).

```rust
use tokio::sync::oneshot;
use tokio::time::{sleep, Duration};

async fn with_timeout<F, T>(fut: F, timeout: Duration) -> Option<T>
where
    F: std::future::Future<Output = T>,
{
    tokio::select! {
        result = fut => Some(result),
        _ = sleep(timeout) => None,
    }
}

// Practical shutdown integration
async fn serve_with_shutdown(shutdown: oneshot::Receiver<()>) {
    let server = run_server();   // some long-running future

    tokio::select! {
        _ = server => {
            eprintln!("server exited unexpectedly");
        }
        _ = shutdown => {
            println!("shutdown signal received, stopping");
        }
    }
}

// Looping select: drain multiple channels
async fn event_loop(
    mut cmd_rx: tokio::sync::mpsc::Receiver<String>,
    mut sig_rx: tokio::sync::mpsc::Receiver<()>,
) {
    loop {
        tokio::select! {
            // biased; // uncomment to prioritize top branch (no random poll order)
            cmd = cmd_rx.recv() => match cmd {
                Some(c) => println!("command: {c}"),
                None => break, // channel closed
            },
            _ = sig_rx.recv() => {
                println!("signal received");
                break;
            }
        }
    }
}

async fn run_server() {}
```

**`select!` cancellation semantics:** The losing branch's future is dropped synchronously the moment another branch completes; its `Drop` impl runs immediately, releasing any resources it held. There is no "next await point" — cancellation is instantaneous. If the dropped future was mid-way through a non-atomic operation (partially written to a buffer, mid-transaction), that partial state must be accounted for. Design futures to be cancel-safe: prefer atomic operations or checkpoint-based state.

### Graceful Shutdown with `CancellationToken`

`tokio_util::sync::CancellationToken` is the idiomatic way to propagate cancellation through a tree of tasks.

```toml
[dependencies]
tokio-util = { version = "0.7", features = ["rt"] }
```

```rust
use tokio_util::sync::CancellationToken;
use tokio::signal;

#[tokio::main]
async fn main() {
    let token = CancellationToken::new();

    // Spawn workers with child tokens
    let mut set = tokio::task::JoinSet::new();
    for i in 0..4 {
        let child = token.child_token(); // child: cancelled when parent is cancelled
        set.spawn(async move {
            worker(i, child).await;
        });
    }

    // Wait for Ctrl+C or SIGTERM
    tokio::select! {
        _ = signal::ctrl_c() => {
            println!("Ctrl+C received, cancelling tasks...");
            token.cancel();
        }
    }

    // Wait for all workers to acknowledge cancellation
    while let Some(res) = set.join_next().await {
        if let Err(e) = res {
            eprintln!("worker error: {e}");
        }
    }
    println!("clean shutdown complete");
}

async fn worker(id: u64, cancel: CancellationToken) {
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                println!("worker {id} shutting down");
                return;
            }
            _ = do_work() => {
                println!("worker {id} completed a unit of work");
            }
        }
    }
}

async fn do_work() {
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
```

**Cancellation propagation tree:** `token.child_token()` creates a token that is cancelled when either it is cancelled directly or the parent is cancelled. Cancellation flows **top-down only** — cancelling a child does NOT cancel the parent or its siblings. This allows coarse-grained top-level cancellation with fine-grained override.

## Channels

Tokio provides four channel types. Choosing the wrong one creates either data loss or unnecessary complexity.

### `mpsc` — Multi-Producer, Single-Consumer

The workhorse channel. Multiple senders, one receiver. Bounded (`channel(capacity)`) or unbounded (`unbounded_channel()`). **Always prefer bounded** — unbounded channels can grow without limit under backpressure.

```rust
use tokio::sync::mpsc;

async fn pipeline() {
    let (tx, mut rx) = mpsc::channel::<String>(32); // buffer of 32

    // Multiple producers: clone the sender
    let tx2 = tx.clone();
    tokio::spawn(async move {
        for i in 0..5 {
            tx.send(format!("producer1: {i}")).await.unwrap();
        }
        // tx dropped here — if this was the last sender, rx.recv() returns None
    });
    tokio::spawn(async move {
        for i in 0..5 {
            tx2.send(format!("producer2: {i}")).await.unwrap();
        }
    });

    // Single consumer
    while let Some(msg) = rx.recv().await {
        println!("{msg}");
    }
    println!("all senders dropped, channel closed");
}
```

### `oneshot` — Single Message, One-Way

For request/reply patterns: send exactly one value from one producer to one consumer. Typically used to return a result back to the caller of a task.

```rust
use tokio::sync::oneshot;

async fn request_response() {
    let (tx, rx) = oneshot::channel::<String>();

    tokio::spawn(async move {
        // Simulate work, then reply
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let _ = tx.send("the answer".to_string());
        // Sending can fail if the receiver was dropped — often OK to ignore
    });

    match rx.await {
        Ok(reply) => println!("got reply: {reply}"),
        Err(_) => println!("sender dropped before replying"),
    }
}

// Typical actor pattern: embed a oneshot in the command
struct Command {
    payload: String,
    reply: oneshot::Sender<Result<String, String>>,
}
```

### `broadcast` — One-to-Many Fan-Out

Single producer, multiple subscribers. Each receiver gets every message. Lagging receivers skip messages (lossy). Use for: event buses, pub/sub, configuration change notifications.

```rust
use tokio::sync::broadcast;

async fn event_bus() {
    let (tx, _rx) = broadcast::channel::<String>(64); // channel retains up to 64 values total (shared ring buffer, NOT per-subscriber); slow subscribers see `Lagged` if they fall behind

    // Spawn multiple subscribers
    for sub_id in 0..3 {
        let mut rx = tx.subscribe(); // each call creates a new receiver
        tokio::spawn(async move {
            while let Ok(msg) = rx.recv().await {
                println!("subscriber {sub_id}: {msg}");
            }
        });
    }

    // Publisher
    for i in 0..5 {
        let _ = tx.send(format!("event {i}"));
    }

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
}

// Handling lag (receiver fell behind and messages were dropped)
async fn resilient_subscriber(mut rx: broadcast::Receiver<String>) {
    loop {
        match rx.recv().await {
            Ok(msg) => println!("got: {msg}"),
            Err(broadcast::error::RecvError::Lagged(n)) => {
                eprintln!("skipped {n} messages — receiver too slow");
                // Continue receiving; next recv() returns the oldest available
            }
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}
```

### `watch` — Latest-Value Notifications

One producer, many consumers. Consumers always see the *latest* value, not every intermediate value. Use for: configuration that changes, connection state, last-known health status.

```rust
use tokio::sync::watch;

async fn config_watch() {
    let (tx, rx) = watch::channel("initial-config".to_string());

    // Multiple readers
    for reader_id in 0..3 {
        let mut rx = rx.clone();
        tokio::spawn(async move {
            // changed() yields when a new value has been sent since last call
            while rx.changed().await.is_ok() {
                let value = rx.borrow_and_update().clone();
                println!("reader {reader_id} sees new config: {value}");
            }
        });
    }

    // Writer updates configuration
    tx.send("config-v2".to_string()).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    tx.send("config-v3".to_string()).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    // tx dropped → rx.changed() returns Err → readers exit
}
```

### Channel Selection Guide

| Need | Channel |
|------|---------|
| Multi-producer to single consumer (stream of work) | `mpsc` |
| Single reply to a single request (RPC call) | `oneshot` |
| Fan-out every event to all active subscribers | `broadcast` |
| Notify consumers of the latest state (not history) | `watch` |

## Shared State

### `Arc<Mutex<T>>` — The Standard Pattern

```rust
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct AppState {
    counter: u64,
    connections: Vec<String>,
}

type SharedState = Arc<Mutex<AppState>>;

async fn increment(state: SharedState) {
    // OK to use std::sync::Mutex here: the guard never crosses an .await.
    // It is acquired, used, and dropped before this async fn yields anywhere.
    // See "Don't hold std::sync::Mutex across .await" rule below — it doesn't apply.
    let mut guard = state.lock().unwrap(); // blocks; see async vs std below
    guard.counter += 1;
}   // guard dropped here — lock released

async fn serve(state: SharedState) {
    let handles: Vec<_> = (0..4).map(|_| {
        let s = Arc::clone(&state);
        tokio::spawn(async move { increment(s).await; })
    }).collect();
    for h in handles { h.await.unwrap(); }
    println!("counter: {}", state.lock().unwrap().counter);
}
```

### `parking_lot` vs `std`

`parking_lot::Mutex` is often preferred over `std::sync::Mutex` in high-contention code:

```toml
[dependencies]
parking_lot = "0.12"
```

```rust
use parking_lot::Mutex;
use std::sync::Arc;

// No .unwrap() — parking_lot Mutex never poisons
let state = Arc::new(Mutex::new(0u64));
let s2 = Arc::clone(&state);
tokio::spawn(async move {
    *s2.lock() += 1;  // no .unwrap() needed
});
```

Key differences:

| | `std::sync::Mutex` | `parking_lot::Mutex` |
|-|-------------------|---------------------|
| Lock poisoning | Yes (panics propagate) | No |
| Memory overhead | Larger (OS primitive) | Smaller |
| `const` constructor | Yes (`Mutex::new` is `const fn` since 1.63) | Yes |
| `try_lock_for` / timeout | No | Yes |
| Performance | Good | Better under contention |

### `RwLock` — Read-Heavy Workloads

```rust
use std::sync::{Arc, RwLock};

async fn read_heavy_cache(cache: Arc<RwLock<std::collections::HashMap<String, String>>>) {
    // Multiple concurrent readers
    {
        let guard = cache.read().unwrap();
        if let Some(val) = guard.get("key") {
            println!("cache hit: {val}");
            return;
        }
    } // read guard dropped

    // Exclusive write
    let value = fetch_from_db("key").await;
    let mut guard = cache.write().unwrap();
    guard.insert("key".to_string(), value);
}

async fn fetch_from_db(_key: &str) -> String { "value".to_string() }
```

`RwLock` can suffer from writer starvation under heavy read load. If writes are frequent, `Mutex` may have better tail latency. Measure before choosing.

### The Critical Rule: Do NOT Hold `std::sync::Mutex` Across `.await`

This is the most important async-specific ownership rule. A `std::sync::Mutex` guard (`MutexGuard<T>`) is `!Send`. Holding it across an `.await` means the future captures a `!Send` type — the future itself becomes `!Send`, which prevents `tokio::spawn` from accepting it.

Note that `tokio::spawn` requires `F: Future + Send + 'static` on *every* runtime flavor — including `current_thread`. The flavor controls thread count, not the `Send` bound. `!Send` futures must use `tokio::task::spawn_local` (inside a `LocalSet`) or be driven directly by `Runtime::block_on`. Even on `current_thread`, code that uses `spawn_local`/`block_on` can still deadlock while holding a `std::sync::Mutex` guard across `.await`: the task yields (suspends), another task on the same thread tries to lock the same mutex — and blocks the thread, preventing the first task from resuming.

```rust
use std::sync::{Arc, Mutex};

async fn wrong_usage(state: Arc<Mutex<Vec<String>>>) {
    let mut guard = state.lock().unwrap();
    guard.push("start".to_string());

    // WRONG: guard held across .await — !Send, potential deadlock
    some_async_operation().await;

    guard.push("end".to_string());
}   // guard dropped here — too late

async fn correct_usage_drop_first(state: Arc<Mutex<Vec<String>>>) {
    {
        let mut guard = state.lock().unwrap();
        guard.push("start".to_string());
    } // guard dropped before .await

    some_async_operation().await;

    {
        let mut guard = state.lock().unwrap();
        guard.push("end".to_string());
    }
}

async fn correct_usage_tokio_mutex(state: Arc<tokio::sync::Mutex<Vec<String>>>) {
    // tokio::sync::Mutex is async-aware: .lock().await yields instead of blocking.
    // CAVEAT: `lock().await` is NOT cancel-safe — cancelling a pending lock
    // loses your place in the fair FIFO queue. Do not put `lock().await` in a
    // `select!` branch that can be cancelled without care.
    let mut guard = state.lock().await;
    guard.push("start".to_string());
    some_async_operation().await; // OK: guard is Send in a tokio::sync::Mutex
    guard.push("end".to_string());
}

async fn some_async_operation() {}
```

### `tokio::sync` Locks — When to Use Async-Aware Variants

| Type | Use when |
|------|---------|
| `std::sync::Mutex` | Lock held briefly, never across `.await`. Higher throughput. |
| `tokio::sync::Mutex` | Must hold lock across `.await` points. |
| `std::sync::RwLock` | Read-heavy, never across `.await`. |
| `tokio::sync::RwLock` | Read-heavy, may cross `.await`. |
| `tokio::sync::Semaphore` | Bounded concurrency (rate limiting, connection pools). |

Prefer `std::sync::Mutex` when the lock is not held across await points — it has lower overhead than the async variant. Only reach for `tokio::sync::Mutex` when you genuinely need to await while holding the lock.

**Cancel-safety note:** `tokio::sync::Mutex::lock().await`, `RwLock::{read,write}().await`, and `Semaphore::acquire().await` are **not** cancel-safe. Cancelling a pending acquisition drops the waiter from the fair FIFO queue — no state is corrupted, but the wait is lost. Wrap these in `select!` branches only when the branch is genuinely "give up on this work" (shutdown, deadline). To wait for a lock under a timeout without losing cancel safety, use `tokio::time::timeout(dur, lock.lock())` rather than racing `sleep` against `lock()` in `select!`.

### Deadlock Avoidance

Tokio's async executor does not detect deadlocks. Common causes:

1. Lock ordering: always acquire locks in a consistent global order.
2. `std::sync::Mutex` held across `.await` (covered above).
3. Sending on a bounded channel while holding a lock that the receiver is waiting on before sending.

```rust
// Deadlock example: task A holds lock, waits on channel; task B waits on lock to send
// Fix: release the lock before waiting on the channel
async fn deadlock_prone(
    lock: Arc<Mutex<i32>>,
    tx: tokio::sync::mpsc::Sender<i32>,
) {
    let guard = lock.lock().unwrap();
    let val = *guard;
    drop(guard);          // release before sending to avoid deadlock
    tx.send(val).await.unwrap();
}
```

## Async Trait Reality Check

### Native Async fn in Traits (Stable since 1.75)

```rust
trait DataSource {
    async fn fetch(&self, id: u64) -> Option<String>;
    async fn store(&mut self, id: u64, value: String) -> bool;
}

struct MemorySource {
    data: std::collections::HashMap<u64, String>,
}

impl DataSource for MemorySource {
    async fn fetch(&self, id: u64) -> Option<String> {
        self.data.get(&id).cloned()
    }

    async fn store(&mut self, id: u64, value: String) -> bool {
        self.data.insert(id, value);
        true
    }
}
```

This is stable for concrete types and generic bounds (`T: DataSource`). Two caveats as of the 1.87 baseline:

1. **Not yet dyn-compatible (object-safe).** A trait containing `async fn` cannot be used as `dyn DataSource`. The compiler tells you to use `#[trait_variant::make]` or the `async-trait` macro if you need trait objects. Native async fn in `dyn` trait objects is still unstable.
2. **`Send` inference is implicit.** Native async fn in traits produces futures that may or may not be `Send` — the compiler infers this from the future body. If your implementations will be used with `tokio::spawn` on a multi-thread runtime, you often need `Send` bounds, which the `async fn` sugar cannot express directly. See workarounds below.

### `?Send` Bounds and Why They Matter

```rust
// This function requires the future returned by DataSource::fetch to be Send
// so it can be used with tokio::spawn on a multi-thread runtime.
async fn spawn_fetch<S>(source: S, id: u64) -> Option<String>
where
    S: DataSource + Send + 'static,
    // The problem: with native `async fn` sugar we can't directly bound the
    // returned future as Send. Workarounds: (a) rewrite the trait method using
    // explicit RPITIT — `fn fetch(&self, id: u64) -> impl Future<Output=...> + Send;`
    // — which lets callers require Send futures; or (b) use the
    // `#[trait_variant::make]` attribute to auto-generate a Send-bound variant
    // of the trait. On nightly, return-type notation (`S: DataSource<fetch(..): Send>`)
    // expresses this bound directly.
{
    tokio::spawn(async move { source.fetch(id).await })
        .await
        .unwrap()
}
```

When you need `Send` bounds on the returned futures (always the case with `tokio::spawn` on a multi-thread runtime), there are two approaches:

### `BoxFuture` — Explicit Erased Future

```rust
use std::future::Future;
use std::pin::Pin;

// BoxFuture type alias (also in futures::future::BoxFuture)
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

trait AsyncHandler: Send + Sync {
    fn handle<'a>(&'a self, input: &'a str) -> BoxFuture<'a, String>;
}

struct EchoHandler;

impl AsyncHandler for EchoHandler {
    fn handle<'a>(&'a self, input: &'a str) -> BoxFuture<'a, String> {
        Box::pin(async move { input.to_uppercase() })
    }
}

// Now usable as dyn AsyncHandler — object-safe because BoxFuture is a concrete type
fn make_handler() -> Box<dyn AsyncHandler> {
    Box::new(EchoHandler)
}
```

### `async-trait` Macro — When It's Still Worth It

The `async-trait` crate transforms async fn in traits into `BoxFuture` under the hood. Use it when:

- You need `dyn Trait` with async methods (object safety).
- You want the ergonomic `async fn` syntax without writing `BoxFuture` manually.
- Dealing with complex lifetime interactions in trait methods.

```toml
[dependencies]
async-trait = "0.1"
```

```rust
use async_trait::async_trait;

#[async_trait]
trait Repository: Send + Sync {
    async fn find_by_id(&self, id: u64) -> Option<String>;
    async fn save(&self, id: u64, value: String) -> Result<(), String>;
}

struct PostgresRepository;

#[async_trait]
impl Repository for PostgresRepository {
    async fn find_by_id(&self, id: u64) -> Option<String> {
        // actual DB call
        Some(format!("row_{id}"))
    }

    async fn save(&self, id: u64, value: String) -> Result<(), String> {
        println!("saving {id}: {value}");
        Ok(())
    }
}

// Usable as dyn Repository
async fn use_repo(repo: &dyn Repository) {
    let val = repo.find_by_id(1).await;
    println!("{val:?}");
}
```

**`async-trait` tradeoffs:**
- Allocates a `Box` for each async method call (heap allocation per call).
- Adds `Send` bound by default; use `#[async_trait(?Send)]` for single-thread code.
- Proc-macro compile overhead.
- Still the most ergonomic option for object-safe async traits in non-hot paths.

**When NOT to use `async-trait`:**
- Hot paths where per-call heap allocation is unacceptable: use generics with `BoxFuture` or static dispatch.
- Traits where all implementations are known at compile time: native async fn in traits suffices.

## Web Frameworks Sidebar

### axum

axum is the tokio-ecosystem web framework built by the tokio team. It uses a tower-based middleware model and extractors.

```rust
use axum::{routing::get, Router, Json};
use serde::Serialize;

#[derive(Serialize)]
struct Health { status: &'static str }

async fn health() -> Json<Health> {
    Json(Health { status: "ok" })
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/health", get(health));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

axum's async model: handler functions return `impl IntoResponse`. Middleware is composed via `tower::ServiceBuilder`. Graceful shutdown integrates with tokio's `signal` module. See the axum docs for extractors, state sharing (`State<T>`), and error handling.

### actix-web

actix-web is a high-performance framework with its own runtime wrapper (uses tokio underneath since v4). Actor model available via the `actix` crate but not required.

```rust
use actix_web::{get, web, App, HttpServer, Responder};

#[get("/health")]
async fn health() -> impl Responder {
    "ok"
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(health))
        .bind("0.0.0.0:3000")?
        .run()
        .await
}
```

actix-web uses its own `#[actix_web::main]` macro which sets up tokio internally. It runs a **per-worker, non-`Send` runtime** (each worker thread owns a `current_thread` runtime), so the `Handler` trait requires only `Clone + 'static` — handlers and per-worker state do **not** need `Send`/`Sync` bounds. Cross-worker shared state still does: use `web::Data<Arc<T>>` (optionally with `Mutex`/`RwLock`) for state that must outlive or be visible across workers.

**axum vs actix-web:** Both are production-grade. axum has tighter integration with the tokio ecosystem and tower middleware. actix-web has a longer track record and slightly higher throughput benchmarks at extreme concurrency. Either is a defensible choice; axum is more common for new projects in 2025.

## Performance Pitfalls

### Holding Locks Across `.await` (Covered Above — Never Do This)

Repeating for emphasis: `std::sync::Mutex` guard held across `.await` makes the future `!Send` and risks deadlock on current-thread runtimes. Restructure to drop before yielding, or switch to `tokio::sync::Mutex`.

### Blocking the Executor Thread

The tokio multi-thread runtime uses a work-stealing scheduler across N OS threads (default: CPU count). If any task blocks an OS thread for more than ~10-100μs (database call with blocking driver, `std::thread::sleep`, blocking file I/O, CPU-intensive work), that thread cannot run other tasks. Under load, all threads can be blocked, starving the entire service.

```rust
// WRONG: blocking the executor thread
async fn handle() {
    std::thread::sleep(std::time::Duration::from_secs(1)); // blocks OS thread
    heavy_cpu_work();                                       // blocks OS thread
    std::fs::read_to_string("/tmp/config").unwrap();        // blocking I/O
}

// CORRECT: offload to blocking thread pool
async fn handle_correctly() {
    tokio::time::sleep(std::time::Duration::from_secs(1)).await; // async sleep
    tokio::task::spawn_blocking(heavy_cpu_work).await.unwrap();   // blocking pool
    tokio::fs::read_to_string("/tmp/config").await.unwrap();      // async I/O
}

fn heavy_cpu_work() { /* ... */ }
```

Detect with: tokio-console shows tasks that are not yielding. `tokio::time::timeout` can surface stuck futures.

### Large Futures on the Stack

Each async state machine captures all locals across yield points. A future that holds large arrays or structs grows proportionally. This can cause stack overflows in deeply recursive async code.

```rust
// WRONG: recursive async function — unbounded stack growth
async fn recursive_bad(n: u64) -> u64 {
    if n == 0 { return 0; }
    // Each await frame stacks the locals of this function
    1 + recursive_bad(n - 1).await  // compile error E0733: recursion in an `async fn` requires boxing
}

// CORRECT: box the recursive future to move it to the heap
fn recursive_ok(n: u64) -> std::pin::Pin<Box<dyn std::future::Future<Output = u64> + Send>> {
    Box::pin(async move {
        if n == 0 { return 0; }
        1 + recursive_ok(n - 1).await
    })
}

// ALSO CORRECT: use an iterative approach
async fn iterative(n: u64) -> u64 {
    let mut sum = 0u64;
    for i in 0..n {
        tokio::task::yield_now().await; // explicit yield point
        sum += i;
    }
    sum
}
```

`tokio::task::yield_now()` is also useful in CPU-bound loops to surrender the executor thread periodically, preventing task monopolization.

### `tokio::spawn` in a Hot Loop Without Backpressure

Spawning a task per unit of work in an unbounded loop can create millions of live tasks, exhausting memory. Backpressure is the mechanism that slows producers when consumers are behind.

```rust
// WRONG: unbounded task spawning
async fn process_all_wrong(items: Vec<String>) {
    for item in items {
        tokio::spawn(async move { process_item(&item).await; });
        // 1M items → 1M live tasks → OOM or extreme latency
    }
}

// CORRECT: bound concurrency with a semaphore
async fn process_all_bounded(items: Vec<String>) {
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(64)); // max 64 concurrent
    let mut set = tokio::task::JoinSet::new();

    for item in items {
        let permit = std::sync::Arc::clone(&sem).acquire_owned().await.unwrap();
        set.spawn(async move {
            let _permit = permit; // held until task completes
            process_item(&item).await;
        });
    }
    while set.join_next().await.is_some() {}
}

// CORRECT: channel-based backpressure (bounded channel blocks sender when full)
async fn process_all_channel(items: Vec<String>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);

    let consumer = tokio::spawn(async move {
        while let Some(item) = rx.recv().await {
            process_item(&item).await;
        }
    });

    for item in items {
        tx.send(item).await.unwrap(); // blocks when channel is full — backpressure
    }
    drop(tx);
    consumer.await.unwrap();
}

async fn process_item(_item: &str) {}
```

### Unbounded Channels

`tokio::sync::mpsc::unbounded_channel()` accepts items without ever blocking the sender. Under any burst or slow consumer, the queue grows without limit. This is almost always wrong for production code.

```rust
// WRONG: unbounded channel — silently buffers unbounded data
let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();

// CORRECT: bounded channel with explicit capacity
let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);
// If the consumer is slow, tx.send().await will yield until space is available
```

The only appropriate uses for unbounded channels: test code, controlled environments where the producer rate is provably bounded, or intentional buffering with explicit memory accounting.

## Anti-Patterns

### Anti-Pattern 1: Holding `std::sync::Mutex` Across `.await`

**Why wrong:** `MutexGuard<T>` is `!Send`. Capturing it in an async block that crosses an await point makes the future `!Send`, preventing use with `tokio::spawn` on a multi-thread runtime. Even on a single-thread runtime, it risks deadlock: the task yields while holding the lock; another task attempts to acquire the same lock; the thread is blocked; the first task can never resume.

```rust
use std::sync::{Arc, Mutex};

// WRONG: guard crosses .await
async fn wrong(state: Arc<Mutex<Vec<i32>>>) {
    let mut guard = state.lock().unwrap();
    guard.push(1);
    network_call().await;   // guard is still live → future is !Send → deadlock risk
    guard.push(2);
}

// FIX 1: drop the guard before .await
async fn fixed_drop(state: Arc<Mutex<Vec<i32>>>) {
    {
        let mut guard = state.lock().unwrap();
        guard.push(1);
    } // guard dropped here
    network_call().await;
    {
        let mut guard = state.lock().unwrap();
        guard.push(2);
    }
}

// FIX 2: use tokio::sync::Mutex (async-aware, Send)
async fn fixed_async(state: Arc<tokio::sync::Mutex<Vec<i32>>>) {
    let mut guard = state.lock().await;
    guard.push(1);
    network_call().await; // OK: tokio::sync::MutexGuard is Send
    guard.push(2);
}

async fn network_call() {}
```

### Anti-Pattern 2: Calling `block_on` Inside an Async Function

**Why wrong:** `Runtime::block_on` and `Handle::block_on` expect to run on a non-async thread. Calling them from within an already-running async context **panics** at runtime with "Cannot start a runtime from within a runtime" (tokio). This is a panic, not undefined behaviour — but the caller is still broken.

```rust
// WRONG: block_on inside async context → panic at runtime
async fn wrong_blocking() {
    let handle = tokio::runtime::Handle::current();
    // This panics: already inside a tokio runtime
    handle.block_on(async { println!("inner"); });
}

// FIX: just .await — that's the correct bridging mechanism
async fn correct() {
    println!("inner");  // or:
    inner_async().await;
}

async fn inner_async() { println!("inner"); }
```

If you need to call async code from a sync function that is itself called from async context, restructure to make the calling function async, or use `tokio::task::spawn_blocking` to run the sync function on a blocking thread that can then create its own runtime if truly needed.

### Anti-Pattern 3: `tokio::spawn` in a Hot Loop Without Backpressure

**Why wrong:** Spawning a new task for every incoming item with no bound on the number of live tasks consumes memory proportional to queue depth. Under load spikes, this can exhaust heap memory and cause cascading failure. Backpressure is how async systems communicate that consumers are saturated.

```rust
// WRONG: one spawn per item, no ceiling
async fn ingest_wrong(items: impl Iterator<Item = Vec<u8>>) {
    for item in items {
        tokio::spawn(async move { store(item).await; });
        // Items arrive faster than store() completes → unbounded task accumulation
    }
}

// FIX: semaphore-bounded concurrency
async fn ingest_bounded(items: impl Iterator<Item = Vec<u8>>) {
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(128));
    let mut set = tokio::task::JoinSet::new();
    for item in items {
        let permit = std::sync::Arc::clone(&sem).acquire_owned().await.unwrap();
        set.spawn(async move {
            let _permit = permit;
            store(item).await;
        });
    }
    while set.join_next().await.is_some() {}
}

async fn store(_: Vec<u8>) {}
```

### Anti-Pattern 4: `std::thread::sleep` in Async Code

**Why wrong:** `std::thread::sleep` suspends the OS thread for the specified duration. The tokio executor cannot reclaim the thread for other tasks. Under load, all executor threads sleeping simultaneously means no async progress.

```rust
use std::time::Duration;

// WRONG: blocks the OS thread, freezes the executor on this thread
async fn wrong_sleep() {
    std::thread::sleep(Duration::from_secs(1)); // blocks the thread
    do_something().await;
}

// CORRECT: async sleep yields the executor thread to run other tasks
async fn correct_sleep() {
    tokio::time::sleep(Duration::from_secs(1)).await; // non-blocking
    do_something().await;
}

// Also common mistake: std::time::Instant for timing loops
async fn wrong_wait_for(deadline: std::time::Instant) {
    while std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1)); // WRONG
    }
}

async fn correct_wait_for(deadline: tokio::time::Instant) {
    tokio::time::sleep_until(deadline).await; // CORRECT
}

async fn do_something() {}
```

### Anti-Pattern 5: Unbounded Channels With No Backpressure Story

**Why wrong:** `unbounded_channel` decouples producer and consumer but removes the mechanism that slows producers when consumers fall behind. Under any sustained load imbalance, the queue grows until the process runs out of memory.

```rust
// WRONG: producer can outrun consumer indefinitely
async fn wrong_pipeline(events: Vec<String>) {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    tokio::spawn(async move {
        while let Some(e) = rx.recv().await {
            slow_process(e).await; // takes 10ms per event
        }
    });

    for event in events {
        tx.send(event).unwrap(); // never blocks → queue grows to 1M items
    }
}

// FIX: use bounded channel; sender yields when consumer is behind
async fn correct_pipeline(events: Vec<String>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64); // bounded!

    tokio::spawn(async move {
        while let Some(e) = rx.recv().await {
            slow_process(e).await;
        }
    });

    for event in events {
        tx.send(event).await.unwrap(); // yields when buffer is full — backpressure
    }
}

async fn slow_process(_: String) {
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
}
```

### Anti-Pattern 6: Calling Blocking Syscalls Directly

**Why wrong:** Synchronous file I/O, DNS resolution via `gethostbyname`, blocking database drivers, and any blocking syscall suspend the OS thread. On tokio's multi-thread runtime with N workers, having all workers in blocking syscalls stops the entire event loop from making progress.

```rust
// WRONG: blocking file read on executor thread
async fn wrong_read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap() // blocks OS thread
}

// FIX 1: use tokio's async filesystem API
async fn correct_async_read(path: &str) -> std::io::Result<String> {
    tokio::fs::read_to_string(path).await
}

// FIX 2: wrap any blocking operation with spawn_blocking
async fn correct_blocking_read(path: String) -> std::io::Result<String> {
    tokio::task::spawn_blocking(move || std::fs::read_to_string(path))
        .await
        .expect("blocking task panicked")
}

// WRONG: blocking DNS resolution (common in connection setup)
async fn wrong_connect(host: &str) {
    // std::net::TcpStream::connect blocks on DNS + TCP handshake
    let _stream = std::net::TcpStream::connect((host, 80)).unwrap();
}

// CORRECT: use tokio's async networking
async fn correct_connect(host: &str) -> tokio::io::Result<tokio::net::TcpStream> {
    tokio::net::TcpStream::connect((host, 80)).await
}
```

## Checklist

Before shipping async Rust code:

- [ ] No `std::sync::MutexGuard` held across `.await` points — verified by `Send` bound compilation, or by code review.
- [ ] No `std::thread::sleep` in async functions — only `tokio::time::sleep(...).await`.
- [ ] Blocking I/O and CPU-intensive work wrapped in `spawn_blocking`.
- [ ] All bounded channels: `mpsc::channel(N)` not `unbounded_channel()` except where explicitly justified.
- [ ] Hot loops with `tokio::spawn` have a concurrency ceiling (semaphore or `JoinSet` + bounded backpressure).
- [ ] `block_on` not called from within an async context.
- [ ] Graceful shutdown implemented: `CancellationToken` or signal handler cancels `JoinSet`, tasks check for cancellation.
- [ ] Channel type matches semantics: `mpsc` for work queues, `oneshot` for single replies, `broadcast` for fan-out, `watch` for latest-value.
- [ ] `tokio::sync::Mutex` used only when the lock must be held across `.await`; `std::sync::Mutex` used otherwise (lower overhead).
- [ ] Async fn in traits with object-safety requirements use `BoxFuture` directly or `async-trait`; `Send` bounds are explicit.
- [ ] Recursive async functions use `Box::pin` to heap-allocate the recursive future.
- [ ] `tokio::task::yield_now().await` inserted in CPU-bound loops to prevent task monopolization.
- [ ] Runtime flavor matches deployment: `multi_thread` for services, `current_thread` for CLIs or tests.
- [ ] Tasks spawned with `tokio::spawn` handle the `JoinError` — panics in tasks do not silently disappear.
- [ ] `tracing` instrumentation added to long-running tasks and request handlers for observability.

## Related Skills

- [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — `Send`/`Sync` auto-traits, `Pin`/`Unpin` and why async state machines are `!Unpin`, `Arc<Mutex<T>>` vs `Rc<RefCell<T>>` decisions, `'static` bounds on spawned tasks.
- [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — Object safety constraints on async traits, `BoxFuture` and `dyn Future`, `async-trait` macro and its interaction with vtable dispatch, `Send` bounds on generic async functions.
- [error-handling-patterns.md](error-handling-patterns.md) — Error propagation across `tokio::spawn` boundaries (`JoinError` wrapping), `anyhow` in async handlers, `thiserror` for service error types.
- [project-structure-and-tooling.md](project-structure-and-tooling.md) — Tokio feature flag selection in `Cargo.toml`, workspace layout for services, `tokio-console` setup for runtime inspection.
- [testing-and-quality.md](testing-and-quality.md) — `#[tokio::test]` for async tests, testing shutdown sequences, mocking time with `tokio::time::pause`, integration tests for async handlers.
- [systematic-delinting.md](systematic-delinting.md) — Clippy lints relevant to async: `clippy::async_yields_async`, `clippy::let_underscore_future`, blocking-in-async detection.
- [performance-and-profiling.md](performance-and-profiling.md) — Tokio console for live task inspection, async flamegraphs, identifying blocking tasks, measuring channel throughput, `spawn_blocking` pool tuning.
- [modern-rust-and-editions.md](modern-rust-and-editions.md) — Native async fn in traits (stable 1.75), RPITIT, async closures (stabilizing), edition-specific async features.
- [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — `Pin::new_unchecked` for custom async primitives, implementing `Future` manually, `Waker` and `RawWaker` internals, FFI with async callbacks.
- [ai-ml-and-interop.md](ai-ml-and-interop.md) — Async Python interop via PyO3 with tokio, streaming inference over async channels, async HTTP clients for model API calls.
