---
name: async-patterns-and-concurrency
description: async/await mastery, asyncio patterns, TaskGroup (3.11+), structured concurrency, event loop management, common pitfalls, concurrent.futures
---

# Async Patterns and Concurrency

## Overview

**Core Principle:** Async code is about I/O concurrency, not CPU parallelism. Use async when waiting for network, files, or databases. Don't use async to speed up CPU-bound work.

Python's async/await (asyncio) enables single-threaded concurrency through cooperative multitasking. Structured concurrency (TaskGroup in 3.11+) makes async code safer and easier to reason about. The most common mistake: blocking the event loop with synchronous operations.

## When to Use

**Use this skill when:**
- "asyncio not working"
- "async/await errors"
- "Event loop issues"
- "Coroutine never awaited"
- "How to use TaskGroup?"
- "When to use async?"
- "Async code is slow"
- "Blocking the event loop"

**Don't use when:**
- CPU-bound work (use multiprocessing or threads)
- Setting up project (use project-structure-and-tooling)
- Profiling needed (use debugging-and-profiling first)

**Symptoms triggering this skill:**
- RuntimeWarning: coroutine was never awaited
- Event loop errors
- Async functions not running concurrently
- Need to parallelize I/O operations

---

## Async Fundamentals

### When to Use Async vs Sync

```python
# ❌ WRONG: Using async for CPU-bound work
async def calculate_fibonacci(n: int) -> int:
    if n < 2:
        return n
    return await calculate_fibonacci(n-1) + await calculate_fibonacci(n-2)
# Problem: No I/O, just CPU work. Async adds overhead without benefit.

# ✅ CORRECT: Use regular function for CPU work
def calculate_fibonacci(n: int) -> int:
    if n < 2:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# ✅ CORRECT: Use async for I/O-bound work
async def fetch_user(user_id: int) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/users/{user_id}") as resp:
            return await resp.json()
# Async shines: waiting for network response, can do other work

# ✅ CORRECT: Use async when orchestrating multiple I/O operations
async def fetch_all_users(user_ids: list[int]) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_user(session, uid) for uid in user_ids]
        return await asyncio.gather(*tasks)
# Multiple network calls run concurrently
```

**Why this matters**: Async adds complexity. Only use when you benefit from I/O concurrency. For CPU work, use threads or multiprocessing.

### Basic async/await Syntax

```python
# ❌ WRONG: Forgetting await
async def get_data():
    return fetch_from_api()  # Returns coroutine, doesn't execute!

result = get_data()  # RuntimeWarning: coroutine never awaited
print(result)  # Prints <coroutine object>, not data

# ✅ CORRECT: Always await async functions
async def get_data():
    return await fetch_from_api()

# ✅ CORRECT: Running from sync code
import asyncio

def main():
    result = asyncio.run(get_data())
    print(result)

# ✅ CORRECT: Running from async code
async def main():
    result = await get_data()
    print(result)

asyncio.run(main())
```

**Why this matters**: Async functions return coroutines. Must `await` them to execute. `asyncio.run()` bridges sync and async worlds.

### Running the Event Loop

```python
# ❌ WRONG: Running event loop multiple times
import asyncio

asyncio.run(task1())
asyncio.run(task2())  # Creates new event loop, inefficient

# ✅ CORRECT: Single event loop for all async work
async def main():
    await task1()
    await task2()

asyncio.run(main())

# ❌ WRONG: Mixing asyncio.run and manual loop management
loop = asyncio.get_event_loop()
loop.run_until_complete(task1())
asyncio.run(task2())  # Error: loop already running

# ✅ CORRECT: Use asyncio.run() (Python 3.7+)
asyncio.run(main())

# ✅ CORRECT: For advanced cases, manual loop management
async def main():
    await task1()
    await task2()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    loop.run_until_complete(main())
finally:
    loop.close()
```

**Why this matters**: `asyncio.run()` handles loop creation and cleanup automatically. Prefer it unless you need fine-grained control.

---

## Structured Concurrency with TaskGroup (Python 3.11+)

### TaskGroup Basics

```python
# ❌ WRONG: Creating tasks without proper cleanup (old style)
async def fetch_all_old(urls: list[str]) -> list[str]:
    tasks = []
    for url in urls:
        task = asyncio.create_task(fetch(url))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results
# Problem: If one task fails, others continue. No automatic cleanup.

# ✅ CORRECT: TaskGroup (Python 3.11+)
async def fetch_all(urls: list[str]) -> list[str]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch(url)) for url in urls]

    # When exiting context, all tasks guaranteed complete or cancelled
    return [task.result() for task in tasks]

# Why this matters: TaskGroup ensures:
# 1. All tasks complete before proceeding
# 2. If any task fails, all others cancelled
# 3. Automatic cleanup, no leaked tasks
```

### Handling Errors with TaskGroup

```python
# ❌ WRONG: Silent failures with gather
async def process_all_gather(items: list[str]) -> list[str]:
    tasks = [asyncio.create_task(process(item)) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]
# Problem: Errors silently ignored, hard to debug

# ✅ CORRECT: TaskGroup raises ExceptionGroup
async def process_all(items: list[str]) -> list[str]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(process(item)) for item in items]
    return [task.result() for task in tasks]

# Usage with error handling
try:
    results = await process_all(items)
except* ValueError as eg:
    # Handle all ValueErrors
    for exc in eg.exceptions:
        log.error(f"Validation error: {exc}")
except* ConnectionError as eg:
    # Handle all ConnectionErrors
    for exc in eg.exceptions:
        log.error(f"Network error: {exc}")

# ✅ CORRECT: Selective error handling with gather
async def process_with_fallback(items: list[str]) -> list[str]:
    tasks = [asyncio.create_task(process(item)) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    for item, result in zip(items, results):
        if isinstance(result, Exception):
            log.warning(f"Failed to process {item}: {result}")
            processed.append(None)  # Or default value
        else:
            processed.append(result)
    return processed
```

**Why this matters**: TaskGroup provides structured concurrency with automatic cleanup. Use `gather` when you need partial results despite failures.

### Timeout Handling

```python
# ❌ WRONG: No timeout on I/O operations
async def fetch_data(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()
# Problem: Can hang forever if server doesn't respond

# ✅ CORRECT: Timeout with asyncio.timeout (Python 3.11+)
async def fetch_data(url: str) -> str:
    async with asyncio.timeout(10.0):  # 10 second timeout
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.text()
# Raises TimeoutError after 10 seconds

# ✅ CORRECT: Timeout on TaskGroup
async def fetch_all_with_timeout(urls: list[str]) -> list[str]:
    async with asyncio.timeout(30.0):  # Total timeout
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(fetch_data(url)) for url in urls]
        return [task.result() for task in tasks]

# ✅ CORRECT: Individual timeouts (Python <3.11)
async def fetch_with_timeout_old(url: str) -> str:
    try:
        return await asyncio.wait_for(fetch_data(url), timeout=10.0)
    except asyncio.TimeoutError:
        log.error(f"Timeout fetching {url}")
        raise
```

**Why this matters**: Always timeout I/O operations. Network calls can hang indefinitely. `asyncio.timeout()` (3.11+) is cleaner than `wait_for()`.

---

## Async Context Managers

### Basic Async Context Manager

```python
# ❌ WRONG: Using sync context manager in async code
class DatabaseConnection:
    def __enter__(self):
        self.conn = connect_to_db()  # Blocking I/O!
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()  # Blocking I/O!

async def query():
    with DatabaseConnection() as conn:  # Blocks event loop
        return await conn.query("SELECT * FROM users")

# ✅ CORRECT: Async context manager
class AsyncDatabaseConnection:
    async def __aenter__(self):
        self.conn = await async_connect_to_db()
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()

async def query():
    async with AsyncDatabaseConnection() as conn:
        return await conn.query("SELECT * FROM users")
```

### Using contextlib for Async Context Managers

```python
from contextlib import asynccontextmanager

# ✅ CORRECT: Simple async context manager with decorator
@asynccontextmanager
async def database_connection(host: str):
    conn = await connect_to_database(host)
    try:
        yield conn
    finally:
        await conn.close()

# Usage
async def fetch_users():
    async with database_connection("localhost") as conn:
        return await conn.query("SELECT * FROM users")

# ✅ CORRECT: Resource pool management
@asynccontextmanager
async def http_session():
    session = aiohttp.ClientSession()
    try:
        yield session
    finally:
        await session.close()

async def fetch_multiple(urls: list[str]):
    async with http_session() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)
```

**Why this matters**: Async context managers ensure resources cleaned up properly. Use `@asynccontextmanager` for simple cases, `__aenter__/__aexit__` for complex ones.

---

## Async Iterators and Generators

### Async Iterators

```python
# ❌ WRONG: Sync iterator doing async work
class DataFetcher:
    def __init__(self, ids: list[int]):
        self.ids = ids
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.ids):
            raise StopIteration
        data = asyncio.run(fetch_data(self.ids[self.index]))  # Don't do this!
        self.index += 1
        return data

# ✅ CORRECT: Async iterator
class AsyncDataFetcher:
    def __init__(self, ids: list[int]):
        self.ids = ids
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.ids):
            raise StopAsyncIteration
        data = await fetch_data(self.ids[self.index])
        self.index += 1
        return data

# Usage
async def process_all():
    async for data in AsyncDataFetcher([1, 2, 3, 4]):
        print(data)
```

### Async Generators

```python
# ✅ CORRECT: Async generator (simpler than iterator)
async def fetch_users_paginated(page_size: int = 100):
    page = 0
    while True:
        users = await fetch_page(page, page_size)
        if not users:
            break
        for user in users:
            yield user
        page += 1

# Usage
async def process_all_users():
    async for user in fetch_users_paginated():
        await process_user(user)

# ✅ CORRECT: Async generator with cleanup
async def stream_file_lines(path: str):
    async with aiofiles.open(path) as f:
        async for line in f:
            yield line.strip()

# Usage with async comprehension
async def load_data(path: str) -> list[str]:
    return [line async for line in stream_file_lines(path)]
```

**Why this matters**: Async iterators/generators enable streaming I/O-bound data without loading everything into memory. Essential for large datasets.

---

## Common Async Pitfalls

### Blocking the Event Loop

```python
# ❌ WRONG: Blocking operation in async function
import time
import requests

async def fetch_data(url: str) -> str:
    # Blocks entire event loop for 2 seconds!
    time.sleep(2)

    # Also blocks event loop (requests is synchronous)
    response = requests.get(url)
    return response.text

# ✅ CORRECT: Use async sleep and async HTTP
import asyncio
import aiohttp

async def fetch_data(url: str) -> str:
    await asyncio.sleep(2)  # Non-blocking sleep

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

# ✅ CORRECT: If must use blocking code, run in executor
import asyncio
import requests

async def fetch_data_sync(url: str) -> str:
    loop = asyncio.get_running_loop()

    # Run blocking code in thread pool
    response = await loop.run_in_executor(
        None,  # Use default executor
        requests.get,
        url
    )
    return response.text

# ✅ CORRECT: CPU-bound work in process pool
async def heavy_computation(data: bytes) -> bytes:
    loop = asyncio.get_running_loop()

    # Run in process pool for CPU work
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, process_data, data)
    return result
```

**Why this matters**: Blocking the event loop stops ALL async code. Use async libraries (aiohttp not requests), async sleep, or run_in_executor for blocking code.

### Forgetting to Await

```python
# ❌ WRONG: Not awaiting async functions
async def main():
    fetch_data()  # Returns coroutine, doesn't run!
    print("Done")

# ✅ CORRECT: Always await
async def main():
    await fetch_data()
    print("Done")

# ❌ WRONG: Collecting coroutines without running them
async def process_all(items: list[str]):
    results = [process_item(item) for item in items]  # List of coroutines!
    return results

# ✅ CORRECT: Await or gather
async def process_all(items: list[str]):
    tasks = [asyncio.create_task(process_item(item)) for item in items]
    return await asyncio.gather(*tasks)

# ✅ BETTER: TaskGroup (Python 3.11+)
async def process_all(items: list[str]):
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(process_item(item)) for item in items]
    return [task.result() for task in tasks]
```

### Shared Mutable State

```python
# ❌ WRONG: Shared mutable state without locks
counter = 0

async def increment():
    global counter
    temp = counter
    await asyncio.sleep(0)  # Yield control
    counter = temp + 1  # Race condition!

async def main():
    await asyncio.gather(*[increment() for _ in range(100)])
    print(counter)  # Not 100! Lost updates due to race

# ✅ CORRECT: Use asyncio.Lock
counter = 0
lock = asyncio.Lock()

async def increment():
    global counter
    async with lock:
        temp = counter
        await asyncio.sleep(0)
        counter = temp + 1

async def main():
    await asyncio.gather(*[increment() for _ in range(100)])
    print(counter)  # 100, as expected

# ✅ BETTER: Avoid shared state
async def increment(current: int) -> int:
    await asyncio.sleep(0)
    return current + 1

async def main():
    results = await asyncio.gather(*[increment(i) for i in range(100)])
    print(sum(results))
```

**Why this matters**: Async code is concurrent. Race conditions exist. Use locks or avoid shared mutable state.

---

## Async Patterns

### Fire and Forget

```python
# ❌ WRONG: Creating task without tracking it
async def main():
    asyncio.create_task(background_job())  # Task may not complete!
    return "Done"

# ✅ CORRECT: Track background tasks
background_tasks = set()

async def main():
    task = asyncio.create_task(background_job())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return "Done"

# ✅ CORRECT: Wait for background tasks before exit
async def main():
    task = asyncio.create_task(background_job())
    try:
        return "Done"
    finally:
        await task
```

### Retry with Exponential Backoff

```python
# ❌ WRONG: Retry without delay
async def fetch_with_retry(url: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            return await fetch_data(url)
        except Exception:
            if attempt == max_retries - 1:
                raise
    # Hammers server, no backoff

# ✅ CORRECT: Exponential backoff with jitter
async def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> str:
    for attempt in range(max_retries):
        try:
            return await fetch_data(url)
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            log.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
            await asyncio.sleep(delay)

    raise RuntimeError("Unreachable")
```

### Rate Limiting

```python
# ❌ WRONG: No rate limiting
async def fetch_all(urls: list[str]) -> list[str]:
    tasks = [asyncio.create_task(fetch(url)) for url in urls]
    return await asyncio.gather(*tasks)
# Can overwhelm server with 1000s of concurrent requests

# ✅ CORRECT: Semaphore for concurrent request limit
async def fetch_all(urls: list[str], max_concurrent: int = 10) -> list[str]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_sem(url: str) -> str:
        async with semaphore:
            return await fetch(url)

    tasks = [asyncio.create_task(fetch_with_sem(url)) for url in urls]
    return await asyncio.gather(*tasks)

# ✅ CORRECT: Token bucket rate limiting
class RateLimiter:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # Tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

# Usage
rate_limiter = RateLimiter(rate=10.0, capacity=10)  # 10 req/sec

async def fetch_with_limit(url: str) -> str:
    await rate_limiter.acquire()
    return await fetch(url)
```

**Why this matters**: Rate limiting prevents overwhelming servers and respects API limits. Semaphore limits concurrency, token bucket smooths bursts.

### Async Queue for Producer/Consumer

```python
# ✅ CORRECT: Producer/consumer with asyncio.Queue
import asyncio

async def producer(queue: asyncio.Queue, items: list[str]):
    for item in items:
        await queue.put(item)
        await asyncio.sleep(0.1)  # Simulate work

    # Signal completion
    await queue.put(None)

async def consumer(queue: asyncio.Queue, consumer_id: int):
    while True:
        item = await queue.get()

        if item is None:
            # Re-queue sentinel for other consumers
            await queue.put(None)
            break

        print(f"Consumer {consumer_id} processing {item}")
        await asyncio.sleep(0.2)  # Simulate work
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=10)
    items = [f"item_{i}" for i in range(20)]

    # Start producer and consumers
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(queue, items))
        for i in range(3):
            tg.create_task(consumer(queue, i))

    # Wait for all items processed
    await queue.join()

# ✅ CORRECT: Multiple producers, multiple consumers
async def worker(name: str, queue: asyncio.Queue):
    while True:
        item = await queue.get()
        if item is None:
            break

        await process_item(item)
        queue.task_done()

async def main():
    queue = asyncio.Queue()

    # Create workers
    workers = [asyncio.create_task(worker(f"worker-{i}", queue)) for i in range(5)]

    # Add work
    for item in items:
        await queue.put(item)

    # Wait for all work done
    await queue.join()

    # Stop workers
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers)
```

**Why this matters**: asyncio.Queue is thread-safe and async-safe. Perfect for producer/consumer patterns in async code.

---

## Threading vs Async vs Multiprocessing

### When to Use What

```python
# CPU-bound work: Use multiprocessing
def cpu_bound(n: int) -> int:
    return sum(i * i for i in range(n))

async def process_cpu_tasks(data: list[int]) -> list[int]:
    loop = asyncio.get_running_loop()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        results = await asyncio.gather(*[
            loop.run_in_executor(pool, cpu_bound, n) for n in data
        ])
    return results

# I/O-bound work: Use async
async def io_bound(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

async def process_io_tasks(urls: list[str]) -> list[str]:
    return await asyncio.gather(*[io_bound(url) for url in urls])

# Blocking I/O (no async library): Use threads
def blocking_io(path: str) -> str:
    with open(path) as f:  # Blocking file I/O
        return f.read()

async def process_files(paths: list[str]) -> list[str]:
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        results = await asyncio.gather(*[
            loop.run_in_executor(pool, blocking_io, path) for path in paths
        ])
    return results
```

**Decision tree:**
```
Is work CPU-bound?
├─ Yes → multiprocessing (ProcessPoolExecutor)
└─ No → I/O-bound
    ├─ Async library available? → async/await
    └─ Only sync library? → threads (ThreadPoolExecutor)
```

### Combining Async and Threads

```python
# ✅ CORRECT: Running async code in thread
import threading

def run_async_in_thread(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def sync_function():
    result = run_async_in_thread(async_operation())
    return result

# ✅ CORRECT: Thread-safe async queue
class AsyncThreadSafeQueue:
    def __init__(self):
        self._queue = queue.Queue()

    async def get(self):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._queue.get)

    async def put(self, item):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._queue.put, item)
```

---

## Debugging Async Code

### Common Errors and Solutions

```python
# Error: RuntimeWarning: coroutine 'fetch' was never awaited
# ❌ WRONG:
async def main():
    fetch_data()  # Missing await

# ✅ CORRECT:
async def main():
    await fetch_data()

# Error: RuntimeError: Event loop is closed
# ❌ WRONG:
asyncio.run(coro1())
asyncio.run(coro2())  # Creates new loop, first loop closed

# ✅ CORRECT:
async def main():
    await coro1()
    await coro2()
asyncio.run(main())

# Error: RuntimeError: Task got Future attached to different loop
# ❌ WRONG:
loop1 = asyncio.new_event_loop()
task = loop1.create_task(coro())
loop2 = asyncio.new_event_loop()
loop2.run_until_complete(task)  # Task from different loop!

# ✅ CORRECT: Use same loop
loop = asyncio.new_event_loop()
task = loop.create_task(coro())
loop.run_until_complete(task)
```

### Enabling Debug Mode

```python
# Enable asyncio debug mode for better errors
import asyncio
import logging

# Method 1: Environment variable
# PYTHONASYNCIODEBUG=1 python script.py

# Method 2: In code
asyncio.run(main(), debug=True)

# Method 3: For existing loop
loop = asyncio.get_event_loop()
loop.set_debug(True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Debug mode enables:
# - Warnings for slow callbacks (>100ms)
# - Warnings for coroutines never awaited
# - Better stack traces
```

### Detecting Blocking Code

```python
# ✅ CORRECT: Monitor event loop lag
import asyncio
import time

class LoopMonitor:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.last_check = time.monotonic()

    async def monitor(self):
        while True:
            now = time.monotonic()
            lag = now - self.last_check - 1.0  # Expecting 1 second sleep

            if lag > self.threshold:
                log.warning(f"Event loop blocked for {lag:.3f}s")

            self.last_check = now
            await asyncio.sleep(1.0)

async def main():
    monitor = LoopMonitor()
    asyncio.create_task(monitor.monitor())

    # Your async code here
    await run_application()
```

---

## Async Libraries Ecosystem

### Essential Async Libraries

```python
# HTTP client
import aiohttp

async def fetch(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

# File I/O
import aiofiles

async def read_file(path: str) -> str:
    async with aiofiles.open(path) as f:
        return await f.read()

# Database (PostgreSQL)
import asyncpg

async def query_db():
    conn = await asyncpg.connect('postgresql://user@localhost/db')
    try:
        rows = await conn.fetch('SELECT * FROM users')
        return rows
    finally:
        await conn.close()

# Redis
import aioredis

async def cache_get(key: str) -> str | None:
    redis = await aioredis.create_redis_pool('redis://localhost')
    try:
        value = await redis.get(key)
        return value.decode() if value else None
    finally:
        redis.close()
        await redis.wait_closed()
```

### Async Testing with pytest-asyncio

```python
# Install: pip install pytest-asyncio

import pytest

# Mark async test
@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data("https://api.example.com")
    assert result is not None

# Async fixture
@pytest.fixture
async def http_session():
    async with aiohttp.ClientSession() as session:
        yield session

@pytest.mark.asyncio
async def test_with_session(http_session):
    async with http_session.get("https://api.example.com") as resp:
        assert resp.status == 200
```

---

## Anti-Patterns

### Async Over Everything

```python
# ❌ WRONG: Making everything async without reason
async def calculate_total(prices: list[float]) -> float:
    total = 0.0
    for price in prices:
        total += price  # No I/O, no benefit from async
    return total

# ✅ CORRECT: Keep sync when no I/O
def calculate_total(prices: list[float]) -> float:
    return sum(prices)

# ❌ WRONG: Async wrapper for sync function
async def async_sum(numbers: list[int]) -> int:
    return sum(numbers)  # Why?

# ✅ CORRECT: Only async when doing I/O
async def fetch_and_sum(urls: list[str]) -> int:
    results = await asyncio.gather(*[fetch_number(url) for url in urls])
    return sum(results)  # sum() is sync, that's fine
```

### Creating Too Many Tasks

```python
# ❌ WRONG: Creating millions of tasks
async def process_all(items: list[str]):  # 1M items
    tasks = [asyncio.create_task(process(item)) for item in items]
    return await asyncio.gather(*tasks)
# Problem: Creates 1M tasks, high memory usage

# ✅ CORRECT: Batch processing with semaphore
async def process_all(items: list[str], max_concurrent: int = 100):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_sem(item: str):
        async with semaphore:
            return await process(item)

    return await asyncio.gather(*[process_with_sem(item) for item in items])

# ✅ BETTER: Process in batches
async def process_all(items: list[str], batch_size: int = 100):
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[process(item) for item in batch])
        results.extend(batch_results)
    return results
```

### Mixing Sync and Async Poorly

```python
# ❌ WRONG: Calling asyncio.run inside async function
async def bad_function():
    result = asyncio.run(some_async_function())  # Error!
    return result

# ✅ CORRECT: Just await
async def good_function():
    result = await some_async_function()
    return result

# ❌ WRONG: Sync wrapper calling async repeatedly
def process_all_sync(items: list[str]) -> list[str]:
    return [asyncio.run(process(item)) for item in items]
# Creates new event loop for each item!

# ✅ CORRECT: Single event loop
def process_all_sync(items: list[str]) -> list[str]:
    async def process_all_async():
        return await asyncio.gather(*[process(item) for item in items])

    return asyncio.run(process_all_async())
```

---

## Decision Trees

### Should I Use Async?

```
Does my code do I/O? (network, files, database)
├─ No → Don't use async (CPU-bound work)
└─ Yes → Does an async library exist?
    ├─ Yes → Use async/await
    └─ No → Can I use sync library with threads?
        ├─ Yes → Use run_in_executor with ThreadPoolExecutor
        └─ No → Rethink approach or write async wrapper
```

### Concurrent Execution Strategy

```
What am I waiting for?
├─ Network/database → async/await (asyncio)
├─ File I/O → async/await with aiofiles
├─ CPU computation → multiprocessing (ProcessPoolExecutor)
├─ Blocking library (no async version) → threads (ThreadPoolExecutor)
└─ Nothing (pure computation) → Regular sync code
```

### Error Handling in Concurrent Tasks

```
Do I need all results?
├─ Yes → TaskGroup (3.11+) or gather without return_exceptions
│   └─ Fails fast on first error
└─ No (partial results OK) → gather with return_exceptions=True
    └─ Filter exceptions from results
```

---

## Integration with Other Skills

**After using this skill:**
- If profiling async code → See @debugging-and-profiling for async profiling
- If testing async code → See @testing-and-quality for pytest-asyncio
- If setting up project → See @project-structure-and-tooling for async dependencies

**Before using this skill:**
- If code is slow → Use @debugging-and-profiling to verify it's I/O-bound first
- If starting project → Use @project-structure-and-tooling to set up dependencies

---

## Quick Reference

### Python 3.11+ Features

| Feature | Description | When to Use |
|---------|-------------|-------------|
| TaskGroup | Structured concurrency | Multiple concurrent tasks, automatic cleanup |
| asyncio.timeout() | Context manager for timeouts | Cleaner than wait_for() |
| except* | Exception group handling | Handle multiple concurrent errors |

### Common Async Patterns

```python
# Concurrent execution
async with asyncio.TaskGroup() as tg:
    tasks = [tg.create_task(func(x)) for x in items]
results = [t.result() for t in tasks]

# Timeout
async with asyncio.timeout(10.0):
    result = await long_operation()

# Rate limiting
semaphore = asyncio.Semaphore(10)
async with semaphore:
    await rate_limited_operation()

# Retry with backoff
for attempt in range(max_retries):
    try:
        return await operation()
    except Exception:
        await asyncio.sleep(2 ** attempt)
```

### When NOT to Use Async

- Pure computation (no I/O)
- Single I/O operation (overhead not worth it)
- CPU-bound work (use multiprocessing)
- When sync code is simpler and performance is acceptable
