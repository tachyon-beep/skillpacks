# External Effects Substitution

## Overview

**A deterministic system is a closed world. Every input either lives in the run config or is recorded as it enters. Wall-clock reads, network responses, third-party API calls, file-system iteration, environment variables, and user input are leaks; each one drags the run's identity into something the run record does not own. The remedy is *substitution* — replace the leak with a recorded source — and the substitution machinery is the load-bearing part of replay infrastructure.**

This sheet defines what counts as an external effect, the three substitution patterns (mock, record-and-replay, deterministic substitute), and how to wire substitution into the replay loop without leaving back-channels open. The deliverable is `10-external-effects-substitution.md`.

## When to Use

Use this sheet when:

- The system reads `time.time()`, `time.monotonic()`, `datetime.now()`, or any wall-clock for any decision-affecting purpose.
- The system makes network calls, reads files whose contents change, queries a database, or calls a third-party API.
- The system reads environment variables, command-line arguments outside the config, or user input mid-run.
- The system iterates a directory, reads a glob, or walks a file tree (order is OS-dependent).
- A determinism class above XS is required.

Do not use this sheet for:

- Pure simulations with no external IO (the issue does not arise).
- Configuration values read once at startup and recorded in the run record (already an input, not a leak).

## Core Principle

> A run is reproducible if and only if every byte of input is in the run record. External effects either become inputs (recorded), become substitutes (deterministic functions of recorded inputs), or become forbidden. There is no "we'll just hit the live API; it's stable enough."

## What Counts as an External Effect

Anything that is *not* a function of `(seed, config, code-version)`. The full list:

| Category | Examples | Smell |
|----------|---------|-------|
| Time | `time.time`, `time.monotonic`, `datetime.now`, `Date.now`, `Instant::now` | Decisions depend on "when this ran" |
| Random ambient | `os.urandom`, `random.SystemRandom`, `/dev/random`, `crypto.randomBytes` | "Cryptographically random" reads |
| File system iteration | `os.listdir`, `glob`, `Path.iterdir`, `find`, `readdir` | Order varies by FS / kernel / capacity |
| File contents (mutable) | Reading a database file, a log being written by another process | Racy reads |
| Network | HTTP, gRPC, websockets, raw sockets | Any call to a service not in the run record |
| Environment variables read after startup | `os.getenv` mid-run, container env vars sniffed at request time | Process state leaks |
| User input | stdin reads, GUI events, terminal interaction | Operator decisions are inputs |
| OS scheduling | thread/process creation timing, IPC arrival order | Already covered in `determinism-under-concurrency.md` |
| Hardware state | CPU frequency, thermal throttling, available memory affecting algorithm choice | Indirect; rare but real |
| Third-party libraries reading any of the above | `tqdm` reads `TERM`; `tensorboard` reads `time`; ML frameworks read `CUDA_VISIBLE_DEVICES` mid-run | Library leaks count |

The audit is: grep for `time(`, `now(`, `urandom`, `getenv`, `listdir`, `glob`, `requests.`, `socket.`, `open(` (with non-config paths). Every hit is a candidate leak.

## The Three Substitution Patterns

### Pattern 1 — Substitute with a deterministic function of the run

The cleanest fix when the external value can be computed from the run's identity.

```python
# Forbidden:
def make_run_id() -> str:
    return f"run-{int(time.time())}"

# Substituted:
def make_run_id(seed: int, config_hash: str) -> str:
    return f"run-{seed:08x}-{config_hash[:8]}"
```

Use when: the value is informational or organisational (a run name, a checkpoint filename, a directory name).

### Pattern 2 — Record at first read; replay from record

The standard fix when the value is genuinely external but should be reproducible.

```python
class ExternalClock:
    def __init__(self, mode: Literal["record", "replay"], log_path: Path):
        self.mode = mode
        self.log_path = log_path
        self.cursor = 0
        self.records: list[float] = []
        if mode == "replay":
            self.records = json.loads(log_path.read_text())

    def now(self) -> float:
        if self.mode == "record":
            t = time.time()
            self.records.append(t)
            return t
        else:
            t = self.records[self.cursor]
            self.cursor += 1
            return t

    def finalise(self) -> None:
        if self.mode == "record":
            self.log_path.write_text(json.dumps(self.records))
```

Use when: the value's actual content matters (timestamps in log entries, server responses, file contents at moment of read) and the run must be replayable.

The recorded log is part of the run output, signed if the run is signed (cross-link `axiom-audit-pipelines`), retained as long as replay is required. `replay` mode reads from the log and asserts cursor advances exactly as the recorded run did — any deviation is a divergence event (cross-link `divergence-detection-and-localisation.md`).

### Pattern 3 — Mock with a fixed deterministic substitute

The fix when the external is irrelevant to the deterministic logic and the test can fix it to any value.

```python
# Test only — not used in production runs:
mock_clock = lambda: 1700000000.0  # frozen
```

Use when: testing, when the actual time/network response/etc. plays no semantic role in the deterministic logic.

**Pattern 3 is for tests only.** A production run that uses mocks for external effects is not a deterministic system; it is a system that hides external effects. The distinction matters: Pattern 2 records the actual external; Pattern 3 hides it. Mocking in production is forbidden by `01-`'s class promise.

## Wiring the Substitution Layer

The substitution layer is a single object the system passes to every component that touches an external. Components do not call `time.time()` directly; they call `effects.now()`. Components do not call `requests.get()`; they call `effects.http_get()`.

```python
@dataclass
class Effects:
    clock: ExternalClock
    rng_external: ExternalRandom    # for /dev/urandom-style needs
    network: ExternalNetwork
    fs: ExternalFileSystem
    env: ExternalEnv

    @classmethod
    def for_record(cls, log_dir: Path) -> "Effects": ...

    @classmethod
    def for_replay(cls, log_dir: Path) -> "Effects": ...
```

The constructor is the choke-point: in record mode, every component reads from the live source and the read is logged; in replay mode, every component reads from the log. There is no third constructor (`for_live`, `for_test`) that lets components touch a real external without logging — that constructor is the back-channel `01-` forbids.

## Time as the Canonical Case

Time is the most-leaked external effect and the most-rationalised. The patterns:

| Time use | Substitution |
|----------|--------------|
| "Tick at 60 Hz" cadence | Logical clock — use a tick counter, not wall-clock; advance once per logical tick. Wall-clock is decoration. |
| Timestamps in log entries | Recorded clock; replay reads from log. |
| `random.seed(time.time())` | Already forbidden by `seed-governance.md`. |
| Performance measurement | Recorded clock if perf is part of the run record; otherwise external observer (separate from the run). |
| Timeouts | Logical timeout (N ticks); not wall-clock duration. |
| TTLs and cache expiry | Logical TTLs; or recorded clock with explicit semantics in replay. |
| Date-stamped paths and filenames | Pattern 1 — derive from run identity. |

## Network as the Hardest Case

Network calls have side effects on the remote, which complicate Pattern 2. Three sub-patterns:

1. **Read-only, idempotent calls** (GET, query): record the response; replay from record.
2. **Side-effecting calls** (POST, write): record the response; replay does NOT re-issue. The replay machinery returns the recorded response without making the call. This is correct: replay is observation of past behaviour, not re-execution against the real world.
3. **Streaming / long-lived connections**: record the stream as a sequence of events; replay reads events from the log.

The trap: side-effecting calls in replay that *do* re-issue corrupt the world (double charge, double email). The Effects layer must distinguish modes; the corollary is that the same code reads `effects.network` but the layer's behaviour differs by mode. This is by design.

## Idempotency and Safety in Replay

Replay must not affect the outside world. The Effects layer's `for_replay` constructor is the seal: any code path that bypasses Effects to touch a real external during replay is a bug. The audit:

```bash
# Confirm: in replay mode, no network/file/clock call escapes the layer
$ grep -rn "requests\." --include="*.py" src/
# Each hit must be inside Effects.network.* or in test code
```

CI runs the audit. Hits outside Effects fail the build.

## Closed-World Assertion

The Effects layer's record mode emits a manifest at run end:

```json
{
  "clock_reads": 8472,
  "network_reads": 42,
  "fs_reads": 12,
  "env_reads": 3,
  "first_read_offset_ns": 1023,
  "log_total_bytes": 28471
}
```

Replay verifies the manifest: same number of reads in the same order. A replay that consumes fewer reads than recorded is a divergence (cross-link `divergence-detection-and-localisation.md` Step 3 — replayed code path differs from recorded). A replay that consumes more is a code change.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `time.time()` called from inside the deterministic loop | Wrap in `Effects.clock.now()`; record/replay. |
| Wall-clock cadence (`time.sleep(1/60)`) | Logical tick counter; cadence is decoration. |
| `os.listdir` iterated raw | Sort the result; or wrap in `Effects.fs.list_sorted()`. |
| Network call in replay mode reaches the real network | Effects.for_replay returns recorded; assert no real socket opens. |
| Pattern 3 (mocking) used in production for "convenience" | Mocks are tests-only. Use Pattern 2. |
| Recorded log not part of run record | The log IS the run; without it the run is unreplayable. |
| Replay log replayed in wrong order | Cursor advances strictly; out-of-order read is a divergence. |
| `os.getenv` read mid-run | Either freeze env at startup and put it in run config, or wrap in Effects.env. |
| Side-effecting POST re-issued during replay | Replay must not re-issue. Effects.network.post in replay mode returns recorded response without HTTP. |
| Per-component clock substitution (each subsystem its own) | Single Effects layer; one substitution machinery per run. |

## Spec Output (`10-external-effects-substitution.md`)

The sheet's deliverable answers:

1. **External-effect inventory** — every category in the table above, with each occurrence in the codebase enumerated. Per occurrence: which substitution pattern, why.
2. **Effects layer interface** — the data class, the methods, the constructor signatures (`for_record`, `for_replay`).
3. **Per-effect substitution rule** — for each external (clock, network, fs, env, ...): the recording format, the replay semantics, the side-effect policy.
4. **Closed-world assertion** — the manifest format, the assertion machinery, the divergence-on-mismatch behaviour.
5. **Audit procedure** — the grep patterns, the build-time check that confirms no external is touched outside Effects.
6. **Test vectors** — for each effect type, at minimum one recorded run + replay producing identical state hashes.
7. **Class-breaking events** — new external introduced, substitution pattern changed for an effect, recording format changed, audit rule changed.
8. **Cross-link rules** — how the recorded log integrates with `04-snapshot-strategy` (size and storage), with `05-divergence-protocol` (mismatch handling), with `06-replay-infrastructure-spec` (constructor wiring).
9. **Forbidden constructors** — the explicit list of constructors that do *not* exist (`for_live_with_logging`, `for_test_in_production`).

Without these nine items the spec is incomplete and Check 14 (external effects) of the consistency gate will fail.

## Cross-Pack Notes

- `axiom-audit-pipelines`: the recorded log of external effects can be written into the audit trail with the same canonical encoding and signing rules; the two share canonical-encoding hygiene (`canonical-state-encoding-for-replay.md`).
- `yzmir-deep-rl`: environment substrates often have a "step" function that internally reads time or filesystem; vec-env wrappers must route through Effects to stay deterministic.
- `axiom-solution-architect`: side-effecting external calls are typical ADR territory; this sheet's "no re-issue in replay" rule is the relevant constraint.
- `axiom-static-analysis-engineering`: linters that flag `time.time()` and `os.listdir` outside Effects layer are cheap and effective; encode the substitution rule as a custom rule.

## The Bottom Line

**A deterministic system is a closed world. Every external read is an input — recorded as it enters, replayed in order, never re-issued in replay mode for side-effecting calls. Wall-clock is decoration; logical tick is the truth. Mocks are for tests; production records what really happened. The Effects layer is the single choke-point; back-channels around it are bugs by definition.**
