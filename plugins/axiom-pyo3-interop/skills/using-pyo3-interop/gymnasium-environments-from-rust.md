---
name: gymnasium-environments-from-rust
description: Use when exposing a Rust simulation as a Gymnasium environment — observation/action contracts, episode boundaries, reset semantics, vectorised environments, observation-copy avoidance. The canonical PyO3-for-RL bridge pattern. Produces `07-gymnasium-environments-from-rust.md`.
---

# Gymnasium Environments Backed by Rust

## Overview

**Core principle: Gymnasium (the maintained successor to OpenAI Gym) is the de facto Python ABI for RL environments. Backing it with Rust gives you 10–100× faster step rates than pure-Python envs while preserving compatibility with every Python policy, every wrapper, every algorithm in the RL ecosystem. The pattern is canonical: `#[pyclass]` exposes `reset`, `step`, `render`, `close`; `observation_space` and `action_space` come from `gymnasium.spaces`; observations are zero-copy NumPy arrays; episode boundaries are explicit. Done well, Python becomes the orchestration layer (policy, training, logging) and Rust the simulation kernel.**

This is the pattern murk ships and that the wider RL/sim community has converged on. The pitfalls are observation-copy cost (per-step allocation kills throughput), episode-boundary semantics (the `terminated` vs `truncated` distinction), vectorised env GIL discipline, and reset determinism.

## The Gymnasium Contract

A Gymnasium environment exposes:

```python
class MyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, **kwargs): ...
    def reset(self, *, seed=None, options=None) -> tuple[obs, info]: ...
    def step(self, action) -> tuple[obs, reward, terminated, truncated, info]: ...
    def render(self) -> np.ndarray | None: ...
    def close(self): ...

    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
```

The Rust-backed env is a `#[pyclass]` that satisfies this contract. The trick is mapping Rust's owned simulation state to Gymnasium's expected types without copying every step.

## Minimal Skeleton

```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use numpy::{IntoPyArray, PyArray1};

#[pyclass]
pub struct MyEnv {
    state: SimState,
    rng: rand_pcg::Pcg64,
    step_count: u32,
    max_steps: u32,
}

#[pymethods]
impl MyEnv {
    #[new]
    #[pyo3(signature = (max_steps = 1000))]
    fn new(max_steps: u32) -> Self {
        MyEnv {
            state: SimState::default(),
            rng: rand_pcg::Pcg64::seed_from_u64(0),
            step_count: 0,
            max_steps,
        }
    }

    /// Returns (observation, info_dict).
    #[pyo3(signature = (*, seed = None, options = None))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
        options: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyDict>)> {
        let _ = options;   // unused in this example
        if let Some(s) = seed {
            self.rng = rand_pcg::Pcg64::seed_from_u64(s);
        }
        self.state = SimState::random(&mut self.rng);
        self.step_count = 0;

        let obs = self.state.observation();      // Vec<f32>
        let info = PyDict::new(py);
        Ok((obs.into_pyarray(py), info))
    }

    /// Returns (observation, reward, terminated, truncated, info).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: i32,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, f32, bool, bool, Bound<'py, PyDict>)> {
        let (reward, terminated) = py.allow_threads(|| self.state.advance(action));
        self.step_count += 1;
        let truncated = self.step_count >= self.max_steps;

        let obs = self.state.observation();
        let info = PyDict::new(py);
        Ok((obs.into_pyarray(py), reward, terminated, truncated, info))
    }

    fn close(&mut self) {
        // Release any Rust-side resources (handles, file descriptors).
    }

    #[getter]
    fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let gym = py.import("gymnasium")?;
        let spaces = gym.getattr("spaces")?;
        let box_cls = spaces.getattr("Box")?;

        let low = vec![-1.0f32; SimState::OBS_DIM].into_pyarray(py);
        let high = vec![1.0f32; SimState::OBS_DIM].into_pyarray(py);
        let kwargs = PyDict::new(py);
        kwargs.set_item("low", low)?;
        kwargs.set_item("high", high)?;
        kwargs.set_item("dtype", py.import("numpy")?.getattr("float32")?)?;
        box_cls.call((), Some(&kwargs))
    }

    #[getter]
    fn action_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let gym = py.import("gymnasium")?;
        let spaces = gym.getattr("spaces")?;
        let discrete = spaces.getattr("Discrete")?;
        discrete.call1((SimState::N_ACTIONS,))
    }
}
```

A Python user does:

```python
import mymod
env = mymod.MyEnv(max_steps=500)
obs, info = env.reset(seed=42)
done = False
total = 0.0
while not done:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    done = terminated or truncated
```

This works with every Gymnasium-compatible policy, every wrapper (`gym.wrappers.TimeLimit`, `RecordVideo`, etc.), every algorithm in stable-baselines3, RLlib, CleanRL, or your custom training loop.

## The Observation-Copy Trap

The skeleton above does `obs.into_pyarray(py)` per step. That allocates a new NumPy array, copies (well, hands ownership) of the `Vec<f32>` into it. For a 100-element observation at 100k steps/sec, that's 100k allocations/sec — measurable allocator pressure.

The fix: pre-allocate one `PyArray1<f32>` per env, reuse:

```rust
#[pyclass]
pub struct MyEnv {
    state: SimState,
    obs_buffer: Py<PyArray1<f32>>,    // owned across calls
    /* ... */
}

#[pymethods]
impl MyEnv {
    #[new]
    fn new(py: Python<'_>) -> Self {
        let obs_buffer = numpy::PyArray1::zeros(py, SimState::OBS_DIM, false).unbind();
        MyEnv { obs_buffer, /* ... */ }
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: i32,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, f32, bool, bool, Bound<'py, PyDict>)> {
        let (reward, terminated) = py.allow_threads(|| self.state.advance(action));

        // Write into the persistent buffer.
        let bound = self.obs_buffer.bind(py).clone();
        {
            let mut view = unsafe { bound.as_array_mut() };  // exclusive access; we're the only owner
            self.state.write_observation(&mut view);
        }

        let truncated = false;
        let info = PyDict::new(py);
        Ok((bound, reward, terminated, truncated, info))
    }
}
```

Caveats:

- The buffer is *the same array each step*. If the Python caller stores past observations (a replay buffer), they need to **copy** the array (`obs.copy()`). Document this clearly in `__init__.pyi` or the docstring; otherwise the replay buffer fills with N copies of the latest obs.
- `unsafe { as_array_mut }` is needed because PyO3's standard borrow tracker wouldn't normally allow this — we are asserting that no other Rust code is reading the buffer. Since the buffer is `Py<>`-owned and only we touch it, this is safe.
- Alternative: emit a *slice* of a larger pre-allocated arena, and have the caller treat each slice as a one-shot view. The lifetime contract is then explicit.

For most RL training, **copy-on-step is acceptable**: the copy cost is < the policy forward pass; the simplicity is worth it. Switch to reused buffers only when the profile shows allocation dominating.

## `terminated` vs `truncated`

A subtle but critical distinction:

- **`terminated`** — the episode ended due to environment dynamics (the agent reached the goal, fell off the cliff, ran out of HP).
- **`truncated`** — the episode ended due to an external clock (max steps reached, wall-clock budget exhausted).

The distinction matters for value-bootstrapping in RL algorithms: `terminated=True` means the next state has no value (terminal state), while `truncated=True` means the next state *would* have value but we're stopping for non-environmental reasons (so bootstrap from the policy).

A Rust-backed env that always returns `truncated = step_count >= max_steps` and `terminated` from the simulation correctly mirrors this. The most common bug is conflating the two: `done = terminated or truncated` is what the user sees, but the env must report them separately.

## Vectorised Environments

For batched RL training, frameworks expect a `VectorEnv` API:

```python
envs = gym.vector.SyncVectorEnv([lambda: mymod.MyEnv() for _ in range(N)])
obs, info = envs.reset()              # obs shape (N, obs_dim)
actions = policy(obs)                 # shape (N,)
obs, rewards, terminated, truncated, info = envs.step(actions)
```

`SyncVectorEnv` calls `env.step()` on each env serially under one Python thread. Performance is bounded by per-step Python overhead × N; for small obs and fast envs the GIL is the bottleneck.

`AsyncVectorEnv` runs each env in its own subprocess; bypasses the GIL but pays IPC cost (typically ~50 µs per step due to pipe serialization).

Best practice for Rust-backed envs: **implement vectorisation in Rust**.

```rust
#[pyclass]
pub struct VecEnv {
    envs: Vec<MyEnv>,
}

#[pymethods]
impl VecEnv {
    #[new]
    fn new(n: usize) -> Self {
        VecEnv { envs: (0..n).map(|_| MyEnv::new(1000)).collect() }
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: numpy::PyReadonlyArray1<'py, i32>,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray2<f32>>,
        Bound<'py, numpy::PyArray1<f32>>,
        Bound<'py, numpy::PyArray1<bool>>,
        Bound<'py, numpy::PyArray1<bool>>,
        Bound<'py, pyo3::types::PyList>,
    )> {
        let actions_view = actions.as_array();
        let n = self.envs.len();
        let obs_dim = SimState::OBS_DIM;

        // Pre-allocate output buffers.
        let mut obs = vec![0f32; n * obs_dim];
        let mut rewards = vec![0f32; n];
        let mut terminated = vec![false; n];
        let mut truncated = vec![false; n];

        py.allow_threads(|| {
            // Parallel step over envs.
            use rayon::prelude::*;
            self.envs.par_iter_mut().enumerate().for_each(|(i, env)| {
                let action = actions_view[i];
                let (r, t) = env.advance(action);
                rewards[i] = r;
                terminated[i] = t;
                truncated[i] = env.step_count >= env.max_steps;
                env.write_observation(&mut obs[i * obs_dim..(i + 1) * obs_dim]);
            });
        });

        let infos = pyo3::types::PyList::empty(py);
        for _ in 0..n {
            infos.append(pyo3::types::PyDict::new(py))?;
        }

        Ok((
            ndarray::Array2::from_shape_vec((n, obs_dim), obs).unwrap().into_pyarray(py),
            rewards.into_pyarray(py),
            terminated.into_pyarray(py),
            truncated.into_pyarray(py),
            infos,
        ))
    }
}
```

One Python call drives N env steps in parallel. The boundary cost is paid once. The simulation runs on all cores. Throughput improves by N× (modulo Amdahl).

## Reset Determinism

`reset(seed=42)` must produce the same first observation every time. The Rust env owns its RNG; reseed it explicitly:

```rust
fn reset<'py>(&mut self, py: Python<'py>, seed: Option<u64>) -> PyResult<...> {
    if let Some(s) = seed {
        self.rng = rand_pcg::Pcg64::seed_from_u64(s);
    }
    /* ... */
}
```

Notes:

- Use a **counter-based or PCG-style** RNG (deterministic, fast, splittable). Avoid `rand::thread_rng()` (per-thread, non-deterministic).
- For vectorised envs, seed each sub-env with `seed + i` (or a more sophisticated stream split) so they don't all produce the same trajectory.
- See `axiom-determinism-and-replay` for the broader determinism discipline.

## Rendering

`render() -> np.ndarray | None` returns a frame for video logging. For a 2D / 3D simulation, the natural form is an `(H, W, 3)` `uint8` array.

```rust
fn render<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, numpy::PyArray3<u8>>>> {
    if self.render_mode != RenderMode::RgbArray {
        return Ok(None);
    }
    let frame: ndarray::Array3<u8> = self.state.render_to_rgb();
    Ok(Some(frame.into_pyarray(py)))
}
```

`render_mode = "human"` is window-based; `render_mode = "rgb_array"` is array-based. Document the supported modes in `metadata`.

## Wrapping the Env in Python

Some Gymnasium-isms are easier in Python. The `__init__.py` wraps the Rust class:

```python
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import numpy as np

from mymod._native import MyEnv as _RustEnv

class MyEnv(Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_steps: int = 1000, render_mode: str | None = None):
        self._inner = _RustEnv(max_steps)
        self.render_mode = render_mode
        self.observation_space = Box(
            low=-1.0, high=1.0, shape=(N,), dtype=np.float32,
        )
        self.action_space = Discrete(N_ACTIONS)

    def reset(self, *, seed=None, options=None):
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        return self._inner.step(action)
```

The wrapper:

- Lets `__init__` validate kwargs Pythonically.
- Gives `gymnasium.make` a clean class to register.
- Hides the `_native` import path from users.

For projects that want to keep all logic in Rust, the wrapper is optional (the `#[pyclass]` itself can satisfy the protocol). For projects that anticipate adding wrappers (action repeat, frame stack, normalisation), the Python class is the better seam.

## Pitfalls

| Pitfall                                                  | Symptom                                                          | Fix                                                                                  |
|----------------------------------------------------------|------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Allocating a new obs array every step                     | Allocator dominates profile in long episodes                     | Reuse a `Py<PyArray1>` buffer; document that callers must `.copy()` to retain        |
| Conflating `terminated` and `truncated`                   | Bootstrapping wrong; advantage estimates biased near step limits  | Return both separately; never collapse to `done`                                      |
| GIL held through the simulation step                       | Vectorised env Python threads serialise                          | Wrap the simulation in `py.allow_threads(|| ...)`                                     |
| `SyncVectorEnv` over Rust env (cross-Python loop)          | Throughput < expected; per-env overhead × N                      | Implement `VecEnv` in Rust; one Python call per batched step                          |
| Non-deterministic reset                                    | RL training non-reproducible despite `seed=`                     | Use a deterministic RNG; explicitly reseed in reset                                   |
| `observation_space` rebuilt every call                     | Slow `env.observation_space` access                               | Cache the space object as a `Py<PyAny>` field; return cached on getter                |
| Holding `Bound<'_, _>` across multiple steps               | Compile error or unsafe lifetime                                  | Use `Py<>` for stored, `Bound<'py, _>` only within a single method call              |
| Wrong dtype on observation                                  | `gym.vector` complains about dtype mismatch with declared space  | Match `np.float32` (or whatever) consistently; verify `observation_space.dtype`       |
| `info` dict shared across envs in vectorised step          | Mutating one env's info changes another's                         | Allocate a fresh `PyDict::new(py)` per env per step                                   |

## Quick Reference

| Concern                                | Pattern                                                                            |
|----------------------------------------|------------------------------------------------------------------------------------|
| Single-env class                        | `#[pyclass]` with `reset`, `step`, `close`, `observation_space`, `action_space`    |
| `reset` signature                       | `(self, *, seed=None, options=None) -> (obs, info)`                                |
| `step` signature                        | `(self, action) -> (obs, reward, terminated, truncated, info)`                     |
| Spaces                                   | Build via `gymnasium.spaces.Box / Discrete / MultiDiscrete / Dict` in a getter      |
| Reused obs buffer                       | `Py<PyArray1<T>>` field; write via `as_array_mut`; document copy-on-store          |
| Vectorised env                           | One `#[pyclass]` owning `Vec<Env>`; rayon-parallel step; pre-allocated outputs     |
| Determinism                              | Counter-based / PCG RNG; reseed in `reset`; seed-stream split for sub-envs         |
| Rendering                                 | `render() -> Option<Bound<'py, PyArray3<u8>>>`; declare modes in `metadata`        |

## Cross-References

- [`pyo3-fundamentals.md`](pyo3-fundamentals.md) — the `#[pyclass]` machinery
- [`numpy-buffer-protocol.md`](numpy-buffer-protocol.md) — observation arrays, zero-copy
- [`gil-release-patterns.md`](gil-release-patterns.md) — release for the simulation step
- [`batched-ffi-operations.md`](batched-ffi-operations.md) — vectorised env is the canonical batch case
- [`error-mapping-and-traceback-fidelity.md`](error-mapping-and-traceback-fidelity.md) — `step()` errors should map cleanly
- `axiom-determinism-and-replay` — determinism discipline (RNG, replayable trajectories)
- `yzmir-deep-rl` — RL training algorithms that consume this env
