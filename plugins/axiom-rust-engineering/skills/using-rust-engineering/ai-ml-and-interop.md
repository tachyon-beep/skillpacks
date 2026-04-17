# AI/ML and Interop

## Overview

**Core Principle:** Rust shines at inference, deployment, and numerical kernels. It rarely wins at research iteration. Know which problem you are solving before choosing the language.

Rust's guarantees — no GC pauses, predictable latency, zero-cost abstractions, fearless concurrency — are valuable in production inference serving, hot-loop numerical kernels, and data preprocessing pipelines. They are largely irrelevant during research: when you are changing the model architecture every afternoon, Python's interactivity and PyTorch's autograd machinery are far more valuable than Rust's type system.

The canonical pattern in modern ML infrastructure is a Python research surface backed by Rust production components. You write the training loop in PyTorch; you write the serving endpoint, tokenizer, and preprocessing transforms in Rust; PyO3 stitches them together when needed. That boundary — Python for iteration, Rust for deployment — is where this sheet lives.

What this sheet covers: PyO3 extension modules, the candle and burn inference frameworks, tch-rs libtorch bindings, ndarray and nalgebra for numerical computation, ML-specific serialization, and performance notes for tight numerical kernels.

What this sheet does not cover: training pipelines, gradient computation, automatic differentiation, or research workflow tooling. Those belong in Python with PyTorch. See the Yzmir skillpack for that work.

Baseline: Rust stable 1.87, 2024 edition, PyO3 0.22+, candle 0.8+, burn 0.16+.

> **PyO3 version drift (0.22 → 0.23 → 0.24)**: all the Python-binding code in
> this sheet targets the `Bound<'py, T>` API that became primary in 0.21 and is
> the only API in 0.22. Key transitions to know about:
>
> - **0.23 (Nov 2024)** introduced `IntoPyObject` to replace the legacy `IntoPy`
>   / `ToPyObject` traits (both deprecated) and added experimental support for
>   free-threaded CPython (3.13t). Expect deprecation warnings on any
>   `impl IntoPy<PyObject>` or `#[derive(ToPyObject)]` — the replacement is
>   `impl<'py> IntoPyObject<'py> for ...`.
> - **0.24 (early 2025)** continues the `Bound` migration and is tightening the
>   GIL-token API; `Python::with_gil` is the stable entry point, and the older
>   `&PyAny` / `Py<PyAny>` return types are being phased out in favour of
>   `Bound<'py, PyAny>`. Track the `pyo3` CHANGELOG before bumping majors.
>
> The `Bound` patterns in this sheet are forward-compatible with 0.23 and 0.24.

## When to Use

Use this sheet when:

- "I need to call a Rust function from Python without ctypes."
- "I'm building a Python extension module with maturin."
- "I want to run inference on a safetensors model in Rust."
- "My Python data pipeline is too slow — can I move the hot loop to Rust?"
- "How do I use candle for embedding inference?"
- "Should I use burn or candle for this inference service?"
- "I need libtorch access from Rust — is tch-rs the answer?"
- "How do I avoid holding the GIL in a PyO3 extension?"
- "What's the right crate for N-dimensional array computation in Rust?"
- "How do I serialize ML model weights in a Rust binary?"

**Trigger keywords**: `pyo3`, `maturin`, `#[pyfunction]`, `#[pyclass]`, `candle`, `burn`, `tch-rs`, `ndarray`, `nalgebra`, `safetensors`, PyO3, GIL, `allow_threads`, `PyResult`, extension module, ONNX, tokenizer, embedding, inference server, numerical kernel, BLAS.

## When NOT to Use

- **Training pipelines**: do not rewrite PyTorch training in Rust. Autograd, mixed-precision scaling, gradient checkpointing, and distributed training are deeply integrated with Python. The velocity cost is not worth it. Stay in Python.
- **Research iteration**: when you are exploring architectures, tuning hyperparameters, or running ablations, Rust's compile cycle is a productivity tax. Python notebooks are the right tool.
- **ONNX export and runtime**: if your model exports cleanly to ONNX, `onnxruntime` has excellent Rust bindings (`ort` crate) and handles many backends. Consider it before building custom inference logic.
- **GPU-heavy training**: `tch-rs` and `burn` support CUDA, but the toolchain coupling (libtorch version, CUDA version, driver version) is significantly harder to manage than Python. Unless you have a specific reason, train in Python and serve in Rust.
- **You have not profiled the Python code**: before porting anything to Rust, prove the Python is actually the bottleneck. A well-vectorized NumPy kernel is often faster than naive Rust, and the Python is easier to maintain.

---

## When Rust Makes Sense for ML

### Inference serving

Rust excels at serving ML models because the requirements — low latency, high throughput, predictable tail latency, no GC pauses — are exactly what Rust is designed for.

- **Tokenizers**: Hugging Face's `tokenizers` library is a Rust library with Python bindings. It is 20–100× faster than pure-Python tokenization at scale.
- **Pre/post processing**: feature engineering, normalization, decoding — these are pure computation with no dynamic dispatch requirement. Rust handles them well.
- **Embedding retrieval and similarity search**: approximate nearest-neighbor lookups benefit from Rust's memory layout control.
- **Serving endpoint**: an Axum or actix-web endpoint adds negligible overhead compared to a Python WSGI/ASGI server. Combined with candle or an ONNX runtime, you get a single-binary ML server.

### Data preprocessing and ETL

Python data pipelines (pandas, Polars, Spark) are appropriate at moderate scale, but when you need:
- Streaming processing of files too large to fit in memory
- Custom binary format parsing
- CPU-bound transformation running in a tight multi-threaded loop
- Deterministic memory usage (no GC pressure surprises)

...Rust with `rayon` and `ndarray` often produces meaningfully simpler production code than equivalent Python.

### Edge and embedded inference

`no_std`-compatible inference with `candle` (CPU-only, no std) or via ONNX runtime on embedded targets. Rust is the natural choice when you cannot run a Python interpreter.

### Hot-loop kernels called from Python

The most common pattern: Python orchestrates, Rust computes. A PyO3 extension wraps a tight Rust function (matrix multiply, tokenization, custom CUDA kernel wrapper, feature hashing) and is called from Python as if it were a C extension. You get Python ergonomics for the orchestration logic and Rust performance for the bottleneck.

### When NOT to use Rust for ML

This deserves explicit reinforcement because the mistake is common:

**Do not rewrite your training pipeline in Rust.** The benefits of Rust (memory safety, no GC) do not address the pain points of training (slow convergence, wrong loss curves, overfitting, GPU utilization). PyTorch's autograd, `torch.compile`, Weights & Biases, and the Python ML ecosystem took a decade to build. You will spend months reimplementing 10% of that capability and ship a worse product.

The threshold for adding Rust to an ML system: a Python profiler run has identified a specific bottleneck that cannot be fixed by vectorization, a better algorithm, or a well-maintained Python library. That threshold is higher than most engineers expect.

---

## PyO3 and maturin

PyO3 lets you write Python extension modules in Rust. Maturin is the build system that compiles the Rust crate and packages it as a Python wheel.

### Choosing a Python-binding approach

```
Do you need to call Rust from Python, or Python from Rust?
├─ Python calls Rust (the common case)
│  ├─ Performance-critical hot loop, tight numeric kernel
│  │  └─ PyO3 + maturin (this sheet)
│  ├─ Wrap an existing C / C++ library, no Rust involved
│  │  └─ cffi or ctypes (not a Rust problem; skip this sheet)
│  ├─ One wheel that works on CPython 3.9..3.13 without per-version rebuilds
│  │  └─ PyO3 with the `abi3-pyXY` feature (stable ABI, slight perf cost)
│  └─ Occasional script bridge, not production
│     └─ PyO3 is still fine; maturin develop; don't overthink it
└─ Rust calls Python (uncommon)
   ├─ Embed a Python interpreter inside a Rust binary
   │  └─ PyO3 with the `auto-initialize` feature and `Python::with_gil`
   └─ Run a subprocess and pipe data
      └─ `std::process::Command` + JSON / msgpack (no PyO3 needed)
```

Rule of thumb: **PyO3 + maturin for ~95% of real use cases**. The alternatives
(`cffi`, raw `ctypes`, subprocess) exist but only win when the Rust side is not
actually Rust (wrapping existing C) or when you do not want a compile step.

### Project layout

```
my_extension/
├── pyproject.toml          # Python packaging + maturin backend
├── Cargo.toml              # Rust crate
└── src/
    └── lib.rs              # Module definition
```

### pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "my_extension"
version = "0.1.0"
requires-python = ">=3.9"

[tool.maturin]
features = ["pyo3/extension-module"]
```

### Cargo.toml

```toml
[package]
name = "my_extension"
version = "0.1.0"
edition = "2024"   # edition 2024 needs Rust >= 1.85; use "2021" if on older toolchains

[lib]
# cdylib = shared library that Python can import
# rlib = optional for integration tests in Rust
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
# On PyO3 0.23+, additionally consider enabling abi3 for a single wheel that
# works across Python minor versions:
#   pyo3 = { version = "0.23", features = ["extension-module", "abi3-py39"] }
```

### A complete extension module: functions and classes

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// A simple Python function exposed from Rust.
///
/// Python docstrings work here — they appear in help().
#[pyfunction]
fn sum_as_string(a: i64, b: i64) -> String {
    (a + b).to_string()
}

/// A stateful Python class.
#[pyclass]
struct FeatureHasher {
    buckets: usize,
    state: Vec<f32>,
}

#[pymethods]
impl FeatureHasher {
    /// Python: hasher = FeatureHasher(buckets=1024)
    #[new]
    fn new(buckets: usize) -> PyResult<Self> {
        if buckets == 0 {
            return Err(PyValueError::new_err("buckets must be > 0"));
        }
        Ok(FeatureHasher {
            buckets,
            state: vec![0.0; buckets],
        })
    }

    /// Update internal state. Returns self for chaining.
    fn update(&mut self, features: Vec<String>) -> PyResult<()> {
        for f in features {
            let idx = hash_feature(&f) % self.buckets;
            self.state[idx] += 1.0;
        }
        Ok(())
    }

    /// Export state as a Python list.
    fn as_list(&self) -> Vec<f32> {
        self.state.clone()
    }
}

fn hash_feature(s: &str) -> usize {
    // FNV-1a hash for illustration
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h as usize
}

/// The module initializer — the name must match the crate name.
#[pymodule]
fn my_extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<FeatureHasher>()?;
    Ok(())
}
```

### Development workflow

```bash
# Install maturin into your venv
pip install maturin

# Compile and install into the active venv (fast iteration loop)
maturin develop

# Build a release wheel for distribution
maturin build --release

# Build for a specific Python version
maturin build --release --interpreter python3.11

# Usage from Python:
# import my_extension
# h = my_extension.FeatureHasher(1024)
# h.update(["user:alice", "country:us"])
# print(h.as_list())
```

### GIL handling: releasing the GIL for long computations

**This is the most important correctness point in PyO3 work.** Python's GIL serializes all Python threads. If your Rust function holds the GIL while doing a 500ms computation, you have converted your Python multi-threaded program into a single-threaded one.

```rust
use pyo3::prelude::*;

// ❌ WRONG: holds the GIL for the entire computation
#[pyfunction]
fn slow_compute_wrong(py: Python<'_>, data: Vec<f64>) -> Vec<f64> {
    // GIL is held throughout — blocks other Python threads
    data.iter().map(|x| x.sqrt() * 3.14159).collect()
}

// ✅ CORRECT: release the GIL before the heavy work
#[pyfunction]
fn slow_compute(py: Python<'_>, data: Vec<f64>) -> PyResult<Vec<f64>> {
    // py.allow_threads releases the GIL for the duration of the closure.
    // Other Python threads can run while we compute.
    let result = py.allow_threads(|| -> Vec<f64> {
        data.iter().map(|x| x.sqrt() * 3.14159).collect()
    });
    Ok(result)
}

// ✅ CORRECT: release GIL across a rayon parallel computation
#[pyfunction]
fn parallel_compute(py: Python<'_>, data: Vec<f64>) -> PyResult<Vec<f64>> {
    use rayon::prelude::*;
    let result = py.allow_threads(|| -> Vec<f64> {
        data.par_iter().map(|x| x.sqrt() * 3.14159).collect()
    });
    Ok(result)
}
```

**Rule**: any Rust code that runs longer than ~1ms should release the GIL via `allow_threads`. If you are unsure, release it — there is no correctness cost.

### Error translation

PyO3 converts `Result<T, PyErr>` automatically. For custom error types, implement `From<MyError> for PyErr`:

```rust
use pyo3::exceptions::{PyIOError, PyValueError, PyRuntimeError};
use pyo3::prelude::*;

#[derive(Debug)]
enum MyError {
    Io(std::io::Error),
    InvalidInput(String),
    Internal(String),
}

impl From<MyError> for PyErr {
    fn from(e: MyError) -> Self {
        match e {
            MyError::Io(e) => PyIOError::new_err(e.to_string()),
            MyError::InvalidInput(msg) => PyValueError::new_err(msg),
            MyError::Internal(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

// Now any function returning Result<T, MyError> works transparently
#[pyfunction]
fn load_data(path: &str) -> Result<Vec<f32>, MyError> {
    std::fs::read(path)
        .map_err(MyError::Io)?
        .chunks(4)
        .map(|chunk| {
            chunk.try_into()
                .map(|b: [u8; 4]| f32::from_le_bytes(b))
                .map_err(|_| MyError::InvalidInput("truncated f32".into()))
        })
        .collect()
}
```

### Buffer protocol for numpy interop

Avoid copying data between Python and Rust when possible. The buffer protocol lets Rust access numpy arrays in-place:

```rust
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// For numpy arrays, use the `numpy` crate (pyo3 extension)
// Add to Cargo.toml:
//   numpy = "0.22"

use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
// When you extend the example to mutable arrays, you'll also need:
//   use numpy::PyArrayMethods;   // required for .as_array_mut(), .readwrite(), etc.
// The Bound-API array methods live on this trait in numpy 0.21+.

#[pyfunction]
fn normalize_inplace<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    // SAFETY / SOUNDNESS: `arr.as_slice()?` hands out a borrow into Python-owned
    // numpy memory. That borrow is only valid while the GIL is held — once the GIL
    // is released, another Python thread can resize, free, or overwrite the array.
    // Never pass a slice derived from Python memory into `py.allow_threads`.
    //
    // The safe pattern is: copy out under the GIL -> compute without the GIL ->
    // return. For small arrays the copy is cheap; for large arrays it still beats
    // the UB of a dangling borrow.
    let owned: Vec<f32> = arr.as_slice()?.to_vec();

    let result: Vec<f32> = py.allow_threads(move || {
        let n = owned.len() as f32;
        let mean: f32 = owned.iter().sum::<f32>() / n;
        let var: f32 = owned.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std_dev = var.sqrt();
        owned.iter().map(|x| (x - mean) / (std_dev + 1e-8)).collect()
    });
    // numpy 0.22: `into_pyarray` returns the deprecated `&'py PyArray<...>`;
    // `into_pyarray_bound` returns the `Bound<'py, PyArray<...>>` this signature
    // expects. numpy 0.23+ collapses them — there `into_pyarray` itself returns
    // `Bound` and `into_pyarray_bound` is removed.
    Ok(result.into_pyarray_bound(py))
}
```

---

## candle

Candle is Hugging Face's pure-Rust ML framework. It targets inference, not training. The design philosophy: minimal dependencies, no Python runtime required, supports CPU/CUDA/Metal, native safetensors support.

Add to `Cargo.toml`:

```toml
[dependencies]
candle-core = "0.8"            # 0.8.x is the Dec-2024 line; check crates.io for latest 0.x
candle-nn = "0.8"
candle-transformers = "0.8"    # optional: pre-built transformer blocks
safetensors = "0.4"
tokenizers = "0.20"            # optional: Hugging Face tokenizer (0.21 available as of early 2025)
```

Candle is pre-1.0 and every 0.x bump is allowed to be a breaking change — pin
an exact minor (`0.8`, not `>=0.8`) in production builds and upgrade
intentionally. The three candle crates (`candle-core`, `candle-nn`,
`candle-transformers`) must move together; mixing versions breaks the `VarBuilder`
and `Module` traits.

### Tensor basics and device abstraction

```rust
use candle_core::{Device, DType, Tensor};

fn tensor_basics() -> candle_core::Result<()> {
    // Device selection: CPU, CUDA (by ordinal), or Metal
    let device = Device::Cpu;
    // let device = Device::new_cuda(0)?;    // first CUDA GPU
    // let device = Device::new_metal(0)?;   // Metal (Apple Silicon)

    // Create tensors
    let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
    let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device)?;

    // Element-wise operations
    let c = (&a + &b)?;
    let d = (&a * &b)?;
    let e = a.log()?;

    // Reshape and transpose
    let mat = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
    let transposed = mat.t()?;

    // Matrix multiply
    let result = mat.matmul(&transposed)?;

    // Extract values back to CPU/host
    let values: Vec<f32> = result.flatten_all()?.to_vec1()?;
    println!("{values:?}");

    Ok(())
}
```

### Model loading from safetensors

```rust
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use safetensors::SafeTensors;

// A simple linear classifier in candle
struct LinearClassifier {
    fc1: Linear,
    fc2: Linear,
}

impl LinearClassifier {
    fn load(weights_path: &str, device: &Device) -> candle_core::Result<Self> {
        // Load weights from safetensors (safe, fast, language-portable).
        // `VarBuilder::from_mmaped_safetensors` mmaps the file internally — no need
        // to construct a separate `MmapedSafetensors` alongside it. The inner mmap
        // is still `unsafe` because another process could mutate the file while
        // candle has it mapped; the constructor is marked unsafe to surface that.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                candle_core::DType::F32,
                device,
            )?
        };

        Ok(Self {
            fc1: candle_nn::linear(128, 64, vb.pp("fc1"))?,
            fc2: candle_nn::linear(64, 10, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.relu()?;
        self.fc2.forward(&x)
    }
}

fn run_inference(model_path: &str, input: Vec<f32>) -> candle_core::Result<Vec<f32>> {
    let device = Device::Cpu;
    let model = LinearClassifier::load(model_path, &device)?;

    // Input shape: [batch=1, features=128]
    let n = input.len();
    let x = Tensor::from_vec(input, (1, n), &device)?;

    let logits = model.forward(&x)?;

    // Softmax for probabilities
    let probs = candle_nn::ops::softmax(&logits, 1)?;
    Ok(probs.flatten_all()?.to_vec1()?)
}
```

### Embedding model inference example

A common production use case: generate embeddings from a sentence transformer.

```rust
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

fn embed_text(
    tokenizer: &Tokenizer,
    vb: VarBuilder,
    device: &Device,
    text: &str,
) -> candle_core::Result<Vec<f32>> {
    // Tokenize
    let encoding = tokenizer.encode(text, true)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    let n = input_ids.len();

    // Build input tensor [1, seq_len]. `from_vec(Vec<u32>, ...)` already produces
    // a U32 tensor, so no `to_dtype` conversion is needed here.
    let input = Tensor::from_vec(input_ids, (1, n), device)?;

    // Run embedding lookup — this assumes a model with a pre-built embedding layer
    // For production, use candle_transformers::models::bert::BertModel
    // or similar pre-built transformer blocks.
    let embed_layer = candle_nn::embedding(30522, 768, vb.pp("embeddings.word_embeddings"))?;
    let hidden = embed_layer.forward(&input)?;

    // Mean pool over sequence dimension
    let pooled = hidden.mean(1)?;     // [1, 768]
    let flat = pooled.flatten_all()?; // [768]

    Ok(flat.to_vec1()?)
}
```

### When to choose candle

- Pure Rust: no Python runtime, single static binary (modulo CUDA libs).
- Fast cold start: no JIT warm-up.
- Ideal for: inference microservices, edge binaries, CLI inference tools.
- Limitations: smaller ecosystem than PyTorch; fewer pre-built model architectures; no autograd / training.

---

## burn

Burn is an alternative ML framework for Rust with a focus on backend abstraction and training support (unlike candle which is inference-focused).

```toml
[dependencies]
burn = { version = "0.16", features = ["wgpu"] }
# Available backends:
#   "wgpu"    — cross-platform GPU via WebGPU (works on macOS, Linux, Windows,
#              and browsers via wasm — this is burn's unique selling point)
#   "candle"  — use candle as burn's backend
#   "ndarray" — CPU-only, no GPU, pure Rust
#   "tch"     — libtorch backend (see tch-rs section)
#   "cuda"    — direct CUDA backend (burn 0.15+, separate from the tch path)
```

Burn is moving faster than candle and has had breaking API changes roughly every
minor release (0.13 → 0.14 → 0.15 → 0.16 all broke something). Pin the exact
minor, read the release notes before upgrading, and expect `Module` derives and
`Backend` bounds to churn. The code below targets 0.16; earlier versions use
slightly different `init` / `LinearConfig` signatures.

### Backend abstraction

The core design: write your model once against the `Backend` trait; swap backends without changing model code.

```rust
use burn::prelude::*;
// burn::prelude re-exports `Tensor`, `Backend`, `Module`, `nn`, but NOT `activation`.
// Import it explicitly for `activation::relu`, `activation::sigmoid`, etc.
use burn::tensor::activation;

// Define a model generically over the backend
#[derive(Module, Debug)]
struct MyModel<B: Backend> {
    fc1: nn::Linear<B>,
    dropout: nn::Dropout,
    fc2: nn::Linear<B>,
}

impl<B: Backend> MyModel<B> {
    fn new(device: &B::Device) -> Self {
        let fc1 = nn::LinearConfig::new(128, 64).init(device);
        let dropout = nn::DropoutConfig::new(0.3).init();
        let fc2 = nn::LinearConfig::new(64, 10).init(device);
        Self { fc1, dropout, fc2 }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = activation::relu(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}

// Run on WGPU (GPU):
fn run_wgpu() {
    use burn::backend::Wgpu;
    type MyBackend = Wgpu;

    let device = Default::default();
    let model: MyModel<MyBackend> = MyModel::new(&device);
    // ... forward pass
}

// Run on NdArray (CPU, no GPU deps):
fn run_ndarray() {
    use burn::backend::NdArray;
    type MyBackend = NdArray<f32>;

    let device = Default::default();
    let model: MyModel<MyBackend> = MyModel::new(&device);
}
```

### When to choose burn over candle

| Criterion | candle | burn |
|-----------|--------|------|
| Inference only | Excellent | Good |
| Training support | Minimal | Full (autodiff, optimizers) |
| Backend portability | CPU/CUDA/Metal | CPU/CUDA/Metal/WebGPU/Vulkan |
| WebGPU / browser | No | Yes (via wgpu) |
| Ecosystem maturity | More pre-built models | Growing quickly |
| Binary size | Smaller | Larger |
| Single-binary inference | Yes | Yes |

Choose **candle** when: you are doing inference only, want minimal dependencies, or need strong Hugging Face model compatibility.

Choose **burn** when: you need training in Rust, want WebGPU support, or need Vulkan/DirectX12 backends.

---

## tch-rs

`tch-rs` provides direct bindings to libtorch (the C++ backend of PyTorch). It gives you full access to the PyTorch tensor API from Rust, including autograd, but comes with significant build complexity.

```toml
[dependencies]
# Gate behind a feature flag — most users should not need tch-rs
tch = { version = "0.16", optional = true }

[features]
libtorch = ["tch"]
```

### Build requirements and complexity

```bash
# libtorch must be installed separately — it is NOT bundled by tch-rs
# Download from https://pytorch.org/get-started/locally/

# Set required environment variables before cargo build:
export LIBTORCH=/opt/libtorch               # path to extracted libtorch
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# CUDA version must match libtorch's CUDA expectation exactly.
# libtorch 2.3 + CUDA 12.1 will not work with CUDA 12.4.
# This is the main pain point.
cargo build --features libtorch
```

### Basic tch-rs usage

```rust
#[cfg(feature = "libtorch")]
mod tch_example {
    use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

    fn create_and_run() {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);

        // Build a network using the PyTorch C++ API
        let net = nn::seq()
            .add(nn::linear(vs.root() / "fc1", 128, 64, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs.root() / "fc2", 64, 10, Default::default()));

        // Inference
        let input = Tensor::randn([1, 128], (Kind::Float, device));
        let output = net.forward(&input);
        println!("{output}");

        // Load a saved PyTorch model (TorchScript)
        let model = tch::CModule::load("model.pt").expect("load failed");
        let result = model.forward_ts(&[input]).expect("forward failed");
    }
}
```

### When tch-rs is the right choice

- You need to load and run TorchScript (`model.pt`) files directly — tch-rs is the cleanest path.
- You need autograd in Rust (rare, but exists).
- Your organization has libtorch already installed and version-pinned in CI.
- You need exact numerical parity with a PyTorch model for validation.

### When tch-rs is the wrong choice

If you are writing a new inference service and do not have existing TorchScript models, start with candle or burn. The libtorch installation, version pinning, and LD_LIBRARY_PATH management add significant operational overhead. Feature-flag the dependency as shown above so users who do not need it are not forced to install libtorch.

---

## ndarray and nalgebra

Rust's two main numerical array libraries serve different use cases and should not be mixed in the same module without a clear reason.

### ndarray: N-dimensional arrays (numpy analogue)

```toml
[dependencies]
ndarray = { version = "0.16", features = ["rayon"] }
ndarray-rand = "0.15"    # optional: random initialization
blas-src = { version = "0.10", features = ["openblas"] }  # optional: BLAS backend
# Note: ndarray-linalg lags ndarray releases — check crates.io for the version that
# matches your `ndarray` version before uncommenting this line. The 0.16.x series
# tracks ndarray 0.15; an ndarray 0.16-compatible release may still be on git only.
# ndarray-linalg = { version = "0.17", features = ["openblas"] }  # optional: linalg
```

```rust
use ndarray::{Array1, Array2, Axis, s};
use ndarray::parallel::prelude::*;

fn ndarray_basics() {
    // Create arrays
    let v: Array1<f32> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let m: Array2<f32> = Array2::zeros((4, 4));

    // Slicing (like numpy)
    let row_0 = m.slice(s![0, ..]);      // first row
    let submat = m.slice(s![1..3, 1..3]); // 2x2 submatrix

    // Elementwise operations via Zip (avoids temporary allocations)
    let a = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
    let b = Array1::from_vec(vec![4.0f32, 5.0, 6.0]);
    let mut c = Array1::zeros(3);

    ndarray::Zip::from(&mut c)
        .and(&a)
        .and(&b)
        .for_each(|c_i, &a_i, &b_i| *c_i = a_i * b_i + 0.5);

    // Parallel map (requires rayon feature)
    // Parallel element-wise map via the `rayon` feature of ndarray.
    // Note: there is no `par_map` on `Array1`; use `Zip::par_map_collect`.
    use ndarray::Zip;
    let d: Array1<f32> = Zip::from(&a).par_map_collect(|&x| x.sqrt());

    // Matrix multiply (uses BLAS if configured)
    let p: Array2<f32> = Array2::eye(3);
    let q: Array2<f32> = Array2::ones((3, 3));
    let r = p.dot(&q);

    // Axis operations
    let m2 = Array2::from_shape_vec((3, 4), (0..12).map(|x| x as f32).collect()).unwrap();
    let col_means = m2.mean_axis(Axis(0)).unwrap(); // mean over rows → shape [4]
    let row_sums = m2.sum_axis(Axis(1));             // sum over cols → shape [3]
}
```

### nalgebra: fixed-size linear algebra (typed dimensions)

nalgebra's type system encodes matrix dimensions at compile time. A `Matrix3<f64>` is a 3×3 matrix; multiplying a `Matrix3x4<f64>` by a `Vector4<f64>` produces a `Vector3<f64>` — and the compiler rejects dimension mismatches.

```toml
[dependencies]
nalgebra = "0.33"
```

```rust
use nalgebra::{Matrix3, Matrix3x4, Vector3, Vector4, SMatrix, SVector};

fn nalgebra_basics() {
    // Fixed-size: dimensions are type-level constants
    let rotation: Matrix3<f64> = Matrix3::identity();
    let translation: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);

    // Dimension mismatch is a COMPILE ERROR — not a runtime panic
    let transform: Matrix3x4<f64> = Matrix3x4::zeros();
    let point: Vector4<f64> = Vector4::new(1.0, 0.0, 0.0, 1.0);
    let result: Vector3<f64> = transform * point;  // type-checks

    // Common operations
    let m = Matrix3::new(
        1.0f64, 2.0, 3.0,
        4.0,    5.0, 6.0,
        7.0,    8.0, 9.0,
    );
    let det = m.determinant();
    let trace = m.trace();
    let transposed = m.transpose();

    // Decompositions (LU, QR, SVD, Cholesky)
    let decomp = m.lu();
    let (l, u, _perm) = decomp.unpack();

    // Quaternions for 3D rotation
    use nalgebra::{UnitQuaternion, Vector3 as Vec3};
    let axis = Vec3::new(0.0f64, 0.0, 1.0);
    let angle = std::f64::consts::FRAC_PI_4; // 45 degrees
    let rotation = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle);
}
```

### When to use each

| Situation | Use |
|-----------|-----|
| Dynamic shapes (batch size unknown at compile time) | `ndarray` |
| numpy interop / data pipelines | `ndarray` |
| Large matrices (> a few hundred elements) | `ndarray` |
| Fixed-size geometry (3D transforms, quaternions) | `nalgebra` |
| Physics simulation (rigid bodies, joints) | `nalgebra` |
| Compile-time dimension checking is valuable | `nalgebra` |
| BLAS/LAPACK integration needed | `ndarray` + `ndarray-linalg` |

**Do not mix them in the same data path.** Converting between `nalgebra` matrices and `ndarray` arrays requires a copy (or careful unsafe reinterpretation). If your pipeline starts with `ndarray` tensors and needs nalgebra for a rotation, that conversion cost may dominate. Pick one and stay consistent within a module boundary.

---

## Serialization for ML

### serde: JSON and binary formats

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
# bincode 2.x gates its serde bridge behind the "serde" feature; without it
# `bincode::serde::encode_to_vec` is not in scope. The default features DO NOT
# include serde — you must enable it explicitly.
bincode = { version = "2", features = ["serde"] }
```

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct ModelConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    activation: String,
    version: String,
}

fn serialize_config(config: &ModelConfig) -> anyhow::Result<()> {
    // JSON (human-readable, portable)
    let json = serde_json::to_string_pretty(config)?;
    std::fs::write("config.json", &json)?;

    // Bincode (compact binary, ~5–10× smaller than JSON, fast)
    // Good for internal caches or IPC; NOT good for cross-language exchange.
    // bincode 2.x requires either the serde bridge (as used here, with
    // `bincode = { version = "2", features = ["serde"] }`) when the type derives
    // `serde::Serialize`/`Deserialize`, or `#[derive(bincode::Encode, Decode)]`
    // on the type itself (no serde needed in that case).
    let bytes = bincode::serde::encode_to_vec(config, bincode::config::standard())?;
    std::fs::write("config.bin", &bytes)?;

    Ok(())
}
```

### safetensors: the right format for model weights

Safetensors is the standard for sharing model weights. Key properties:
- Safe: no arbitrary code execution (unlike pickle).
- Fast: memory-mapped; loading is O(1) regardless of file size.
- Language-portable: Python, Rust, JavaScript, C++ all have libraries.
- Self-describing: stores dtype, shape, and name for each tensor.

```toml
[dependencies]
safetensors = "0.4"
memmap2 = "0.9"
bytemuck = "1"   # for the f32 -> &[u8] reinterpretation in the write path
```

```rust
use safetensors::{SafeTensors, tensor::TensorView};
use memmap2::MmapOptions;
use std::fs::File;

fn load_weights(path: &str) -> anyhow::Result<()> {
    // Memory-map the file — no copy into RAM for the full file
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // Parse the header (metadata only — tensors are still mmap'd)
    let tensors = SafeTensors::deserialize(&mmap)?;

    // Access individual tensors
    for (name, view) in tensors.tensors() {
        let shape = view.shape();
        let dtype = view.dtype();
        println!("{name}: {dtype:?} {shape:?}");
    }

    // Get a specific tensor's data
    let embed = tensors.tensor("embeddings.weight")?;
    let data: &[u8] = embed.data(); // raw bytes; cast based on dtype

    Ok(())
}

fn save_weights(weights: &[(&str, Vec<f32>, Vec<usize>)]) -> anyhow::Result<()> {
    use safetensors::serialize_to_file;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;
    use std::collections::HashMap;

    // safetensors has no `Tensor::new` — the public construction API is
    // `TensorView::new(dtype, shape: Vec<usize>, data: &[u8])`. Data must be raw
    // bytes, so reinterpret the f32 backing store as a `&[u8]` slice.
    let views: HashMap<String, TensorView<'_>> = weights
        .iter()
        .map(|(name, data, shape)| {
            let bytes: &[u8] = bytemuck::cast_slice(data.as_slice());
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .expect("shape/data length mismatch");
            (name.to_string(), view)
        })
        .collect();

    // serialize_to_file takes the data by value (move), not by reference.
    serialize_to_file(views, &None, std::path::Path::new("weights.safetensors"))?;
    Ok(())
}
```

### prost: protobuf schemas for ML pipelines

When you need versioned, schema-enforced wire format for model inputs/outputs (e.g., a gRPC serving endpoint or a Kafka pipeline):

```toml
[build-dependencies]
prost-build = "0.13"

[dependencies]
prost = "0.13"
```

```proto
// proto/inference.proto
syntax = "proto3";

message InferenceRequest {
  string model_version = 1;
  repeated float features = 2;
  map<string, string> metadata = 3;
}

message InferenceResponse {
  repeated float logits = 1;
  string predicted_class = 2;
  float confidence = 3;
}
```

```rust
// build.rs
fn main() {
    prost_build::compile_protos(&["proto/inference.proto"], &["proto/"]).unwrap();
}
```

Protobuf is worth the setup cost when: the serving boundary crosses teams, services, or languages; or when you need schema evolution guarantees (adding fields without breaking existing clients).

---

## Performance Notes

### Parallelism with rayon

Rayon is the standard for data-parallel computation in Rust. It integrates cleanly with ndarray and handles work-stealing automatically.

```rust
use rayon::prelude::*;
use ndarray::{Array1, Array2};

// ✅ CORRECT: parallel map over a large dataset
fn process_batch(inputs: &[Vec<f32>]) -> Vec<f32> {
    inputs
        .par_iter()
        .map(|row| row.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect()
}

// ✅ CORRECT: parallel ndarray operation with Zip
fn elementwise_parallel(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
    use ndarray::parallel::prelude::*;
    let mut result = Array1::zeros(a.len());
    ndarray::Zip::from(&mut result)
        .and(a)
        .and(b)
        .par_for_each(|r, &ai, &bi| *r = ai.mul_add(bi, 0.5));
    result
}
```

### Avoiding allocations in tight numerical loops

Each `Vec` allocation in a hot loop is a heap round-trip. In numerical code this compounds quickly.

```rust
// ❌ WRONG: allocates a Vec on every iteration
fn score_batch_wrong(inputs: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    inputs.iter().map(|row| {
        let weighted: Vec<f32> = row.iter().zip(weights).map(|(x, w)| x * w).collect();
        weighted.iter().sum::<f32>()
    }).collect()
}

// ✅ CORRECT: compute without intermediate allocation
fn score_batch(inputs: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    inputs.iter().map(|row| {
        row.iter().zip(weights).map(|(x, w)| x * w).sum::<f32>()
    }).collect()
}

// ✅ CORRECT: reuse a scratch buffer across calls
struct Scorer {
    scratch: Vec<f32>,
}

impl Scorer {
    fn score(&mut self, row: &[f32], weights: &[f32]) -> f32 {
        self.scratch.clear();
        self.scratch.extend(row.iter().zip(weights).map(|(x, w)| x * w));
        self.scratch.iter().sum()
    }
}
```

### Zip for elementwise operations

`ndarray::Zip` is the idiomatic way to write elementwise kernels. It avoids temporary arrays and enables auto-vectorization.

```rust
use ndarray::{Array1, Zip};

fn fused_kernel(
    out: &mut Array1<f32>,
    a: &Array1<f32>,
    b: &Array1<f32>,
    scale: f32,
) {
    // ✅ CORRECT: single pass, no temporaries, LLVM can vectorize
    Zip::from(out).and(a).and(b).for_each(|o, &ai, &bi| {
        *o = (ai + bi) * scale;
    });
}
```

### SIMD

For most numerical Rust code, write clean scalar code and let LLVM auto-vectorize. Check with `RUSTFLAGS="-C target-cpu=native" cargo rustc --release -- --emit=asm` or via godbolt.org whether vectorization is happening. Only hand-write SIMD when:

1. The function is a proven bottleneck via Criterion.
2. Auto-vectorization is confirmed to not be happening.
3. A hand-written version benchmarks measurably faster.

```toml
# For stable SIMD without nightly, use the wide crate
[dependencies]
wide = "0.7"
```

```rust
// ✅ CORRECT: let LLVM vectorize first; verify with godbolt
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// Only after profiling confirms the above is the bottleneck
// and godbolt shows it is NOT being vectorized:
use wide::f32x8;

fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    let chunks = a.len() / 8;
    let mut acc = f32x8::splat(0.0);
    for i in 0..chunks {
        let va = f32x8::from(&a[i * 8..][..8]);
        let vb = f32x8::from(&b[i * 8..][..8]);
        acc += va * vb;
    }
    let simd_sum: f32 = acc.to_array().iter().sum();
    // Handle tail
    let tail_start = chunks * 8;
    let tail: f32 = a[tail_start..].iter().zip(&b[tail_start..]).map(|(x, y)| x * y).sum();
    simd_sum + tail
}
```

---

## Anti-Patterns

### 1. Rewriting Python training pipelines in Rust

**Why wrong:** Training in PyTorch benefits from autograd, `torch.compile`, dynamic shapes, mixed precision, gradient checkpointing, and an ecosystem of optimizers, schedulers, and monitoring tools that took a decade to build. A Rust training pipeline replicates none of that for free. Research velocity — the ability to change the architecture on Tuesday, add a new loss term on Wednesday, and run an ablation on Thursday — depends on the Python ecosystem. Rust's compile cycle and static typing work against this velocity.

**The fix:** Train in Python. Export to safetensors. Serve in Rust. This is the proven architecture: PyTorch/HuggingFace for training, Rust (candle, burn, or an ONNX runtime) for inference.

---

### 2. Holding the GIL across long computations in PyO3

**Why wrong:** Python's Global Interpreter Lock (GIL) prevents multiple Python threads from running concurrently. If your Rust extension holds the GIL for a 200ms compute operation, every other Python thread blocks for 200ms. In an async Python server (asyncio, FastAPI), this can stall the entire event loop. This is the most common correctness bug in PyO3 extensions — it silently destroys concurrency.

**The fix:** Call `py.allow_threads(|| { ... })` around any Rust code that takes more than ~1ms and does not need to call back into Python.

```rust
// ❌ WRONG: GIL held throughout
#[pyfunction]
fn process(py: Python<'_>, data: Vec<f32>) -> Vec<f32> {
    data.iter().map(|x| x.sqrt()).collect()
}

// ✅ CORRECT
#[pyfunction]
fn process(py: Python<'_>, data: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(py.allow_threads(|| data.iter().map(|x| x.sqrt()).collect()))
}
```

---

### 3. Using tch-rs when candle would suffice

**Why wrong:** `tch-rs` requires a manual libtorch installation, exact CUDA version alignment, and `LD_LIBRARY_PATH` configuration. This breaks in CI unless you install libtorch there too. It ties your Rust binary to a specific libtorch ABI. The build complexity is significant — a single CUDA version mismatch produces cryptic link errors.

**The fix:** For inference from safetensors weights, candle is self-contained (libtorch not required) and covers most use cases. Only reach for `tch-rs` when you need TorchScript (`.pt`) model loading, exact PyTorch numerical parity for validation, or autograd in Rust. Gate the dependency behind a feature flag so the default build does not require libtorch.

```toml
# ✅ CORRECT: optional, feature-gated
[dependencies]
tch = { version = "0.16", optional = true }

[features]
libtorch-backend = ["tch"]
```

---

### 4. Mixing ndarray and nalgebra in the same data path

**Why wrong:** `ndarray` and `nalgebra` are not interchangeable. Converting between them requires a data copy (or unsafe reinterpretation). In a numerical pipeline, an unexpected conversion at the boundary of two modules can dominate runtime. More subtly: if half your code uses `ndarray::Array2` and half uses `nalgebra::DMatrix`, neither half can use the other's optimized operations, BLAS integration, or rayon parallel iterators directly.

**The fix:** Choose one per codebase area. Use `ndarray` for data pipeline work (dynamic shapes, numpy interop, large batches). Use `nalgebra` for geometry and physics (fixed-size, compile-time dimension checking). Define the boundary explicitly and do any conversion once at the edge.

---

### 5. Shipping ML binaries without safetensors versioning

**Why wrong:** Pickle-based PyTorch weights (`.pt`, `.pth`) execute arbitrary Python code on load — this is a Remote Code Execution (RCE) vector. Even if you trust your own weights, a supply chain compromise of a model hub file can silently compromise every machine that loads it. Beyond security: without version metadata in your weight files, you cannot detect when a binary is running with stale weights after a model update. Silent model drift produces wrong predictions with no error signal.

**The fix:** Use safetensors for all production weight files. Store model version and config hash alongside weights (either in the safetensors metadata dict or in a companion `config.json`). At binary startup, verify the version matches what the code expects.

```rust
use safetensors::SafeTensors;

fn load_and_verify(path: &str, expected_version: &str) -> anyhow::Result<()> {
    let file = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&file)?;

    // safetensors supports a metadata dict for exactly this purpose
    // Store version in Python: save_file(tensors, path, metadata={"version": "1.2.0"})
    // (The safetensors Rust crate exposes metadata via the header)
    // If version metadata is absent, treat as unversioned and warn.

    Ok(())
}
```

---

### 6. Neglecting to release ndarray temporaries in loops

**Why wrong:** Expressions like `a + b` in ndarray return an owned `Array` — a heap allocation. Chained operations (`(a + b) * c + d`) produce N-1 temporary arrays. In a batch processing loop, this creates allocation pressure that the profiler will flag as "allocator time" rather than "compute time," making it hard to diagnose.

**The fix:** Use `Zip` for multi-operand elementwise kernels. Use `ScaledAdd` or in-place operations (`+=`, `*=`) when accumulating.

```rust
use ndarray::{Array1, Zip};

// ❌ WRONG: 2 temporary arrays per call
fn fused_wrong(a: &Array1<f32>, b: &Array1<f32>, c: &Array1<f32>) -> Array1<f32> {
    (a + b) * c   // alloc for (a+b), then alloc for result
}

// ✅ CORRECT: single output allocation, no intermediates.
// Zip::from needs something that *produces elements* (`&mut out`), not the
// dimension descriptor. `Zip::from(out.raw_dim())` would iterate shape axes, not
// array elements. Use `&mut out` to walk the `MaybeUninit<f32>` slots, and call
// `.for_each` (the `.apply` name was removed in ndarray 0.15+).
fn fused_correct(a: &Array1<f32>, b: &Array1<f32>, c: &Array1<f32>) -> Array1<f32> {
    let mut out = Array1::<f32>::uninit(a.raw_dim());
    Zip::from(&mut out)
        .and(a).and(b).and(c)
        .for_each(|out_i, &ai, &bi, &ci| {
            out_i.write((ai + bi) * ci);
        });
    // SAFETY: the Zip above writes every element exactly once.
    unsafe { out.assume_init() }
}

// Simpler alternative for this case:
fn fused_simple(a: &Array1<f32>, b: &Array1<f32>, c: &Array1<f32>) -> Array1<f32> {
    Zip::from(a).and(b).and(c).map_collect(|&ai, &bi, &ci| (ai + bi) * ci)
}
```

---

## Checklist

Before shipping a Rust component that touches ML or Python interop:

- [ ] Confirmed the bottleneck is Python (profiled with cProfile or py-spy) before porting to Rust.
- [ ] Training pipeline stays in Python; Rust handles only inference, preprocessing, or hot-loop kernels.
- [ ] All PyO3 functions that run longer than ~1ms call `py.allow_threads`.
- [ ] Custom error types implement `From<MyError> for PyErr` so Python sees meaningful exception types.
- [ ] numpy interop uses the buffer protocol or the `numpy` crate — no manual pointer arithmetic.
- [ ] Model weights are in safetensors format; no pickle (`.pt`/`.pth`) in production paths.
- [ ] Weight files include version metadata that the binary verifies at startup.
- [ ] `tch-rs` is behind a feature flag if present; default build does not require libtorch.
- [ ] ndarray and nalgebra are not mixed in the same data path without a documented conversion boundary.
- [ ] ndarray hot loops use `Zip` or in-place operations; no chained `+`/`*` that produce temporaries.
- [ ] rayon parallelism is enabled via `ndarray`'s `rayon` feature; `par_for_each` used for batch work.
- [ ] SIMD is only hand-written after Criterion confirms the scalar version is a bottleneck.
- [ ] `maturin build --release` produces a wheel that installs cleanly in a fresh venv.
- [ ] GIL is released before any rayon parallel section in a PyO3 function.
- [ ] safetensors loading uses memory-mapping (`MmapOptions`) for large weight files.

---

## Related Skills

- [modern-rust-and-editions.md](modern-rust-and-editions.md) — edition and feature compatibility; what stable 1.87 provides
- [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — lifetime annotations in PyO3 `'py` lifetimes; GIL token semantics
- [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — burn's `Backend` trait pattern; generic numerical code
- [error-handling-patterns.md](error-handling-patterns.md) — `From<MyError> for PyErr`; error translation at FFI boundaries
- [project-structure-and-tooling.md](project-structure-and-tooling.md) — feature flags for optional libtorch; workspace layout for mixed Python/Rust projects
- [testing-and-quality.md](testing-and-quality.md) — testing numerical code with `approx`; roundtrip tests for serialization
- [systematic-delinting.md](systematic-delinting.md) — clippy lints for ndarray allocation patterns; PyO3 correctness lints
- [async-and-concurrency.md](async-and-concurrency.md) — serving inference endpoints with Axum; back-pressure in batch inference pipelines
- [performance-and-profiling.md](performance-and-profiling.md) — Criterion for numerical benchmarks; flamegraph for hot-path identification
- [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — buffer protocol and raw pointer interop; mmap safety; manual SIMD intrinsics
