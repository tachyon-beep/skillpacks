
# Attention Mechanisms Catalog

## When to Use This Skill

Use this skill when you need to:
- ✅ Select attention mechanism for long sequences (> 2k tokens)
- ✅ Optimize memory usage (GPU OOM errors)
- ✅ Speed up training or inference
- ✅ Understand exact vs approximate attention trade-offs
- ✅ Choose between Flash, sparse, or linear attention
- ✅ Implement cross-attention for multimodal models

**Do NOT use this skill for:**
- ❌ Basic Transformer understanding (use `transformer-architecture-deepdive`)
- ❌ High-level architecture selection (use `using-neural-architectures`)
- ❌ LLM-specific optimization (use `llm-specialist/llm-inference-optimization`)


## Core Principle

**Not all attention is O(n²).** Standard self-attention has quadratic complexity, but modern variants achieve:
- **O(n²) with less memory**: Flash Attention (exact, 4x less memory)
- **O(n × w)**: Sparse attention (exact, sliding window)
- **O(n)**: Linear attention (approximate, 1-3% accuracy loss)

**Default recommendation:** Flash Attention (exact + fast + memory-efficient)


## Part 1: Complexity Hierarchy

### Standard Self-Attention (Baseline)

**Formula:**
```python
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

**Complexity:**
- Time: O(n² · d) where n = seq_len, d = d_model
- Memory: O(n²) for attention matrix
- Exact: Yes (no approximation)

**Memory breakdown (4k tokens, d=768), materializing the score matrix:**
```
Attention scores: 4096² × 4 bytes = 64MB per HEAD
Multi-head (12 heads): 64MB × 12 = 768MB per layer
16 layers: 768MB × 16 = 12GB just for attention!
Batch size 8: 12GB × 8 = 96GB (impossible on single GPU)
```
This is exactly the cost FlashAttention eliminates — it never materializes
the n² matrix, so in practice you do not pay this. The table above is the
*reason* Flash exists, not the memory profile of a modern model.

**When to use:**
- Sequence length < 2k tokens
- Standard use case (most models)
- Pair with Flash Attention optimization

**Limitations:**
- Memory explosion for long sequences
- Quadratic scaling impractical beyond 4k tokens


## Part 2: Flash Attention ⭐ (Modern Default)

### What is Flash Attention?

**Breakthrough (2022):** Exact attention with 4x less memory, 2-3x faster

**Key insight:**
- Standard attention is **memory-bound** (not compute-bound)
- GPUs: Fast compute (TFLOPS), slow memory bandwidth (GB/s)
- Bottleneck: Moving n² attention matrix to/from HBM

**Solution:**
- Tile attention computation
- Recompute instead of store intermediate values
- Fuse operations (reduce memory transfers)
- Result: Same O(n²) compute, O(n) memory

### Algorithm

```
Standard attention (3 memory operations):
1. Compute scores: S = Q K^T (store n² matrix)
2. Softmax: P = softmax(S) (store n² matrix)
3. Output: O = P V (store n×d matrix)

Flash Attention (tiled):
1. Divide Q, K, V into blocks
2. For each Q block:
   - Load block to SRAM (fast memory)
   - For each K, V block:
     - Compute attention for this tile
     - Update output incrementally
   - Never materialize full n² matrix!
3. Result: Same output, O(n) memory
```

### Performance

**Benchmarks (A100 GPU, 2k tokens):**

Standard attention:
- Memory: 4GB for batch_size=8
- Speed: 150ms/batch
- Max batch size: 16

Flash Attention:
- Memory: 1GB for batch_size=8 **(4x reduction)**
- Speed: 75ms/batch **(2x faster)**
- Max batch size: 64 **(4x larger)**

**Flash Attention 2 (2023 update):**
- Further optimized: 2-3x faster than Flash Attention 1
- Better parallelism
- Supports more head dimensions

### When to Use

✅ **ALWAYS use Flash Attention when:**
- You are running attention on a CUDA GPU — at **any** sequence length
- Need exact attention (no approximation)
- Available in your framework

**It's a FREE LUNCH:**
- No accuracy loss (mathematically exact)
- Faster training AND inference
- Less memory usage
- Drop-in replacement

### Implementation

**PyTorch 2.0+ (built-in):**
```python
import torch.nn.functional as F

# Automatic Flash Attention (if available)
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False
)
# PyTorch automatically uses Flash Attention if:
# - CUDA available
# - Sequence length suitable
# - No attention mask (or causal mask)
```

**HuggingFace Transformers:**
```python
from transformers import AutoModel

# Enable Flash Attention 2
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    attn_implementation="flash_attention_2",  # Requires flash-attn package
    torch_dtype=torch.float16
)
```

**Manual installation:**
```bash
pip install flash-attn --no-build-isolation
```

### Limitations

❌ **Flash Attention NOT suitable when:**
- Inference on CPU (CUDA-only; PyTorch falls back to a math kernel)
- Very exotic attention patterns that no kernel implements (rare — see
  FlexAttention below)

**What is NOT a limitation:** long sequences. FlashAttention memory is
**O(n), not O(n²)** — it never materializes the score matrix (see the O(n)
memory claim in the algorithm section above). *Compute* remains O(n²), so
long sequences get slower, but they do not blow up memory. Production models
run **exact** FlashAttention at 128k+ context. Do not switch to sparse or
linear attention merely because the sequence exceeds some length.

**Newer variants worth knowing:**
- **FlashAttention-3 (2024):** rewritten for Hopper (H100) — asynchrony,
  warp specialization, FP8 support. Large speedup over FA-2 on H100-class
  hardware.
- **FlexAttention (PyTorch 2.5+, 2024):** compiles an arbitrary
  `score_mod` / `mask_mod` function into a fused Flash-style kernel. This
  **removes the "custom masks unsupported" limitation** — sliding window,
  ALiBi, document masking, prefix-LM and causal variants all get
  Flash-quality kernels without hand-writing CUDA. Reach for this before
  reaching for an approximate attention mechanism.


## Part 3: Sparse Attention (Exact for Long Sequences) — *largely legacy*

> **Status note (2026):** Longformer and BigBird were designed for a world in
> which 4k tokens of exact attention did not fit in memory. FlashAttention
> removed that constraint. Today they are **legacy** for new builds: exact
> Flash + RoPE context scaling serves 128k+ context, and where you genuinely
> want a sparse *pattern* (sliding window, document masking), FlexAttention
> compiles it into a fused exact kernel without switching model families.
>
> Read this section to understand deployed models and the sliding-window idea
> (still live in Mistral-class models and in FlexAttention `mask_mod`s), not
> as a default recommendation.

### Concept

**Idea:** Each token attends to subset of tokens (not all)
- Sliding window: Local context
- Global tokens: Long-range connections
- Result: O(n × window_size) instead of O(n²)

**Key property:** Still EXACT attention (not approximate)
- Just more structured attention pattern
- No accuracy loss if pattern matches task

### Variant 1: Longformer

**Pattern:** Sliding window + global attention

```
Attention pattern (window=2, global=[0]):
    0  1  2  3  4  5
0 [ 1  1  1  1  1  1 ]  ← Global token (attends to all)
1 [ 1  1  1  0  0  0 ]  ← Window: tokens 0-2
2 [ 1  1  1  1  0  0 ]  ← Window: tokens 1-3
3 [ 1  0  1  1  1  0 ]  ← Window: tokens 2-4
4 [ 1  0  0  1  1  1 ]  ← Window: tokens 3-5
5 [ 1  0  0  0  1  1 ]  ← Window: tokens 4-5

Complexity: O(n × (window + num_global))
```

**Components:**
1. **Sliding window**: Each token attends to w/2 tokens before and after
2. **Global tokens**: Special tokens (like [CLS]) attend to all tokens
3. **Dilated windows**: Optional (stride > 1 for longer context)

**Implementation:**
```python
from transformers import LongformerModel

model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

# Attention mask (shape: batch, seq_len)
attention_mask = torch.ones(batch_size, seq_len)
attention_mask[:, 0] = 2  # 2 = global attention for [CLS] token

output = model(input_ids, attention_mask=attention_mask)
```

**Memory comparison (4k tokens, window=512):**
```
Standard: 4096² = 16M elements → 64MB
Longformer: 4096 × 512 = 2M elements → 8MB (8x reduction!)
```

**When to use:**
- Documents: 4k-16k tokens (legal, scientific papers)
- Need full context but can't fit O(n²)
- Task has local + global structure

**Pretrained models:**
- `allenai/longformer-base-4096`: Max 4096 tokens
- `allenai/longformer-large-4096`: Larger version

### Variant 2: BigBird

**Pattern:** Random + window + global

```
Attention pattern:
- Sliding window: Like Longformer
- Random connections: Each token attends to r random tokens
- Global tokens: Special tokens attend to all

Complexity: O(n × (window + r + num_global))
```

**Key difference from Longformer:**
- Random connections help information flow
- Theoretically proven to approximate full attention

**When to use:**
- Similar to Longformer
- Slightly better for tasks needing long-range
- Less widely adopted than Longformer

**Implementation:**
```python
from transformers import BigBirdModel

model = BigBirdModel.from_pretrained(
    "google/bigbird-roberta-base",
    attention_type="block_sparse"  # or "original_full"
)
```

### Sparse Attention Decision

```
Any sequence length, new build:
→ Exact FlashAttention first. It is O(n) memory; length alone is not a
  reason to leave it.

You want a specific sparsity PATTERN (sliding window, doc masking, prefix-LM):
→ FlexAttention `mask_mod` — keeps exactness AND the fused kernel.
→ Or a model natively trained with sliding-window attention (Mistral-class).

You are maintaining a deployed Longformer / BigBird checkpoint:
→ This section explains it. Don't port the pattern to a new model.
```


## Part 4: Linear Attention (Approximate for Very Long) — *largely legacy*

> **Status note (2026):** Performer, Linformer and the linear-attention family
> lost. They trade exactness for an asymptotic win that FlashAttention made
> unnecessary at the lengths people actually run, and they consistently
> underperformed on retrieval-style long-context tasks. Where sub-quadratic
> sequence modeling did succeed, it was **state-space / hybrid** models
> (Mamba-2, Jamba), not softmax approximation — see
> [sequence-models-comparison.md](sequence-models-comparison.md).
>
> Know these as background and for reading older papers. Do not pick one for
> a new build without a measured reason.

### Concept

**Idea:** Approximate softmax attention with linear operations
- Complexity: O(n × k) where k << n
- Trade-off: real accuracy loss, worst on long-range retrieval
- Benefit: sub-quadratic *compute* at extreme lengths

**Key property:** APPROXIMATE (not exact)
- Do NOT use if accuracy critical
- Only consider when exact Flash is genuinely compute-bound at your length
  and you have measured the quality cost

### Variant 1: Performer

**Method:** Random Fourier Features to approximate softmax(Q K^T)

**Formula:**
```python
# Standard attention
Attention(Q, K, V) = softmax(Q K^T) V

# Performer approximation: a random feature map φ(·) such that
φ(Q) φ(K)^T ≈ softmax(Q K^T)
# Associativity then lets you avoid the n² product entirely:
Attention(Q, K, V) ≈ φ(Q) (φ(K)^T V)

# Complexity: O(n × k) where k = feature dimension
```

**Key trick:**
- Compute φ(K)^T V first: (k × d) matrix (small!)
- Then multiply by φ(Q): O(n × k × d) instead of O(n² × d)
- Never materialize n² attention matrix

**Implementation:**
```python
# From performer-pytorch library
from performer_pytorch import Performer

model = Performer(
    dim=512,
    depth=6,
    heads=8,
    dim_head=64,
    causal=False,
    nb_features=256  # k = number of random features
)
```

**Accuracy:**
- Typical loss: 1-2% vs standard attention
- Depends on nb_features (more features = better approximation)
- k=256 usually sufficient

**When to use:**
- Sequence length > 16k tokens
- Accuracy loss acceptable (not critical task)
- Need better than sparse attention (no structure assumptions)

### Variant 2: Linformer

**Method:** Project K and V to lower dimension

**Formula:**
```python
# Standard attention (n × n attention matrix)
Attention(Q, K, V) = softmax(Q K^T / √d_k) V

# Linformer (project K, V to n × k where k << n)
K_proj = E K  # E: (k × n) projection matrix
V_proj = F V  # F: (k × n) projection matrix

Attention(Q, K, V) ≈ softmax(Q K_proj^T / √d_k) V_proj
# Attention matrix: (n × k) instead of (n × n)
```

**Complexity:**
- Time: O(n × k × d) where k << n
- Memory: O(n × k) instead of O(n²)

**Implementation:**
```python
# From linformer library
from linformer import Linformer

model = Linformer(
    dim=512,
    seq_len=8192,
    depth=12,
    heads=8,
    k=256  # Projected dimension
)
```

**Accuracy:**
- Typical loss: 1-3% vs standard attention
- More loss than Performer
- Fixed sequence length (k is tied to max_seq_len)

**When to use:**
- Fixed-length long sequences
- Memory more critical than speed
- Accuracy loss OK (2-3%)

### Linear Attention Decision

```
Need exact attention (almost always):
→ FlashAttention at any length; FlexAttention if you need a custom pattern

Long context (16k-128k+), accuracy critical:
→ Still exact FlashAttention + RoPE scaling. This is what production
  long-context models do. Linear attention is NOT the answer here.

Streaming / constant-memory decoding, or genuinely compute-bound at extreme
length:
→ SSM or hybrid (Mamba-2, Jamba) — see sequence-models-comparison.md

Performer / Linformer:
→ Legacy. Only with a measured accuracy budget and a measured speedup.
```


## Part 5: Cross-Attention (Multimodal)

### Concept

**Self-attention:** Q, K, V from same source
**Cross-attention:** Q from one source, K/V from another

**Use cases:**
- Multimodal: vision → language (image captioning)
- Seq2seq: source language → target language (translation)
- RAG: query → document retrieval
- Conditioning: generation conditioned on context

### Architecture

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(self, query_source, key_value_source, mask=None):
        # query_source: (batch, n_q, d_model) - e.g., text tokens
        # key_value_source: (batch, n_kv, d_model) - e.g., image patches

        # Q from query source
        Q = self.W_q(query_source)

        # K, V from key-value source
        K = self.W_k(key_value_source)
        V = self.W_v(key_value_source)

        # Attention: (batch, n_q, d_model)
        output = attention(Q, K, V, mask)
        return output
```

### Example: Image Captioning

**Task:** Generate caption from image

**Architecture:**
1. **Image Encoder:** ViT processes image → image features (n_patches × d)
2. **Text Decoder:** Autoregressive text generation
3. **Cross-Attention:** Text queries image features

```python
class ImageCaptioningDecoder(nn.Module):
    def forward(self, text_tokens, image_features):
        # 1. Self-attention on text (causal)
        text = self.text_self_attention(
            query=text,
            key=text,
            value=text,
            causal_mask=True  # Don't see future words
        )

        # 2. Cross-attention (text queries image)
        text = self.cross_attention(
            query=text,               # From text decoder
            key=image_features,       # From image encoder
            value=image_features      # From image encoder
            # No causal mask! Can attend to all image patches
        )

        # 3. Feed-forward
        text = self.feed_forward(text)

        return text
```

**Attention flow:**
- Text token "cat" → High attention to cat region in image
- Text token "sitting" → High attention to posture in image

### Example: Retrieval-Augmented Generation (RAG)

**Task:** Generate answer using retrieved documents

```python
class RAGDecoder(nn.Module):
    def forward(self, query_tokens, document_embeddings):
        # 1. Self-attention on query
        query = self.query_self_attention(query, query, query)

        # 2. Cross-attention (query → documents)
        query = self.cross_attention(
            query=query,                    # What we're generating
            key=document_embeddings,        # Retrieved docs
            value=document_embeddings       # Retrieved docs
        )

        # Query learns to extract relevant info from docs

        return query
```

### When to Use Cross-Attention

✅ **Use cross-attention when:**
- Two different modalities (vision + language)
- Conditioning generation on context (RAG)
- Seq2seq with different input/output (translation)
- Query-document matching

❌ **Don't use cross-attention when:**
- Same modality (use self-attention)
- No clear query vs key-value separation


## Part 6: Other Attention Variants

### Axial Attention (2D Images)

**Idea:** For 2D data (images), attend along each axis separately

```
Standard 2D attention: H×W tokens → (HW)² attention matrix
Axial attention:
  - Row attention: Each row attends to itself (H × W²)
  - Column attention: Each column attends to itself (W × H²)
  - Total: O(HW × (H + W)) << O((HW)²)
```

**When to use:**
- High-resolution images
- 2D positional structure important

### Block-Sparse Attention

**Idea:** Divide attention into blocks, attend only within/across blocks

**Pattern:**
```
Block size = 64 tokens
- Local block: Attend within same block
- Vertical stripe: Attend to corresponding position in other blocks
```

**Used in:** Sparse Transformer (Child et al., OpenAI, **2019**), GPT-3

### Multi-Query Attention (MQA)

**Idea:** One K/V head shared across all Q heads

**Benefit:**
- Smaller KV cache during inference
- Much faster decoding (4-8x)
- Trade-off: ~1% accuracy loss

**Used in:** PaLM, Falcon

### Grouped-Query Attention (GQA)

**Idea:** Middle ground between multi-head and multi-query
- Group Q heads share K/V heads
- Example: 32 Q heads → 8 K/V heads (4:1 ratio)

**Benefit:**
- 4x smaller KV cache
- Minimal accuracy loss (< 0.5%)

**Used in:** LLaMA-2, Mistral


## Part 7: Decision Framework

### By Sequence Length

**The headline: exact attention scales further than most people think.**
Sequence length is not by itself a reason to leave exact FlashAttention.

```
Any length, up to 128k+:
→ Exact FlashAttention (FA-2, or FA-3 on H100-class hardware)
   + RoPE context scaling (YaRN / NTK-aware) if extending a pretrained model
   This is what production long-context models actually do.
   Memory is O(n); only COMPUTE is quadratic.

Need a specific attention PATTERN (sliding window, doc masking, prefix-LM):
→ FlexAttention `mask_mod` / `score_mod`
   Still exact, still a fused kernel, no model-family change.

Compute (not memory) is the measured bottleneck at extreme length:
→ Sliding-window attention (Mistral-class), or
→ State-space / hybrid models (Mamba-2, Jamba) for streaming and
   constant-memory decoding — see sequence-models-comparison.md
→ Linear attention (Performer/Linformer) only with a measured quality budget

Maintaining a deployed Longformer / BigBird:
→ Parts 3-4 explain them. Legacy for new builds.
```

### By Memory Constraints

```
GPU OOM with standard attention:
1. Use Flash Attention (removes the n² score matrix entirely — free lunch)
2. Reduce batch size / use gradient accumulation
3. Use AMP (bf16) — roughly halves activation memory
4. Gradient checkpointing — genuinely useful; it trades compute for
   activation memory and composes WITH Flash Attention
5. Shard the model (FSDP / ZeRO) if the weights, not the activations, dominate

NOTE: at long context, decode-time memory is usually the KV CACHE, not the
attention computation. Fix that with GQA/MQA, MLA, or KV-cache quantization —
not by swapping in an approximate attention.
```

### By Accuracy Requirements

```
Must be exact (no approximation):
→ FlashAttention (any length), or FlexAttention if you need a custom pattern
   Never use linear attention!

Accuracy loss acceptable:
→ Linear Attention (Performer, Linformer) — legacy; measure before adopting
   The modern sub-quadratic answer is an SSM/hybrid, not softmax approximation

Critical task (medical, legal):
→ Exact attention only — FlashAttention / FlexAttention
```

### By Task Type

```
Classification / Understanding:
→ Standard + Flash Attention
   Sequence usually < 2k

Document processing:
→ A long-context Transformer with exact FlashAttention (32k-128k+)
   Longformer/BigBird only if you already run one

Generation (LLM):
→ Flash Attention for training
→ + GQA/MQA for inference (faster decoding)

Multimodal (vision + language):
→ Cross-attention for modality fusion
→ Self-attention within each modality

Retrieval-augmented:
→ Cross-attention (query → documents)
```


## Part 8: Implementation Checklist

### Using Flash Attention

**PyTorch 2.0+:**
```python
# Automatic (recommended)
output = F.scaled_dot_product_attention(query, key, value)

# Verify Flash Attention is used
import torch.backends.cuda
print(torch.backends.cuda.flash_sdp_enabled())  # Should be True
```

**HuggingFace:**
```python
model = AutoModel.from_pretrained(
    "model-name",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16  # Flash Attention needs fp16/bf16
)
```

**Requirements:**
- CUDA GPU (not CPU)
- PyTorch >= 2.0 OR flash-attn package
- fp16 or bf16 dtype (not fp32)

### Using Sparse Attention

**Longformer:**
```python
from transformers import LongformerModel, LongformerTokenizer

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

# Attention mask
# 0 = no attention, 1 = local attention, 2 = global attention
attention_mask = torch.ones(batch_size, seq_len)
attention_mask[:, 0] = 2  # [CLS] token gets global attention

outputs = model(input_ids, attention_mask=attention_mask)
```

**Custom sparse pattern:**
```python
# Create custom block-sparse mask
def create_block_sparse_mask(seq_len, block_size):
    num_blocks = seq_len // block_size
    mask = torch.zeros(seq_len, seq_len)

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        mask[start:end, start:end] = 1  # Local block

    return mask
```

### Using Cross-Attention

```python
class DecoderWithCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

    def forward(self, decoder_input, encoder_output, causal_mask=None):
        # Self-attention (causal)
        x = self.self_attn(
            query=decoder_input,
            key=decoder_input,
            value=decoder_input,
            mask=causal_mask
        )

        # Cross-attention (Q from decoder, K/V from encoder)
        x = self.cross_attn(
            query=x,                  # From decoder
            key=encoder_output,       # From encoder
            value=encoder_output,     # From encoder
            mask=None                 # No causal mask for cross-attention!
        )

        return x
```


## Part 9: Common Mistakes

### Mistake 1: Ignoring Flash Attention

**Symptom:** Training slow, high memory usage
**Fix:** Use Flash Attention at every sequence length, not just short ones

### Mistake 2: Abandoning Exact Attention Because the Sequence Is Long

**Symptom:** Reaching for Longformer/Performer at 16k-128k and eating an
accuracy loss for nothing
**Fix:** FlashAttention memory is O(n). Exact attention serves 128k+ in
production. Use FlexAttention if you need a custom mask; only consider
approximation with a measured quality budget.

### Mistake 3: Assuming Custom Masks Rule Out Flash

**Symptom:** Falling back to a naive O(n²) implementation for a sliding
window, document mask, or prefix-LM mask
**Fix:** FlexAttention (PyTorch 2.5+) compiles arbitrary `mask_mod` /
`score_mod` into a fused Flash-style kernel

### Mistake 4: Cross-Attention with Causal Mask

**Symptom:** Decoder can't attend to encoder properly
**Fix:** Causal mask only for self-attention, NOT cross-attention

### Mistake 5: Accepting O(n²) Memory

**Symptom:** GPU OOM for > 4k tokens
**Fix:** Use Flash Attention, don't just add GPUs

### Mistake 6: Blaming Attention for a KV-Cache Problem

**Symptom:** Long-context *inference* OOMs even with Flash Attention
**Fix:** At decode time the KV cache dominates, not the attention kernel.
Use GQA/MQA, MLA, or KV-cache quantization — swapping attention mechanisms
will not help


## Summary: Quick Reference

### Attention Selection

```
Sequence length:
  Any length → exact FlashAttention (FA-2; FA-3 on H100-class)
  Extending a pretrained model → + RoPE scaling (YaRN / NTK-aware)
  Custom mask/bias needed → FlexAttention (still exact, still fused)
  Legacy only → Longformer / BigBird / Performer / Linformer

Memory constrained:
  First: Flash Attention (removes the n² score matrix)
  Then: bf16/AMP, smaller batch + grad accumulation, gradient checkpointing
  Long-context inference: fix the KV cache (GQA/MQA/MLA, cache quantization)
  Weights dominate: FSDP / ZeRO sharding

Speed critical:
  Training: Flash Attention (2x faster)
  Inference: Flash Attention + GQA/MQA

Accuracy critical:
  Use exact attention only (Flash or Sparse)
  NEVER linear attention

Multimodal:
  Cross-attention for modality fusion
```

### Implementation

```
PyTorch 2.0+:
  F.scaled_dot_product_attention() # Auto Flash Attention

HuggingFace:
  attn_implementation="flash_attention_2"

Longformer:
  LongformerModel.from_pretrained("allenai/longformer-base-4096")

Custom:
  Inherit from nn.Module, implement forward()
```


## Next Steps

After mastering this skill:
- `llm-specialist/llm-inference-optimization`: Apply attention optimizations to inference
- `llm-specialist/context-window-management`: Manage long contexts in LLMs
- `architecture-design-principles`: Understand broader design trade-offs

**Remember:** Exact FlashAttention is the modern default at *every* sequence
length — its memory is O(n), so length alone is never the reason to leave it.
If you need a custom mask or bias, use FlexAttention rather than an
approximate mechanism. Sparse and linear attention are legacy for new builds.
