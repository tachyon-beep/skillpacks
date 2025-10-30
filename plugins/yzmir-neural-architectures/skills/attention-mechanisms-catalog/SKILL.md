---
name: attention-mechanisms-catalog
description: Comprehensive catalog of attention mechanisms from standard self-attention to modern variants (Flash, sparse, linear). Use when selecting attention for sequence length, memory constraints, or speed requirements. Covers exact vs approximate attention, complexity analysis, and practical trade-offs.
dependencies:
  - transformer-architecture-deepdive
  - using-neural-architectures
related:
  - sequence-models-comparison
  - llm-specialist/llm-inference-optimization
---

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

---

## Core Principle

**Not all attention is O(n²).** Standard self-attention has quadratic complexity, but modern variants achieve:
- **O(n²) with less memory**: Flash Attention (exact, 4x less memory)
- **O(n × w)**: Sparse attention (exact, sliding window)
- **O(n)**: Linear attention (approximate, 1-3% accuracy loss)

**Default recommendation:** Flash Attention (exact + fast + memory-efficient)

---

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

**Memory breakdown (4k tokens, d=768):**
```
Attention scores: 4096² × 4 bytes = 64MB per layer
Multi-head (12 heads): 64MB × 12 = 768MB per layer
16 layers: 768MB × 16 = 12GB just for attention!
Batch size 8: 12GB × 8 = 96GB (impossible on single GPU)
```

**When to use:**
- Sequence length < 2k tokens
- Standard use case (most models)
- Pair with Flash Attention optimization

**Limitations:**
- Memory explosion for long sequences
- Quadratic scaling impractical beyond 4k tokens

---

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
- Sequence length < 16k tokens
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
- Sequence length > 16k (memory still grows quadratically)
- Custom attention masks (complex patterns not supported)
- Inference on CPU (CUDA-only)

**For > 16k tokens:** Use sparse or linear attention

---

## Part 3: Sparse Attention (Exact for Long Sequences)

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
Sequence length < 4k:
→ Flash Attention (exact, no pattern needed)

Sequence length 4k-16k:
→ Longformer (sliding window + global)
→ Best for: Documents, long-form text

Sequence length > 16k:
→ Longformer if possible
→ Linear attention if Longformer too slow
```

---

## Part 4: Linear Attention (Approximate for Very Long)

### Concept

**Idea:** Approximate softmax attention with linear operations
- Complexity: O(n × k) where k << n
- Trade-off: 1-3% accuracy loss
- Benefit: Can handle very long sequences (> 16k)

**Key property:** APPROXIMATE (not exact)
- Do NOT use if accuracy critical
- Good for extremely long sequences where exact is impossible

### Variant 1: Performer

**Method:** Random Fourier Features to approximate softmax(Q K^T)

**Formula:**
```python
# Standard attention
Attention(Q, K, V) = softmax(Q K^T) V

# Performer approximation
φ(Q) ≈ φ(K)^T ≈ softmax(Q K^T)
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
Need exact attention:
→ Flash Attention or Sparse Attention (NOT linear)

Sequence > 16k, accuracy critical:
→ Sparse Attention (Longformer)

Sequence > 16k, accuracy loss OK:
→ Performer (better) or Linformer

Sequence > 100k:
→ State space models (S4, Mamba, not attention)
```

---

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

---

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

**Used in:** Sparse Transformer (OpenAI), GPT-3

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

---

## Part 7: Decision Framework

### By Sequence Length

```
< 2k tokens:
→ Flash Attention
   Exact, fast, standard

2k-4k tokens:
→ Flash Attention
   Still manageable with modern GPUs

4k-16k tokens:
→ Sparse Attention (Longformer, BigBird)
   Exact, designed for documents
→ OR Flash Attention if batch size = 1

> 16k tokens:
→ Sparse Attention
   If task has local structure
→ Linear Attention (Performer)
   If accuracy loss OK (1-2%)
→ State Space Models (S4, Mamba)
   If sequence > 100k
```

### By Memory Constraints

```
GPU OOM with standard attention:
1. Try Flash Attention (4x less memory, free lunch)
2. If still OOM, reduce batch size
3. If batch size = 1 and still OOM, use sparse attention
4. Last resort: Linear attention (if accuracy loss OK)

DON'T:
- Gradient checkpointing (slower, use Flash Attention instead)
- Throwing more GPUs (algorithmic problem, not hardware)
```

### By Accuracy Requirements

```
Must be exact (no approximation):
→ Flash Attention or Sparse Attention
   Never use linear attention!

Accuracy loss acceptable (1-3%):
→ Linear Attention (Performer, Linformer)
   Only for very long sequences (> 16k)

Critical task (medical, legal):
→ Exact attention only
   Flash Attention or Sparse Attention
```

### By Task Type

```
Classification / Understanding:
→ Standard + Flash Attention
   Sequence usually < 2k

Document processing:
→ Longformer (4096 tokens)
   Designed for documents

Generation (LLM):
→ Flash Attention for training
→ + GQA/MQA for inference (faster decoding)

Multimodal (vision + language):
→ Cross-attention for modality fusion
→ Self-attention within each modality

Retrieval-augmented:
→ Cross-attention (query → documents)
```

---

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

---

## Part 9: Common Mistakes

### Mistake 1: Ignoring Flash Attention

**Symptom:** Training slow, high memory usage
**Fix:** Always use Flash Attention for < 16k tokens

### Mistake 2: Using Linear Attention Unnecessarily

**Symptom:** 1-3% accuracy loss for no reason
**Fix:** Use Flash Attention (exact) unless sequence > 16k

### Mistake 3: Gradient Checkpointing Instead of Flash Attention

**Symptom:** Training 20% slower
**Fix:** Flash Attention gives memory savings AND speed

### Mistake 4: Cross-Attention with Causal Mask

**Symptom:** Decoder can't attend to encoder properly
**Fix:** Causal mask only for self-attention, NOT cross-attention

### Mistake 5: Accepting O(n²) Memory

**Symptom:** GPU OOM for > 4k tokens
**Fix:** Use sparse or Flash Attention, don't just add GPUs

---

## Summary: Quick Reference

### Attention Selection

```
Sequence length:
  < 2k → Flash Attention (default)
  2-4k → Flash Attention
  4-16k → Longformer (documents) or Flash Attention (batch=1)
  > 16k → Sparse or Linear Attention

Memory constrained:
  First: Try Flash Attention (4x less memory)
  Still OOM: Use sparse attention (Longformer)
  Last resort: Linear attention (accuracy loss)

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

---

## Next Steps

After mastering this skill:
- `llm-specialist/llm-inference-optimization`: Apply attention optimizations to inference
- `llm-specialist/context-window-management`: Manage long contexts in LLMs
- `architecture-design-principles`: Understand broader design trade-offs

**Remember:** Flash Attention is the modern default. Use it unless you have a specific reason not to (> 16k tokens, custom patterns).
