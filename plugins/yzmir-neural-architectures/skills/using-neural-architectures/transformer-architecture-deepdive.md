
# Transformer Architecture Deep Dive

## When to Use This Skill

Use this skill when you need to:
- ✅ Implement a Transformer from scratch
- ✅ Understand HOW and WHY self-attention works
- ✅ Choose between encoder, decoder, or encoder-decoder architectures
- ✅ Decide if Vision Transformer (ViT) is appropriate for your vision task
- ✅ Understand modern variants (RoPE, ALiBi, GQA, MQA)
- ✅ Debug Transformer implementation issues
- ✅ Optimize Transformer performance

**Do NOT use this skill for:**
- ❌ High-level architecture selection (use `using-neural-architectures`)
- ❌ Attention mechanism comparison (use `attention-mechanisms-catalog`)
- ❌ LLM-specific topics like prompt engineering (use `llm-specialist` pack)


## Core Principle

**Transformers are NOT magic.** They are:
1. Self-attention mechanism (information retrieval)
2. + Position encoding (break permutation invariance)
3. + Residual connections + Layer norm (training stability)
4. + Feed-forward networks (non-linearity)

Understanding the mechanism beats cargo-culting implementations.


## Part 1: Self-Attention Mechanism Explained

### The Information Retrieval Analogy

**Self-attention = Querying a database:**
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I have?"

**Process:**
1. Compare your query with all keys (compute similarity)
2. Weight values by similarity
3. Return weighted sum of values

**Example:** Sentence: "The cat sat on the mat"
Token "sat" (verb):
- High attention to "cat" (subject) → Learns verb-subject relationship
- High attention to "mat" (object) → Learns verb-object relationship
- Low attention to "the", "on" (function words)

### Mathematical Breakdown

**Given input X:** (batch, seq_len, d_model)

**Step 1: Project to Q, K, V**
```python
Q = X @ W_Q  # (batch, seq_len, d_k)
K = X @ W_K  # (batch, seq_len, d_k)
V = X @ W_V  # (batch, seq_len, d_v)

# Typically: d_k = d_v = d_model / num_heads
```

**Step 2: Compute attention scores** (similarity)
```python
scores = Q @ K.transpose(-2, -1)  # (batch, seq_len, seq_len)
# scores[i, j] = similarity between query_i and key_j
```

**Geometric interpretation:**
- Dot product measures vector alignment
- q · k = ||q|| ||k|| cos(θ)
- Similar vectors → Large dot product → High attention
- Orthogonal vectors → Zero dot product → No attention

**Step 3: Scale by √d_k** (CRITICAL!)
```python
scores = scores / math.sqrt(d_k)
```

**WHY scaling?**
- Dot products grow with dimension: Var(q · k) = d_k
- Example: d_k=64 → Random dot products ~ ±64
- Large scores → Softmax saturates → Gradients vanish
- Scaling: Keep scores ~ O(1) regardless of dimension

**Without scaling:** Softmax([30, 25, 20]) ≈ [0.99, 0.01, 0.00] (saturated!)
**With scaling:** Softmax([3, 2.5, 2]) ≈ [0.50, 0.30, 0.20] (healthy gradients)

**Step 4: Softmax to get attention weights**
```python
attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
# Each row sums to 1 (probability distribution)
# attn_weights[i, j] = "how much token i attends to token j"
```

**Step 5: Weight values**
```python
output = attn_weights @ V  # (batch, seq_len, d_v)
# Each token's output = weighted average of all values
```

**Complete formula:**
```python
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

### Why Three Matrices (Q, K, V)?

**Could we use just one?** Attention(X, X, X)
**Yes, but Q/K/V separation enables:**

1. **Asymmetry**: Query can differ from key (search ≠ database)
2. **Decoupling**: What you search for (Q@K) ≠ what you retrieve (V)
3. **Cross-attention**: Q from one source, K/V from another
   - Example: Decoder queries encoder (translation)

**Modern optimization:** Multi-Query Attention (MQA), Grouped-Query Attention (GQA)
- Share K/V across heads (fewer parameters, faster inference)

### Implementation Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None):
        super().__init__()
        self.d_k = d_k or d_model

        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_k)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, seq_len, seq_len)

        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # Weighted sum of values
        output = torch.matmul(attn_weights, V)  # (batch, seq_len, d_k)

        return output, attn_weights
```

**Complexity:** O(n² · d) where n = seq_len, d = d_model
- **Quadratic in sequence length** (bottleneck for long sequences)
- For n=1000, d=512: 1000² × 512 = 512M operations


## Part 2: Multi-Head Attention

### Why Multiple Heads?

**Single-head attention** learns one attention pattern.
**Multi-head attention** learns multiple parallel patterns:
- Head 1: Syntactic relationships (subject-verb)
- Head 2: Semantic similarity
- Head 3: Positional proximity
- Head 4: Long-range dependencies

**Analogy:** Ensemble of attention functions, each specializing in different patterns.

### Head Dimension Calculation

**CRITICAL CONSTRAINT:** num_heads must divide d_model evenly!

```python
d_model = 512
num_heads = 8
d_k = d_model // num_heads  # 512 / 8 = 64

# Each head operates in d_k dimensions
# Concatenate all heads → back to d_model dimensions
```

**Common configurations:**
- BERT-base: d_model=768, heads=12, d_k=64
- GPT-2: d_model=768, heads=12, d_k=64
- GPT-3 175B: d_model=12288, heads=96, d_k=128
- LLaMA-2 70B: d_model=8192, heads=64, d_k=128

**Rule of thumb:** d_k (head dimension) should be 64-128
- Too small (d_k < 32): Limited representational capacity
- Too large (d_k > 256): Redundant, wasteful

### Multi-Head Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Single linear layers for all heads (more efficient)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # Output projection

    def split_heads(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        # Reshape to (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Linear projections
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (batch, num_heads, seq_len, d_k)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (batch, seq_len, num_heads, d_k)
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        # (batch, seq_len, d_model)

        # Final linear projection
        output = self.W_o(attn_output)

        return output, attn_weights
```

### Modern Variants: GQA and MQA

**Problem:** K/V caching during inference is memory-intensive
- LLaMA-2 70B: 8192 × 64 heads × 2 (K + V) = 1M parameters per token cached!

**Solution 1: Multi-Query Attention (MQA)**
- **One** K/V head shared across **all** Q heads
- Benefit: Dramatically faster inference (smaller KV cache)
- Trade-off: ~1-2% accuracy loss

```python
# MQA: Single K/V projection
self.W_k = nn.Linear(d_model, d_k)  # Not d_model!
self.W_v = nn.Linear(d_model, d_k)
self.W_q = nn.Linear(d_model, d_model)  # Multiple Q heads
```

**Solution 2: Grouped-Query Attention (GQA)**
- Middle ground: Group multiple Q heads per K/V head
- Example: 32 Q heads → 8 K/V heads (4 Q per K/V)
- Benefit: 4x smaller KV cache, minimal accuracy loss

**Used in:** LLaMA-2/3, Mistral, Mixtral, Gemma, Qwen2, DeepSeek-V2/V3.

### Multi-head Latent Attention (MLA)

**Paper:** DeepSeek-AI — *DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model* (2024).

**Idea:** Project K and V into a low-rank "latent" before caching, instead of
sharing K/V heads (GQA/MQA). Result: KV cache shrinks ~10× vs MHA, often
beats GQA on quality at the same cache size.

**Used in:** DeepSeek-V2, DeepSeek-V3, DeepSeek-R1.

**When to consider:** Long-context decoder-only models where KV cache is the
bottleneck and you're training from scratch (MLA is harder to retrofit).


## Part 2.5: Mixture-of-Experts (MoE) Transformers

A decade-old idea that became the dominant frontier-model pattern after Mixtral
(late 2023): replace the dense feed-forward block with a *sparse* one — many
"expert" FFNs plus a router that activates only a few per token.

**Why it matters:** A 47B-parameter MoE that activates 13B per token (Mixtral
8×7B) often matches or beats a 70B dense model on quality, at ~13B FLOPs per
token. Total parameters scale; compute per token does not.

### Origin and key papers

| Paper | Contribution |
|-------|-------------|
| Shazeer et al. 2017 — *Outrageously Large Neural Networks* | Sparsely-gated MoE in LSTMs; introduced top-k routing + load-balancing loss |
| GShard (Lepikhin et al. 2020) | Token-level expert routing for Transformers, expert parallelism |
| Switch Transformer (Fedus et al. 2021) | Top-1 routing simplification, capacity factor, scaling laws for sparse models |
| ST-MoE (Zoph et al. 2022) | Stability tricks (router z-loss), better fine-tuning |
| Mixtral 8×7B (Jiang et al. 2024) | First widely-used open-weights production MoE; 8 experts, top-2 routing |
| DeepSeek-MoE (Dai et al. 2024) | Fine-grained experts (many small experts) + shared experts; better expert specialization |
| OLMoE (Muennighoff et al. 2024) | Fully-open MoE recipe (data, code, weights) |

### How MoE replaces the FFN

A standard Transformer block:

```
y = x + MHA(LN(x))
y = y + FFN(LN(y))      # FFN(x) = W2 · GeLU(W1 · x)
```

An MoE block:

```
y = x + MHA(LN(x))
y = y + MoE(LN(y))
where MoE(x) = sum_{e in TopK(router(x))} g_e(x) · FFN_e(x)
```

- `router(x)` is typically a single linear layer producing logits over experts
- `TopK` keeps the top-1 (Switch) or top-2 (Mixtral) experts
- `g_e(x)` are router-softmax weights, used to combine selected experts

### Routing strategies

| Strategy | When activated | Notes |
|----------|---------------|-------|
| **Top-1 (Switch)** | Single expert per token | Simplest, fastest, slightly lower quality |
| **Top-2 (Mixtral, GShard)** | Two experts per token | Standard production choice |
| **Expert Choice (Zhou et al. 2022)** | Each expert picks top-N tokens (inverted) | Balances load by construction; harder to mask for autoregressive training |
| **Fine-grained experts (DeepSeek-MoE)** | Many small experts, plus a "shared" expert active for every token | Better expert specialization; total expert count explodes |
| **Hash routing (HASH layers)** | Random fixed routing | Used as a baseline; surprisingly strong |

### Load balancing — the central problem

If routing is unconstrained, a few experts collapse and absorb all tokens (the
classic mode-collapse failure). Standard mitigations:

- **Auxiliary load-balancing loss** (Shazeer 2017): penalize variance in
  per-expert token counts.
- **Router z-loss** (ST-MoE): penalize large router logits to stabilize
  training.
- **Capacity factor**: cap how many tokens an expert can accept; overflow
  tokens skip the MoE layer (residual passes through).
- **Auxiliary-loss-free balancing** (DeepSeek-V3): per-expert bias updated
  online to equalize load without an explicit loss term — currently the
  state-of-the-art recipe for very-large MoEs.

### Expert parallelism

MoE forces a new parallelism dimension. With E experts, you typically:

- Shard experts across devices (each device holds a subset of experts)
- All-to-all the activations: tokens routed to expert e are sent to that
  device, processed, sent back
- Combine with tensor / pipeline / data parallelism (4D-parallel training)

The all-to-all is the bandwidth bottleneck. Modern MoE training relies on fast
interconnects (NVLink, NVSwitch) and overlap-friendly kernels (MegaBlocks,
DeepEP).

### When MoE pays off

| Condition | Pay off? |
|-----------|----------|
| Very large total budget (>30B total params) | ✅ Strong gains over dense at equal active params |
| Inference is memory-bandwidth-bound, not compute-bound | ⚠️ Memory cost is total params, not active — you still need to hold all experts in VRAM |
| Single-GPU on-device deployment | ❌ Total VRAM cost dominates; dense is usually better |
| Training data << model size | ❌ Sparsity hurts when data is the bottleneck |
| Long-tail / multi-domain data | ✅ Experts can specialize |

**Rule of thumb (2026):** Frontier models are MoE (Mixtral, DeepSeek-V3,
Qwen2-MoE, GPT-4-class internal models). Most production fine-tunes and
on-device models are still dense — MoE adds VRAM cost, deployment complexity,
and serving constraints that aren't worth it below the very-large scale.

### Mixtral, DeepSeek-MoE, OLMoE — what's different

- **Mixtral 8×7B (2024).** 8 experts, top-2 routing, 47B total / 13B active.
  Standard reference design. Open weights.
- **DeepSeek-MoE / V2 / V3.** Fine-grained experts (V3: 256 routed + 1 shared
  per layer, 8 active), MLA attention, auxiliary-loss-free load balancing.
  V3 is 671B total / 37B active. Currently the strongest open-weights MoE.
- **OLMoE 7B (2024).** 64 experts, top-8 routing. Fully open including data
  and training code; the cleanest reference for studying MoE recipes.

### What NOT to do

- **Don't train a tiny MoE.** Below ~3–5B total parameters, the routing
  overhead and expert imbalance erase the sparsity gains. Use a dense
  Transformer.
- **Don't ignore load balancing.** Skipping the aux loss/z-loss/bias scheme
  produces dead experts and broken capacity utilization.
- **Don't assume MoE is "just a faster Transformer at inference."** Active
  FLOPs are lower; total VRAM is higher. Plan around the cache.

### Why Position Encoding?

**Problem:** Self-attention is **permutation-invariant**
- Attention("cat sat mat") = Attention("mat cat sat")
- No inherent notion of position or order!

**Solution:** Add position information to embeddings

### Strategy 1: Sinusoidal Position Encoding (Original)

**Formula:**
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Implementation:**
```python
def sinusoidal_position_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Usage: Add to input embeddings
x = token_embeddings + positional_encoding
```

**Properties:**
- Deterministic (no learned parameters)
- Extrapolates to unseen lengths (geometric properties)
- Relative positions: PE(pos+k) is linear function of PE(pos)

**When to use:** Variable-length sequences in NLP

### Strategy 2: Learned Position Embeddings

```python
self.pos_embedding = nn.Embedding(max_seq_len, d_model)

# Usage
positions = torch.arange(seq_len, device=x.device)
x = token_embeddings + self.pos_embedding(positions)
```

**Properties:**
- Learnable (adapts to data)
- Cannot extrapolate beyond max_seq_len

**When to use:**
- Fixed-length sequences
- Vision Transformers (image patches)
- When training data covers all positions

### Strategy 3: Rotary Position Embeddings (RoPE) ⭐

**Modern approach (2021+):** Rotate Q and K in complex plane

**Key advantages:**
- Encodes **relative** positions naturally
- Better long-range decay properties
- No addition to embeddings (applied in attention)

**Used in:** GPT-NeoX, PaLM, LLaMA, LLaMA-2, Mistral

```python
def apply_rotary_pos_emb(x, cos, sin):
    # x: (batch, num_heads, seq_len, d_k)
    # Split into even/odd
    x1, x2 = x[..., ::2], x[..., 1::2]
    # Rotate
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

### Strategy 4: ALiBi (Attention with Linear Biases) ⭐

**Simplest modern approach:** Add bias to attention scores (no embeddings!)

```python
# Bias matrix: -1 * distance
# [[0, -1, -2, -3],
#  [0,  0, -1, -2],
#  [0,  0,  0, -1],
#  [0,  0,  0,  0]]

scores = Q @ K^T / √d_k + alibi_bias
```

**Key advantages:**
- **Best extrapolation** to longer sequences
- No positional embeddings (simpler)
- Per-head slopes (different decay rates)

**Used in:** BLOOM

### Position Encoding Selection Guide

| Use Case | Recommended | Why |
|----------|-------------|-----|
| NLP (variable length) | RoPE (with extension if needed) | Production standard since LLaMA |
| NLP (fixed length) | Learned embeddings | Adapts to data |
| Vision (ViT) | 2D learned embeddings or RoPE-2D | Spatial structure |
| Long sequences (>2k) | RoPE + YaRN / NTK-aware scaling | Extends pretrained context |
| Legacy/compatibility | Sinusoidal | Original Transformer |

**Modern trend (2023-2026):** RoPE has won. Almost every open-weights LLM
released since LLaMA-1 uses RoPE. ALiBi is now a minority choice (BLOOM,
some research models). Sinusoidal is essentially historical.

### Strategy 5: Extending RoPE Context (YaRN / NTK-aware / Position Interpolation)

**Problem:** A model pretrained at 4k context degrades fast above its training
length, even with RoPE. You want 32k–128k+ context without retraining from
scratch.

**Solutions, in order of how invasive they are:**

1. **Position Interpolation (PI)** — Chen et al. 2023. Linearly down-scale
   position indices so test-time positions sit inside the trained range.
   Cheap, but blurs short-range resolution.
2. **NTK-aware scaling** — Adjust the RoPE base frequency rather than scaling
   positions linearly. Preserves high-frequency components better than PI.
3. **YaRN — Yet another RoPE extensioN** (Peng et al. 2023). Combines
   NTK-by-parts scaling with an attention temperature correction. The most
   widely deployed RoPE extension; lets a 4k-trained model fine-tune to 128k+
   with minimal extra training.
4. **LongRoPE** (Microsoft, 2024). Search-based per-dimension scaling factors
   pushing context to 2M+ tokens with light fine-tuning.

**Practical rule:** If a published checkpoint has a YaRN or LongRoPE variant,
use it. Don't rewrite RoPE math by hand.

### Strategy 6: NoPE — No Position Encoding

**Paper:** Kazemnejad et al. — *The Impact of Positional Encoding on Length
Generalization in Transformers* (NeurIPS 2023).

**Surprising finding:** Decoder-only Transformers with the causal mask but **no
positional encoding at all** can outperform RoPE/ALiBi on length generalization
in some controlled settings. The causal mask itself breaks permutation
invariance enough to learn position implicitly.

**Status:** Mostly research signal. Production models still use RoPE because the
ecosystem (caching, extension techniques, kernels) is built around it. Worth
knowing about; not worth defaulting to.


## Part 4: Architecture Variants

### Variant 1: Encoder-Only (Bidirectional)

**Architecture:**
- Self-attention: Each token attends to **ALL** tokens (past + future)
- No masking (bidirectional context)

**Examples:** BERT, RoBERTa, ELECTRA, DeBERTa

**Use cases:**
- Text classification
- Named entity recognition
- Question answering (extract span from context)
- Sentence embeddings

**Key property:** Sees full context → Good for **understanding**

**Implementation:**
```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)  # No causal mask!
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_output, _ = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)

        # Feed-forward + residual + norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x
```

### Variant 2: Decoder-Only (Autoregressive)

**Architecture:**
- Self-attention with **causal masking**
- Each token attends ONLY to past tokens (not future)

**Causal mask (lower triangular):**
```python
# mask[i, j] = 1 if j <= i else 0
[[1, 0, 0, 0],   # Token 0 sees only itself
 [1, 1, 0, 0],   # Token 1 sees tokens 0-1
 [1, 1, 1, 0],   # Token 2 sees tokens 0-2
 [1, 1, 1, 1]]   # Token 3 sees all
```

**Examples:** GPT, GPT-2, GPT-3, GPT-4, LLaMA, Mistral

**Use cases:**
- Text generation
- Language modeling
- Code generation
- Autoregressive prediction

**Key property:** Generates sequentially → Good for **generation**

**Implementation:**
```python
def create_causal_mask(seq_len, device):
    # Lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len, x.device)

        for layer in self.layers:
            x = layer(x, causal_mask)  # Apply causal mask!
        return x
```

**Modern trend (2023+):** Decoder-only architectures dominate
- Can do both generation AND understanding (via prompting)
- Simpler than encoder-decoder (no cross-attention)
- Scales better to massive sizes

### Variant 3: Encoder-Decoder (Seq2Seq)

**Architecture:**
- **Encoder**: Bidirectional self-attention (understands input)
- **Decoder**: Causal self-attention (generates output)
- **Cross-attention**: Decoder queries encoder outputs

**Cross-attention mechanism:**
```python
# Q from decoder, K and V from encoder
Q = decoder_hidden @ W_q
K = encoder_output @ W_k
V = encoder_output @ W_v

cross_attn = softmax(Q K^T / √d_k) V
```

**Examples:** T5, BART, mT5, original Transformer (2017)

**Use cases:**
- Translation (input ≠ output language)
- Summarization (long input → short output)
- Question answering (generate answer, not extract)

**When to use:** Input and output are fundamentally different

**Implementation:**
```python
class EncoderDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # NEW!
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, decoder_input, encoder_output, causal_mask=None):
        # 1. Self-attention (causal)
        self_attn_out, _ = self.self_attn(decoder_input, causal_mask)
        x = self.norm1(decoder_input + self_attn_out)

        # 2. Cross-attention (Q from decoder, K/V from encoder)
        cross_attn_out, _ = self.cross_attn.forward_cross(
            query=x,
            key=encoder_output,
            value=encoder_output
        )
        x = self.norm2(x + cross_attn_out)

        # 3. Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + ff_out)

        return x
```

### Architecture Selection Guide

| Task | Architecture | Why |
|------|--------------|-----|
| Classification | Encoder-only | Need full bidirectional context |
| Text generation | Decoder-only | Autoregressive generation |
| Translation | Encoder-decoder or Decoder-only | Different languages, or use prompting |
| Summarization | Encoder-decoder or Decoder-only | Length mismatch, or use prompting |
| Q&A (extract) | Encoder-only | Find span in context |
| Q&A (generate) | Decoder-only | Generate freeform answer |

**2023+ trend:** Decoder-only can do everything via prompting (but less parameter-efficient for some tasks)


## Part 5: Vision Transformers (ViT)

### From Images to Sequences

**Key insight:** Treat image as sequence of patches

**Process:**
1. Split image into patches (e.g., 16×16 pixels)
2. Flatten each patch → 1D vector
3. Linear projection → token embeddings
4. Add 2D positional embeddings
5. Prepend [CLS] token (for classification)
6. Feed to Transformer encoder

**Example:** 224×224 image, 16×16 patches
- Number of patches: (224/16)² = 196
- Each patch: 16 × 16 × 3 = 768 dimensions
- Transformer input: 197 tokens (196 patches + 1 [CLS])

### ViT Implementation

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 d_model=768, num_heads=12, num_layers=12, num_classes=1000):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        # Patch embedding (linear projection of flattened patches)
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # [CLS] token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Position embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        # Transformer encoder
        self.encoder = TransformerEncoder(d_model, num_heads,
                                         d_ff=4*d_model, num_layers=num_layers)

        # Classification head
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, channels, height, width)
        batch_size = x.size(0)

        # Divide into patches and flatten
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        # (batch, channels, num_patches_h, num_patches_w, patch_size, patch_size)

        x = x.contiguous().view(batch_size, -1, self.patch_size ** 2 * 3)
        # (batch, num_patches, patch_dim)

        # Linear projection
        x = self.patch_embed(x)  # (batch, num_patches, d_model)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches+1, d_model)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer encoder
        x = self.encoder(x)

        # Classification: Use [CLS] token
        cls_output = x[:, 0]  # (batch, d_model)
        logits = self.head(cls_output)

        return logits
```

### ViT vs CNN: Critical Differences

**1. Inductive Bias**

| Property | CNN | ViT |
|----------|-----|-----|
| Locality | Strong (conv kernel) | Weak (global attention) |
| Translation invariance | Strong (weight sharing) | Weak (position embeddings) |
| Hierarchy | Strong (pooling layers) | None (flat patches) |

**Implication:** CNN has strong priors, ViT learns from data

**2. Data Requirements**

| Dataset Size | CNN | ViT (supervised from scratch) | ViT (with MAE/DINOv2-style SSL) | ViT (fine-tuned from pretrained) |
|--------------|-----|------------------------------|---------------------------------|----------------------------------|
| Small (< 100k) | ✅ Good | ❌ Fails | ⚠️ Limited | ✅ Good |
| Medium (100k-1M) | ✅ Excellent | ⚠️ Poor | ✅ Good | ✅ Excellent |
| Large (> 1M) | ✅ Excellent | ⚠️ OK | ✅ Excellent | ✅ Excellent |
| Huge (> 100M) | ✅ Excellent | ✅ SOTA | ✅ SOTA | N/A |

**Original 2020-era finding (Dosovitskiy et al.):** ViT needed JFT-300M to beat
CNNs from scratch. **Updated picture (2022+):**

- **DeiT** (Touvron et al. 2021): With aggressive augmentation + distillation,
  ViT reaches CNN-level accuracy on ImageNet-1k (~1.3M images) from scratch.
- **MAE — Masked Autoencoders** (He et al. 2022): Self-supervised pretraining
  on ImageNet-1k (mask 75% of patches, reconstruct) makes ViT competitive
  without supervised labels.
- **DINOv2** (Oquab et al. 2024): Self-supervised ViT trained on a curated
  142M-image dataset; produces general-purpose features that beat
  CLIP/SigLIP on dense prediction tasks. This is now the default off-the-shelf
  vision encoder when you don't want a CLIP-aligned representation.
- **EVA / EVA-02** (Fang et al. 2023): Masked image modeling with CLIP targets;
  scales cleanly to multi-billion-parameter ViTs.
- **Register tokens** (Darcet et al. ICLR 2024): Adding a few extra learnable
  "register" tokens to ViT removes attention-map artifacts and improves dense
  prediction — a small, near-free architectural fix that should be in any new
  ViT recipe.

**Practical rule for 2026:** "ViT needs 100M images" is no longer true.
For most vision tasks, fine-tune a DINOv2 / SigLIP / EVA-02 / MAE checkpoint
rather than train from scratch — same as the NLP playbook.

**3. Computational Cost**

**Example: 224×224 images**

| Model | Parameters | GFLOPs | Inference (GPU) |
|-------|-----------|--------|-----------------|
| ResNet-50 | 25M | 4.1 | ~30ms |
| EfficientNet-B0 | 5M | 0.4 | ~10ms |
| ViT-B/16 | 86M | 17.6 | ~100ms |

**Implication:** ViT is 40x more expensive than EfficientNet!

### When to Use ViT

✅ **Use ViT when:**
- Large dataset (> 1M images) OR using pretrained weights
- Computational cost acceptable (cloud, large GPU)
- Best possible accuracy needed
- Can fine-tune from ImageNet-21k checkpoint

❌ **Use CNN when:**
- Small/medium dataset (< 1M images) and training from scratch
- Limited compute/memory
- Edge deployment (mobile, embedded)
- Need architectural inductive biases

### Hybrid Approaches (2022-2024)

**ConvNeXt / ConvNeXt v2:** CNN with ViT design choices.
- ConvNeXt (Liu et al. 2022) ports ViT recipes (large kernels, LayerNorm,
  GELU, fewer activations) into a pure-conv backbone.
- ConvNeXt v2 (Woo et al. 2023) adds a fully convolutional masked autoencoder
  (FCMAE) pretraining objective and Global Response Normalization.
- Result: matches ViT accuracy at similar scale with CNN-friendly inference.

**Swin / Swin v2:** Hierarchical ViT with shifted local windows.
- Original Swin (Liu et al. 2021): O(n) complexity, strong on dense
  prediction.
- Swin v2 (Liu et al. 2022): scales cleanly past 3B params; fixes the
  attention-instability issues that hurt original Swin at scale.

**MaxViT** (Tu et al. ECCV 2022): Block (local) + grid (sparse global)
attention alternation, plus mobile-conv blocks. Strong Pareto curve at
moderate compute.

**Hiera** (Ryali et al. ICML 2023): Strips Swin back to a plain hierarchical
ViT trained with MAE; matches Swin/MaxViT with a much simpler architecture.

**FastViT** (Vasu et al. ICCV 2023): Reparameterizable hybrid CNN/ViT
optimized for mobile inference latency.

**CoAtNet** (Dai et al. NeurIPS 2021): Conv layers early, Transformer layers
late. Was state-of-the-art on ImageNet for a window in 2021–2022; still a
clean reference for the conv-then-attention pattern.


## Part 6: Implementation Checklist

### Critical Details

**1. Layer Norm Placement**

**Post-norm (original):**
```python
x = x + self_attn(x)
x = layer_norm(x)
```

**Pre-norm (modern, recommended):**
```python
x = x + self_attn(layer_norm(x))
```

**Why pre-norm?** More stable training, less sensitive to learning rate

**2. Attention Dropout**

Apply dropout to **attention weights**, not Q/K/V!

```python
attn_weights = F.softmax(scores, dim=-1)
attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)  # HERE!
output = torch.matmul(attn_weights, V)
```

**3. Feed-Forward Dimension**

Typically: d_ff = 4 × d_model
- BERT: d_model=768, d_ff=3072
- GPT-2: d_model=768, d_ff=3072

**4. Residual Connections**

ALWAYS use residual connections (essential for training)!

```python
x = x + self_attn(x)  # Residual
x = x + feed_forward(x)  # Residual
```

**5. Initialization**

Use Xavier/Glorot initialization for attention weights:
```python
nn.init.xavier_uniform_(self.W_q.weight)
nn.init.xavier_uniform_(self.W_k.weight)
nn.init.xavier_uniform_(self.W_v.weight)
```


## Part 7: When NOT to Use Transformers

### Limitation 1: Small Datasets

**Problem:** Transformers have weak inductive bias (learn from data)

**Impact:**
- ViT: Fails on < 100k images without pretraining
- NLP: BERT needs 100M+ tokens for pretraining

**Solution:** Use models with stronger priors (CNN for vision, smaller models for text)

### Limitation 2: Long Sequences

**Problem:** O(n²) memory complexity

**Impact:**
- Standard Transformer: n=10k → 100M attention scores
- GPU memory: 10k² × 4 bytes = 400MB per sample!

**Solution:**
- Sparse attention (Longformer, BigBird)
- Linear attention (Linformer, Performer)
- Flash Attention (memory-efficient kernel)
- State space models (S4, Mamba)

### Limitation 3: Edge Deployment

**Problem:** Large model size, high latency

**Impact:**
- ViT-B: 86M parameters, ~100ms inference
- Mobile/embedded: Need < 10M parameters, < 50ms

**Solution:** Efficient CNNs (MobileNet, EfficientNet) or distilled models

### Limitation 4: Real-Time Processing

**Problem:** Sequential generation in decoder (cannot parallelize at inference)

**Impact:** GPT-style models generate one token at a time

**Solution:** Non-autoregressive models, speculative decoding, or smaller models


## Part 8: Common Mistakes

### Mistake 1: Forgetting Causal Mask

**Symptom:** Decoder "cheats" by seeing future tokens

**Fix:** Always apply causal mask to decoder self-attention!

```python
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
```

### Mistake 2: Wrong Dimension for Multi-Head

**Symptom:** Runtime error or dimension mismatch

**Fix:** Ensure d_model % num_heads == 0

```python
assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
```

### Mistake 3: Forgetting Position Encoding

**Symptom:** Model ignores word order

**Fix:** Always add position information!

```python
x = token_embeddings + positional_encoding
```

### Mistake 4: Wrong Softmax Dimension

**Symptom:** Attention weights don't sum to 1 per query

**Fix:** Softmax over last dimension (keys)

```python
attn_weights = F.softmax(scores, dim=-1)  # Sum over keys for each query
```

### Mistake 5: No Residual Connections

**Symptom:** Training diverges or converges very slowly

**Fix:** Always add residual connections!

```python
x = x + self_attn(x)
x = x + feed_forward(x)
```


## Summary: Quick Reference

### Architecture Selection

```
Classification/Understanding → Encoder-only (BERT-style)
Generation/Autoregressive → Decoder-only (GPT-style)
Seq2Seq (input ≠ output) → Encoder-decoder (T5-style) or Decoder-only with prompting
```

### Position Encoding Selection

```
NLP (variable length) → RoPE or ALiBi
NLP (fixed length) → Learned embeddings
Vision (ViT) → 2D learned embeddings
Long sequences (> 2k) → ALiBi (best extrapolation)
```

### Multi-Head Configuration

```
Small models (d_model < 512): 4-8 heads
Medium models (d_model 512-1024): 8-12 heads
Large models (d_model > 1024): 12-32 heads
Rule: d_k (head dimension) should be 64-128
```

### ViT vs CNN

```
ViT: Large dataset (> 1M) OR pretrained weights
CNN: Small dataset (< 1M) OR edge deployment
```

### Implementation Essentials

```
✅ Pre-norm (more stable than post-norm)
✅ Residual connections (essential!)
✅ Causal mask for decoder
✅ Attention dropout (on weights, not Q/K/V)
✅ d_ff = 4 × d_model (feed-forward dimension)
✅ Check: d_model % num_heads == 0
```


## Next Steps

After mastering this skill:
- `attention-mechanisms-catalog`: Explore attention variants (sparse, linear, Flash)
- `llm-specialist/llm-finetuning-strategies`: Apply to language models
- `architecture-design-principles`: Understand design trade-offs

**Remember:** Transformers are NOT magic. Understanding the mechanism (information retrieval via Q/K/V) beats cargo-culting implementations.
