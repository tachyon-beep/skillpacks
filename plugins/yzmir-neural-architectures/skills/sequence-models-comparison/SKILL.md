---
name: sequence-models-comparison
description: Sequence models: RNN/LSTM/GRU, Transformers, TCN, Sparse Transformers, S4 with selection
pack: neural-architectures
faction: yzmir
---

# Sequence Models Comparison: Choosing the Right Architecture for Sequential Data

<CRITICAL_CONTEXT>
Sequence modeling has evolved rapidly:
- 2014-2017: LSTM/GRU dominated
- 2017+: Transformers revolutionized the field
- 2018+: TCN emerged as efficient alternative
- 2021+: Sparse Transformers for very long sequences
- 2022+: State Space Models (S4) for extreme lengths

Don't default to LSTM (outdated) or blindly use Transformers (not always appropriate).
Match architecture to your sequence characteristics.
</CRITICAL_CONTEXT>

## When to Use This Skill

Use this skill when:
- ✅ Selecting model for sequential/temporal data
- ✅ Comparing RNN vs LSTM vs Transformer
- ✅ Deciding on sequence architecture for time series, text, audio
- ✅ Understanding modern alternatives to LSTM
- ✅ Optimizing for sequence length, speed, or accuracy

DO NOT use for:
- ❌ Vision tasks (use cnn-families-and-selection)
- ❌ Graph-structured data (use graph-neural-networks-basics)
- ❌ LLM-specific questions (use llm-specialist pack)

**When in doubt:** If data is sequential/temporal → this skill.

---

## Selection Framework

### Step 1: Identify Key Characteristics

**Before recommending, ask:**

| Characteristic | Question | Impact |
|----------------|----------|--------|
| **Sequence Length** | Typical length? | Short (< 100) → LSTM/CNN, Medium (100-1k) → Transformer, Long (> 1k) → Sparse Transformer/S4 |
| **Data Type** | Language, time series, audio? | Language → Transformer, Time series → TCN/Transformer, Audio → Specialized |
| **Data Volume** | Training examples? | Small (< 10k) → LSTM/TCN, Large (> 100k) → Transformer |
| **Latency** | Real-time needed? | Yes → TCN/LSTM, No → Transformer |
| **Deployment** | Cloud/edge/mobile? | Edge → TCN/LSTM, Cloud → Any |

### Step 2: Apply Decision Tree

```
START: What's your primary constraint?

┌─ SEQUENCE LENGTH
│  ├─ Short (< 100 steps)
│  │  ├─ Language → BiLSTM or small Transformer
│  │  └─ Time series → TCN or LSTM
│  │
│  ├─ Medium (100-1000 steps)
│  │  ├─ Language → Transformer (BERT-style)
│  │  └─ Time series → Transformer or TCN
│  │
│  ├─ Long (1000-10000 steps)
│  │  ├─ Sparse Transformer (Longformer, BigBird)
│  │  └─ Hierarchical models
│  │
│  └─ Very Long (> 10000 steps)
│     └─ State Space Models (S4)
│
├─ DATA TYPE
│  ├─ Natural Language
│  │  ├─ < 50k data → BiLSTM or DistilBERT
│  │  └─ > 50k data → Transformer (BERT, RoBERTa)
│  │
│  ├─ Time Series
│  │  ├─ Fast training → TCN
│  │  ├─ Long sequences → Transformer
│  │  └─ Multivariate → Transformer with cross-series attention
│  │
│  └─ Audio
│     ├─ Waveform → WaveNet (TCN-based)
│     └─ Spectrograms → CNN + Transformer
│
└─ COMPUTATIONAL CONSTRAINT
   ├─ Edge device → TCN or small LSTM
   ├─ Real-time latency → TCN (parallel inference)
   └─ Cloud, no constraint → Transformer
```

---

## Architecture Catalog

### 1. RNN (Recurrent Neural Networks) - Legacy Foundation

**Architecture:** Basic recurrent cell with hidden state

**Status:** **OUTDATED** - don't use for new projects

**Why it existed:**
- First neural approach to sequences
- Hidden state captures temporal information
- Theoretically can model any sequence

**Why it failed:**
- Vanishing gradient (can't learn long dependencies)
- Very slow training (sequential processing)
- Replaced by LSTM in 2014

**When to mention:**
- Historical context only
- Teaching purposes
- Never recommend for production

**Key Insight:** Proved neural nets could handle sequences, but impractical due to vanishing gradients

---

### 2. LSTM (Long Short-Term Memory) - Legacy Standard

**Architecture:** Gated recurrent cell (forget, input, output gates)

**Complexity:** O(n) memory, sequential processing

**Strengths:**
- Solves vanishing gradient (gates maintain long-term info)
- Works well for short-medium sequences (< 500 steps)
- Small datasets (< 10k examples)
- Low memory footprint

**Weaknesses:**
- Sequential processing (slow training, can't parallelize)
- Still struggles with very long sequences (> 1000 steps)
- Slow inference (especially bidirectional)
- Superseded by Transformers for most language tasks

**When to Use:**
- ✅ Small datasets (< 10k examples)
- ✅ Short sequences (< 100 steps)
- ✅ Edge deployment (low memory)
- ✅ Baseline comparison

**When NOT to Use:**
- ❌ Large datasets (Transformer better)
- ❌ Long sequences (> 500 steps)
- ❌ Modern NLP (Transformer standard)
- ❌ Fast training needed (TCN better)

**Code Example:**
```python
class SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                           num_layers=2,
                           batch_first=True,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last timestep
        out = self.fc(lstm_out[:, -1, :])
        return out
```

**Status:** Legacy but still useful for specific cases (small data, edge deployment)

---

### 3. GRU (Gated Recurrent Unit) - Simplified LSTM

**Architecture:** Simplified gating (2 gates instead of 3)

**Advantages over LSTM:**
- Fewer parameters (faster training)
- Similar performance in many tasks
- Lower memory

**Disadvantages:**
- Still sequential (same as LSTM)
- No major advantage over LSTM in practice
- Also superseded by Transformers

**When to Use:**
- Same as LSTM, but prefer LSTM for slightly better performance
- Use if computational savings matter

**Status:** Rarely recommended - if using recurrent, prefer LSTM or move to Transformer/TCN

---

### 4. Transformer - Modern Standard

**Architecture:** Self-attention mechanism, parallel processing

**Complexity:**
- Memory: O(n²) for sequence length n
- Compute: O(n²d) where d is embedding dimension

**Strengths:**
- ✅ Parallel processing (fast training)
- ✅ Captures long-range dependencies (better than LSTM)
- ✅ State-of-the-art for language (BERT, GPT)
- ✅ Pre-trained models available
- ✅ Scales with data (more data = better performance)

**Weaknesses:**
- ❌ Quadratic memory (struggles with sequences > 1000)
- ❌ Needs more data than LSTM (> 10k examples)
- ❌ Slower inference than TCN
- ❌ Harder to interpret than RNN

**When to Use:**
- ✅ **Natural language** (current standard)
- ✅ Medium sequences (100-1000 tokens)
- ✅ Large datasets (> 50k examples)
- ✅ Pre-training available (BERT, GPT)
- ✅ Accuracy priority

**When NOT to Use:**
- ❌ Short sequences (< 50 tokens) - LSTM/CNN competitive, simpler
- ❌ Very long sequences (> 2000) - quadratic memory explodes
- ❌ Small datasets (< 10k) - will overfit
- ❌ Edge deployment - large model size

**Memory Analysis:**
```python
# Standard Transformer attention

# For sequence length n=1000, batch_size=32, embedding_dim=512:
attention_weights = softmax(Q @ K^T / sqrt(d))  # Shape: (32, 1000, 1000)
# Memory: 32 * 1000 * 1000 * 4 bytes = 128 MB just for attention!

# For n=5000:
# Memory: 32 * 5000 * 5000 * 4 bytes = 3.2 GB per batch!
# → Impossible on most GPUs
```

**Code Example:**
```python
from transformers import BertModel, BertTokenizer

# Pre-trained BERT for text classification
class TransformerClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.pooler_output
        return self.classifier(pooled)

# Fine-tuning
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TransformerClassifier(num_classes=2)
```

**Status:** **Current standard for NLP**, competitive for time series with large data

---

### 5. TCN (Temporal Convolutional Network) - Underrated Alternative

**Architecture:** 1D convolutions with dilated causal convolutions

**Complexity:** O(n) memory, fully parallel processing

**Strengths:**
- ✅ **Parallel training** (much faster than LSTM)
- ✅ **Parallel inference** (faster than LSTM/Transformer)
- ✅ Linear memory (no quadratic blow-up)
- ✅ Large receptive field (dilation)
- ✅ Works well for time series
- ✅ Simple architecture

**Weaknesses:**
- ❌ Less popular (fewer pre-trained models)
- ❌ Not standard for language (Transformer dominates)
- ❌ Fixed receptive field (vs adaptive attention)

**When to Use:**
- ✅ **Time series forecasting** (often BETTER than LSTM)
- ✅ **Fast training needed** (2-3x faster than LSTM)
- ✅ **Fast inference** (real-time applications)
- ✅ Long sequences (linear memory)
- ✅ Audio processing (WaveNet is TCN-based)

**When NOT to Use:**
- ❌ Natural language with pre-training available (use Transformer)
- ❌ Need very large receptive field (Transformer better)

**Performance Comparison:**
```
Time series forecasting (1000-step sequences):

Training speed:
- LSTM: 100% (baseline, sequential)
- TCN: 35% (2.8x faster, parallel)
- Transformer: 45% (2.2x faster)

Inference speed:
- LSTM: 100% (sequential)
- TCN: 20% (5x faster, parallel)
- Transformer: 60% (1.7x faster)

Accuracy (similar across all three):
- LSTM: Baseline
- TCN: Equal or slightly better
- Transformer: Equal or slightly better (needs more data)

Conclusion: TCN wins on speed, matches accuracy
```

**Code Example:**
```python
class TCN(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size=3):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            # Causal dilated convolution
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                         padding=(kernel_size-1) * dilation_size,
                         dilation=dilation_size)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Usage for time series
# Input: (batch, channels, sequence_length)
model = TCN(input_channels=1, num_channels=[64, 128, 256])
```

**Key Insight:** Dilated convolutions create exponentially large receptive field (2^k) while maintaining linear memory

**Status:** **Excellent for time series**, underrated, should be considered before LSTM

---

### 6. Sparse Transformers - Long Sequence Specialists

**Architecture:** Modified attention patterns to reduce complexity

**Variants:**
- **Longformer**: Local + global attention
- **BigBird**: Random + local + global attention
- **Linformer**: Low-rank projection of keys/values
- **Performer**: Kernel approximation of attention

**Complexity:** O(n log n) or O(n) depending on variant

**When to Use:**
- ✅ **Long sequences** (1000-10000 tokens)
- ✅ Document processing (multi-page documents)
- ✅ Long-context language modeling
- ✅ When standard Transformer runs out of memory

**Trade-offs:**
- Slightly lower accuracy than full attention (approximation)
- More complex implementation
- Fewer pre-trained models

**Example Use Cases:**
- Legal document analysis (10k+ tokens)
- Scientific paper understanding
- Long-form text generation
- Time series with thousands of steps

**Status:** Specialized for long sequences, active research area

---

### 7. State Space Models (S4) - Cutting Edge

**Architecture:** Structured state space with efficient recurrence

**Complexity:** O(n log n) training, O(n) inference

**Strengths:**
- ✅ **Very long sequences** (10k-100k steps)
- ✅ Linear inference complexity
- ✅ Strong theoretical foundations
- ✅ Handles continuous-time sequences

**Weaknesses:**
- ❌ Newer (less mature ecosystem)
- ❌ Complex mathematics
- ❌ Fewer pre-trained models
- ❌ Harder to implement

**When to Use:**
- ✅ Extremely long sequences (> 10k steps)
- ✅ Audio (raw waveforms, 16kHz sampling)
- ✅ Medical signals (ECG, EEG)
- ✅ Research applications

**Status:** **Cutting edge** (2022+), promising for very long sequences

---

## Practical Selection Guide

### Scenario 1: Natural Language Processing

**Short text (< 50 tokens, e.g., tweets, titles):**
```
Small dataset (< 10k):
→ BiLSTM or 1D CNN (simple, effective)

Large dataset (> 10k):
→ DistilBERT (smaller Transformer, 40M params)
→ Or BiLSTM if latency critical
```

**Medium text (50-512 tokens, e.g., reviews, articles):**
```
Standard approach:
→ BERT, RoBERTa, or similar (110M params)
→ Fine-tune on task-specific data

Small dataset:
→ DistilBERT (66M params, faster, similar accuracy)
```

**Long documents (> 512 tokens):**
```
→ Longformer (4096 tokens max)
→ BigBird (4096 tokens max)
→ Hierarchical: Process in chunks, aggregate
```

---

### Scenario 2: Time Series Forecasting

**Short sequences (< 100 steps):**
```
Fast training:
→ TCN (2-3x faster than LSTM)

Small dataset:
→ LSTM or simple models (ARIMA, Prophet)

Baseline:
→ LSTM (well-tested)
```

**Medium sequences (100-1000 steps):**
```
Best accuracy:
→ Transformer (if data > 50k examples)

Fast training/inference:
→ TCN (parallel processing)

Multivariate:
→ Transformer with cross-series attention
```

**Long sequences (> 1000 steps):**
```
→ Sparse Transformer (Informer for time series)
→ Hierarchical models (chunk + aggregate)
→ State Space Models (S4)
```

---

### Scenario 3: Audio Processing

**Waveform (raw audio, 16kHz):**
```
→ WaveNet (TCN-based)
→ State Space Models (S4)
```

**Spectrograms (mel-spectrograms):**
```
→ CNN + BiLSTM (traditional)
→ CNN + Transformer (modern)
```

**Speech recognition:**
```
→ Transformer (Wav2Vec 2.0, Whisper)
→ Pre-trained models available
```

---

## Trade-Off Analysis

### Speed Comparison

**Training speed (1000-step sequences):**
```
LSTM:         100% (baseline, sequential)
GRU:          75% (simpler gates)
TCN:          35% (2.8x faster, parallel)
Transformer:  45% (2.2x faster, parallel)

Conclusion: TCN fastest for training
```

**Inference speed:**
```
LSTM:         100% (sequential)
BiLSTM:       200% (2x passes)
TCN:          20% (5x faster, parallel)
Transformer:  60% (faster, but attention overhead)

Conclusion: TCN fastest for inference
```

---

### Memory Comparison

**Sequence length n=1000, batch=32:**
```
LSTM:              ~500 MB (linear in n)
Transformer:       ~2 GB (quadratic in n)
TCN:               ~400 MB (linear in n)
Sparse Transformer: ~800 MB (n log n)

For n=5000:
LSTM:              ~2 GB
Transformer:       OUT OF MEMORY (50 GB needed!)
TCN:               ~2 GB
Sparse Transformer: ~4 GB
```

---

### Accuracy vs Data Size

**Small dataset (< 10k examples):**
```
LSTM:       ★★★★☆ (works well with little data)
Transformer: ★★☆☆☆ (overfits, needs more data)
TCN:        ★★★★☆ (similar to LSTM)

Winner: LSTM or TCN
```

**Large dataset (> 100k examples):**
```
LSTM:       ★★★☆☆ (good but plateaus)
Transformer: ★★★★★ (best, scales with data)
TCN:        ★★★★☆ (competitive)

Winner: Transformer
```

---

## Common Pitfalls

### Pitfall 1: Using LSTM in 2025 Without Considering Modern Alternatives
**Symptom:** Defaulting to LSTM for all sequence tasks

**Why it's wrong:** Transformers (language) and TCN (time series) often better

**Fix:** Consider Transformer for language, TCN for time series, LSTM for small data/edge only

---

### Pitfall 2: Using Standard Transformer for Very Long Sequences
**Symptom:** Running out of memory on sequences > 1000 tokens

**Why it's wrong:** O(n²) memory explodes

**Fix:** Use Sparse Transformer (Longformer, BigBird) or hierarchical approach

---

### Pitfall 3: Not Trying TCN for Time Series
**Symptom:** Struggling with slow LSTM training

**Why it's wrong:** TCN is 2-3x faster, often more accurate

**Fix:** Try TCN before optimizing LSTM

---

### Pitfall 4: Using Transformer for Small Datasets
**Symptom:** Transformer overfits on < 10k examples

**Why it's wrong:** Transformers need large datasets to work well

**Fix:** Use LSTM or TCN for small datasets, or use pre-trained Transformer

---

### Pitfall 5: Ignoring Sequence Length Constraints
**Symptom:** Choosing architecture without considering typical sequence length

**Why it's wrong:** Architecture effectiveness varies dramatically with length

**Fix:** Match architecture to sequence length (short → LSTM/CNN, long → Sparse Transformer)

---

## Evolution Timeline

**Understanding why architectures evolved:**

```
2010-2013: Basic RNN
→ Vanishing gradient problem
→ Can't learn long dependencies

2014: LSTM (Hochreiter & Schmidhuber)
→ Gates solve vanishing gradient
→ Became standard for sequences

2014: GRU
→ Simplified LSTM
→ Similar performance, fewer parameters

2017: Transformer (Attention Is All You Need)
→ Self-attention replaces recurrence
→ Parallel processing (fast training)
→ Revolutionized NLP

2018: TCN (Temporal Convolutional Networks)
→ Dilated convolutions for sequences
→ Often better than LSTM for time series
→ Underrated alternative

2020: Sparse Transformers
→ Reduce quadratic complexity
→ Enable longer sequences

2021: State Space Models (S4)
→ Very long sequences (10k-100k)
→ Theoretical foundations
→ Cutting edge research

Current (2025):
- NLP: Transformer standard (BERT, GPT)
- Time Series: TCN or Transformer
- Audio: Specialized (WaveNet, Transformer)
- Edge: LSTM or TCN (low memory)
```

---

## Decision Checklist

Before choosing sequence model:

```
☐ Sequence length? (< 100 / 100-1k / > 1k)
☐ Data type? (language / time series / audio / other)
☐ Dataset size? (< 10k / 10k-100k / > 100k)
☐ Latency requirement? (real-time / batch / offline)
☐ Deployment target? (cloud / edge / mobile)
☐ Pre-trained models available? (yes / no)
☐ Training speed critical? (yes / no)

Based on answers:
→ Language + large data → Transformer
→ Language + small data → BiLSTM or DistilBERT
→ Time series + speed → TCN
→ Time series + accuracy + large data → Transformer
→ Very long sequences → Sparse Transformer or S4
→ Edge deployment → TCN or LSTM
→ Real-time latency → TCN
```

---

## Integration with Other Skills

**For language-specific questions:**
→ `yzmir/llm-specialist/using-llm-specialist`
- LLM-specific Transformers (GPT, BERT variants)
- Fine-tuning strategies
- Prompt engineering

**For Transformer internals:**
→ `yzmir/neural-architectures/transformer-architecture-deepdive`
- Attention mechanisms
- Positional encoding
- Transformer variants

**After selecting architecture:**
→ `yzmir/training-optimization/using-training-optimization`
- Optimizer selection
- Learning rate schedules
- Handling sequence-specific training issues

---

## Summary

**Quick Reference Table:**

| Use Case | Best Choice | Alternative | Avoid |
|----------|-------------|-------------|-------|
| Short text (< 50 tokens) | BiLSTM, DistilBERT | 1D CNN | Full BERT (overkill) |
| Long text (> 512 tokens) | Longformer, BigBird | Hierarchical | Standard BERT (memory) |
| Time series (< 1k steps) | TCN, Transformer | LSTM | Basic RNN |
| Time series (> 1k steps) | Sparse Transformer, S4 | Hierarchical | Standard Transformer |
| Small dataset (< 10k) | LSTM, TCN | Simple models | Transformer (overfits) |
| Large dataset (> 100k) | Transformer | TCN | LSTM (plateaus) |
| Edge deployment | TCN, LSTM | Quantized Transformer | Large Transformer |
| Real-time inference | TCN | Small LSTM | BiLSTM, Transformer |

**Key Principles:**
1. **Don't default to LSTM** (outdated for most tasks)
2. **Transformer for language** (current standard, if data sufficient)
3. **TCN for time series** (fast, effective, underrated)
4. **Match to sequence length** (short → LSTM/CNN, long → Sparse Transformer)
5. **Consider modern alternatives** (don't stop at LSTM vs Transformer)

---

**END OF SKILL**
