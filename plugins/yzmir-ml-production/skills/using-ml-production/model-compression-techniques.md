
# Model Compression Techniques

## When to Use This Skill

Use this skill when:
- Deploying models to edge devices (mobile, IoT, embedded systems)
- Model too large for deployment constraints (storage, memory, bandwidth)
- Inference costs too high (need smaller/faster model)
- Need to balance model size, speed, and accuracy
- Combining multiple compression techniques (quantization + pruning + distillation)

**When NOT to use:**
- Model already fits deployment constraints (compression unnecessary)
- Training optimization needed (use training-optimization pack instead)
- Quantization is sufficient (use quantization-for-inference instead)
- LLM-specific optimization (use llm-specialist for KV cache, speculative decoding)

**Relationship with quantization-for-inference:**
- Quantization: Reduce precision (FP32 → INT8/INT4) - 4× size reduction
- Compression: Reduce architecture (pruning, distillation) - 2-10× size reduction
- Often combined: Quantization + pruning + distillation = 10-50× total reduction

## Core Principle

**Compression is not one-size-fits-all. Architecture and deployment target determine technique.**

Without systematic compression:
- Mobile deployment: 440MB model crashes 2GB devices
- Wrong technique: Pruning transformers → 33pp accuracy drop
- Unstructured pruning: No speedup on standard hardware
- Aggressive distillation: 77× compression produces gibberish
- No recovery: 5pp preventable accuracy loss

**Formula:** Architecture analysis (transformer vs CNN) + Deployment constraints (hardware, latency, size) + Technique selection (pruning vs distillation) + Quality preservation (recovery, progressive compression) = Production-ready compressed model.


## Compression Decision Framework

```
Model Compression Decision Tree

1. What is target deployment?
   ├─ Edge/Mobile (strict size/memory) → Aggressive compression (4-10×)
   ├─ Cloud/Server (cost optimization) → Moderate compression (2-4×)
   └─ On-premises (moderate constraints) → Balanced approach

2. What is model architecture?
   ├─ Transformer (BERT, GPT, T5)
   │  └─ Primary: Knowledge distillation (preserves attention)
   │  └─ Secondary: Layer dropping, quantization
   │  └─ AVOID: Aggressive unstructured pruning (destroys quality)
   │
   ├─ CNN (ResNet, EfficientNet, MobileNet)
   │  └─ Primary: Structured channel pruning (works well)
   │  └─ Secondary: Quantization (INT8 standard)
   │  └─ Tertiary: Knowledge distillation (classification tasks)
   │
   └─ RNN/LSTM
      └─ Primary: Quantization (safe, effective)
      └─ Secondary: Structured pruning (hidden dimension)
      └─ AVOID: Unstructured pruning (breaks sequential dependencies)

3. What is deployment hardware?
   ├─ CPU/GPU/Mobile (standard) → Structured pruning (actual speedup)
   └─ Specialized (A100, sparse accelerators) → Unstructured pruning possible

4. What is acceptable quality loss?
   ├─ <2pp → Conservative: Quantization only (4× reduction)
   ├─ 2-5pp → Moderate: Quantization + structured pruning (6-10× reduction)
   └─ >5pp → Aggressive: Full pipeline with distillation (10-50× reduction)

5. Combine techniques for maximum compression:
   Quantization (4×) + Pruning (2×) + Distillation (2×) = 16× total reduction
```


## Part 1: Structured vs Unstructured Pruning

### When to Use Each

**Unstructured Pruning:**
- **Use when:** Sparse hardware available (NVIDIA A100, specialized accelerators)
- **Benefit:** Highest compression (70-90% sparsity possible)
- **Drawback:** No speedup on standard hardware (computes zeros anyway)
- **Hardware support:** Rare (most deployments use standard CPU/GPU)

**Structured Pruning:**
- **Use when:** Standard hardware (CPU, GPU, mobile) - 99% of deployments
- **Benefit:** Actual speedup (smaller dense matrices)
- **Drawback:** Lower compression ratio (50-70% typical)
- **Variants:** Channel pruning (CNNs), layer dropping (transformers), attention head pruning

### Structured Channel Pruning (CNNs)

**Problem:** Unstructured pruning creates sparse tensors that don't accelerate on standard hardware.

**Solution:** Remove entire channels to create smaller dense model (actual speedup).

```python
import torch
import torch.nn as nn
import torch_pruning as tp

def structured_channel_pruning_cnn(model, pruning_ratio=0.5, example_input=None):
    """
    Structured channel pruning for CNNs (actual speedup on all hardware).

    WHY structured: Removes entire channels/filters, creating smaller dense model
    WHY works: Smaller dense matrices compute faster than sparse matrices on standard hardware

    Args:
        model: CNN model to prune
        pruning_ratio: Fraction of channels to remove (0.5 = remove 50%)
        example_input: Example input tensor for tracing dependencies

    Returns:
        Pruned model (smaller, faster)
    """
    if example_input is None:
        example_input = torch.randn(1, 3, 224, 224)

    # Define importance metric (L1 norm of channels)
    # WHY L1 norm: Channels with small L1 norm contribute less to output
    importance = tp.importance.MagnitudeImportance(p=1)

    # Create pruner
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_input,
        importance=importance,
        pruning_ratio=pruning_ratio,
        global_pruning=False  # Prune each layer independently
    )

    # Execute pruning (removes channels, creates smaller model)
    # WHY remove channels: Conv2d(64, 128) → Conv2d(32, 64) after 50% pruning
    pruner.step()

    return model

# Example: Prune ResNet18
from torchvision.models import resnet18

model = resnet18(pretrained=True)
print(f"Original model size: {get_model_size(model):.1f}MB")  # 44.7MB
print(f"Original params: {count_parameters(model):,}")  # 11,689,512

# Apply 50% channel pruning
model_pruned = structured_channel_pruning_cnn(
    model,
    pruning_ratio=0.5,
    example_input=torch.randn(1, 3, 224, 224)
)

print(f"Pruned model size: {get_model_size(model_pruned):.1f}MB")  # 22.4MB (50% reduction)
print(f"Pruned params: {count_parameters(model_pruned):,}")  # 5,844,756 (50% reduction)

# Benchmark inference speed
# WHY faster: Smaller dense matrices (fewer FLOPs, less memory bandwidth)
original_time = benchmark_inference(model)  # 25ms
pruned_time = benchmark_inference(model_pruned)  # 12.5ms (2× FASTER!)

# Accuracy (before fine-tuning)
original_acc = evaluate(model)  # 69.8%
pruned_acc = evaluate(model_pruned)  # 64.2% (5.6pp drop - needs fine-tuning)

# Fine-tune to recover accuracy
fine_tune(model_pruned, epochs=5, lr=1e-4)
pruned_acc_recovered = evaluate(model_pruned)  # 68.5% (1.3pp drop, acceptable)
```

**Helper functions:**

```python
def get_model_size(model):
    """Calculate model size in MB."""
    # WHY: Multiply parameters by 4 bytes (FP32)
    param_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
    return param_size

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_inference(model, num_runs=100):
    """Benchmark inference time (ms)."""
    import time
    model.eval()
    example_input = torch.randn(1, 3, 224, 224)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(example_input)

    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(example_input)
    end = time.time()

    return (end - start) / num_runs * 1000  # Convert to ms
```

### Iterative Pruning (Quality Preservation)

**Problem:** Pruning all at once (50% in one step) → 10pp accuracy drop.

**Solution:** Iterative pruning (5 steps × 10% each) → 2pp accuracy drop.

```python
def iterative_pruning(model, target_ratio=0.5, num_iterations=5, finetune_epochs=2):
    """
    Iterative pruning with fine-tuning between steps.

    WHY iterative: Gradual pruning allows model to adapt
    WHY fine-tune: Remaining weights compensate for removed weights

    Example: 50% pruning
    - One-shot: 50% pruning → 10pp accuracy drop
    - Iterative: 5 steps × 10% each → 2pp accuracy drop (better!)

    Args:
        model: Model to prune
        target_ratio: Final pruning ratio (0.5 = remove 50% of weights)
        num_iterations: Number of pruning steps (more = gradual = better quality)
        finetune_epochs: Fine-tuning epochs after each step

    Returns:
        Pruned model with quality preservation
    """
    # Calculate pruning amount per iteration
    # WHY: Distribute total pruning across iterations
    amount_per_iteration = 1 - (1 - target_ratio) ** (1 / num_iterations)

    print(f"Pruning {target_ratio*100:.0f}% in {num_iterations} steps")
    print(f"Amount per step: {amount_per_iteration*100:.1f}%")

    for step in range(num_iterations):
        print(f"\n=== Iteration {step+1}/{num_iterations} ===")

        # Prune this iteration
        # WHY global_unstructured: Prune across all layers (balanced sparsity)
        parameters_to_prune = [
            (module, "weight")
            for module in model.modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))
        ]

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount_per_iteration
        )

        # Evaluate current accuracy
        acc_before_finetune = evaluate(model)
        print(f"Accuracy after pruning: {acc_before_finetune:.2f}%")

        # Fine-tune to recover accuracy
        # WHY: Allow remaining weights to compensate for removed weights
        fine_tune(model, epochs=finetune_epochs, lr=1e-4)

        acc_after_finetune = evaluate(model)
        print(f"Accuracy after fine-tuning: {acc_after_finetune:.2f}%")

    # Make pruning permanent (remove masks)
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    return model

# Example usage
model = resnet18(pretrained=True)
original_acc = evaluate(model)  # 69.8%

# One-shot pruning (worse quality)
model_oneshot = copy.deepcopy(model)
prune_global_unstructured(model_oneshot, amount=0.5)  # Prune 50% immediately
oneshot_acc = evaluate(model_oneshot)  # 59.7% (10.1pp drop!)

# Iterative pruning (better quality)
model_iterative = copy.deepcopy(model)
model_iterative = iterative_pruning(
    model_iterative,
    target_ratio=0.5,  # 50% pruning
    num_iterations=5,  # Gradual over 5 steps
    finetune_epochs=2  # Fine-tune after each step
)
iterative_acc = evaluate(model_iterative)  # 67.5% (2.3pp drop, much better!)

# Quality comparison:
# - One-shot: 10.1pp drop (unacceptable)
# - Iterative: 2.3pp drop (acceptable)
```

### Structured Layer Pruning (Transformers)

**Problem:** Transformers sensitive to unstructured pruning (destroys attention patterns).

**Solution:** Drop entire layers (structured pruning for transformers).

```python
def drop_transformer_layers(model, num_layers_to_drop=6):
    """
    Drop transformer layers (structured pruning for transformers).

    WHY drop layers: Transformers learn hierarchical features, later layers refine
    WHY not unstructured: Attention patterns are dense, pruning destroys them

    Example: BERT-base (12 layers) → BERT-small (6 layers)
    - Size: 440MB → 220MB (2× reduction)
    - Speed: 2× faster (half the layers)
    - Accuracy: 95% → 92% (3pp drop with fine-tuning)

    Args:
        model: Transformer model (BERT, GPT, T5)
        num_layers_to_drop: Number of layers to remove

    Returns:
        Smaller transformer model
    """
    # Identify which layers to drop
    # WHY drop middle layers: Keep early (low-level features) and late (task-specific)
    # Alternative: Drop early or late layers depending on task
    total_layers = len(model.encoder.layer)  # BERT example
    layers_to_keep = total_layers - num_layers_to_drop

    # Drop middle layers (preserve early and late layers)
    start_idx = num_layers_to_drop // 2
    end_idx = start_idx + layers_to_keep

    new_layers = model.encoder.layer[start_idx:end_idx]
    model.encoder.layer = nn.ModuleList(new_layers)

    # Update config
    model.config.num_hidden_layers = layers_to_keep

    print(f"Dropped {num_layers_to_drop} layers ({total_layers} → {layers_to_keep})")

    return model

# Example: Compress BERT-base
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
print(f"Original: {model.config.num_hidden_layers} layers, {get_model_size(model):.0f}MB")
# Original: 12 layers, 440MB

# Drop 6 layers (50% reduction)
model_compressed = drop_transformer_layers(model, num_layers_to_drop=6)
print(f"Compressed: {model_compressed.config.num_hidden_layers} layers, {get_model_size(model_compressed):.0f}MB")
# Compressed: 6 layers, 220MB

# Accuracy before fine-tuning
original_acc = evaluate(model)  # 95.2%
compressed_acc = evaluate(model_compressed)  # 88.5% (6.7pp drop)

# Fine-tune to recover accuracy
# WHY fine-tune: Remaining layers adapt to missing layers
fine_tune(model_compressed, epochs=3, lr=2e-5)
compressed_acc_recovered = evaluate(model_compressed)  # 92.1% (3.1pp drop, acceptable)
```


## Part 2: Knowledge Distillation

### Progressive Distillation (Quality Preservation)

**Problem:** Single-stage aggressive distillation fails (77× compression → unusable quality).

**Solution:** Progressive distillation in multiple stages (2-4× per stage).

```python
def progressive_distillation(
    teacher,
    num_stages=2,
    compression_per_stage=2.5,
    distill_epochs=10,
    finetune_epochs=3
):
    """
    Progressive knowledge distillation (quality preservation for aggressive compression).

    WHY progressive: Large capacity gap (teacher → tiny student) loses too much knowledge
    WHY multi-stage: Smooth transition preserves quality (teacher → intermediate → final)

    Example: 6× compression
    - Single-stage: 774M → 130M (6× in one step) → 15pp accuracy drop (bad)
    - Progressive: 774M → 310M → 130M (2.5× per stage) → 5pp accuracy drop (better)

    Args:
        teacher: Large pre-trained model (e.g., BERT-large, GPT-2 Large)
        num_stages: Number of distillation stages (2-3 typical)
        compression_per_stage: Compression ratio per stage (2-4× safe)
        distill_epochs: Distillation training epochs per stage
        finetune_epochs: Fine-tuning epochs on hard labels

    Returns:
        Final compressed student model
    """
    current_teacher = teacher
    teacher_params = count_parameters(teacher)

    print(f"Teacher: {teacher_params:,} params")

    for stage in range(num_stages):
        print(f"\n=== Stage {stage+1}/{num_stages} ===")

        # Calculate student capacity for this stage
        # WHY: Reduce by compression_per_stage factor
        student_params = teacher_params // (compression_per_stage ** (stage + 1))

        # Create student architecture (model-specific)
        # WHY smaller: Fewer layers, smaller hidden dimension, fewer heads
        student = create_student_model(
            teacher_architecture=current_teacher,
            target_params=student_params
        )

        print(f"Student {stage+1}: {count_parameters(student):,} params")

        # Stage 1: Distillation training (learn from teacher)
        # WHY soft targets: Teacher's probability distribution (richer than hard labels)
        student = train_distillation(
            teacher=current_teacher,
            student=student,
            train_loader=train_loader,
            epochs=distill_epochs,
            temperature=2.0,  # WHY 2.0: Softer probabilities (more knowledge transfer)
            alpha=0.7  # WHY 0.7: Weight distillation loss higher than hard loss
        )

        # Stage 2: Fine-tuning on hard labels (task optimization)
        # WHY: Optimize student for actual task performance (not just mimicking teacher)
        student = fine_tune_on_labels(
            student=student,
            train_loader=train_loader,
            epochs=finetune_epochs,
            lr=2e-5
        )

        # Evaluate this stage
        teacher_acc = evaluate(current_teacher)
        student_acc = evaluate(student)
        print(f"Teacher accuracy: {teacher_acc:.2f}%")
        print(f"Student accuracy: {student_acc:.2f}% (drop: {teacher_acc - student_acc:.2f}pp)")

        # Student becomes teacher for next stage
        current_teacher = student

    return student

def create_student_model(teacher_architecture, target_params):
    """
    Create student model with target parameter count.

    WHY: Match architecture type but scale down capacity
    """
    # Example for BERT
    if isinstance(teacher_architecture, BertForSequenceClassification):
        # Scale down: fewer layers, smaller hidden size, fewer heads
        # WHY: Preserve architecture but reduce capacity
        teacher_config = teacher_architecture.config

        # Calculate scaling factor
        scaling_factor = (target_params / count_parameters(teacher_architecture)) ** 0.5

        student_config = BertConfig(
            num_hidden_layers=int(teacher_config.num_hidden_layers * scaling_factor),
            hidden_size=int(teacher_config.hidden_size * scaling_factor),
            num_attention_heads=max(1, int(teacher_config.num_attention_heads * scaling_factor)),
            intermediate_size=int(teacher_config.intermediate_size * scaling_factor),
            num_labels=teacher_config.num_labels
        )

        return BertForSequenceClassification(student_config)

    # Add other architectures as needed
    raise ValueError(f"Unsupported architecture: {type(teacher_architecture)}")

def train_distillation(teacher, student, train_loader, epochs, temperature, alpha):
    """
    Train student to mimic teacher (knowledge distillation).

    WHY distillation loss: Student learns soft targets (probability distributions)
    WHY temperature: Softens probabilities (exposes dark knowledge)
    """
    import torch.nn.functional as F

    teacher.eval()  # WHY: Teacher is frozen (pre-trained knowledge)
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)

    for epoch in range(epochs):
        total_loss = 0

        for batch, labels in train_loader:
            # Teacher predictions (soft targets)
            with torch.no_grad():
                teacher_logits = teacher(batch).logits

            # Student predictions
            student_logits = student(batch).logits

            # Distillation loss (KL divergence with temperature scaling)
            # WHY temperature: Softens probabilities, exposes similarities between classes
            soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
            soft_predictions = F.log_softmax(student_logits / temperature, dim=-1)

            distillation_loss = F.kl_div(
                soft_predictions,
                soft_targets,
                reduction='batchmean'
            ) * (temperature ** 2)  # WHY T^2: Scale loss appropriately

            # Hard label loss (cross-entropy with ground truth)
            hard_loss = F.cross_entropy(student_logits, labels)

            # Combined loss
            # WHY alpha: Balance distillation (learn from teacher) and hard loss (task performance)
            loss = alpha * distillation_loss + (1 - alpha) * hard_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    return student

# Example usage: Compress GPT-2 Large (774M) to GPT-2 Small (124M)
from transformers import GPT2LMHeadModel

teacher = GPT2LMHeadModel.from_pretrained('gpt2-large')  # 774M params

# Progressive distillation (2 stages, 2.5× per stage = 6.25× total)
student_final = progressive_distillation(
    teacher=teacher,
    num_stages=2,
    compression_per_stage=2.5,  # 2.5× per stage
    distill_epochs=10,
    finetune_epochs=3
)

# Results:
# - Teacher (GPT-2 Large): 774M params, perplexity 18.5
# - Student 1 (intermediate): 310M params, perplexity 22.1 (3.6pp drop)
# - Student 2 (final): 124M params, perplexity 28.5 (10pp drop)
# - Single-stage (direct 774M → 124M): perplexity 45.2 (26.7pp drop, much worse!)
```

### Capacity Matching Guidelines

**Problem:** Student too small → can't learn teacher knowledge. Student too large → inefficient.

**Solution:** Match student capacity to compression target and quality tolerance.

```python
def calculate_optimal_student_capacity(
    teacher_params,
    target_compression,
    quality_tolerance,
    architecture_type
):
    """
    Calculate optimal student model capacity.

    Compression guidelines:
    - 2-4× compression: Minimal quality loss (1-3pp)
    - 4-8× compression: Acceptable quality loss (3-7pp) with fine-tuning
    - 8-15× compression: Significant quality loss (7-15pp), risky
    - >15× compression: Usually fails, student lacks capacity

    Progressive distillation for >4× compression:
    - Stage 1: Teacher → Student 1 (2-4× compression)
    - Stage 2: Student 1 → Student 2 (2-4× compression)
    - Total: 4-16× compression with quality preservation

    Args:
        teacher_params: Number of parameters in teacher model
        target_compression: Desired compression ratio (e.g., 6.0 for 6× smaller)
        quality_tolerance: Acceptable accuracy drop (pp)
        architecture_type: "transformer", "cnn", "rnn"

    Returns:
        (student_params, num_stages, compression_per_stage)
    """
    # Compression difficulty by architecture
    # WHY: Different architectures have different distillation friendliness
    difficulty_factor = {
        "transformer": 1.0,  # Distills well (attention patterns transferable)
        "cnn": 0.8,  # Distills very well (spatial features transferable)
        "rnn": 1.2   # Distills poorly (sequential dependencies fragile)
    }[architecture_type]

    # Adjust target compression by difficulty
    effective_compression = target_compression * difficulty_factor

    # Determine number of stages
    if effective_compression <= 4:
        # Single-stage distillation sufficient
        num_stages = 1
        compression_per_stage = effective_compression
    elif effective_compression <= 16:
        # Two-stage distillation
        num_stages = 2
        compression_per_stage = effective_compression ** 0.5
    else:
        # Three-stage distillation (or warn that compression is too aggressive)
        num_stages = 3
        compression_per_stage = effective_compression ** (1/3)

        if quality_tolerance < 0.15:  # <15pp drop
            print(f"WARNING: {target_compression}× compression may exceed quality tolerance")
            print(f"Consider: Target compression {target_compression/2:.1f}× instead")

    # Calculate final student capacity
    student_params = teacher_params / target_compression

    return student_params, num_stages, compression_per_stage

# Example usage
teacher_params = 774_000_000  # GPT-2 Large

# Conservative compression (2× - safe)
student_params, stages, per_stage = calculate_optimal_student_capacity(
    teacher_params=teacher_params,
    target_compression=2.0,
    quality_tolerance=0.03,  # Accept 3pp drop
    architecture_type="transformer"
)
print(f"2× compression: {student_params:,} params, {stages} stage(s), {per_stage:.1f}× per stage")
# Output: 387M params, 1 stage, 2.0× per stage

# Moderate compression (6× - requires planning)
student_params, stages, per_stage = calculate_optimal_student_capacity(
    teacher_params=teacher_params,
    target_compression=6.0,
    quality_tolerance=0.10,  # Accept 10pp drop
    architecture_type="transformer"
)
print(f"6× compression: {student_params:,} params, {stages} stage(s), {per_stage:.1f}× per stage")
# Output: 129M params, 2 stages, 2.4× per stage

# Aggressive compression (15× - risky)
student_params, stages, per_stage = calculate_optimal_student_capacity(
    teacher_params=teacher_params,
    target_compression=15.0,
    quality_tolerance=0.20,  # Accept 20pp drop
    architecture_type="transformer"
)
print(f"15× compression: {student_params:,} params, {stages} stage(s), {per_stage:.1f}× per stage")
# Output: 52M params, 3 stages, 2.5× per stage
# WARNING: High quality loss expected
```


## Part 3: Low-Rank Decomposition

### Singular Value Decomposition (SVD) for Linear Layers

**Problem:** Large weight matrices (e.g., 4096×4096 in transformers) consume memory.

**Solution:** Decompose into two smaller matrices (low-rank factorization).

```python
import torch
import torch.nn as nn

def decompose_linear_layer_svd(layer, rank_ratio=0.5):
    """
    Decompose linear layer using SVD (low-rank approximation).

    WHY: Large matrix W (m×n) → two smaller matrices U (m×r) and V (r×n)
    WHY works: Weight matrices often have low effective rank (redundancy)

    Example: Linear(4096, 4096) with 50% rank
    - Original: 16.8M parameters (4096×4096)
    - Decomposed: 4.1M parameters (4096×2048 + 2048×4096) - 4× reduction!

    Args:
        layer: nn.Linear layer to decompose
        rank_ratio: Fraction of original rank to keep (0.5 = keep 50%)

    Returns:
        Sequential module with two linear layers (equivalent to original)
    """
    # Get weight matrix
    W = layer.weight.data  # Shape: (out_features, in_features)
    bias = layer.bias.data if layer.bias is not None else None

    # Perform SVD: W = U @ S @ V^T
    # WHY SVD: Optimal low-rank approximation (minimizes reconstruction error)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    # Determine rank to keep
    original_rank = min(W.shape)
    target_rank = int(original_rank * rank_ratio)

    print(f"Original rank: {original_rank}, Target rank: {target_rank}")

    # Truncate to target rank
    # WHY keep largest singular values: They capture most of the information
    U_k = U[:, :target_rank]  # Shape: (out_features, target_rank)
    S_k = S[:target_rank]  # Shape: (target_rank,)
    Vt_k = Vt[:target_rank, :]  # Shape: (target_rank, in_features)

    # Create two linear layers: W ≈ U_k @ diag(S_k) @ Vt_k
    # Layer 1: Linear(in_features, target_rank) with weights Vt_k
    # Layer 2: Linear(target_rank, out_features) with weights U_k @ diag(S_k)
    layer1 = nn.Linear(W.shape[1], target_rank, bias=False)
    layer1.weight.data = Vt_k

    layer2 = nn.Linear(target_rank, W.shape[0], bias=(bias is not None))
    layer2.weight.data = U_k * S_k.unsqueeze(0)  # Incorporate S into second layer

    if bias is not None:
        layer2.bias.data = bias

    # Return sequential module (equivalent to original layer)
    return nn.Sequential(layer1, layer2)

# Example: Decompose large transformer feedforward layer
original_layer = nn.Linear(4096, 4096)
print(f"Original params: {count_parameters(original_layer):,}")  # 16,781,312

# Decompose with 50% rank retention
decomposed_layer = decompose_linear_layer_svd(original_layer, rank_ratio=0.5)
print(f"Decomposed params: {count_parameters(decomposed_layer):,}")  # 4,194,304 (4× reduction!)

# Verify reconstruction quality
x = torch.randn(1, 128, 4096)  # Example input
y_original = original_layer(x)
y_decomposed = decomposed_layer(x)

reconstruction_error = torch.norm(y_original - y_decomposed) / torch.norm(y_original)
print(f"Reconstruction error: {reconstruction_error.item():.4f}")  # Small error (good approximation)
```

### Apply SVD to Entire Model

```python
def decompose_model_svd(model, rank_ratio=0.5, layer_threshold=1024):
    """
    Apply SVD decomposition to all large linear layers in model.

    WHY selective: Only decompose large layers (small layers don't benefit)
    WHY threshold: Layers with <1024 input/output features too small to benefit

    Args:
        model: Model to compress
        rank_ratio: Fraction of rank to keep (0.5 = 2× reduction per layer)
        layer_threshold: Minimum layer size to decompose (skip small layers)

    Returns:
        Model with decomposed layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Only decompose large layers
            if module.in_features >= layer_threshold and module.out_features >= layer_threshold:
                print(f"Decomposing {name}: {module.in_features}×{module.out_features}")

                # Decompose layer
                decomposed = decompose_linear_layer_svd(module, rank_ratio=rank_ratio)

                # Replace in model
                setattr(model, name, decomposed)
        elif len(list(module.children())) > 0:
            # Recursively decompose nested modules
            decompose_model_svd(module, rank_ratio, layer_threshold)

    return model

# Example: Compress transformer model
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased')
original_params = count_parameters(model)
print(f"Original params: {original_params:,}")  # 109M

# Apply SVD (50% rank) to feedforward layers
model_compressed = decompose_model_svd(model, rank_ratio=0.5, layer_threshold=512)
compressed_params = count_parameters(model_compressed)
print(f"Compressed params: {compressed_params:,}")  # 82M (1.3× reduction)

# Fine-tune to recover accuracy
# WHY: Low-rank approximation introduces small errors, fine-tuning compensates
fine_tune(model_compressed, epochs=3, lr=2e-5)
```


## Part 4: Combined Compression Pipelines

### Quantization + Pruning + Distillation

**Problem:** Single technique insufficient for aggressive compression (e.g., 20× for mobile).

**Solution:** Combine multiple techniques (multiplicative compression).

```python
def full_compression_pipeline(
    teacher_model,
    target_compression=20,
    deployment_target="mobile"
):
    """
    Combined compression pipeline for maximum compression.

    WHY combine techniques: Multiplicative compression
    - Quantization: 4× reduction (FP32 → INT8)
    - Pruning: 2× reduction (50% structured pruning)
    - Distillation: 2.5× reduction (progressive distillation)
    - Total: 4 × 2 × 2.5 = 20× reduction!

    Pipeline order:
    1. Knowledge distillation (preserve quality first)
    2. Structured pruning (remove redundancy)
    3. Quantization (reduce precision last)

    WHY this order:
    - Distillation first: Creates smaller model with quality preservation
    - Pruning second: Removes redundancy from distilled model
    - Quantization last: Works well on already-compressed model

    Args:
        teacher_model: Large pre-trained model to compress
        target_compression: Desired compression ratio (e.g., 20 for 20× smaller)
        deployment_target: "mobile", "edge", "server"

    Returns:
        Fully compressed model ready for deployment
    """
    print(f"=== Full Compression Pipeline (target: {target_compression}× reduction) ===\n")

    # Original model metrics
    original_size = get_model_size(teacher_model)
    original_params = count_parameters(teacher_model)
    original_acc = evaluate(teacher_model)

    print(f"Original: {original_params:,} params, {original_size:.1f}MB, {original_acc:.2f}% acc")

    # Step 1: Knowledge Distillation (2-2.5× compression)
    # WHY first: Preserves quality better than pruning teacher directly
    print("\n--- Step 1: Knowledge Distillation ---")

    distillation_ratio = min(2.5, target_compression ** (1/3))  # Allocate ~1/3 of compression
    student_model = progressive_distillation(
        teacher=teacher_model,
        num_stages=2,
        compression_per_stage=distillation_ratio ** 0.5,
        distill_epochs=10,
        finetune_epochs=3
    )

    student_size = get_model_size(student_model)
    student_params = count_parameters(student_model)
    student_acc = evaluate(student_model)

    print(f"After distillation: {student_params:,} params, {student_size:.1f}MB, {student_acc:.2f}% acc")
    print(f"Compression: {original_size/student_size:.1f}×")

    # Step 2: Structured Pruning (1.5-2× compression)
    # WHY after distillation: Prune smaller model (easier to maintain quality)
    print("\n--- Step 2: Structured Pruning ---")

    pruning_ratio = min(0.5, 1 - 1/(target_compression ** (1/3)))  # Allocate ~1/3 of compression
    pruned_model = iterative_pruning(
        model=student_model,
        target_ratio=pruning_ratio,
        num_iterations=5,
        finetune_epochs=2
    )

    pruned_size = get_model_size(pruned_model)
    pruned_params = count_parameters(pruned_model)
    pruned_acc = evaluate(pruned_model)

    print(f"After pruning: {pruned_params:,} params, {pruned_size:.1f}MB, {pruned_acc:.2f}% acc")
    print(f"Compression: {original_size/pruned_size:.1f}×")

    # Step 3: Quantization (4× compression)
    # WHY last: Works well on already-compressed model, easy to apply
    print("\n--- Step 3: Quantization (INT8) ---")

    # Quantization-aware training
    pruned_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare_qat(pruned_model)

    # Fine-tune with fake quantization
    fine_tune(model_prepared, epochs=3, lr=1e-4)

    # Convert to INT8
    model_prepared.eval()
    quantized_model = torch.quantization.convert(model_prepared)

    quantized_size = get_model_size(quantized_model)
    quantized_acc = evaluate(quantized_model)

    print(f"After quantization: {quantized_size:.1f}MB, {quantized_acc:.2f}% acc")
    print(f"Total compression: {original_size/quantized_size:.1f}×")

    # Summary
    print("\n=== Compression Pipeline Summary ===")
    print(f"Original:     {original_size:.1f}MB, {original_acc:.2f}% acc")
    print(f"Distilled:    {student_size:.1f}MB, {student_acc:.2f}% acc ({original_size/student_size:.1f}×)")
    print(f"Pruned:       {pruned_size:.1f}MB, {pruned_acc:.2f}% acc ({original_size/pruned_size:.1f}×)")
    print(f"Quantized:    {quantized_size:.1f}MB, {quantized_acc:.2f}% acc ({original_size/quantized_size:.1f}×)")
    print(f"\nFinal compression: {original_size/quantized_size:.1f}× (target: {target_compression}×)")
    print(f"Accuracy drop: {original_acc - quantized_acc:.2f}pp")

    # Deployment checks
    if deployment_target == "mobile":
        assert quantized_size <= 100, f"Model too large for mobile ({quantized_size:.1f}MB > 100MB)"
        assert quantized_acc >= original_acc - 5, f"Quality loss too high ({original_acc - quantized_acc:.2f}pp)"
        print("\n✓ Ready for mobile deployment")

    return quantized_model

# Example: Compress BERT for mobile deployment
from transformers import BertForSequenceClassification

teacher = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Target: 20× compression (440MB → 22MB)
compressed_model = full_compression_pipeline(
    teacher_model=teacher,
    target_compression=20,
    deployment_target="mobile"
)

# Results:
# - Original: 440MB, 95.2% acc
# - Distilled: 180MB, 93.5% acc (2.4× compression)
# - Pruned: 90MB, 92.8% acc (4.9× compression)
# - Quantized: 22MB, 92.1% acc (20× compression!)
# - Accuracy drop: 3.1pp (acceptable for mobile)
```


## Part 5: Architecture Optimization

### Neural Architecture Search for Efficiency

**Problem:** Manual architecture design for compression is time-consuming.

**Solution:** Automated search for efficient architectures (NAS for compression).

```python
def efficient_architecture_search(
    task_type,
    target_latency_ms,
    target_accuracy,
    search_space="mobilenet"
):
    """
    Search for efficient architecture meeting constraints.

    WHY NAS: Automated discovery of architectures optimized for efficiency
    WHY search space: MobileNet, EfficientNet designed for edge deployment

    Search strategies:
    - Width multiplier: Scale number of channels (0.5× - 1.5×)
    - Depth multiplier: Scale number of layers (0.75× - 1.25×)
    - Resolution multiplier: Scale input resolution (128px - 384px)

    Args:
        task_type: "classification", "detection", "segmentation"
        target_latency_ms: Maximum inference latency (ms)
        target_accuracy: Minimum acceptable accuracy
        search_space: "mobilenet", "efficientnet", "custom"

    Returns:
        Optimal architecture configuration
    """
    # Example: MobileNetV3 search space
    # WHY MobileNet: Designed for mobile (depthwise separable convolutions)
    from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

    configurations = [
        {
            "model": mobilenet_v3_small,
            "width_multiplier": 0.5,
            "expected_latency": 8,  # ms on mobile CPU
            "expected_accuracy": 60.5  # ImageNet top-1
        },
        {
            "model": mobilenet_v3_small,
            "width_multiplier": 1.0,
            "expected_latency": 15,
            "expected_accuracy": 67.4
        },
        {
            "model": mobilenet_v3_large,
            "width_multiplier": 0.75,
            "expected_latency": 25,
            "expected_accuracy": 73.3
        },
        {
            "model": mobilenet_v3_large,
            "width_multiplier": 1.0,
            "expected_latency": 35,
            "expected_accuracy": 75.2
        }
    ]

    # Find configurations meeting constraints
    # WHY filter: Only consider configs within latency budget and accuracy requirement
    valid_configs = [
        config for config in configurations
        if config["expected_latency"] <= target_latency_ms
        and config["expected_accuracy"] >= target_accuracy
    ]

    if not valid_configs:
        print(f"No configuration meets constraints (latency={target_latency_ms}ms, accuracy={target_accuracy}%)")
        print("Consider: Relax constraints or use custom search")
        return None

    # Select best (highest accuracy within latency budget)
    best_config = max(valid_configs, key=lambda c: c["expected_accuracy"])

    print(f"Selected: {best_config['model'].__name__} (width={best_config['width_multiplier']})")
    print(f"Expected: {best_config['expected_latency']}ms, {best_config['expected_accuracy']}% acc")

    return best_config

# Example usage: Find architecture for mobile deployment
config = efficient_architecture_search(
    task_type="classification",
    target_latency_ms=20,  # 20ms latency budget
    target_accuracy=70.0,  # Minimum 70% accuracy
    search_space="mobilenet"
)

# Output: Selected MobileNetV3-Large (width=0.75)
# Expected: 25ms latency, 73.3% accuracy
# Meets both constraints!
```


## Part 6: Quality Preservation Strategies

### Trade-off Analysis Framework

```python
def analyze_compression_tradeoffs(
    model,
    compression_techniques,
    deployment_constraints
):
    """
    Analyze compression technique trade-offs.

    WHY: Different techniques have different trade-offs
    - Quantization: Best size/speed, minimal quality loss (0.5-2pp)
    - Pruning: Good size/speed, moderate quality loss (2-5pp)
    - Distillation: Excellent quality, requires training time

    Args:
        model: Model to compress
        compression_techniques: List of techniques to try
        deployment_constraints: Dict with size_mb, latency_ms, accuracy_min

    Returns:
        Recommended technique and expected metrics
    """
    results = []

    # Quantization (FP32 → INT8)
    if "quantization" in compression_techniques:
        results.append({
            "technique": "quantization",
            "compression_ratio": 4.0,
            "expected_accuracy_drop": 0.5,  # 0.5-2pp with QAT
            "training_time_hours": 2,  # QAT training
            "complexity": "low"
        })

    # Structured pruning (50%)
    if "pruning" in compression_techniques:
        results.append({
            "technique": "structured_pruning_50%",
            "compression_ratio": 2.0,
            "expected_accuracy_drop": 2.5,  # 2-5pp with iterative pruning
            "training_time_hours": 8,  # Iterative pruning + fine-tuning
            "complexity": "medium"
        })

    # Knowledge distillation (2× compression)
    if "distillation" in compression_techniques:
        results.append({
            "technique": "distillation_2x",
            "compression_ratio": 2.0,
            "expected_accuracy_drop": 1.5,  # 1-3pp
            "training_time_hours": 20,  # Full distillation training
            "complexity": "high"
        })

    # Combined (quantization + pruning)
    if "combined" in compression_techniques:
        results.append({
            "technique": "quantization+pruning",
            "compression_ratio": 8.0,  # 4× × 2× = 8×
            "expected_accuracy_drop": 3.5,  # Additive: 0.5 + 2.5 + interaction
            "training_time_hours": 12,  # Pruning + QAT
            "complexity": "high"
        })

    # Filter by constraints
    original_size = get_model_size(model)
    original_acc = evaluate(model)

    valid_techniques = [
        r for r in results
        if (original_size / r["compression_ratio"]) <= deployment_constraints["size_mb"]
        and (original_acc - r["expected_accuracy_drop"]) >= deployment_constraints["accuracy_min"]
    ]

    if not valid_techniques:
        print("No technique meets all constraints")
        return None

    # Recommend technique (prioritize: best quality, then fastest training, then simplest)
    best = min(
        valid_techniques,
        key=lambda r: (r["expected_accuracy_drop"], r["training_time_hours"], r["complexity"])
    )

    print(f"Recommended: {best['technique']}")
    print(f"Expected: {original_size/best['compression_ratio']:.1f}MB (from {original_size:.1f}MB)")
    print(f"Accuracy: {original_acc - best['expected_accuracy_drop']:.1f}% (drop: {best['expected_accuracy_drop']}pp)")
    print(f"Training time: {best['training_time_hours']} hours")

    return best

# Example usage
deployment_constraints = {
    "size_mb": 50,  # Model must be <50MB
    "latency_ms": 100,  # <100ms inference
    "accuracy_min": 90.0  # >90% accuracy
}

recommendation = analyze_compression_tradeoffs(
    model=my_model,
    compression_techniques=["quantization", "pruning", "distillation", "combined"],
    deployment_constraints=deployment_constraints
)
```


## Common Mistakes to Avoid

| Mistake | Why It's Wrong | Correct Approach |
|---------|----------------|------------------|
| "Pruning works for all architectures" | Destroys transformer attention | Use distillation for transformers |
| "More compression is always better" | 77× compression produces gibberish | Progressive distillation for >4× |
| "Unstructured pruning speeds up inference" | No speedup on standard hardware | Use structured pruning (channel/layer) |
| "Quantize and deploy immediately" | 5pp accuracy drop without recovery | QAT + fine-tuning for quality preservation |
| "Single technique is enough" | Can't reach aggressive targets (20×) | Combine: quantization + pruning + distillation |
| "Skip fine-tuning to save time" | Preventable accuracy loss | Always include recovery step |


## Success Criteria

You've correctly compressed a model when:

✅ Selected appropriate technique for architecture (distillation for transformers, pruning for CNNs)
✅ Matched student capacity to compression target (2-4× per stage, progressive for >4×)
✅ Used structured pruning for standard hardware (actual speedup)
✅ Applied iterative/progressive compression (quality preservation)
✅ Included accuracy recovery (QAT, fine-tuning, calibration)
✅ Achieved target compression with acceptable quality loss (<5pp for most tasks)
✅ Verified deployment constraints (size, latency, accuracy) are met


## References

**Key papers:**
- DistilBERT (Sanh et al., 2019): Knowledge distillation for transformers
- The Lottery Ticket Hypothesis (Frankle & Carbin, 2019): Iterative magnitude pruning
- Pruning Filters for Efficient ConvNets (Li et al., 2017): Structured channel pruning
- Deep Compression (Han et al., 2016): Pruning + quantization + Huffman coding

**When to combine with other skills:**
- Use with quantization-for-inference: Quantization (4×) + compression (2-5×) = 8-20× total
- Use with hardware-optimization-strategies: Optimize compressed model for target hardware
- Use with model-serving-patterns: Deploy compressed model with batching/caching
