---
name: llm-finetuning-strategies
description: Master LLM fine-tuning strategies including when to fine-tune vs prompt engineering, LoRA vs full fine-tuning, dataset preparation, hyperparameter selection, and evaluation. Use when considering fine-tuning, preparing datasets, or optimizing fine-tuned models.
---

# LLM Fine-Tuning Strategies

## Context

You're considering fine-tuning an LLM or debugging a fine-tuning process. Common mistakes:
- **Fine-tuning when prompts would work** (unnecessary cost/time)
- **Full fine-tuning instead of LoRA** (100× less efficient)
- **Poor dataset quality** (garbage in, garbage out)
- **Wrong hyperparameters** (catastrophic forgetting)
- **No validation strategy** (overfitting undetected)

**This skill provides effective fine-tuning strategies: when to fine-tune, efficient methods (LoRA), data quality, hyperparameters, and evaluation.**

---

## Decision Tree: Prompt Engineering vs Fine-Tuning

**Start with prompt engineering. Fine-tuning is last resort.**

### Step 1: Try Prompt Engineering

```python
# System message + few-shot examples
system = """
You are a {role} with {characteristics}.
{guidelines}
"""

few_shot = [
    # 3-5 examples of desired behavior
]

# Test quality
quality = evaluate(system, few_shot, test_set)
```

**If quality ≥ 90%:** ✅ STOP. Use prompts (no fine-tuning needed)

**If quality < 90%:** Continue to Step 2

### Step 2: Optimize Prompts

- Add more examples (5-10)
- Add chain-of-thought
- Specify output format more clearly
- Try different system messages
- Use temperature=0 for consistency

**If quality ≥ 90%:** ✅ STOP. Use optimized prompts

**If quality < 90%:** Continue to Step 3

### Step 3: Consider Fine-Tuning

**Fine-tune when:**

✅ **Prompts fail** (quality < 90% after optimization)
✅ **Have 1000+ examples** (minimum for meaningful fine-tuning)
✅ **Need consistency** (can't rely on prompt variations)
✅ **Reduce latency** (shorter prompts → faster inference)
✅ **Teach new capability** (not in base model)

**Don't fine-tune for:**

❌ **Tone/style matching** (use system message)
❌ **Output formatting** (use format specification in prompt)
❌ **Few examples** (< 100 examples insufficient)
❌ **Quick experiments** (prompts iterate faster)
❌ **Recent information** (use RAG, not fine-tuning)

---

## When to Fine-Tune: Detailed Criteria

### Criterion 1: Task Complexity

**Simple tasks (prompt engineering):**
- Classification (sentiment, category)
- Extraction (entities, dates, names)
- Formatting (JSON, CSV conversion)
- Tone matching (company voice)

**Complex tasks (consider fine-tuning):**
- Multi-step reasoning (not in base model)
- Domain-specific language (medical, legal)
- Consistent complex behavior (100+ edge cases)
- New capabilities (teach entirely new skill)

### Criterion 2: Dataset Size

```
< 100 examples: Prompts only (insufficient for fine-tuning)
100-1000: Prompts preferred (fine-tuning risky - overfitting)
1000-10k: Fine-tuning viable if prompts fail
> 10k: Fine-tuning effective
```

### Criterion 3: Cost-Benefit

**Prompt engineering:**
- Cost: $0 (just dev time)
- Time: Minutes to hours (fast iteration)
- Maintenance: Easy (just update prompt)

**Fine-tuning:**
- Cost: $100-1000+ (compute + data prep)
- Time: Days to weeks (data prep + training + eval)
- Maintenance: Hard (need retraining for updates)

**ROI calculation:**
```python
# Prompt engineering cost
prompt_dev_hours = 4
hourly_rate = 100
prompt_cost = 4 * 100 = $400

# Fine-tuning cost
data_prep_hours = 40
training_cost = 500
total_ft_cost = 40 * 100 + 500 = $4,500

# Cost ratio: Fine-tuning is 11× more expensive
# Only worth it if quality improvement > 10%
```

### Criterion 4: Performance Requirements

**Quality:**
- Need 90-95%: Prompts usually sufficient
- Need 95-98%: Fine-tuning may help
- Need 98%+: Fine-tuning + careful data curation

**Latency:**
- > 1 second acceptable: Prompts fine (long prompts OK)
- 200-1000ms: Fine-tuning may help (reduce prompt size)
- < 200ms: Fine-tuning + optimization required

**Consistency:**
- Variable outputs acceptable: Prompts OK (temperature > 0)
- High consistency needed: Prompts (temperature=0) or fine-tuning
- Perfect consistency: Fine-tuning + validation

---

## Fine-Tuning Methods

### 1. Full Fine-Tuning

**Updates all model parameters.**

**Pros:**
- Maximum flexibility (can change any behavior)
- Best quality (when you have massive data)

**Cons:**
- Expensive (7B model = 28GB memory for weights alone)
- Slow (hours to days)
- Risk of catastrophic forgetting
- Hard to merge multiple fine-tunes

**When to use:**
- Massive dataset (100k+ examples)
- Fundamental behavior change needed
- Have large compute resources (multi-GPU)

**Memory requirements:**
```python
# 7B parameter model (FP32)
weights = 7B * 4 bytes = 28 GB
gradients = 28 GB
optimizer_states = 56 GB (Adam: 2× weights)
activations = ~8 GB (batch_size=8)
total = 120 GB  # Need multi-GPU!
```

### 2. LoRA (Low-Rank Adaptation)

**Freezes base model, trains small adapter matrices.**

**How it works:**
```
Original linear layer: W (d × k)
LoRA: W + (A × B)
  where A (d × r), B (r × k), r << d,k

Example:
W: 4096 × 4096 = 16.7M parameters
A: 4096 × 8 = 32K parameters
B: 8 × 4096 = 32K parameters
A + B = 64K parameters (0.4% of original!)
```

**Pros:**
- Extremely efficient (1% of parameters)
- Fast training (10× faster than full FT)
- Low memory (fits single GPU)
- Easy to merge multiple LoRAs
- No catastrophic forgetting (base model frozen)

**Cons:**
- Slightly lower capacity than full FT (99% quality usually)
- Need to keep base model + adapters

**When to use:**
- 99% of fine-tuning cases
- Limited compute (single GPU)
- Fast iteration needed
- Multiple tasks (train separate LoRAs, swap as needed)

**Configuration:**
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,  # Rank (4-16 typical, higher = more capacity)
    lora_alpha=32,  # Scaling (usually 2× rank)
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, config)
print(model.print_trainable_parameters())
# trainable params: 8.4M || all params: 7B || trainable%: 0.12%
```

**Rank selection:**
```
r=4: Minimal (fast, low capacity) - simple tasks
r=8: Standard (balanced) - most tasks
r=16: High capacity (slower, better quality) - complex tasks
r=32+: Approaching full FT quality (diminishing returns)

Start with r=8, increase only if quality insufficient
```

### 3. QLoRA (Quantized LoRA)

**LoRA + 4-bit quantization of base model.**

**Pros:**
- Extremely memory efficient (4× less than LoRA)
- 7B model fits on 16GB GPU
- Same quality as LoRA

**Cons:**
- Slower than LoRA (quantization overhead)
- More complex setup

**When to use:**
- Limited GPU memory (< 24GB)
- Large models on consumer GPUs
- Cost optimization (cheaper GPUs)

**Setup:**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Then add LoRA as usual
model = get_peft_model(model, lora_config)
```

**Memory comparison:**
```
Method         | 7B Model | 13B Model | 70B Model
---------------|----------|-----------|----------
Full FT        | 120 GB   | 200 GB    | 1000 GB
LoRA           | 40 GB    | 60 GB     | 300 GB
QLoRA          | 12 GB    | 20 GB     | 80 GB
```

### Method Selection:

```python
if gpu_memory < 24:
    use_qlora()
elif gpu_memory < 80:
    use_lora()
elif have_massive_data and multi_gpu_cluster:
    use_full_finetuning()
else:
    use_lora()  # Default choice
```

---

## Dataset Preparation

**Quality > Quantity. 1,000 clean examples > 10,000 noisy examples.**

### 1. Data Collection

**Good sources:**
- Human-labeled data (gold standard)
- Curated conversations (high-quality)
- Expert-written examples
- Validated user interactions

**Bad sources:**
- Raw logs (errors, incomplete, noise)
- Scraped data (quality varies wildly)
- Automated generation (may have artifacts)
- Untested user inputs (edge cases, adversarial)

### 2. Data Cleaning

```python
def clean_dataset(raw_data):
    clean = []

    for example in raw_data:
        # Filter 1: Remove errors
        if any(err in example for err in ['error', 'exception', 'failed']):
            continue

        # Filter 2: Length checks
        if len(example['input']) < 10 or len(example['output']) < 10:
            continue  # Too short
        if len(example['input']) > 2000 or len(example['output']) > 2000:
            continue  # Too long (may be malformed)

        # Filter 3: Completeness
        if not example['output'].strip().endswith(('.', '!', '?')):
            continue  # Incomplete response

        # Filter 4: Language check
        if not is_valid_language(example['output']):
            continue  # Gibberish or wrong language

        # Filter 5: Duplicates
        if is_duplicate(example, clean):
            continue

        clean.append(example)

    return clean

cleaned = clean_dataset(raw_data)
print(f"Filtered: {len(raw_data)} → {len(cleaned)}")
# Example: 10,000 → 3,000 (but high quality!)
```

### 3. Manual Validation

**Critical step: Spot check 100+ random examples.**

```python
import random

sample = random.sample(cleaned, min(100, len(cleaned)))

for i, ex in enumerate(sample):
    print(f"\n--- Example {i+1}/100 ---")
    print(f"Input: {ex['input']}")
    print(f"Output: {ex['output']}")

    response = input("Quality (good/bad/skip)? ")
    if response == 'bad':
        # Investigate pattern, add filtering rule
        print("Why bad?")
        reason = input()
        # Update filtering logic
```

**What to check:**
- ☐ Output is correct and complete
- ☐ Output matches desired format/style
- ☐ No errors or hallucinations
- ☐ Appropriate length
- ☐ Natural language (not robotic)
- ☐ Consistent with other examples

### 4. Dataset Format

**OpenAI format (for GPT fine-tuning):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

**Hugging Face format:**
```python
from datasets import Dataset

data = {
    'input': ["question 1", "question 2", ...],
    'output': ["answer 1", "answer 2", ...]
}

dataset = Dataset.from_dict(data)
```

### 5. Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

# 70% train, 15% val, 15% test
train, temp = train_test_split(data, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
# Example: Train: 2100, Val: 450, Test: 450

# Stratified split for imbalanced data
train, temp = train_test_split(
    data, test_size=0.3, stratify=data['label'], random_state=42
)
```

**Split guidelines:**
- Minimum validation: 100 examples
- Minimum test: 100 examples
- Large datasets (> 10k): 80/10/10 split
- Small datasets (< 5k): 70/15/15 split

### 6. Data Augmentation (Optional)

**When you need more data:**

```python
# Paraphrasing
"What's the weather?" → "How's the weather today?"

# Back-translation
English → French → English (introduces variation)

# Synthetic generation (use carefully!)
few_shot_examples = [...]
new_examples = llm.generate(
    f"Generate 10 examples similar to: {few_shot_examples}"
)
# ALWAYS manually validate synthetic data!
```

**Warning:** Synthetic data can introduce artifacts. Always validate!

---

## Hyperparameters

### Learning Rate

**Most critical hyperparameter.**

```python
# Pre-training LR: 1e-3 to 3e-4
# Fine-tuning LR: 100-1000× smaller!

training_args = TrainingArguments(
    learning_rate=1e-5,  # Start here for 7B models
    # Or even more conservative:
    learning_rate=1e-6,  # For larger models or small datasets
)
```

**Guidelines:**
```
Model size     | Pre-train LR | Fine-tune LR
---------------|--------------|-------------
1B params      | 3e-4         | 3e-5 to 1e-5
7B params      | 3e-4         | 1e-5 to 1e-6
13B params     | 2e-4         | 5e-6 to 1e-6
70B+ params    | 1e-4         | 1e-6 to 1e-7

Rule: Fine-tune LR ≈ Pre-train LR / 100
```

**LR scheduling:**
```python
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,  # Gradual LR increase (10% of training)
    num_training_steps=total_steps
)
```

**Signs of wrong LR:**

Too high (LR > 1e-4):
- Training loss oscillates wildly
- Model generates gibberish
- Catastrophic forgetting (fails on general tasks)

Too low (LR < 1e-7):
- Training loss barely decreases
- Model doesn't adapt to new data
- Very slow convergence

### Epochs

```python
training_args = TrainingArguments(
    num_train_epochs=3,  # Standard: 3-5 epochs
)
```

**Guidelines:**
```
Dataset size | Epochs
-------------|-------
< 1k         | 5-10 (more passes needed)
1k-5k        | 3-5 (standard)
5k-10k       | 2-3
> 10k        | 1-2 (large dataset, fewer passes)

Rule: Smaller dataset → more epochs (but watch for overfitting!)
```

**Too many epochs:**
- Training loss → 0 but val loss increases (overfitting)
- Model memorizes training data
- Catastrophic forgetting

**Too few epochs:**
- Model hasn't fully adapted
- Training and val loss still decreasing

### Batch Size

```python
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # Depends on GPU memory
    gradient_accumulation_steps=4,   # Effective batch = 8 × 4 = 32
)
```

**Guidelines:**
```
GPU Memory | Batch Size (7B model)
-----------|----------------------
16 GB      | 1-2 (use gradient accumulation!)
24 GB      | 2-4
40 GB      | 4-8
80 GB      | 8-16

Effective batch size (with accumulation): 16-64 typical
```

**Gradient accumulation:**
```python
# Simulate batch_size=32 with only 8 examples fitting in memory:
per_device_train_batch_size=8
gradient_accumulation_steps=4
# Effective batch = 8 × 4 = 32
```

### Weight Decay

```python
training_args = TrainingArguments(
    weight_decay=0.01,  # L2 regularization (prevent overfitting)
)
```

**Guidelines:**
- Standard: 0.01
- Strong regularization: 0.1 (small dataset, high overfitting risk)
- Light regularization: 0.001 (large dataset)

### Warmup

```python
training_args = TrainingArguments(
    warmup_steps=100,  # Or warmup_ratio=0.1 (10% of training)
)
```

**Why warmup:**
- Prevents initial instability (large gradients early)
- Gradual LR increase: 0 → target_LR over warmup steps

**Guidelines:**
- Warmup: 5-10% of total training steps
- Longer warmup for larger models

---

## Training

### Basic Training Loop

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",

    # Hyperparameters
    learning_rate=1e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    warmup_steps=100,

    # Evaluation
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Logging
    logging_steps=10,
    logging_dir="./logs",

    # Optimization
    fp16=True,  # Mixed precision (faster, less memory)
    gradient_checkpointing=True,  # Trade compute for memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Monitoring Training

**Key metrics to watch:**

```python
# 1. Training loss (should decrease steadily)
# 2. Validation loss (should decrease, then plateau)
# 3. Validation metrics (accuracy, F1, BLEU, etc.)

# Warning signs:
# - Train loss → 0 but val loss increasing: Overfitting
# - Train loss oscillating: LR too high
# - Train loss not decreasing: LR too low or data issues
```

**Logging:**
```python
import wandb

wandb.init(project="fine-tuning")

training_args = TrainingArguments(
    report_to="wandb",  # Log to Weights & Biases
    logging_steps=10,
)
```

### Early Stopping

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    ...
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3,  # Stop if no improvement for 3 evals
        early_stopping_threshold=0.01,  # Minimum improvement
    )]
)
```

**Why early stopping:**
- Prevents overfitting (stops before val loss increases)
- Saves compute (don't train unnecessary epochs)
- Automatically finds optimal epoch count

---

## Evaluation

### 1. Validation During Training

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(decoded_labels, decoded_preds)
    f1 = f1_score(decoded_labels, decoded_preds, average='weighted')

    return {'accuracy': accuracy, 'f1': f1}

trainer = Trainer(
    ...
    compute_metrics=compute_metrics,
)
```

### 2. Test Set Evaluation (Final)

```python
# After training completes, evaluate on held-out test set ONCE
test_results = trainer.evaluate(test_dataset)

print(f"Test accuracy: {test_results['accuracy']:.2%}")
print(f"Test F1: {test_results['f1']:.2%}")
```

### 3. Qualitative Evaluation

**Critical: Manually test on real examples!**

```python
def test_model(model, tokenizer, test_examples):
    for ex in test_examples:
        prompt = ex['input']
        expected = ex['output']

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Input: {prompt}")
        print(f"Expected: {expected}")
        print(f"Generated: {generated}")
        print(f"Match: {'✓' if generated == expected else '✗'}")
        print("-" * 80)

# Test on 20-50 examples (including edge cases)
test_model(model, tokenizer, test_examples)
```

### 4. A/B Testing (Production)

```python
# Route 50% traffic to base model, 50% to fine-tuned
import random

def get_model():
    if random.random() < 0.5:
        return base_model
    else:
        return finetuned_model

# Measure:
# - User satisfaction (thumbs up/down)
# - Task success rate
# - Response time
# - Cost per request

# After 1000+ requests, analyze results
```

### 5. Catastrophic Forgetting Check

**Critical: Ensure fine-tuning didn't break base capabilities!**

```python
# Test on general knowledge tasks
general_tasks = [
    "What is the capital of France?",  # Basic knowledge
    "Translate to Spanish: Hello",    # Translation
    "2 + 2 = ?",                       # Basic math
    "Who wrote Hamlet?",               # Literature
]

for task in general_tasks:
    before = base_model.generate(task)
    after = finetuned_model.generate(task)

    print(f"Task: {task}")
    print(f"Before: {before}")
    print(f"After: {after}")
    print(f"Preserved: {'✓' if before == after else '✗'}")
```

---

## Common Issues and Solutions

### Issue 1: Overfitting

**Symptoms:**
- Train loss → 0, val loss increases
- Perfect on training data, poor on test data

**Solutions:**
```python
# 1. Reduce epochs
num_train_epochs=3  # Instead of 10

# 2. Increase regularization
weight_decay=0.1  # Instead of 0.01

# 3. Early stopping
early_stopping_patience=3

# 4. Collect more data
# 5. Data augmentation

# 6. Use LoRA (less prone to overfitting than full FT)
```

### Issue 2: Catastrophic Forgetting

**Symptoms:**
- Fine-tuned model fails on general tasks
- Lost pre-trained knowledge

**Solutions:**
```python
# 1. Lower learning rate (most important!)
learning_rate=1e-6  # Instead of 1e-4

# 2. Fewer epochs
num_train_epochs=2  # Instead of 10

# 3. Use LoRA (base model frozen, can't forget)

# 4. Add general examples to training set (10-20% general data)
```

### Issue 3: Poor Quality

**Symptoms:**
- Model output is low quality (incorrect, incoherent)

**Solutions:**
```python
# 1. Check dataset quality (most common cause!)
# - Manual validation
# - Remove noise
# - Fix labels

# 2. Increase model size
# - 7B → 13B → 70B

# 3. Increase training data
# - Need 1000+ high-quality examples

# 4. Adjust hyperparameters
# - Try higher LR (1e-5 → 3e-5) if underfit
# - Train longer (3 → 5 epochs)

# 5. Check if base model has capability
# - If base model can't do task, fine-tuning won't help
```

### Issue 4: Slow Training

**Symptoms:**
- Training takes days/weeks

**Solutions:**
```python
# 1. Use LoRA (10× faster than full FT)

# 2. Mixed precision
fp16=True  # 2× faster

# 3. Gradient checkpointing (trade speed for memory)
gradient_checkpointing=True

# 4. Smaller batch size + gradient accumulation
per_device_train_batch_size=2
gradient_accumulation_steps=16

# 5. Use multiple GPUs
# 6. Use faster GPU (A100 > V100 > T4)
```

### Issue 5: Out of Memory

**Symptoms:**
- CUDA out of memory error

**Solutions:**
```python
# 1. Use QLoRA (4× less memory)

# 2. Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=32

# 3. Gradient checkpointing
gradient_checkpointing=True

# 4. Use smaller model
# 7B → 3B → 1B

# 5. Reduce sequence length
max_seq_length=512  # Instead of 2048
```

---

## Best Practices Summary

### Before Fine-Tuning:

1. ☐ Try prompt engineering first (90% of cases, prompts work!)
2. ☐ Have 1000+ high-quality examples
3. ☐ Clean and validate dataset (quality > quantity)
4. ☐ Create train/val/test split (70/15/15)
5. ☐ Define success metrics (what does "good" mean?)

### During Fine-Tuning:

6. ☐ Use LoRA (unless specific reason for full FT)
7. ☐ Set tiny learning rate (1e-5 to 1e-6 for 7B models)
8. ☐ Train for 3-5 epochs (not 50!)
9. ☐ Monitor val loss (stop when it stops improving)
10. ☐ Log everything (wandb, tensorboard)

### After Fine-Tuning:

11. ☐ Evaluate on test set (quantitative metrics)
12. ☐ Manual testing (qualitative, 20-50 examples)
13. ☐ Check for catastrophic forgetting (general tasks)
14. ☐ A/B test in production (before full rollout)
15. ☐ Document hyperparameters (for reproducibility)

---

## Quick Reference

| Task | Method | Dataset | LR | Epochs |
|------|--------|---------|----|----|
| Tone matching | Prompts | N/A | N/A | N/A |
| Simple classification | Prompts | N/A | N/A | N/A |
| Complex domain task | LoRA | 1k-10k | 1e-5 | 3-5 |
| Fundamental change | Full FT | 100k+ | 1e-5 | 1-3 |
| Limited GPU | QLoRA | 1k-10k | 1e-5 | 3-5 |

**Default recommendation:** Try prompts first. If that fails, use LoRA with LR=1e-5, epochs=3, and high-quality dataset.

---

## Summary

**Core principles:**

1. **Prompt engineering first**: 90% of tasks don't need fine-tuning
2. **LoRA by default**: 100× more efficient than full fine-tuning, same quality
3. **Data quality matters**: 1,000 clean examples > 10,000 noisy examples
4. **Tiny learning rate**: Fine-tune LR = Pre-train LR / 100 to / 1000
5. **Validation essential**: Train/val/test split + early stopping + catastrophic forgetting check

**Decision tree:**
1. Try prompts (system message + few-shot)
2. If quality < 90%, optimize prompts
3. If still < 90% and have 1000+ examples, consider fine-tuning
4. Use LoRA (default), QLoRA (limited GPU), or full FT (rare)
5. Set LR = 1e-5, epochs = 3-5, monitor val loss
6. Evaluate on test set + manual testing + general tasks

**Key insight**: Fine-tuning is powerful but expensive and slow. Start with prompts, fine-tune only when prompts demonstrably fail and you have high-quality data.
