---
description: Check gradient health - detect vanishing, exploding, or NaN gradients before they cause training failure
allowed-tools: ["Read", "Bash", "Grep", "Glob", "Skill"]
argument-hint: "[training_script.py]"
---

# Check Gradients Command

Monitor gradient health to detect problems before they cause training failure.

## Core Principle

**Gradient issues cause 80% of "unexplained" training failures.**

Checking gradients proactively catches:
- Exploding gradients → NaN loss
- Vanishing gradients → No learning
- Dead neurons → Wasted capacity

## Gradient Health Check

### Step 1: Add Gradient Monitoring

Add this to training loop (after `loss.backward()`, before `optimizer.step()`):

```python
def check_gradient_health(model, step):
    total_norm = 0.0
    num_params = 0
    nan_count = 0
    zero_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            num_params += 1

            if torch.isnan(param.grad).any():
                nan_count += 1
                print(f"  NaN gradient in: {name}")

            if param_norm == 0:
                zero_count += 1

    total_norm = total_norm ** 0.5

    # Report
    print(f"Step {step}: grad_norm={total_norm:.4f}, nan={nan_count}, zero={zero_count}")

    return {
        'total_norm': total_norm,
        'nan_count': nan_count,
        'zero_count': zero_count,
        'num_params': num_params
    }
```

### Step 2: Interpret Results

| Finding | Status | Meaning | Action |
|---------|--------|---------|--------|
| grad_norm 0.1-10 | ✅ Healthy | Normal range | None needed |
| grad_norm > 100 | ⚠️ Warning | Potentially exploding | Add clipping (max_norm=1.0) |
| grad_norm > 1000 | ❌ Critical | Exploding | Clip immediately, reduce LR |
| grad_norm < 0.001 | ⚠️ Warning | Potentially vanishing | Check architecture, initialization |
| grad_norm = 0 | ❌ Critical | No learning happening | Check frozen params, dead ReLUs |
| nan_count > 0 | ❌ Critical | NaN gradients | Check loss function, reduce LR |
| zero_count > 50% | ⚠️ Warning | Many dead parameters | Check initialization, use LeakyReLU |

### Step 3: Common Fixes

**For Exploding Gradients (norm > 100):**

```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Or use per-parameter clipping
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

**For Vanishing Gradients (norm < 0.001):**

```python
# Check for frozen parameters
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"Frozen: {name}")

# Consider initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

model.apply(init_weights)
```

**For NaN Gradients:**

```python
# Add numerical stability to loss
loss = F.cross_entropy(logits, targets, label_smoothing=0.1)

# Or for custom loss
def safe_log(x, eps=1e-8):
    return torch.log(x + eps)
```

### Step 4: Gradient Norm Over Time

Log gradient norms across training to spot trends:

```python
# During training
grad_norms = []
for step, (x, y) in enumerate(dataloader):
    # ... forward, backward ...

    # After backward, before step
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
    grad_norms.append(total_norm.item())

    optimizer.step()

# Plot or analyze
import matplotlib.pyplot as plt
plt.plot(grad_norms)
plt.xlabel('Step')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm Over Training')
```

| Trend | Meaning | Action |
|-------|---------|--------|
| Stable around 1-10 | Healthy training | Continue |
| Increasing steadily | Building to explosion | Add clipping preemptively |
| Decreasing to near 0 | Vanishing, training stalling | Check LR, architecture |
| Spikes | Unstable, specific batches problematic | Clip, check data quality |

## Output Format

```markdown
## Gradient Health Report

### Summary
| Metric | Value | Status |
|--------|-------|--------|
| Total grad norm | X.XX | ✅/⚠️/❌ |
| NaN parameters | X | ✅/⚠️/❌ |
| Zero-grad parameters | X% | ✅/⚠️/❌ |

### Issues Detected
1. [Issue with severity and affected layers]

### Recommendations
1. [Specific fix]

### Code to Add
```python
[Gradient clipping or fix code]
```
```

## When to Check Gradients

**Proactively (before problems):**
- New model architecture
- New loss function
- Switching to mixed precision
- Training Transformers, RNNs, or very deep networks

**Reactively (when problems occur):**
- Loss becomes NaN
- Loss stuck at plateau
- Training unstable/oscillating

## Load Detailed Guidance

For comprehensive gradient management:
```
Load skill: yzmir-training-optimization:using-training-optimization
Then read: gradient-management.md
```
