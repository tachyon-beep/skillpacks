---
description: Compare CNN architectures for vision tasks - ResNet vs EfficientNet vs MobileNet
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[constraint: cloud|edge|mobile|accuracy]"
---

# CNN Comparison Command

You are comparing CNN architectures for a computer vision task. Provide data-driven comparison based on constraints.

## Core Principle

**EfficientNet dominates ResNet on efficiency. MobileNet wins on mobile.** Match architecture to deployment target.

## Quick Reference Table

| Architecture | Params | GFLOPs | ImageNet Acc | Best For |
|--------------|--------|--------|--------------|----------|
| **ResNet-18** | 11M | 1.8 | 69.8% | Small datasets, baselines |
| **ResNet-50** | 25M | 4.1 | 76.1% | General cloud, detection backbone |
| **ResNet-101** | 44M | 7.8 | 77.4% | High accuracy cloud |
| **EfficientNet-B0** | 5M | 0.4 | 77.3% | Efficiency priority |
| **EfficientNet-B2** | 9M | 1.0 | 80.3% | Balanced production |
| **EfficientNet-B4** | 19M | 4.2 | 82.9% | High accuracy, efficient |
| **EfficientNet-B7** | 66M | 37 | 84.4% | Maximum accuracy |
| **MobileNetV3-Small** | 2.5M | 0.06 | 67.4% | Mobile real-time |
| **MobileNetV3-Large** | 5.4M | 0.2 | 75.2% | Mobile balanced |
| **EfficientNet-Lite0** | 4.7M | 0.4 | 75.0% | Edge devices |

## Comparison by Constraint

### Constraint: Cloud Deployment (No limits)

**Comparison:**

```
Same ~76% accuracy:
├─ ResNet-50:        25M params, 4.1 GFLOPs
└─ EfficientNet-B0:   5M params, 0.4 GFLOPs ← 5× fewer params, 10× fewer FLOPs!

Better accuracy (83%):
├─ ResNet-152:       60M params, 11.6 GFLOPs → 78.3%
└─ EfficientNet-B4:  19M params, 4.2 GFLOPs  → 82.9% ← Better accuracy, 3× smaller!
```

**Recommendation:**
- Default: **EfficientNet-B2** (80%, balanced)
- Max accuracy: **EfficientNet-B4** (83%)
- Legacy/compatibility: **ResNet-50** (well-tested)

### Constraint: Edge Deployment (Jetson, Coral)

**Latency benchmarks (Jetson Nano, FP16):**

| Architecture | Latency | Accuracy |
|--------------|---------|----------|
| ResNet-50 | 150ms | 76.1% |
| ResNet-18 | 45ms | 69.8% |
| EfficientNet-Lite0 | 25ms | 75.0% |
| MobileNetV3-Large | 20ms | 75.2% |
| MobileNetV3-Small | 12ms | 67.4% |

**Recommendation:**
- Real-time (<20ms): **MobileNetV3-Small**
- Balanced: **MobileNetV3-Large** or **EfficientNet-Lite0**
- Best quality: **EfficientNet-Lite2** (30ms, 77%)

### Constraint: Mobile (iOS/Android)

**Latency benchmarks (iPhone 12, INT8):**

| Architecture | Latency | Accuracy |
|--------------|---------|----------|
| MobileNetV3-Small | 5-10ms | 67.4% |
| MobileNetV3-Large | 15-25ms | 75.2% |
| EfficientNet-Lite0 | 20-30ms | 75.0% |

**Recommendation:**
- All mobile: **MobileNetV3** family
- Plus INT8 quantization (route to ml-production)
- CoreML (iOS) or TFLite (Android) optimization

### Constraint: Small Dataset (<10k images)

**Overfitting risk by model size:**

| Architecture | Params | Risk for 10k samples |
|--------------|--------|---------------------|
| ResNet-152 | 60M | HIGH (60M/10k = 6000:1) |
| ResNet-50 | 25M | MEDIUM (25M/10k = 2500:1) |
| ResNet-18 | 11M | LOW (11M/10k = 1100:1) |
| EfficientNet-B0 | 5M | LOWEST (5M/10k = 500:1) |

**Recommendation:**
- <5k samples: **ResNet-18** or **EfficientNet-B0**
- 5-10k samples: **EfficientNet-B1** or **ResNet-34**
- Always use: pretrained weights, data augmentation

### Constraint: Object Detection Backbone

**Detection frameworks expect:**
- FPN-compatible backbones
- Multi-scale features
- Pretrained weights

| Framework | Recommended Backbone |
|-----------|---------------------|
| Faster R-CNN | ResNet-50 + FPN |
| Mask R-CNN | ResNet-101 + FPN |
| EfficientDet | EfficientNet + BiFPN |
| YOLOv8 | CSPDarknet (built-in) |
| RetinaNet | ResNet-50 + FPN |

**Recommendation:** Use framework's default backbone unless specific constraint.

## Pareto Analysis

**Best architecture for each point on accuracy-efficiency frontier:**

```
Accuracy:  67%     70%     75%     77%     80%     83%     84%
           │       │       │       │       │       │       │
           ▼       ▼       ▼       ▼       ▼       ▼       ▼
FLOPs:   0.06G   0.2G    0.4G    0.4G    1.0G    4.2G    37G
         MNv3-S  MNv3-L  EffB0   EffB0   EffB2   EffB4   EffB7

Key insight: EfficientNet dominates every point except ultra-mobile
```

## Decision Flowchart

```
START: What's your deployment target?

├─ CLOUD (no constraints)
│  └─ Dataset size?
│     ├─ Small (<10k) → EfficientNet-B0 or ResNet-18
│     ├─ Medium (10k-100k) → EfficientNet-B2
│     └─ Large (>100k) → EfficientNet-B4
│
├─ EDGE (Jetson, Coral)
│  └─ Latency requirement?
│     ├─ <15ms → MobileNetV3-Small
│     ├─ <30ms → MobileNetV3-Large or EfficientNet-Lite0
│     └─ <60ms → EfficientNet-Lite2
│
├─ MOBILE (iOS/Android)
│  └─ MobileNetV3 + INT8 quantization (always)
│     ├─ Speed priority → MobileNetV3-Small
│     └─ Accuracy priority → MobileNetV3-Large
│
└─ DETECTION/SEGMENTATION
   └─ Use framework default or ResNet-50 + FPN
```

## Code Examples

### Cloud (EfficientNet)
```python
import timm

# Balanced production
model = timm.create_model('efficientnet_b2', pretrained=True)

# Maximum accuracy
model = timm.create_model('efficientnet_b4', pretrained=True)
```

### Edge (EfficientNet-Lite)
```python
import timm

# Edge optimized
model = timm.create_model('efficientnet_lite0', pretrained=True)
```

### Mobile (MobileNet)
```python
import torchvision.models as models

# Mobile balanced
model = models.mobilenet_v3_large(pretrained=True)

# Mobile fastest
model = models.mobilenet_v3_small(pretrained=True)
```

### Small Dataset (ResNet-18)
```python
import torchvision.models as models

# Avoid overfitting on small data
model = models.resnet18(pretrained=True)
# Freeze early layers
for param in model.layer1.parameters():
    param.requires_grad = False
```

## Output Format

After comparison, provide:

```markdown
## CNN Comparison Results

**Constraint**: [deployment/dataset/accuracy]
**Winner**: [Architecture name]

### Comparison Table
| Metric | [Arch 1] | [Arch 2] | [Arch 3] |
|--------|----------|----------|----------|
| Params | | | |
| FLOPs | | | |
| Accuracy | | | |
| Latency* | | | |

*on target hardware

### Recommendation
**Choose [Winner]** because [constraint-specific reason]

### Alternatives
- [Alternative]: Consider if [condition]

### Next Steps
1. [Implementation step]
2. [Training step]
3. [Deployment step]
```
