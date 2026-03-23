
# CNN Families and Selection: Choosing the Right Convolutional Network

<CRITICAL_CONTEXT>
CNNs are the foundation of computer vision. Different families have vastly different trade-offs:
- Accuracy vs Speed vs Size
- Dataset size requirements
- Deployment target (cloud vs edge vs mobile)
- Task type (classification vs detection vs segmentation)

This skill helps you choose the RIGHT CNN for YOUR constraints.
</CRITICAL_CONTEXT>

## When to Use This Skill

Use this skill when:
- ✅ Selecting CNN for vision task (classification, detection, segmentation)
- ✅ Comparing CNN families (ResNet vs EfficientNet vs MobileNet)
- ✅ Optimizing for specific constraints (latency, size, accuracy)
- ✅ Understanding CNN evolution (why newer architectures exist)
- ✅ Deployment-specific selection (cloud, edge, mobile)

DO NOT use for:
- ❌ Non-vision tasks (use sequence-models-comparison or other skills)
- ❌ Training optimization (use training-optimization pack)
- ❌ Implementation details (use pytorch-engineering pack)

**When in doubt:** If choosing WHICH CNN → this skill. If implementing/training CNN → other skills.


## Selection Framework

### Step 1: Identify Constraints

**Before recommending ANY architecture, ask:**

| Constraint | Question | Impact |
|------------|----------|--------|
| **Deployment** | Where will model run? | Cloud → Any, Edge → MobileNet/EfficientNet-Lite, Mobile → MobileNetV3 |
| **Latency** | Speed requirement? | Real-time (< 10ms) → MobileNet, Batch (> 100ms) → Any |
| **Model Size** | Parameter/memory budget? | < 10M params → MobileNet, < 50M → ResNet/EfficientNet, Any → Large models OK |
| **Dataset Size** | Training images? | < 10k → Small models, 10k-100k → Medium, > 100k → Large |
| **Accuracy** | Required accuracy? | Competitive → EfficientNet-B4+, Production → ResNet-50/EfficientNet-B2 |
| **Task Type** | Classification/detection/segmentation? | Detection → FPN-compatible, Segmentation → Multi-scale |

**Critical:** Get answers to these BEFORE recommending architecture.

### Step 2: Apply Decision Tree

```
START: What's your primary constraint?

┌─ DEPLOYMENT TARGET
│  ├─ Cloud / Server
│  │  └─ Dataset size?
│  │     ├─ Small (< 10k) → ResNet-18, EfficientNet-B0
│  │     ├─ Medium (10k-100k) → ResNet-50, EfficientNet-B2
│  │     └─ Large (> 100k) → ResNet-101, EfficientNet-B4, ViT
│  │
│  ├─ Edge Device (Jetson, Coral)
│  │  └─ Latency requirement?
│  │     ├─ Real-time (< 10ms) → MobileNetV3-Small, EfficientNet-Lite0
│  │     ├─ Medium (10-50ms) → MobileNetV3-Large, EfficientNet-Lite2
│  │     └─ Relaxed (> 50ms) → EfficientNet-B0, ResNet-18
│  │
│  └─ Mobile (iOS/Android)
│     └─ MobileNetV3-Small (fastest), MobileNetV3-Large (balanced)
│        + INT8 quantization (route to ml-production)
│
├─ ACCURACY PRIORITY (cloud deployment assumed)
│  ├─ Maximum accuracy → EfficientNet-B7, ResNet-152, ViT-Large
│  ├─ Balanced → EfficientNet-B2/B3, ResNet-50
│  └─ Fast training → ResNet-18, EfficientNet-B0
│
├─ EFFICIENCY PRIORITY
│  └─ Best accuracy per FLOP → EfficientNet family (B0-B7)
│     (EfficientNet dominates ResNet on Pareto frontier)
│
└─ TASK TYPE
   ├─ Classification → Any CNN (use constraint-based selection above)
   ├─ Object Detection → ResNet + FPN, EfficientDet, YOLOv8 (CSPDarknet)
   └─ Segmentation → ResNet + U-Net, EfficientNet + DeepLabV3
```


## CNN Family Catalog

### 1. ResNet Family (2015) - The Standard Baseline

**Architecture:** Residual connections (skip connections) enable very deep networks

**Variants:**
- ResNet-18: 11M params, 1.8 GFLOPs, 69.8% ImageNet
- ResNet-34: 22M params, 3.7 GFLOPs, 73.3% ImageNet
- ResNet-50: 25M params, 4.1 GFLOPs, 76.1% ImageNet
- ResNet-101: 44M params, 7.8 GFLOPs, 77.4% ImageNet
- ResNet-152: 60M params, 11.6 GFLOPs, 78.3% ImageNet

**When to Use:**
- ✅ **Baseline choice**: Well-tested, widely supported
- ✅ **Transfer learning**: Excellent pre-trained weights available
- ✅ **Object detection**: Standard backbone for Faster R-CNN, Mask R-CNN
- ✅ **Interpretability**: Simple architecture, easy to understand

**When NOT to Use:**
- ❌ **Edge/mobile deployment**: Too large and slow
- ❌ **Efficiency priority**: EfficientNet beats ResNet on accuracy/FLOP
- ❌ **Small datasets (< 10k)**: Use ResNet-18, not ResNet-50+

**Key Insight:** Skip connections solve vanishing gradient, enable depth

**Code Example:**
```python
import torchvision.models as models

# For cloud/server (good dataset)
model = models.resnet50(pretrained=True)

# For small dataset or faster training
model = models.resnet18(pretrained=True)

# For maximum accuracy (cloud only)
model = models.resnet101(pretrained=True)
```


### 2. EfficientNet Family (2019) - Best Efficiency

**Architecture:** Compound scaling (depth + width + resolution) optimized via neural architecture search

**Variants:**
- EfficientNet-B0: 5M params, 0.4 GFLOPs, 77.3% ImageNet
- EfficientNet-B1: 8M params, 0.7 GFLOPs, 79.2% ImageNet
- EfficientNet-B2: 9M params, 1.0 GFLOPs, 80.3% ImageNet
- EfficientNet-B3: 12M params, 1.8 GFLOPs, 81.7% ImageNet
- EfficientNet-B4: 19M params, 4.2 GFLOPs, 82.9% ImageNet
- EfficientNet-B7: 66M params, 37 GFLOPs, 84.4% ImageNet

**When to Use:**
- ✅ **Efficiency matters**: Best accuracy per FLOP/parameter
- ✅ **Cloud deployment**: B2-B4 sweet spot for production
- ✅ **Limited compute**: B0 matches ResNet-50 accuracy at 10x fewer FLOPs
- ✅ **Scaling needs**: Want to scale model up/down systematically

**When NOT to Use:**
- ❌ **Real-time mobile**: Use MobileNet (EfficientNet has more layers)
- ❌ **Very small datasets**: Can overfit despite efficiency
- ❌ **Simplicity needed**: More complex than ResNet

**Key Insight:** Compound scaling balances depth, width, and resolution optimally

**Efficiency Comparison:**
```
Same accuracy as ResNet-50 (76%):
- ResNet-50: 25M params, 4.1 GFLOPs
- EfficientNet-B0: 5M params, 0.4 GFLOPs (10x more efficient!)

Better accuracy (82.9%):
- ResNet-152: 60M params, 11.6 GFLOPs → 78.3% ImageNet
- EfficientNet-B4: 19M params, 4.2 GFLOPs → 82.9% ImageNet
  (Better accuracy with 3x fewer params and 3x less compute)
```

**Code Example:**
```python
import timm  # PyTorch Image Models library

# Balanced choice (production)
model = timm.create_model('efficientnet_b2', pretrained=True)

# Efficiency priority (edge)
model = timm.create_model('efficientnet_b0', pretrained=True)

# Accuracy priority (research)
model = timm.create_model('efficientnet_b4', pretrained=True)
```


### 3. MobileNet Family (2017-2019) - Mobile Optimized

**Architecture:** Depthwise separable convolutions (drastically reduce compute)

**Variants:**
- MobileNetV1: 4.2M params, 0.6 GFLOPs, 70.6% ImageNet
- MobileNetV2: 3.5M params, 0.3 GFLOPs, 72.0% ImageNet
- MobileNetV3-Small: 2.5M params, 0.06 GFLOPs, 67.4% ImageNet
- MobileNetV3-Large: 5.4M params, 0.2 GFLOPs, 75.2% ImageNet

**When to Use:**
- ✅ **Mobile deployment**: iOS/Android apps
- ✅ **Edge devices**: Raspberry Pi, Jetson Nano
- ✅ **Real-time inference**: < 100ms latency
- ✅ **Extreme efficiency**: < 10M parameters budget

**When NOT to Use:**
- ❌ **Cloud deployment with no constraints**: EfficientNet or ResNet better accuracy
- ❌ **Accuracy priority**: Sacrifices accuracy for speed
- ❌ **Large datasets with compute**: Can afford better models

**Key Insight:** Depthwise separable convolutions = standard conv split into depthwise + pointwise (9x fewer operations)

**Deployment Performance:**
```
Raspberry Pi 4 inference (224×224 image):
- ResNet-50: ~2000ms (unusable)
- ResNet-18: ~600ms (slow)
- MobileNetV2: ~150ms (acceptable)
- MobileNetV3-Large: ~80ms (good)
- MobileNetV3-Small: ~40ms (fast)

With INT8 quantization:
- MobileNetV3-Large: ~30ms (production-ready)
- MobileNetV3-Small: ~15ms (real-time)
```

**Code Example:**
```python
import torchvision.models as models

# For mobile deployment
model = models.mobilenet_v3_large(pretrained=True)

# For ultra-low latency (sacrifice accuracy)
model = models.mobilenet_v3_small(pretrained=True)

# Quantization for mobile (route to ml-production skill for details)
# Achieves 2-4x speedup with minimal accuracy loss
```


### 4. Inception Family (2014-2016) - Multi-Scale Features

**Architecture:** Multi-scale convolutions in parallel (inception modules)

**Variants:**
- InceptionV3: 24M params, 5.7 GFLOPs, 77.5% ImageNet
- InceptionV4: 42M params, 12.3 GFLOPs, 80.0% ImageNet
- Inception-ResNet: Hybrid with residual connections

**When to Use:**
- ✅ **Multi-scale features**: Objects at different sizes
- ✅ **Object detection**: Good backbone for detection
- ✅ **Historical interest**: Understanding multi-scale approaches

**When NOT to Use:**
- ❌ **Simplicity needed**: Complex architecture, hard to modify
- ❌ **Efficiency priority**: EfficientNet better
- ❌ **Modern projects**: Largely superseded by ResNet/EfficientNet

**Key Insight:** Parallel multi-scale convolutions (1×1, 3×3, 5×5) capture different receptive fields

**Status:** Mostly historical - ResNet and EfficientNet have replaced Inception in practice


### 5. DenseNet Family (2017) - Dense Connections

**Architecture:** Every layer connects to every other layer (dense connections)

**Variants:**
- DenseNet-121: 8M params, 2.9 GFLOPs, 74.4% ImageNet
- DenseNet-169: 14M params, 3.4 GFLOPs, 75.6% ImageNet
- DenseNet-201: 20M params, 4.3 GFLOPs, 76.9% ImageNet

**When to Use:**
- ✅ **Parameter efficiency**: Good accuracy with few parameters
- ✅ **Feature reuse**: Dense connections enable feature reuse
- ✅ **Small datasets**: Better gradient flow helps with limited data

**When NOT to Use:**
- ❌ **Inference speed priority**: Dense connections slow (high memory bandwidth)
- ❌ **Training speed**: Slower to train than ResNet
- ❌ **Production deployment**: Less mature ecosystem than ResNet

**Key Insight:** Dense connections improve gradient flow, enable feature reuse, but slow inference

**Status:** Theoretically elegant, but ResNet/EfficientNet more practical


### 6. VGG Family (2014) - Historical Baseline

**Architecture:** Very deep (16-19 layers), small 3×3 convolutions, many parameters

**Variants:**
- VGG-16: 138M params, 15.5 GFLOPs, 71.5% ImageNet
- VGG-19: 144M params, 19.6 GFLOPs, 71.1% ImageNet

**When to Use:**
- ❌ **DON'T use VGG for new projects**
- Historical understanding only

**Why NOT to Use:**
- Massive parameter count (138M vs ResNet-50's 25M)
- Poor accuracy for size
- Superseded by ResNet (2015)

**Key Insight:** Proved that depth matters, but skip connections (ResNet) are better

**Status:** **Obsolete** - use ResNet or EfficientNet instead


## Practical Selection Guide

### Scenario 1: Cloud/Server Deployment

**Goal:** Best accuracy, no compute constraints

**Recommendation:**
```
Small dataset (< 10k images):
→ EfficientNet-B0 or ResNet-18
  (Avoid overfitting with smaller model)

Medium dataset (10k-100k images):
→ EfficientNet-B2 or ResNet-50
  (Balanced accuracy and efficiency)

Large dataset (> 100k images):
→ EfficientNet-B4 or ResNet-101
  (Can afford larger model)

Maximum accuracy (research):
→ EfficientNet-B7 or Vision Transformer
  (If dataset > 1M images and compute unlimited)
```


### Scenario 2: Edge Deployment (Jetson, Coral TPU)

**Goal:** Optimize for edge hardware latency

**Recommendation:**
```
Real-time requirement (< 10ms):
→ MobileNetV3-Small or EfficientNet-Lite0
  + INT8 quantization

Medium latency (10-50ms):
→ MobileNetV3-Large or EfficientNet-Lite2

Relaxed latency (> 50ms):
→ EfficientNet-B0 or ResNet-18
```

**Critical:** Profile on actual edge hardware. Quantization is mandatory (route to ml-production).


### Scenario 3: Mobile Deployment (iOS/Android)

**Goal:** On-device inference, minimal battery drain

**Recommendation:**
```
All mobile deployments:
→ MobileNetV3-Large (balanced)
→ MobileNetV3-Small (fastest, less accurate)

Always use:
- INT8 quantization (2-4x speedup)
- CoreML (iOS) or TFLite (Android) optimization
- Benchmark on target device before deploying
```

**Expected latency (iPhone 12, INT8 quantized):**
- MobileNetV3-Small: 5-10ms
- MobileNetV3-Large: 15-25ms


### Scenario 4: Object Detection

**Goal:** Select backbone for detection framework

**Recommendation:**
```
Faster R-CNN:
→ ResNet-50 + FPN (standard)
→ ResNet-101 + FPN (more accuracy)

YOLOv8:
→ CSPDarknet (built-in, optimized)

EfficientDet:
→ EfficientNet + BiFPN (best efficiency)

Custom detection:
→ ResNet or EfficientNet as backbone
→ Add Feature Pyramid Network (FPN) for multi-scale
```

**Note:** Detection adds significant compute on top of backbone. Choose efficient backbone.


### Scenario 5: Semantic Segmentation

**Goal:** Dense pixel-wise prediction

**Recommendation:**
```
U-Net style:
→ ResNet-18/34 as encoder (fast)
→ EfficientNet-B0 as encoder (efficient)

DeepLabV3:
→ ResNet-50 (standard)
→ MobileNetV3 (mobile deployment)

Key: Segmentation requires multi-scale features
→ Ensure backbone has skip connections or FPN
```


## Trade-Off Analysis

### Accuracy vs Efficiency (Pareto Frontier)

**ImageNet Top-1 Accuracy vs FLOPs:**

```
Efficiency Winners (best accuracy per FLOP):
1. EfficientNet-B0: 77.3% @ 0.4 GFLOPs (best efficiency)
2. EfficientNet-B2: 80.3% @ 1.0 GFLOPs
3. EfficientNet-B4: 82.9% @ 4.2 GFLOPs

Accuracy Winners (best absolute accuracy):
1. EfficientNet-B7: 84.4% @ 37 GFLOPs
2. ViT-Large: 85.2% @ 190 GFLOPs (requires huge dataset)
3. ResNet-152: 78.3% @ 11.6 GFLOPs (dominated by EfficientNet)

Speed Winners (lowest latency):
1. MobileNetV3-Small: 67.4% @ 0.06 GFLOPs (50ms on mobile)
2. MobileNetV3-Large: 75.2% @ 0.2 GFLOPs (100ms on mobile)
3. EfficientNet-Lite0: 75.0% @ 0.4 GFLOPs
```

**Key Takeaway:** EfficientNet dominates ResNet on Pareto frontier (better accuracy at same compute).


### Parameters vs Accuracy

**For same ~75% ImageNet accuracy:**
```
VGG-16:           138M params (❌ terrible efficiency)
ResNet-50:         25M params
EfficientNet-B0:    5M params (✅ 5x fewer parameters!)
MobileNetV3-Large:  5M params (fast inference)
```

**Conclusion:** Modern architectures (EfficientNet, MobileNet) achieve same accuracy with far fewer parameters.


## Common Pitfalls

### Pitfall 1: Defaulting to ResNet-50
**Symptom:** Using ResNet-50 without considering alternatives

**Why it's wrong:** EfficientNet-B0 matches ResNet-50 accuracy with 10x less compute

**Fix:** Consider EfficientNet family first (better efficiency)


### Pitfall 2: Choosing Large Model for Small Dataset
**Symptom:** Using ResNet-101 with < 10k images

**Why it's wrong:** Model will overfit (too many parameters for data)

**Fix:**
- < 10k images → ResNet-18 or EfficientNet-B0
- 10k-100k → ResNet-50 or EfficientNet-B2
- > 100k → Can use larger models


### Pitfall 3: Using Desktop Model on Mobile
**Symptom:** Trying to run ResNet-50 on mobile device

**Why it's wrong:** 2000ms inference time is unusable

**Fix:** Use MobileNetV3 + quantization for mobile (15-30ms)


### Pitfall 4: Ignoring Task Type
**Symptom:** Using standard CNN for object detection without FPN

**Why it's wrong:** Detection needs multi-scale features

**Fix:** Use detection-specific frameworks (YOLOv8, Faster R-CNN) with appropriate backbone


### Pitfall 5: Believing "Bigger = Better"
**Symptom:** Choosing ResNet-152 over ResNet-50 without justification

**Why it's wrong:** Diminishing returns - 3x compute for 1.3% accuracy, will overfit on small data

**Fix:** Match model capacity to dataset size, consider efficiency


## Evolution and Historical Context

**Why CNNs evolved the way they did:**

```
2012: AlexNet
→ Proved deep learning works for vision
→ 8 layers, 60M params

2014: VGG
→ Deeper is better (16-19 layers)
→ But: 138M params (too many)

2014: Inception/GoogLeNet
→ Multi-scale convolutions
→ More efficient than VGG

2015: ResNet ★
→ Skip connections enable very deep networks (152 layers)
→ Solved vanishing gradient problem
→ Became standard baseline

2017: MobileNet
→ Mobile deployment needs
→ Depthwise separable convolutions (9x fewer ops)

2017: DenseNet
→ Dense connections for feature reuse
→ Parameter efficient but slow inference

2019: EfficientNet ★
→ Compound scaling (depth + width + resolution)
→ Neural architecture search
→ Dominates Pareto frontier (best accuracy per FLOP)
→ New standard for efficiency

2020: Vision Transformer
→ Attention-based (no convolutions)
→ Requires very large datasets (> 1M images)
→ For research/large-scale applications
```

**Current Recommendations (2025):**
- Cloud: **EfficientNet** (best efficiency) or ResNet (simplicity)
- Edge: **EfficientNet-Lite** or MobileNetV3
- Mobile: **MobileNetV3** + quantization
- Detection: **EfficientDet** or YOLOv8
- Baseline: **ResNet** (simple, well-tested)


## Decision Checklist

Before choosing CNN, answer these:

```
☐ Deployment target? (cloud/edge/mobile)
☐ Latency requirement? (< 10ms / 10-100ms / > 100ms)
☐ Model size budget? (< 10M / 10-50M / unlimited params)
☐ Dataset size? (< 10k / 10k-100k / > 100k images)
☐ Accuracy priority? (maximum / production / fast iteration)
☐ Task type? (classification / detection / segmentation)
☐ Efficiency matters? (yes → EfficientNet, no → flexibility)

Based on answers:
→ Mobile → MobileNetV3
→ Edge → EfficientNet-Lite or MobileNetV3
→ Cloud + efficiency → EfficientNet
→ Cloud + simplicity → ResNet
→ Maximum accuracy → EfficientNet-B7 or ViT
→ Small dataset → Small models (ResNet-18, EfficientNet-B0)
```


## Integration with Other Skills

**After selecting CNN architecture:**

**Training the model:**
→ `yzmir/training-optimization/using-training-optimization`
- Optimizer selection (Adam, SGD, AdamW)
- Learning rate schedules
- Data augmentation

**Implementing in PyTorch:**
→ `yzmir/pytorch-engineering/using-pytorch-engineering`
- Custom modifications to pre-trained models
- Multi-GPU training
- Performance optimization

**Deploying to production:**
→ `yzmir/ml-production/using-ml-production`
- Quantization (INT8, FP16)
- Model serving (TorchServe, ONNX)
- Optimization for edge/mobile (TFLite, CoreML)

**If architecture is unstable (very deep):**
→ `yzmir/neural-architectures/normalization-techniques`
- Normalization layers (BatchNorm, LayerNorm)
- Skip connections
- Initialization strategies


## Summary

**CNN Selection in One Table:**

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Cloud, balanced | EfficientNet-B2 | Best efficiency, 80% accuracy |
| Cloud, max accuracy | EfficientNet-B4 | 83% accuracy, reasonable compute |
| Cloud, simple baseline | ResNet-50 | Well-tested, widely used |
| Edge device | MobileNetV3-Large | Optimized for edge, 75% accuracy |
| Mobile app | MobileNetV3-Small + quantization | < 20ms inference |
| Small dataset (< 10k) | ResNet-18 or EfficientNet-B0 | Avoid overfitting |
| Object detection | ResNet-50 + FPN, EfficientDet | Multi-scale features |
| Segmentation | ResNet + U-Net, DeepLabV3 | Dense prediction |

**Key Principles:**
1. **Match model capacity to dataset size** (small data → small model)
2. **EfficientNet dominates ResNet on efficiency** (same accuracy, less compute)
3. **Mobile needs mobile-specific architectures** (MobileNet, quantization)
4. **Task type matters** (detection/segmentation need multi-scale features)
5. **Bigger ≠ always better** (diminishing returns, overfitting risk)

**When in doubt:** Start with **EfficientNet-B2** (cloud) or **MobileNetV3-Large** (edge/mobile).


**END OF SKILL**
