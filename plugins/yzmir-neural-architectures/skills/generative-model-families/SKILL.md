---
name: generative-model-families
description: Comprehensive guide to generative model families (VAE, GAN, Diffusion, Autoregressive, Flow). Use when selecting generative architecture for images, audio, or other data. Covers quality trade-offs, training stability, speed, and modern recommendations (2025).
dependencies:
  - using-neural-architectures
  - transformer-architecture-deepdive
related:
  - cnn-families-and-selection
  - llm-specialist (for autoregressive text models)
---

# Generative Model Families

## When to Use This Skill

Use this skill when you need to:
- ✅ Select generative model for image/audio/video generation
- ✅ Understand VAE vs GAN vs Diffusion trade-offs
- ✅ Decide between training from scratch vs fine-tuning
- ✅ Address mode collapse in GANs
- ✅ Choose between quality, speed, and training stability
- ✅ Understand modern landscape (Stable Diffusion, StyleGAN, etc.)

**Do NOT use this skill for:**
- ❌ Text generation (use `llm-specialist` pack)
- ❌ Architecture implementation details (use model-specific docs)
- ❌ High-level architecture selection (use `using-neural-architectures`)

---

## Core Principle

**Generative models have fundamental trade-offs:**
- **Quality vs Stability**: GANs sharp but unstable, VAEs blurry but stable
- **Quality vs Speed**: Diffusion high-quality but slow, GANs fast
- **Explicitness vs Flexibility**: Autoregressive/Flow have likelihood, GANs don't

**Modern default (2025):** Diffusion models (best quality + stability)

---

## Part 1: Model Family Overview

### The Five Families

**1. VAE (Variational Autoencoder)**
- **Approach**: Learn latent space with encoder-decoder
- **Quality**: Blurry (6/10)
- **Training**: Very stable
- **Use**: Latent space exploration, NOT high-quality generation

**2. GAN (Generative Adversarial Network)**
- **Approach**: Adversarial game (generator vs discriminator)
- **Quality**: Sharp (9/10)
- **Training**: Unstable (adversarial dynamics)
- **Use**: High-quality generation, fast inference

**3. Diffusion Models**
- **Approach**: Iterative denoising
- **Quality**: Very sharp (9.5/10)
- **Training**: Stable
- **Use**: Modern default for high-quality generation

**4. Autoregressive Models**
- **Approach**: Sequential generation (pixel-by-pixel, token-by-token)
- **Quality**: Good (7-8/10)
- **Training**: Stable
- **Use**: Explicit likelihood, sequential data

**5. Flow Models**
- **Approach**: Invertible transformations
- **Quality**: Good (7-8/10)
- **Training**: Stable
- **Use**: Exact likelihood, invertibility needed

### Quick Comparison

| Model | Quality | Training Stability | Inference Speed | Mode Collapse | Likelihood |
|-------|---------|-------------------|----------------|---------------|------------|
| VAE | 6/10 (blurry) | 10/10 | Fast | No | Approximate |
| GAN | 9/10 | 3/10 | Fast | Yes | No |
| Diffusion | 9.5/10 | 9/10 | Slow | No | Approximate |
| Autoregressive | 7-8/10 | 9/10 | Very slow | No | Exact |
| Flow | 7-8/10 | 8/10 | Fast (both ways) | No | Exact |

---

## Part 2: VAE (Variational Autoencoder)

### Architecture

**Components:**
1. **Encoder**: x → z (image to latent)
2. **Latent space**: z ~ N(μ, σ²)
3. **Decoder**: z → x' (latent to reconstruction)

**Loss function:**
```python
# ELBO (Evidence Lower Bound)
loss = reconstruction_loss + KL_divergence

# Reconstruction: How well decoder reconstructs input
reconstruction_loss = MSE(x, x_reconstructed)

# KL: How close latent is to standard normal
KL_divergence = KL(q(z|x) || p(z))
```

### Why VAE is Blurry

**Problem**: MSE loss encourages pixel-wise averaging

**Example:**
- Dataset: Faces with both smiles and no smiles
- VAE learns: "Average face has half-smile blur"
- Result: Blurry, hedges between modes

**Mathematical reason:**
- MSE minimization = mean prediction
- Mean of sharp images = blurry image

### When to Use VAE

✅ **Use VAE for:**
- Latent space exploration (interpolation, arithmetic)
- Anomaly detection (reconstruction error)
- Disentangled representations (β-VAE)
- Compression (lossy, with latent codes)

❌ **DON'T use VAE for:**
- High-quality image generation (use GAN or Diffusion!)
- Sharp, realistic outputs

### Implementation

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = μ + σ * ε
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode
        h = self.fc_decode(z)
        h = h.view(-1, 128, 8, 8)
        x_recon = self.decoder(h)

        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss
```

---

## Part 3: GAN (Generative Adversarial Network)

### Architecture

**Components:**
1. **Generator**: z → x (noise to image)
2. **Discriminator**: x → [0, 1] (image to real/fake probability)

**Adversarial Training:**
```python
# Discriminator loss: Classify real as real, fake as fake
D_loss = -log(D(x_real)) - log(1 - D(G(z)))

# Generator loss: Fool discriminator
G_loss = -log(D(G(z)))

# Minimax game:
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

### Training Instability

**Problem**: Adversarial dynamics are unstable

**Common issues:**
1. **Mode collapse**: Generator produces limited variety
2. **Non-convergence**: Oscillation, never settles
3. **Vanishing gradients**: Discriminator too strong, generator can't learn
4. **Hyperparameter sensitivity**: Learning rates critical

**Solutions:**
- Spectral normalization (StyleGAN2)
- Progressive growing (start low-res, increase)
- Minibatch discrimination (penalize lack of diversity)
- Wasserstein loss (WGAN, more stable)

### Mode Collapse

**What is it?**
- Generator produces subset of distribution
- Example: Face GAN only generates 10 face types

**Why it happens:**
- Generator exploits discriminator weaknesses
- Finds "easy" samples that fool discriminator
- Forgets other modes

**Detection:**
```python
# Check diversity: Generate many samples
samples = generator.generate(n=1000)
diversity = compute_pairwise_distance(samples)
if diversity < threshold:
    print("Mode collapse detected!")
```

**Solutions:**
- Minibatch discrimination
- Unrolled GANs (slow but helps)
- Switch to diffusion (no mode collapse by design!)

### Modern GANs

**StyleGAN2 (2020):**
- State-of-the-art for faces
- Style-based generator
- Spectral normalization for stability
- Resolution: 1024×1024

**StyleGAN3 (2021):**
- Alias-free architecture
- Better animation/video

**When to use GAN:**
✅ Fast inference needed (50ms per image)
✅ Pretrained model available (StyleGAN2)
✅ Can tolerate training difficulty

❌ Training instability unacceptable
❌ Mode collapse problematic
❌ Starting from scratch (use diffusion instead)

### Implementation (Basic GAN)

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # 32x32 -> 64x64
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),            # 32x32 -> 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),           # 16x16 -> 8x8
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Training loop
for real_images in dataloader:
    # Train discriminator
    fake_images = generator(noise)
    D_real = discriminator(real_images)
    D_fake = discriminator(fake_images.detach())

    D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
    D_loss.backward()
    optimizer_D.step()

    # Train generator
    D_fake = discriminator(fake_images)
    G_loss = -torch.mean(torch.log(D_fake))
    G_loss.backward()
    optimizer_G.step()
```

---

## Part 4: Diffusion Models (Modern Default)

### Architecture

**Concept**: Learn to reverse a diffusion (noising) process

**Forward process** (fixed):
```python
# Gradually add noise to image
x_0 (original) → x_1 → x_2 → ... → x_T (pure noise)

# At each step:
x_t = √(1 - β_t) * x_{t-1} + √β_t * ε
where ε ~ N(0, I), β_t = noise schedule
```

**Reverse process** (learned):
```python
# Model learns to denoise
x_T (noise) → x_{T-1} → ... → x_1 → x_0 (image)

# Model predicts: ε_θ(x_t, t)
# Then: x_{t-1} = (x_t - √β_t * ε_θ(x_t, t)) / √(1 - β_t)
```

**Training:**
```python
# Simple loss: Predict the noise
loss = MSE(ε, ε_θ(x_t, t))

# x_t = noisy image at step t
# ε = actual noise added
# ε_θ(x_t, t) = model's noise prediction
```

### Why Diffusion is Excellent

**Advantages:**
1. **High quality**: State-of-the-art (better than GAN)
2. **Stable training**: Standard MSE loss (no adversarial dynamics)
3. **No mode collapse**: By design, covers full distribution
4. **Controllable**: Easy to add conditioning (text, class, etc.)

**Disadvantages:**
1. **Slow inference**: 50-1000 denoising steps (vs GAN's 1 step)
2. **Compute intensive**: T forward passes (T = 50-1000)

**Speed comparison:**
```
GAN: 1 forward pass = 50ms
Diffusion (T=50): 50 forward passes = 2.5 seconds
Diffusion (T=1000): 1000 forward passes = 50 seconds
```

**Speedup techniques:**
- DDIM (fewer steps, 10-50 instead of 1000)
- DPM-Solver (fast sampler)
- Latent diffusion (Stable Diffusion, denoise in latent space)

### Modern Diffusion Models

**Stable Diffusion (2022+):**
- Latent diffusion (denoise in VAE latent space)
- Text conditioning (CLIP text encoder)
- Pretrained on billions of images
- Fine-tunable

**DALL-E 2 (2022):**
- Prior network (text → image embedding)
- Diffusion decoder (embedding → image)

**Imagen (2022, Google):**
- Text conditioning with T5 encoder
- Cascaded diffusion (64×64 → 256×256 → 1024×1024)

**When to use Diffusion:**
✅ High-quality generation (best quality)
✅ Stable training (standard loss)
✅ Diversity needed (no mode collapse)
✅ Conditioning (text-to-image, class-conditional)

❌ Need fast inference (< 1 second)
❌ Real-time generation

### Implementation (DDPM)

```python
class DiffusionModel(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        # U-Net architecture
        self.model = UNet(
            in_channels=img_channels,
            out_channels=img_channels,
            time_embedding_dim=256
        )

    def forward(self, x_t, t):
        # Predict noise ε at timestep t
        return self.model(x_t, t)

# Training
def train_step(model, x_0):
    # Sample random timestep
    t = torch.randint(0, T, (batch_size,))

    # Sample noise
    ε = torch.randn_like(x_0)

    # Create noisy image x_t
    α_t = alpha_schedule[t]
    x_t = torch.sqrt(α_t) * x_0 + torch.sqrt(1 - α_t) * ε

    # Predict noise
    ε_pred = model(x_t, t)

    # Loss: MSE between actual and predicted noise
    loss = F.mse_loss(ε_pred, ε)
    return loss

# Sampling (generation)
@torch.no_grad()
def sample(model, shape):
    # Start from pure noise
    x_t = torch.randn(shape)

    # Iteratively denoise
    for t in reversed(range(T)):
        # Predict noise
        ε_pred = model(x_t, t)

        # Denoise one step
        α_t = alpha_schedule[t]
        x_t = (x_t - (1 - α_t) / torch.sqrt(1 - ᾱ_t) * ε_pred) / torch.sqrt(α_t)

        # Add noise (except last step)
        if t > 0:
            x_t += torch.sqrt(β_t) * torch.randn_like(x_t)

    return x_t  # x_0 (generated image)
```

---

## Part 5: Autoregressive Models

### Concept

**Idea**: Model probability as product of conditionals
```
p(x) = p(x_1) * p(x_2|x_1) * p(x_3|x_1,x_2) * ... * p(x_n|x_1,...,x_{n-1})
```

**For images**: Generate pixel-by-pixel (or patch-by-patch)

**Architectures:**
- **PixelCNN**: Convolutional with masked kernels
- **PixelCNN++**: Improved with mixture of logistics
- **VQ-VAE + PixelCNN**: Two-stage (learn discrete codes, model codes)
- **ImageGPT**: GPT-style Transformer for images

### Advantages

✅ **Explicit likelihood**: Can compute p(x) exactly
✅ **Stable training**: Standard cross-entropy loss
✅ **Theoretical guarantees**: Proper probability model

### Disadvantages

❌ **Very slow generation**: Sequential (can't parallelize)
❌ **Limited quality**: Worse than GAN/Diffusion for high-res
❌ **Resolution scaling**: Impractical for 1024×1024 (1M pixels!)

**Speed comparison:**
```
GAN: Generate 1024×1024 in 50ms (parallel)
PixelCNN: Generate 32×32 in 5 seconds (sequential!)
ImageGPT: Generate 256×256 in 30 seconds

For 1024×1024: 1M pixels × 5ms/pixel = 83 minutes!
```

### When to Use

✅ **Use autoregressive for:**
- Explicit likelihood needed (compression, evaluation)
- Small images (32×32, 64×64)
- Two-stage models (VQ-VAE + Transformer)

❌ **Don't use for:**
- High-resolution images (too slow)
- Real-time generation
- Quality-critical applications (use diffusion)

### Modern Usage

**Two-stage approach (DALL-E, VQ-GAN):**
1. **Stage 1**: VQ-VAE learns discrete codes
   - Image → 32×32 grid of codes (instead of 1M pixels)
2. **Stage 2**: Autoregressive model (Transformer) on codes
   - Much faster (32×32 = 1024 codes, not 1M pixels)

---

## Part 6: Flow Models

### Concept

**Idea**: Invertible transformations
```
z ~ N(0, I)  ←→  x ~ p_data

f: z → x (forward)
f⁻¹: x → z (inverse)
```

**Requirement**: f must be invertible and differentiable

**Advantage**: Exact likelihood via change-of-variables
```
log p(x) = log p(z) + log |det(∂f⁻¹/∂x)|
```

### Architectures

**RealNVP (2017):**
- Coupling layers (affine transformations)
- Invertible by design

**Glow (2018, OpenAI):**
- Actnorm, invertible 1×1 convolutions
- Multi-scale architecture

**When to use Flow:**
✅ Exact likelihood needed (better than VAE)
✅ Invertibility needed (both z→x and x→z)
✅ Stable training (standard loss)

❌ Architecture constraints (must be invertible)
❌ Quality not as good as GAN/Diffusion

### Modern Status

**Mostly superseded by Diffusion:**
- Diffusion has better quality
- Diffusion more flexible (no invertibility constraint)
- Flow models still used in specialized applications

---

## Part 7: Decision Framework

### By Primary Goal

```
Goal: High-quality images
→ Diffusion (modern default)
→ OR GAN if pretrained available

Goal: Fast inference
→ GAN (50ms per image)
→ Avoid Diffusion (too slow for real-time)

Goal: Training stability
→ Diffusion or VAE (standard loss)
→ Avoid GAN (adversarial training hard)

Goal: Latent space exploration
→ VAE (smooth interpolation)
→ Avoid GAN (no encoder)

Goal: Explicit likelihood
→ Autoregressive or Flow
→ For evaluation, compression

Goal: Diversity (no mode collapse)
→ Diffusion (by design)
→ OR VAE (stable)
→ Avoid GAN (mode collapse common)
```

### By Data Type

```
Images (high-quality):
→ Diffusion (Stable Diffusion)
→ OR GAN (StyleGAN2)

Images (small, 32×32):
→ Any model works
→ Try VAE first (simplest)

Audio waveforms:
→ WaveGAN
→ OR Diffusion (WaveGrad)

Video:
→ Video Diffusion (limited)
→ OR GAN (StyleGAN-V)

Text:
→ Autoregressive (GPT)
→ NOT VAE/GAN/Diffusion (discrete tokens)
```

### By Training Budget

```
Large budget (millions $, pretrain from scratch):
→ Diffusion (Stable Diffusion scale)
→ Billions of images, weeks on cluster

Medium budget (thousands $, train from scratch):
→ GAN or Diffusion
→ 10k-1M images, days on GPU

Small budget (hundreds $, fine-tune):
→ Fine-tune Stable Diffusion (LoRA)
→ 1k-10k images, hours on consumer GPU

Tiny budget (research, small scale):
→ VAE (simplest, most stable)
→ Few thousand images, CPU possible
```

### Modern Recommendations (2025)

**For new projects:**
1. **Default: Diffusion**
   - Fine-tune Stable Diffusion or train from scratch
   - Best quality + stability

2. **If need speed: GAN**
   - Use pretrained StyleGAN2 if available
   - Or train GAN (if can tolerate instability)

3. **If need latent space: VAE**
   - For interpolation, not generation quality

**AVOID:**
- Training GAN from scratch (unless necessary)
- Using VAE for high-quality generation
- Autoregressive for high-res images

---

## Part 8: Training from Scratch vs Fine-Tuning

### Stable Diffusion Example

**Pretraining (what Stability AI did):**
- Dataset: LAION-5B (5 billion images)
- Compute: 150,000 A100 GPU hours
- Cost: ~$600,000
- Time: Weeks on massive cluster
- **DON'T DO THIS!**

**Fine-tuning (what users do):**
- Dataset: 10k-100k domain images
- Compute: 100-1000 GPU hours
- Cost: $100-1,000
- Time: Days on single A100
- **DO THIS!**

**LoRA (Low-Rank Adaptation):**
- Efficient fine-tuning (fewer parameters)
- Dataset: 1k-5k images
- Compute: 10-100 GPU hours
- Cost: $10-100
- Time: Hours on consumer GPU (RTX 3090)
- **Best for small budgets!**

### Decision

```
Have pretrained model in your domain:
→ Fine-tune (don't retrain!)

No pretrained model:
→ Train from scratch (small model)
→ OR find closest pretrained and fine-tune

Budget < $1000:
→ LoRA fine-tuning
→ OR train small model (64×64)

Budget < $100:
→ LoRA with free Colab
→ OR VAE from scratch (cheap)
```

---

## Part 9: Common Mistakes

### Mistake 1: VAE for High-Quality Generation

**Symptom:** Blurry outputs
**Fix:** Use GAN or Diffusion for quality
**VAE is for:** Latent space, not generation

### Mistake 2: Ignoring Mode Collapse

**Symptom:** GAN generates same images
**Fix:** Spectral norm, minibatch discrimination
**Better:** Switch to Diffusion (no mode collapse)

### Mistake 3: Training Stable Diffusion from Scratch

**Symptom:** Burning money, poor results
**Fix:** Fine-tune pretrained model
**Reality:** Pretraining costs $600k+

### Mistake 4: Slow Inference with Diffusion

**Symptom:** 50 seconds per image
**Fix:** Use DDIM (fewer steps, 10-50)
**OR:** Use GAN if speed critical

### Mistake 5: Wrong Loss for GAN

**Symptom:** Training diverges
**Fix:** Use Wasserstein loss (WGAN)
**OR:** Spectral normalization
**Better:** Switch to Diffusion (standard loss)

---

## Summary: Quick Reference

### Model Selection

```
High quality + stable training:
→ Diffusion (modern default)

Fast inference required:
→ GAN (if pretrained) or trained GAN

Latent space exploration:
→ VAE

Explicit likelihood:
→ Autoregressive or Flow

Small images (< 64×64):
→ Any model (start with VAE)

Large images (> 256×256):
→ Diffusion or GAN (avoid autoregressive)
```

### Quality Ranking

```
1. Diffusion (9.5/10)
2. GAN (9/10)
3. Autoregressive (7-8/10)
4. Flow (7-8/10)
5. VAE (6/10 - blurry)
```

### Training Stability Ranking

```
1. VAE (10/10)
2. Diffusion (9/10)
3. Autoregressive (9/10)
4. Flow (8/10)
5. GAN (3/10 - very unstable)
```

### Modern Stack (2025)

```
Image generation: Stable Diffusion (fine-tuned)
Fast inference: StyleGAN2 (if available)
Latent space: VAE
Research: Diffusion (easiest to train)
```

---

## Next Steps

After mastering this skill:
- `llm-specialist/llm-finetuning-strategies`: Apply to text generation
- `architecture-design-principles`: Understand design trade-offs
- `training-optimization`: Optimize training for your chosen model

**Remember:** Diffusion models dominate in 2025. Use them unless you have specific reason not to (speed, latent space, likelihood).
