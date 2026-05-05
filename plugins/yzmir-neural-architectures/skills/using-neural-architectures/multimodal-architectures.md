
# Multimodal Architectures: Vision-Language and Beyond

## When to Use This Skill

Use this skill when you need to:
- ✅ Connect a vision encoder to a language model
- ✅ Choose between contrastive (CLIP-style) and generative (LLaVA-style)
  multimodal recipes
- ✅ Decide between bolt-on (projector / adapter / Q-Former) and
  trained-from-scratch native multimodal models
- ✅ Pick a vision encoder for a multimodal LLM (CLIP vs SigLIP vs DINOv2 vs
  EVA-02)
- ✅ Understand the modern vision-language architecture landscape (LLaVA,
  Flamingo, BLIP-2, Idefics, PaliGemma, Qwen-VL, native multimodal models)

**Do NOT use this skill for:**
- ❌ Pure-text LLMs (use `yzmir-llm-specialist`)
- ❌ Pure-vision tasks (use `cnn-families-and-selection.md` or
  `transformer-architecture-deepdive.md`)
- ❌ Audio-only or video-only models (covered briefly here for fusion;
  unimodal specialists in their respective sheets)


## Core Principle

There are essentially **three families** of vision-language architecture, and
they answer different questions:

1. **Contrastive (CLIP family).** "Are this image and this text the same
   thing?" → produces aligned image and text embeddings. Use for retrieval,
   zero-shot classification, and as the **vision encoder** for downstream
   multimodal LLMs.
2. **Generative bolt-on (LLaVA / BLIP-2 / Flamingo family).** "Given this
   image, write text about it." → take a frozen pretrained LLM and a frozen
   pretrained vision encoder, train only a small bridge between them. The
   dominant production recipe for vision-language assistants.
3. **Native multimodal (Gemini-style, Chameleon, NaVL family).** Train one
   Transformer end-to-end on interleaved tokens from multiple modalities.
   More expensive, often higher ceiling.

Almost every multimodal system you'll touch is one of these three; the rest
of this sheet is about choosing between them and choosing the components
inside.


## Part 1: Contrastive Vision-Language (CLIP family)

### CLIP

**Paper:** Radford et al. — *Learning Transferable Visual Models From Natural
Language Supervision* (CLIP, ICML 2021).

Two encoders (image, text) trained with a contrastive loss on 400M
image-text pairs scraped from the web. After training, image and text
embeddings live in a shared space.

**Why it changed everything:**
- Zero-shot image classification: encode "a photo of a {class}" for each
  class, take argmax cosine similarity. No fine-tuning needed.
- Strong **off-the-shelf image encoder** for downstream multimodal LLMs.
  Almost every LLaVA-class model uses a CLIP or CLIP-derivative ViT.

### OpenCLIP / LAION-CLIP

Open-weights reproductions of CLIP at multiple scales (ViT-B/32 up to
ViT-G/14) trained on the LAION-2B / LAION-5B datasets. The de facto
open-source CLIP.

### SigLIP / SigLIP 2

**Papers:** Zhai et al. — *Sigmoid Loss for Language-Image Pre-Training*
(SigLIP, ICCV 2023); Tschannen et al. — *SigLIP 2: Multilingual
Vision-Language Encoders with Improved Semantic Understanding,
Localization, and Dense Features* (2025).

**Key change:** Replace CLIP's softmax contrastive loss with a per-pair
**sigmoid loss**. This decouples the loss from batch size — you can train
with smaller batches without hurting alignment, and large batches still
work. Result: better quality at the same compute, especially for smaller
models.

**SigLIP is the default modern choice over CLIP** when you want a
contrastive vision-language encoder. PaliGemma, Idefics3, several Qwen-VL
variants, and many recent VLMs use SigLIP rather than the original CLIP.

### EVA-CLIP

**Paper:** Sun et al. — *EVA-CLIP: Improved Training Techniques for CLIP at
Scale* (2023).

CLIP-style training initialized from EVA-02 ViT weights with stronger
optimizer + LAMB. State-of-the-art among open contrastive models at the
multi-billion-parameter scale.

### When to pick which contrastive encoder

| Need | Pick |
|------|------|
| Strong general-purpose VL encoder, balanced size | **SigLIP** ViT-L or SigLIP 2 |
| Largest open contrastive encoder, max quality | **EVA-CLIP** ViT-G |
| Compatibility with the OpenCLIP ecosystem and tooling | **OpenCLIP** ViT-L/14 or ViT-H/14 |
| Best dense / pixel-level features (segmentation, detection downstream) | **DINOv2** (self-supervised, *not* contrastive — see ViT section) |
| Multilingual VL retrieval | SigLIP 2 |


## Part 2: Generative Bolt-on Multimodal (LLaVA / BLIP-2 / Flamingo)

This is the dominant production pattern as of 2024-2026: take a strong
pretrained LLM, take a strong pretrained vision encoder, **train only a
small connector** between them on a relatively modest amount of multimodal
instruction data. Three connector designs to know:

### 2.1 Linear / MLP projector — LLaVA family

**Papers:**
- Liu et al. — *Visual Instruction Tuning* (LLaVA, NeurIPS 2023)
- Liu et al. — *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5,
  CVPR 2024)
- Li et al. — *LLaVA-NeXT* (2024) — higher-resolution, more careful data
- Li et al. — *LLaVA-OneVision* (2024) — image, multi-image, video unification

**Architecture:**

```
Image  ─→ Vision encoder (CLIP/SigLIP ViT, frozen during stage 1)
         ─→ Projector (a 2-layer MLP — really)
            ─→ Image tokens
                ↘
                  Text tokens ─→ LLM (Vicuna / LLaMA / Qwen, frozen stage 1)
                                   ─→ Generated text
```

**Training stages:**
1. Pretrain only the projector on image-caption pairs (cheap)
2. Fine-tune projector + LLM on visual instruction-following data

**Why it works so well:** The CLIP/SigLIP image embedding is already aligned
to language semantics. A trivial 2-layer MLP is enough to map it into the
LLM's input embedding space. You get a competitive vision-language model for
~$1k of fine-tuning compute.

**This is the modern default recipe.** Idefics3, Qwen-VL (early variants),
InternVL, the Llama 3.2-V family, and many domain VLMs are all variations on
the LLaVA template.

### 2.2 Q-Former — BLIP-2

**Paper:** Li et al. — *BLIP-2: Bootstrapping Language-Image Pre-training
with Frozen Image Encoders and Large Language Models* (ICML 2023).

**Architecture:** Between a frozen vision encoder and a frozen LLM, insert a
small **Querying Transformer (Q-Former)** with a fixed set of learnable
query tokens that attend to image features and produce a compact set of
language-aligned tokens.

**Why it existed:** When BLIP-2 came out, dumping hundreds of patch tokens
straight into an LLM was prohibitively expensive. The Q-Former compresses
to a fixed small set (e.g., 32 tokens).

**Status (2026):** Q-Former is conceptually elegant but largely supplanted
in practice by:
- Plain MLP projectors (LLaVA family — simpler, often matches Q-Former when
  data is sufficient)
- Resampling schemes / Perceiver-style fixed-token outputs (used in
  Flamingo, Idefics, Mini-CPM-V) when token-budget pressure remains.

Worth knowing as a reference design and for low-token-budget settings.

### 2.3 Cross-attention / Perceiver-Resampler — Flamingo / Idefics

**Papers:**
- Alayrac et al. — *Flamingo: a Visual Language Model for Few-Shot Learning*
  (NeurIPS 2022)
- Laurençon et al. — *OBELICS: An Open Web-Scale Filtered Dataset of
  Interleaved Image-Text Documents* and the Idefics 1/2/3 model family
  (2023-2024)

**Architecture:**

```
Image  ─→ Vision encoder
         ─→ Perceiver-Resampler (compress to fixed-length visual tokens)
            ─→ injected via NEW cross-attention layers inserted into a
               frozen pretrained LLM
                ↘
                  Text tokens ─→ frozen LLM (with new XATTN layers active)
```

**Differences from LLaVA:**
- Does not concatenate visual tokens into the input sequence; instead injects
  them via cross-attention layers added between the LLM's existing decoder
  blocks.
- Naturally supports interleaved multi-image / multi-turn input (images can
  appear anywhere in the text stream).

**When to pick this style:** Multi-image / interleaved-document inputs;
tasks where you want to bolt vision onto an LLM without changing its
self-attention layers.

### Bolt-on family — what to actually pick (2026)

| Need | Recipe |
|------|--------|
| Best general single-image VL assistant, smallest budget | LLaVA-NeXT / LLaVA-OneVision style: SigLIP + MLP + open LLM |
| High-res images / OCR-heavy tasks | LLaVA-NeXT or InternVL with patch-tiling |
| Multi-image / interleaved docs | Idefics3-style cross-attention (Flamingo lineage) |
| Token-budget-constrained edge | Q-Former or Perceiver-Resampler to fixed token count |
| You already have great in-domain text data and a frozen open LLM | LLaVA-style MLP projector, fine-tune end-to-end on instruction data |

### High-resolution and patch tiling

A core practical issue: vision encoders are usually pretrained at 224² or
336². For dense text in images (charts, documents, OCR), you want
1000²+ effective resolution. Common solutions:

- **Patch tiling** (LLaVA-NeXT, InternVL, Qwen-VL): tile the input image into
  several encoder-resolution sub-images, embed each, concatenate as visual
  tokens, plus a low-res thumbnail.
- **Native high-res ViT**: use a ViT pretrained at 384²/448² (e.g., SigLIP
  or DINOv2 large variants).

These are not architectural revolutions but they're the difference between
a VL model that can read invoices and one that can't. Mention them when
choosing a recipe.


## Part 3: Native Multimodal Models

End-to-end training where multiple modalities share one Transformer trunk
from the start, rather than bolting modalities onto a frozen LLM.

### Chameleon

**Paper:** Chameleon Team (Meta) — *Chameleon: Mixed-Modal Early-Fusion
Foundation Models* (2024).

Unified token vocabulary (text tokens + VQ image tokens), single
Transformer, autoregressive over both modalities. Generates text and images
in one decoder.

### Gemini-style native multimodal

Google's Gemini (Anil et al. technical report, 2023; updated reports
through 2024-2025) is described as natively multimodal across text,
images, audio, and video using a single Transformer trunk with
modality-specific tokenizers. The architectural details published are
limited; the takeaway is that frontier-scale **closed** models have largely
moved to native-multimodal training.

### When native multimodal pays off

| Condition | Pay off? |
|-----------|----------|
| Frontier-scale training budget (>$10M class) | ✅ Higher ceiling than bolt-on |
| Targeting cross-modal generation (text+image out) | ✅ Bolt-on doesn't naturally generate images |
| Need cross-modal in-context learning | ✅ Stronger than projector approach |
| Production fine-tune on a budget | ❌ Use bolt-on (LLaVA-style) instead |
| Edge / on-device | ❌ Bolt-on is much smaller and easier to ship |

**Practical rule:** Frontier closed models are increasingly native
multimodal. Almost every open-source VL model you'll touch is bolt-on.
Recommend native multimodal only for very-large-scale work or when you need
cross-modal generation (text+image out from one model).


## Part 4: Beyond Vision-Language

### Audio-visual / video-language

- **Whisper** (Radford et al. 2022): audio encoder, can be bolted to an LLM
  the same way CLIP/SigLIP are (audio embeddings → projector → LLM).
- **Qwen2-Audio**, **Qwen2.5-Omni** (Alibaba 2024-2025): treat audio as
  another modality with its own tokenizer/encoder, bolt-on into a multimodal
  LLM.
- **VideoLLaMA**, **LLaVA-NeXT-Video**, **LLaVA-OneVision**: extend the
  LLaVA recipe to sampled video frames; primary architectural question is
  how to handle long sequences of frame tokens (resampling, temporal
  pooling, sliding windows).

### Generic fusion strategies (when you don't have a foundation model)

When neither modality has a strong pretrained encoder for your domain
(rare in 2026, but happens with novel sensor data):

| Fusion | When to use |
|--------|------------|
| **Early fusion** — concatenate raw inputs / patches | Small models, modalities are tightly correlated, plenty of joint data |
| **Late fusion** — separate encoders, combine at the head | Modalities are independent, want to reuse unimodal pretrained encoders |
| **Cross-attention fusion** — encoder per modality, attention bridges | Modalities each rich enough to be encoded separately, but interactions matter (Flamingo pattern) |
| **Tokenize-and-stitch** — separate tokenizers, one shared decoder | Native-multimodal pattern (Chameleon / Gemini); works at scale |

For most real work, **don't design fusion from scratch** — pick the closest
foundation-model recipe (LLaVA-style for VL, Qwen2.5-Omni for audio+vision,
etc.) and adapt.


## Decision Tree

```
You need a multimodal system. What's the primary task?

├─ Zero-shot retrieval / classification (image ↔ text)
│  └─ Use CLIP-family encoder
│     ├─ Default: SigLIP / SigLIP 2
│     ├─ Largest open: EVA-CLIP
│     └─ Ecosystem fit: OpenCLIP
│
├─ Visual question answering / image-grounded chat
│  └─ Use bolt-on VL recipe (LLaVA family)
│     ├─ Single image, general → LLaVA-NeXT / LLaVA-OneVision
│     ├─ Multi-image / interleaved docs → Idefics3 (Flamingo lineage)
│     ├─ OCR / chart / document-heavy → patch-tiling LLaVA-NeXT or InternVL
│     └─ Token-budget constrained → BLIP-2 / Q-Former or Perceiver-Resampler
│
├─ Generate images from text
│  └─ Use diffusion (see generative-model-families.md), not VL
│
├─ Cross-modal generation (text+image in same model, in/out)
│  └─ Native multimodal (Chameleon, Gemini-class)
│     — generally only worth it at frontier scale
│
├─ Audio + text
│  └─ Whisper bolt-on to LLM, or Qwen2-Audio / Qwen2.5-Omni recipe
│
├─ Video + text
│  └─ LLaVA-NeXT-Video / LLaVA-OneVision; resample frames, watch token count
│
└─ Novel sensor + text (no good pretrained encoder)
   └─ Train a unimodal encoder first (sensor → embeddings),
      then apply LLaVA-style MLP-projector recipe
```


## Common Mistakes

1. **Picking CLIP for the dense-prediction backbone.** CLIP/SigLIP are great
   for classification and as VL connectors but **DINOv2 dominates** for
   dense pixel-level features (segmentation, depth). If you're feeding
   features to a segmentation head, prefer DINOv2.
2. **Training a Q-Former when an MLP projector would do.** The LLaVA result
   showed that a 2-layer MLP between SigLIP and an LLM is enough for most
   VL chat tasks. Q-Former adds complexity that only pays off when token
   budget is tight.
3. **Going native multimodal without frontier-scale data and compute.**
   Bolt-on recipes (LLaVA-style) are the right choice below frontier scale;
   native multimodal needs much more multimodal data to break even.
4. **Ignoring resolution.** A SigLIP-336 + MLP + LLaMA stack will not read
   small text in document images, no matter how good the LLM is. Use
   patch tiling or a high-res vision encoder.
5. **Over-resampling visual tokens.** Compressing to too few tokens (e.g.,
   <16 per image with Q-Former) loses fine detail. The trend is *more*
   visual tokens, not fewer, as LLM context length has grown.


## Integration with Other Skills

- For the **vision encoder** internals (ViT, DINOv2, SigLIP, MAE), see the
  ViT section of [transformer-architecture-deepdive.md](transformer-architecture-deepdive.md).
- For **CNN backbones** sometimes used in multimodal stacks (ConvNeXt v2,
  EfficientNetV2), see [cnn-families-and-selection.md](cnn-families-and-selection.md).
- For the **LLM trunk** of a bolt-on VL model (architecture, fine-tuning,
  RAG, evaluation), see `yzmir-llm-specialist`.
- For **deployment / quantization** of a bolt-on VL model, see
  `yzmir-ml-production`.


**Remember:**
- 2026 default for "I want a vision-language model": **SigLIP + MLP
  projector + open LLM**, fine-tuned with LLaVA-style instruction data.
- Native multimodal is for frontier scale and cross-modal generation, not
  for typical production work.
- The vision encoder choice (CLIP vs SigLIP vs DINOv2 vs EVA) often matters
  more than the connector design.
