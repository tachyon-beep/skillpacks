
# LLM Fine-Tuning Strategies

## Context

You're considering fine-tuning an LLM, or debugging one already in flight. Common mistakes:
- **Fine-tuning when prompts (or RAG, or a stronger model) would work** — the most common and most expensive mistake.
- **Full fine-tuning when LoRA/QLoRA would match it** — a 100× cost premium for a few percent.
- **Poor data quality** — garbage in, garbage out, regardless of method.
- **Wrong learning rate** — easily 100× off; the dominant cause of catastrophic forgetting.
- **Stopping at SFT when a preference step would help** — or jumping straight to RLHF when SFT alone would have been fine.
- **Picking the algorithm by hype, not by fit** (DPO vs. KTO vs. GRPO are not interchangeable).

**This skill provides decision criteria for when (and what) to fine-tune, the modern preference-tuning lineage (PPO → DPO → IPO/KTO/SimPO/ORPO → GRPO), the LoRA family (LoRA, QLoRA, DoRA, rsLoRA, LoftQ, LongLoRA), and the practical stacks (TRL, Axolotl, Unsloth, LLaMA-Factory).**

This sheet covers **fine-tuning strategy and method choice**. Training-dynamics specifics (FSDP2, FP8, optimizer-state sharding, learning-rate transfer via muP) live in `yzmir-training-optimization` — cross-ref, don't duplicate.


## Decision Tree: Don't Fine-Tune If You Don't Have To

The order of operations has not changed; the threshold for "good enough without fine-tuning" has risen sharply with frontier models.

### Step 1: Try a stronger prompt with a stronger model

```
- Use a frontier instruction-tuned model in its highest-capability tier.
- System message + 3-8 few-shot examples + clear output schema.
- Temperature 0 (or low) for consistency; chain-of-thought or structured
  output schema where appropriate.
```

If quality clears your bar (commonly ≥ 90% of human-rated examples) → **stop**. You don't need fine-tuning.

### Step 2: Add RAG, reasoning, or tool use

If the failure mode is "model doesn't know X," the answer is almost always RAG, not fine-tuning. See `rag-architecture-patterns.md`.

If the failure mode is "model doesn't think hard enough," see `reasoning-models.md` (extended thinking, reasoning-tier models).

If it's "model can't act on the world," see `agentic-patterns-and-mcp.md`.

### Step 3: Consider fine-tuning

Fine-tune when:
- ✅ Prompts demonstrably fail after honest iteration.
- ✅ You have ≥ 1,000 high-quality examples (or can build them).
- ✅ You need consistent behavior that prompts cannot deliver.
- ✅ You need to reduce per-call latency / cost (shorter prompts → cheaper inference, or distill to a smaller model).
- ✅ You need to teach a capability or domain pattern not in the base model.

Don't fine-tune for:
- ❌ Tone or style (use system prompt + examples).
- ❌ Output formatting (use structured-output / JSON schema).
- ❌ Very small datasets (< 100 examples).
- ❌ Recent or changing facts (use RAG).
- ❌ Quick experiments (prompts iterate in minutes).


## What Kind of Fine-Tuning Do You Actually Need?

Three distinct things get called "fine-tuning"; conflating them is half the field's confusion.

| Goal | Method family | When |
|------|---------------|------|
| Teach a format, domain, or new task | **SFT** (supervised) on instruction/response pairs | Most "fine-tuning" requests |
| Align outputs with human (or AI) preferences | **Preference tuning** (DPO, IPO, KTO, SimPO, ORPO) on preference pairs | After SFT, when ranked-pairs data exists |
| Optimize against a reward signal (reasoning, tool success, evals) | **Online RL** (PPO, GRPO) | Reasoning models, tool-use agents, when reward is computable |

The standard modern recipe for an instruction-following model: SFT → DPO (or SimPO/ORPO). For a reasoning model: SFT → GRPO with verifiable rewards. For frontier alignment (harmlessness, helpfulness): SFT → preference tuning, often with RLAIF / Constitutional AI.


## Preference-Tuning Lineage

### PPO + RLHF — InstructGPT (the predecessor)

Ouyang et al., 2022 — <https://arxiv.org/abs/2203.02155>. Three stages:
1. **SFT** on demonstrations.
2. **Reward model** trained on human pairwise preferences.
3. **PPO** to maximize reward while a KL penalty pins the policy near the SFT model.

This is what trained ChatGPT. It works, but it's complex (four models in memory: policy, reference, reward, value), unstable, and hyperparameter-sensitive. Almost no one runs vanilla PPO+RM-RLHF in 2026 outside frontier labs and reasoning training (where it morphed into GRPO).

### DPO — the new default

Rafailov et al., NeurIPS 2023 — <https://arxiv.org/abs/2305.18290>. Direct Preference Optimization shows that the constrained-reward problem RLHF solves has a closed-form optimum, so you can skip the reward model and PPO entirely. You train directly on preference pairs `(prompt, chosen, rejected)` with a contrastive loss against a frozen reference model.

DPO is now the standard preference-tuning algorithm: stable, fits in standard SFT infrastructure, no sampling rollouts, no value model. Use TRL's `DPOTrainer`.

### Variants worth knowing (each fixes a specific DPO weakness)

| Method | Citation | One-liner |
|--------|----------|-----------|
| **IPO** | Azar et al., AISTATS 2024 — <https://arxiv.org/abs/2310.12036> | DPO can over-fit on near-deterministic preferences; IPO replaces DPO's logit with identity, regularizing better. Use when preference data is noisy or has many ties. |
| **KTO** | Ethayarajh et al., ICML 2024 — <https://arxiv.org/abs/2402.01306> | Drops the *pairwise* requirement: needs only a binary "good / bad" label per response, using Kahneman–Tversky prospect-theory utility. Use when you have thumbs-up/down logs but no clean pairs. |
| **SimPO** | Meng et al., NeurIPS 2024 — <https://arxiv.org/abs/2405.14734> | **Reference-free** DPO variant; uses average log-prob per token as the implicit reward and adds a target margin. Saves the reference-model memory + forward pass; reported to outperform DPO on AlpacaEval 2 / Arena-Hard. |
| **ORPO** | Hong et al., EMNLP 2024 — <https://arxiv.org/abs/2403.07691> | **Monolithic**: combines SFT and preference signal in a single loss via an odds-ratio term; no separate SFT phase, no reference model. Use when you want one training run instead of two. |

**Practical decision matrix:**

```
Have clean pairwise preferences, want the safe default?         → DPO
Have only binary good/bad labels (thumbs up/down logs)?         → KTO
GPU memory tight or want to skip reference-model forward pass?  → SimPO
Want SFT + preference in one training run?                      → ORPO
Preferences are noisy or ties are common?                       → IPO
```

All of these are implemented in [TRL](https://huggingface.co/docs/trl) under their respective trainers (`DPOTrainer`, `KTOTrainer`, `ORPOTrainer`, `SimPO` via DPOTrainer with reference-free flags), in [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), and in [Unsloth](https://github.com/unslothai/unsloth).

### GRPO — the algorithm behind reasoning-model RL

Shao et al., DeepSeekMath, 2024 — <https://arxiv.org/abs/2402.03300>. Group Relative Policy Optimization is the on-policy RL algorithm that powers DeepSeek-R1 and most modern reasoning RL stacks.

GRPO drops PPO's value/critic model. For each prompt, sample a *group* of G outputs, score each with a (potentially programmatic) reward, and use the **group's normalized rewards** as advantages — the baseline is the group mean. This makes RL fine-tuning practical without training a separate value head, which roughly halves memory footprint vs. PPO.

GRPO shines when the reward is **verifiable** (math answer correctness, code execution, tool-call success) — these are exactly the regimes driving the reasoning-model wave. Implementations: TRL `GRPOTrainer`, Unsloth (single-GPU GRPO), Axolotl (production GRPO).

```python
# Sketch of TRL GRPOTrainer usage; check current TRL docs for exact arguments.
from trl import GRPOConfig, GRPOTrainer

def reward_correct_answer(prompts, completions, **_):
    # Returns a list[float] — one reward per completion.
    return [1.0 if check_answer(c) else 0.0 for c in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    reward_funcs=[reward_correct_answer],
    args=GRPOConfig(
        num_generations=8,           # group size G
        learning_rate=5e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        beta=0.04,                   # KL coefficient
    ),
    train_dataset=ds,
)
trainer.train()
```

When to reach for GRPO: reasoning training, tool-use agents, anywhere you can write a verifiable reward function. Cross-ref `yzmir-training-optimization` for the training-dynamics knobs (KL schedule, group size, reward shaping) and `reasoning-models.md` (this campaign) for the reasoning-tier model context.

### Constitutional AI / RLAIF

Bai et al., 2022 — <https://arxiv.org/abs/2212.08073>. Replace human preference labelers with an AI judge guided by a written constitution (a list of principles). The model self-critiques and revises during SFT, then RLAIF does preference training using AI-generated preferences. Used in production at Anthropic and increasingly elsewhere; useful when human labeling is the bottleneck or coverage of edge cases is impractical.

The technique generalizes: any time you have a stronger model that can rank outputs more cheaply than humans, RLAIF is on the table. DPO/SimPO/ORPO over AI-judged preferences is now common.


## When to use what — decision matrix

| Situation | SFT alone | SFT + DPO/SimPO/ORPO | SFT + GRPO | RLHF (PPO+RM) |
|-----------|-----------|----------------------|------------|---------------|
| Format / schema / domain task, 1k–10k examples | ✅ | optional | — | — |
| Want preference alignment, have pair data | — | ✅ | — | rare |
| Reasoning, math, code, verifiable reward | start with SFT | optional | ✅ | rare |
| Frontier helpfulness/harmlessness alignment | — | ✅ (often with RLAIF) | sometimes | ✅ (frontier labs) |
| Budget is small, single-GPU | ✅ | ✅ (Unsloth) | ✅ (Unsloth GRPO) | ❌ |

**Honest "when not to bother fine-tuning at all":**
- Your data has < 1,000 examples and you haven't tried 8-shot prompting on the strongest model.
- You haven't tried RAG and the problem is "the model doesn't know X."
- You're trying to fix tone, voice, or output formatting (system prompt + structured output is the answer).
- Your eval is "vibes" — without a measurable target you can't tell if FT helped.


## LoRA Family — Modern PEFT

Full fine-tuning rewrites every weight; **PEFT** (Parameter-Efficient Fine-Tuning) freezes the base and learns small adapters. This is the right default for ~99% of fine-tuning, including all the preference and RL methods above.

### LoRA (the foundation)

Hu et al., ICLR 2022 — <https://arxiv.org/abs/2106.09685>. For a frozen weight `W ∈ R^{d×k}`, learn `ΔW = B A` with `A ∈ R^{r×k}, B ∈ R^{d×r}`, `r << min(d, k)`. Train only `A, B`. Memory and compute drop by ~100×; quality is usually within a couple percent of full FT.

```python
from peft import LoraConfig, get_peft_model

cfg = LoraConfig(
    r=16,                  # 8-32 typical; higher with rsLoRA, see below
    lora_alpha=32,         # commonly 2*r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # all attention + MLP
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, cfg)
model.print_trainable_parameters()
```

**Rank guidelines (vanilla LoRA):**
- `r=4–8`: small task, simple format/style adjustments.
- `r=16`: default for most domain SFT.
- `r=32–64`: complex tasks, large datasets — but see rsLoRA below before going high.

**Target modules:** target attention + MLP projections (`q,k,v,o,gate,up,down`). Targeting only `q,v` (the original LoRA paper's choice) leaves quality on the table on modern LLMs.

### QLoRA — the cost-saver

Dettmers et al., NeurIPS 2023 — <https://arxiv.org/abs/2305.14314>. LoRA + 4-bit NF4 quantization of the frozen base. Same quality as LoRA in the paper; ~4× less memory, lets you fit a 70B-class model on a single 48GB GPU and a small (7-8B) model on 16GB.

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",  # or any current open base
    quantization_config=bnb,
    device_map="auto",
)
model = get_peft_model(model, lora_config)
```

**Memory comparison (rough, bfloat16 base, AdamW optimizer):**

| Method | 7-8B model | 70B model | 405B model |
|--------|-----------|-----------|------------|
| Full FT (FP16, AdamW) | ~120 GB | ~1 TB | ~6 TB |
| LoRA | ~28 GB | ~180 GB | ~900 GB |
| QLoRA | ~10–14 GB | ~48 GB | ~240 GB |

These are coarse capability tiers — exact numbers depend on sequence length, batch size, optimizer, gradient checkpointing, and ZeRO/FSDP sharding. For multi-GPU training detail, cross-ref `yzmir-training-optimization`.

### LoRA-family advances

| Method | Citation | What it gives you |
|--------|----------|-------------------|
| **DoRA** | Liu et al., ICML 2024 (Oral) — <https://arxiv.org/abs/2402.09353> | Decomposes pretrained weight into magnitude + direction; LoRA updates only the direction. Consistently beats LoRA at the same rank with no extra inference cost. Supported in PEFT (`use_dora=True`). Use when you have budget for a slightly-slower-to-train run and want LoRA-quality closer to full FT. |
| **rsLoRA** | Kalajdzievski, 2023 — <https://arxiv.org/abs/2312.03732> | Shows the standard `α/r` scaling stunts learning at high rank; replacing with `α/√r` ("rank-stabilized") unlocks high-rank LoRA. Use when you want to push `r` to 64+ for complex tasks. PEFT supports it via `use_rslora=True`. |
| **LoftQ** | Li et al., ICLR 2024 — <https://arxiv.org/abs/2310.08659> | Quantization-aware *initialization* for QLoRA: jointly chooses 4-bit weights and LoRA factors so their sum approximates the FP16 weight. Closes much of QLoRA's small quality gap to LoRA. Use when QLoRA is underperforming LoRA on your eval. |
| **LongLoRA** | Chen et al., ICLR 2024 — <https://arxiv.org/abs/2309.12307> | Long-context LoRA via shifted-sparse attention during training (S²-Attn) plus tuning of embeddings and norms. Extends Llama-2 7B from 4k → 100k context with LoRA-style cost. Use when extending context length, not when fine-tuning at the base context. |

**Default recommendation in 2026:** **vanilla LoRA + QLoRA is still the right starting point**. Reach for DoRA when you want a quality bump at the same rank, rsLoRA when going to high rank, LoftQ when you specifically need to close a QLoRA-vs-LoRA gap, and LongLoRA when you're extending context.

### Full fine-tuning

Still has its place: you have ≥ 100k high-quality examples, a multi-GPU cluster, and you need behavioral changes that adapters can't fully capture (deep distribution shift, e.g., a new modality token, a different tokenizer's coverage). For everything else, PEFT.


## Modern Stacks

| Stack | Repo | When to use it |
|-------|------|----------------|
| **TRL** (Hugging Face) | <https://github.com/huggingface/trl> | The reference implementation of every preference and RL trainer (SFT, DPO, IPO, KTO, ORPO, GRPO, PPO, RLOO). Use when you need the latest algorithm or a custom reward function. |
| **Axolotl** | <https://github.com/axolotl-ai-cloud/axolotl> | YAML-config-driven, production-leaning, multi-GPU first-class. Supports SFT/DPO/ORPO/GRPO and quantization-aware training. Use for reproducible production fine-tuning pipelines. |
| **Unsloth** | <https://github.com/unslothai/unsloth> | Hand-tuned single-GPU LoRA/QLoRA with custom Triton kernels. ~2× faster, ~50% less memory than baseline at time of writing; supports DPO and GRPO. Use when you have one GPU and need it to count. |
| **LLaMA-Factory** | <https://github.com/hiyouga/LLaMA-Factory> | Web UI + CLI, 100+ supported model families, zero-code workflows. Use for fast prototyping and for teams without ML infra engineers. |

Hardware-side, **FlashAttention-3** (Shah, Dao, et al., NeurIPS 2024 — <https://arxiv.org/abs/2407.08608>) gives a 1.5–2× speedup on H100 over FA2, with FP8 support. Most of the stacks above pick it up automatically when available. For attention variants beyond FA, see `yzmir-pytorch-engineering` (FlexAttention).


## Dataset Preparation

**Quality > quantity.** This has not changed and will not change. 1,000 clean, diverse, domain-representative examples beat 50,000 noisy ones almost every time. The most reliable single intervention available to most fine-tuners is "spend more time cleaning data."

### Sources

Good: human-labeled, expert-written, validated production logs, frontier-model-distilled with human review.

Bad: raw logs without filtering, scraped without quality control, automated generation without validation, anything containing PII you don't have rights to.

### Cleaning checklist

```python
def clean_example(ex: dict) -> bool:
    if len(ex["input"]) < 10 or len(ex["output"]) < 10:        return False
    if len(ex["input"]) > 8000 or len(ex["output"]) > 8000:    return False
    if any(bad in ex["output"].lower()
           for bad in ["error", "exception", "failed to"]):    return False
    if ex["output"] == ex["input"]:                            return False
    if not ex["output"].strip().endswith((".", "!", "?", "}", "]", ")")):
        return False
    return True
```

Then **always spot-check 100+ random examples by hand**. There is no substitute. Watch for: copy-paste artifacts, leaked system prompts, format inconsistencies, label errors, stale references.

### Dataset format

OpenAI / Anthropic chat format (`[{role, content}, …]`) is now the de facto standard. Hugging Face `datasets` + a `tokenizer.apply_chat_template` call handles the conversion to whatever the base model expects.

```python
from datasets import Dataset

def to_messages(ex):
    return {"messages": [
        {"role": "system",    "content": ex["system"]},
        {"role": "user",      "content": ex["input"]},
        {"role": "assistant", "content": ex["output"]},
    ]}

ds = Dataset.from_list(records).map(to_messages)
```

For preference data (DPO/IPO/SimPO/ORPO):

```python
preference_record = {
    "prompt":   "...",
    "chosen":   "...",
    "rejected": "...",
}
```

For KTO, just `(prompt, completion, label: bool)`.

For GRPO, `(prompt,)` only — completions are sampled by the trainer; rewards are computed by your reward function.

### Splits

70/15/15 for small datasets (< 5k), 80/10/10 for larger. Hold the test set out — never tune on it. For preference data, split by *prompt* not by individual pair, to avoid leakage.


## Hyperparameters

### Learning rate — the one that actually matters

The single most important knob. **Fine-tuning learning rates are 100–1000× smaller than pre-training rates.** Order of magnitude is the dominant signal; the second decimal place is noise.

| Method | Typical LR (capability tier-aware) |
|--------|------------------------------------|
| SFT, full FT | ~ 1/100 of the model's pre-training LR |
| LoRA / QLoRA SFT | 1e-4 to 5e-5 (LoRA can take higher LR than full FT — adapters are small) |
| DPO / SimPO / ORPO | 5e-7 to 5e-6 (much lower than SFT — preference loss is tiny) |
| KTO | 5e-7 to 5e-6 |
| GRPO / PPO | 5e-7 to 5e-6 |

For exact base-model-LR transfer, see **muP / mu-Transfer** (Yang et al., NeurIPS 2021 — <https://arxiv.org/abs/2203.03466>) — it lets you tune at small scale and transfer the LR to large. Cross-ref `yzmir-training-optimization` for the full muP / muTransfer recipe.

**Symptoms of wrong LR:**
- Loss oscillates / spikes / NaN → LR too high.
- Loss barely moves → LR too low or schedule wrong.
- Catastrophic forgetting (model forgets general capability) → LR too high or too many epochs (these often coincide).

### Schedule

Cosine decay with 3–10% warmup is the standard default. Linear is fine for short runs. Constant-with-warmup is sometimes used in DPO.

### Epochs

| Dataset size | Epochs (SFT) |
|--------------|--------------|
| < 1k | 3–5 |
| 1k–10k | 2–3 |
| 10k–100k | 1–2 |
| > 100k | 1 (or fewer) |

For DPO/SimPO/ORPO: 1–3 epochs over preference pairs is plenty; more usually overfits.

For GRPO: train for *steps*, not epochs — monitor reward and KL divergence and stop when reward plateaus or KL blows up.

### Batch size, gradient accumulation, weight decay

Effective batch sizes of 64–256 are typical for SFT, 16–64 for preference tuning, 32–128 (groups × batch) for GRPO. Use gradient accumulation aggressively when memory is tight. Weight decay 0.01–0.1; use lower with LoRA (the adapters are already a regularizer of sorts).

For deeper training-dynamics tuning (gradient clipping, optimizer choice — Lion vs. AdamW vs. AdamW-8bit, FP8 mixed precision, FSDP2 sharding, sequence packing), see `yzmir-training-optimization`. This sheet stops at "what value, roughly."


## Training Skeleton (TRL SFT + DPO)

```python
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

base = "meta-llama/Meta-Llama-3.1-8B"
tok  = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16")

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM", use_rslora=True,
)

# Stage 1 — SFT
sft = SFTTrainer(
    model=model, peft_config=lora, tokenizer=tok,
    train_dataset=sft_train, eval_dataset=sft_val,
    args=SFTConfig(
        output_dir="ckpt-sft",
        learning_rate=2e-4,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps", eval_steps=200,
    ),
)
sft.train()

# Stage 2 — DPO on preference pairs (chosen / rejected)
dpo = DPOTrainer(
    model="ckpt-sft",                # SFT checkpoint as starting policy
    ref_model=None,                  # TRL infers a frozen ref from policy
    tokenizer=tok,
    train_dataset=pref_train,
    args=DPOConfig(
        output_dir="ckpt-dpo",
        learning_rate=5e-7,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        beta=0.1,                    # DPO temperature; 0.1 is the canonical default
        bf16=True,
        gradient_checkpointing=True,
    ),
)
dpo.train()
```

For SimPO, set `loss_type="simpo"` in `DPOConfig` and use a target reward margin. For ORPO, swap to `ORPOTrainer` and skip the SFT stage. Always validate on a held-out preference set during DPO; reward hacking on the eval set is real.


## Evaluation

### During training

- **Loss curves** (train + val): healthy curves are monotonically decreasing then plateau.
- **Held-out task metrics**: accuracy, F1, BLEU, ROUGE, exact-match — pick what fits the task.
- **Catastrophic-forgetting probe**: a fixed set of 50–200 general-knowledge / safety prompts evaluated every N steps. If the model loses general capability, drop the LR or stop earlier.

### After training

- **Test-set metrics** (run once at the end).
- **Pairwise human or LLM-judge evaluation** vs. the base model (or vs. SFT, when evaluating preference tuning).
- **Adversarial / safety eval** if it's a deployed model — cross-ref `ordis-security-architect` for prompt-injection and jailbreak harnesses.
- **Production A/B** before full rollout.

For preference-tuned models, **AlpacaEval 2** and **Arena-Hard** are the de-facto open benchmarks; for reasoning RL, MATH/GSM8K/code-execution rewards. Always report against the SFT baseline, not just the base model.


## Common Issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Loss → 0, val loss rising | Overfitting | Fewer epochs, more regularization, more data, LoRA instead of full FT. |
| Model "forgets" general knowledge | Catastrophic forgetting (LR too high, too many epochs) | Drop LR 10×; use LoRA; add 5-15% general-purpose mix into SFT data. |
| DPO model becomes incoherent or repeats | DPO over-optimization, β too low, bad preference data | Raise β (0.1 → 0.3), shorter training, better preference pairs, switch to IPO. |
| GRPO reward goes up but quality drops | Reward hacking | Tighten reward function, add KL penalty, evaluate on held-out distribution. |
| OOM on 7-8B model with single 24GB GPU | FP16 full FT or unbatched generation | QLoRA, gradient checkpointing, sequence packing, `bf16`, smaller batch + accumulation. |
| QLoRA significantly worse than LoRA on this task | Quantization-init mismatch | Try LoftQ initialization. |
| LoRA at high rank stops improving | Standard `α/r` scaling stunts large `r` | Switch to rsLoRA (`use_rslora=True`). |
| Need 32k+ context but base is 4k | Context extension required | LongLoRA (S²-Attn + tuned embeddings/norms), or use a base with native long context. |


## Best Practices Checklist

Before:
1. Try the strongest model with the strongest prompt. Honestly.
2. Try RAG if the gap is "doesn't know X."
3. Have ≥ 1,000 clean, manually-spot-checked examples.
4. Define the success metric and the failure-mode probe before you train.

During:
5. LoRA / QLoRA by default; full FT only with explicit justification.
6. SFT learning rate ~ pre-train LR / 100; preference-tuning LR 10–100× lower than SFT.
7. Track loss, eval metric, and a forgetting probe on the same dashboard.
8. Cosine schedule, 3–10% warmup, early stopping on val metric.

After:
9. Test-set evaluation, pairwise LLM-judge vs. baseline, forgetting probe, adversarial eval.
10. Document everything (data version, model commit, hyperparameters, eval results) — fine-tuning runs that aren't reproducible are uneconomic to maintain.


## Quick Reference

| You want to… | Use |
|--------------|-----|
| Make the model follow a format / domain | SFT (LoRA), 1k–10k examples, LR ~2e-4 |
| Align with human preferences (have pair data) | SFT + DPO, β=0.1, LR ~5e-7 |
| Same, GPU-tight | SFT + SimPO (reference-free) |
| Same, in one training run | ORPO |
| Have only thumbs-up/thumbs-down logs | KTO |
| Train a reasoning model with verifiable reward | SFT + GRPO, group size 4–16 |
| Single GPU, fast iteration | Unsloth + QLoRA |
| Production-grade reproducible pipeline | Axolotl + YAML configs |
| Frontier algorithm, latest paper | TRL |
| Web UI, no code | LLaMA-Factory |


## Summary

1. **Don't fine-tune unless prompts and RAG have honestly failed.**
2. **Three things are called fine-tuning**: SFT (teach), preference tuning (align), RL (optimize against reward). Pick the right one.
3. **DPO is the preference-tuning default**; reach for IPO (noisy data), KTO (binary labels), SimPO (no ref model), ORPO (one-shot SFT+pref) when their constraints fit.
4. **GRPO is the algorithm for verifiable-reward RL** — reasoning, code, tool use.
5. **LoRA + QLoRA is the right default**; DoRA, rsLoRA, LoftQ, and LongLoRA are targeted improvements, not the default.
6. **Data quality dominates**. So does learning rate. Most of the rest is engineering.
7. **Evaluate honestly**, including a forgetting probe and an adversarial probe. A model that wins your eval but loses general capability is not an improvement.

Cross-references: `yzmir-training-optimization` (FSDP2, FP8, optimizer, muP, gradient diagnostics), `yzmir-pytorch-engineering` (FlashAttention/FlexAttention, OOM debugging), `yzmir-ml-production` (serving the fine-tuned model), `ordis-security-architect` (adversarial / jailbreak eval), `reasoning-models.md` (this campaign — reasoning-tier model context for GRPO).

---

*Model lineup current as of 2026-05; revisit quarterly.*
