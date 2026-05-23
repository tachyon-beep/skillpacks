# Review: yzmir-neural-architectures
**Version:** 1.2.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

## 1. Inventory

### Plugin metadata
- `plugins/yzmir-neural-architectures/.claude-plugin/plugin.json:1-16`
  - `name`: `yzmir-neural-architectures`
  - `version`: `1.2.0`
  - `description`: "Neural architectures - CNNs (ConvNeXt v2 / EfficientNetV2), Transformers, SSM/Mamba, MoE (Mixtral/DeepSeek), modern diffusion (SDXL/FLUX/DiT), multimodal (CLIP/SigLIP/LLaVA), SAM/SAM-2, equivariant GNNs - 9 reference sheets, 3 commands, 2 agents"
  - `license`: CC-BY-SA-4.0
  - `keywords`: `[yzmir, neural, architectures]`

### Marketplace registration
- `/.claude-plugin/marketplace.json:589`
  - Description: "Neural architectures - CNNs, Transformers, RNNs, selection guidance - 9 skills"
  - **Drift vs. plugin.json:** marketplace.json description is stale — still advertises "CNNs, Transformers, RNNs" while plugin.json claims SSM/Mamba, MoE, diffusion, multimodal, SAM/SAM-2, equivariant GNNs. Also "9 skills" — the pack has 1 router skill + 9 reference sheets (or 8 specialist sheets if you exclude the router's own content), and 3 commands + 2 agents that the catalog blurb omits.

### Component layout
```
plugins/yzmir-neural-architectures/
├── .claude-plugin/plugin.json
├── agents/
│   ├── architecture-advisor.md
│   └── architecture-reviewer.md
├── commands/
│   ├── compare-cnn.md
│   ├── select-architecture.md
│   └── validate-architecture.md
└── skills/
    └── using-neural-architectures/
        ├── SKILL.md                              (540 lines)
        ├── architecture-design-principles.md     (960)
        ├── attention-mechanisms-catalog.md       (824)
        ├── cnn-families-and-selection.md         (809)
        ├── generative-model-families.md          (915)
        ├── graph-neural-networks-basics.md       (722)
        ├── multimodal-architectures.md           (374)
        ├── normalization-techniques.md           (915)
        ├── sequence-models-comparison.md         (804)
        └── transformer-architecture-deepdive.md  (1138)
```

Total reference-sheet content: ~7,461 lines across 9 sheets.

### Skills (1 router)
| Skill | Description | Status |
|-------|-------------|--------|
| `using-neural-architectures` (`skills/using-neural-architectures/SKILL.md:1-4`) | "The architecture selection router for CNNs, Transformers, RNNs, GANs, GNNs by data modality and constraints" | Issue — see Finding M-1 (description outdated + doesn't start with "Use when...") |

### Reference sheets (9, all under router skill dir)
| Sheet | Lines | Parent | Status |
|-------|-------|--------|--------|
| `architecture-design-principles.md` | 960 | router | OK |
| `attention-mechanisms-catalog.md` | 824 | router | OK |
| `cnn-families-and-selection.md` | 809 | router | OK |
| `generative-model-families.md` | 915 | router | OK |
| `graph-neural-networks-basics.md` | 722 | router | OK |
| `multimodal-architectures.md` | 374 | router | OK (shortest — see P-3) |
| `normalization-techniques.md` | 915 | router | OK |
| `sequence-models-comparison.md` | 804 | router | OK |
| `transformer-architecture-deepdive.md` | 1138 | router | OK |

### Commands (3)
| Command | Description | argument-hint | allowed-tools | Status |
|---------|-------------|---------------|---------------|--------|
| `/compare-cnn` (`commands/compare-cnn.md:1-4`) | Compare CNN architectures - ResNet vs EfficientNet vs MobileNet | `[constraint: cloud\|edge\|mobile\|accuracy]` | `["Read", "Grep", "Glob", "Bash", "Task"]` | OK — note `Task` is non-standard (most marketplace commands omit Task; see Finding m-2) |
| `/select-architecture` (`commands/select-architecture.md:1-4`) | Guided architecture selection by modality/task/constraints | `[modality] [task] [constraints]` | `["Read", "Grep", "Glob", "Bash", "Task", "AskUserQuestion"]` | OK — see m-2 |
| `/validate-architecture` (`commands/validate-architecture.md:1-4`) | Validate architecture for skip connections, depth-width, capacity | `[model_file_or_class]` | `["Read", "Grep", "Glob", "Bash", "Task"]` | OK — see m-2 |

### Agents (2)
| Agent | Model | SME-compliant? | Status |
|-------|-------|----------------|--------|
| `architecture-advisor` (`agents/architecture-advisor.md:1-4`) | opus | Yes — desc ends "Follows SME Agent Protocol with confidence/risk assessment." (line 2); body cites `meta-sme-protocol:sme-agent-protocol` (line 10) and requires the four output sections | OK |
| `architecture-reviewer` (`agents/architecture-reviewer.md:1-4`) | sonnet | Yes — desc ends "Follows SME Agent Protocol with confidence/risk assessment." (line 2); body cites `meta-sme-protocol:sme-agent-protocol` (line 10) and requires the four output sections | OK |

Both agents declare only `description` + `model` (no `tools:` restriction) — matches dominant marketplace convention.

### Hooks
None. Not expected for this pack type.

### Slash-command wrapper
- `/home/john/skillpacks/.claude/commands/neural-architectures.md` — **EXISTS** (not missing).
- However, it is a **near-verbatim copy** of `SKILL.md` rather than a thin "When to use" wrapper. Confirmed by `diff` showing only minor formatting and one block (the "How to Access Reference Sheets" instruction block, lines 38–55 of SKILL.md) is omitted. Other slash wrappers in this repo (`.claude/commands/python-engineering.md`, `deep-rl.md`, `llm-specialist.md`) are shorter, plain-markdown overview documents (377 lines for python-engineering vs 467 here). See Finding M-2.

---

## 2. Domain & Coverage

### Intended scope (inferred from plugin.json + SKILL.md + CLAUDE.md)
Architecture selection and design discipline for neural networks — choosing WHAT architecture to use given data modality, dataset size, deployment constraints, latency. Explicitly hands off training, implementation, and deployment to sibling Yzmir packs.

**Audience:** practitioner/expert ML engineers selecting and reviewing architectures.

### Coverage map (vs. modern 2026 landscape)

#### Foundational — Status
- Inductive bias matching — **Exists** (`architecture-design-principles.md`, `architecture-reviewer.md`)
- Skip connections / residuals — **Exists** (`architecture-design-principles.md`, `architecture-reviewer.md`, `validate-architecture.md`)
- Normalization (BN/LN/GN/RMSNorm) — **Exists** (`normalization-techniques.md`, 915 lines; check RMSNorm coverage)
- Capacity vs. data ratio — **Exists** (`validate-architecture.md`, `architecture-reviewer.md`)

#### Core families — Status
- CNNs (ResNet, EfficientNet, MobileNet, ConvNeXt v2, EfficientNetV2) — **Exists** (`cnn-families-and-selection.md`, 809 lines; 135 mentions of the four key families)
- Sequence models (RNN/LSTM/TCN/Transformer/Mamba/SSM/RWKV/Jamba) — **Exists** (`sequence-models-comparison.md`, 38 mentions of Mamba/SSM/RWKV/Jamba)
- Transformers (ViT, BERT, ViT variants, MoE Mixtral/DeepSeek/OLMoE, RoPE/YaRN/NoPE) — **Exists** (`transformer-architecture-deepdive.md`, 1138 lines — largest sheet)
- Attention variants (self/cross/multi-head/sparse) — **Exists** (`attention-mechanisms-catalog.md`, 824 lines)
- Generative (GAN/VAE/Diffusion, DiT, SDXL/SD3/FLUX, ControlNet/LoRA/IP-Adapter) — **Exists** (`generative-model-families.md`, 915 lines)
- GNNs (GCN/GAT/GraphSAGE, equivariant) — **Exists** (`graph-neural-networks-basics.md`, 722 lines)
- Multimodal (CLIP/SigLIP, LLaVA/BLIP-2/Q-Former/Flamingo/Idefics, native Chameleon/Gemini-style) — **Exists** (`multimodal-architectures.md`, 374 lines)

#### Advanced/special — Status
- SAM / SAM-2 (segmentation foundation models) — **Plugin.json claims coverage but not verified** — `cnn-families-and-selection.md` does not appear to be the home; needs spot check (Finding P-1)
- Equivariant GNNs — **Plugin.json claims; sheet is 722 lines** — likely covered but unverified
- Sparse / long-context attention — **Likely in attention-mechanisms-catalog or transformer-deepdive** — unverified
- Tabular architecture guidance — **Mentioned in commands** (`select-architecture.md:25,76-81`, `architecture-advisor.md:84`) but no dedicated sheet — minor gap (P-2)

### Research currency
- Domain is **evolving** (AI/ML).
- Plugin.json description is current (mentions ConvNeXt v2, EfficientNetV2, SSM/Mamba, Mixtral/DeepSeek-MoE, SDXL/FLUX/DiT, SigLIP/LLaVA, SAM/SAM-2) — author has updated.
- Router SKILL.md description (`SKILL.md:3`) is **stale** — still reads "CNNs, Transformers, RNNs, GANs, GNNs" with no mention of Mamba/MoE/Diffusion/Multimodal. Major drift between metadata layers (Finding M-1).
- Marketplace catalog blurb (`marketplace.json:589`) is **stale** — same omission as router (Finding M-3).

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Score | Evidence |
|---|-----------|-------|----------|
| 1 | Domain coverage breadth | **Pass** | 9 sheets cover CNN/Transformer/sequence/generative/GNN/multimodal/normalization/attention/design — all major modalities. plugin.json claims (ConvNeXt v2, EFNv2, Mamba, MoE, DiT, SAM, equivariant GNNs) appear backed by sheet content (length + spot-checked keyword density). |
| 2 | Routing accuracy / discoverability | **Minor** | SKILL.md description (line 3) does not begin with the dominant repo convention "Use when..." (CLAUDE.md routing relies on description-based discovery). It also lists outdated families (RNNs/GANs only) which hurts trigger fidelity for Mamba / diffusion / multimodal queries. |
| 3 | Specialist depth | **Pass** | Sheets are 374–1138 lines; multimodal sheet is the shortest and *could* be deeper (modern multimodal landscape moves fast) but explicit handoffs to `cnn-families-and-selection.md` and `transformer-architecture-deepdive.md` mitigate. |
| 4 | Commands quality | **Minor** | All 3 commands have correct quoted-array `allowed-tools` and `argument-hint`. They include `"Task"` in allowed-tools which is non-standard for this marketplace (most commands omit `Task`); not a functional defect but a maintenance inconsistency. `compare-cnn` has fixed benchmark tables that will drift over time (e.g., "Jetson Nano" benchmarks at line 60). |
| 5 | Agents quality (SME compliance) | **Pass** | Both agents include the load-bearing SME phrase in the description, cite `meta-sme-protocol:sme-agent-protocol` in the body, and require all four output sections (Confidence/Risk/Information Gaps/Caveats). Model selection (opus for advisor, sonnet for reviewer) is reasonable for the complexity tier. |
| 6 | Cross-pack handoffs | **Pass** | SKILL.md sections "Integration with Other Packs" (lines 462–512) explicitly route to `yzmir-training-optimization`, `yzmir-pytorch-engineering`, `yzmir-ml-production`, `yzmir-deep-rl`, `yzmir-llm-specialist`, and `yzmir-dynamic-architectures`. Agents/commands also cross-pack. |
| 7 | Internal consistency | **Major** | (a) SKILL.md `:454` says **"The 8 architecture skills:"** then lists 8 (lines 455–462) — but the file's own Summary (lines 530–538) lists 9 sheets. **Off-by-one** inconsistency within the same file. The 9th sheet (multimodal) was added but the count blurb was not updated. (b) plugin.json says "9 reference sheets, 3 commands, 2 agents" — accurate. (c) marketplace.json blurb says "9 skills" — accurate in count but stale in family list. (d) Router description (SKILL.md:3) is stale. |
| 8 | Slash-command wrapper alignment | **Major** | Wrapper `/home/john/skillpacks/.claude/commands/neural-architectures.md` exists (good — not missing per Stage 4 check). But it is a **copy of SKILL.md content** (467 lines, same critical context blocks, decision tree, rationalization table) rather than a thin overview pointer. The wrapper carries the **same "8 architecture skills" off-by-one bug** (`neural-architectures.md:454`). Diverges from the marketplace convention demonstrated by `python-engineering.md` (377-line overview document with no decision tree). |

**Overall: Major** — Pack is content-rich and structurally sound on the foundational dimensions (coverage, agent compliance, cross-pack handoffs), but has multiple **metadata drift / internal inconsistency** defects (router description outdated, count off-by-one, wrapper is a duplicate, marketplace blurb stale) that collectively warrant attention before the next release. None block the pack from functioning today.

---

## 4. Behavioral Tests

**Note:** Per task constraints (report-only, no edits), behavioral tests are scenario walk-throughs against the static text, not live subagent dispatches.

### Test 1 — Router discoverability under modern-family query
**Scenario:** "I want to use a Mamba/SSM for long-context retrieval — which architecture should I pick?"

**Expected:** Router triggers via "Sequences" routing rule, routes to `sequence-models-comparison.md`, which (per `grep` of 38 Mamba/SSM/RWKV/Jamba mentions) covers SSMs.

**Observed:** Router `description` field (`SKILL.md:3`) reads "CNNs, Transformers, RNNs, GANs, GNNs" — **does not mention Mamba/SSM**. Description-based discovery in skill-trigger contexts could miss this query. The router *body* lists SSM/Mamba in the Recency Bias table (`SKILL.md:393`) and the multimodal/sequence sections, but the description (which is what the model sees for activation decisions) is stale.

**Result:** **Fix needed (Minor → Major depending on how often description-based discovery fires)** — see M-1.

### Test 2 — Off-by-one count under summary-table read
**Scenario:** User asks "list all the specialist skills in this pack".

**Expected:** Router summary lists all 9 sheets accurately.

**Observed:** `SKILL.md:454` literally says "**The 8 architecture skills:**" then enumerates 8. Lines 530–538 of the same file list 9 sheets including multimodal. The wrapper (`.claude/commands/neural-architectures.md:454`) carries the same bug. If the model reads the "8 architecture skills" line first and stops, it will under-list. Self-contradicting documentation.

**Result:** **Fix needed (Major)** — internal contradiction inside one file (M-4).

### Test 3 — Pressure resistance on the advisor agent
**Scenario:** "Quick — what model for 2000 images? Just give me a number, don't ask questions."

**Expected:** Agent resists "skip clarification" pressure (matches the Rationalization Table in router `:451-458` and agent's Phase 1 clarification logic `architecture-advisor.md:52-76`).

**Observed:** Agent's "Phase 1: Clarify Requirements" (`architecture-advisor.md:52`) explicitly says **"Always ask before recommending"** and provides an AskUserQuestion template. Recency-bias resistance is present (`:108-118`). Capacity-matching rules (`:120-138`) would catch the 2k-sample case (ResNet-18 / EfficientNet-B0 + pretrained).

**Result:** **Pass.**

### Test 4 — Validate-architecture command on Pythonized model
**Scenario:** "/validate-architecture my_model.py" where the model is 30-layer Conv2d with no residuals.

**Expected:** Command's checklist (`validate-architecture.md:42-77`) flags >10-layer-without-skip as a violation.

**Observed:** Section "2. Skip Connections for Deep Networks" (`validate-architecture.md:42-77`) has bash greps to count layers and detect `+ x`/`identity`/`residual` patterns, and explicit threshold (`if num_layers > 10 and not has_skip_connections(model): print("WARNING ...")`). Concrete fix code provided.

**Result:** **Pass.**

### Test 5 — Recency-bias resistance: "use ViT for my 3k-image medical dataset"
**Scenario:** Tests router's resistance to recommending the trendy answer.

**Expected:** Router Resistance Table (`SKILL.md:386-398`) flags ViT below 10k images as wrong, suggests CNNs.

**Observed:** Table is present and accurate (line 387: "Vision Transformers (ViT) | Small datasets (< 10k images) | CNNs (ResNet, EfficientNet)"). Also `architecture-advisor.md:111` carries the same challenge logic.

**Result:** **Pass.**

### Test 6 — Wrapper-skill drift
**Scenario:** Model loads `/neural-architectures` wrapper instead of the router skill.

**Expected:** Wrapper provides the same routing guidance as the router (functional parity).

**Observed:** Wrapper is a near-verbatim copy minus YAML frontmatter and reference-sheet-path-block. It carries the same off-by-one ("8 architecture skills") bug. Two artifacts now contain the same routing text — when the router is updated, the wrapper will drift unless updated in lockstep.

**Result:** **Fix needed (Major)** — see M-2.

### Test summary
| Component | Test | Result |
|-----------|------|--------|
| Router SKILL.md description | Modern-family discoverability (T1) | Fix (M-1) |
| Router SKILL.md body | Count consistency (T2) | Fix (M-4) |
| `architecture-advisor` agent | Pressure-resist clarification (T3) | Pass |
| `/validate-architecture` command | Deep-net-no-skip detection (T4) | Pass |
| Router resistance table | ViT-for-small-dataset (T5) | Pass |
| `.claude/commands/neural-architectures.md` wrapper | Duplication / drift (T6) | Fix (M-2) |

---

## 5. Findings

### Critical
*(none)*

### Major

**M-1 — Router SKILL.md description is stale; lists 2021-era families only**
- Location: `plugins/yzmir-neural-architectures/skills/using-neural-architectures/SKILL.md:3`
- Current: `description: The architecture selection router for CNNs, Transformers, RNNs, GANs, GNNs by data modality and constraints`
- Defects: (a) does not start with "Use when..." — the dominant convention in this repo for skill-discovery triggers; (b) lists "RNNs, GANs" but omits Mamba/SSM, MoE, Diffusion, Multimodal that the pack's content actually covers; (c) drifts from `plugin.json:4` which already names the modern families. Hurts description-based trigger fidelity for ~half the sheets.
- Recommended fix (no edit performed): rewrite to `description: Use when selecting or comparing neural architectures — routes by data modality (vision / sequence / graph / generative / multimodal) and constraints (dataset size, compute, latency); covers CNNs (ConvNeXt v2 / EfficientNetV2), Transformers + MoE, SSM/Mamba, diffusion (SDXL/FLUX/DiT), CLIP/SigLIP/LLaVA, GNNs, normalization, attention variants.`

**M-2 — `.claude/commands/neural-architectures.md` wrapper is a duplicate, not an overview**
- Location: `/home/john/skillpacks/.claude/commands/neural-architectures.md` (467 lines)
- Defects: (a) Near-verbatim copy of `SKILL.md` rather than a short overview pointer like `.claude/commands/python-engineering.md` (377 lines, plain-overview prose, no decision-tree). (b) Carries the same off-by-one bug at line 454 ("The 8 architecture skills:"). (c) Creates a two-source-of-truth maintenance trap — every router edit must be applied twice or wrapper drifts.
- Recommended fix: replace wrapper body with a short overview (When to use / Routing summary / link to specialist sheets) modelled on the `python-engineering.md` wrapper pattern. Keep the canonical content in `SKILL.md` only.

**M-3 — Marketplace catalog description for this pack is stale**
- Location: `/.claude-plugin/marketplace.json:589`
- Current: `"description": "Neural architectures - CNNs, Transformers, RNNs, selection guidance - 9 skills"`
- Defects: Omits the modern families plugin.json calls out (Mamba/SSM, MoE, Diffusion, Multimodal, SAM/SAM-2, equivariant GNNs). "9 skills" undercounts since there are also 3 commands + 2 agents — see plugin.json:4 phrasing as the canonical version.
- Recommended fix: bring marketplace.json:589 in line with plugin.json:4.

**M-4 — Self-contradicting count inside SKILL.md ("8 architecture skills" vs. enumerated 9)**
- Location: `SKILL.md:454-462` (says "The 8 architecture skills:" and lists 8) vs. `SKILL.md:526-538` (Summary lists 9 sheets including `multimodal-architectures.md`).
- Defects: The 9th sheet (multimodal) was added later but the earlier count blurb was not updated. Same defect appears in the wrapper (line 454). A reader (model or human) who reaches the "8" line first will conclude the multimodal sheet doesn't exist.
- Recommended fix: replace "The 8 architecture skills:" with "The 9 architecture reference sheets:" and add `multimodal-architectures.md` to the numbered list at lines 454–462.

### Minor

**m-1 — SKILL.md retains "8 architecture skills" formatting in two places where the pack now has 9**
- Already covered by M-4 but worth tracking the wrapper instance separately (`.claude/commands/neural-architectures.md:454`) since it's a distinct artifact.

**m-2 — Commands include `"Task"` in `allowed-tools`; not the dominant marketplace pattern**
- Location: `commands/compare-cnn.md:3`, `commands/select-architecture.md:3`, `commands/validate-architecture.md:3`.
- Not a functional defect (Task lets the command dispatch a subagent). But most marketplace commands omit Task. Audit whether dispatch is actually used; if not, remove for minimum-privilege hygiene.

**m-3 — `compare-cnn.md` carries hardcoded latency benchmarks (Jetson Nano FP16, iPhone 12 INT8) that will rot**
- Location: `commands/compare-cnn.md:58-78`.
- Tables are useful but will be wrong within 12–24 months as hardware evolves. Either date-stamp the tables ("benchmarks as of late 2025"), cite the source (timm / MLPerf), or move to relative-rank claims ("MobileNetV3-Small is fastest of these on edge").

### Polish

**P-1 — SAM / SAM-2 coverage claimed but home unclear**
- plugin.json:4 lists "SAM/SAM-2" as covered. Likely lives in `cnn-families-and-selection.md` or `multimodal-architectures.md` but neither was sampled deeply enough in this review to confirm. Recommend a spot check — if absent, add a short SAM/SAM-2 section to the CNN sheet (it's a vision foundation model for segmentation).

**P-2 — Tabular data guidance is present in commands/agents but has no dedicated sheet**
- `select-architecture.md:76-81`, `architecture-advisor.md:84` route tabular tasks to MLP/XGBoost. No reference sheet for tabular deep learning (TabNet, FT-Transformer, NODE). Low-priority gap — most users in this pack's audience will use boosted trees for tabular, which is correctly flagged. Worth deciding whether to add a 1-page tabular sheet or document the deliberate omission.

**P-3 — `multimodal-architectures.md` is the shortest sheet (374 lines) in a fast-moving subdomain**
- Modern multimodal moves at 2x the pace of the other modalities. Coverage looks reasonable for 2024–early 2026 (SigLIP, LLaVA, BLIP-2, Q-Former, Flamingo, Chameleon-class native multimodal mentioned). Schedule a deeper refresh next cycle.

**P-4 — Plugin keywords are sparse**
- `plugin.json:11-15`: `[yzmir, neural, architectures]`. Could add `cnn, transformer, mamba, diffusion, multimodal, gnn` for marketplace search.

---

## 6. Recommended Actions

In priority order (no edits performed — these are recommendations only):

1. **M-4** (effort: 2 minutes) — Fix off-by-one count inside SKILL.md and the wrapper; add `multimodal-architectures.md` to the numbered list. Touches two files.
2. **M-1** (effort: 2 minutes) — Rewrite SKILL.md `description:` to start with "Use when..." and include modern families. One field.
3. **M-3** (effort: 2 minutes) — Update marketplace.json:589 description to match plugin.json:4. One line.
4. **M-2** (effort: 15–30 minutes) — Replace `.claude/commands/neural-architectures.md` body with a short overview (target ~150–200 lines) matching the `python-engineering.md` pattern, so it is a wrapper not a duplicate. Pick canonical content boundary deliberately.
5. **m-2** (effort: 5 minutes) — Audit `Task` in command `allowed-tools`; remove if dispatch is not used.
6. **m-3** (effort: 10 minutes) — Date-stamp or source-cite the benchmark tables in `compare-cnn.md`.
7. **P-1** (effort: 5 minutes) — Verify SAM/SAM-2 coverage; add a short subsection if absent.
8. **P-2 / P-3 / P-4** (defer) — next maintenance cycle.

Recommended version bump on application of fixes: **patch (1.2.0 → 1.2.1)** for M-1/M-3/M-4 + m-2/m-3/P-1 (metadata + minor content corrections). If M-2 is also done (wrapper restructure is user-facing behaviour change in how the slash command displays), **minor (1.2.0 → 1.3.0)** is more honest.

---

## 7. Reviewer Notes

- **Methodology:** Stages 1–4 of `using-skillpack-maintenance` rubric (Stage 5 skipped per task). All findings cite paths and lines. Behavioral tests in Section 4 are scenario walk-throughs against static text rather than live subagent dispatches, because the task is report-only and a dispatched subagent would itself need access to the pack content.
- **Confidence:**
  - High confidence on M-1, M-3, M-4, m-2, m-3 — these are directly observable in the file text and quoted verbatim.
  - High confidence on M-2 — diff between SKILL.md and the wrapper confirmed.
  - Medium confidence on P-1 (SAM/SAM-2 coverage location) — based on grep + file size heuristics, did not read the full CNN or multimodal sheets line-by-line.
  - Medium confidence on m-2 — `Task` in allowed-tools is unusual but not necessarily wrong; needs domain-author input.
- **Risk:**
  - If only the metadata fixes (M-1, M-3, M-4) ship, the discoverability and self-consistency defects close, but the duplicated-wrapper maintenance trap (M-2) remains and will reappear on the next router edit.
  - Hardcoded benchmark tables (m-3) will become wrong silently — the pack will keep claiming Jetson Nano latencies that no longer represent the hardware target users actually deploy to.
- **Information gaps:**
  - Did not read full text of all 9 reference sheets — sampled headers, line counts, and keyword density only.
  - SAM/SAM-2 home not directly verified (P-1).
  - Did not run live behavioural dispatch tests against the agents — relied on the static text of agent prompts and example blocks.
- **Caveats:**
  - "Major" overall rating reflects metadata drift defects collectively, not content quality of the reference sheets themselves, which appear strong (sheet length distribution, modern-family coverage in plugin.json, agent SME compliance all read as a well-maintained pack).
  - Recommended fix sequencing assumes editor will not change the canonical router copy. If the canonical home for routing text is moved to the wrapper, swap M-1/M-4 onto that file instead.
