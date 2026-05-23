# Review: yzmir-pytorch-engineering
**Version:** 1.2.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

---

## 1. Inventory

### Plugin metadata
- `plugins/yzmir-pytorch-engineering/.claude-plugin/plugin.json:1-15`
- Name: `yzmir-pytorch-engineering`, version `1.2.0`
- Self-description: "PyTorch 2.9+ mastery - torch.compile (modes/dynamic/fullgraph), torch.amp (BF16/FP16/FP8), FSDP1/FSDP2 (fully_shard, MixedPrecisionPolicy), DTensor + device_mesh, FlexAttention, scaled_dot_product_attention, CUDA Graphs, NVTX/Nsight, torch.profiler, channels_last, expandable_segments. 8 reference sheets + 1 router, 3 commands, 2 agents."
- License/repo fields present and correct.

### Skills (1 router + 8 reference sheets)

| Skill / Sheet | Path | LOC | Status |
|---------------|------|-----|--------|
| `using-pytorch-engineering` (router SKILL.md) | `skills/using-pytorch-engineering/SKILL.md` | 442 | OK |
| `tensor-operations-and-memory` | same dir | 1110 | OK |
| `module-design-patterns` | same dir | 2012 | OK |
| `distributed-training-strategies` | same dir | 1949 | OK |
| `mixed-precision-and-optimization` | same dir | 1553 | OK |
| `performance-profiling` | same dir | 2237 | OK |
| `debugging-techniques` | same dir | 2023 | OK |
| `checkpointing-and-reproducibility` | same dir | 2039 | OK |
| `custom-autograd-functions` | same dir | 2828 | OK |

Total content: ~16k LOC.

### Commands (3)

| Command | Path | argument-hint | allowed-tools (quoted array?) | Notes |
|---------|------|---------------|-------------------------------|-------|
| `debug-nan` | `commands/debug-nan.md:1-5` | `"[file_or_layer_name]"` | yes | Modern `torch.amp` API used, anomaly detection workflow, 2.9 awareness included. |
| `debug-oom` | `commands/debug-oom.md:1-5` | `"[file_or_description]"` | yes | 6-step methodology; uses modern `torch.amp.GradScaler('cuda')`. |
| `profile` | `commands/profile.md:1-5` | `"[script_path] [--cpu\|--gpu\|--memory\|--io]"` | yes (incl. `Write`) | 4-phase profiling; references `torch.profiler`, memory snapshot, Inductor. |

All three include the `Task` tool — appropriate for subagent dispatch from a command.

### Agents (2)

| Agent | Path | Model | `tools:`? | SME-protocol compliant? |
|-------|------|-------|-----------|--------------------------|
| `pytorch-code-reviewer` | `agents/pytorch-code-reviewer.md:1-5` | `sonnet` | omitted (correct default) | Yes — `agents/pytorch-code-reviewer.md:10` cites `meta-sme-protocol:sme-agent-protocol` and requires Confidence / Risk / Information Gaps / Caveats; description ends with "Follows SME Agent Protocol with confidence/risk assessment." (line 2). |
| `memory-diagnostician` | `agents/memory-diagnostician.md:1-5` | `sonnet` | omitted (correct default) | Yes — `agents/memory-diagnostician.md:10` cites SME protocol with four-section requirement. |

### Hooks
None present. Not expected for this pack.

### Slash-command wrapper
- `/.claude/commands/pytorch-engineering.md` — exists, 363 lines (see Findings §5 — the wrapper is stale).

### Marketplace registration
- Registered in `/.claude-plugin/marketplace.json`. Entry description: `"PyTorch mastery - tensors, modules, distributed training, profiling - 9 skills"` — see Findings §5 (counter-claim issue).

---

## 2. Domain & Coverage

### Domain stability
PyTorch is an **evolving** framework domain — APIs migrate (e.g. `torch.cuda.amp` → `torch.amp` at 2.4; FSDP1 → FSDP2 / `fully_shard`; FairScale ZeRO deprecated; FlexAttention added in 2.5+). Currency matters more than for stable algorithmic domains.

The pack explicitly calibrates itself to "PyTorch 2.9+ as of 2026-05" (`SKILL.md:14`) with a dedicated "About This Pack's API Currency" reconciliation gate (`SKILL.md:18-28`).

### Coverage map vs. inventory

**Foundational**
- Tensor ops / memory mgmt — covered (`tensor-operations-and-memory.md`).
- `nn.Module` design — covered (`module-design-patterns.md`).
- Checkpointing + determinism — covered (`checkpointing-and-reproducibility.md`), including DCP / FSDP2 sharded state dict at `checkpointing-and-reproducibility.md:1951-1964`.

**Core techniques**
- Mixed precision (`torch.amp` BF16/FP16/FP8) — covered with explicit deprecation notes at `mixed-precision-and-optimization.md:47-57`.
- `torch.compile` (modes, `dynamic=`, `fullgraph`, recompiles, graph breaks) — covered; debug pipeline at `debugging-techniques.md:1773-1821`.
- Distributed: DDP + FSDP1 + FSDP2 (`fully_shard`, `MixedPrecisionPolicy`, `OffloadPolicy`) + DTensor + device mesh — covered (`distributed-training-strategies.md:963-1184`).
- Profiling: `torch.profiler`, NVTX, Nsight Systems, CUDA Graphs, allocator stats, `_record_memory_history` — covered (`performance-profiling.md:813-2232`).
- Memory format `channels_last`, `expandable_segments` — covered (`tensor-operations-and-memory.md:1018-1033` and 813-822).

**Advanced**
- Custom autograd / `torch.autograd.Function` — covered in 2828-line dedicated sheet.
- FlexAttention / SDPA backend selection — covered in mixed-precision sheet.
- Sharded checkpointing (DCP) — covered.

**Cross-cutting**
- Diagnosis-first principle, routing table, common rationalizations — present in router (`SKILL.md:295-380`).
- Cross-pack handoffs to `yzmir-training-optimization`, `yzmir-llm-specialist`, `yzmir-ml-production` — explicit (`SKILL.md:430-442`).

### Coverage gaps observed
None material. Pack is comprehensive for the stated PyTorch-implementation scope. Quantization (e.g. `torch.ao.quantization`, AWQ/GPTQ workflows) is out of scope by design — handled in `yzmir-ml-production` / `yzmir-llm-specialist` per `SKILL.md:386-393`.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Evidence |
|---|-----------|--------|----------|
| 1 | Router quality (description, triggers, routing table accuracy) | **Pass** | `SKILL.md:2-3` description triggers on PyTorch symptoms; routing-by-symptom and routing-mistakes tables (`SKILL.md:295-310`) cover modern (FSDP2, FlexAttention, torch.compile) and deprecated (FairScale, `torch.cuda.amp`) cases explicitly. |
| 2 | Skill / reference-sheet coverage of the domain | **Pass** | 8 sheets covering foundational → advanced; no observable gap for PyTorch-implementation scope. |
| 3 | Skill quality (actionable, depth, examples) | **Pass** | Sheets are 1.1k–2.8k LOC each with code examples, anti-patterns, and migration notes (e.g. `mixed-precision-and-optimization.md:299-356` BF16 vs FP16 selection). |
| 4 | Commands (entry points, tool discipline, argument-hint) | **Pass** | All three have quoted `allowed-tools`, sensible argument-hint, modern API throughout. |
| 5 | Agents (scope, SME-protocol compliance, model selection) | **Pass** | Both agents cite SME protocol, declare four output sections, omit `tools:` (inherit context — correct repo norm), model `sonnet` is appropriate for review/diagnosis. |
| 6 | API currency vs. PyTorch 2.9 baseline | **Pass** | Explicit reconciliation gate (`SKILL.md:18-28`); deprecated aliases flagged everywhere they appear (`mixed-precision-and-optimization.md:47-57`); FSDP1+FSDP2 distinction maintained; FlexAttention / SDPA present. One legacy reference: `pytorch-code-reviewer.md:115-117` calls FSDP2 "(experimental)" — this is no longer accurate (FSDP2 is stable in 2.9). Minor. |
| 7 | Cross-pack boundaries and handoffs | **Pass** | `SKILL.md:430-442` explicit. Each sheet I sampled (e.g. `mixed-precision-and-optimization.md:10` boundary statement) cross-refs the strategic pack. |
| 8 | Slash-command wrapper alignment with router skill | **Major issue** | `/.claude/commands/pytorch-engineering.md` is the **old (pre-1.2.0) routing copy**. It still echoes `torch.cuda.amp` as a routing keyword (`pytorch-engineering.md:115`), lacks FSDP2 / `fully_shard` / DTensor / FlexAttention / `torch.compile` / NVTX / Nsight / CUDA Graphs / `channels_last` / `expandable_segments` triggers, omits the "Common Routing Mistakes" rows for FairScale / SDPA / compile-slower-than-eager that the SKILL.md added, and ends with a "Phase 1 - Standalone… Future cross-references" block (`pytorch-engineering.md:355-362`) that is contradicted by the SKILL.md's actual cross-pack section (`SKILL.md:430-442`). Users invoking `/pytorch-engineering` get stale, deprecated-API routing instructions instead of the v1.2.0 router. |

**Overall:** **Pass with one Major (and a few Minors).** Pack core (skills, commands, agents, marketplace registration) is structurally sound and API-current. The slash-command wrapper has drifted and is the only finding likely to affect users in the wild.

---

## 4. Behavioral Tests

Read-only review — no live subagent dispatch performed. Behavioral assessment is based on inspection of the router and reference sheets against the gauntlet categories.

### A. Pressure resistance (router)

**Scenario:** "My training is slow, just enable AMP and `torch.compile`."

- Router: `SKILL.md:142-156` ("MUST profile before optimizing") explicitly resists the pressure and routes to `performance-profiling` first. Routing-mistakes table reinforces (`SKILL.md:299`: "Training slow, optimize my optimizer → performance-profiling.md FIRST").
- Common-rationalizations table (`SKILL.md:333-346`) lists "User suggested Z", "Simple issue", "Direct answer is helpful" with counter-prescriptions.
- **Assessment:** Strong. The router would route, not act.

### B. Edge cases (deprecated API)

**Scenario:** "Should I use `torch.cuda.amp.GradScaler` or `torch.amp.GradScaler`?"

- Routing-mistakes table row 7 (`SKILL.md:305`) and Red-Flags row (`SKILL.md:322`) both intercept "type `torch.cuda.amp.autocast` / `GradScaler`" and route to the modern API.
- `mixed-precision-and-optimization.md:47-57` reproduces the actual deprecation warning string PyTorch emits.
- **Assessment:** Strong. Will not echo deprecated API.

### C. Real-world complexity (compile + FSDP2)

**Scenario:** "Set up FSDP2 with mixed precision and CPU offload, composed with torch.compile."

- Cross-cutting section `SKILL.md:252-256` sequences this correctly: distributed → memory.
- `distributed-training-strategies.md:1152-1184` covers `fully_shard` + `MixedPrecisionPolicy` + `OffloadPolicy` + composition with `torch.compile`.
- **Assessment:** Strong. Both router and sheet handle it.

### D. Wrapper drift (the only behavioral concern)

**Scenario:** A user types `/pytorch-engineering` because of the project CLAUDE.md instructions ("All router skills are available as slash commands due to skill context limits").

- The wrapper they get (`/.claude/commands/pytorch-engineering.md`) is the v0.x routing content. It mentions `torch.cuda.amp` as a routing trigger and never mentions FSDP2, FlexAttention, `torch.compile`, CUDA Graphs, NVTX, `expandable_segments`, `channels_last`, or DCP. It also tells the model "Phase 1 - Standalone: PyTorch skills are self-contained" / "Future cross-references" (`pytorch-engineering.md:355-362`) — contradicting the SKILL.md's documented cross-pack handoffs.
- **Assessment:** Failure mode is silent — the wrapper "works" (reads as a coherent router), it just routes per a deprecated worldview.

---

## 5. Findings

### Critical
None.

### Major

**M1. Slash-command wrapper is out of sync with SKILL.md (v1.2.0).**
- Path: `/.claude/commands/pytorch-engineering.md`
- Evidence:
  - Line 115: echoes `"torch.cuda.amp"` as a routing keyword (deprecated alias since 2.4).
  - Lines 64-82: distributed-training keywords list omits FSDP / FSDP1 / FSDP2 / `fully_shard` / DTensor / device mesh; only DDP appears.
  - Lines 110-128: "Mixed Precision and Optimization" section omits `torch.compile`, FlexAttention, `scaled_dot_product_attention`, FP8.
  - Lines 86-108: "Performance and Speed" omits CUDA Graphs, NVTX, Nsight, host-bound vs comm-bound triage.
  - Lines 24-42: "Memory Issues" omits `expandable_segments`, `channels_last`, fragmentation.
  - Lines 237-247: "Common Routing Mistakes" table is the 5-row pre-1.2 version (missing the FairScale, `torch.cuda.amp`, hand-rolled attention, compile-slower-than-eager, CUDA-Graphs-blindly rows that the SKILL.md added).
  - Lines 355-362: "Integration Notes" claims "Phase 1 - Standalone: PyTorch skills are self-contained… Future cross-references: training-optimization (… )" — directly contradicting SKILL.md's "Cross-Pack References" section (`SKILL.md:430-442`).
- Impact: Users invoking `/pytorch-engineering` get the pre-2.9 routing worldview. Behavior under deprecated-API queries will be wrong despite the SKILL.md being correct, because the wrapper is the content loaded for slash-command invocation.
- Recommended fix: Replace the wrapper body with content derived from the current `SKILL.md`. Match the SKILL.md's "About This Pack's API Currency", routing table, common-routing-mistakes table, and cross-pack references. The wrapper does not need to duplicate every sheet hyperlink, but it must reflect the same routing worldview.

### Minor

**m1. Marketplace description says "9 skills".**
- Path: `/.claude-plugin/marketplace.json`
- Evidence: entry `"description": "PyTorch mastery - tensors, modules, distributed training, profiling - 9 skills"`.
- Reality: 1 router SKILL.md + 8 reference sheets = 1 skill (with reference-sheet companions). The plugin.json correctly says "8 reference sheets + 1 router". Marketplace catalog hasn't been resynced with the pack's self-description.
- Fix: Update the marketplace entry's `description` to align with plugin.json (e.g. "PyTorch 2.9+ mastery — torch.compile, torch.amp (BF16/FP16/FP8), FSDP1/FSDP2, FlexAttention, profiling — 8 reference sheets + 1 router, 3 commands, 2 agents").

**m2. `pytorch-code-reviewer` agent calls FSDP2 "experimental".**
- Path: `agents/pytorch-code-reviewer.md:115-117` (under "Enhanced Tensor Subclassing" / earlier in Category-5 section): the body mentions FSDP2 only in passing, but `memory-diagnostician.md:135` says "FSDP2 (experimental): Next-gen sharding for very large models".
- Reality: FSDP2 (`fully_shard`) is stable in PyTorch 2.9 and is the recommended new-code path per the router and the distributed sheet.
- Fix: Drop the "(experimental)" qualifier in `memory-diagnostician.md:135`.

**m3. Memory-diagnostician's PyTorch-2.9 awareness block is light on FSDP2 / DCP / `expandable_segments`.**
- Path: `agents/memory-diagnostician.md:130-143`
- The agent mentions FSDP2 in one bullet but does not mention `expandable_segments:True`, `_record_memory_history` workflow, or DCP — all of which are present in the pack's reference sheets and central to its current memory story.
- Fix: Extend the 2.9 awareness block to surface `expandable_segments`, `torch.cuda.memory._record_memory_history()`, `channels_last`, and DCP for FSDP checkpoints, so the agent's cross-references stay aligned with the sheets.

**m4. Code-reviewer's PyTorch-2.9 section is shallow vs. the pack's actual depth.**
- Path: `agents/pytorch-code-reviewer.md:90-125`
- "Enhanced Tensor Subclassing" subsection is a stub ("Check for compatibility if using custom tensor types") and "Better Error Messages" subsection is one comment line. These contribute little to a review.
- Fix: Either expand to actionable checks (e.g. "look for `__torch_function__` / `__torch_dispatch__` subclasses and verify compose with `torch.compile`"; "if user pasted a shape-mismatch traceback, walk it through the 2.9 enhanced error format") or trim to one sentence pointing the agent at the relevant sheet.

### Polish

**p1. Router's "When NOT to Use PyTorch Skills" list could explicitly name `yzmir-neural-architectures`.**
- Path: `SKILL.md:386-393`. It currently says "Model architecture selection (use `yzmir-neural-architectures`)" — good, but the cross-pack references section (`SKILL.md:430-442`) only enumerates training-optimization, llm-specialist, ml-production. Adding `yzmir-neural-architectures` to the cross-pack references would close the loop.

**p2. Reference-sheet path explainer block (`SKILL.md:45-58`) is verbose.**
- Useful for the model, but the same content appears in many packs. Consider keeping it but moving the WRONG-path negative example out of bold/red callout once you trust the convention.

**p3. `profile` command's argument-hint mentions four mutually exclusive flags but the body doesn't dispatch on them.**
- Path: `commands/profile.md:4` argument-hint = `"[script_path] [--cpu|--gpu|--memory|--io]"`. The body runs all four phases regardless. Either implement flag-conditional logic or simplify the hint to `"[script_path]"`.

---

## 6. Recommended Actions

In priority order:

1. **(Major) Rewrite `/.claude/commands/pytorch-engineering.md` wrapper** to match the v1.2.0 `SKILL.md`. Drop the deprecated `torch.cuda.amp` routing keyword, add FSDP2 / `fully_shard` / DTensor / FlexAttention / `torch.compile` / CUDA Graphs / NVTX / Nsight / `expandable_segments` / `channels_last` triggers, replace the 5-row "Common Routing Mistakes" table with the 10-row v1.2.0 version, and replace the "Phase 1 - Standalone" integration block with the current cross-pack references. This is the highest-impact change because slash-command invocation is the canonical user entry point per the repo's CLAUDE.md.

2. **(Minor)** Resync `.claude-plugin/marketplace.json` description for `yzmir-pytorch-engineering` with `plugin.json` self-description (8 reference sheets + 1 router, 3 commands, 2 agents).

3. **(Minor)** Drop "(experimental)" from `agents/memory-diagnostician.md:135` and expand the 2.9 awareness block to mention `expandable_segments`, `_record_memory_history`, `channels_last`, and DCP.

4. **(Minor)** Tighten `agents/pytorch-code-reviewer.md` Category-5 ("PyTorch 2.9 Considerations") — collapse the placeholder subsections or expand to actionable checks.

5. **(Polish)** Decide whether `profile` command's argument-hint is aspirational or implemented; align body and hint either way.

6. **(Polish)** Add `yzmir-neural-architectures` to the cross-pack references list in the router for symmetry with the "When NOT to Use" guidance.

Recommended version bump after applying M1 + m1–m4: **patch → 1.2.1** (no philosophy or component changes, no new skills, all fixes are content correctness / synchronisation).

---

## 7. Reviewer Notes

- This was a Stage 1–4 read-only review per the rubric in `plugins/meta-skillpack-maintenance/skills/using-skillpack-maintenance/`. No edits applied.
- The pack itself (SKILL.md + 8 sheets + 3 commands + 2 agents) is in good shape and clearly received a thorough refresh for PyTorch 2.9: explicit reconciliation gate, deprecation-warning text reproduced verbatim where users will actually see it, FSDP1↔FSDP2 distinction maintained across router and sheets, FlexAttention / SDPA / `torch.compile` first-class, profiling sheet covers the full modern toolchain (Profiler + NVTX + Nsight + CUDA Graphs + allocator stats + `_record_memory_history`).
- The single Major finding is structural drift between the pack and its repo-root slash-command wrapper. This is an easy fix and a representative failure mode for any pack with a router-skill / slash-command-wrapper pair; a future maintenance pass might add a marketplace-level audit step that diffs every `skills/using-*/SKILL.md` against its `.claude/commands/<X>.md`.
- I did not dispatch live subagents to behavior-test the router under pressure; the assessment in §4 is from reading the routing tables and cross-checking against the sheets. If desired, a follow-up could run subagent-based pressure tests on (a) `/pytorch-engineering` with a "just enable AMP" prompt and (b) a deprecated-API query, to confirm the wrapper-drift behavior empirically.
- No project-specific data leaked into this review (per memory note `feedback_no_project_leaks.md`). All examples cited are from the public skillpack.
