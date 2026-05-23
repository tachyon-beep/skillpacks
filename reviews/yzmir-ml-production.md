# Review: yzmir-ml-production
**Version:** 1.2.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

Pack manifest: `/home/john/skillpacks/plugins/yzmir-ml-production/.claude-plugin/plugin.json`
Marketplace catalog: `/home/john/skillpacks/.claude-plugin/marketplace.json:yzmir-ml-production`

---

## 1. Inventory

### 1.1 Plugin metadata

- `plugin.json` (`.claude-plugin/plugin.json:1-20`)
  - `name`: `yzmir-ml-production`
  - `version`: `1.2.0`
  - `description`: Production ML — quantization (torch.ao.quantization, AWQ/GPTQ/AQLM/FP8), LLM serving stacks (vLLM/SGLang/TensorRT-LLM/TGI/Triton), MLOps (Prefect/Dagster/Flyte/Airflow), observability (Phoenix/Langfuse/OTel GenAI), deployment, monitoring, debugging. "10 reference sheets, 3 commands, 2 agents."
  - `license`: CC-BY-SA-4.0
  - Keywords: yzmir, ml-production, mlops, serving, quantization, monitoring, deployment

- Marketplace entry (`.claude-plugin/marketplace.json` around `yzmir-ml-production`)
  - `source: ./plugins/yzmir-ml-production`
  - `description: "Production ML - quantization, serving, MLOps, monitoring, debugging - 11 skills"` — **mismatch:** plugin.json says "10 reference sheets" (1 router + 10 sheets = 11 SKILL.md-equivalents only if you count the router). Counting `find … -name "SKILL.md"` returns 1 file. The marketplace string "11 skills" is misleading.

### 1.2 Skill files

| File | Lines | Sections | Path |
|------|------:|---------:|------|
| Router SKILL.md | 476 | 33 | `skills/using-ml-production/SKILL.md` |
| quantization-for-inference | 936 | 54 | `skills/using-ml-production/quantization-for-inference.md` |
| model-compression-techniques | 1230 | 25 | `skills/using-ml-production/model-compression-techniques.md` |
| hardware-optimization-strategies | 1436 | 66 | `skills/using-ml-production/hardware-optimization-strategies.md` |
| model-serving-patterns | 1178 | 34 | `skills/using-ml-production/model-serving-patterns.md` |
| deployment-strategies | 3561 | 37 | `skills/using-ml-production/deployment-strategies.md` |
| scaling-and-load-balancing | 2872 | 24 | `skills/using-ml-production/scaling-and-load-balancing.md` |
| experiment-tracking-and-versioning | 934 | 41 | `skills/using-ml-production/experiment-tracking-and-versioning.md` |
| mlops-pipeline-automation | 801 | 40 | `skills/using-ml-production/mlops-pipeline-automation.md` |
| production-monitoring-and-alerting | 918 | 36 | `skills/using-ml-production/production-monitoring-and-alerting.md` |
| production-debugging-techniques | 3550 | 49 | `skills/using-ml-production/production-debugging-techniques.md` |

Total content lines: 17,892 (1 router + 10 reference sheets). Median sheet: ~1200 lines. Largest sheets are `deployment-strategies` (3561 — A/B + canary + shadow + blue-green + rollback at depth) and `production-debugging-techniques` (3550 — multi-part RED/GREEN/REFACTOR + post-mortem templates + LLM-specific debugging in Part 10).

**Section-header structure (selected sheets):**

- `model-serving-patterns.md` — 9 named parts: FastAPI / TorchServe (Legacy-Maintenance flag) / gRPC / ONNX Runtime / Request Batching / Containerization / General-Purpose Framework Selection (non-LLM) / **LLM Serving Stacks** / Cloud Managed Inference.
- `production-monitoring-and-alerting.md` — 9 sections: RED metrics / Model quality / Drift detection / Dashboards / Alert rules / SLAs and SLOs / Monitoring stack (Prometheus + Grafana + OTel) / **LLM Observability** / End-to-end example.
- `scaling-and-load-balancing.md` — RED / GREEN / REFACTOR + **Part 4 LLM-Specific Scaling**.
- `mlops-pipeline-automation.md` — Architecture / RED / GREEN / **MLOps for LLM Applications** / REFACTOR pressure tests.
- `production-debugging-techniques.md` — RED-phase anti-patterns / GREEN-phase methodology / Impact / Timeline / Root Cause / Contributing Factors / What Went Well / What Went Wrong / Action Items / **Part 10 LLM-Specific Debugging** / REFACTOR pressure tests.
- `experiment-tracking-and-versioning.md` — RED / GREEN tooling landscape / **Part 3 Experiment Tracking for LLM Work** / Reproducibility checklist / REFACTOR / Integration patterns.
- `quantization-for-inference.md` — explicit `torch.ao.quantization` namespace migration note (lines 44-51, calls out PT2E `prepare_pt2e` / `convert_pt2e` path); explicit ops-vs-format-choice cross-reference to llm-specialist (line 29).

Cross-sheet LLM coverage (keyword presence):
- vLLM / SGLang / TensorRT-LLM / Phoenix / Langfuse / OpenTelemetry / OTel — present in 9 sheets (all except `model-compression-techniques.md` and `experiment-tracking-and-versioning.md`).
- AWQ / GPTQ / AQLM / FP8 / PagedAttention / continuous batching / speculative decoding — present in 7 sheets.

The LLM-era content is woven through the pack, not isolated in a single sheet.

### 1.3 Commands

| Command | Frontmatter shape | Path |
|---------|-------------------|------|
| `/optimize-inference` | `description`, `allowed-tools: ["Read","Bash","Glob","Grep","Skill"]`, `argument-hint: "[model_path]"` | `commands/optimize-inference.md:1-5` |
| `/diagnose-inference` | `description`, `allowed-tools: ["Read","Bash","Glob","Grep","Skill"]`, `argument-hint: "[symptom_or_model_name]"` | `commands/diagnose-inference.md:1-5` |
| `/deploy-model` | `description`, `allowed-tools: ["Read","Bash","Glob","Grep","Write","AskUserQuestion"]`, `argument-hint: "[model_path_or_name]"` | `commands/deploy-model.md:1-5` |

Note: `/deploy-model` omits `"Skill"` from `allowed-tools`. The body at line 300-308 instructs the model to *Load skill: yzmir-ml-production:using-ml-production*, but without `Skill` in the tool list the command cannot actually dispatch via the Skill tool. (Minor — the skill can still be loaded by description-based discovery in the parent context.)

### 1.4 Agents

| Agent | description tail | model | Path |
|-------|------------------|-------|------|
| `mlops-architect` | "...Follows SME Agent Protocol with confidence/risk assessment." | sonnet | `agents/mlops-architect.md:1-3` |
| `inference-debugger` | "...Follows SME Agent Protocol with confidence/risk assessment." | sonnet | `agents/inference-debugger.md:1-3` |

Both agents:
- Declare the SME-tail in `description`.
- Have a `**Protocol**:` body line citing `meta-sme-protocol:sme-agent-protocol` and naming the four mandatory output sections (Confidence, Risk, Information Gaps, Caveats).
- Provide positive AND negative `<example>` activation cases (mlops-architect lines 18-41; inference-debugger lines 18-41).
- Provide explicit `## Scope Boundaries` ("I design / I do NOT" or "I debug / I do NOT").
- Omit `tools:` — inherit parent context per repo convention.

`mlops-architect` body structure: Core Principle → When to Activate (4 positive + 2 negative examples redirecting `/diagnose-inference` and `/deploy-model`) → 5-step Design Protocol (Maturity assessment → Experiment Tracking → Model Registry → CI/CD pipeline → Automated Retraining) → Output Format template → MLOps Patterns (Feature Store, Model Versioning, Data Validation with great_expectations) → Scope Boundaries. Strong cross-pack handoff into `/diagnose-inference`, `/deploy-model`, `/optimize-inference`, `neural-architectures`.

`inference-debugger` body structure: Core Principle ("Production ML failures are rarely model bugs") → When to Activate (3 positive + 2 negative) → 4-step Debugging Protocol (Categorize → Gather Evidence → Reproduce in Isolation → Isolate Component) → Issue-Specific Debugging (Slow / Accuracy / OOM / Intermittent) → Output Format with Investigation Timeline + Evidence + Root Cause + Verification + Prevention → Quick Diagnostic Commands → Scope Boundaries.

Both agents partition cleanly: `mlops-architect` explicitly redirects to `inference-debugger` and vice versa. No overlap.

### 1.5 Hooks

None present. Acceptable — production ML guidance is advisory; nothing to enforce via PreToolUse / PostToolUse.

### 1.6 Slash-command wrapper

- `/home/john/skillpacks/.claude/commands/ml-production.md` **exists** (336 lines).
- **However:** the wrapper is materially out of sync with the router SKILL.md.

Drift table (wrapper vs. router SKILL.md):

| Topic | SKILL.md (current) | Wrapper file | Severity |
|-------|--------------------|--------------|----------|
| LLM-serving stacks (vLLM, SGLang, TensorRT-LLM, TGI, Triton) | Listed as Category-2 symptom keywords, separate LLM-serving Part of serving-patterns | Not mentioned anywhere | Major |
| LLM observability stacks (Phoenix, Langfuse, OTel GenAI) | Category-4 owns "plumbing"; explicit split with llm-specialist for methodology | Not mentioned | Major |
| AWQ / GPTQ / `torch.ao.quantization` (quantization ops vs. format-choice split) | Explicit split with llm-specialist | Not mentioned | Major |
| TorchServe deprecation flag | Routing-mistakes table flags TorchServe as maintenance-mode | "We use TorchServe — still route to serving-patterns. Tool choice doesn't change routing" — misses the flag | Major |
| Performance triage trichotomy (speed / accuracy / LLM output quality) | Three-way clarification + hallucination/cost examples | Two-way (speed / accuracy) — pre-LLM era | Major |
| `ordis-security-architect` cross-reference (prompt-injection, PII, threat modeling) | Section "With ordis-security-architect" + multi-concern row | Not mentioned | Minor |
| `reasoning-models`, `agentic-patterns-and-mcp`, `context-engineering-and-prompt-caching`, MCP | Cited via llm-specialist split | Not mentioned | Minor |

The wrapper duplicates content from an earlier router (pre-LLM-refresh) and now contradicts the SKILL.md it should mirror. A user invoking `/ml-production` gets a stale routing rubric.

### 1.7 References

Router SKILL.md cites `docs/plans/2025-10-30-ml-production-pack-design.md` (SKILL.md:470) and a sister-pack `yzmir/ai-engineering-expert/using-ai-engineering` (SKILL.md:471). The latter path uses the legacy `yzmir/` prefix; verify it resolves under the current `plugins/yzmir-ai-engineering-expert/` layout. (Minor — likely a stale path string.)

---

## 2. Domain & Coverage

### 2.1 User-defined scope (inferred from SKILL.md)

- **Intent:** Production-side operations for ML — moving classical and LLM models to production, optimising inference, automating MLOps, monitoring/debugging at scale.
- **Boundaries:** Explicit exclusions
  - Training optimization → `training-optimization` (SKILL.md:27)
  - Model architecture → `neural-architectures` (SKILL.md:27)
  - PyTorch infrastructure → `pytorch-engineering` (SKILL.md:27)
  - Prompt/RAG/agent design quality → `llm-specialist` (SKILL.md:27)
  - AI/LLM threat modeling → `ordis-security-architect` (SKILL.md:291-292, 431-432)
- **Audience:** Practitioners with production responsibility (not first-time ML learners — assumes profiling, Kubernetes, GPU familiarity).

### 2.2 Coverage map (production-ML domain, 2026-05)

**Foundational (must-have):**
- Quantization (PTQ / QAT / dynamic / static / PT2E) — **Exists** (`quantization-for-inference.md`)
- Pruning / distillation — **Exists** (`model-compression-techniques.md`)
- Hardware-specific optimization (GPU/CPU/edge, TensorRT, ONNX RT) — **Exists** (`hardware-optimization-strategies.md`)
- General serving (FastAPI, gRPC, ONNX, batching, containers) — **Exists** (`model-serving-patterns.md` Parts 1-6)
- Deployment rollout (canary, shadow, A/B, blue-green) — **Exists** (`deployment-strategies.md`, 3561 lines)
- Scaling (autoscaling, HPA, replicas, traffic patterns) — **Exists** (`scaling-and-load-balancing.md`, 2872 lines)
- Experiment tracking & registries (MLflow, W&B, model registry) — **Exists** (`experiment-tracking-and-versioning.md`)
- MLOps automation (CI/CD, feature store, retraining) — **Exists** (`mlops-pipeline-automation.md`)
- Monitoring/alerting (RED, drift, dashboards, SLOs) — **Exists** (`production-monitoring-and-alerting.md`)
- Debugging / post-mortems — **Exists** (`production-debugging-techniques.md`, 3550 lines)

**LLM-era extensions (2024-2026):**
- LLM serving stacks: vLLM, SGLang, TensorRT-LLM, TGI, Triton, continuous batching, PagedAttention, speculative decoding, KV-cache management — **Exists**: `model-serving-patterns.md` Part 8 "LLM Serving Stacks"; cross-referenced from router.
- LLM-targeted quantization: AWQ, GPTQ, AQLM, FP8, INT4 weight-only — **Exists**: covered in `quantization-for-inference.md` (line 29 ops/format split with llm-specialist).
- LLM observability: Phoenix, Langfuse, OpenTelemetry GenAI semantic conventions — **Exists**: `production-monitoring-and-alerting.md` Section 8 "LLM Observability".
- Prompt / RAG / eval-set versioning — **Exists**: `experiment-tracking-and-versioning.md` Part 3 "Experiment Tracking for LLM Work".
- Tool-call telemetry, prompt-injection signals — **Exists** in monitoring + debugging (Part 10 LLM-Specific Debugging).
- LLM-specific scaling (KV-cache pressure, batch-size autoscaling) — **Exists**: `scaling-and-load-balancing.md` Part 4 "LLM-Specific Scaling".
- LLM-specific MLOps (RAG-index rebuilds, fine-tune pipelines, eval gates) — **Exists**: `mlops-pipeline-automation.md` "MLOps for LLM Applications".

**Cross-cutting:**
- Knowledge-cutoff acknowledgement — **Exists** (SKILL.md:16 "Tool inventory is calibrated to 2026-05")
- Capability-tier abstraction (avoid naming vendor SKUs) — **Exists** (SKILL.md:14)
- TorchServe maintenance-mode flag — **Exists** in router (SKILL.md:304-305) but NOT in wrapper (see §1.6)
- Sister-pack triggers (llm-specialist, training-optimization, pytorch-engineering, ordis-security-architect) — **Exists** in router; absent/incomplete in wrapper.

**Coverage status:** No structural gaps identified. Pack is exhaustive for its declared scope.

### 2.3 Currency (research-flag)

Domain is fast-evolving (LLM serving and observability tools shift quarterly). Pack acknowledges this in `SKILL.md:16` ("Tool inventory is calibrated to 2026-05 ... should be re-checked against vendor docs"). The 2026-05 calibration matches today's date — sheets are current at time of review.

**Specific currency markers observed:**

- `quantization-for-inference.md:44-51` — explicit PyTorch namespace migration note: "PyTorch 2.x has consolidated quantization under `torch.ao.quantization`. The legacy `torch.quantization` module forwards ... and is deprecated for new code." Calls out PT2E (PyTorch 2 Export) graph-based path with `prepare_pt2e` / `convert_pt2e`. Cites docs URLs.
- `model-serving-patterns.md:Part 2` — TorchServe titled "Legacy / Maintenance Mode" (not removed; kept for existing-system support).
- `model-serving-patterns.md:Part 8` — vLLM, SGLang, TensorRT-LLM, TGI, Triton as the current LLM serving stacks. Continuous batching, PagedAttention, speculative decoding, KV-cache management named explicitly.
- `production-monitoring-and-alerting.md:Section 8` — Phoenix (Arize), Langfuse, OpenTelemetry GenAI semantic conventions.
- `mlops-pipeline-automation.md` — Prefect, Dagster, Flyte, Airflow all enumerated in plugin.json keywords; section "MLOps for LLM Applications" covers prompt versioning, RAG-index rebuilds, fine-tune pipelines, eval gates.
- `experiment-tracking-and-versioning.md:Part 3` — explicit LLM-work tracking: prompt versions, RAG evaluation sets, lineage.

### 2.4 Discoverability and triggers

Router SKILL.md `description` (frontmatter line 3): "Router skill directing to deployment, optimization, MLOps, and monitoring guides." — terse, but follows the router convention used throughout this marketplace (most `using-X` skills have similarly terse descriptions because they are loaded via the slash-command wrapper or via parent meta-router rather than via description-based discovery).

Trigger surface (SKILL.md:21-25):
- Symptoms: "production", "deploy", "serve model", "MLOps", "monitoring", "optimize inference", "vLLM", "SGLang", "TensorRT-LLM", "Phoenix", "Langfuse"
- Negative triggers (SKILL.md:27): training optimization, model architecture, PyTorch infrastructure, prompt/RAG/agent design quality.

Discovery via the wrapper at `.claude/commands/ml-production.md` is degraded by the drift documented in §1.6 — wrapper-based users miss the LLM-tool trigger surface entirely.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Evidence |
|---|-----------|:------:|----------|
| 1 | Domain coverage completeness | Pass | All 10 expected sub-skills exist; LLM era covered explicitly in 5 sheets + router |
| 2 | Router quality (description, decision tree, dual-routing) | Pass | SKILL.md frontmatter + decision tree + integration examples + 8-row sister-pack triggers + clarification questions for ambiguous queries (SKILL.md:177-208) |
| 3 | Sheet depth & TDD/RED-GREEN-REFACTOR rigor | Pass | 4 of 10 sheets use explicit RED/GREEN/REFACTOR structure (deployment-strategies, scaling, mlops, debugging, experiment-tracking); two sheets exceed 3000 lines each |
| 4 | Cross-pack boundaries (llm-specialist, training-optimization, pytorch-engineering, security-architect) | Pass | Six bidirectional trigger tables + explicit ops-vs-generation-quality boundary (SKILL.md:232-263) |
| 5 | Commands (frontmatter shape, scope, tool restrictions) | Minor | `/deploy-model` lacks `"Skill"` in `allowed-tools` though body cites a Skill load |
| 6 | Agents (SME protocol compliance, scope boundaries) | Pass | Both agents end description with SME tail; both cite `meta-sme-protocol`; both require four protocol sections; both have positive+negative activation examples |
| 7 | Slash-command wrapper alignment with router | **Major** | Wrapper at `.claude/commands/ml-production.md` is pre-LLM-refresh; missing vLLM/SGLang/TensorRT-LLM/Phoenix/Langfuse/TorchServe-deprecation/LLM-output-quality clarification — material drift from the v1.2.0 SKILL.md |
| 8 | Marketplace registration & metadata accuracy | Minor | Catalog says "11 skills" while plugin.json says "10 reference sheets, 3 commands, 2 agents"; SKILL.md count is 1, sheet count is 10 — verbiage is loose; legacy `yzmir/ai-engineering-expert/...` path in SKILL.md:471 may not resolve |

**Overall: Pass with one Major (wrapper drift) and a handful of Minor issues.** Pack is structurally sound, comprehensively scoped, and behaviourally rigorous. The wrapper file is the load-bearing defect — it actively misroutes users away from LLM-era coverage that the SKILL.md provides.

---

## 4. Behavioral Tests

Spot-checks against the SKILL.md routing logic. Subagent dispatch not run (report-only); reasoning-based traces follow.

### Test A — Pressure: "Just deploy this LLM with TorchServe, we're already set up"

- **Expected (router):** Acknowledge existing setup, route to `model-serving-patterns.md` (Part 2 TorchServe, plus Part 8 LLM-serving stacks). Flag TorchServe as maintenance-mode (SKILL.md:304-305, "Common Routing Mistakes" row). Suggest evaluation of vLLM/SGLang/TGI for new LLM work; cross-trigger llm-specialist for strategy/capability choice.
- **Wrapper behaviour:** Wrapper file says "Tool choice doesn't change routing" (commands/ml-production.md:221) and **never mentions LLM-serving stacks or the deprecation flag**. A user reaching this via the slash command would not learn that TorchServe is in maintenance mode for new systems.
- **Verdict:** Router PASS, wrapper FAIL. Reinforces §1.6 Major.

### Test B — Edge case: "Our LLM is hallucinating and burning $20k/day"

- **Expected (router):** Dual-route. ml-production: `production-monitoring-and-alerting.md` (Phoenix/Langfuse + cost-budget alerts + hallucination drift) + `production-debugging-techniques.md` (root-cause cost spikes — token bloat, retry storms, bad routing). llm-specialist: `llm-evaluation-metrics.md` (methodology), `prompt-engineering-patterns.md` / `rag-architecture-patterns.md` (remediation).
- **SKILL.md trace:** Example 4 (SKILL.md:413-420) gives exactly this four-step recipe by name. Pass.
- **Verdict:** Router PASS.

### Test C — Ambiguity: "Set up MLOps for my team"

- **Expected (router):** Ask clarifying question — current pain point: tracking, automation, prompt/eval-set versioning, or combination (SKILL.md:202-207). Route to one or more of `experiment-tracking-and-versioning.md` and `mlops-pipeline-automation.md` based on the answer.
- **SKILL.md trace:** Lines 200-208 and Example 3 (lines 395-410) both implement this. Pass.

### Test D — Real-world complexity: "RAG production system, costs spiking, retrieval quality degrading"

- **Expected:** Triple-cross-trigger: llm-specialist `rag-architecture-patterns.md` (architecture), ml-production `production-monitoring-and-alerting.md` (retrieval-quality drift + cost monitoring), ml-production `scaling-and-load-balancing.md` (vector-store deployment, latency monitoring).
- **SKILL.md trace:** Multi-Concern row "RAG production system" (line 222) names the exact sequence. Pass.

### Test E — Boundary: "My training is slow"

- **Expected:** Hand off to `training-optimization` (SKILL.md:265-275, "With training-optimization" + ambiguity-resolution).
- **SKILL.md trace:** Clarifying question in "My model is too slow" (line 179): "Is this inference latency, or training time? Training → Route to `training-optimization` (wrong pack)." Pass.

### Test F — Agent activation: `mlops-architect` vs `inference-debugger`

- Activation examples (mlops-architect.md:18-41, inference-debugger.md:18-41) cover positive AND negative cases. Each agent explicitly redirects the other's tasks ("I do NOT debug" / "I do NOT design"). Scope boundaries cleanly partitioned. Pass.

### Test G — Pressure on commands: `/deploy-model "my_model.pt"`

- Frontmatter offers `[model_path_or_name]` shape (correct).
- Body provides decision tree, REST/gRPC/Batch pattern templates, containerization, canary/shadow scaffolds, monitoring metrics, rollout checklist, rollback procedure.
- Issue: command body recommends `Load skill: yzmir-ml-production:using-ml-production` (lines 300-308), but `Skill` tool is missing from `allowed-tools` (frontmatter line 3). User can still load by description match, but the explicit command-driven Skill dispatch is gated off. Minor.

### Test H — Shortcut pressure: "Just quantize this — INT8 should be fine"

- **Expected (router):** Resist single-route to `quantization-for-inference.md`. Multi-concern row line 219: "Quantize model and serve with vLLM" → 1. Quantization (ops) → 2. LLM-serving Part of serving-patterns. For LLMs specifically, the router dual-routes: ml-production owns the ops (AWQ/GPTQ wiring, calibration, hardware fit), llm-specialist owns the format-choice (quality tradeoff). Router lines 305-307 (Common Routing Mistakes): "Quantize an LLM → Route both: llm-specialist (which format suits the task) + ml-production (AWQ/GPTQ ops, hardware fit)".
- **Sheet trace:** `quantization-for-inference.md:29` explicitly defers LLM format-choice to llm-specialist and retains ops.
- **Verdict:** PASS — router resists the shortcut and dual-routes.

### Test I — Tool-name-baiting: "Kubernetes is acting weird with our model server"

- **Expected:** Route by concern, not tool. Router lines 298-299 (Routing Mistakes table) says "Deploy with Kubernetes → Defer to Kubernetes docs (wrong) / Category 2: serving-patterns or deployment-strategies (correct) — Kubernetes is tool choice, not routing concern". Symptom triage first: is this a serving-pattern issue, a deployment-strategy issue (canary misconfig), or a scaling issue (autoscaler thrash)? Ask clarifying question.
- **Verdict:** PASS.

### Test J — Negative test: agent activation on out-of-domain

- `mlops-architect.md:33-41`: explicit negative example "My model inference is slow → Do NOT activate — performance issue, use /diagnose-inference". And "Deploy this model to production → Do NOT activate — deployment task, use /deploy-model".
- `inference-debugger.md:33-41`: explicit negative example "Design MLOps pipeline → Do NOT activate — architecture task, use mlops-architect" and "Deploy this model → Do NOT activate — deployment task, use /deploy-model".
- Both agents clearly decline and redirect.
- **Verdict:** PASS — agents have correct out-of-scope refusal patterns.

### Summary

| Scenario | Result |
|----------|--------|
| A — Pressure: TorchServe LLM | Router Pass, **Wrapper Fail** (Major) |
| B — LLM hallucination + cost spike | Pass |
| C — MLOps ambiguity | Pass |
| D — RAG production triple-trigger | Pass |
| E — Training vs inference boundary | Pass |
| F — Agent activation symmetry | Pass |
| G — `/deploy-model` tool dispatch | Pass with Minor (missing `Skill` in `allowed-tools`) |
| H — Shortcut: "just quantize this" | Pass (router resists single-route, dual-routes for LLMs) |
| I — Tool-name baiting (Kubernetes) | Pass (route by concern, not tool) |
| J — Agent out-of-domain refusal | Pass (explicit redirect for both agents) |

---

## 5. Findings

### Critical

None.

### Major

- **M1 — Slash-command wrapper drift.** `.claude/commands/ml-production.md` is a stale snapshot of the router SKILL.md (looks like pre-LLM-era v1.0/v1.1 content) while the plugin.json declares v1.2.0 with explicit LLM coverage. The wrapper omits vLLM/SGLang/TensorRT-LLM/TGI/Triton; Phoenix/Langfuse/OTel-GenAI; AWQ/GPTQ; TorchServe-deprecation flag; three-way "performance" disambiguation (speed/accuracy/LLM output quality); `ordis-security-architect` cross-reference; reasoning-models / agentic-patterns / context-engineering coverage. Users reaching the pack via `/ml-production` get materially worse routing than users who load the skill directly. Evidence: full wrapper file vs. SKILL.md cross-comparison in §1.6 above.

### Minor

- **m1 — `/deploy-model` missing `"Skill"` in `allowed-tools`.** `commands/deploy-model.md:3` lists `["Read","Bash","Glob","Grep","Write","AskUserQuestion"]`. Body lines 300-308 instruct the model to *Load skill: yzmir-ml-production:using-ml-production* but the tool is not permitted. Either add `"Skill"` or remove the explicit Skill-load instructions in favour of description-based discovery.

- **m2 — Marketplace catalog string says "11 skills".** `.claude-plugin/marketplace.json` entry for `yzmir-ml-production` reads `"Production ML - quantization, serving, MLOps, monitoring, debugging - 11 skills"`. plugin.json says "10 reference sheets, 3 commands, 2 agents". The skill-file count is 1 (router SKILL.md), reference-sheet count is 10. Pick one convention and propagate. Recommend: "10 reference sheets, 3 commands, 2 agents".

- **m3 — Legacy path in SKILL.md References.** SKILL.md:471 references `yzmir/ai-engineering-expert/using-ai-engineering` (no `plugins/` prefix and the legacy `yzmir/` faction directory). Modern layout is `plugins/yzmir-ai-engineering-expert/skills/using-ai-engineering/`. Verify and update.

- **m4 — Design-doc reference may be dangling.** SKILL.md:470 cites `docs/plans/2025-10-30-ml-production-pack-design.md`. Existence not verified during review. If the file is gone, drop the reference.

### Polish

- **p1 — Some sheets use mixed RED/GREEN/REFACTOR rigour while others use straight prose.** `deployment-strategies.md`, `scaling-and-load-balancing.md`, `mlops-pipeline-automation.md`, `experiment-tracking-and-versioning.md`, `production-debugging-techniques.md` use explicit RED/GREEN/REFACTOR; others (`quantization-for-inference.md`, `model-compression-techniques.md`, `hardware-optimization-strategies.md`, `model-serving-patterns.md`, `production-monitoring-and-alerting.md`) use Part 1..N or Section 1..N. Both styles are defensible; mention is purely consistency observation, not a defect.

- **p2 — Router SKILL.md is 476 lines.** This sits at the top of the per-router context budget. Not unusual for this marketplace (other routers run similar length), but worth flagging — the wrapper's existence is precisely what justifies a heavy router skill. With the wrapper out of date (M1), users get the heavy router unfiltered.

- **p3 — Agent `tools:` audit.** Both agents correctly omit `tools:`. Repo-consistent. No action.

---

## 6. Recommended Actions

(Report-only — no edits performed.)

| Priority | Action | Effort |
|:--------:|--------|:-----:|
| Major | Regenerate `.claude/commands/ml-production.md` from the current `SKILL.md` (or simply replace its body with an `> See: yzmir-ml-production:using-ml-production` pointer plus a 30-line summary that matches the v1.2.0 router exactly). | M |
| Minor | Add `"Skill"` to `commands/deploy-model.md` `allowed-tools` OR remove the explicit `Load skill:` block. | XS |
| Minor | Reconcile marketplace catalog string: replace `- 11 skills` with `- 10 reference sheets, 3 commands, 2 agents` (mirror plugin.json). | XS |
| Minor | Verify / fix `yzmir/ai-engineering-expert/using-ai-engineering` reference path in SKILL.md:471. | XS |
| Minor | Verify existence of `docs/plans/2025-10-30-ml-production-pack-design.md`; drop the citation if absent. | XS |
| Polish | Optional: harmonize sheet rhetorical structure (RED/GREEN/REFACTOR vs. Part-N) on the next major refresh. | M |

Suggested version bump on remediation: **patch (1.2.0 → 1.2.1)**. The wrapper realignment is a maintenance fix to existing content, not new capability. Demote to minor (1.2.x → 1.3.0) only if the wrapper rewrite materially expands coverage.

---

## 7. Reviewer Notes

### 7.1 What works well in this pack

1. **Ops-vs-generation-quality boundary is the strongest in the marketplace.** The pack draws an explicit operational line against `llm-specialist`: serving stacks / quantization ops / observability platforms belong here; eval methodology / prompt design / RAG architecture / fine-tune strategy choice belong there. This is enforced via six bidirectional trigger tables (SKILL.md:255-261) and a per-row hand-off matrix in "Common Routing Mistakes" (SKILL.md:304-307). A user with an LLM production question is reliably dual-routed.
2. **Knowledge-cutoff and capability-tier discipline.** SKILL.md:14-16 establishes both "capability tier, not model ID" and "tool inventory calibrated to 2026-05" — protections against vendor SKU churn and against silent staleness. Few packs in this marketplace bother with this.
3. **TorchServe deprecation flag.** SKILL.md:304-305 (Routing Mistakes table) and `model-serving-patterns.md` Part 2 both flag TorchServe as maintenance-mode rather than silently recommending it. Models with stale training data would otherwise route happily to a maintenance-mode serving framework. This is good safety guidance.
4. **RED/GREEN/REFACTOR rigour on five of ten sheets.** Deployment-strategies, scaling, mlops-pipeline, experiment-tracking, and production-debugging all use the explicit anti-pattern-then-pattern-then-pressure-test structure that matches the rest of this marketplace's discipline.
5. **Clean agent partition with explicit negative examples.** Both agents include `Action: Do NOT activate` examples that name the *other* agent or command. Avoids the common SME-agent trap where two agents both claim a borderline scenario.

### 7.2 Watch items for future maintenance

1. **Wrapper drift is a class of defect, not a one-off.** Every plugin in this marketplace has a `using-X` skill paired with a `.claude/commands/X.md` wrapper, and any router refresh that touches the SKILL.md must touch the wrapper or be flagged as incomplete. Recommend adding a cross-check to the pre-commit hook (or to the meta-skillpack-maintenance audit pass) that diffs the wrapper against the SKILL.md and emits a warning when they diverge significantly.
2. **Sheet structural inconsistency** (p1) is purely cosmetic — but on a v2.0 refresh, harmonizing all sheets to RED/GREEN/REFACTOR would simplify both maintenance and discoverability.
3. **The `Skill` tool gap on `/deploy-model`** (m1) is a small symptom of a larger question: do command bodies in this marketplace consistently declare every tool they actually use? Worth a marketplace-wide sweep at the next maintenance pass.

### 7.3 Methodology used in this review

- **Stage 1 (Investigation):** Read all metadata files (`plugin.json`, marketplace catalog entry, wrapper, all SKILL.md and reference-sheet headers, both agent frontmatter blocks, all three command frontmatter blocks). Counted section headers across all 11 skill files. Sampled mid-content of `quantization-for-inference.md` (lines 1-60) to verify currency markers (PT2E reference confirmed).
- **Stage 2 (Structure Review):** Generated 8-dimension scorecard against the rubric in `reviewing-pack-structure.md`. Cross-referenced agent SME-protocol compliance against `meta-sme-protocol:sme-agent-protocol`. Diffed wrapper against SKILL.md row-by-row across 7 LLM-era topics.
- **Stage 3 (Behavioral Testing):** 10 reasoning-based traces against the router and the agent activation rules. No subagent dispatch executed (report-only). For a remediation pass, scenarios H and I (shortcut pressure and tool-name baiting) and the wrapper-vs-router contrast in scenario A would be the highest-value subagent dispatches.
- **Stage 4 (Discussion):** Findings sorted into Critical / Major / Minor / Polish per the rubric. One Major, four Minor, three Polish. No Critical.
- **Stage 5 (Execution):** Skipped per task brief — this is a report-only review.

### 7.4 Bottom line

- The pack is in excellent shape on substance. The router SKILL.md is one of the most thorough in this marketplace — explicit ops-vs-generation-quality boundary, six bidirectional trigger tables with llm-specialist, three-way performance disambiguation, TorchServe maintenance-mode flag, capability-tier abstraction, knowledge-cutoff acknowledgement, multi-concern routing in execution order.
- The single Major finding (wrapper drift) is structurally important because the wrapper is the *user-facing entry point* per the repo's CLAUDE.md convention. A v1.2.0 plugin with a v1.0-era wrapper undermines the entire LLM-refresh effort.
- Both agents are SME-protocol-compliant with positive AND negative activation examples and clear scope boundaries — repo-best-practice.
- Reference sheets show appropriate depth (934-3561 lines) and contain LLM-era content embedded throughout (Part 8/Part 10/Section 8 patterns across sheets), not merely tacked on.
- Behavioral testing was reasoning-based (report-only, no subagent dispatch). Spot-checks of routing logic against the SKILL.md decision tree showed expected behaviour on all 7 scenarios for the router itself; the wrapper failed the one scenario where it was exercised.
- No content gaps identified. The pack covers its declared production-ML scope exhaustively as of 2026-05 calibration.

`yzmir-ml-production` v1.2.0 is a **Pass** on substance and a **Pass-with-Major** on integration. The single Major (wrapper drift) is the only blocker preventing a clean Pass and is straightforward to remediate (~30 minutes of mechanical regeneration). The remaining Minor findings are housekeeping items that can be batched with the next routine maintenance pass. Patch-level bump (1.2.1) is the appropriate next version on remediation; no Critical issues, no Rebuild recommendation, no Major structural change required.

---

## Appendix A — Wrapper-vs-Router Diff (concrete evidence for Major M1)

Side-by-side excerpts demonstrating the drift between `.claude/commands/ml-production.md` (wrapper) and `plugins/yzmir-ml-production/skills/using-ml-production/SKILL.md` (router):

**A.1 Category-2 "Serving Infrastructure" symptom list.**

Wrapper (`commands/ml-production.md:43`):
> **Symptoms**: "How to serve model", "need API endpoint", "deploy to production", "containerize model", "scale serving", "load balancing", "traffic management"

Router (`SKILL.md:74`):
> **Symptoms**: "How to serve model", "need API endpoint", "deploy to production", "containerize model", "scale serving", "load balancing", "traffic management", "vLLM", "SGLang", "TensorRT-LLM", "TGI", "Triton", "continuous batching", "PagedAttention", "speculative decoding", "KV-cache management"

The wrapper trigger surface misses 8 LLM-specific keywords. A user asking "should we use vLLM or SGLang?" via `/ml-production` will not get this skill activated by description match.

**A.2 Category-4 "Observability" symptom list.**

Wrapper (`commands/ml-production.md:85`):
> **Symptoms**: "Monitor production", "model degrading", "detect drift", "production debugging", "alert on failures", "model not working in prod", "performance issues in production"

Router (`SKILL.md:119`):
> **Symptoms**: "Monitor production", "model degrading", "detect drift", "production debugging", "alert on failures", "model not working in prod", "performance issues in production", "LLM in production", "hallucination rate", "tool-call success", "cost-per-request spiking", "prompt-injection attempts", "drift in RAG retrieval quality"

Wrapper misses 6 LLM-observability keywords including hallucination, cost spikes, and prompt-injection.

**A.3 Performance disambiguation.**

Wrapper (`commands/ml-production.md:102-104`):
> "By performance, do you mean inference speed or model accuracy?"
> - Speed → Category 1 (optimization) or Category 2 (serving/scaling)
> - Accuracy → Category 4 (observability — drift detection)

Router (`SKILL.md:138-142`):
> If "performance" = speed/latency → might be Category 1 (optimization) or Category 2 (serving/scaling)
> If "performance" = accuracy degradation → Category 4 (observability — drift detection)
> If "performance" = LLM output quality → Category 4 (observability — wire Phoenix/Langfuse) **and** llm-specialist (eval methodology)
> **Ask clarifying question**: "By performance, do you mean inference speed, model accuracy, or LLM output quality?"

Wrapper has a two-way clarification; router has a three-way clarification with the LLM-output-quality branch that includes dual-routing into llm-specialist.

**A.4 "Common Routing Mistakes" table.**

Wrapper (`commands/ml-production.md:215-222`) has 5 rows; the TorchServe row reads:
> "We use TorchServe → Skip routing → Still route to serving-patterns → Tool choice doesn't change routing"

Router (`SKILL.md:298-307`) has 8 rows. The TorchServe row reads:
> "Just use TorchServe → Route uncritically to TorchServe content (wrong) → Flag: TorchServe is in maintenance mode (PyTorch deprecated active development). Route to LLM-serving Part of `model-serving-patterns.md` (vLLM/SGLang/TensorRT-LLM/TGI/Triton) for LLMs..."

Plus three additional rows the wrapper lacks entirely:
- "Deploy LLM" → dual-route into llm-specialist + ml-production
- "Hallucinations in production" → route both ml-production (Phoenix/Langfuse) + llm-specialist (eval methodology + remediation)
- "Quantize an LLM" → route both llm-specialist (format choice) + ml-production (AWQ/GPTQ ops)

**A.5 Sister-pack section.**

Wrapper has three sister-pack subsections (llm-specialist, training-optimization, pytorch-engineering). Router has four — adds `ordis-security-architect` (SKILL.md:289-292):
> "For AI/LLM threat modeling — prompt injection, data exfiltration via tool-calls, supply-chain risk on model artifacts, PII leakage in logs, model-registry access control — route to `ordis-security-architect`..."

Wrapper users get no path to security-architect for AI threat modeling.

**A.6 Specialist skills catalog.**

Wrapper has no enumerated catalog of the 10 reference sheets. Router has a numbered catalog (SKILL.md:451-464) with per-sheet content summaries including:
- Sheet 4 explicitly lists "**LLM-serving Part: vLLM, SGLang, TensorRT-LLM, TGI, Triton, continuous batching, PagedAttention, speculative decoding, KV-cache management**"
- Sheet 9 explicitly lists "**LLM observability: Phoenix, Langfuse, OpenTelemetry GenAI semantic conventions**"

Wrapper users get no machine-readable map of which sheet covers what — they must rely on the truncated symptom lists in §A.1 and §A.2.

**Net assessment:** The wrapper file appears to predate the v1.2.0 LLM refresh and was not regenerated when the SKILL.md was updated. Wrapper users get a strictly worse routing experience and may be steered toward maintenance-mode tools (TorchServe) or denied access to relevant sister packs (llm-specialist dual-routing, ordis-security-architect for threat modeling). This is the load-bearing defect of the review and the primary remediation target.
