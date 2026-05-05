---
name: using-ml-production
description: Router skill directing to deployment, optimization, MLOps, and monitoring guides.
---

# Using ML Production

## Overview

This meta-skill routes you to the right production deployment skill based on your concern. Load this when you need to move ML models to production but aren't sure which specific aspect to address.

**Core Principle**: Production concerns fall into four categories. Identify the concern first, then route to the appropriate skill. Tools and infrastructure choices are implementation details, not routing criteria.

**Capability tiers, not model IDs.** Sheets describe model and hardware capabilities (e.g., "frontier reasoning model", "mid-tier serving GPU", "edge accelerator") rather than naming specific SKUs. Vendor model names and exact GPU SKUs change quarterly; capability tiers are stable.

**Knowledge-cutoff acknowledgement.** Tool inventory is calibrated to 2026-05. Fast-moving areas — LLM serving stacks (vLLM, SGLang, TensorRT-LLM, TGI, Triton), observability platforms (Phoenix, Langfuse, OTel GenAI), quantization toolchains (`torch.ao.quantization`, AWQ, GPTQ), and MLOps platforms — should be re-checked against vendor docs before architecting new systems. Treat sheet content as a structured starting point, not a substitute for current documentation.

## When to Use

Load this skill when:
- Deploying ML models (classical or LLM) to production
- Optimizing model inference (speed, size, cost)
- Setting up MLOps workflows (tracking, automation, CI/CD)
- Monitoring or debugging production models, including LLM-specific signals (hallucination rate, tool-call success, prompt-injection attempts)
- User mentions: "production", "deploy", "serve model", "MLOps", "monitoring", "optimize inference", "vLLM", "SGLang", "TensorRT-LLM", "Phoenix", "Langfuse"

**Don't use for**: Training optimization (use `training-optimization`), model architecture selection (use `neural-architectures`), PyTorch infrastructure (use `pytorch-engineering`), prompt/RAG/agent design quality (use `llm-specialist`).

---

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-ml-production/SKILL.md`

Reference sheets like `quantization-for-inference.md` are at:
  `skills/using-ml-production/quantization-for-inference.md`

NOT at:
  `skills/quantization-for-inference.md` ← WRONG PATH

When you see a link like `[quantization-for-inference.md](quantization-for-inference.md)`, read the file from the same directory as this SKILL.md.

---

## Routing by Concern

### Category 1: Model Optimization

**Symptoms**: "Model too slow", "inference latency high", "model too large", "need to optimize for edge", "reduce model size", "speed up inference", "quantize model", "INT8/INT4", "AWQ", "GPTQ"

**When to route here**:
- Model itself is the bottleneck (not infrastructure)
- Need to reduce model size or increase inference speed
- Deploying to resource-constrained hardware (edge, mobile)
- Cost optimization through model efficiency
- Quantization *operations* (the mechanics of converting weights, calibration, kernel fit)

**Routes to**:
- [quantization-for-inference.md](quantization-for-inference.md) - `torch.ao.quantization`, AWQ, GPTQ, INT8/INT4, post-training quantization, QAT, calibration
- [model-compression-techniques.md](model-compression-techniques.md) - Pruning, distillation, architecture optimization
- [hardware-optimization-strategies.md](hardware-optimization-strategies.md) - GPU/CPU/edge tuning, batch sizing, hardware/quant-format fit

**Key question to ask**: "Is the MODEL the bottleneck, or is it infrastructure/serving?"

**LLM quantization split**: Choosing *which* quantization format suits a given LLM task (quality vs throughput tradeoffs, instruction-following degradation) is `llm-specialist` territory. The *operations* — wiring up AWQ/GPTQ, running calibration, validating kernel support — live here.

---

### Category 2: Serving Infrastructure

**Symptoms**: "How to serve model", "need API endpoint", "deploy to production", "containerize model", "scale serving", "load balancing", "traffic management", "vLLM", "SGLang", "TensorRT-LLM", "TGI", "Triton", "continuous batching", "PagedAttention", "speculative decoding", "KV-cache management"

**When to route here**:
- Need to expose model as API or service (classical model or LLM)
- Questions about serving patterns (REST, gRPC, batch, streaming)
- Deployment strategies (gradual rollout, A/B testing)
- Scaling concerns (traffic, replicas, autoscaling)
- LLM-serving stacks: vLLM, SGLang, TensorRT-LLM, TGI, Triton — **covered in the LLM-serving Part of [model-serving-patterns.md](model-serving-patterns.md)** alongside continuous batching, PagedAttention, and speculative decoding ops

**Routes to**:
- [model-serving-patterns.md](model-serving-patterns.md) - FastAPI, TorchServe (maintenance mode — see routing mistakes), gRPC, ONNX, batching, containerization, **plus LLM-serving Part: vLLM / SGLang / TensorRT-LLM / TGI / Triton, continuous batching, PagedAttention, speculative decoding**
- [deployment-strategies.md](deployment-strategies.md) - A/B testing, canary, shadow mode, rollback procedures
- [scaling-and-load-balancing.md](scaling-and-load-balancing.md) - Horizontal scaling, autoscaling, load balancing, cost optimization

**Key distinction**:
- Serving patterns = HOW to expose model (API, container, batching, LLM-serving stack)
- Deployment strategies = HOW to roll out safely (gradual, testing, rollback)
- Scaling = HOW to handle traffic (replicas, autoscaling, balancing)

---

### Category 3: MLOps Tooling

**Symptoms**: "Track experiments", "version models", "automate deployment", "reproducibility", "CI/CD for ML", "feature store", "model registry", "experiment management", "version prompts", "version RAG eval set"

**When to route here**:
- Need workflow/process improvements
- Want to track experiments, version models, version prompts, or version RAG eval sets
- Need to automate training-to-deployment pipeline
- Team collaboration and reproducibility concerns

**Routes to**:
- [experiment-tracking-and-versioning.md](experiment-tracking-and-versioning.md) - MLflow, Weights & Biases, Comet, Hugging Face Hub, model registries, prompt versioning, RAG/eval-set versioning, lineage
- [mlops-pipeline-automation.md](mlops-pipeline-automation.md) - CI/CD for ML, feature stores, data validation, automated retraining, orchestration

**Key distinction**:
- Experiment tracking = Research/development phase (track runs, version models, version prompts/eval sets)
- Pipeline automation = Production phase (automate workflows, CI/CD)

**Multi-concern**: Queries like "track experiments AND automate deployment" → route to BOTH skills

---

### Category 4: Observability

**Symptoms**: "Monitor production", "model degrading", "detect drift", "production debugging", "alert on failures", "model not working in prod", "performance issues in production", "LLM in production", "hallucination rate", "tool-call success", "cost-per-request spiking", "prompt-injection attempts", "drift in RAG retrieval quality"

**When to route here**:
- Model already deployed, need to monitor or debug
- Detecting production issues (drift, errors, degradation, accuracy regressions)
- Setting up alerts and dashboards
- Root cause analysis for production failures
- LLM-specific telemetry: trace generations, score outputs, track cost/token spend, capture tool-call traces, surface prompt-injection signals

**Routes to**:
- [production-monitoring-and-alerting.md](production-monitoring-and-alerting.md) - Metrics, drift detection, dashboards, alerts, SLAs, **LLM observability platforms: Phoenix, Langfuse, OpenTelemetry GenAI semantic conventions** for trace/eval/cost monitoring
- [production-debugging-techniques.md](production-debugging-techniques.md) - Error analysis, profiling, rollback procedures, post-mortems, tool-call failure forensics

**Key distinction**:
- Monitoring = Proactive (set up metrics, alerts, detect issues early)
- Debugging = Reactive (diagnose and fix existing issues)

**LLM observability split**: The *plumbing* — Phoenix/Langfuse/OTel-GenAI deployment, retention, dashboards, alerts on cost/latency/error budgets — lives here. The *eval methodology* feeding those dashboards (which metrics to compute, how to score hallucinations, how to build judges) lives in `llm-specialist` (`llm-evaluation-metrics.md`). Wire them together: methodology defines the signal; this pack ships it.

**"Performance" ambiguity**:
- If "performance" = speed/latency → might be Category 1 (optimization) or Category 2 (serving/scaling)
- If "performance" = accuracy degradation → Category 4 (observability - drift detection)
- If "performance" = LLM output quality → Category 4 (observability — wire Phoenix/Langfuse) **and** llm-specialist (eval methodology)
- **Ask clarifying question**: "By performance, do you mean inference speed, model accuracy, or LLM output quality?"

---

## Routing Decision Tree

```
User query → Identify primary concern

Is model THE problem (size/speed)?
  YES → Category 1: Model Optimization
  NO → Continue

Is it about HOW to expose/deploy model (incl. LLM-serving stack)?
  YES → Category 2: Serving Infrastructure
  NO → Continue

Is it about workflow/process/automation?
  YES → Category 3: MLOps Tooling
  NO → Continue

Is it about monitoring/debugging in production (incl. LLM telemetry)?
  YES → Category 4: Observability
  NO → Ask clarifying question

Is the question about LLM generation quality, prompt design, RAG retrieval design, or agent behavior design?
  YES → Hand off to llm-specialist (this pack does ops, not generation quality)

Ambiguous? → Ask ONE question to clarify concern category
```

---

## Clarification Questions for Ambiguous Queries

### Query: "My model is too slow"

**Ask**: "Is this inference latency (how fast predictions are), or training time?"
- Training → Route to `training-optimization` (wrong pack)
- Inference → Follow-up: "Have you profiled to find bottlenecks?"
  - Model is bottleneck → Category 1 (optimization)
  - Infrastructure/batching issue → Category 2 (serving)
  - LLM-specific (KV cache, batching strategy) → Category 2 LLM-serving Part **+** llm-specialist `llm-inference-optimization.md`

### Query: "I need to deploy my model"

**Ask**: "What's your deployment target — cloud server, edge device, batch processing, or LLM endpoint?"
- Cloud/server → Category 2 (serving-patterns, then maybe deployment-strategies if gradual rollout needed)
- Edge/mobile → Category 1 (optimization first for size/speed) + Category 2 (serving)
- Batch → Category 2 (serving-patterns - batch processing)
- LLM endpoint → Category 2 LLM-serving Part of `model-serving-patterns.md` **+** llm-specialist (`llm-inference-optimization.md` for capability/strategy choice)

### Query: "My model isn't performing well in production"

**Ask**: "By performance, do you mean inference speed, prediction accuracy, or LLM output quality (hallucination, tool-call failure)?"
- Speed → Category 1 (optimization) or Category 2 (serving/scaling)
- Accuracy → Category 4 (observability — drift detection, monitoring)
- LLM output quality → Category 4 (Phoenix/Langfuse plumbing) + llm-specialist (`llm-evaluation-metrics.md` for methodology, `prompt-engineering-patterns.md` / `rag-architecture-patterns.md` for fixes)

### Query: "Set up MLOps for my team"

**Ask**: "What's the current pain point — experiment tracking, automated deployment, prompt/eval-set versioning, or some combination?"
- Tracking/versioning (incl. prompt/eval-set) → Category 3 (experiment-tracking-and-versioning)
- Automation/CI/CD → Category 3 (mlops-pipeline-automation)
- Multiple → Route to multiple skills

---

## Multi-Concern Scenarios

Some queries span multiple categories. Route to ALL relevant skills in logical order:

| Scenario | Route Order | Why |
|----------|-------------|-----|
| "Optimize and deploy model" | 1. Optimization → 2. Serving | Optimize BEFORE deploying |
| "Deploy and monitor model" | 1. Serving → 2. Observability | Deploy BEFORE monitoring |
| "Track experiments and automate deployment" | 1. Experiment tracking → 2. Pipeline automation | Track BEFORE automating |
| "Quantize model and serve with vLLM" | 1. Quantization (ops) → 2. LLM-serving Part of serving-patterns | Optimize BEFORE serving |
| "Deploy with A/B testing and monitor" | 1. Deployment strategies → 2. Monitoring | Deploy strategy BEFORE monitoring |
| "Deploy LLM to production" | 1. llm-specialist (model/strategy choice) → 2. ml-production serving + 3. ml-production observability | Generation-quality choice frames the ops |
| "RAG production system" | 1. llm-specialist (`rag-architecture-patterns.md`) → 2. ml-production (vector-store deployment, retrieval-quality observability, latency monitoring) | Architecture decisions frame the ops |
| "Agent in production" | 1. llm-specialist (`agentic-patterns-and-mcp.md`) → 2. ml-production (tool-call observability, error rates, cost monitoring) → 3. axiom-engineering-foundations (sandboxing, system design) | Agent design frames ops + safety |

**Principle**: Route in execution order (what needs to happen first).

---

## Relationship with Other Packs

### With llm-specialist

The two packs split along the **ops vs. generation-quality boundary**.

**ml-production owns** (this pack):
- General serving stacks: vLLM, SGLang, TensorRT-LLM, TGI, Triton — deployment, batching ops, KV-cache config, scaling
- Quantization for inference *as ops*: `torch.ao.quantization`, AWQ, GPTQ wiring, calibration, hardware fit
- MLOps: CI/CD, registries, feature stores, retraining
- Deployment patterns: containers, canary, A/B, rollback
- Monitoring & observability platforms: Phoenix, Langfuse, OpenTelemetry GenAI semantic conventions, dashboards, alerts, drift detection

**llm-specialist owns** (sister pack — 10 reference sheets):
- Prompt engineering (`prompt-engineering-patterns.md`)
- Reasoning-model use (`reasoning-models.md`)
- Agentic patterns + MCP (`agentic-patterns-and-mcp.md`)
- RAG architecture (`rag-architecture-patterns.md`)
- Fine-tuning strategy choice (`llm-finetuning-strategies.md`)
- Context engineering & prompt caching (`context-engineering-and-prompt-caching.md`, `context-window-management.md`)
- Evaluation methodology (`llm-evaluation-metrics.md`)
- Inference-strategy choice (`llm-inference-optimization.md`)
- Safety / alignment (`llm-safety-alignment.md`)

**Concrete bidirectional triggers**:

| Query | llm-specialist contributes | ml-production contributes |
|-------|---------------------------|---------------------------|
| "Deploy LLM to production" | Inference-strategy + capability-tier choice | Serving stack, monitoring, scaling |
| "Quantize LLM" | Which format suits the task (quality tradeoffs) | `torch.ao.quantization`, AWQ/GPTQ ops, hardware fit |
| "LLM observability" | Eval methodology to feed into dashboards | Phoenix / Langfuse / OTel-GenAI plumbing |
| "RAG production system" | RAG architecture + retrieval design | Vector-store deployment, retrieval-quality observability, latency monitoring |
| "Agent in production" | Agentic patterns, MCP, anti-patterns | Tool-call observability, error rates, cost monitoring (also: axiom-engineering-foundations for sandboxing/system design) |

**Rule of thumb**: If the question is "what should the model/prompt/agent do and how should it be designed?" → llm-specialist. If the question is "how do we run, observe, and operate it at scale?" → ml-production. Most LLM production questions need **both**.

### With training-optimization

**Clear boundary**:
- training-optimization = Training phase (convergence, hyperparameters, training speed)
- ml-production = Inference phase (deployment, serving, monitoring)

**"Too slow" disambiguation**:
- Training slow → training-optimization
- Inference slow → ml-production

**Bidirectional**: training-optimization should send users here when they're done training; ml-production should send users back when they need to retrain (drift response, dataset updates).

### With pytorch-engineering

**pytorch-engineering covers**: Foundation (distributed training, profiling, memory management)

**ml-production covers**: Production-specific (serving APIs, deployment patterns, MLOps, production observability)

**When to use both**:
- "Profile production inference" → pytorch-engineering (profiling techniques) + ml-production (production context, observability wiring)
- "Optimize serving performance" → ml-production (serving patterns) + pytorch-engineering (low-level profiling, CUDA/memory)

**Bidirectional**: pytorch-engineering profiling skills point here for production-context observability; this pack's debugging sheet points back to pytorch-engineering for low-level CUDA/memory work.

### With ordis-security-architect

For AI/LLM threat modeling — prompt injection, data exfiltration via tool-calls, supply-chain risk on model artifacts, PII leakage in logs, model-registry access control — route to `ordis-security-architect` (threat modeling, controls design, security review). This pack handles the *operational signals* (prompt-injection attempt rate as an observability metric) but the *threat model and control design* live in security-architect.

---

## Common Routing Mistakes

| Query | Wrong Route | Correct Route | Why |
|-------|-------------|---------------|-----|
| "Model too slow in production" | Immediately to quantization | Ask: inference or training? Then model vs infrastructure? | Could be serving/batching issue, not model |
| "Deploy with Kubernetes" | Defer to Kubernetes docs | Category 2: serving-patterns or deployment-strategies | Kubernetes is tool choice, not routing concern |
| "Set up MLOps" | Route to one skill | Ask about specific pain point — could be tracking AND automation AND prompt/eval-set versioning | MLOps spans multiple skills |
| "Performance issues" | Assume accuracy | Ask: speed, accuracy, or LLM output quality? | Performance is ambiguous |
| "Deploy LLM" | Skip llm-specialist, route only here | Route both: llm-specialist (strategy/capability choice), ml-production (serving stack, observability, scaling) | Generation-quality choices frame the ops |
| "Just use TorchServe" | Route uncritically to TorchServe content | Flag: TorchServe is in maintenance mode (PyTorch deprecated active development). Route to LLM-serving Part of `model-serving-patterns.md` (vLLM/SGLang/TensorRT-LLM/TGI/Triton) for LLMs; for classical models route to serving-patterns but recommend evaluating Triton or framework-native serving | Don't recommend a maintenance-mode tool for new systems |
| "We use TorchServe" (existing system) | Skip routing | Route to serving-patterns, note migration path | Existing systems still need ops guidance, but flag the deprecation |
| "Hallucinations in production" | Route only to llm-specialist | Route both: ml-production (Phoenix/Langfuse to detect+alert) + llm-specialist (eval methodology + fix via prompt/RAG/fine-tune) | Detection is ops; remediation is generation quality |
| "Quantize an LLM" | Route only to quantization-for-inference | Route both: llm-specialist (which format suits the task) + ml-production (AWQ/GPTQ ops, hardware fit) | Format choice and ops are different concerns |

---

## Common Rationalizations (Don't Do These)

| Excuse | Reality |
|--------|---------|
| "User mentioned Kubernetes, route to deployment" | Tools are implementation details. Route by concern first. |
| "Slow = optimization, route to quantization" | Slow could be infrastructure. Clarify model vs serving bottleneck. |
| "They said deploy, must be serving-patterns" | Could need serving + deployment-strategies + monitoring. Don't assume single concern. |
| "MLOps = experiment tracking" | MLOps spans tracking AND automation AND prompt/eval-set versioning. Ask which pain point. |
| "Performance obviously means speed" | Could mean accuracy or LLM output quality. Clarify. |
| "They're technical, skip clarification" | Technical users still benefit from clarifying questions. |
| "LLM question, route only to llm-specialist" | Most LLM production questions need both packs. Default to dual-route. |
| "TorchServe still works, just route there" | TorchServe is in maintenance mode. Flag and offer alternatives. |

---

## Red Flags Checklist

If you catch yourself thinking ANY of these, STOP and clarify:

- "I'll guess optimization vs serving" → ASK which is the bottleneck
- "Performance probably means speed" → ASK speed, accuracy, or LLM output quality
- "Deploy = serving-patterns only" → Consider deployment-strategies and monitoring too
- "They mentioned [tool], route based on tool" → Route by CONCERN, not tool
- "MLOps = one skill" → Could span experiment tracking, automation, and prompt/eval-set versioning
- "Skip question to save time" → Clarifying prevents wrong routing
- "LLM in production = llm-specialist alone" → Default to dual-routing

**When in doubt**: Ask ONE clarifying question. 10 seconds of clarification prevents minutes of wrong-skill loading.

---

## Routing Summary Table

| User Concern | Ask Clarifying | Route To | Also Consider |
|--------------|----------------|----------|---------------|
| Model slow/large | Inference or training? | Optimization skills | If inference, check serving too |
| Deploy classical model | Target (cloud/edge/batch)? | Serving patterns | Deployment strategies for gradual rollout |
| Deploy LLM | Capability tier and target? | LLM-serving Part of serving-patterns + llm-specialist | Observability + cost monitoring |
| Production monitoring | Proactive or reactive? | Monitoring OR debugging | Both if setting up + fixing issues |
| LLM observability | Plumbing or methodology? | Monitoring (plumbing) + llm-specialist (methodology) | Cost + drift alerts |
| MLOps setup | Tracking, automation, or prompt/eval versioning? | Experiment tracking AND/OR automation | Often multiple needed |
| Performance issues | Speed, accuracy, or LLM quality? | Optimization OR observability OR llm-specialist | Depends on clarification |
| Scale serving | Traffic pattern? | Scaling-and-load-balancing | Serving patterns if not set up yet |

---

## Integration Examples

### Example 1: Full Production Pipeline

**Query**: "I trained a model, now I need to put it in production"

**Routing**:
1. Ask: "What's your deployment target and are there performance concerns? Is it an LLM or a classical model?"
2. If "cloud deployment, classical model, fast enough":
   - [model-serving-patterns.md](model-serving-patterns.md) (expose as API)
   - [deployment-strategies.md](deployment-strategies.md) (if gradual rollout needed)
   - [production-monitoring-and-alerting.md](production-monitoring-and-alerting.md) (set up observability)
3. If "edge device, model too large":
   - [quantization-for-inference.md](quantization-for-inference.md) (reduce size first)
   - [model-serving-patterns.md](model-serving-patterns.md) (edge deployment pattern)
   - [production-monitoring-and-alerting.md](production-monitoring-and-alerting.md) (if possible on edge)
4. If "LLM endpoint":
   - llm-specialist `llm-inference-optimization.md` (capability tier + strategy choice)
   - [model-serving-patterns.md](model-serving-patterns.md) (LLM-serving Part: vLLM/SGLang/TensorRT-LLM/TGI/Triton)
   - [production-monitoring-and-alerting.md](production-monitoring-and-alerting.md) (Phoenix/Langfuse + cost monitoring)
   - llm-specialist `llm-evaluation-metrics.md` (methodology to feed observability)

### Example 2: Optimization Decision

**Query**: "My inference is slow"

**Routing**:
1. Ask: "Have you profiled to find the bottleneck — is it the model, serving infrastructure, or KV-cache/batching for an LLM?"
2. If "not profiled yet":
   - [production-debugging-techniques.md](production-debugging-techniques.md) (profile first to diagnose)
   - Then route based on findings
3. If "model is bottleneck":
   - [hardware-optimization-strategies.md](hardware-optimization-strategies.md) (check if hardware tuning helps)
   - If not enough → [quantization-for-inference.md](quantization-for-inference.md) or [model-compression-techniques.md](model-compression-techniques.md)
4. If "infrastructure/batching is bottleneck":
   - [model-serving-patterns.md](model-serving-patterns.md) (batching strategies, LLM-serving Part for continuous batching/PagedAttention/speculative decoding)
   - [scaling-and-load-balancing.md](scaling-and-load-balancing.md) (if traffic-related)

### Example 3: MLOps Maturity

**Query**: "We need better ML workflows"

**Routing**:
1. Ask: "What's the current pain point — can't reproduce experiments, manual deployment, prompt/eval-set drift, or some combination?"
2. If "can't reproduce, need to track experiments":
   - [experiment-tracking-and-versioning.md](experiment-tracking-and-versioning.md)
3. If "manual deployment is slow":
   - [mlops-pipeline-automation.md](mlops-pipeline-automation.md)
4. If "prompts and eval sets keep drifting":
   - [experiment-tracking-and-versioning.md](experiment-tracking-and-versioning.md) (prompt + eval-set versioning patterns)
   - llm-specialist `llm-evaluation-metrics.md` (what to put in those eval sets)
5. If multiple:
   - [experiment-tracking-and-versioning.md](experiment-tracking-and-versioning.md) (establish tracking first)
   - [mlops-pipeline-automation.md](mlops-pipeline-automation.md) (then automate workflow)

### Example 4: LLM in Production with Quality Issues

**Query**: "Our LLM keeps hallucinating in production and costs are spiking"

**Routing**:
1. [production-monitoring-and-alerting.md](production-monitoring-and-alerting.md) — wire Phoenix/Langfuse, capture traces, set cost-budget alerts, surface hallucination/eval-score drift
2. llm-specialist `llm-evaluation-metrics.md` — define hallucination scoring methodology to feed the observability stack
3. llm-specialist `prompt-engineering-patterns.md` and/or `rag-architecture-patterns.md` — remediate the underlying generation-quality issue
4. [production-debugging-techniques.md](production-debugging-techniques.md) — root-cause cost spikes (token bloat? retry storms? bad routing?)

---

## When NOT to Use ml-production Skills

**Skip ml-production when:**
- Still designing/training model → Use neural-architectures, training-optimization
- PyTorch infrastructure issues → Use pytorch-engineering
- LLM generation quality only (prompts, RAG retrieval design, fine-tuning strategy choice, agent design, eval methodology, safety/alignment) → Use llm-specialist
- Classical ML deployment → ml-production still applies; consider gradient boosting / sklearn serving paths inside serving-patterns
- AI/LLM threat modeling → Use ordis-security-architect

**Red flag**: If model isn't trained yet, probably don't need ml-production. Finish training first.

---

## Success Criteria

You've routed correctly when:
- Identified concern category (optimization, serving, MLOps, observability)
- Asked clarifying question for ambiguous queries
- Routed to appropriate skill(s) in logical order
- Didn't let tool choices (Kubernetes, TorchServe, vLLM) dictate routing
- Recognized multi-concern scenarios and routed to multiple skills
- For LLM questions, defaulted to dual-routing across ml-production and llm-specialist unless the question is purely ops or purely generation-quality
- Flagged maintenance-mode tools (e.g., TorchServe) instead of recommending them uncritically
- Pointed at ordis-security-architect when threat modeling, not just observability, was at issue

---

## ML Production Specialist Skills Catalog

After routing, load the appropriate specialist skill for detailed guidance. **This pack contains exactly 10 reference sheets** (no new sheets added in this refresh):

1. [quantization-for-inference.md](quantization-for-inference.md) - `torch.ao.quantization`, AWQ, GPTQ, INT8/INT4 ops, post-training quantization, QAT, calibration, kernel/hardware fit (LLM format *choice* lives in llm-specialist)
2. [model-compression-techniques.md](model-compression-techniques.md) - Pruning (structured/unstructured), knowledge distillation, architecture optimization, model size reduction
3. [hardware-optimization-strategies.md](hardware-optimization-strategies.md) - GPU/CPU/edge tuning, batch sizing, memory optimization, hardware-specific acceleration (TensorRT, ONNX Runtime), capability-tier hardware selection
4. [model-serving-patterns.md](model-serving-patterns.md) - FastAPI, gRPC, ONNX, batching, containerization (Docker), REST/gRPC APIs; **LLM-serving Part: vLLM, SGLang, TensorRT-LLM, TGI, Triton, continuous batching, PagedAttention, speculative decoding, KV-cache management**; TorchServe covered as legacy/maintenance-mode
5. [deployment-strategies.md](deployment-strategies.md) - A/B testing, canary deployment, shadow mode, gradual rollout, rollback procedures, blue-green deployment
6. [scaling-and-load-balancing.md](scaling-and-load-balancing.md) - Horizontal scaling, autoscaling, load balancing, traffic management, cost optimization, replica management
7. [experiment-tracking-and-versioning.md](experiment-tracking-and-versioning.md) - MLflow, Weights & Biases, Comet, Hugging Face Hub, model registries, prompt versioning, RAG / eval-set versioning, lineage, reproducibility
8. [mlops-pipeline-automation.md](mlops-pipeline-automation.md) - CI/CD for ML, feature stores, data validation, automated retraining, orchestration (Airflow, Kubeflow, Prefect, Dagster)
9. [production-monitoring-and-alerting.md](production-monitoring-and-alerting.md) - Metrics tracking, drift detection, dashboards, alerting, SLAs; **LLM observability: Phoenix, Langfuse, OpenTelemetry GenAI semantic conventions**; tool-call telemetry, prompt-injection signal, RAG-retrieval-quality drift, cost/token monitoring
10. [production-debugging-techniques.md](production-debugging-techniques.md) - Error analysis, production profiling, rollback procedures, post-mortems, root cause analysis, tool-call failure forensics

---

## References

- See design doc: `docs/plans/2025-10-30-ml-production-pack-design.md`
- Primary router: `yzmir/ai-engineering-expert/using-ai-engineering`
- Sister packs (bidirectional):
  - `yzmir-llm-specialist/using-llm-specialist` — generation quality, prompt/RAG/agent/eval/safety design (10 sheets including `reasoning-models.md`, `agentic-patterns-and-mcp.md`, `context-engineering-and-prompt-caching.md`)
  - `yzmir-training-optimization/using-training-optimization` — training-phase convergence and speed
  - `yzmir-pytorch-engineering/using-pytorch-engineering` — distributed training, low-level profiling, CUDA/memory
  - `ordis-security-architect/using-security-architect` — AI/LLM threat modeling, controls, security review
