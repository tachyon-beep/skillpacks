# Skill relevance review and consolidation — 9 October 2026

## Decision and scope

The source assessment covered all 61 published skill entrypoints in 52 packs at `ca22e854b09ff02cb3b85140b709c3dc22718b6a`, plus an inventory and targeted sampling of 564 reference sheets. It recommended 24 KEEP, 24 REFOCUS, three MERGE and one RETIRE pack. KEEP preserves a distinct capability; it does not certify every recipe. REFOCUS preserves a domain while changing activation/workflow/content.

The resulting catalog contains **48 packs and 54 entrypoints**. All retained routers were rewritten around task, constraints, evidence and selective reference loading. Commands/agents were refocused, duplicated tutorial references reduced, and obsolete runtime dependencies removed. No controlled model ablation or end-to-end external deployment was performed: this establishes source consistency and review disposition, not a measured performance gain or certification of every retained example.

## Every original pack

| Original pack | Decision | Rationale / retained capability |
|---|---|---|
| [axiom-audit-pipelines](../plugins/axiom-audit-pipelines) | KEEP | Decision provenance, canonical bytes, signing/export integrity and replay limits are concrete evidence contracts. Shorten prescribed artifact production. |
| [axiom-contract-engineering](../plugins/axiom-contract-engineering) | KEEP | Absence semantics, versioned meaning, blinded views and independent boundary tests address costly silent failures. Scope strict policies explicitly. |
| [axiom-determinism-and-replay](../plugins/axiom-determinism-and-replay) | KEEP | RNG ownership, snapshot closure, external effects, equivalence predicates and divergence localization form a distinct reproducibility contract. |
| [axiom-devops-engineering](../plugins/axiom-devops-engineering) | KEEP | Deployment identity, observed health, rollback and restore evidence require real operational checks. Validate snippets and simplify routing. |
| [axiom-distributed-systems](../plugins/axiom-distributed-systems) | KEEP | Failure assumptions, per-operation consistency, idempotent effects, leases and retry budgets remain useful. Compress textbook explanations. |
| [axiom-embedded-database](../plugins/axiom-embedded-database) | KEEP | Connection PRAGMAs, write concurrency, atomic claims, WAL and recovery probes capture practical embedded-database failure modes. |
| `axiom-engineering-foundations` | MERGE | Preserve concise debugging/refactoring evidence checks in shared engineering conventions; merge incident/debt/coverage material with its specialist owners. |
| [axiom-experiment-formalisation](../plugins/axiom-experiment-formalisation) | KEEP | Verified ontology vocabulary, competency questions, provenance and annotation boundaries supply niche knowledge. Keep opt-in and preserve its honest no-work exit. |
| [axiom-mcp-engineering](../plugins/axiom-mcp-engineering) | KEEP | Agent-facing tool contracts, atomic retries, structured errors, output budgets and testing remain distinct. Correct overclaims in replay tests and review rules. |
| [axiom-planning](../plugins/axiom-planning) | REFOCUS | Keep implementable handoffs, dependencies and verification criteria. Remove speculative complete implementation code, fixed microsteps and default reviewer fan-out. |
| [axiom-procedural-architecture](../plugins/axiom-procedural-architecture) | REFOCUS | Keep stage invariants, decision readiness, reachability and handoff contracts. Remove required disagreement and clarify control-flow cycles versus dependency DAGs. |
| [axiom-product-management](../plugins/axiom-product-management) | KEEP | Durable product state, authority grants, decision provenance, acceptance and continuity are external context a model cannot infer from intelligence alone. |
| [axiom-program-management](../plugins/axiom-program-management) | REFOCUS | Keep dependency agreements, RAID, honest forecasting and calculations from actual data. Turn generic management instruction into optional reference material. |
| [axiom-pyo3-interop](../plugins/axiom-pyo3-interop) | KEEP | Buffer ownership, Python/Rust lifetimes, cancellation, interpreter teardown and wheel distribution define a valuable integration boundary. Parameterize build modes and timing heuristics. |
| [axiom-python-engineering](../plugins/axiom-python-engineering) | REFOCUS | Keep safe lint/type repair, cancellation/resource handling, profiling and specific tool integration. Remove beginner lessons and routing before every simple Python answer. |
| [axiom-rust-engineering](../plugins/axiom-rust-engineering) | REFOCUS | Keep unsafe/FFI obligations, cancellation, feature/toolchain checks and diagnostic repair. Move elementary ownership/Result/generics teaching to optional documentation. |
| [axiom-rust-workspaces](../plugins/axiom-rust-workspaces) | KEEP | Feature-unification probes, policy inheritance, crate visibility, publishing order and Miri subsets are distinct multi-crate concerns. Remove universal structural prescriptions. |
| [axiom-sdlc-engineering](../plugins/axiom-sdlc-engineering) | REFOCUS | Make this an explicitly selected organizational governance profile. Preserve traceability; remove default maturity mandates, elapsed-time quotas and substantive-comment requirements. |
| [axiom-solution-architect](../plugins/axiom-solution-architect) | REFOCUS | Keep quantified requirements, alternatives, migration/cutover and consistency checks. Size deliverables to risk; a small change should not require ten documents. |
| [axiom-static-analysis-engineering](../plugins/axiom-static-analysis-engineering) | KEEP | Abstract-domain assumptions, termination, conservative unknown calls, stable findings, cache invalidation and suppression lifecycle are a coherent specialist discipline. |
| [axiom-system-archaeologist](../plugins/axiom-system-archaeologist) | REFOCUS | Keep source provenance, bounded coverage, schemas, unknowns and incremental refresh. Remove source-reading bans and fixed five-agent loops per module. |
| `axiom-system-architect` | MERGE | Preserve evidence/impact/debt fields in solution architecture and shared review conventions. Retire the separate posture and critique router. |
| [axiom-tensor-compiler-engineering](../plugins/axiom-tensor-compiler-engineering) | KEEP | Reference/gradient conformance, numerical budgets, artifact identity, manifests and miscompile bisection remain valuable. Scope gradient and bespoke-IR requirements to actual obligations. |
| [axiom-web-backend](../plugins/axiom-web-backend) | REFOCUS | Keep production API contracts, compatibility, authorization and failure-mode checks. Replace framework tutorials and copied scaffolds with current-source recipes. |
| [bravos-game-design](../plugins/bravos-game-design) | KEEP | Causal design, competing explanations, bounded prototypes and human playtesting form a useful workflow. Preserve its distinction between structural checks and enjoyment evidence. |
| [bravos-simulation-tactics](../plugins/bravos-simulation-tactics) | REFOCUS | Keep LOD transitions, fidelity/readability tradeoffs, frame budgets and desync diagnosis. Slim generic algorithms and long engine implementations. |
| [bravos-systems-as-experience](../plugins/bravos-systems-as-experience) | REFOCUS | Keep interaction hypotheses, discovery/feedback, prototype evidence and mod compatibility. Consolidate overlapping emergence/sandbox/game-design exposition. |
| [lyra-creative-writing](../plugins/lyra-creative-writing) | REFOCUS | Preserve author intent, voice, continuity and excerpt-based revision. Remove compulsory mode questions, five-way review and arbitrary genre/reference caps. |
| [lyra-site-designer](../plugins/lyra-site-designer) | REFOCUS | Preserve docs-site URL/version/deprecation policy and tested design recipes. Remove generic CSS instruction and automatic specialist delegation. |
| [lyra-tui-designer](../plugins/lyra-tui-designer) | KEEP | Terminal capabilities, restoration, signals, redraw, focus and accessibility fallbacks are a genuine substrate specialization. Verify partial-initialization cleanup examples. |
| [lyra-ux-designer](../plugins/lyra-ux-designer) | REFOCUS | Keep accessibility validation, user evidence, AI interaction recovery and platform constraints. Compress fundamentals and duplicated routing tables. |
| [meta-skillpack-maintenance](../plugins/meta-skillpack-maintenance) | REFOCUS | Keep repository packaging contracts and skill evaluation. Measure incremental utility and cost, rather than treating compliance with the skill as success. |
| `meta-sme-protocol` | MERGE | Move evidence, uncertainty and coverage expectations into a concise shared contract. Retire the separate long protocol and repeated mandatory prose sections. |
| [muna-document-designer](../plugins/muna-document-designer) | KEEP | Pandoc/Typst production, rendering, print/multilingual constraints and accessibility preflight remain concrete toolchain workflows. Narrow to users selecting that toolchain. |
| [muna-panel-review](../plugins/muna-panel-review) | REFOCUS | Keep staged exposure and precommitted expectations as an optional synthetic experiment. Simplify orchestration and treat audience reactions as hypotheses requiring human validation. |
| [muna-technical-writer](../plugins/muna-technical-writer) | REFOCUS | Keep source tracing, executable documentation, editorial registers and fact-check artifacts. Remove basic writing lectures, line-count ceremonies and flawed verification rules. |
| [muna-wiki-management](../plugins/muna-wiki-management) | KEEP | Derivation graphs, claim registries, change-impact propagation and consistency encode persistent relationships beyond model memory. Keep adoption proportional. |
| [ordis-quality-engineering](../plugins/ordis-quality-engineering) | REFOCUS | Keep failure-driven diagnostics, isolation probes and actual test evidence. Consolidate the testing encyclopedia and remove universal ratios and routing overrides. |
| [ordis-security-architect](../plugins/ordis-security-architect) | KEEP | Trust boundaries, controls, residual risk and human authorization require project facts. Compress discovery and verify standards/version claims against primary sources. |
| `yzmir-ai-engineering-expert` | RETIRE | Retire the standalone routing-only skill. Preserve a short specialist catalog and boundaries in discovery metadata; remove repeated routing examples and mandatory questions. |
| [yzmir-counterfactual-statistics](../plugins/yzmir-counterfactual-statistics) | KEEP | Statistical units, pairing, grouped splits, selection bias and abstention calibration form a distinctive experimental contract. Correct pressure against clean audits. |
| [yzmir-deep-rl](../plugins/yzmir-deep-rl) | KEEP | Environment/reward validation, data regime, replay and multi-seed evaluation are useful failure checks. Move algorithm derivations and long implementations out of defaults. |
| [yzmir-dynamic-architectures](../plugins/yzmir-dynamic-architectures) | KEEP | Gradient isolation, topology-dependent optimizer state, lifecycle transitions and forgetting controls remain useful niche research constraints. |
| [yzmir-llm-specialist](../plugins/yzmir-llm-specialist) | REFOCUS | Keep provider-aware evaluation, retrieval/context measurements and tool boundaries. Consolidate prompt/context tutorials and fix the contradictory tool-message hierarchy. |
| [yzmir-ml-production](../plugins/yzmir-ml-production) | REFOCUS | Keep train/serve skew, model/data/prompt lineage, model rollout gates and serving constraints. Delegate generic backend/cluster deployment material to canonical owners. |
| [yzmir-morphogenetic-rl](../plugins/yzmir-morphogenetic-rl) | KEEP | Controller/governor separation, topology-changing replay, delayed rollback credit and off-switch baselines are distinct. Choose statistics from experimental design rather than a blanket test rule. |
| [yzmir-neural-architectures](../plugins/yzmir-neural-architectures) | REFOCUS | Keep measured architecture comparisons under data/compute constraints. Remove textbook catalogs, brittle thresholds and contradictory fast-generation recommendations. |
| [yzmir-pytorch-engineering](../plugins/yzmir-pytorch-engineering) | KEEP | Checkpoint completeness, AMP ordering, distributed bring-up and profiler evidence are concrete framework checks. Compress tensor/Module/autograd tutorials. |
| [yzmir-simulation-foundations](../plugins/yzmir-simulation-foundations) | REFOCUS | Keep invariant, stability, precision and error-budget checks. Move derivations/course plans to optional teaching material and replace guaranteed outcomes with test obligations. |
| [yzmir-structure-synthesis](../plugins/yzmir-structure-synthesis) | KEEP | Typed grammar, verification/canonicalization ordering, semantic identity and post-canonical diversity provide a specific generator contract. Permit evidenced clean reviews. |
| [yzmir-systems-thinking](../plugins/yzmir-systems-thinking) | REFOCUS | Keep explicit stock/flow assumptions, polarity checks and intervention comparisons. Remove compulsory archetype narratives, course durations and uncalibrated predictions. |
| [yzmir-training-optimization](../plugins/yzmir-training-optimization) | REFOCUS | Keep controlled diagnosis, objective alignment, accumulation/AMP order, budgeted searches and leakage gates. Compress optimizer/loss/dropout tutorials and duplicate tracking material. |

## Migrations and corrections

- Engineering foundations: focused debugging/refactoring and change evidence now live in solution architecture; delivery/recovery techniques use existing specialist owners. The standalone pack and shortcut were removed.
- System architect: evidence, impact, alternatives, technical debt and review coverage now live in solution architecture. The separate posture/router was removed.
- SME protocol: [shared maintainer guidance](sme-agent-protocol.md) and inline evidence/uncertainty/coverage rules replace the standalone dependency and repeated mandatory report sections. A clean review is valid.
- AI router: [the compact specialist catalog](ai-specialist-catalog.md) preserves discovery. Specialists are directly installable; no routing-only pack remains.
- SDLC: design-and-build, quality-assurance and platform-integration leaves were removed after retaining governance, acceptance/change/defect traceability and review evidence. Technical methods belong with solution architecture, quality engineering and DevOps. Governance is explicitly selected by organizational need.
- Python/Rust: broad tutorial sheets became focused compatibility, diagnostic, cancellation, resource, test and integration checks. Unsafe/FFI details remain available on demand.
- Review processes: fixed fan-out, minimum durations, finding/comment quotas and required disagreement were removed from active entrypoints. User authorization and actual evidence consumers govern scope.
- Fact checking: hedged claims still need examination; inability to refute is not verification. Positive support, uncertainty and reproducible claim artifacts govern the result.
- Panel review: staged exposure and precommitted expectations remain optional synthetic experiments. Audience reactions are hypotheses requiring human validation; unavailable isolation/capabilities are reported.
- Technical corrections include MCP transcript-replay limits, queueing assumptions, signature-verification claims, provider/tool instruction hierarchy, statistics selected from study design, build-specific PyO3 behavior, intentional one-member workspaces, and tensor inference/training/layout/device applicability.

## Entrypoint coverage

The table maps every original entrypoint to its disposition and surviving source. Removed entrypoints have no compatibility runtime shim. KEEP/REFOCUS retain the capability with rewritten defaults; MERGE/RETIRE refer to the migrations above.

| Original pack / skill | Disposition | Current source |
|---|---|---|
| `axiom-audit-pipelines` / `using-audit-pipelines` | KEEP | [SKILL.md](../plugins/axiom-audit-pipelines/skills/using-audit-pipelines/SKILL.md) |
| `axiom-contract-engineering` / `using-contract-engineering` | KEEP | [SKILL.md](../plugins/axiom-contract-engineering/skills/using-contract-engineering/SKILL.md) |
| `axiom-determinism-and-replay` / `using-determinism-and-replay` | KEEP | [SKILL.md](../plugins/axiom-determinism-and-replay/skills/using-determinism-and-replay/SKILL.md) |
| `axiom-devops-engineering` / `using-devops-engineering` | KEEP | [SKILL.md](../plugins/axiom-devops-engineering/skills/using-devops-engineering/SKILL.md) |
| `axiom-distributed-systems` / `using-distributed-systems` | KEEP | [SKILL.md](../plugins/axiom-distributed-systems/skills/using-distributed-systems/SKILL.md) |
| `axiom-embedded-database` / `using-embedded-database` | KEEP | [SKILL.md](../plugins/axiom-embedded-database/skills/using-embedded-database/SKILL.md) |
| `axiom-engineering-foundations` / `using-software-engineering` | MERGE | Removed; see migrations |
| `axiom-experiment-formalisation` / `using-experiment-formalisation` | KEEP | [SKILL.md](../plugins/axiom-experiment-formalisation/skills/using-experiment-formalisation/SKILL.md) |
| `axiom-mcp-engineering` / `using-mcp-engineering` | KEEP | [SKILL.md](../plugins/axiom-mcp-engineering/skills/using-mcp-engineering/SKILL.md) |
| `axiom-planning` / `implementation-planning` | REFOCUS | [SKILL.md](../plugins/axiom-planning/skills/implementation-planning/SKILL.md) |
| `axiom-planning` / `plan-review` | REFOCUS | [SKILL.md](../plugins/axiom-planning/skills/plan-review/SKILL.md) |
| `axiom-procedural-architecture` / `using-procedural-architecture` | REFOCUS | [SKILL.md](../plugins/axiom-procedural-architecture/skills/using-procedural-architecture/SKILL.md) |
| `axiom-product-management` / `using-product-management` | KEEP | [SKILL.md](../plugins/axiom-product-management/skills/using-product-management/SKILL.md) |
| `axiom-program-management` / `using-program-management` | REFOCUS | [SKILL.md](../plugins/axiom-program-management/skills/using-program-management/SKILL.md) |
| `axiom-pyo3-interop` / `using-pyo3-interop` | KEEP | [SKILL.md](../plugins/axiom-pyo3-interop/skills/using-pyo3-interop/SKILL.md) |
| `axiom-python-engineering` / `using-python-engineering` | REFOCUS | [SKILL.md](../plugins/axiom-python-engineering/skills/using-python-engineering/SKILL.md) |
| `axiom-rust-engineering` / `using-rust-engineering` | REFOCUS | [SKILL.md](../plugins/axiom-rust-engineering/skills/using-rust-engineering/SKILL.md) |
| `axiom-rust-workspaces` / `using-rust-workspaces` | KEEP | [SKILL.md](../plugins/axiom-rust-workspaces/skills/using-rust-workspaces/SKILL.md) |
| `axiom-sdlc-engineering` / `design-and-build` | MERGE | Removed; see migrations |
| `axiom-sdlc-engineering` / `governance-and-risk` | REFOCUS | [SKILL.md](../plugins/axiom-sdlc-engineering/skills/governance-and-risk/SKILL.md) |
| `axiom-sdlc-engineering` / `lifecycle-adoption` | REFOCUS | [SKILL.md](../plugins/axiom-sdlc-engineering/skills/lifecycle-adoption/SKILL.md) |
| `axiom-sdlc-engineering` / `platform-integration` | MERGE | Removed; see migrations |
| `axiom-sdlc-engineering` / `quality-assurance` | MERGE | Removed; see migrations |
| `axiom-sdlc-engineering` / `quantitative-management` | REFOCUS | [SKILL.md](../plugins/axiom-sdlc-engineering/skills/quantitative-management/SKILL.md) |
| `axiom-sdlc-engineering` / `requirements-lifecycle` | REFOCUS | [SKILL.md](../plugins/axiom-sdlc-engineering/skills/requirements-lifecycle/SKILL.md) |
| `axiom-sdlc-engineering` / `using-sdlc-engineering` | REFOCUS | [SKILL.md](../plugins/axiom-sdlc-engineering/skills/using-sdlc-engineering/SKILL.md) |
| `axiom-solution-architect` / `using-solution-architect` | REFOCUS | [SKILL.md](../plugins/axiom-solution-architect/skills/using-solution-architect/SKILL.md) |
| `axiom-static-analysis-engineering` / `using-static-analysis-engineering` | KEEP | [SKILL.md](../plugins/axiom-static-analysis-engineering/skills/using-static-analysis-engineering/SKILL.md) |
| `axiom-system-archaeologist` / `using-system-archaeologist` | REFOCUS | [SKILL.md](../plugins/axiom-system-archaeologist/skills/using-system-archaeologist/SKILL.md) |
| `axiom-system-architect` / `using-system-architect` | MERGE | Removed; see migrations |
| `axiom-tensor-compiler-engineering` / `using-tensor-compiler-engineering` | KEEP | [SKILL.md](../plugins/axiom-tensor-compiler-engineering/skills/using-tensor-compiler-engineering/SKILL.md) |
| `axiom-web-backend` / `using-web-backend` | REFOCUS | [SKILL.md](../plugins/axiom-web-backend/skills/using-web-backend/SKILL.md) |
| `bravos-game-design` / `using-game-design` | KEEP | [SKILL.md](../plugins/bravos-game-design/skills/using-game-design/SKILL.md) |
| `bravos-simulation-tactics` / `using-simulation-tactics` | REFOCUS | [SKILL.md](../plugins/bravos-simulation-tactics/skills/using-simulation-tactics/SKILL.md) |
| `bravos-systems-as-experience` / `using-systems-as-experience` | REFOCUS | [SKILL.md](../plugins/bravos-systems-as-experience/skills/using-systems-as-experience/SKILL.md) |
| `lyra-creative-writing` / `using-creative-writing` | REFOCUS | [SKILL.md](../plugins/lyra-creative-writing/skills/using-creative-writing/SKILL.md) |
| `lyra-site-designer` / `using-site-designer` | REFOCUS | [SKILL.md](../plugins/lyra-site-designer/skills/using-site-designer/SKILL.md) |
| `lyra-tui-designer` / `using-tui-designer` | KEEP | [SKILL.md](../plugins/lyra-tui-designer/skills/using-tui-designer/SKILL.md) |
| `lyra-ux-designer` / `using-ux-designer` | REFOCUS | [SKILL.md](../plugins/lyra-ux-designer/skills/using-ux-designer/SKILL.md) |
| `meta-skillpack-maintenance` / `using-skillpack-maintenance` | REFOCUS | [SKILL.md](../plugins/meta-skillpack-maintenance/skills/using-skillpack-maintenance/SKILL.md) |
| `meta-sme-protocol` / `sme-agent-protocol` | MERGE | Removed; see migrations |
| `muna-document-designer` / `using-document-designer` | KEEP | [SKILL.md](../plugins/muna-document-designer/skills/using-document-designer/SKILL.md) |
| `muna-panel-review` / `reader-panel-review` | REFOCUS | [SKILL.md](../plugins/muna-panel-review/skills/reader-panel-review/SKILL.md) |
| `muna-technical-writer` / `fact-checking` | REFOCUS | [SKILL.md](../plugins/muna-technical-writer/skills/fact-checking/SKILL.md) |
| `muna-technical-writer` / `using-technical-writer` | REFOCUS | [SKILL.md](../plugins/muna-technical-writer/skills/using-technical-writer/SKILL.md) |
| `muna-wiki-management` / `using-wiki-manager` | KEEP | [SKILL.md](../plugins/muna-wiki-management/skills/using-wiki-manager/SKILL.md) |
| `ordis-quality-engineering` / `using-quality-engineering` | REFOCUS | [SKILL.md](../plugins/ordis-quality-engineering/skills/using-quality-engineering/SKILL.md) |
| `ordis-security-architect` / `using-security-architect` | KEEP | [SKILL.md](../plugins/ordis-security-architect/skills/using-security-architect/SKILL.md) |
| `yzmir-ai-engineering-expert` / `using-ai-engineering` | RETIRE | Removed; see migrations |
| `yzmir-counterfactual-statistics` / `using-counterfactual-statistics` | KEEP | [SKILL.md](../plugins/yzmir-counterfactual-statistics/skills/using-counterfactual-statistics/SKILL.md) |
| `yzmir-deep-rl` / `using-deep-rl` | KEEP | [SKILL.md](../plugins/yzmir-deep-rl/skills/using-deep-rl/SKILL.md) |
| `yzmir-dynamic-architectures` / `using-dynamic-architectures` | KEEP | [SKILL.md](../plugins/yzmir-dynamic-architectures/skills/using-dynamic-architectures/SKILL.md) |
| `yzmir-llm-specialist` / `using-llm-specialist` | REFOCUS | [SKILL.md](../plugins/yzmir-llm-specialist/skills/using-llm-specialist/SKILL.md) |
| `yzmir-ml-production` / `using-ml-production` | REFOCUS | [SKILL.md](../plugins/yzmir-ml-production/skills/using-ml-production/SKILL.md) |
| `yzmir-morphogenetic-rl` / `using-morphogenetic-rl` | KEEP | [SKILL.md](../plugins/yzmir-morphogenetic-rl/skills/using-morphogenetic-rl/SKILL.md) |
| `yzmir-neural-architectures` / `using-neural-architectures` | REFOCUS | [SKILL.md](../plugins/yzmir-neural-architectures/skills/using-neural-architectures/SKILL.md) |
| `yzmir-pytorch-engineering` / `using-pytorch-engineering` | KEEP | [SKILL.md](../plugins/yzmir-pytorch-engineering/skills/using-pytorch-engineering/SKILL.md) |
| `yzmir-simulation-foundations` / `using-simulation-foundations` | REFOCUS | [SKILL.md](../plugins/yzmir-simulation-foundations/skills/using-simulation-foundations/SKILL.md) |
| `yzmir-structure-synthesis` / `using-structure-synthesis` | KEEP | [SKILL.md](../plugins/yzmir-structure-synthesis/skills/using-structure-synthesis/SKILL.md) |
| `yzmir-systems-thinking` / `using-systems-thinking` | REFOCUS | [SKILL.md](../plugins/yzmir-systems-thinking/skills/using-systems-thinking/SKILL.md) |
| `yzmir-training-optimization` / `using-training-optimization` | REFOCUS | [SKILL.md](../plugins/yzmir-training-optimization/skills/using-training-optimization/SKILL.md) |

## Source size and publication inventory

| Measure | Before | After |
|---|---:|---:|
| Packs | 52 | 48 |
| Skill entrypoints | 61 | 54 |
| Optional reference sheets | 564 | 530 |
| Entrypoint lines | 20,760 | 2,631 |
| Skill/reference words | 1,979,516 | 1,169,488 |

The current publication also has 141 plugin commands and 113 agents. Counts exclude intentional maintenance fixtures, operational skills and an unrelated ignored local program-management command. Source size is a mechanical inventory, not a measure of quality or model performance.

## Independent review and corrections

Three independent cross-reviews fully read the 40 retained entrypoints outside their own edit cohorts; the coordinating reviewer read the remaining 14 ML/simulation entrypoints. Reviews also checked selected references, commands/agents, old-versus-new source and central packaging. Deep references were sampled, not exhaustively revalidated.

The reviews found and resolved retained-reference defects in free-threaded detachment, NumPy lifetime versus aliasing safety, tensor source/target identity and oracle independence, product-state commit authorization, workflow-net soundness, and fact-check output schema. Reviewers rechecked the material fixes. Follow-up corrections removed unsupported workspace quotas and scoped tensor numerical/layout exactness. No unresolved material finding remains within that review coverage.

## Validation performed

- `python3 scripts/check_marketplace_integrity.py`: 48 catalog entries match 48 packs; zero errors and warnings for active pack/command references, current review coverage and README discovery.
- `python3 scripts/check_skillpack_contracts.py`: all 54 skill entrypoints, 141 plugin commands, 113 agents and repository wrappers have valid required frontmatter; current entrypoint/catalog links resolve; no retired runtime dependencies. Counts are 48/54/530/141/113.
- Parsed frontmatter in all published skill/reference Markdown where present. Checked local links outside fenced examples in changed plugin Markdown; no missing candidate targets. Scoped reviewer checks also covered their reference links and fence closure.
- Executed the MCP replay example: dynamic lease capture/substitution succeeds; changed scalar, extra object field, bool/int mismatch and reordered lists fail.
- Executed the optional forecast calculator: deterministic known completion, zero remaining scope, invalid/empty/all-zero data, censored trials and seeded reproducibility pass.
- Exercised the new contract checker against an isolated valid publication fixture and injected missing links, malformed YAML, retired dependency and source mismatch; the valid fixture passes and each defect is rejected.
- Both catalog/contract checkers also pass against an isolated snapshot of the staged publication, without ignored checkout-local files.
- `git diff --check` passes. The two pre-existing dirty Filigree reference files retain their original SHA-256 values and are excluded from the commit.

The executable checks above were local, deterministic checks. No controlled no-skill/current/reduced model comparison, training run, wheel-build matrix, deployment, hosted CI or exhaustive retained-snippet execution was performed. Version-sensitive examples require task-time verification; links to primary guidance support specific corrections rather than certifying whole packs.

Marketplace 4.0.0 and retained plugin major bumps make removed entrypoints and changed default contracts explicit. Historical per-pack reports remain labeled as historical. Previous tutorial/process material is recoverable in Git history.
