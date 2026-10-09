# Skill Packs Marketplace

48 independently installable specialist packs · 54 skill entrypoints · 530 optional reference sheets

Focused contracts and failure checks for engineering, AI/ML, design, writing and delivery. Skills assume a capable model: routine requests need no routing ceremony, references load selectively, and reviewers report evidence and uncertainty without quotas.

## Installation

In Claude Code, add the marketplace, then install the packs you need:

```text
/plugin marketplace add tachyon-beep/skillpacks
/plugin install axiom-python-engineering
/plugin install yzmir-deep-rl
```

For local development, clone this repository and use `/plugin marketplace add .` from its root. Repository shortcuts under `.claude/commands/` point to the same source contracts; plugin commands live inside each independently installable pack. See [command shortcuts](.claude/SLASH_COMMANDS.md).

## Catalog

Counts describe actual `SKILL.md` entrypoints, not reference sheets. Each pack links to its source and carries its own metadata, optional references, commands and agents. The faction names are organizing themes; see [FACTIONS.md](FACTIONS.md).

| Pack | Skills | Purpose |
|---|---:|---|
| [axiom-audit-pipelines](plugins/axiom-audit-pipelines) | 1 | Use when decisions require verifiable provenance, tamper-evident exports, or defensible retention and replay limits. Ordinary application logging does not require this workflow. |
| [axiom-contract-engineering](plugins/axiom-contract-engineering) | 1 | Use when typed records cross subsystem authority boundaries and absence, meaning changes, blinded views, or derived decisions must remain explicit and reproducible. |
| [axiom-determinism-and-replay](plugins/axiom-determinism-and-replay) | 1 | Use when past stateful execution must be reproduced or investigated: RNG isolation, snapshots, external-effect substitution, scheduling, rollback or first-divergence localization. |
| [axiom-devops-engineering](plugins/axiom-devops-engineering) | 1 | Use when building or fixing delivery and operations: artifact identity, rollout health, rollback/restore evidence, infrastructure and runtime reliability. |
| [axiom-distributed-systems](plugins/axiom-distributed-systems) | 1 | Use when multiple processes or failure domains must remain correct through crashes, partitions, retries, duplicates, clock changes or overload. |
| [axiom-embedded-database](plugins/axiom-embedded-database) | 1 | Use when SQLite or DuckDB application behavior depends on connection configuration, write concurrency, atomic claims, migrations, backup/restore, FTS or query/storage boundaries. |
| [axiom-experiment-formalisation](plugins/axiom-experiment-formalisation) | 1 | Use when experiment records need ontology interoperability, verified EXPO/SUMO/BFO/PROV mappings, executable competency questions or auditable description. Do not activate for ordinary experiment execution. |
| [axiom-mcp-engineering](plugins/axiom-mcp-engineering) | 1 | Use when designing or reviewing MCP server surfaces: model-visible capability contracts, host negotiation, retries, authorization, bounded output and real task evaluation. |
| [axiom-planning](plugins/axiom-planning) | 2 | Use when reviewing an implementation plan for source accuracy, missing dependencies, failure paths or acceptance evidence. |
| [axiom-procedural-architecture](plugins/axiom-procedural-architecture) | 1 | Use when designing or reviewing procedures, runbooks or interactive flows whose stages, decisions and handoffs must be correct for a particular audience. |
| [axiom-product-management](plugins/axiom-product-management) | 1 | Use when deciding product value or maintaining product ownership across sessions: durable state, decision provenance, falsifiable acceptance and authority. |
| [axiom-program-management](plugins/axiom-program-management) | 1 | Use when coordinating delivery across workstreams with real dependency agreements, risks, capacity limits and evidence-based forecasts. |
| [axiom-pyo3-interop](plugins/axiom-pyo3-interop) | 1 | Use when a PyO3 extension needs safe ownership/lifetimes, array exchange, thread attachment, async cancellation, interpreter teardown, wheel compatibility or measured FFI performance. |
| [axiom-python-engineering](plugins/axiom-python-engineering) | 1 | Use for Python-specific compatibility, type/lint repair, cancellation/resource handling, profiling, packaging or Textual lifecycle issues that need focused checks against a real project. |
| [axiom-rust-engineering](plugins/axiom-rust-engineering) | 1 | Use for Rust-specific ownership/API, async cancellation, unsafe/FFI soundness, feature/toolchain compatibility, diagnostics, packaging or profiling problems in a real crate. |
| [axiom-rust-workspaces](plugins/axiom-rust-workspaces) | 1 | Use when multi-crate Cargo composition, feature unification, workspace policy inheritance, internal/public APIs, publishing order or target-specific test subsets cause concrete problems. |
| [axiom-sdlc-engineering](plugins/axiom-sdlc-engineering) | 5 | Use when an organization explicitly selects a software governance profile requiring lifecycle evidence, traceability, risk decisions or measurement. |
| [axiom-solution-architect](plugins/axiom-solution-architect) | 1 | Use when making or reviewing system design choices against measurable requirements, alternatives, migration constraints and source evidence. |
| [axiom-static-analysis-engineering](plugins/axiom-static-analysis-engineering) | 1 | Use when building analyzers or rules whose traversal, abstract domain, unknown-call handling, termination, incremental cache or findings output must have explicit guarantees. |
| [axiom-system-archaeologist](plugins/axiom-system-archaeologist) | 1 | Use when reconstructing an existing codebase: source-linked architecture, dependency and risk findings, coverage limits and incremental refresh. |
| [axiom-tensor-compiler-engineering](plugins/axiom-tensor-compiler-engineering) | 1 | Use when tensor lowering/transforms, torch.fx/compile capture, numerical tolerances, artifact identity or cache behavior need reference-versus-compiled conformance checks. |
| [axiom-web-backend](plugins/axiom-web-backend) | 1 | Use when implementing or reviewing production API behavior: authorization, compatibility, transaction boundaries, retries, bounded results and contract tests. |
| [bravos-game-design](plugins/bravos-game-design) | 1 | Use when inventing, diagnosing, repairing, balancing, or playtesting games across digital, tabletop, live, physical, educational, asynchronous, or hybrid media. |
| [bravos-simulation-tactics](plugins/bravos-simulation-tactics) | 1 | Use when game simulation needs a fidelity/LOD decision, engine implementation, frame-budget evidence or desync diagnosis. |
| [bravos-systems-as-experience](plugins/bravos-systems-as-experience) | 1 | Use when gameplay depends on interacting mechanics, player experimentation, optimization, emergent stories or a modding ecosystem. |
| [lyra-creative-writing](plugins/lyra-creative-writing) | 1 | Use when drafting, revising, critiquing, or planning prose fiction or creative nonfiction, especially when author voice, continuity, genre expectations, or truth claims need deliberate treatment. |
| [lyra-site-designer](plugins/lyra-site-designer) | 1 | Use when building or maintaining static developer-tool, open-source, or documentation sites, including navigation, versioned URLs, code examples, theming, and build or deployment pipelines. |
| [lyra-tui-designer](plugins/lyra-tui-designer) | 1 | Use when building or reviewing interactive terminal applications, especially capability detection, rendering, input/focus, terminal restoration, accessibility, and cross-environment behavior. |
| [lyra-ux-designer](plugins/lyra-ux-designer) | 1 | Use when designing or reviewing application interfaces, audience needs, interaction flows, accessibility, or AI surfaces across web, mobile, desktop, and games. |
| [meta-skillpack-maintenance](plugins/meta-skillpack-maintenance) | 1 | Use when reviewing, simplifying, extending, or validating this marketplace’s plugins, skills, commands, agents, references, hooks, or packaging. |
| [muna-document-designer](plugins/muna-document-designer) | 1 | Use when producing or validating Pandoc/Typst documents, reusable templates, print-ready PDFs, multilingual layouts, or accessible publication artifacts. |
| [muna-panel-review](plugins/muna-panel-review) | 1 | Use for optional staged synthetic reader reviews: precommitted expectations, controlled exposure and evidence-linked hypotheses requiring human validation. |
| [muna-technical-writer](plugins/muna-technical-writer) | 2 | Use when documentation accuracy, executable procedures, institutional register, sensitive information, or source-to-document traceability needs deliberate review. |
| [muna-wiki-management](plugins/muna-wiki-management) | 1 | Use when related documents need explicit source lineage, audience-specific derivatives, claim consistency, reading paths, or controlled change propagation. |
| [ordis-quality-engineering](plugins/ordis-quality-engineering) | 1 | Use when test reliability, isolation, coverage gaps, performance experiments, resilience, or release evidence needs investigation or a concrete verification strategy. |
| [ordis-security-architect](plugins/ordis-security-architect) | 1 | Use when system trust boundaries, threat models, control verification, AI/tool privileges, supply-chain assurance, classified information flows, or authorization evidence need design or review. |
| [yzmir-counterfactual-statistics](plugins/yzmir-counterfactual-statistics) | 1 | Use when paired or branched ML experiments need uncertainty estimates, grouped splits, selection-bias controls, power analysis, or abstention calibration. |
| [yzmir-deep-rl](plugins/yzmir-deep-rl) | 1 | Use when reinforcement-learning environment, reward, data-regime, algorithm or evaluation decisions need implementation checks. |
| [yzmir-dynamic-architectures](plugins/yzmir-dynamic-architectures) | 1 | Use when a neural network grows, prunes, composes modules or adapts across tasks and needs lifecycle, gradient-isolation or state-integrity checks. |
| [yzmir-llm-specialist](plugins/yzmir-llm-specialist) | 1 | Use when an LLM application needs model/configuration evaluation, retrieval, context/cost controls, fine-tuning, tool reliability or safety checks. |
| [yzmir-ml-production](plugins/yzmir-ml-production) | 1 | Use when a model or ML dataset must be released, served, monitored or recovered with operational evidence. |
| [yzmir-morphogenetic-rl](plugins/yzmir-morphogenetic-rl) | 1 | Use when an RL controller decides when or how to change neural-network topology and needs governor, rollback, replay or fair-evaluation checks. |
| [yzmir-neural-architectures](plugins/yzmir-neural-architectures) | 1 | Use when neural architecture families or components must be compared against data, quality, compute, latency or adaptation constraints. |
| [yzmir-pytorch-engineering](plugins/yzmir-pytorch-engineering) | 1 | Use when PyTorch code has tensor/gradient, checkpoint, AMP, compile, distributed, memory or profiling obligations. |
| [yzmir-simulation-foundations](plugins/yzmir-simulation-foundations) | 1 | Use when a simulation needs numerical-model, integration, stability, control, stochastic or determinism checks. |
| [yzmir-structure-synthesis](plugins/yzmir-structure-synthesis) | 1 | Use when a generator emits typed graph/program candidates and needs legality, canonical identity, diversity or search-space checks. |
| [yzmir-systems-thinking](plugins/yzmir-systems-thinking) | 1 | Use when recurring interventions, feedback, delays or accumulation need a causal model and evidence-backed comparison. |
| [yzmir-training-optimization](plugins/yzmir-training-optimization) | 1 | Use when a training run needs evidence about convergence, objective, gradients, regularization, search budget, batch or precision strategy. |

## October 2026 consolidation

Marketplace 4.0.0 reduces 52 packs/61 entrypoints to 48 packs/54 entrypoints. Specialist domains remain; broad tutorials and compulsory orchestration were reduced. This is a content refocus, not a measured claim of improved model performance.

| Retired entrypoint | Retained owner |
|---|---|
| `yzmir-ai-engineering-expert` | [AI/ML specialist catalog](docs/ai-specialist-catalog.md); install the relevant specialist directly |
| `axiom-engineering-foundations` | Solution-architecture debugging/refactoring checks and existing delivery specialists |
| `axiom-system-architect` | Solution-architecture evidence, impact and debt assessment |
| `meta-sme-protocol` | [Shared evidence contract](docs/sme-agent-protocol.md) and inline agent rules |
| SDLC design-and-build, quality-assurance, platform-integration | Retained SDLC governance/requirements; optional solution architecture, quality and DevOps owners |

The [consolidation record](docs/relevance-refresh.md) covers every original pack, migrations and validation limits. Historical review files remain labeled as historical. Retired plugins are removed from this marketplace; uninstall/update any separately cached old installation in your client.

## Maintenance and validation

See [CONTRIBUTING.md](CONTRIBUTING.md) for packaging, evidence and versioning policy. Run the repository integrity checks before committing. Technical examples must be verified against the project's actual toolchain before use. Current standards/compliance claims need named authoritative sources; local checks do not establish certification.

Licensed under [CC BY-SA 4.0](LICENSE), with the [license addendum](LICENSE_ADDENDUM.md). [Code of conduct](CODE_OF_CONDUCT.md).
