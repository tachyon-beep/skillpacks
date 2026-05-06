# Claude Code Skill Packs Marketplace

## Professional AI/ML, Python & Rust engineering, web backend, DevOps, SDLC, solution architecture, game development, security, documentation, and UX skills for Claude Code

38 complete skillpacks • 200+ skills • Install what you need

> **Recent additions (May 2026)**: 6 new packs — `yzmir-morphogenetic-rl`,
> `axiom-determinism-and-replay`, `axiom-audit-pipelines`,
> `axiom-static-analysis-engineering`, `axiom-rust-workspaces`,
> `axiom-pyo3-interop`. Plus a 2026-era refresh sweep across the AI/ML cluster
> (PyTorch 2.9+, FSDP2, vLLM/SGLang, GRPO/DreamerV3, Mamba/SSM, modern PEFT) and
> the design/security/quality packs (WCAG 2.2, Material 3, Pydantic v2,
> Typst 0.14, SLSA/SBOM/Sigstore).

---

## Installation

### Via Marketplace (Recommended)

```bash
# Add the skillpacks marketplace
/plugin marketplace add tachyon-beep/skillpacks

# Browse available plugins
/plugin

# Install specific packs
/plugin install yzmir-deep-rl
/plugin install lyra-ux-designer
/plugin install ordis-security-architect
```

### Via Git Clone (Development)

```bash
git clone https://github.com/tachyon-beep/skillpacks
cd skillpacks
/plugin marketplace add .
```

---

## Available Skillpacks

*Skillpack groupings are inspired by the factions from [Altered TCG](https://www.altered.gg). See [FACTIONS.md](FACTIONS.md) for the philosophical connections between factions and skillpack domains.*

### 🔒 Security (Ordis)

**ordis-security-architect** - 11 skills, 3 commands, 2 agents _(refreshed for 2026)_

- Threat modeling (STRIDE / CWE / CVSS / ATT&CK), security controls
- Compliance: NIST CSF 2.0, ISO 27001:2022, PCI-DSS v4.0.1, GDPR, NIS2, EU AI Act
- LLM/AI security (OWASP LLM Top 10:2025, MITRE ATLAS), supply-chain
  (SLSA, SBOM, Sigstore), ATO/RMF
- `/plugin install ordis-security-architect`

### 📝 Documentation (Muna) - 4 Packs

**muna-document-designer** - 1 skill, 1 agent, 1 command

- Professional document design with Pandoc and Typst
- Typography, layout, templates, branded document systems
- Reports, proposals, specifications, resumes, brochures
- `/plugin install muna-document-designer`

**muna-technical-writer** - 10 skills, 4 commands, 3 agents

- Documentation structure, clarity & style, diagram conventions
- Security-aware docs, incident response, ITIL/governance
- Includes `complex-writer` / `complex-reviewer` agent pair for surgical edits
  in large files (≥2000 lines), cross-language
- `/plugin install muna-technical-writer`

**muna-panel-review** - 3 agents, 1 skill, 3 commands

- Simulated audience panel review with persona-driven editorial intelligence
- Spawn reader personas, collect mood journals, synthesize cross-panel feedback
- `/plugin install muna-panel-review`

**muna-wiki-management** - 7 skills, 4 commands, 2 agents

- Document set management as wikis with architecture and governance
- Derivation tracking, consistency auditing, evolution management
- `/plugin install muna-wiki-management`

### 🔬 Development (Axiom) - 15 Packs

**axiom-python-engineering** - 10 skills

- Modern Python 3.13: uv-first tooling, ty/pyrefly, current pre-commit revs
- Testing, async, debugging, profiling, delinting
- Scientific computing (NumPy/pandas), ML workflows
- `/plugin install axiom-python-engineering`

**axiom-rust-engineering** - router + 11 reference sheets, 5 commands, 3 agents

- Rust 2024 edition: ownership, traits, async (tokio), testing
- Clippy/cargo tooling, performance, unsafe/FFI, AI/ML interop (PyO3, candle)
- Single-crate-shaped — composes with `axiom-rust-workspaces` and `axiom-pyo3-interop`
- `/plugin install axiom-rust-engineering`

**axiom-rust-workspaces** - router + 13 sheets, 3 commands, 1 agent _(new — May 2026)_

- Rust at workspace scope: multi-crate composition, `[workspace.dependencies]`,
  `[workspace.lints]`, `deny.toml` with waiver lifecycle
- Feature unification (resolver-2/3), Miri-on-subset, public-vs-internal crate
  boundaries (internal-traits-crate / sealed-trait), 10-pattern anti-pattern list
- `/plugin install axiom-rust-workspaces`

**axiom-pyo3-interop** - router + 13 sheets, 3 commands, 1 agent _(new — May 2026)_

- Python ↔ Rust FFI discipline: `Bound<'py, T>`, GIL release, abi3, maturin
- Batched FFI to amortise crossing cost, zero-copy NumPy buffer protocol,
  Gymnasium environments backed by Rust, async across the boundary
- Wheel matrix (cibuildwheel, manylinux), interpreter-teardown discipline
- `/plugin install axiom-pyo3-interop`

**axiom-system-archaeologist** - 5 skills

- Deep codebase architecture analysis through subagent coordination
- C4 diagrams (Context, Container, Component levels)
- Subsystem catalog generation with validation gates
- Stakeholder-ready architecture documentation
- `/plugin install axiom-system-archaeologist`

**axiom-system-architect** - 4 skills

- TDD-validated architectural assessment with professional discipline
- Prevents diplomatic softening, analysis paralysis, security compromise
- Router + 3 specialist skills for technical debt and architecture review
- `/plugin install axiom-system-architect`

**axiom-solution-architect** - 9 skills, 2 agents, 2 commands

- Forward solution architecture: brief/HLD/epic → traceable artifact set
- ADRs, C4, NFRs, RTM, integration/migration, risks, TOGAF/ArchiMate
- Consolidated Solution Architecture Document (SAD) with consistency gate
- `/plugin install axiom-solution-architect`

**axiom-web-backend** - 12 skills

- FastAPI, Django, Express.js development patterns
- REST/GraphQL API design, microservices architecture
- Authentication, database integration, message queues
- API testing, documentation, production deployment
- `/plugin install axiom-web-backend`

**axiom-devops-engineering** - 1 skill

- CI/CD pipeline architecture, deployment strategies
- Zero-downtime deployments, infrastructure reliability
- `/plugin install axiom-devops-engineering`

**axiom-sdlc-engineering** - 8 skills + 4 agents

- CMMI-based SDLC framework (Levels 2-4) with GitHub/Azure DevOps integration
- Requirements lifecycle, design/build, quality assurance, governance & risk
- Platform integration, quantitative management, lifecycle adoption
- Specialist agents for architecture decisions, bug triage, QA, and process routing
- `/plugin install axiom-sdlc-engineering`

**axiom-planning** - 2 skills, 5 agents, 1 command

- TDD-validated implementation planning with plan review quality gate
- Plan review spawns 4 parallel reviewers (reality, architecture, quality, systems) + synthesizer
- Catches hallucinations, convention violations, and risks before code execution
- `/plugin install axiom-planning`

**axiom-engineering-foundations** - 6 skills

- Universal software engineering methodology (language-agnostic)
- Systematic debugging, safe refactoring, code review, incident response
- Technical debt triage, codebase confidence building
- `/plugin install axiom-engineering-foundations`

**axiom-audit-pipelines** - router + 11 sheets, 3 commands, 2 agents _(new — May 2026)_

- Audit-grade decision pipelines: canonical encoding (RFC 8785 JCS), append-only
  decision logs, fingerprint chains (linked-hash and Merkle)
- HMAC and Ed25519 signed exports, immutable storage, decision provenance
  (inputs/ruleset/code closure), threat model for the log itself
- Retention reconciled with right-to-be-forgotten, partial replay, performance budgets
- `/plugin install axiom-audit-pipelines`

**axiom-determinism-and-replay** - router + 13 sheets, 3 commands, 2 agents _(new — May 2026)_

- Architecture-level determinism and replay for RL substrates, multi-agent
  systems, multiplayer lockstep engines, replay-debuggable services
- Seed governance, RNG isolation, snapshot strategy, divergence detection,
  rollback, concurrency strategies, floating-point and GPU determinism
- External-effects substitution, canonical state encoding, property tests,
  cost-of-determinism accounting
- `/plugin install axiom-determinism-and-replay`

**axiom-static-analysis-engineering** - router + 13 sheets, 3 commands, 2 agents _(new — May 2026, v0.2)_

- Building static analyzers as engines, not running them as users
- AST visitation, taint-lattice (abstract-domain) design with monotonicity,
  three-phase inference (variable → function → callgraph) with termination proofs
- Plugin architecture for rules, false-positive economics with auditable
  suppression lifecycle, the static-vs-runtime boundary
- Callgraph construction (Rung 0–4 resolution, dynamic-feature handling),
  cross-module flow with stub libraries, decorator-as-assertion (runtime + static
  agreement contract), manifest-driven configuration with coherence validation,
  SARIF / CI integration, incremental analysis at scale, LLM-assisted finding
  explanations as a substitutable pattern
- `/plugin install axiom-static-analysis-engineering`

### 🤖 AI/ML Engineering (Yzmir) - 11 Packs

**yzmir-ai-engineering-expert** - 1 router skill

- Primary router that directs to specialized AI/ML packs
- `/plugin install yzmir-ai-engineering-expert`

**yzmir-pytorch-engineering** - 9 skills _(refreshed for PyTorch 2.9+)_

- Tensors, modules, distributed training, profiling, debugging
- torch.compile, FSDP1/FSDP2, FlexAttention, torch.amp
- `/plugin install yzmir-pytorch-engineering`

**yzmir-training-optimization** - 11 skills _(refreshed)_

- Modern optimizers, schedules, mixed-precision, batch-size strategy
- Learning rates, convergence, hyperparameter tuning
- `/plugin install yzmir-training-optimization`

**yzmir-deep-rl** - 13 skills _(refreshed)_

- DQN, PPO, SAC, reward shaping, exploration
- Modern offline RL, GRPO, DreamerV3 / TD-MPC2, REDQ/DroQ/CrossQ,
  MAPPO/IPPO, BBF/Agent57, Go-Explore
- `/plugin install yzmir-deep-rl`

**yzmir-morphogenetic-rl** - router + 7 sheets, 2 commands, 2 agents _(new — May 2026)_

- RL controllers that decide WHEN/HOW to mutate a network's topology during training
- Controller action/observation/reward design, governor and safety gates
- Rollback-as-RL-signal, deterministic morphogenesis, ablation under topology change
- Companion to `yzmir-dynamic-architectures`
- `/plugin install yzmir-morphogenetic-rl`

**yzmir-neural-architectures** - 9 skills _(refreshed)_

- CNNs, Transformers, RNNs, attention mechanisms, architecture selection
- Mamba/SSM, MoE (Mixtral/DeepSeek/OLMoE), modern diffusion (SDXL/FLUX/DiT/SD3),
  multimodal (CLIP/SigLIP/LLaVA), SAM/SAM-2, ConvNeXt v2, equivariant GNNs
- `/plugin install yzmir-neural-architectures`

**yzmir-llm-specialist** - 8 skills _(refreshed)_

- Reasoning models, agentic + MCP, modern serving stack, DPO/GRPO
- RAG with contextual retrieval, fine-tuning, inference optimization
- `/plugin install yzmir-llm-specialist`

**yzmir-ml-production** - 11 skills _(refreshed)_

- torch.ao.quantization, vLLM/SGLang/TensorRT-LLM serving
- LLM observability, MLOps, monitoring, debugging
- `/plugin install yzmir-ml-production`

**yzmir-simulation-foundations** - 9 skills

- Differential equations, stability analysis, control theory (game math)
- `/plugin install yzmir-simulation-foundations`

**yzmir-systems-thinking** - 6 skills

- Systems thinking methodology, patterns, leverage points
- Archetypes, modeling, visualization
- `/plugin install yzmir-systems-thinking`

**yzmir-dynamic-architectures** - 6 skills, 1 agent, 2 commands _(refreshed)_

- Dynamic/morphogenetic neural networks: grow, prune, adapt topology
- Continual learning, gradient isolation, lifecycle orchestration
- Modern PEFT (VeRA/PiSSA/LoftQ/LoRA+/rsLoRA/LongLoRA), post-Mixtral MoE,
  adapter merging (TIES/DARE/SLERP/MergeKit)
- `/plugin install yzmir-dynamic-architectures`

### 🎮 Game Development (Bravos) - 2 Packs

**bravos-simulation-tactics** - 11 skills

- Physics simulation, ecosystem simulation, crowd simulation
- `/plugin install bravos-simulation-tactics`

**bravos-systems-as-experience** - 9 skills

- Emergent gameplay, player-driven narratives, strategic depth
- `/plugin install bravos-systems-as-experience`

### 🎨 UX & Site Design (Lyra) - 2 Packs

**lyra-site-designer** - 1 skill, 1 agent, 1 command, 6 reference sheets

- Static site design for developer tools and documentation sites
- Information architecture, HTML/CSS craftsmanship, design tokens
- Developer UX patterns: code blocks, dark mode, search, responsive layouts
- Static site tooling guidance (Hugo, Astro, Eleventy)
- `/plugin install lyra-site-designer`

**lyra-ux-designer** - 12 reference sheets _(refreshed for 2026)_

- WCAG 2.2 accessibility, modern web platform primitives (dialog, popover,
  `:has()`, container queries)
- iOS 17+ and Material 3 mobile patterns, AI-experience patterns
- `/plugin install lyra-ux-designer`

### ✅ Quality Engineering (Ordis)

**ordis-quality-engineering** - 21 skills

- E2E testing, API testing, integration testing, performance testing
- Chaos engineering, flaky test diagnosis, mutation testing
- Observability, test automation architecture, coverage gap analysis
- `/plugin install ordis-quality-engineering`

### 🛠️ Meta Packs

**meta-sme-protocol** - 1 skill

- SME (Subject Matter Expert) Agent Protocol
- Mandatory protocol for all specialist agents: fact-finding, confidence/risk assessment
- `/plugin install meta-sme-protocol`

**meta-skillpack-maintenance** - 3 skills

- Systematic maintenance and enhancement of skill packs
- Investigative domain analysis, RED-GREEN-REFACTOR testing
- Automated quality improvements
- `/plugin install meta-skillpack-maintenance`

---

## Quick Examples

### AI/ML Engineering

```bash
/plugin install yzmir-deep-rl
/deep-rl  # Use slash command to load router skill
```

Claude will guide you to the right specialized skill:

```plaintext
I'm using yzmir/deep-rl/policy-gradient-methods to implement PPO for continuous character control
```

### Web Backend Development

```bash
/plugin install axiom-web-backend
/web-backend  # Use slash command to load router
```

```plaintext
I'm using axiom/web-backend/fastapi-development to build production-ready async APIs with dependency injection
```

### Game Development

```bash
/plugin install bravos-systems-as-experience
/systems-as-experience  # Use slash command to load router
```

```plaintext
I'm using bravos/systems-as-experience/emergent-gameplay-design to create systemic interactions in my RPG
```

### Security Architecture

```bash
/plugin install ordis-security-architect
/security-architect  # Use slash command to load router
```

```
I'm using ordis/security-architect/threat-modeling to analyze this authentication system with STRIDE
```

### UX Design

```bash
/plugin install lyra-ux-designer
/ux-designer  # Use slash command to load router
```

```
I'm using lyra/ux-designer/accessibility-and-inclusive-design to ensure WCAG 2.1 AA compliance
```

### Architecture Analysis

```bash
/plugin install axiom-system-archaeologist
/system-archaeologist  # Use slash command to load router
```

```plaintext
Then Claude routes you to specialized analysis skills for C4 diagrams, subsystem catalogs, etc.
```

---

## Using Router Skills: Slash Commands vs Direct Invocation

### Slash Commands (Recommended)

All router skills (`using-X` skills) are available as **slash commands** to avoid skill context limits:

```bash
/ai-engineering          # Route to AI/ML specialized packs
/deep-rl                # Route to RL algorithm skills
/system-archaeologist   # Route to architecture analysis
/python-engineering     # Route to Python skills
/web-backend           # Route to web backend skills
/ux-designer           # Route to UX design skills
# ... and 12 more
```

**Why slash commands?** Router skills are comprehensive guides that exceeded the context budget for automatic skill discovery. Slash commands provide explicit, user-controlled invocation without context limits.

See [`.claude/SLASH_COMMANDS.md`](.claude/SLASH_COMMANDS.md) for the complete list of all router commands.

### Direct Skill Invocation

If you have **only a few plugins installed**, direct skill invocation still works:

```plaintext
I'm using yzmir/deep-rl/policy-gradient-methods to implement PPO
```

**Caveat**: With many plugins installed (5+), you may hit skill discovery context limits. In this case, use slash commands for routers and direct invocation for specialized skills.

**Best Practice**:

- Use **slash commands** for router skills (`/ai-engineering`, `/deep-rl`, etc.)
- Use **direct invocation** for specialized skills after the router guides you

---

## What's Inside

- **200+ Skills**: Production-ready guidance across 7 domains
- **38 Packs**: Install only what you need
- **Per-Pack Plugins**: Independent versioning, clean dependencies
- **Complete Coverage**: Design → Foundation → Training → Production

---

## Repository Structure

```
skillpacks/
├── .claude-plugin/
│   └── marketplace.json       # Marketplace catalog
├── plugins/                   # All 38 skillpacks
│   ├── axiom-*/               # Development (15 packs)
│   ├── bravos-*/              # Game development (2 packs)
│   ├── lyra-*/                # UX & site design (2 packs)
│   ├── meta-*/                # Meta utilities (2 packs)
│   ├── muna-*/                # Documentation (4 packs)
│   ├── ordis-*/               # Quality & security (2 packs)
│   └── yzmir-*/               # AI/ML (11 packs)
└── docs/                      # Planning documents
```

---

## Contributing

See [CONTRIBUTING.md](source/CONTRIBUTING.md) for guidelines on adding skills or creating new packs.

---

## License

CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International) - See [LICENSE](LICENSE) for details.

**Note**: Faction names (Axiom, Bravos, Lyra, Muna, Ordis, Yzmir) from Altered TCG are NOT covered by this license - see [LICENSE_ADDENDUM.md](LICENSE_ADDENDUM.md).

---

## About

Built by [@tachyon-beep](https://github.com/tachyon-beep)

**Skillpacks** provides modular, production-ready expertise for Claude Code across professional domains. Each skillpack is independently versioned and installable, containing systematically validated skills that guide Claude through expert-level implementations.
