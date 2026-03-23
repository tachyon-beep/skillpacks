# Claude Code Skill Packs Marketplace

## Professional AI/ML, game development, security, documentation, and UX skills for Claude Code

26 complete skillpacks • 173+ skills • Install what you need

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

**ordis-security-architect** - 9 skills

- Threat modeling (STRIDE), security controls, compliance frameworks
- ATO processes, classified systems, security architecture review
- `/plugin install ordis-security-architect`

### 📝 Documentation (Muna)

**muna-technical-writer** - 9 skills

- Documentation structure, clarity & style, diagram conventions
- Security-aware docs, incident response, ITIL/governance
- `/plugin install muna-technical-writer`

### 🔬 Development (Axiom) - 7 Packs

**axiom-python-engineering** - 10 skills

- Modern Python 3.12+: types, syntax, project structure, delinting
- Testing, async, debugging, profiling
- Scientific computing (NumPy/pandas), ML workflows
- `/plugin install axiom-python-engineering`

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

### 🤖 AI/ML Engineering (Yzmir) - 10 Packs

**yzmir-ai-engineering-expert** - 1 router skill

- Primary router that directs to specialized AI/ML packs
- `/plugin install yzmir-ai-engineering-expert`

**yzmir-pytorch-engineering** - 9 skills

- Tensors, modules, distributed training, profiling, debugging
- `/plugin install yzmir-pytorch-engineering`

**yzmir-training-optimization** - 11 skills

- Optimizers, learning rates, convergence, hyperparameter tuning
- `/plugin install yzmir-training-optimization`

**yzmir-deep-rl** - 13 skills

- DQN, PPO, SAC, reward shaping, exploration, offline RL
- `/plugin install yzmir-deep-rl`

**yzmir-neural-architectures** - 9 skills

- CNNs, Transformers, RNNs, attention mechanisms, architecture selection
- `/plugin install yzmir-neural-architectures`

**yzmir-llm-specialist** - 8 skills

- Fine-tuning, RLHF, RAG, inference optimization, prompt engineering
- `/plugin install yzmir-llm-specialist`

**yzmir-ml-production** - 11 skills

- Quantization, model serving, MLOps, monitoring, debugging
- `/plugin install yzmir-ml-production`

**yzmir-simulation-foundations** - 9 skills

- Differential equations, stability analysis, control theory (game math)
- `/plugin install yzmir-simulation-foundations`

**yzmir-systems-thinking** - 6 skills

- Systems thinking methodology, patterns, leverage points
- Archetypes, modeling, visualization
- `/plugin install yzmir-systems-thinking`

**yzmir-dynamic-architectures** - 7 skills

- Dynamic/morphogenetic neural networks
- Continual learning, gradient isolation, PEFT/LoRA
- `/plugin install yzmir-dynamic-architectures`

### 🎮 Game Development (Bravos) - 2 Packs

**bravos-simulation-tactics** - 11 skills

- Physics simulation, ecosystem simulation, crowd simulation
- `/plugin install bravos-simulation-tactics`

**bravos-systems-as-experience** - 9 skills

- Emergent gameplay, player-driven narratives, strategic depth
- `/plugin install bravos-systems-as-experience`

### 🎨 UX Design (Lyra)

**lyra-ux-designer** - 11 skills

- Visual design, accessibility (WCAG), interaction patterns
- Mobile/web/desktop/game UI design
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

**meta-skillpack-maintenance** - 1 skill

- Systematic maintenance and enhancement of skill packs
- RED-GREEN-REFACTOR testing methodology
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

- **160+ Skills**: Production-ready guidance across 6 domains
- **23 Packs**: Install only what you need
- **Per-Pack Plugins**: Independent versioning, clean dependencies
- **Complete Coverage**: Design → Foundation → Training → Production

---

## Repository Structure

```
skillpacks/
├── .claude-plugin/
│   └── marketplace.json       # Marketplace catalog
├── plugins/                   # All 23 skillpacks
│   ├── axiom-*/               # Development (5 packs)
│   ├── bravos-*/              # Game development (2 packs)
│   ├── lyra-*/                # UX design (1 pack)
│   ├── meta-*/                # Meta utilities (2 packs)
│   ├── muna-*/                # Documentation (1 pack)
│   ├── ordis-*/               # Quality & security (2 packs)
│   └── yzmir-*/               # AI/ML (10 packs)
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
