# Claude Code Skill Packs Marketplace

## Professional AI/ML, game development, security, documentation, and UX skills for Claude Code**

13 complete skillpacks • 120+ production-ready skills • Install what you need

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

### 🤖 AI/ML Engineering (Yzmir) - 7 Packs

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

---

## Quick Examples

### AI/ML Engineering

```bash
/plugin install yzmir-deep-rl
```

```plaintext
I'm using yzmir/deep-rl/policy-gradient-methods to implement PPO for continuous character control
```

### Game Development

```bash
/plugin install bravos-systems-as-experience
```

``````plaintext
I'm using bravos/systems-as-experience/emergent-gameplay-design to create systemic interactions in my RPG
```

### Security Architecture

```bash
/plugin install ordis-security-architect
```

```
I'm using ordis/security-architect/threat-modeling to analyze this authentication system with STRIDE
```

### UX Design

```bash
/plugin install lyra-ux-designer
```

```
I'm using lyra/ux-designer/accessibility-and-inclusive-design to ensure WCAG 2.1 AA compliance
```

---

## What's Inside

- **120+ Skills**: Production-ready guidance across 5 domains
- **13 Packs**: Install only what you need
- **Per-Pack Plugins**: Independent versioning, clean dependencies
- **Complete Coverage**: Design → Foundation → Training → Production

---

## Repository Structure

```
skillpacks/
├── .claude-plugin/
│   └── marketplace.json       # Marketplace catalog
├── plugins/                   # All 13 skillpacks
│   ├── ordis-security-architect/
│   ├── muna-technical-writer/
│   ├── yzmir-deep-rl/
│   ├── lyra-ux-designer/
│   └── ... (9 more)
└── source/                    # Original structure (archived)
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
