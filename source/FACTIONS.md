# Faction Organization

**Inspired by**: Altered TCG faction themes
**Purpose**: Thematic grouping of skill packs for discoverability and conceptual clarity

---

## The Six Factions

### Ordis - Protectors of Order
**Theme**: Security architecture, controls, governance, protection
**Archetype**: Guardians who establish and maintain security boundaries
**Skills**: Security architecture, threat modeling, compliance, access control

**Current Skill Packs**:
- `security-architect` - Security architecture and threat analysis (Phase 1)

**Future Potential**:
- Access control and authorization patterns
- Security incident response
- Audit and compliance frameworks
- Security testing and verification

---

### Muna - Weavers of Harmony
**Theme**: Cross-discipline patterns, documentation, connection-making
**Archetype**: Harmonizers who weave understanding across boundaries
**Skills**: Technical writing, documentation, knowledge management, synthesis

**Current Skill Packs**:
- `technical-writer` - Documentation structure and clarity (Phase 1)

**Future Potential**:
- Knowledge management systems
- Cross-functional communication
- API documentation patterns
- Developer experience design

---

### Axiom - Creators of Marvels
**Theme**: Tech-forward, science, tooling, innovation
**Archetype**: Engineers who build sophisticated technical systems
**Skills**: Build systems, CI/CD, developer tools, automation

**Current Skill Packs**: (None yet)

**Future Potential**:
- Build system optimization
- CI/CD pipeline design
- Developer tooling creation
- Infrastructure as code
- Performance testing frameworks

---

### Bravos - Champions of Action
**Theme**: Practical recipes, doers' tactics, pushing through adversity
**Archetype**: Pragmatic heroes who deliver results under pressure
**Skills**: Debugging, incident response, production reliability, practical patterns

**Current Skill Packs**: (None yet)

**Future Potential**:
- Production debugging techniques
- Incident response playbooks
- Reliability engineering patterns
- On-call best practices
- Quick-fix tactical patterns

---

### Lyra - Nomadic Artists
**Theme**: Creative personas, UX craft, disruptive innovation
**Archetype**: Artists who create delightful, human-centered experiences
**Skills**: UX design, interaction patterns, accessibility, visual design

**Current Skill Packs**: (None yet)

**Future Potential**:
- UX research and testing
- Interaction design patterns
- Accessibility engineering
- Design system creation
- User journey mapping

---

### Yzmir - Magicians of the Mind
**Theme**: Deep theory, game theory, cryptography, complex algorithms
**Archetype**: Theoreticians who wield mathematical and algorithmic power
**Skills**: Algorithms, cryptography, formal methods, complex system design

**Current Skill Packs**: (None yet)

**Future Potential**:
- Cryptographic protocol design
- Algorithm optimization patterns
- Formal verification methods
- Distributed systems theory
- Game theory applications

---

## Design Principles

### 1. Faction Coherence
Each skill pack should align with its faction's archetype. If a skill pack doesn't clearly fit, reconsider the classification or create a new faction.

### 2. Cross-Faction References
Skills can (and should!) reference skills from other factions. Example:
- Ordis `security-architect` → Muna `technical-writer` (for security documentation)
- Axiom build tools → Ordis security controls (for supply chain security)

### 3. Faction Independence
Each faction should be independently valuable. You can use Ordis skills without Muna, though they enhance each other.

### 4. Future Growth
Factions provide a framework for unlimited growth. As new skill packs emerge, they find natural homes in existing factions.

---

## Directory Structure

```
source/
├── ordis/
│   └── security-architect/
│       ├── using-security-architect/
│       ├── threat-modeling/
│       ├── security-controls-design/
│       └── ...
├── muna/
│   └── technical-writer/
│       ├── using-technical-writer/
│       ├── documentation-structure/
│       ├── clarity-and-style/
│       └── ...
├── axiom/
│   └── (future skill packs)
├── bravos/
│   └── (future skill packs)
├── lyra/
│   └── (future skill packs)
├── yzmir/
│   └── (future skill packs)
└── planning/
    └── (project planning docs)
```

---

## Naming Convention

**Skill Pack Path**: `{faction}/{pack-name}/{skill-name}/SKILL.md`

**Examples**:
- `ordis/security-architect/threat-modeling/SKILL.md`
- `muna/technical-writer/documentation-structure/SKILL.md`
- `axiom/build-systems/ci-cd-patterns/SKILL.md` (future)

---

## When to Create New Faction

**Don't create new factions lightly.** The six factions cover broad thematic ground:
- **Security/Governance** → Ordis
- **Documentation/Synthesis** → Muna
- **Tools/Infrastructure** → Axiom
- **Tactical/Practical** → Bravos
- **UX/Creative** → Lyra
- **Theory/Algorithms** → Yzmir

**Create new faction only if**:
1. Skill pack doesn't fit any existing faction theme
2. There's enough thematic coherence for 3+ future skill packs
3. The archetype is distinct and clear

---

## Cross-Faction Skill Examples

Some skills naturally span factions. These go in the faction that **uses** the skill, and reference the faction that **teaches** related techniques:

**Example**: `ordis/security-architect/documenting-threats-and-controls`
- **Primary faction**: Ordis (security architects use this)
- **References**: Muna `technical-writer/documentation-structure` (how to write docs)
- **Nature**: Cross-cutting skill that bridges factions

---

## Faction Flavor (Optional Style Guide)

When writing skills, you can (optionally) incorporate faction flavor:

**Ordis**: Formal, structured, protective language. "Establish boundaries." "Enforce controls."
**Muna**: Harmonious, connective language. "Weave understanding." "Bridge concepts."
**Axiom**: Technical, innovative language. "Engineer solutions." "Build systems."
**Bravos**: Direct, action-oriented language. "Ship it." "Push through."
**Lyra**: Creative, human-centered language. "Craft experiences." "Delight users."
**Yzmir**: Theoretical, precise language. "Prove correctness." "Reason about."

**This is optional** - clarity and utility trump flavor. But faction voice can enhance identity.

---

## Current Status

**Phase 1 (Active)**:
- ✅ Ordis: `security-architect` (4 skills planned)
- ✅ Muna: `technical-writer` (4 skills planned)

**Future Phases**:
- Ordis: Additional security packs (compliance, incident response)
- Muna: Additional documentation packs (API docs, knowledge management)
- Axiom, Bravos, Lyra, Yzmir: New skill packs as needs emerge

---

## Benefits of Faction Organization

1. **Discovery**: "I need security skills" → Browse Ordis
2. **Coherence**: Related skills naturally grouped
3. **Scalability**: Framework supports dozens of skill packs
4. **Identity**: Each faction has clear thematic identity
5. **Cross-Faction Value**: See connections between domains
6. **Fun**: Thematic organization is more engaging than flat lists

---

## Getting Started

**Current work**: Phase 1 skills in Ordis and Muna
**Next**: Follow RED-GREEN-REFACTOR for each skill
**Future**: As new skill packs emerge, place them in appropriate factions

The faction structure is **governance from day one** - every skill has a clear thematic home.
