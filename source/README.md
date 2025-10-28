# Security Architect & Technical Writer Skills for Claude Code

Professional security architecture and technical writing skills that teach Claude Code how to perform threat modeling, design security controls, create compliance documentation, write incident response runbooks, and more.

**18 skills total:** 2 meta-skills for routing + 8 universal core skills + 8 specialized extension skills

**Ready for:** Security professionals, technical writers, and developers learning these domains

---

## Quick Start

**1. Clone this repository:**
```bash
git clone https://github.com/tachyon-beep/skillpacks
cd skillpacks
```

**2. Load a skill in Claude Code:**
```
I'm using ordis/security-architect/threat-modeling to analyze this API for security threats
```

**3. Claude applies the skill to your project**

That's it! Claude now has expert-level knowledge of threat modeling methodology.

---

## Skill Catalog

### Ordis - Security Architect (9 skills)

**Meta-Skill:**
- **using-security-architect** - Routes to appropriate security skills based on your task

**Core Skills (Universal):**
- **threat-modeling** - STRIDE methodology, attack trees, risk scoring matrices
- **security-controls-design** - Defense-in-depth, trust boundaries, control effectiveness
- **secure-by-design-patterns** - Zero-trust architecture, immutable infrastructure, least privilege
- **security-architecture-review** - Systematic review checklists for design and implementation

**Extension Skills (Specialized Contexts):**
- **classified-systems-security** - Bell-LaPadula MLS model, fail-fast validation, classified data handling
- **compliance-awareness-and-mapping** - Framework discovery (ISM/HIPAA/SOC2/GDPR), control mapping
- **security-authorization-and-accreditation** - Government ATO/AIS processes, RMF 7-step, SSP/SAR/POA&M
- **documenting-threats-and-controls** - Security ADRs, threat documentation, control traceability

### Muna - Technical Writer (9 skills)

**Meta-Skill:**
- **using-technical-writer** - Routes to appropriate documentation skills based on your task

**Core Skills (Universal):**
- **documentation-structure** - ADR patterns, README templates, documentation organization
- **clarity-and-style** - Active voice, concrete examples, avoiding ambiguity
- **diagram-conventions** - Decision trees, semantic labels, visual clarity
- **documentation-testing** - 5 dimensions: completeness, accuracy, findability, examples, walkthrough

**Extension Skills (Specialized Contexts):**
- **security-aware-documentation** - Sanitizing examples (fake credentials), preventing leaks, threat disclosure
- **incident-response-documentation** - 5-phase response template, time-critical runbooks, escalation paths
- **itil-and-governance-documentation** - RFC templates, service catalogs, SLA/OLA, DR plans
- **operational-acceptance-documentation** - Production readiness checklists, go-live approval, handover

---

## Example Usage

### Example 1: Threat Model a REST API

```
User: I'm building a user authentication API. Help me identify security threats.

Claude: I'm using ordis/security-architect/threat-modeling to analyze your API.

Let me apply STRIDE methodology:

**Spoofing:** Attacker impersonates legitimate user
- Threat: Stolen credentials used to access API
- Severity: HIGH

**Tampering:** Attacker modifies authentication tokens
- Threat: JWT token signature forged
- Severity: HIGH

[Complete STRIDE analysis with attack trees and risk scoring]
```

### Example 2: Write Security-Aware Documentation

```
User: I need to document our API authentication flow with examples.

Claude: I'm using muna/technical-writer/security-aware-documentation.

I'll create examples with obviously fake credentials to prevent leaks:

```bash
# Authenticate user
curl -X POST https://api.example.com/auth \
  -H "Authorization: Bearer fake_api_key_abc123_for_docs_only" \
  -d '{"user": "jane.doe@example.com", "password": "fake_password_example_only"}'
```

[Complete documentation with sanitized examples]
```

### Example 3: Government System Authorization

```
User: We need ATO documentation for a classified system.

Claude: I'm using ordis/security-architect/security-authorization-and-accreditation.

Let me guide you through the RMF 7-step process:

**Step 1: PREPARE** - Define system boundary
**Step 2: CATEGORIZE** - Apply FIPS 199 (your system is HIGH confidentiality)
**Step 3: SELECT** - Choose NIST 800-53 controls (HIGH baseline)
...

[Complete ATO process with SSP/SAR/POA&M templates]
```

---

## Who Should Use This

**Security Professionals & Architects:**
- Apply proven threat modeling frameworks (STRIDE, attack trees)
- Design defense-in-depth security controls
- Navigate compliance frameworks (ISM/HIPAA/SOC2/GDPR)
- Prepare government authorization packages (ATO/AIS)

**Technical Writers & Documentation Engineers:**
- Structure documentation effectively (ADRs, README patterns)
- Write clear, unambiguous content
- Create security-aware documentation (sanitized examples)
- Develop incident response runbooks

**Developers Learning Security & Documentation:**
- Learn threat modeling by applying it to your projects
- Understand security best practices through real examples
- Write better documentation with proven patterns
- Build skills in regulated/high-security contexts

**Teams in Regulated Industries:**
- Healthcare (HIPAA compliance documentation)
- Government/Defense (classified systems, ATO processes)
- Finance (compliance mapping, security controls)
- Enterprise (ITIL, incident response, DR planning)

---

## Installation

### Option 1: Git Clone (Available Now)

```bash
git clone https://github.com/tachyon-beep/skillpacks
cd skillpacks
```

Then in Claude Code:
```
I'm using ordis/security-architect/threat-modeling
```

### Option 2: Claude Code Plugin (Coming Soon)

Install from Claude Code plugin marketplace - one-click installation, automatic updates.

---

## Tutorials

End-to-end tutorials showing realistic skill usage:

1. **[Secure a REST API from Scratch](docs/tutorials/01-secure-rest-api.md)** - Apply threat modeling and security controls to new API
2. **[Healthcare System Compliance Documentation](docs/tutorials/02-healthcare-compliance.md)** - Create HIPAA-compliant documentation with sanitized examples
3. **[Government System Authorization](docs/tutorials/03-government-ato.md)** - Complete ATO package for classified system
4. **[Incident Response Readiness](docs/tutorials/04-incident-response.md)** - Build incident response capabilities from scratch
5. **[Security + Documentation Lifecycle](docs/tutorials/05-security-documentation-lifecycle.md)** - Complete lifecycle from threat model to public docs

---

## Architecture

**Layered Design:**
- **Meta-skills** route to appropriate skills based on task
- **Core skills** are universal (any project, any domain)
- **Extension skills** specialize for high-security/regulated contexts

**Bidirectional Cross-References:**
Skills reference each other, creating a knowledge graph. Example: threat-modeling references security-controls-design, which references documenting-threats-and-controls.

**TDD for Documentation:**
Every skill tested with RED-GREEN-REFACTOR methodology before deployment. See [DEVELOPMENT.md](DEVELOPMENT.md) for details.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- How to propose new skills
- RED-GREEN-REFACTOR testing requirements
- Skill creation guidelines
- Pull request process

---

## Design Principles

1. **Layered:** Core (universal) + Extensions (specialized contexts)
2. **Modular:** Load only what you need
3. **Cross-Referenced:** Bidirectional knowledge graph between skills
4. **TDD for Documentation:** Every skill tested before deployment
5. **Meta-Aware:** Teach patterns and methodology, not exhaustive lists

---

## Project Status

**Phase 3 Complete:** All 18 skills implemented and tested
- ~45-50 hours invested across Phases 1-3
- Ready for personal/team use
- Phase 4 (public polish) in progress

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Origin Story

Designed alongside **Elspeth** (security-first LLM orchestration platform) but universal and reusable. Informed by:
- Bell-LaPadula MLS enforcement in classified systems
- ADR-quality documentation practices
- TDD-based complexity reduction methodology
- High-security/regulated environment patterns (government, healthcare, finance)

**Applies to any project:** Web apps, mobile, embedded, data pipelines, infrastructure.

---

## Questions?

- **GitHub Issues:** Report bugs or request features
- **GitHub Discussions:** Ask questions, share usage examples
- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md)
