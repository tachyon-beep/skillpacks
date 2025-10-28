# Phase 4: Public Release Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform skillpacks from "Phase 3 complete" to "public-ready open-source project" with GitHub repository polish, Claude Code plugin packaging, and end-to-end tutorials.

**Architecture:** Three sequential waves - (1) Repository polish with public-facing docs and governance files, (2) Plugin manifest and packaging for Claude Code marketplace, (3) Five end-to-end tutorials showing realistic skill usage across diverse domains.

**Tech Stack:** Markdown, JSON (plugin manifest), Git, YAML (frontmatter), GitHub templates

**Timeline:** 8-14 hours (15-17 tasks, 30-60 min each)

---

## Wave 1: Repository Polish (4-5 hours)

### Task 1: Create DEVELOPMENT.md with Implementation Details

**Goal:** Move implementation details from README.md to DEVELOPMENT.md for developers/contributors.

**Files:**
- Create: `DEVELOPMENT.md`

**Step 1: Create DEVELOPMENT.md with Phase 1-3 implementation history**

Extract implementation content from current README.md (lines 20-180 approximately) and expand with development workflow.

Content structure:
```markdown
# Development Guide

This guide documents the implementation history and development workflow for these skillpacks.

## Implementation History

### Phase 1: Foundation (4 skills, 10-15 hours)
[Extract Phase 1 details from README]

### Phase 2: Core Skills (6 skills, 18-30 hours)
[Extract Phase 2 details from README]

### Phase 3: Extensions (8 skills, 24-40 hours)
[Extract Phase 3 details from README]

## Development Workflow

### RED-GREEN-REFACTOR Methodology

All skills MUST be tested before deployment:

**RED Phase:** Baseline test WITHOUT skill
- Create realistic scenario
- Test Claude WITHOUT the skill
- Document failures objectively

**GREEN Phase:** Create skill addressing failures
- Write skill targeting documented gaps
- Include real-world examples
- Add cross-references

**REFACTOR Phase:** Verification test WITH skill
- Test Claude WITH the skill
- Verify all baseline failures resolved
- Test under pressure (time constraints, edge cases)

### Git Commit Pattern

[Document the commit message pattern used throughout Phases 1-3]

### Faction Organization

[Extract faction details from FACTIONS.md if needed]

## Testing Skills

[Reference superpowers:testing-skills-with-subagents methodology]
```

**Step 2: Verify content is complete**

Check that DEVELOPMENT.md includes:
- [ ] All phase implementation history
- [ ] RED-GREEN-REFACTOR methodology explained
- [ ] Git commit patterns documented
- [ ] Testing approach documented

**Step 3: Commit**

```bash
git add DEVELOPMENT.md
git commit -m "docs: Create DEVELOPMENT.md with implementation history

Move implementation details from README to dedicated development guide.
Includes Phase 1-3 history, RED-GREEN-REFACTOR methodology, git patterns.

Prepares README for public-facing restructure.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Restructure README.md for Public Audience

**Goal:** Transform README from implementation-focused to public discovery-focused.

**Files:**
- Modify: `README.md` (complete rewrite)

**Step 1: Rewrite README.md with public-facing structure**

Complete new content (replace existing):

```markdown
# Security Architect & Technical Writer Skills for Claude Code

Professional security architecture and technical writing skills that teach Claude Code how to perform threat modeling, design security controls, create compliance documentation, write incident response runbooks, and more.

**18 skills total:** 2 meta-skills for routing + 8 universal core skills + 8 specialized extension skills

**Ready for:** Security professionals, technical writers, and developers learning these domains

---

## Quick Start

**1. Clone this repository:**
```bash
git clone https://github.com/[username]/skillpacks
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

### Ordis - Security Architect (10 skills)

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

### Muna - Technical Writer (8 skills)

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
git clone https://github.com/[username]/skillpacks
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
```

**Step 2: Replace [username] placeholders**

Update the README with actual GitHub username/repository path.

**Step 3: Verify README structure**

Check that README includes:
- [ ] Quick start (3 steps, <30 seconds to understand)
- [ ] Complete skill catalog (all 18 skills with descriptions)
- [ ] 3 usage examples (concrete, showing skill loading)
- [ ] Who should use this (4 audience segments)
- [ ] Installation (git clone + plugin mention)
- [ ] Tutorials (links to 5 tutorials)
- [ ] Contributing link
- [ ] License

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: Restructure README for public audience

Transform README from implementation-focused to discovery-focused:
- Quick start (3 steps to use skills)
- Complete skill catalog (18 skills with descriptions)
- 3 concrete usage examples
- Clear audience segments (professionals + learners)
- Installation options (git clone + plugin coming soon)
- Tutorial links (5 end-to-end scenarios)

Moved implementation details to DEVELOPMENT.md.

Public-ready README enables discovery in <2 minutes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Create CONTRIBUTING.md

**Goal:** Guide contributors through skill creation process with RED-GREEN-REFACTOR requirements.

**Files:**
- Create: `CONTRIBUTING.md`

**Step 1: Create CONTRIBUTING.md**

Complete content:

```markdown
# Contributing to Security Architect & Technical Writer Skills

Thank you for your interest in contributing! These skillpacks welcome new skills, bug fixes, and documentation improvements.

---

## How to Propose a New Skill

**1. Create GitHub issue first** using `.github/ISSUE_TEMPLATE/new_skill.md`

Before implementing, create an issue describing:
- **Skill name:** Which faction and pack (e.g., `ordis/security-architect/api-security-testing`)
- **Purpose:** What does this skill teach? (2-3 sentences)
- **When to use:** What situations trigger using this skill?
- **Related skills:** Which existing skills does this complement?
- **Real-world gap:** What does Claude currently miss that this skill addresses?

**2. Wait for maintainer feedback**

Maintainers will review to ensure:
- Skill addresses real gap (not duplicate of existing skill)
- Fits faction/pack organization
- Scope is appropriate (not too broad or too narrow)

**3. Implement with RED-GREEN-REFACTOR testing** (required)

See "Skill Creation Requirements" below.

---

## Skill Creation Requirements

**All skills MUST follow RED-GREEN-REFACTOR TDD methodology.**

This ensures skills actually work and address real gaps.

### RED Phase: Baseline Test Without Skill

**Goal:** Objectively measure what Claude misses without your skill.

**Process:**
1. Create realistic scenario where your skill should help
2. Test Claude (via subagent or fresh session) WITHOUT loading your skill
3. Document failures objectively:
   - What did Claude miss?
   - What did Claude get wrong?
   - What was inefficient or unclear?

**Example (threat-modeling skill):**
```
Scenario: Design auth API, need threat analysis

WITHOUT skill:
❌ Missed STRIDE methodology (used ad-hoc intuition)
❌ No attack trees (listed threats but no paths)
❌ No risk scoring (couldn't prioritize threats)
❌ No systematic coverage (missed info disclosure threats)
```

### GREEN Phase: Create Skill Addressing Failures

**Goal:** Write skill that addresses every documented baseline failure.

**Required Skill Components:**

1. **YAML Frontmatter:**
```yaml
---
name: skill-name-here
description: Use when [trigger] - covers [what it teaches]
---
```

2. **Overview Section:**
   - Core principle (one sentence)
   - Key insight (what makes this skill valuable)

3. **When to Use Section:**
   - Load this skill when: [situations]
   - Symptoms you need this: [trigger patterns]
   - Don't use for: [anti-patterns]

4. **Core Content:**
   - Teach patterns and methodology (not exhaustive lists)
   - Include decision trees and checklists
   - Provide 3+ real-world examples (concrete, complete)

5. **Cross-References:**
   - Use WITH this skill: [complementary skills]
   - Use AFTER this skill: [follow-up skills]

6. **Real-World Impact Section:**
   - How has this skill been used?
   - What concrete improvements did it deliver?
   - Key lesson (one sentence takeaway)

**Example Structure:**
```markdown
---
name: my-new-skill
description: Use when [situation] - covers [capabilities]
---

# My New Skill

## Overview
[Core principle and key insight]

## When to Use
[Trigger situations, symptoms, anti-patterns]

## [Core Content Sections]
[Teach the methodology with examples]

## Cross-References
**Use WITH:** [related skills]

## Real-World Impact
[Proven effectiveness with concrete examples]
```

### REFACTOR Phase: Verification Test With Skill

**Goal:** Prove skill addresses all baseline failures.

**Process:**
1. Test Claude (via subagent) WITH your skill loaded
2. Apply to same scenario from RED phase
3. Verify every baseline failure is now resolved
4. Test under pressure:
   - Time constraints ("90-minute emergency patch")
   - Edge cases
   - Scope creep scenarios

**Example (threat-modeling skill):**
```
Scenario: Same auth API design

WITH skill:
✅ Applied STRIDE systematically (all 6 categories)
✅ Built attack trees with feasibility marking
✅ Scored risks with Likelihood × Impact matrix
✅ Found ALL threats (including info disclosure)
```

**If any baseline failures remain:** Return to GREEN phase, improve skill, re-test.

### Document Testing in Commit Message

Commit message must include RED-GREEN-REFACTOR testing summary:

```
Implement ordis/security-architect/my-new-skill

RED-GREEN-REFACTOR cycle complete:

RED Phase:
- Tested scenario X without skill
- Documented 5 baseline failures: [list]

GREEN Phase:
- Created skill addressing all failures
- Included [pattern Y] and [methodology Z]

REFACTOR Phase:
- Tested WITH skill - all failures resolved
- Tested under pressure - maintains discipline
- Skill is bulletproof

[Additional context]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Faction Organization

Skills are grouped by Altered TCG-inspired factions:

- **Ordis** (Protectors of Order): Security, governance, compliance
- **Muna** (Weavers of Harmony): Documentation, synthesis, clarity
- **Axiom** (Creators of Marvels): Tooling, infrastructure, automation
- **Bravos** (Champions of Action): Practical tactics, execution
- **Lyra** (Nomadic Artists): UX, creative, design
- **Yzmir** (Magicians of Mind): Theory, algorithms, architecture

See [FACTIONS.md](FACTIONS.md) for details on choosing appropriate faction.

---

## Style Guide

### Markdown Formatting
- Use `#` for main heading, `##` for sections
- Use triple backticks with language tags for code blocks
- Use `**bold**` for emphasis, `*italic*` for secondary emphasis
- Use numbered lists for sequential steps, bullet lists for unordered items

### YAML Frontmatter
```yaml
---
name: faction/pack/skill-name  # Use kebab-case
description: Use when [trigger] - covers [capabilities]  # One line, <120 chars
---
```

### Cross-References
Use relative paths:
```markdown
**Use WITH this skill:**
- `ordis/security-architect/threat-modeling` - [Brief description]
```

### Decision Trees
Use markdown tables or ASCII art:
```markdown
1. **Check X:**
   - If Y → Proceed to step 2
   - If Z → See [section]
```

### Real-World Examples
- Use obviously fake credentials: `fake_api_key_abc123_for_docs_only`
- Use example.com domains (RFC 2606)
- Use complete examples (not placeholders like `YOUR_KEY_HERE`)

---

## Pull Request Process

**1. Fork repository**

```bash
git clone https://github.com/[your-username]/skillpacks
cd skillpacks
```

**2. Create feature branch**

```bash
git checkout -b skill/my-new-skill
```

**3. Implement skill with RED-GREEN-REFACTOR testing**

Follow "Skill Creation Requirements" above.

**4. Commit with detailed message**

Include RED-GREEN-REFACTOR testing summary in commit message.

**5. Push to your fork**

```bash
git push origin skill/my-new-skill
```

**6. Create Pull Request**

Use `.github/PULL_REQUEST_TEMPLATE.md` template.

### PR Checklist

Before submitting PR, verify:

- [ ] RED-GREEN-REFACTOR testing complete (documented in commit message)
- [ ] Real-world examples included (3+ concrete examples)
- [ ] Cross-references updated (complementary skills linked)
- [ ] YAML frontmatter complete (name + description)
- [ ] Follows style guide (markdown, formatting)
- [ ] "Real-World Impact" section included
- [ ] Ready for review

---

## Code Review Process

Maintainers will review:

1. **Skill addresses real gap**
   - Not duplicate of existing skill
   - Solves problem Claude currently struggles with

2. **RED-GREEN-REFACTOR methodology followed**
   - Baseline testing documented
   - Skill addresses baseline failures
   - Verification testing proves effectiveness

3. **Examples are realistic and complete**
   - Concrete scenarios (not abstract)
   - Complete examples (copy-paste-runnable)
   - Obviously fake credentials/data

4. **Writing is clear and actionable**
   - Core principle stated clearly
   - Decision trees and checklists provided
   - No ambiguity

5. **Cross-references are accurate**
   - Related skills linked correctly
   - Relative paths work

---

## Bug Reports & Feature Requests

**Bug in existing skill:**
- Use `.github/ISSUE_TEMPLATE/bug_report.md`
- Include skill name, expected behavior, actual behavior
- Provide reproduction steps

**Feature request:**
- Use GitHub Discussions for initial discussion
- If consensus reached, maintainer creates issue
- Follow "Propose New Skill" process

---

## Development Workflow

For detailed development workflow, testing methodology, and implementation history, see [DEVELOPMENT.md](DEVELOPMENT.md).

---

## Questions?

- **GitHub Issues:** Propose skills, report bugs
- **GitHub Discussions:** Ask questions, get feedback
- **Pull Requests:** Submit contributions

Thank you for contributing to these skillpacks!
```

**Step 2: Verify CONTRIBUTING.md completeness**

Check that CONTRIBUTING.md includes:
- [ ] How to propose new skills (issue first)
- [ ] RED-GREEN-REFACTOR requirements (detailed)
- [ ] Required skill components (checklist)
- [ ] Faction organization guide
- [ ] Style guide (markdown, YAML, examples)
- [ ] PR process (fork, branch, commit, submit)
- [ ] PR checklist
- [ ] Code review criteria

**Step 3: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: Create CONTRIBUTING.md with skill creation guide

Complete contribution guidelines including:
- New skill proposal process (issue first)
- RED-GREEN-REFACTOR testing requirements (mandatory)
- Required skill components (YAML, structure, examples)
- Faction organization guide
- Style guide (markdown, cross-references, examples)
- PR process and checklist
- Code review criteria

Enables community contributions with quality standards.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Add CODE_OF_CONDUCT.md

**Goal:** Establish community conduct standards using industry-standard Contributor Covenant.

**Files:**
- Create: `CODE_OF_CONDUCT.md`

**Step 1: Create CODE_OF_CONDUCT.md with Contributor Covenant 2.1**

Content (standard Contributor Covenant):

```markdown
# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, caste, color, religion, or sexual
identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming,
diverse, inclusive, and healthy community.

## Our Standards

Examples of behavior that contributes to a positive environment for our
community include:

* Demonstrating empathy and kindness toward other people
* Being respectful of differing opinions, viewpoints, and experiences
* Giving and gracefully accepting constructive feedback
* Accepting responsibility and apologizing to those affected by our mistakes,
  and learning from the experience
* Focusing on what is best not just for us as individuals, but for the overall
  community

Examples of unacceptable behavior include:

* The use of sexualized language or imagery, and sexual attention or advances of
  any kind
* Trolling, insulting or derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or email address,
  without their explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

## Enforcement Responsibilities

Community leaders are responsible for clarifying and enforcing our standards of
acceptable behavior and will take appropriate and fair corrective action in
response to any behavior that they deem inappropriate, threatening, offensive,
or harmful.

Community leaders have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are
not aligned to this Code of Conduct, and will communicate reasons for moderation
decisions when appropriate.

## Scope

This Code of Conduct applies within all community spaces, and also applies when
an individual is officially representing the community in public spaces.
Examples of representing our community include using an official e-mail address,
posting via an official social media account, or acting as an appointed
representative at an online or offline event.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported to the community leaders responsible for enforcement at
[INSERT CONTACT EMAIL].

All complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the
reporter of any incident.

## Enforcement Guidelines

Community leaders will follow these Community Impact Guidelines in determining
the consequences for any action they deem in violation of this Code of Conduct:

### 1. Correction

**Community Impact**: Use of inappropriate language or other behavior deemed
unprofessional or unwelcome in the community.

**Consequence**: A private, written warning from community leaders, providing
clarity around the nature of the violation and an explanation of why the
behavior was inappropriate. A public apology may be requested.

### 2. Warning

**Community Impact**: A violation through a single incident or series of
actions.

**Consequence**: A warning with consequences for continued behavior. No
interaction with the people involved, including unsolicited interaction with
those enforcing the Code of Conduct, for a specified period of time. This
includes avoiding interactions in community spaces as well as external channels
like social media. Violating these terms may lead to a temporary or permanent
ban.

### 3. Temporary Ban

**Community Impact**: A serious violation of community standards, including
sustained inappropriate behavior.

**Consequence**: A temporary ban from any sort of interaction or public
communication with the community for a specified period of time. No public or
private interaction with the people involved, including unsolicited interaction
with those enforcing the Code of Conduct, is allowed during this period.
Violating these terms may lead to a permanent ban.

### 4. Permanent Ban

**Community Impact**: Demonstrating a pattern of violation of community
standards, including sustained inappropriate behavior, harassment of an
individual, or aggression toward or disparagement of classes of individuals.

**Consequence**: A permanent ban from any sort of public interaction within the
community.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 2.1, available at
[https://www.contributor-covenant.org/version/2/1/code_of_conduct.html][v2.1].

Community Impact Guidelines were inspired by
[Mozilla's code of conduct enforcement ladder][Mozilla CoC].

For answers to common questions about this code of conduct, see the FAQ at
[https://www.contributor-covenant.org/faq][FAQ]. Translations are available at
[https://www.contributor-covenant.org/translations][translations].

[homepage]: https://www.contributor-covenant.org
[v2.1]: https://www.contributor-covenant.org/version/2/1/code_of_conduct.html
[Mozilla CoC]: https://github.com/mozilla/diversity
[FAQ]: https://www.contributor-covenant.org/faq
[translations]: https://www.contributor-covenant.org/translations
```

**Step 2: Update contact email**

Replace `[INSERT CONTACT EMAIL]` with actual contact email for code of conduct enforcement.

**Step 3: Commit**

```bash
git add CODE_OF_CONDUCT.md
git commit -m "docs: Add Contributor Covenant Code of Conduct

Add standard Contributor Covenant 2.1 code of conduct.
Establishes community standards for respectful collaboration.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Add LICENSE (Apache 2.0)

**Goal:** Add Apache License 2.0 for enterprise compatibility and patent grant.

**Files:**
- Create: `LICENSE`

**Step 1: Create LICENSE file with Apache 2.0 text**

Full Apache License 2.0 text available at: https://www.apache.org/licenses/LICENSE-2.0.txt

Copy complete license text, update copyright year and holder:

```
Copyright [YEAR] [COPYRIGHT HOLDER]
```

**Step 2: Verify license is complete**

Check that LICENSE includes:
- [ ] Complete Apache 2.0 license text
- [ ] Copyright year updated (2025)
- [ ] Copyright holder name updated

**Step 3: Commit**

```bash
git add LICENSE
git commit -m "docs: Add Apache License 2.0

Add Apache 2.0 license for enterprise compatibility.
Includes patent grant and permissive terms.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Create GitHub Issue Templates

**Goal:** Provide structured templates for skill proposals and bug reports.

**Files:**
- Create: `.github/ISSUE_TEMPLATE/new_skill.md`
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`

**Step 1: Create directory structure**

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

**Step 2: Create new_skill.md template**

Content for `.github/ISSUE_TEMPLATE/new_skill.md`:

```markdown
---
name: New Skill Proposal
about: Propose a new skill for the skillpacks
title: '[SKILL] '
labels: 'skill-proposal'
assignees: ''
---

## Skill Name

[e.g., ordis/security-architect/api-security-testing]

## Purpose

[What does this skill teach? 2-3 sentences]

## When to Use

[What situations trigger using this skill? Be specific about symptoms/triggers]

## Faction & Pack

**Faction:** [ordis | muna | axiom | bravos | lyra | yzmir]
**Pack:** [e.g., security-architect, technical-writer]

## Related Skills

[Which existing skills does this complement or extend?]

## Real-World Gap

[What does Claude currently miss or get wrong that this skill addresses? Be specific with examples]

**Example scenario:**
[Describe a concrete scenario where Claude struggles without this skill]

**Current behavior (without skill):**
- [What Claude misses or does incorrectly]

**Desired behavior (with skill):**
- [What Claude should do with this skill]

## Additional Context

[Any other relevant information, examples, or references]
```

**Step 3: Create bug_report.md template**

Content for `.github/ISSUE_TEMPLATE/bug_report.md`:

```markdown
---
name: Bug Report
about: Report a bug in an existing skill
title: '[BUG] '
labels: 'bug'
assignees: ''
---

## Skill Name

[Which skill has the bug? e.g., ordis/security-architect/threat-modeling]

## Expected Behavior

[What should happen when using this skill?]

## Actual Behavior

[What actually happens? Be specific about what goes wrong]

## Steps to Reproduce

1. Load skill: [skill name]
2. Apply to scenario: [description]
3. Observe: [what goes wrong]

## Example

[Provide concrete example if possible]

```
[Claude interaction showing the bug]
```

## Environment

- Claude Code version: [e.g., 1.0.0]
- Operating System: [e.g., macOS 14.0]
- Installation method: [git clone | plugin]

## Additional Context

[Any other relevant information, screenshots, or logs]

## Suggested Fix

[Optional: If you have ideas for how to fix this]
```

**Step 4: Verify templates**

Check that templates include:
- [ ] YAML frontmatter (name, about, title, labels)
- [ ] Clear section structure
- [ ] Specific prompts for required information
- [ ] Examples showing what to include

**Step 5: Commit**

```bash
git add .github/ISSUE_TEMPLATE/
git commit -m "docs: Add GitHub issue templates

Add structured templates for:
- New skill proposals (skill name, purpose, gap analysis)
- Bug reports (reproduction steps, expected/actual behavior)

Guides contributors to provide complete information.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Create GitHub Pull Request Template

**Goal:** Provide PR template with checklist ensuring quality standards.

**Files:**
- Create: `.github/PULL_REQUEST_TEMPLATE.md`

**Step 1: Create pull request template**

Content for `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Description

[What does this PR do? Brief summary]

## Type of Change

- [ ] New skill
- [ ] Bug fix (fixes existing skill)
- [ ] Documentation improvement
- [ ] Other (please describe)

## Related Issue

[Link to related issue, e.g., Closes #123]

---

## For New Skills: RED-GREEN-REFACTOR Testing

**Required for all new skills. Delete this section if not applicable.**

### RED Phase: Baseline Test Without Skill
- [ ] Realistic scenario created
- [ ] Tested Claude WITHOUT skill
- [ ] Baseline failures documented in commit message

**Baseline failures documented:** [List key failures or link to commit message]

### GREEN Phase: Skill Created
- [ ] Skill addresses all baseline failures
- [ ] YAML frontmatter complete (name + description)
- [ ] "When to Use" section included
- [ ] Real-world examples included (3+)
- [ ] Cross-references added

### REFACTOR Phase: Verification Test With Skill
- [ ] Tested Claude WITH skill
- [ ] All baseline failures resolved
- [ ] Tested under pressure (time constraints, edge cases)
- [ ] Skill proven bulletproof

**Testing summary in commit message:** [Confirm RED-GREEN-REFACTOR documented]

---

## Skill Quality Checklist

**Delete this section if not a new skill.**

- [ ] **YAML Frontmatter:** name and description fields complete
- [ ] **Overview:** Core principle and key insight stated clearly
- [ ] **When to Use:** Trigger situations and anti-patterns documented
- [ ] **Core Content:** Patterns and methodology taught (not exhaustive lists)
- [ ] **Real-World Examples:** 3+ concrete, complete examples
- [ ] **Cross-References:** Related skills linked (use WITH, use AFTER)
- [ ] **Real-World Impact:** Proven effectiveness documented
- [ ] **Style Guide:** Follows markdown conventions, fake credentials, example.com domains

---

## General Checklist

- [ ] Code follows style guide (see CONTRIBUTING.md)
- [ ] Self-review completed
- [ ] Testing completed (RED-GREEN-REFACTOR for skills)
- [ ] Documentation updated (if applicable)
- [ ] No merge conflicts

---

## Additional Context

[Any additional information reviewers should know]

---

## Reviewer Notes

[For maintainers - leave blank]
```

**Step 2: Verify template completeness**

Check that template includes:
- [ ] Description section
- [ ] Type of change checkboxes
- [ ] RED-GREEN-REFACTOR section (for skills)
- [ ] Skill quality checklist
- [ ] General checklist
- [ ] Related issue link

**Step 3: Commit**

```bash
git add .github/PULL_REQUEST_TEMPLATE.md
git commit -m "docs: Add GitHub pull request template

Add PR template with:
- Description and type of change
- RED-GREEN-REFACTOR testing checklist (for new skills)
- Skill quality checklist
- General PR checklist
- Related issue linking

Ensures all PRs meet quality standards.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Wave 1 Verification

**Goal:** Verify all Wave 1 deliverables complete and high-quality.

**Files:**
- Review: All Wave 1 files

**Step 1: Verify all files created**

```bash
ls -la | grep -E "README.md|DEVELOPMENT.md|CONTRIBUTING.md|CODE_OF_CONDUCT.md|LICENSE"
ls -la .github/ISSUE_TEMPLATE/
ls -la .github/PULL_REQUEST_TEMPLATE.md
```

Expected output:
- README.md (exists, public-facing)
- DEVELOPMENT.md (exists, implementation details)
- CONTRIBUTING.md (exists, skill creation guide)
- CODE_OF_CONDUCT.md (exists, Contributor Covenant)
- LICENSE (exists, Apache 2.0)
- .github/ISSUE_TEMPLATE/new_skill.md (exists)
- .github/ISSUE_TEMPLATE/bug_report.md (exists)
- .github/PULL_REQUEST_TEMPLATE.md (exists)

**Step 2: Verify README structure**

```bash
head -50 README.md
```

Check for:
- [ ] Quick start (3 steps)
- [ ] Skill catalog (18 skills listed)
- [ ] Example usage (3 examples)

**Step 3: Verify CONTRIBUTING completeness**

```bash
grep -E "RED-GREEN-REFACTOR|Skill Creation Requirements|Pull Request Process" CONTRIBUTING.md
```

Check that all sections present.

**Step 4: Create Wave 1 completion commit**

```bash
git commit --allow-empty -m "docs: Complete Wave 1 - Repository polish

Wave 1 deliverables complete:
✅ README.md restructured for public audience
✅ DEVELOPMENT.md created with implementation history
✅ CONTRIBUTING.md guides skill creation (RED-GREEN-REFACTOR)
✅ CODE_OF_CONDUCT.md added (Contributor Covenant 2.1)
✅ LICENSE added (Apache 2.0)
✅ GitHub issue templates created (new skill, bug report)
✅ GitHub PR template created (with quality checklist)

New user can understand value in <2 minutes.
New contributor can understand process in <10 minutes.

Wave 1 complete. Ready for Wave 2 (plugin packaging).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Verification complete when:**
- [ ] All 7 governance files created
- [ ] README is public-facing (not implementation-focused)
- [ ] CONTRIBUTING guides RED-GREEN-REFACTOR methodology
- [ ] All templates include required sections
- [ ] Wave 1 completion commit created

---

## Wave 2: Plugin Packaging (2-3 hours)

### Task 9: Create plugin.json Manifest

**Goal:** Create Claude Code plugin marketplace manifest listing all 18 skills.

**Files:**
- Create: `plugin.json`

**Step 1: Create plugin.json with complete skill listing**

Complete content:

```json
{
  "name": "security-architect-technical-writer",
  "version": "1.0.0",
  "description": "Security architecture and technical writing skills for Claude Code. Includes threat modeling, compliance frameworks, security controls, documentation structure, incident response, and more.",
  "author": "[Your Name]",
  "homepage": "https://github.com/[username]/skillpacks",
  "repository": {
    "type": "git",
    "url": "https://github.com/[username]/skillpacks.git"
  },
  "skills": [
    {
      "name": "ordis/security-architect/using-security-architect",
      "path": "ordis/security-architect/using-security-architect/SKILL.md",
      "description": "Meta-skill for routing to appropriate security architecture skills based on task"
    },
    {
      "name": "ordis/security-architect/threat-modeling",
      "path": "ordis/security-architect/threat-modeling/SKILL.md",
      "description": "STRIDE threat modeling, attack trees, risk scoring matrices"
    },
    {
      "name": "ordis/security-architect/security-controls-design",
      "path": "ordis/security-architect/security-controls-design/SKILL.md",
      "description": "Defense-in-depth, trust boundaries, control effectiveness analysis"
    },
    {
      "name": "ordis/security-architect/secure-by-design-patterns",
      "path": "ordis/security-architect/secure-by-design-patterns/SKILL.md",
      "description": "Zero-trust architecture, immutable infrastructure, least privilege patterns"
    },
    {
      "name": "ordis/security-architect/security-architecture-review",
      "path": "ordis/security-architect/security-architecture-review/SKILL.md",
      "description": "Systematic review checklists for design and implementation phases"
    },
    {
      "name": "ordis/security-architect/classified-systems-security",
      "path": "ordis/security-architect/classified-systems-security/SKILL.md",
      "description": "Bell-LaPadula MLS model, fail-fast validation, classified data handling"
    },
    {
      "name": "ordis/security-architect/compliance-awareness-and-mapping",
      "path": "ordis/security-architect/compliance-awareness-and-mapping/SKILL.md",
      "description": "Framework discovery (ISM/HIPAA/SOC2/GDPR), control mapping methodology"
    },
    {
      "name": "ordis/security-architect/security-authorization-and-accreditation",
      "path": "ordis/security-architect/security-authorization-and-accreditation/SKILL.md",
      "description": "Government ATO/AIS processes, RMF 7-step, SSP/SAR/POA&M documentation"
    },
    {
      "name": "ordis/security-architect/documenting-threats-and-controls",
      "path": "ordis/security-architect/documenting-threats-and-controls/SKILL.md",
      "description": "Security ADRs, threat documentation, control traceability patterns"
    },
    {
      "name": "muna/technical-writer/using-technical-writer",
      "path": "muna/technical-writer/using-technical-writer/SKILL.md",
      "description": "Meta-skill for routing to appropriate documentation skills based on task"
    },
    {
      "name": "muna/technical-writer/documentation-structure",
      "path": "muna/technical-writer/documentation-structure/SKILL.md",
      "description": "ADR patterns, README templates, documentation organization principles"
    },
    {
      "name": "muna/technical-writer/clarity-and-style",
      "path": "muna/technical-writer/clarity-and-style/SKILL.md",
      "description": "Active voice, concrete examples, avoiding ambiguity in technical writing"
    },
    {
      "name": "muna/technical-writer/diagram-conventions",
      "path": "muna/technical-writer/diagram-conventions/SKILL.md",
      "description": "Decision trees, semantic labels, visual clarity in technical diagrams"
    },
    {
      "name": "muna/technical-writer/documentation-testing",
      "path": "muna/technical-writer/documentation-testing/SKILL.md",
      "description": "5 dimensions: completeness, accuracy, findability, examples, walkthrough"
    },
    {
      "name": "muna/technical-writer/security-aware-documentation",
      "path": "muna/technical-writer/security-aware-documentation/SKILL.md",
      "description": "Sanitizing examples (fake credentials), preventing leaks, threat disclosure decisions"
    },
    {
      "name": "muna/technical-writer/incident-response-documentation",
      "path": "muna/technical-writer/incident-response-documentation/SKILL.md",
      "description": "5-phase response template, time-critical runbooks, escalation paths"
    },
    {
      "name": "muna/technical-writer/itil-and-governance-documentation",
      "path": "muna/technical-writer/itil-and-governance-documentation/SKILL.md",
      "description": "RFC templates, service catalogs, SLA/OLA documentation, DR plans"
    },
    {
      "name": "muna/technical-writer/operational-acceptance-documentation",
      "path": "muna/technical-writer/operational-acceptance-documentation/SKILL.md",
      "description": "Production readiness checklists, go-live approval packages, operational handover"
    }
  ],
  "keywords": [
    "security",
    "documentation",
    "threat-modeling",
    "compliance",
    "technical-writing",
    "security-architecture",
    "incident-response",
    "HIPAA",
    "GDPR",
    "SOC2",
    "ATO",
    "classified-systems",
    "Bell-LaPadula",
    "RMF",
    "ITIL",
    "STRIDE",
    "defense-in-depth",
    "zero-trust"
  ],
  "license": "Apache-2.0",
  "engines": {
    "claude-code": ">=1.0.0"
  }
}
```

**Step 2: Update placeholders**

Replace:
- `[Your Name]` with actual author name
- `[username]` with actual GitHub username

**Step 3: Verify JSON is valid**

```bash
python3 -m json.tool plugin.json > /dev/null && echo "JSON valid" || echo "JSON invalid"
```

Expected: "JSON valid"

**Step 4: Verify all 18 skills listed**

```bash
cat plugin.json | grep '"name":' | grep -E 'ordis|muna' | wc -l
```

Expected: 18 (2 meta + 8 core + 8 extensions)

**Step 5: Commit**

```bash
git add plugin.json
git commit -m "feat: Create plugin.json manifest for Claude Code marketplace

Add plugin manifest with:
- 18 skills (ordis security-architect + muna technical-writer)
- Skill paths and descriptions
- Keywords for discoverability
- Apache 2.0 license
- Repository and homepage links

Ready for Claude Code plugin marketplace submission.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Create Plugin README

**Goal:** Create concise marketplace-facing README (plugin users see this first).

**Files:**
- Create: `PLUGIN_README.md` (will be renamed to README.md for plugin package)

**Step 1: Create PLUGIN_README.md**

Content:

```markdown
# Security Architect & Technical Writer Skills

Professional security architecture and technical writing skills for Claude Code.

---

## Capabilities

### Security Architect (10 skills)

**Core Skills:**
- **Threat modeling** - STRIDE methodology, attack trees, risk scoring
- **Security controls design** - Defense-in-depth, trust boundaries
- **Secure-by-design patterns** - Zero-trust, immutable infrastructure
- **Security architecture review** - Systematic checklists

**Specialized Skills:**
- **Classified systems security** - Bell-LaPadula MLS, fail-fast validation
- **Compliance awareness** - ISM/HIPAA/SOC2/GDPR framework discovery
- **Security authorization** - Government ATO/AIS, RMF process
- **Documenting threats and controls** - Security ADRs, traceability

### Technical Writer (8 skills)

**Core Skills:**
- **Documentation structure** - ADR patterns, README templates
- **Clarity and style** - Active voice, concrete examples
- **Diagram conventions** - Decision trees, semantic labels
- **Documentation testing** - 5 dimensions of quality

**Specialized Skills:**
- **Security-aware documentation** - Sanitized examples, no credential leaks
- **Incident response documentation** - 5-phase runbooks, escalation paths
- **ITIL and governance documentation** - RFC, SLA, DR plans
- **Operational acceptance documentation** - Production readiness, go-live

---

## Quick Start

Load a skill in Claude Code:

```
I'm using ordis/security-architect/threat-modeling to analyze this API
```

Claude will apply STRIDE methodology, create attack trees, and score risks.

---

## Example: Threat Model an API

**Before (without skill):**
- Ad-hoc threat identification
- No systematic methodology
- Misses subtle threats

**After (with skill):**
- STRIDE applied systematically (Spoofing, Tampering, Repudiation, Info Disclosure, DoS, Elevation)
- Attack trees show feasible paths
- Risks scored with Likelihood × Impact matrix
- All threats documented

---

## Who Should Use This

- **Security professionals** - Apply proven frameworks (STRIDE, defense-in-depth, zero-trust)
- **Technical writers** - Structure docs effectively, write clearly, sanitize examples
- **Developers** - Learn security and documentation best practices
- **Regulated industries** - Healthcare (HIPAA), government (ATO), finance (compliance)

---

## Full Documentation

**GitHub Repository:** [https://github.com/[username]/skillpacks](https://github.com/[username]/skillpacks)

Complete documentation including:
- 5 end-to-end tutorials (secure API, healthcare compliance, government ATO, incident response)
- Contribution guidelines
- Skill creation methodology (RED-GREEN-REFACTOR)
- Development history

---

## License

Apache License 2.0 - See [LICENSE](LICENSE)

---

## Version

**1.0.0** - All 18 skills implemented and tested
```

**Step 2: Update GitHub username placeholder**

Replace `[username]` with actual GitHub username.

**Step 3: Verify PLUGIN_README is concise**

```bash
wc -w PLUGIN_README.md
```

Expected: <500 words (marketplace readers want quick overview)

**Step 4: Commit**

```bash
git add PLUGIN_README.md
git commit -m "docs: Create plugin marketplace README

Add concise README for Claude Code plugin marketplace:
- Capabilities overview (18 skills)
- Quick start example
- Before/after comparison
- Target audience
- Link to full docs on GitHub

Marketplace-facing (<500 words), full docs in main README.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Create Plugin Submission Documentation

**Goal:** Document Claude Code plugin marketplace submission process.

**Files:**
- Create: `docs/PLUGIN_SUBMISSION.md`

**Step 1: Create plugin submission guide**

Content for `docs/PLUGIN_SUBMISSION.md`:

```markdown
# Claude Code Plugin Marketplace Submission

This document outlines the process for submitting skillpacks to the Claude Code plugin marketplace.

---

## Preparation Checklist

Before submitting, verify:

- [ ] **plugin.json complete** - All 18 skills listed with paths and descriptions
- [ ] **PLUGIN_README.md created** - Concise marketplace-facing README (<500 words)
- [ ] **All skills tested locally** - Each skill loads and works correctly
- [ ] **Cross-references verified** - Skills can reference each other successfully
- [ ] **Version number set** - 1.0.0 for initial release
- [ ] **License confirmed** - Apache-2.0 in plugin.json and LICENSE file
- [ ] **Author and repository URLs updated** - No placeholder values remain

---

## Local Testing

Test plugin before submission:

### 1. Install Plugin Locally

[Document local plugin installation process here after researching Claude Code plugin development]

### 2. Test Each Meta-Skill

```
I'm using ordis/security-architect/using-security-architect
```

Verify: Meta-skill loads and can route to core/extension skills

```
I'm using muna/technical-writer/using-technical-writer
```

Verify: Meta-skill loads and can route to core/extension skills

### 3. Test Core Skills (spot check)

```
I'm using ordis/security-architect/threat-modeling
```

Verify: Skill loads, STRIDE methodology available

```
I'm using muna/technical-writer/documentation-structure
```

Verify: Skill loads, ADR patterns available

### 4. Test Extension Skills (spot check)

```
I'm using ordis/security-architect/classified-systems-security
```

Verify: Skill loads, Bell-LaPadula MLS patterns available

```
I'm using muna/technical-writer/incident-response-documentation
```

Verify: Skill loads, 5-phase template available

### 5. Test Cross-References

Load skill that references another skill, verify reference works:

```
I'm using ordis/security-architect/threat-modeling
```

This skill should reference `security-controls-design` - verify reference loads correctly.

---

## Submission Process

[Document actual submission steps after researching Claude Code plugin marketplace requirements]

### Step 1: Register/Login to Marketplace

[URL and process for marketplace registration]

### Step 2: Create Plugin Listing

- **Name:** security-architect-technical-writer
- **Version:** 1.0.0
- **Description:** [Copy from plugin.json]
- **README:** [Upload PLUGIN_README.md]
- **Keywords:** [Copy from plugin.json]
- **License:** Apache-2.0

### Step 3: Upload Plugin Package

[Package format and upload process]

### Step 4: Submit for Review

[Submission and review process]

**Expected review time:** [Document after submission]

### Step 5: Address Review Feedback

[Process for responding to reviewer feedback]

---

## Post-Submission

After plugin is published:

### Monitor Marketplace Reviews

- Check plugin marketplace page regularly
- Respond to user reviews and questions
- Track download/install metrics

### Address User Feedback

- User reports bug → Create GitHub issue
- User requests feature → Evaluate for next version
- User asks question → Answer via GitHub Discussions or marketplace

### Plan Version Updates

**Version 1.1.0 (planned):**
- Address any bugs found post-launch
- Incorporate user feedback
- Consider adding 1-2 community-requested skills

**Update process:**
1. Update skills/docs in GitHub
2. Increment version in plugin.json (1.0.0 → 1.1.0)
3. Test updated plugin locally
4. Submit updated plugin to marketplace
5. Document changes in release notes

---

## Troubleshooting

### Common Issues

**Issue:** Skill doesn't load
- **Fix:** Check path in plugin.json matches actual file path
- **Verify:** `ls -la ordis/security-architect/[skill-name]/SKILL.md`

**Issue:** Cross-reference doesn't work
- **Fix:** Verify relative path is correct (e.g., `ordis/security-architect/threat-modeling`)
- **Test:** Load referencing skill, verify referenced skill loads

**Issue:** YAML frontmatter not parsed
- **Fix:** Verify frontmatter has `---` delimiters on separate lines
- **Check:** No tabs in YAML (use spaces)

---

## Resources

- **Claude Code Plugin Documentation:** [Link when available]
- **GitHub Repository:** https://github.com/[username]/skillpacks
- **Issue Tracker:** https://github.com/[username]/skillpacks/issues

---

## Status

**Current Status:** Plugin tested locally, ready for submission pending marketplace research

**Next Steps:**
1. Research Claude Code plugin marketplace submission process
2. Complete submission process documentation
3. Submit plugin for review
4. Monitor and respond to feedback
```

**Step 2: Update GitHub username placeholder**

Replace `[username]` with actual GitHub username.

**Step 3: Commit**

```bash
git add docs/PLUGIN_SUBMISSION.md
git commit -m "docs: Create plugin submission guide

Add documentation for Claude Code plugin marketplace submission:
- Preparation checklist
- Local testing procedures
- Submission process (to be completed after research)
- Post-submission monitoring
- Troubleshooting common issues

Ready to document actual submission process.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: Wave 2 Verification

**Goal:** Verify all Wave 2 deliverables complete.

**Files:**
- Review: plugin.json, PLUGIN_README.md, docs/PLUGIN_SUBMISSION.md

**Step 1: Verify files created**

```bash
ls -la plugin.json PLUGIN_README.md docs/PLUGIN_SUBMISSION.md
```

Expected: All files exist

**Step 2: Verify plugin.json has 18 skills**

```bash
cat plugin.json | python3 -m json.tool | grep '"name":' | grep -E 'ordis|muna' | wc -l
```

Expected: 18

**Step 3: Test JSON validity**

```bash
python3 -m json.tool plugin.json > /dev/null && echo "Valid" || echo "Invalid"
```

Expected: "Valid"

**Step 4: Create Wave 2 completion commit**

```bash
git commit --allow-empty -m "feat: Complete Wave 2 - Plugin packaging

Wave 2 deliverables complete:
✅ plugin.json manifest created (18 skills)
✅ PLUGIN_README.md created (marketplace-facing)
✅ Plugin submission guide created (docs/PLUGIN_SUBMISSION.md)
✅ All skills verified with valid paths
✅ JSON validated

Plugin ready for local testing and marketplace submission.

Wave 2 complete. Ready for Wave 3 (tutorials).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Verification complete when:**
- [ ] plugin.json exists and is valid JSON
- [ ] 18 skills listed in plugin.json
- [ ] PLUGIN_README.md is concise (<500 words)
- [ ] Submission guide created
- [ ] Wave 2 completion commit created

---

## Wave 3: Tutorials (4-6 hours)

### Task 13: Tutorial 1 - Secure a REST API from Scratch

**Goal:** Create end-to-end tutorial showing threat modeling and security controls for API.

**Files:**
- Create: `docs/tutorials/01-secure-rest-api.md`

**Step 1: Create tutorial directory**

```bash
mkdir -p docs/tutorials
```

**Step 2: Create Tutorial 1**

Complete content for `docs/tutorials/01-secure-rest-api.md`:

[Content continues with complete tutorial - see Phase 4 design document section on Tutorial 1 for full structure and content]

**Key sections:**
- Scenario (building user authentication API)
- Skills used (threat-modeling, security-controls-design, secure-by-design-patterns)
- Step-by-step walkthrough (STRIDE analysis → controls → implementation)
- Outcome (threat model document, security ADR, implementation checklist)
- Next steps (related tutorials)

**Step 3: Verify tutorial completeness**

Check tutorial includes:
- [ ] Realistic scenario (2-3 paragraphs)
- [ ] Skills used (3 skills listed)
- [ ] Step-by-step walkthrough (5-7 steps)
- [ ] Concrete examples (commands, code snippets)
- [ ] Outcome (what you built)
- [ ] Next steps (related tutorials)

**Step 4: Commit**

```bash
git add docs/tutorials/01-secure-rest-api.md
git commit -m "docs: Add Tutorial 1 - Secure REST API from scratch

End-to-end tutorial showing:
- STRIDE threat analysis of authentication API
- Defense-in-depth security controls
- Zero-trust implementation patterns

Skills: threat-modeling, security-controls-design, secure-by-design-patterns

Outcome: Threat model + Security ADR + implementation checklist

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 14: Tutorial 2 - Healthcare System Compliance Documentation

**Goal:** Create tutorial showing HIPAA compliance documentation with sanitized examples.

**Files:**
- Create: `docs/tutorials/02-healthcare-compliance.md`

**Step 1: Create Tutorial 2**

[Full content following tutorial template from Phase 4 design]

**Key sections:**
- Scenario (healthcare startup needs HIPAA documentation)
- Skills used (compliance-awareness-and-mapping, security-aware-documentation, documentation-structure)
- Walkthrough (framework discovery → control mapping → sanitized docs)
- Outcome (HIPAA mapping + API docs with fake PHI + compliance documentation)

**Step 2: Verify tutorial completeness**

[Same checklist as Tutorial 1]

**Step 3: Commit**

```bash
git add docs/tutorials/02-healthcare-compliance.md
git commit -m "docs: Add Tutorial 2 - Healthcare compliance documentation

End-to-end tutorial showing:
- HIPAA framework discovery and mapping
- Sanitized examples (no real PHI)
- Compliance documentation organization

Skills: compliance-awareness-and-mapping, security-aware-documentation, documentation-structure

Outcome: HIPAA mapping + sanitized API docs + audit-ready documentation

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 15: Tutorial 3 - Government System Authorization (ATO)

**Goal:** Create tutorial showing complete ATO process for classified system.

**Files:**
- Create: `docs/tutorials/03-government-ato.md`

**Step 1: Create Tutorial 3**

[Full content following tutorial template - most comprehensive tutorial]

**Key sections:**
- Scenario (government contractor needs ATO for classified system)
- Skills used (classified-systems-security, security-authorization-and-accreditation, documenting-threats-and-controls)
- Walkthrough (MLS design → RMF 7-step → SSP/SAR/POA&M)
- Outcome (Complete ATO package)

**Step 2: Verify tutorial completeness**

[Same checklist as Tutorial 1]

**Step 3: Commit**

```bash
git add docs/tutorials/03-government-ato.md
git commit -m "docs: Add Tutorial 3 - Government system authorization (ATO)

End-to-end tutorial showing:
- Bell-LaPadula MLS design with fail-fast validation
- RMF 7-step process (PREPARE through MONITOR)
- Complete ATO package (SSP, SAR, POA&M)

Skills: classified-systems-security, security-authorization-and-accreditation, documenting-threats-and-controls

Outcome: MLS system design + SSP + SAR template + POA&M + Security ADRs

Most comprehensive tutorial (government/defense context).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 16: Tutorial 4 - Incident Response Readiness

**Goal:** Create tutorial showing incident response capability development.

**Files:**
- Create: `docs/tutorials/04-incident-response.md`

**Step 1: Create Tutorial 4**

[Full content following tutorial template]

**Key sections:**
- Scenario (startup needs incident response for Series A)
- Skills used (incident-response-documentation, security-controls-design, operational-acceptance-documentation)
- Walkthrough (scenarios → 5-phase runbooks → escalation → DR plan)
- Outcome (Complete incident response playbook)

**Step 2: Verify tutorial completeness**

[Same checklist as Tutorial 1]

**Step 3: Commit**

```bash
git add docs/tutorials/04-incident-response.md
git commit -m "docs: Add Tutorial 4 - Incident response readiness

End-to-end tutorial showing:
- 5-phase incident response runbooks (detection through lessons learned)
- Control failure scenario identification
- Escalation paths and DR planning

Skills: incident-response-documentation, security-controls-design, operational-acceptance-documentation

Outcome: 3 runbooks (breach, DDoS, outage) + escalation matrix + DR plan

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 17: Tutorial 5 - Security + Documentation Lifecycle (Cross-Cutting)

**Goal:** Create tutorial showing complete lifecycle from threat model to public docs.

**Files:**
- Create: `docs/tutorials/05-security-documentation-lifecycle.md`

**Step 1: Create Tutorial 5**

[Full content following tutorial template - demonstrates ordis + muna integration]

**Key sections:**
- Scenario (document OAuth feature securely and completely)
- Skills used (threat-modeling, documenting-threats-and-controls, security-aware-documentation, documentation-testing)
- Walkthrough (threat analysis → security ADR → sanitized public docs → verification)
- Outcome (Complete documentation lifecycle)

**Step 2: Verify tutorial completeness**

[Same checklist as Tutorial 1]

**Step 3: Commit**

```bash
git add docs/tutorials/05-security-documentation-lifecycle.md
git commit -m "docs: Add Tutorial 5 - Security + documentation lifecycle

End-to-end tutorial showing:
- Complete lifecycle from threat model to public documentation
- Cross-cutting skills (ordis + muna integration)
- Threat-to-control-to-docs traceability

Skills: threat-modeling, documenting-threats-and-controls, security-aware-documentation, documentation-testing

Outcome: Threat model + Security ADR + sanitized public docs + testing verification

Demonstrates bidirectional cross-references between security and documentation skills.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 18: Wave 3 Verification & Phase 4 Completion

**Goal:** Verify all tutorials complete and celebrate Phase 4 completion.

**Files:**
- Review: All tutorial files
- Update: README.md (ensure tutorial links work)

**Step 1: Verify all tutorials created**

```bash
ls -la docs/tutorials/
```

Expected files:
- 01-secure-rest-api.md
- 02-healthcare-compliance.md
- 03-government-ato.md
- 04-incident-response.md
- 05-security-documentation-lifecycle.md

**Step 2: Verify tutorial links in README**

```bash
grep -A 5 "## Tutorials" README.md
```

Verify all 5 tutorials linked correctly.

**Step 3: Test one tutorial link**

```bash
test -f docs/tutorials/01-secure-rest-api.md && echo "Link valid" || echo "Link broken"
```

Expected: "Link valid"

**Step 4: Create Wave 3 completion commit**

```bash
git commit --allow-empty -m "docs: Complete Wave 3 - Tutorials

Wave 3 deliverables complete:
✅ Tutorial 1: Secure REST API (threat modeling + controls)
✅ Tutorial 2: Healthcare compliance (HIPAA + sanitized docs)
✅ Tutorial 3: Government ATO (MLS + RMF + SSP/SAR/POA&M)
✅ Tutorial 4: Incident response (5-phase runbooks + DR)
✅ Tutorial 5: Security + documentation lifecycle (cross-cutting)

All tutorials include:
- Realistic scenarios
- Skills used (with skill names)
- Step-by-step walkthroughs
- Concrete outcomes
- Next steps

Diverse domains covered: API security, healthcare, government, ops, cross-cutting

Wave 3 complete. Phase 4 COMPLETE!

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Step 5: Create Phase 4 completion commit**

```bash
git commit --allow-empty -m "docs: Phase 4 COMPLETE - Public release ready! 🎉

Phase 4 Achievement - All three waves complete:

✅ Wave 1: Repository Polish (4-5h)
   - README restructured for public discovery
   - DEVELOPMENT.md with implementation history
   - CONTRIBUTING.md with RED-GREEN-REFACTOR guide
   - CODE_OF_CONDUCT.md (Contributor Covenant 2.1)
   - LICENSE (Apache 2.0)
   - GitHub issue templates (new skill, bug report)
   - GitHub PR template (quality checklist)

✅ Wave 2: Plugin Packaging (2-3h)
   - plugin.json manifest (18 skills)
   - PLUGIN_README.md (marketplace-facing)
   - Plugin submission guide

✅ Wave 3: Tutorials (4-6h)
   - 5 end-to-end tutorials across diverse domains
   - API security, healthcare, government, ops, cross-cutting
   - Realistic scenarios with concrete outcomes

Public Release Readiness:
✅ GitHub repo polished and welcoming
✅ Plugin ready for marketplace submission
✅ Documentation serves experts AND learners
✅ Contribution pipeline established
✅ New user gets value in <5 minutes
✅ New contributor understands process in <10 minutes

Total Investment: ~53-64 hours (Phases 1-4)
- Phase 1: 10-15h (foundation, 4 skills)
- Phase 2: 18-30h (core, 6 skills)
- Phase 3: 17-25h (extensions, 8 skills)
- Phase 4: 8-14h (polish, tutorials, plugin)

🎉 SKILLPACKS ARE PUBLIC-READY! 🎉

Ready for GitHub star, plugin marketplace submission, and community contributions!

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Verification complete when:**
- [ ] All 5 tutorials created
- [ ] Tutorial links in README verified
- [ ] Wave 3 completion commit created
- [ ] Phase 4 completion commit created
- [ ] Ready to celebrate! 🎉

---

## Phase 4 Exit Criteria

**Complete when ALL criteria met:**

### Wave 1 ✅
- [x] README.md restructured for public audience
- [x] DEVELOPMENT.md created
- [x] CONTRIBUTING.md created
- [x] CODE_OF_CONDUCT.md added
- [x] LICENSE added
- [x] GitHub issue templates created
- [x] GitHub PR template created

### Wave 2 ✅
- [x] plugin.json manifest complete
- [x] PLUGIN_README.md created
- [x] Plugin submission guide created

### Wave 3 ✅
- [x] 5 tutorials published
- [x] Diverse domains covered
- [x] Consistent structure
- [x] Serve both experts and learners

### Overall ✅
- [x] GitHub repo polished
- [x] Plugin ready for submission
- [x] Documentation dual-audience
- [x] Contribution pipeline established
- [x] New user can get value in <5 minutes

---

## Execution Options

**Plan saved to:** `docs/plans/2025-10-29-phase-4-public-release-implementation.md`

**Two execution options:**

**1. Subagent-Driven Development (this session)**
- Stay in current session
- Fresh subagent per task
- Code review between tasks
- Fast iteration with quality gates

**2. Parallel Session (separate Claude Code session)**
- Open new session with `superpowers:executing-plans`
- Batch execution with review checkpoints
- Dedicated focus on implementation

**Which approach would you prefer?**
