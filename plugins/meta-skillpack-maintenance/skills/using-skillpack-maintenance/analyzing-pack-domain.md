# Analyzing Pack Domain

**Purpose:** Investigative process to establish "what this pack should cover" from first principles.

## Adaptive Investigation Workflow

**Sequence: D → B → C → A (conditional)**

### Phase D: User-Guided Scope

**Ask user:**
- "What is the intended scope and purpose of [pack-name]?"
- "What boundaries should this pack respect?"
- "Who is the target audience? (beginners / practitioners / experts)"
- "What depth of coverage is expected? (overview / comprehensive / exhaustive)"

**Document:**
- User's vision as baseline
- Explicit boundaries (what's IN scope, what's OUT of scope)
- Success criteria (what makes this pack "complete"?)

### Phase B: LLM Knowledge-Based Analysis

**Leverage model knowledge to map the domain:**

1. **Generate comprehensive coverage map:**
   - What are the major concepts/algorithms/techniques in this domain?
   - What are the core patterns practitioners need to know?
   - What are common implementation challenges?

2. **Identify structure:**
   - Foundational concepts (must understand first)
   - Core techniques (bread-and-butter patterns)
   - Advanced topics (expert-level material)
   - Cross-cutting concerns (testing, debugging, optimization)

3. **Flag research currency:**
   - Is this domain stable or rapidly evolving?
   - Stable examples: Design patterns, basic algorithms, established protocols
   - Evolving examples: AI/ML, security, modern web frameworks
   - If evolving → Flag for Phase A research

**Output:** Coverage map with categorization (foundational/core/advanced)

### Phase C: Existing Pack Audit

**Read all current skills in the pack:**

1. **Inventory:**
   - List all SKILL.md files
   - Note skill names and descriptions
   - Check router skill (if exists) for specialist list

2. **Compare against coverage map:**
   - What's covered? (existing skills matching coverage areas)
   - What's missing? (gaps in foundational/core/advanced areas)
   - What overlaps? (multiple skills covering same concept)
   - What's obsolete? (outdated approaches, deprecated patterns)

3. **Quality check:**
   - Are descriptions accurate?
   - Do skills match their descriptions?
   - Are there broken cross-references?

**Output:** Gap list, duplicate list, obsolescence flags

### Phase A: Research (Conditional)

**ONLY if Phase B flagged domain as rapidly evolving.**

**Research authoritative sources:**

1. **For AI/ML domains:**
   - Latest survey papers (search: "[domain] survey 2024/2025")
   - Current textbooks (check publication dates)
   - Official library documentation (PyTorch, TensorFlow, etc.)
   - Research benchmarks (Papers with Code, etc.)

2. **For security domains:**
   - OWASP guidelines
   - NIST standards
   - Recent CVE patterns
   - Current threat landscape

3. **For framework domains:**
   - Official documentation (latest version)
   - Migration guides (breaking changes)
   - Best practices (official recommendations)

**Update coverage map:**
- Add new techniques/patterns
- Flag deprecated approaches in existing skills
- Note version-specific considerations

**Decision criteria for Phase A:**
- **Skip research** for: Math, algorithms, design patterns, established protocols
- **Run research** for: AI/ML, security, modern frameworks, evolving standards

## Outputs

Generate comprehensive report:

### 1. Domain Coverage Map

```
Foundational:
- [Concept 1] - [Status: Exists / Missing / Needs enhancement]
- [Concept 2] - [Status: Exists / Missing / Needs enhancement]

Core Techniques:
- [Technique 1] - [Status: ...]
- [Technique 2] - [Status: ...]

Advanced Topics:
- [Topic 1] - [Status: ...]
- [Topic 2] - [Status: ...]

Cross-Cutting:
- [Concern 1] - [Status: ...]
```

### 2. Current State Assessment

```
Existing skills: [count]
- [Skill 1 name] - covers [domain area]
- [Skill 2 name] - covers [domain area]
...
```

### 3. Gap Analysis

```
Missing (High Priority):
- [Gap 1] - foundational concept not covered
- [Gap 2] - core technique missing

Missing (Medium Priority):
- [Gap 3] - advanced topic not covered

Duplicates:
- [Skill A] and [Skill B] overlap on [topic]

Obsolete:
- [Skill C] uses deprecated approach [old pattern]
```

### 4. Research Currency Flag

```
Domain stability: [Stable / Evolving]
Research conducted: [Yes / No / Not needed]
Currency concerns: [None / List specific areas needing updates]
```

## Proceeding to Next Stage

After completing investigation, hand off to `reviewing-pack-structure.md` for scorecard generation.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Skipping user input (Phase D) | Always start with user vision - they define scope |
| Over-relying on LLM knowledge | For evolving domains, run research (Phase A) |
| Skipping gap analysis | Compare coverage map vs. existing skills systematically |
| Treating all domains as stable | Flag AI/ML/security/frameworks for research |
| Vague gap descriptions | Be specific: "Missing TaskGroup patterns" not "async needs work" |
