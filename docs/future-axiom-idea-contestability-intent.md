# Axiom/Idea-Contestability - Intent & Design Document

**Created**: 2025-10-29
**Status**: Intent Document (Urgent - High Priority)
**Faction**: Axiom (Creators of Marvels - Process, Methodology, Infrastructure)
**Priority**: URGENT (Addresses Critical Human-AI Interaction Flaw)
**Grounded In**: Formal government contestability practice (red-teaming proposals)

---

## The Contestability Framework (Professional Foundation)

**This pack is grounded in formal government contestability practice:**

In government organizations, a "contestability" team reviews proposals from other departments before senior board approval. Their job:

1. **Write companion papers** identifying:
   - What they didn't think of
   - What they haven't done
   - Where intellectual rigor is lacking
   - Example: "A chain of air conditioner shops in Antarctica lacks appropriate rigor"

2. **If approved**, help them execute:
   - Project management: Cost, risk, schedule
   - Capability building: What they need to learn/acquire
   - Resource planning: What's required vs available

**This pack applies that professional practice to human-AI collaboration.**

---

## The Problem

**Humanity has a cognitive flaw that LLMs dangerously amplify:**

LLMs can explain ANY concept in simple, accessible terms. This creates an illusion of understanding that leads humans to:

1. **Attempt projects beyond their capability** - "Claude can explain quantum computing, so I can build a quantum algorithm"
2. **Develop crackpot theories with AI validation** - Prompting until LLM agrees, mistaking eloquent explanation for correctness
3. **Skip foundational learning** - "I don't need to learn X, Claude knows it"
4. **Massively overscope** - "Everything seems easy when explained simply"
5. **Dunning-Kruger on steroids** - Confusing ability to discuss a topic with ability to work in it

**Core insight**: "Unlimited access to information dumbed down to our level" means humans can now get PhD-level explanations in 5th-grade language, creating false confidence in their ability to tackle PhD-level problems.

---

## Why This Is URGENT

**Before LLMs**: Humans hit natural barriers when attempting things beyond their capability
- Can't understand academic papers → realize they're not ready
- Can't grasp technical docs → recognize knowledge gap
- Complex explanations → natural difficulty feedback

**With LLMs**: All barriers feel surmountable
- LLM explains paper simply → feels understood
- LLM translates technical concepts → feels accessible
- LLM provides encouraging responses → feels validated

**Result**: Humans waste months/years on projects they're fundamentally not ready for, never building the foundations needed for success.

---

## What This Skill Should DO

**Primary Function**: Help LLMs recognize and address human capability overreach in real-time.

### Detection Patterns (When to Activate)

1. **Scope-Reality Mismatch**: Project scope vastly exceeds demonstrated capability
   - "I want to build a secure distributed database" (has never written a database)
   - "Let's create a game engine from scratch" (has never made a game)

2. **Foundation Gaps**: Missing prerequisite knowledge for stated goal
   - Wants to implement neural architecture search (doesn't know backpropagation)
   - Plans to build compiler (hasn't parsed a CSV file)

3. **Theory-Practice Disconnect**: Can discuss concepts but can't apply them
   - Eloquently explains Byzantine fault tolerance (can't implement basic consensus)
   - Describes game balance theory (has never balanced a single mechanic)

4. **Crackpot Theory Red Flags**: Using LLM to validate questionable ideas
   - Repeatedly rephrasing until LLM agrees
   - Ignoring established theory/practice
   - "Everyone is wrong except me" + LLM validation

5. **Tutorial Hell**: Infinite learning, zero building
   - 6 months of "learning" with no shipped projects
   - Can explain everything, has built nothing
   - "Just one more tutorial before I start"

### Contestability Assessment Framework

**Two-phase process mirroring government practice:**

#### Phase 1: CHALLENGE (The Contestability Paper)
Write formal assessment identifying gaps and evaluating rigor.

**Structured template:**
```markdown
# CONTESTABILITY ASSESSMENT: [Idea Name]

## EXECUTIVE SUMMARY
[One paragraph: Viable? Key concerns? Recommendation?]

## PROPOSAL OVERVIEW
[What they're actually proposing]

## INTELLECTUAL RIGOR ASSESSMENT
### Assumptions Analysis
- [What they're assuming without stating]
- [What needs to be validated]

### Evidence Evaluation
- [Quality of supporting evidence]
- [What's missing]

### Logical Coherence
- [Does the reasoning hold up?]
- [Internal contradictions?]

### Domain Expertise
- [Do they understand the field?]
- [What expertise gaps exist?]

## FEASIBILITY ASSESSMENT
### Problem Definition
- [Is the problem clear and real?]
- [Who actually has this problem?]

### Solution Appropriateness
- [Does this solution fit the problem?]
- [Are there simpler approaches?]

### Resource Reality
- [Time required vs available]
- [Cost required vs available]
- [Skills required vs possessed]
- [Infrastructure needed]

### Risk Profile
- [What could go wrong?]
- [What's the downside exposure?]
- [Is this reversible?]

## RECOMMENDATION
[ ] APPROVE - Proceed to execution planning
[ ] CONDITIONAL - Requires following refinements: [list]
[ ] REJECT - Not viable because: [reasons]
```

#### Phase 2: EXECUTE (The Project Management Support)
If approved, provide structured execution plan.

**Structured template:**
```markdown
## EXECUTION PLAN

### Phase 1: Validation (Week 1-2)
- Literature review
- Expert consultation
- Proof of concept design

### Phase 2: Capability Building (Week 3-8)
- Skills to acquire: [specific skills]
- Resources to obtain: [tools, access]
- Network to build: [experts, community]

### Phase 3: Execution (Week 9+)
- Build prototype
- Test with users
- Iterate based on feedback

### Resource Requirements
- **Time**: [X hours/week for Y weeks]
- **Money**: $[amount] for [what]
- **Infrastructure**: [tools, access, etc.]
- **People**: [collaborators, experts needed]

### Risk Mitigation
- [Top 3 risks and mitigation strategies]

### Success Criteria
- [How do we know this worked?]
- [What does "done" look like?]

### Cost-Schedule-Risk Analysis
- **Optimistic**: [best case timeline/cost]
- **Realistic**: [likely timeline/cost]
- **Pessimistic**: [worst case timeline/cost]
```

---

## Core Principles

### 1. **Evidence-Based Capability Assessment**
Don't accept self-reported skill levels. Look for:
- ✅ Completed projects (code in repo, shipped products)
- ✅ Demonstrated debugging ability (solved real problems)
- ✅ Domain-specific vocabulary use (correctly, not parroting)
- ❌ "I understand X" (understanding ≠ capability)
- ❌ "I read about Y" (reading ≠ doing)
- ❌ "Claude explained Z" (explanation ≠ expertise)

### 2. **Foundation-First Philosophy**
You cannot skip foundations, even with AI assistance:
- Want to build distributed systems? Build single-threaded first
- Want to create game engine? Make simple games with existing engines first
- Want to implement research papers? Implement textbook algorithms first

**Why**: Foundations teach the JUDGMENT needed to use advanced tools effectively.

### 3. **Small Wins Build Capability**
Success is more valuable than ambitious failure:
- Completing 10 small projects > abandoning 1 large project
- Shipping simple version > planning perfect version forever
- Building foundations incrementally > attempting everything at once

### 4. **Honest Difficulty Communication**
LLMs should STOP minimizing difficulty:
- ❌ "This is straightforward" (when it's not)
- ❌ "We can easily implement" (when it's complex)
- ❌ "Just needs a few tweaks" (when it needs complete redesign)
- ✅ "This is a 6-month project requiring X, Y, Z skills"
- ✅ "This requires deep knowledge of [domain] you haven't demonstrated"
- ✅ "Let's build something simpler that gets you 80% of the value"

### 5. **Comfortable Saying "You're Not Ready"**
Most important capability: Telling humans hard truths
- "You're not ready for this project yet"
- "You need to build these foundations first"
- "This is beyond your current capability"
- "Let's do something you CAN succeed at"

**Why this is hard**: LLMs are trained to be helpful and encouraging. This skill requires being helpful through HONESTY, not validation.

---

## Pack Structure

**Pack**: `axiom/idea-contestability` (~10-12 skills across 3 sub-packs)

### Meta-Skill: `axiom/idea-contestability/using-idea-contestability`
**Primary router** - Determines when contestability review is needed and routes to appropriate phase.

**Triggers**:
- Human proposes new project/idea
- Scope dramatically exceeds demonstrated capability
- Request for validation of questionable theory
- "Let's build X" without evidence of prerequisites
- Pattern of learning without shipping

**Routing logic**:
1. First, always run **CHALLENGE phase** (contestability assessment)
2. Based on recommendation:
   - **APPROVE** → Route to EXECUTE phase (project planning)
   - **CONDITIONAL** → Provide refinements, then reassess
   - **REJECT** → Explain why, suggest alternatives

---

### Sub-Pack 1: `axiom/idea-contestability/challenge` (4-5 skills)
**The Contestability Papers** - Red-teaming proposals

**Meta-skill**: `using-challenge-framework`
- Routes between rigor and feasibility assessment skills

**Core skills**:
1. **intellectual-rigor-adversarial** - Evaluate logical coherence, assumptions, evidence quality
   - Assumption identification
   - Evidence evaluation
   - Logical coherence checking
   - Domain expertise assessment

2. **project-approval-adversarial** - Evaluate practical feasibility
   - Problem definition clarity
   - Solution appropriateness
   - Resource reality checking
   - Risk profiling

3. **crackpot-theory-detection** - Recognize invalid ideas seeking validation
   - Physics/math crackpot patterns
   - LLM-validation-seeking behavior
   - Established-theory-ignorance patterns

4. **capability-assessment-frameworks** - Evidence-based skill evaluation
   - Portfolio analysis (what have they BUILT?)
   - Domain vocabulary assessment
   - Theory-vs-practice gap identification

5. **recommendation-synthesis** - Combine assessments into APPROVE/CONDITIONAL/REJECT
   - Risk-benefit analysis
   - Alternative suggestion
   - Refinement identification

**Output**: Formal contestability assessment report with recommendation

---

### Sub-Pack 2: `axiom/idea-contestability/execute` (4-5 skills)
**The Project Management Support** - If approved, here's how to execute

**Meta-skill**: `using-execution-framework`
- Routes between capability building and resource planning

**Core skills**:
1. **idea-to-execution-pipeline** - Break approved idea into phases
   - Validation phase design
   - Capability building phase
   - Execution phase
   - Success criteria definition

2. **capability-building-strategic** - Systematic skill acquisition planning
   - Prerequisite identification
   - Learning path design
   - Practice project selection
   - Progress validation

3. **resource-planning-realistic** - What's actually needed
   - Time estimation (optimistic/realistic/pessimistic)
   - Cost analysis
   - Infrastructure requirements
   - People/expertise needs

4. **risk-mitigation-planning** - What could go wrong, how to handle
   - Risk identification
   - Impact assessment
   - Mitigation strategies
   - Contingency planning

5. **descoping-and-phasing** - Achievable versions and roadmaps
   - MVP identification
   - Progressive enhancement path
   - Foundation-first sequencing
   - Milestone definition

**Output**: Formal execution plan with cost-schedule-risk analysis

---

### Sub-Pack 3: `axiom/idea-contestability/foundations` (2 skills)
**Cross-cutting utilities used by both challenge and execute**

**Core skills**:
1. **evidence-based-assessment** - Separating claims from reality
   - Portfolio analysis techniques
   - Capability demonstration evidence
   - Theory vs practice gap detection

2. **honest-difficulty-communication** - Delivering hard truths constructively
   - Professional skepticism tone
   - Constructive rejection framing
   - Alternative suggestion patterns

---

## Implementation Strategy

### Phase 1: Core Framework (URGENT)
Build minimum viable contestability process:

1. **`using-idea-contestability`** (primary meta-skill) - ~5-8 hours
   - Trigger detection
   - Challenge → Execute routing
   - Recommendation handling

2. **`intellectual-rigor-adversarial`** (challenge skill) - ~5-8 hours
   - Assumption analysis
   - Evidence evaluation
   - Domain expertise assessment
   - **Output**: Rigor section of contestability report

3. **`project-approval-adversarial`** (challenge skill) - ~5-8 hours
   - Feasibility assessment
   - Resource reality checking
   - Risk profiling
   - **Output**: Feasibility section of contestability report

4. **`idea-to-execution-pipeline`** (execute skill) - ~5-8 hours
   - Phase breakdown (validation/capability/execution)
   - Success criteria definition
   - **Output**: Execution plan

**Phase 1 Total**: ~20-32 hours (core contestability loop functional)

### Phase 2: Full Pack Build
Complete remaining skills:

**Challenge pack** (~10-15 hours):
- `capability-assessment-frameworks`
- `crackpot-theory-detection`
- `recommendation-synthesis`

**Execute pack** (~10-15 hours):
- `capability-building-strategic`
- `resource-planning-realistic`
- `risk-mitigation-planning`
- `descoping-and-phasing`

**Foundations** (~5-8 hours):
- `evidence-based-assessment`
- `honest-difficulty-communication`

**Phase 2 Total**: ~25-38 hours

### Phase 3: Integration & Testing
- Test across all existing domains (Ordis, Muna, Bravos, Yzmir, Lyra)
- RED-GREEN-REFACTOR methodology
- Real-world scenario validation

**Phase 3 Total**: ~10-15 hours

**TOTAL PACK**: ~55-85 hours (realistic: 60-70 hours)

---

## Cross-Faction Integration

### Contestability Assessment Applied to ALL Domains

**Integration pattern**: When human proposes project in ANY domain, idea-contestability can be invoked to assess viability before engaging domain skills.

**Example workflows:**

#### Ordis/Security-Architect + Contestability
**Human**: "I want to design a Zero Trust architecture for my startup"

**Contestability assessment**:
- **Rigor check**: Have you implemented authentication? Authorization? Network segmentation?
- **Feasibility check**: Zero Trust is 6-12 month program, not single project
- **Recommendation**: CONDITIONAL - Start with OAuth 2.0 + network segmentation, build toward Zero Trust

**If approved** → Route to `ordis/security-architect/security-controls-design` with scoped goal

#### Muna/Technical-Writer + Contestability
**Human**: "I want to write comprehensive API documentation"

**Contestability assessment**:
- **Rigor check**: Do you have an API? Is it stable? Who's the audience?
- **Feasibility check**: Docs require stable API and understanding of user needs
- **Recommendation**: CONDITIONAL - Build API first, document iteratively as you develop

**If approved** → Route to `muna/technical-writer/api-documentation-design`

#### Bravos/Game-Systems + Contestability
**Human**: "For my first game, I want to build an MMO with crafting, PvP, and procedural worlds"

**Contestability assessment**:
- **Rigor check**: Have you made ANY game? Single-player? Local multiplayer?
- **Feasibility check**: MMO is 50-person, 3-year project, not first game
- **Recommendation**: REJECT - Build Pong, then simple survival game, then consider multiplayer
- **Alternative**: "Build single-player survival game with crafting as MVP"

**Descoped version approved** → Route to `bravos/gameplay-mechanics/core-mechanic-patterns`

#### Yzmir/AI-ML-Engineering + Contestability
**Human**: "I want to implement GPT from scratch to understand transformers"

**Contestability assessment**:
- **Rigor check**: Can you implement backpropagation by hand? Trained any neural network?
- **Feasibility check**: GPT from scratch requires deep understanding of attention, optimization, distributed training
- **Recommendation**: CONDITIONAL - Implement simple CNN on MNIST, then RNN for text, then attention mechanism
- **Capability path**: 3-6 months of foundational work before transformers

**Phased plan approved** → Route to `yzmir/deep-learning/foundational-architectures`

---

### Integration Benefits

1. **Prevents wasted effort** - Catches unrealistic projects before weeks of work
2. **Builds foundations** - Ensures humans have prerequisites before advanced work
3. **Maintains domain expertise** - Domain skills work with capable humans, not novices attempting expert work
4. **Produces better outcomes** - Humans succeed at appropriately-scoped projects

---

## Success Criteria

- ✅ Skill can accurately assess human capability from conversation history
- ✅ Skill can identify foundation gaps before project starts
- ✅ Skill can descope ambitious goals to achievable alternatives
- ✅ Skill can create realistic capability-building roadmaps
- ✅ Skill reduces abandoned projects due to capability mismatch
- ✅ Skill comfortable delivering hard truths about readiness
- ✅ Skill prevents "6 months wasted on wrong project" scenarios

---

## Real-World Scenarios (Contestability Assessments)

### Scenario 1: The "Let's Build Everything" Trap

**Human**: "I want to build a distributed database with ACID guarantees, Byzantine fault tolerance, and Raft consensus"

**Contestability Assessment**:
```
INTELLECTUAL RIGOR ASSESSMENT:
- Assumptions: Assumes distributed systems knowledge without evidence
- Evidence: No demonstrated experience with databases or consensus
- Domain Expertise: Can discuss concepts but has not built even simple KV store

FEASIBILITY ASSESSMENT:
- Problem: Real problem (distributed systems are needed)
- Solution Appropriateness: Massively over-scoped for capability level
- Resource Reality:
  * Required: 2-3 years, deep CS background, distributed systems expertise
  * Available: Unknown capability, likely weeks-months timeline expectation
- Risk: 99% chance of failure without foundations

RECOMMENDATION: ❌ REJECT
- Not viable at current capability level
- Alternative: Build single-node key-value store first
- Capability path: KV store → replication → simple consensus → Raft
- Timeline: 6-12 months of foundational work before distributed database
```

### Scenario 2: The Crackpot Theory

**Human**: "I've discovered a flaw in Einstein's relativity. Can you help me write a paper?"

**Contestability Assessment**:
```
INTELLECTUAL RIGOR ASSESSMENT:
- Assumptions: Assumes understanding of GR without demonstrated capability
- Evidence: No physics background, hasn't worked through GR mathematics
- Logical Coherence: Extraordinary claims require extraordinary evidence (lacking)
- Domain Expertise: Likely misunderstanding of established physics

FEASIBILITY ASSESSMENT:
- Problem: "Problem" likely stems from misunderstanding, not actual flaw
- Solution: Writing paper on invalid theory wastes time
- Risk: Embarrassment, reinforcement of misunderstanding

RECOMMENDATION: ❌ REJECT
- This is classic crackpot theory pattern
- Action: Work through GR mathematics (Carroll textbook)
- Validation: Reproduce known GR results before claiming flaws
- Reality check: "If thousands of physicists missed this, you probably misunderstood something"
```

### Scenario 3: Tutorial Hell

**Human**: "I've spent 6 months learning React, Vue, Angular, Svelte... which framework should I learn next?"

**Contestability Assessment**:
```
INTELLECTUAL RIGOR ASSESSMENT:
- Evidence: 6 months learning, zero shipping
- Pattern: Tutorial hell - infinite learning, no building

FEASIBILITY ASSESSMENT:
- Problem: Not "which framework?" but "why not building?"
- Solution: STOP learning, START building
- Resource Reality: Have capability to build, not using it

RECOMMENDATION: 🔄 CONDITIONAL
- Condition: Build and deploy 3 projects with React BEFORE learning new framework
- Projects:
  1. Todo app (simple CRUD)
  2. API-consuming dashboard (real data)
  3. User-facing app (deployed to production)
- Timeline: 4-6 weeks to build foundations through practice
- Then revisit if new framework actually needed
```

### Scenario 4: Scope Explosion

**Human**: "For my first game, I want to create an open-world survival MMO with crafting, building, PvP, and procedural generation"

**Contestability Assessment**:
```
INTELLECTUAL RIGOR ASSESSMENT:
- Assumptions: Assumes game dev capability without evidence
- Evidence: Has made zero games
- Domain Expertise: Can describe features but not implement

FEASIBILITY ASSESSMENT:
- Problem: Real problem (survival MMO would be fun)
- Solution Appropriateness: 50-person, 3-year project as first game
- Resource Reality:
  * Required: Team, $5M budget, 3+ years
  * Available: Solo dev, hobby timeline
- Risk: 100% chance of abandonment

RECOMMENDATION: ❌ REJECT
- MMO not viable as first game
- Alternative: Single-player survival game with crafting
- Capability path:
  1. Make Pong (learn game loop)
  2. Make simple platformer (learn physics, input)
  3. Make small survival game (learn systems)
  4. Add local multiplayer
  5. THEN consider networked multiplayer
- Timeline: 6-12 months before networked multiplayer capability
```

### Scenario 5: The "I Understand It" Trap

**Human**: "I understand how transformers work from reading papers. Let's implement GPT from scratch."

**Contestability Assessment**:
```
INTELLECTUAL RIGOR ASSESSMENT:
- Assumptions: Understanding explanation ≠ implementation capability
- Evidence: Has read papers but not implemented neural networks
- Theory-Practice Gap: Large gap between conceptual understanding and coding ability

FEASIBILITY ASSESSMENT:
- Problem: Real problem (understanding transformers is valuable)
- Solution Appropriateness: GPT from scratch requires:
  * Backpropagation mastery
  * Attention mechanism understanding
  * Optimization and training expertise
  * Distributed training knowledge
- Resource Reality:
  * Required: 3-6 months ML foundations
  * Available: Conceptual understanding only
- Risk: High frustration, likely abandonment

RECOMMENDATION: 🔄 CONDITIONAL
- Condition: Build foundations through progressive implementation
- Capability path:
  1. Implement backpropagation by hand (understand gradients)
  2. Train simple CNN on MNIST (understand training loop)
  3. Implement RNN for text (understand sequence modeling)
  4. Implement attention mechanism (understand core transformer concept)
  5. THEN implement small transformer from scratch
- Timeline: 3-6 months of foundational work
- Approved: Once completed steps 1-4, GPT implementation is viable
```

---

## Why This Belongs in Skillpacks

This addresses a **systemic problem in human-AI collaboration**: LLMs' ability to explain anything creates false confidence that humans can DO anything.

**Without this skill**: Skillpacks help humans fail faster at ambitious projects
**With this skill**: Skillpacks guide humans to build foundations needed for eventual success

This is not about discouraging ambition. It's about **sequencing ambition correctly**:
- Not "don't build a game engine"
- But "make games with Unity first, then consider building engine"

---

## Faction Placement

**Confirmed**: **Axiom** (Creators of Marvels - Process, Methodology, Infrastructure)

**Why Axiom**:
- This is **meta-process work** - HOW to evaluate and execute ideas
- Parallels Axiom's domain: Tooling, infrastructure, methodology
- Not domain-specific (Bravos/Yzmir) or creative (Lyra) or communication (Muna)
- But **process** that applies across all domains

**Path**: `source/axiom/idea-contestability/`

**Integration model**:
- Can be invoked explicitly when human proposes new project
- Could potentially be auto-triggered on scope red flags (future enhancement)
- Produces formal assessment reports as artifacts

---

## Resolved Questions

1. ✅ **Faction placement**: Axiom (process/methodology domain)
2. ✅ **Foundation**: Formal government contestability practice
3. ✅ **Structure**: Full pack (10-12 skills across 3 sub-packs: challenge, execute, foundations)
4. ✅ **Tone**: Professional skepticism (like government contestability papers - rigorous, not mean)
5. ✅ **Two-phase model**: Challenge (red-team) → Execute (project management)

## Remaining Open Questions

1. **Build priority?** URGENT (as stated) or build after other Axiom packs?
2. **Integration depth?** Explicitly invoked, or auto-triggered on scope red flags?
3. **Report artifacts?** Should formal assessment reports be saved as files in conversation?
4. **Cross-domain testing?** Which existing domain should we test against first (Ordis? Bravos game design?)?

---

**End of Intent Document**

**Next Steps**:
1. User validates this interpretation
2. Decide on faction placement and structure
3. Implement RED-GREEN-REFACTOR testing
4. Deploy as highest-priority addition to skillpacks

---

**This skill exists because**: LLMs make everything SOUND achievable, and humanity needs help distinguishing "I understand the explanation" from "I can build the thing".
