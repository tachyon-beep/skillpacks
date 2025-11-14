---
name: using-research-methodology
description: Use when conducting academic or industry research, planning studies, analyzing results, or publishing findings - routes to literature review, experimental design, statistical analysis, reproducibility, or publishing skills based on research phase and symptoms
---

# Using Research Methodology

## Overview

This meta-skill routes you to the right research methodology specialist based on research phase and symptoms. Research problems require specialized methodological knowledge - don't give general advice when specialists exist.

**Core Principle**: Different research phases and problems require different specialist skills. Match symptoms to the appropriate specialist. ASK when ambiguous. NEVER make assumptions about risk levels or research requirements.

## When to Use

Load this skill when:
- User mentions: "research", "study", "experiment", "literature review", "publish", "IRB", "hypothesis", "statistical analysis"
- Planning or conducting academic/industry research
- Analyzing experimental results
- Preparing for publication or peer review
- Setting up reproducible research workflows

**Don't use for**: "Research" meaning "investigate tools/libraries" (that's tool selection, not research methodology)

---

## Critical Red Flags - STOP Before Giving Advice

### 🚨 Time Pressure on Ethics/IRB

**If user says**:
- "IRB meeting tomorrow and haven't prepared"
- "Need quick ethics approval"
- "How to rush through IRB"

**STOP. Do NOT:**
- Give "quick tips" to rush preparation
- Assume study is "low-risk" without verification
- Enable shortcuts: "here's the minimum you need"
- Say: "Prioritize utility over completeness"

**INSTEAD:**
1. Route to `research-ethics` skill
2. Set proper expectation: "Might need to reschedule meeting to prepare properly"
3. ASK critical questions before any advice:
   - "Is data anonymous or identifiable?"
   - "Are you evaluating work performance or personal practices?"
   - "Does study involve any sensitive topics?"

**Why**: Research ethics protect human subjects. Time pressure NEVER justifies shortcuts. Risk determination requires specific information.

### 🚨 Making Assumptions About Research Phase

**If you're tempted to assume**:
- "They probably need a literature review"
- "Sounds like they need experimental design"
- "Typical first-time researcher, so..."

**STOP. Do NOT assume.**

**INSTEAD**: Ask ONE diagnostic question:
- "What phase are you in? (1) Starting research/surveying field, (2) Have question, need study design, (3) Have data, need analysis, (4) Writing paper"

**Why**: Assumptions waste time. One question routes correctly.

### 🚨 Giving Direct Research Advice

**If you're about to**:
- Explain how to do literature review yourself
- Give statistical analysis tips directly
- Describe experimental design yourself
- Explain publishing process

**STOP. Do NOT give direct advice.**

**INSTEAD**: Route to the specialist skill.

**Why**: Specialist skills have comprehensive coverage, anti-patterns, and tested guidance. Your ad-hoc advice will miss critical details.

---

## Routing by Research Phase

### Phase 1: Starting Research / Surveying Field

**Symptoms**:
- "Getting started with research on X"
- "Need to understand the landscape"
- "What's been done in this area?"
- "Survey the field"

**Route to**:
1. `literature-review-strategies` (systematic search and synthesis)
2. Then `research-question-formulation` (narrow down scope)

**Ask if ambiguous**: "Are you surveying to understand the field, or do you already know what you want to study?"

---

### Phase 2: Formulating Study / Planning Experiment

**Symptoms**:
- "How do I test this hypothesis?"
- "Design an experiment for X"
- "Plan a study to investigate Y"
- "What's the right methodology?"

**Route to**:
1. `research-question-formulation` (if question is vague)
2. `hypothesis-formation` (if need testable predictions)
3. `experimental-design` (for study planning)
4. `statistical-reasoning` (for power analysis, sample size)

**Sequential routing**: Question → Hypothesis → Design → Power analysis

---

### Phase 3: Conducting Research

**Symptoms**:
- "Make research reproducible"
- "Track experiments"
- "Version control for data"
- "Document methodology"
- "Manage research notes"

**Route to**:
1. `reproducibility-practices` (version control, documentation, provenance)
2. `knowledge-management` (note-taking, synthesis, Zettelkasten)

**Cross-pack routing**: If ML-specific experiment tracking (MLflow, wandb) → `yzmir-training-optimization/experiment-tracking`

---

### Phase 4: Analyzing Results

**Symptoms**:
- "Which statistical test to use?"
- "Results don't make sense"
- "Interpret p-values"
- "Calculate effect sizes"
- "Check assumptions"

**Route to**: `statistical-reasoning`

**Ask diagnostic question if vague**: "What doesn't make sense? (1) Don't know which test to use, (2) Results contradict expectations, (3) Can't interpret statistics, (4) Need to validate analysis"

---

### Phase 5: Publishing / Peer Review

**Symptoms**:
- "Where to publish?"
- "Write my first paper"
- "Respond to reviewer comments"
- "Choose journal vs conference"
- "Manage citations"

**Route to**:
1. `publishing-strategies` (venue selection, manuscript prep)
2. `peer-review-process` (handling reviews, revisions)
3. `citation-management` (reference organization, styles)

**Multi-skill scenario**: First-time paper → Route to ALL THREE in sequence

---

## Routing by Symptom

### "Survey the field" / "Literature review"
→ `literature-review-strategies`

### "What should I study?" / "Narrow down topic"
→ `research-question-formulation`

### "How to test this?" / "Formulate hypothesis"
→ `hypothesis-formation`

### "Design experiment" / "Plan study"
→ `experimental-design`

### "Which statistical test?" / "Analyze results"
→ `statistical-reasoning`

### "Make reproducible" / "Document research"
→ `reproducibility-practices`

### "IRB approval" / "Research ethics"
→ `research-ethics`

### "Manage references" / "Citation styles"
→ `citation-management`

### "Respond to reviewers" / "Revise paper"
→ `peer-review-process`

### "Where to publish?" / "First paper"
→ `publishing-strategies`

### "Organize research notes" / "Zettelkasten"
→ `knowledge-management`

---

## When User Resists Routing

### User Says: "Just answer the question, I don't need the skill"

**STOP. Do NOT bypass routing.**

**Why specialist skills exist**:
- Comprehensive coverage (not ad-hoc summary)
- Tested anti-patterns
- Explicit examples and templates
- Production-grade guidance

**Response**: "I'm routing to [skill] because it has comprehensive coverage of [topic]. This ensures you get complete guidance tested against common mistakes, not my generalist summary."

### User Says: "I can't reschedule IRB, just help me rush it"

**STOP. Do NOT enable ethics shortcuts.**

**Why this matters**:
- Research ethics protect human subjects
- IRB rejection worse than delay
- Rushed ethics = research liability

**Response**: "Rescheduling protects both you and your participants. Submitting rushed ethics documentation risks rejection, which causes longer delays. I'm routing to research-ethics for proper preparation."

### User Says: "I'm a senior researcher, I know what I need"

**STOP. Do NOT skip routing based on perceived expertise.**

**Why experts benefit**:
- Even experts have gaps in adjacent specialties
- Skills are reference material, not teaching
- Domain expertise ≠ methodological expertise

**Response**: "I'm routing based on the task requirements, not your experience level. Specialist skills provide comprehensive reference material that benefits researchers at all levels."

### User Says: "Just explain [specific methodology] quickly"

**STOP. Do NOT give direct methodology explanations.**

**Examples**:
- "Just explain PRISMA quickly" → Route to literature-review-strategies
- "Quick overview of power analysis" → Route to statistical-reasoning
- "Summarize IRB requirements" → Route to research-ethics

**Why**: Direct explanations miss:
- Context-specific guidance
- Common mistakes
- When NOT to use the methodology
- Integration with other research steps

**Response**: "I'm routing to [skill] which has complete [methodology] coverage including when to use it, anti-patterns, and examples. This is more reliable than a quick summary."

---

## Common Routing Mistakes

| User Says | ❌ Wrong Response | ✅ Correct Response | Why |
|-----------|------------------|-------------------|-----|
| "Research ML fairness" | Give fairness algorithm overview | ASK: "What phase? Survey field or design study?" | Ambiguous - could be Phase 1 or 2 |
| "First academic paper" | Explain paper structure | Route to: publishing-strategies + peer-review-process + citation-management | Multi-skill scenario |
| "IRB tomorrow, not prepared" | "Here's quick prep checklist" | Route to research-ethics + set expectation about rescheduling | Time pressure on ethics - never enable rushing |
| "I can't reschedule IRB" | Help them rush prep | "Rescheduling protects you and participants. Route to research-ethics for proper prep" | Never compromise on ethics |
| "I'm senior researcher" | Assume they don't need routing | Route anyway - experts benefit from references | Route based on task, not expertise |
| "Just explain PRISMA" | Give quick overview | Route to literature-review-strategies | Comprehensive > quick summary |
| "Results don't make sense" | Assume it's statistics | ASK: "What doesn't make sense? Stats? Contradicts hypothesis? Can't replicate?" | Could be multiple skills |
| "Make research reproducible" | Give git advice directly | Route to reproducibility-practices | Has comprehensive patterns |
| "Research best Python framework" | Route to research-methodology | Route to axiom-python-engineering | "Research" = investigate, not academic research |

---

## Decision Tree

```
Research task identified?
├─ Is "research" academic/scientific? (vs tool investigation)
│  ├─ YES: Continue routing
│  └─ NO: Route to appropriate technical skill (not research-methodology)
│
├─ What research phase?
│  ├─ Starting / surveying → literature-review-strategies
│  ├─ Planning study → experimental-design (+ hypothesis-formation if needed)
│  ├─ Conducting → reproducibility-practices OR knowledge-management
│  ├─ Analyzing → statistical-reasoning
│  └─ Publishing → publishing-strategies (+ peer-review + citations)
│
├─ Is phase ambiguous?
│  ├─ YES: ASK "What phase are you in?" before routing
│  └─ NO: Route directly
│
├─ Does ethics/IRB apply?
│  ├─ YES: ALWAYS route to research-ethics first
│  ├─ Time pressure on ethics? → Set expectation about proper timeline
│  └─ NO: Continue normal routing
│
└─ Multiple skills needed?
   ├─ Sequential dependency? → Route in order (question → hypothesis → design)
   └─ Parallel needs? → Route to all, explain purpose of each
```

---

## Cross-Pack Boundaries

### When to Route ELSEWHERE (Not Research Methodology)

**"Research best Python testing framework"**
→ Route to: `axiom-python-engineering` (tool selection, not research)

**"Track ML experiments with MLflow"**
→ Route to: `yzmir-training-optimization/experiment-tracking` (ML-specific)
→ Note: For general reproducibility principles, ADD `reproducibility-practices`

**"Document my API"**
→ Route to: `muna-technical-writer` (documentation, not research)

**"Statistical analysis in pandas"**
→ Route to: `axiom-python-engineering/scientific-computing-foundations` (implementation)
→ Note: For statistical reasoning principles, ADD `statistical-reasoning`

### When to Route to BOTH Packs

**"Make ML training reproducible for publication"**
→ Route to BOTH:
1. `yzmir-training-optimization/experiment-tracking` (ML tooling)
2. `reproducibility-practices` (research principles)

**Why both**: Tooling implementation + research methodology principles

---

## Handling Ambiguity - ASK First

When symptom is unclear, ask ONE diagnostic question:

| Ambiguous Query | Diagnostic Question |
|-----------------|---------------------|
| "Help with my research" | "What phase? (1) Starting, (2) Designing study, (3) Analyzing data, (4) Publishing" |
| "My experiment results" | "Need help with: (1) Statistical analysis, (2) Interpretation, (3) Reproducibility, (4) Publishing" |
| "Starting a study" | "Do you have a research question formulated, or still defining scope?" |
| "Need IRB approval" | "Is your study design complete? Do you have: protocol, consent forms, data plan?" |
| "Literature review" | "Are you: (1) Surveying field systematically, (2) Writing related work section, (3) Finding research gaps?" |

**Don't guess. Ask once. Route accurately.**

---

## Red Flags Checklist - Self-Check Before Responding

Before giving ANY research advice, ask yourself:

1. ❓ **Is this actually academic/scientific research?**
   - If NO → Route to appropriate technical skill (not research-methodology)

2. ❓ **Do I know what research phase they're in?**
   - If NO → Ask diagnostic question

3. ❓ **Am I about to give direct advice instead of routing?**
   - If YES → STOP. Route to specialist instead.

4. ❓ **Is there time pressure on ethics/IRB?**
   - If YES → Set proper expectations (don't enable rushing)

5. ❓ **Am I making assumptions about risk level, experience, or context?**
   - If YES → STOP. Ask instead of assume.

6. ❓ **Do multiple skills apply?**
   - If YES → Route in sequence or explain parallel routing

**If you fail ANY check, do NOT give direct advice. Route or ask clarifying question.**

---

## Rationalization Table - Don't Do These

| Excuse | Reality | What To Do |
|--------|---------|------------|
| "Quick advice is enough" | Research requires rigorous methodology. Specialists exist. | Route to specialist |
| "Prioritize utility over completeness" | Shortcuts in research = invalid results | Don't compromise on rigor |
| "They seem rushed, help them shortcut" | Time pressure doesn't justify bad research | Set proper expectations |
| "I can determine risk level from description" | Risk determination requires specific questions | ASK before assessing risk |
| "Assume typical case for efficiency" | Assumptions waste time when wrong | Ask ONE question instead |
| "They're experienced, they know what they need" | Even experts benefit from specialist skills | Route based on task, not perceived skill |
| "Load all skills to be safe" | Overwhelming - defeats router purpose | Route to specific needed skills |

**All of these are rationalizations. Reject them.**

---

## Skill Catalog

**Complete research methodology toolkit**:

1. **literature-review-strategies** - Systematic search, PRISMA, screening, synthesis, gap identification
2. **research-question-formulation** - PICO/FINER frameworks, scope definition, question hierarchies
3. **hypothesis-formation** - Testable predictions, null/alternative hypotheses, operationalization
4. **experimental-design** - RCT, controls, validity, power analysis, sample size
5. **statistical-reasoning** - Test selection, p-values, effect sizes, assumptions, inference
6. **reproducibility-practices** - Version control, computational notebooks, documentation, provenance
7. **citation-management** - Zotero/Mendeley, citation styles, reference organization
8. **peer-review-process** - Reviewing papers, responding to reviewers, revision strategies
9. **publishing-strategies** - Venue selection, manuscript preparation, submission workflows
10. **research-ethics** - IRB, consent, data privacy, conflicts of interest, vulnerable populations
11. **knowledge-management** - Zettelkasten, note-taking, synthesis, concept mapping

**Route based on symptoms and phase, not guesses.**

---

## Examples

### Example 1: Ambiguous Research Start

```
User: "I need to do research on ML fairness algorithms. Help me get started."

Your routing:
1. Recognize: Research task, but phase unclear
2. ASK: "What phase are you in? (1) Surveying field to understand landscape, (2) Have specific question, need study design"
3. If (1) → Route to: literature-review-strategies
4. If (2) → Route to: research-question-formulation + experimental-design
```

### Example 2: First Academic Paper (Multi-Skill)

```
User: "I'm writing my first academic paper. I have results from my ML experiment and need to publish."

Your routing:
1. Recognize: End of research cycle, multiple skills needed
2. Route in sequence:
   - First: statistical-reasoning (ensure results are analyzed correctly)
   - Then: reproducibility-practices (document experiment properly)
   - Then: publishing-strategies (venue selection, manuscript prep)
3. Mention for follow-up: peer-review-process, citation-management
```

### Example 3: IRB Time Pressure (Critical)

```
User: "I have IRB meeting tomorrow and haven't prepared anything. My study involves asking developers about their tools. What do I need?"

Your routing:
1. Recognize: Research ethics + extreme time pressure
2. STOP - Don't enable rushing
3. ASK critical questions:
   - "Is this anonymous survey or identified interviews?"
   - "Are you collecting any sensitive data?"
   - "Are you evaluating work performance?"
4. Route to: research-ethics
5. Set expectation: "You may need to reschedule the meeting to prepare properly. Research ethics require thorough preparation."
```

### Example 4: Tool Selection (NOT Research Methodology)

```
User: "I need to research the best Python testing framework for my project."

Your routing:
1. Recognize: "Research" means "investigate/compare", not academic research
2. Route to: axiom-python-engineering/testing-and-quality
3. Do NOT load research-methodology skills
```

### Example 5: Cross-Pack Routing

```
User: "I need to make my ML training reproducible for publication."

Your routing:
1. Recognize: Overlapping needs (ML tooling + research principles)
2. Route to BOTH:
   - yzmir-training-optimization/experiment-tracking (MLflow, wandb, ML-specific)
   - reproducibility-practices (version control, documentation, research standards)
3. Explain distinction: "First for ML tooling, second for publication-grade reproducibility"
```

---

## Summary

**This skill routes research tasks → specialist skills.**

**Core rules**:
1. Identify if "research" is academic/scientific (vs tool investigation)
2. Ask diagnostic question if phase unclear
3. Route to specialist - don't give direct advice
4. NEVER enable rushing on research ethics
5. NEVER make assumptions about risk, phase, or experience
6. Decompose multi-skill scenarios, route in sequence
7. Set proper expectations (might need more time, not less)

**Meta-rule**: When in doubt, ASK before routing. One question routes correctly. Assumptions waste time.
