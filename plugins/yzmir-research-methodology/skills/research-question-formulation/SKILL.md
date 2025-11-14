---
name: research-question-formulation
description: Use when defining research scope, refining vague ideas into specific questions, or unclear what to investigate - prevents scope creep, unanswerable questions, multiple-questions-as-one, and solution-first framing using PICO/FINER frameworks
---

# Research Question Formulation

## Overview

A good research question is **specific, answerable, and appropriately scoped**. Research questions define what you'll investigate - they come BEFORE hypotheses, BEFORE methods, BEFORE literature review completion.

**Core Principle**: One PhD = one core research question (with sub-questions). Questions must be neither too broad (unanswerable in 3-5 years) nor too narrow (trivial). Use systematic frameworks (PICO/FINER) to force specificity.

## When to Use

Use this skill when:
- Starting research and unclear what to study
- Have broad topic but need specific question
- Refining vague research ideas
- Advisor says "narrow your scope"
- Formulating PhD proposal research question
- Converting interests into answerable questions

**Don't use for**: Hypothesis formation (that comes AFTER question), literature review (that informs question)

---

## Critical Red Flags - STOP Before Accepting Question

### 🚨 Too Broad / Scope Creep

**If research question is**:
- "How does AI impact society?"
- "What are the effects of climate change?"
- "How do neural networks work?"
- Topic, not question

**STOP. This is too broad.**

**Why**: PhD timeline = 3-5 years. Broad questions are unanswerable in that timeframe. You'll either:
- Never finish (scope creep keeps adding)
- Answer superficially (breadth without depth)
- Pivot randomly (no clear endpoint)

**Test**: Can you answer this in 3-5 years with focused research? If no, too broad.

---

### 🚨 Too Narrow / Trivial

**If research question is**:
- "What is GPT-4's accuracy on specifically question 37 of MMLU at temperature 0.73?"
- "How many times does the word 'fairness' appear in NeurIPS 2023 papers?"
- Single measurement, not investigation

**STOP. This is too narrow/trivial.**

**Why**: Research questions should contribute to field knowledge. A single data point or arbitrary measurement doesn't.

**Test**: Why does this specific question matter? What larger question does it help answer? If it's just one measurement, it's not a research question.

---

### 🚨 "What is Best" / Unfalsifiable

**If research question asks**:
- "What is the best ML algorithm for fairness?"
- "Which architecture is optimal?"
- "What's the right approach to X?"

**STOP. "Best" is unanswerable without context.**

**Why**: "Best" depends on:
- Best for what use case?
- Best by what metric?
- Best under what constraints?
- Different contexts = different "best"

**Test**: Can this question have one definitive answer that's true across all contexts? If no, it's unfalsifiable as stated.

**Fix**: Reframe as comparative with specific context: "How do algorithms X, Y, Z compare on metrics A, B in context C?"

---

###  🚨 Multiple Questions Disguised as One

**If research question has multiple "and" clauses**:
- "How do architectures affect training time AND inference accuracy AND what are implications for edge deployment AND how does this interact with compression?"

**STOP. Count the questions.**

**Test**: Count "and" clauses. Each potentially separates a different question.

**Analysis**:
1. "How do architectures affect training time?" (Question 1)
2. "AND inference accuracy?" (Question 2)
3. "AND what are implications for edge deployment?" (Question 3)
4. "AND how does this interact with compression?" (Question 4)

This is FOUR questions, not one.

**Fix**: Choose ONE as core question. Others become sub-questions or future work.

---

### 🚨 Solution-First (Not Problem-First)

**If question starts with a predetermined solution**:
- "How can transformers improve time-series forecasting?"
- "How can blockchain solve X?"
- "How can I apply technique Y to domain Z?"

**STOP. This is solution-first, not problem-first.**

**Why**: You've already decided the answer (transformers, blockchain, Y) before investigating the problem. This is confirmation bias.

**Test**: Does your question assume a specific solution? If yes, solution-first.

**Fix**: Start with problem: "What are current limitations in time-series forecasting?" Research question investigates problem; solutions are what you TEST, not assume.

---

### 🚨 Undefined Success Criteria

**If you can't answer "How will I know I've answered this?"**:
- "How do users interact with chatbots?" (Too vague - which interaction aspects?)
- "What are the effects of X?" (Which effects? Measured how?)

**STOP. No success criteria = unanswerable.**

**Why**: If you can't define what "answering" means, you'll never finish research (or you'll answer arbitrarily).

**Test**: How will you know you've answered this question? What evidence would count as an answer?

**Fix**: Force specificity: "What factors influence user error rates when interacting with chatbots for customer service tasks?"

---

## Systematic Question Formulation Frameworks

Use these frameworks to force specificity:

### PICO Framework (Empirical Research)

**Use when**: Your research involves interventions, comparisons, empirical data

**Components**:
- **P**opulation: Who/what is being studied?
- **I**ntervention: What is being applied, tested, or changed?
- **C**omparison: Compared to what? (baseline, alternative, control)
- **O**utcome: What is being measured?

**Example**:

**Vague**: "AI fairness in hiring"

**PICO**:
- **P**: Hiring algorithms for tech company recruitment
- **I**: Fairness-aware ML algorithms (e.g., adversarial debiasing)
- **C**: Standard ML algorithms without fairness constraints
- **O**: Demographic parity and prediction accuracy

**Question**: "How do fairness-aware ML algorithms compare to standard algorithms on demographic parity and accuracy metrics when applied to tech company hiring decisions?"

---

### FINER Framework (Any Research)

**Use when**: Evaluating if question is worth pursuing

**Components**:
- **F**easible: Can you actually do this research? (data available, timeline reasonable, resources accessible)
- **I**nteresting: Does it matter to you and others?
- **N**ovel: Adds something new to knowledge?
- **E**thical: No problematic implications?
- **R**elevant: Matters to your field?

**Example**:

**Question**: "How do neural network architectures affect training time and accuracy in image classification?"

**FINER Evaluation**:
- ✅ **Feasible**: Yes - public datasets (ImageNet), existing architectures, computational resources obtainable
- ✅ **Interesting**: Yes - practitioners care about accuracy/speed tradeoffs
- ⚠️ **Novel**: Maybe - many papers on this; what's new about your angle?
- ✅ **Ethical**: Yes - no human subjects, no dual-use concerns
- ✅ **Relevant**: Yes - important for deployment decisions

**Result**: Question is mostly good, but needs novelty angle (new architectures? new datasets? new analysis?).

---

## Research Question Refinement Process

Follow this sequence to go from broad topic to specific question:

### Step 1: Start with Broad Interest (Topic)

**Example**: "I'm interested in AI fairness"

This is a TOPIC, not a question. That's okay - start here.

---

### Step 2: Apply PICO to Force Specificity

**P**: What population/domain?
- Hiring? Lending? Criminal justice? Healthcare? Education?
- Choose ONE: Hiring algorithms

**I**: What intervention/phenomenon?
- Fairness metrics? Debiasing techniques? Fairness definitions? Transparency?
- Choose ONE: Fairness-aware training techniques

**C**: Compared to what?
- Traditional ML? Human decisions? Different fairness approaches?
- Choose ONE: Traditional ML without fairness constraints

**O**: What outcome matters?
- Accuracy? Fairness metrics (which one)? Demographic representation? Discrimination reduction?
- Choose: Demographic parity and accuracy tradeoffs

---

### Step 3: Write Specific Question

**Result**: "How do fairness-aware training techniques compare to traditional ML on demographic parity and accuracy metrics in hiring algorithm contexts?"

---

### Step 4: Evaluate with FINER

- **Feasible**: Can I get hiring algorithm datasets? (Check: yes, public benchmarks exist)
- **Interesting**: Do practitioners care? (Check: yes, legal compliance + performance)
- **Novel**: What's new? (Check existing literature - maybe focus on specific industry or recent techniques)
- **Ethical**: Any concerns? (Check: no human subjects if using public data, but consider implications of findings)
- **Relevant**: Matters to field? (Check: yes, active research area)

---

### Step 5: Refine Based on FINER Gaps

If novelty is weak, add angle:
- Focus on specific underexplored industry (e.g., healthcare hiring)
- Focus on recent techniques (e.g., post-2022 fairness methods)
- Focus on interaction effects (e.g., fairness + privacy)

**Refined**: "How do recent fairness-aware training techniques (2022-2024) compare to traditional ML on demographic parity and accuracy metrics in healthcare hiring algorithm contexts?"

---

## Answerable Question Checklist

Before finalizing your question, verify:

- [ ] **Specific**: Includes population, context, boundaries (not "AI impacts society")
- [ ] **Answerable**: Has clear success criteria (you know what evidence would answer it)
- [ ] **Falsifiable**: Could empirically find evidence for or against
- [ ] **Scoped**: Can be answered in 3-5 years (PhD), 1-2 years (Masters), 6-12 months (project)
- [ ] **Single core question**: If multiple "and" clauses, they're sub-components not separate questions
- [ ] **Problem-first**: Asks about problem, not predetermined solution
- [ ] **Operationalized**: Subjective terms ("best", "better", "optimal") defined with metrics
- [ ] **Bounded**: Has limits (timeframe, geography, population, domain)
- [ ] **Novel**: Adds something new (check with FINER)
- [ ] **Feasible**: You can actually do this research

If ANY checkbox fails, refine the question.

---

## Question Types and When to Use

| Type | Format | When to Use | Example |
|------|--------|-------------|---------|
| **Descriptive** | "What is X?" | Understudied phenomenon, need characterization | "What fairness metrics are used in practice for hiring algorithms?" |
| **Relational** | "How does X relate to Y?" | Exploring associations | "How does model complexity relate to fairness-accuracy tradeoffs?" |
| **Causal** | "Does X cause Y?" | Establishing causation | "Does adversarial debiasing reduce demographic disparity in hiring outcomes?" |
| **Comparative** | "How do X and Y compare on Z?" | Evaluating alternatives | "How do fairness-aware and standard algorithms compare on accuracy and parity?" |
| **Exploratory** | "What factors influence X?" | Multiple causes unknown | "What factors influence fairness metric selection in deployed hiring systems?" |

**PhD-level**: Usually comparative or causal (more rigorous)
**Exploratory**: Good for understudied areas, but add structure (which factors? which aspects?)

---

## Common Mistakes

| Mistake | Example | Why Bad | How to Fix |
|---------|---------|---------|------------|
| **Too broad** | "How does AI impact society?" | Unanswerable in PhD timeline | Apply PICO: specific AI tech, specific societal aspect, specific population |
| **Too narrow** | "Accuracy on question 37 at temp 0.73" | Trivial, doesn't contribute | Ask: Why does this matter? What larger question does it serve? |
| **Unfalsifiable** | "What is the best algorithm?" | "Best" is context-dependent | Operationalize: "How do algorithms X, Y, Z compare on metrics A, B in context C?" |
| **Multiple questions** | "How does X affect Y and Z and what about W?" | Actually 3+ separate questions | Count "and" clauses; choose ONE core question |
| **Solution-first** | "How can transformers improve forecasting?" | Assumes solution before investigating problem | Reframe: "What limits current forecasting?" Then test transformers as one approach |
| **Undefined criteria** | "How do users interact with chatbots?" | No success criteria | Specify: "What factors influence error rates in chatbot interactions?" |
| **Not falsifiable** | "Should we use AI for hiring?" | Normative, no empirical test | Reframe empirically: "What are accuracy-fairness tradeoffs in AI hiring systems?" |
| **No boundaries** | "Effects of X on Y" (all effects, all contexts) | Infinite scope | Bound: timeframe, population, geography, specific effects |

---

## "And" Clause Analysis Method

When evaluating a question with multiple "and" clauses:

### Step 1: List Each Clause as Separate Question

**Original**: "How do architectures affect training time AND accuracy AND what are implications for edge deployment AND how does compression interact?"

**Separated**:
1. "How do architectures affect training time?"
2. "How do architectures affect accuracy?"
3. "What are implications for edge deployment?"
4. "How does compression interact with architectures?"

### Step 2: Evaluate Each as Standalone

Can each be a complete research question by itself?
- (1) Yes - complete question
- (2) Yes - complete question
- (3) Yes - complete question
- (4) Yes - complete question

**Result**: 4 separate questions, not 1.

### Step 3: Choose Primary Question

Which ONE is the core contribution?
- If (1) and (2) are your focus, combine them: "How do architectures trade off training time and accuracy?"
- Others become future work or sub-questions IF directly necessary

### Step 4: Test Necessity of Sub-Questions

For each sub-question: "Can I answer my primary question WITHOUT this?"
- If YES → It's a separate question (save for future work)
- If NO → It's a necessary sub-question (include)

**Example**: If studying architecture-accuracy tradeoffs, do you NEED compression interaction?
- If NO → Remove it (separate future question)
- If YES → Include as sub-question: "How do architectures trade off accuracy and training time, and how does compression affect these tradeoffs?"

---

## Rationalization Table - Don't Do These

| Excuse | Reality | What To Do |
|--------|---------|------------|
| "Important topics deserve broad questions" | Broad = unanswerable in PhD timeline | Use PICO to narrow to specific aspect of important topic |
| "Narrowing limits impact" | Depth in narrow area > superficial coverage of broad area | One focused contribution > scattered incomplete work |
| "Comprehensive questions are better" | Comprehensive = unfocused | PhD = 1 core question with depth |
| "Specificity limits publishability" | Specific questions get clearer answers = better papers | Vague questions get vague answers = weak papers |
| "PhD should cover everything about topic" | Impossible in 3-5 years | PhD = focused contribution, not encyclopedia |
| "'Best' can be defined in methodology" | "Best" is still context-dependent | Reframe as comparative with specific metrics |
| "These questions are all related" | Related ≠ one question | Count "and" clauses; choose primary |
| "I should leverage my expertise in X" | Research is problem-first, not solution-first | Start with problem; X becomes one approach to test |
| "Exploratory research doesn't need specific questions" | Exploratory still needs boundaries | "What factors influence X?" is exploratory but specific |
| "Qualitative research is open-ended" | Qualitative ≠ undefined | "How do users experience X?" has clear success criteria |

---

## Quick Reference

### From Broad to Specific (One-Page Workflow)

1. **Start**: "I'm interested in [broad topic]"
2. **Apply PICO**:
   - Population: Who/what specifically?
   - Intervention: What specifically?
   - Comparison: Compared to what?
   - Outcome: What do you measure?
3. **Write question**: Combine PICO components
4. **Check FINER**:
   - Feasible? Interesting? Novel? Ethical? Relevant?
5. **Verify checklist**: Specific, answerable, scoped, single, problem-first, bounded
6. **Count "and" clauses**: Each = potential separate question; choose primary
7. **Test success criteria**: How will you know you've answered it?

### Red Flags Checklist

- 🚨 Too broad (can't answer in 3-5 years)
- 🚨 Too narrow (trivial, single measurement)
- 🚨 "What is best" (unfalsifiable)
- 🚨 Multiple "and" clauses (multiple questions)
- 🚨 Solution-first framing ("How can X...")
- 🚨 Undefined success criteria (don't know what counts as answer)

---

## Examples

### Example 1: Too Broad → Specific

**User**: "I want to study how AI impacts society for my PhD. What should my research question be?"

**Response**:

STOP - "AI impacts society" is a topic, not a question. It's far too broad for a PhD. Let's apply PICO to narrow:

**PICO Questions**:
- **P**: Which AI technology? (LLMs, recommendation systems, autonomous vehicles, hiring algorithms, healthcare diagnostics?)
- **I**: Which societal aspect? (employment, privacy, education, healthcare access, inequality, misinformation?)
- **C**: Compared to what? (Pre-AI systems, human decisions, different AI approaches?)
- **O**: What matters? (Outcomes, equity, adoption, displacement, accuracy?)

**Example Narrowing**:

Let's say you care about employment + AI hiring tools:

- **P**: Tech company recruitment in US
- **I**: AI-powered hiring algorithms
- **C**: Traditional resume screening by humans
- **O**: Demographic diversity of hired candidates

**Specific Question**: "How has the adoption of AI-powered hiring tools affected demographic diversity in tech company recruitment in the US from 2018-2024, compared to traditional human-based screening?"

**FINER Check**:
- ✅ Feasible: Public data on tech hiring, EEOC reports, company diversity reports
- ✅ Interesting: Yes - fairness in hiring is hot topic
- ✅ Novel: Longitudinal analysis 2018-2024 could be new
- ✅ Ethical: Uses public data
- ✅ Relevant: Active policy debate

This is answerable in a PhD timeline.

---

### Example 2: "What is Best" → Comparative

**User**: "My research question is: What is the best machine learning algorithm for fairness?"

**Response**:

STOP - "best" is unanswerable without context. Fairness is multidimensional and context-dependent:
- "Best" for hiring vs. lending vs. criminal justice?
- "Best" by which fairness metric? (demographic parity, equalized odds, individual fairness - these can conflict)
- "Best" trading off how much accuracy?

**Reframe as comparative with specific context**:

Instead of: "What is best?"
Ask: "How do X, Y, Z compare on metrics A, B in context C?"

**Example Reframe**:
"How do fairness-aware algorithms (adversarial debiasing, reweighting, post-processing) compare to standard logistic regression on demographic parity and accuracy metrics in hiring classification tasks?"

**Why this is answerable**:
- ✅ Specific algorithms (X, Y, Z)
- ✅ Specific metrics (demographic parity, accuracy)
- ✅ Specific context (hiring classification)
- ✅ Shows tradeoffs (not declaring "best")

Your research reveals tradeoffs, not winners.

---

### Example 3: Multiple Questions → Primary Question

**User**: "My research question is: How do neural network architectures affect both training time and inference accuracy and what are the implications for edge deployment and how does this interact with model compression techniques?"

**Response**:

STOP - Let me unpack the separate questions here:

1. "How do architectures affect training time?"
2. "How do architectures affect inference accuracy?"
3. "What are implications for edge deployment?"
4. "How does compression interact with architectures?"

These are FOUR separate questions. For a PhD, choose ONE as your primary question.

**Choosing Primary Question**:

Option A: If you care about architecture-performance tradeoffs:
- **Primary**: "How do neural network architectures trade off training time and inference accuracy in image classification?"
- Edge + compression = future work or motivation (not core contribution)

Option B: If you care about edge deployment specifically:
- **Primary**: "What architectural properties enable efficient inference on edge devices for computer vision tasks?"
- Training time and compression become sub-questions IF necessary to answer primary

**Test**: Can you answer your primary question without studying compression?
- If YES → Compression is a separate question (save for later)
- If NO → Include as sub-question (but justify why it's necessary)

One PhD = one focused contribution with depth.

---

## Summary

**Research questions define what you'll investigate - they come BEFORE hypotheses and methods.**

**Core rules**:
1. Use PICO to force specificity (Population, Intervention, Comparison, Outcome)
2. Use FINER to evaluate quality (Feasible, Interesting, Novel, Ethical, Relevant)
3. PhD = 1 core question (3-5 years to answer)
4. Count "and" clauses - each potentially separates questions
5. Problem-first, not solution-first (investigate problem, not predetermined solution)
6. Operationalize subjective terms ("best" → "compare on metrics X, Y in context Z")
7. Define success criteria (how will you know you've answered it?)
8. Bound your question (timeframe, population, geography, specific outcomes)
9. Check answerable question checklist (9 criteria)
10. Too broad > refine with PICO; too narrow > ask what larger question it serves

**Meta-rule**: If you can't articulate how you'd know you've answered your question, it's not well-formed yet.
