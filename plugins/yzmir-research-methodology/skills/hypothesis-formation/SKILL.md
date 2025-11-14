---
name: hypothesis-formation
description: Use when converting research questions into testable predictions, unclear how to make falsifiable hypotheses, or need null/alternative hypothesis pairs - prevents unfalsifiable claims, question/hypothesis confusion, and missing operational definitions
---

# Hypothesis Formation

## Overview

A **hypothesis** is a specific, falsifiable prediction about the relationship between variables. Hypotheses come AFTER research questions and literature review, and BEFORE experimental design.

**Core Principle**: Hypotheses are testable predictions (declarative statements), NOT questions (interrogative). ALWAYS state both null (H0) and alternative (H1). ALL variables must be operationally defined IN the hypothesis.

## When to Use

Use this skill when:
- Converting research question into testable hypothesis
- Formulating predictions before experiments
- Preparing hypothesis section for proposals/papers
- Need to make vague predictions specific and testable
- Setting up null hypothesis significance testing

**Don't use for**: Research question formulation (that comes first), experimental design (that comes after)

---

## Critical Red Flags - STOP Before Accepting Hypothesis

### 🚨 Question Disguised as Hypothesis

**If hypothesis uses interrogative form**:
- "How do fairness-aware algorithms affect hiring outcomes?"
- "What is the relationship between X and Y?"
- "Does X influence Y?"

**STOP. This is a research QUESTION, not a hypothesis.**

**Difference**:
- **Research Question** (interrogative): "How does X affect Y?"
- **Hypothesis** (declarative prediction): "X will increase Y by 15% in context Z"

**Test**: Does it end with a question mark? → It's a question, not a hypothesis.

**Fix**: Convert interrogative to declarative prediction (see conversion process below).

---

### 🚨 Missing Null Hypothesis

**If only alternative hypothesis (H1) is stated**:
- "Transformers will outperform LSTMs"
- "Fairness constraints will reduce bias"
- Only H1, no H0

**STOP. ALWAYS state BOTH null and alternative.**

**Why**: Statistical testing compares your data against the null hypothesis. Without H0, you can't properly test H1.

**Required format**:
- **H0** (null): No effect, no difference, or status quo
- **H1** (alternative): Your predicted effect, difference, or change

**Example**:
- ❌ "Transformers will outperform LSTMs" (H1 only)
- ✅ **H0**: Transformers and LSTMs have equivalent perplexity on long sequences
- ✅ **H1**: Transformers achieve ≥10% lower perplexity than LSTMs on sequences >500 tokens

---

### 🚨 Unfalsifiable / Vague Predictions

**If hypothesis contains undefined terms**:
- "AI will lead to better outcomes for everyone"
- "Algorithm X improves performance"
- "Model Y is more fair"
- Uses: "better", "improved", "success", "fair", "optimal"

**STOP. These are unfalsifiable as stated.**

**Why**: "Better" by what measure? "Everyone" includes contradictory stakeholders. Can't test what you can't measure.

**Test**: What specific evidence would DISPROVE this? If unclear, it's unfalsifiable.

**Fix**: Operationalize all terms (see operational definitions below).

---

### 🚨 No Operational Definitions

**If hypothesis uses constructs without defining measurement**:
- "Bias" (which bias metric?)
- "Performance" (which performance metric?)
- "Long sequences" (how many tokens?)
- "Fairness" (demographic parity? equalized odds?)

**STOP. Operational definitions belong IN the hypothesis.**

**Common deflection**: "I'll define it in methods section"

**Reality**: Hypothesis should be testable AS STATED. If reader can't tell what you're testing, it's not specific enough.

**Fix**: Define variables in hypothesis:
- "Bias (measured by demographic parity ratio)"
- "Performance (perplexity on held-out test set)"
- "Long sequences (>500 tokens)"

---

### 🚨 No Magnitude or Direction

**If hypothesis predicts effect without specificity**:
- "X will affect Y" (direction unclear)
- "X will improve Y" (how much?)
- "A and B will differ" (in what direction? by how much?)

**STOP. Specify direction AND magnitude when possible.**

**Why**: Vague predictions prevent meaningful evaluation. "Statistically significant" doesn't mean "practically significant."

**Fix**: Add direction and magnitude:
- "X will INCREASE Y by 10-20%"
- "A will achieve ≥15% higher accuracy than B"
- "Method C will REDUCE bias (DPR) to ≥0.8"

---

## Hypothesis Anatomy

A well-formed hypothesis has these components:

### 1. Null Hypothesis (H0)

**Format**: "There is no difference/effect/relationship between X and Y"

**Examples**:
- H0: Fairness-aware and standard algorithms have equivalent accuracy on hiring tasks
- H0: Transformers and LSTMs achieve equivalent perplexity on long-sequence modeling
- H0: There is no correlation between model complexity and fairness metrics

**Purpose**: Default assumption you're trying to disprove with evidence.

---

### 2. Alternative Hypothesis (H1)

**Format**: "X will [increase/decrease] Y by [magnitude] in [context]"

**Components**:
- **Independent variable** (X): What you're manipulating/comparing
- **Dependent variable** (Y): What you're measuring
- **Direction**: Increase, decrease, or differ (if non-directional)
- **Magnitude**: Specific predicted effect size (when possible)
- **Context**: Population, conditions, boundaries
- **Operational definitions**: How variables are measured

**Examples**:
- H1: Fairness-aware algorithms will achieve ≥10 percentage point increase in demographic parity ratio compared to standard algorithms, while maintaining ≥80% accuracy, in tech hiring contexts
- H1: Transformers will achieve ≥15% lower perplexity than LSTMs on sequences longer than 500 tokens, measured on WikiText-103 test set
- H1: There is a negative correlation (r < -0.5) between model complexity (parameter count) and fairness (demographic parity) in hiring algorithms

---

## Research Question → Hypothesis Conversion

Follow this process to convert questions into hypotheses:

### Step 1: Start with Research Question

**Example**: "How do fairness-aware training techniques affect hiring algorithm outcomes?"

This is a good research question (specific, bounded), but it's not a hypothesis yet.

---

### Step 2: Review Literature for Direction

**What does existing research suggest?**
- Do papers show fairness techniques reduce bias? (direction: positive)
- Do they show accuracy-fairness tradeoffs? (competing outcomes)
- Is there consensus or contradiction?

**From literature**: "Fairness constraints typically reduce discrimination but at ~5-15% accuracy cost"

---

### Step 3: Formulate Null Hypothesis (H0)

**H0 = No effect or no difference**

**Example**:
- H0: Fairness-aware training (adversarial debiasing) and standard training produce equivalent demographic parity ratios AND equivalent accuracy in hiring classification tasks

---

### Step 4: Formulate Alternative Hypothesis (H1)

**Based on literature direction, predict specific effect:**

**Example (directional)**:
- H1: Fairness-aware training (adversarial debiasing) will achieve ≥10 percentage point increase in demographic parity ratio compared to standard training, while reducing accuracy by 5-15%, in hiring classification tasks on real-world datasets

**Components check**:
- ✅ Independent variable: Fairness-aware vs standard training
- ✅ Dependent variables: Demographic parity ratio AND accuracy
- ✅ Direction: Increase parity, decrease accuracy
- ✅ Magnitude: ≥10 pp for parity, 5-15% for accuracy
- ✅ Context: Hiring classification, real-world datasets
- ✅ Operational: Demographic parity ratio (measurable)

---

### Step 5: Verify Falsifiability

**Ask**: What evidence would DISPROVE H1?

**Example answers**:
- If fairness-aware training increases parity by <10 pp → H1 disproven
- If accuracy reduction is >15% → H1 disproven
- If parity decreases → H1 disproven

**Result**: ✅ Falsifiable - clear conditions for disproving.

---

## Directional vs Non-Directional Hypotheses

### Use Directional (One-Tailed) When:

**Theory/literature suggests specific direction**:
- Prior research shows X increases Y
- Theoretical framework predicts decrease
- Mechanism is well-understood

**Example**: "Transformers will outperform LSTMs on long sequences"
- ✅ Use directional: Literature clearly shows transformers excel at long-range dependencies

**Statistical implication**: One-tailed test (more power to detect effect in predicted direction)

---

### Use Non-Directional (Two-Tailed) When:

**No theoretical expectation of direction**:
- Novel combination of techniques
- Contradictory prior findings
- Genuinely exploratory investigation

**Example**: "Quantum-inspired optimization will differ from classical optimization on fairness metrics"
- ✅ Use non-directional: No prior work on quantum methods for fairness

**Statistical implication**: Two-tailed test (can detect effects in either direction)

---

### Common Mistake: Hedging with Non-Directional

**Rationalization**: "Non-directional is safer - I don't want to be wrong about direction"

**Reality**: If literature/theory suggests direction, use directional. Non-directional isn't "more rigorous" - it's appropriate when direction is genuinely unknown.

**Don't**: Use non-directional to avoid committing to a direction when you actually have an expectation.

---

## Operational Definitions IN Hypothesis

Don't defer operational definitions to methods section - include them IN hypothesis statement.

### Common Undefined Terms Requiring Operationalization

| Vague Term | Operational Definition Examples |
|------------|--------------------------------|
| **Bias** | Demographic parity ratio, equalized odds, calibration gap |
| **Performance** | Accuracy, F1-score, AUC-ROC, perplexity, BLEU score |
| **Fairness** | Demographic parity, equalized odds, individual fairness (Lipschitz) |
| **Better/Improved** | ≥10% increase in [specific metric] |
| **Long sequences** | Sequences >500 tokens (specify threshold) |
| **High-stakes** | Decisions affecting: employment, lending, healthcare, criminal justice |
| **Everyone** | All demographic groups (specify: race, gender, age?) |
| **Success** | Task completion rate ≥90%, user satisfaction ≥4/5 |

### Example Operationalization

**❌ Vague**: "H1: Fairness interventions reduce bias in hiring algorithms"

**✅ Operationalized**: "H1: Adversarial debiasing will increase demographic parity ratio from baseline 0.7 to ≥0.8 (measured as min(P(hire|protected)/P(hire|non-protected), P(hire|non-protected)/P(hire|protected))) for race and gender in hiring algorithms trained on real-world recruitment data"

---

## Multiple Hypotheses

### Primary vs Secondary Hypotheses

**Primary hypotheses** (1-3 max):
- What study is designed to test
- What sample size is powered for
- Main contribution of research

**Secondary hypotheses** (clearly labeled):
- Exploratory analyses
- Subgroup analyses
- Interaction effects

**Why distinguish**: Multiple primary hypotheses require correction for multiple comparisons (Bonferroni, FDR), reducing statistical power.

---

### Multiple Comparison Problem

**If testing 15 hypotheses at α=0.05**:
- Expected false positives: 15 × 0.05 = 0.75 (likely to find at least one "significant" result by chance)

**Solutions**:
1. **Limit primary hypotheses** to 1-3
2. **Label others as exploratory** (don't claim significance)
3. **Adjust α** for multiple comparisons:
   - Bonferroni: α_adjusted = 0.05 / number of tests
   - FDR (less conservative): Control false discovery rate

**Rationalization to avoid**: "All hypotheses are equally important"

**Reality**: PhD/study has ONE main research question → 1-3 primary hypotheses that directly test it.

---

## Hypothesis Quality Checklist

Before finalizing hypotheses, verify:

- [ ] **Declarative** (statement, not question)
- [ ] **Both H0 and H1** (null and alternative paired)
- [ ] **Falsifiable** (can identify evidence that would disprove)
- [ ] **Specific** (direction and magnitude when possible)
- [ ] **Operationalized** (all variables measurably defined IN hypothesis)
- [ ] **Testable** (you can actually collect this data)
- [ ] **Connected to question** (hypothesis tests research question)
- [ ] **Literature-informed** (direction based on theory/prior work)
- [ ] **Primary limited** (1-3 primary hypotheses max)
- [ ] **Contextually bounded** (population, timeframe, conditions specified)

If ANY checkbox fails, refine the hypothesis.

---

## Common Mistakes

| Mistake | Example | Why Bad | How to Fix |
|---------|---------|---------|------------|
| **Question not hypothesis** | "How does X affect Y?" | Interrogative, not predictive | Convert: "X will increase Y by 15%" |
| **Only H1, no H0** | "Transformers will outperform LSTMs" | Can't do null hypothesis testing | Add H0: "Transformers and LSTMs have equivalent performance" |
| **Unfalsifiable** | "AI improves outcomes for everyone" | Can't measure "everyone", "improve" | Specify: "AI increases accuracy ≥10% for demographic groups A, B, C" |
| **Vague terms** | "Algorithm reduces bias" | "Bias" undefined | Specify bias metric: "... reduces demographic parity ratio gap to ≤0.2" |
| **No magnitude** | "X will improve Y" | How much? | "X will improve Y by 10-20%" |
| **No direction** | "X will affect Y" | Increase or decrease? | "X will INCREASE Y by 15%" |
| **Deferred definitions** | "I'll define bias in methods" | Hypothesis not testable as stated | Define IN hypothesis: "bias (demographic parity ratio)" |
| **Too many primary** | 15 hypotheses, all primary | Multiple comparison problem | 1-3 primary, rest exploratory |
| **Non-directional hedging** | "X and Y will differ" | When literature suggests direction | Use directional when justified |

---

## Rationalization Table - Don't Do These

| Excuse | Reality | What To Do |
|--------|---------|------------|
| "Research question and hypothesis are similar" | Question asks, hypothesis predicts | Distinguish: interrogative vs declarative prediction |
| "Everyone knows what the null is" | Many researchers skip or misstate it | Always explicitly state H0 |
| "Operational definitions go in methods" | Hypothesis should be testable as stated | Define variables IN hypothesis |
| "Non-directional is safer" | Should be theory-driven, not safety-driven | Use directionality when literature/theory suggests |
| "I'll measure multiple outcomes" | Hypothesis picking = p-hacking | Choose ONE primary outcome beforehand |
| "'Better' is clear enough" | Subjective without operationalization | Define: better by what specific measure? |
| "All hypotheses are equally important" | PhD has ONE main question | Designate 1-3 primary, rest exploratory |
| "I don't want to commit to direction" | If literature suggests direction, use it | Directional hypotheses have more power |
| "Qualitative research doesn't need hypotheses" | Different methods, still need predictions | Adapt format, but predictions still needed |
| "Magnitude predictions are too risky" | Vague predictions aren't useful | Estimate from literature, be conservative |

---

## Quick Reference

### Hypothesis Conversion Workflow

1. **Research Question**: "How does X affect Y?"
2. **Review Literature**: What direction does theory/evidence suggest?
3. **Formulate H0**: "X has no effect on Y" or "X and Y are equivalent on outcome Z"
4. **Formulate H1**: "X will [increase/decrease] Y by [magnitude] in [context]"
5. **Operationalize**: Define all variables measurably
6. **Specify**: Add direction, magnitude, boundaries
7. **Test Falsifiability**: What evidence would disprove this?

### Red Flags Checklist

- 🚨 Interrogative form (ends with ?)
- 🚨 Only H1, no H0
- 🚨 Contains "better", "improved", "success", "fair" without defining them
- 🚨 No operational definitions
- 🚨 No magnitude or direction
- 🚨 "Everyone" or other universal claims
- 🚨 Can't identify what would disprove it

---

## Examples

### Example 1: Question → Hypothesis Conversion

**User**: "My hypothesis is: How do fairness-aware algorithms affect hiring outcomes? I'm ready to start my experiment."

**Response**:

STOP - this is a research QUESTION, not a hypothesis. It's interrogative ("How do...?"), not a predictive statement.

**Conversion process**:

**Step 1: Research Question** (you have this):
"How do fairness-aware algorithms affect hiring outcomes?"

**Step 2: Review Literature**:
Existing research suggests fairness-aware training reduces demographic disparity but often at ~5-15% accuracy cost.

**Step 3: Formulate Null (H0)**:
"H0: Fairness-aware algorithms (adversarial debiasing) and standard algorithms produce equivalent demographic parity ratios AND equivalent accuracy on hiring classification tasks."

**Step 4: Formulate Alternative (H1)**:
"H1: Fairness-aware algorithms (adversarial debiasing) will achieve ≥10 percentage point increase in demographic parity ratio compared to standard algorithms, while reducing accuracy by 5-15%, on real-world hiring datasets (evaluated on protected classes: race, gender)."

**Step 5: Verify Falsifiability**:
- If parity increases by <10 pp → H1 disproven
- If accuracy drops >15% → H1 disproven
- If parity decreases → H1 disproven

✅ Now you're ready to start your experiment.

---

### Example 2: Unfalsifiable → Falsifiable

**User**: "My hypothesis is that using AI in hiring will lead to better outcomes for everyone. I'll measure success and fairness."

**Response**:

STOP - "better outcomes for everyone" is unfalsifiable as stated.

**Problems**:
1. "Better" - by what measure? (accuracy? speed? satisfaction? retention?)
2. "Everyone" - employers AND candidates? (These stakeholders have conflicting interests)
3. "Success" - undefined
4. "Fairness" - which fairness metric? (There are many, and they can conflict)

**Operationalization needed**:

**Revised H0**:
"H0: AI-powered hiring algorithms and human resume screening produce equivalent demographic parity ratios AND equivalent time-to-hire metrics."

**Revised H1**:
"H1: AI-powered hiring algorithms will reduce time-to-hire by ≥20% compared to human screening, while maintaining demographic parity ratio ≥0.8 for protected classes (race, gender), measured in Fortune 500 tech company recruitment from 2020-2024."

**Why this works**:
- ✅ "Better" operationalized: ≥20% faster hiring
- ✅ "Everyone" operationalized: Employers (speed) + candidates (fairness ≥0.8)
- ✅ "Success" and "fairness" defined with specific metrics
- ✅ Falsifiable: Clear conditions that would disprove

---

### Example 3: Adding Null Hypothesis

**User**: "My hypothesis is that transformers outperform LSTMs on long-sequence modeling. I'll test this and publish if it's true."

**Response**:

This is a good start, but you're missing the null hypothesis (H0). ALWAYS state both null and alternative.

Also, let's add operational precision:

**H0 (null hypothesis)**:
"Transformers and LSTMs achieve equivalent perplexity on long-sequence language modeling tasks (sequences >500 tokens)."

**H1 (alternative hypothesis)**:
"Transformers achieve ≥10% lower perplexity than LSTMs on long-sequence language modeling tasks (sequences >500 tokens), measured on WikiText-103 test set."

**Why both are needed**:
- H0: What you're testing AGAINST (default assumption)
- H1: What you're testing FOR (your prediction)
- Statistical test compares data to H0 to evaluate H1

**Operational precision added**:
- "Outperform" → "≥10% lower perplexity" (specific metric + magnitude)
- "Long-sequence" → ">500 tokens" (specific boundary)
- Added dataset: "WikiText-103 test set" (reproducibility)

**Note on "publish if true"**: Be careful of publication bias. You should publish regardless of outcome - null results matter. A well-designed study showing NO difference is valuable.

---

## Summary

**Hypotheses are testable predictions that come AFTER research questions and BEFORE experiments.**

**Core rules**:
1. Hypothesis = declarative prediction, NOT interrogative question
2. ALWAYS state both null (H0) and alternative (H1)
3. Operationalize ALL variables IN the hypothesis (don't defer to methods)
4. Specify direction AND magnitude when literature/theory supports it
5. Use directional when theory suggests; non-directional when truly exploratory
6. Falsifiable = you can identify evidence that would disprove it
7. Primary hypotheses limited to 1-3 (prevent multiple comparison problem)
8. Connect to research question (hypothesis tests the question)
9. All terms measurably defined (bias, performance, fairness, better, etc.)
10. Null (H0) = no effect/no difference; Alternative (H1) = specific predicted effect

**Meta-rule**: If you can't identify what evidence would disprove your hypothesis, it's not falsifiable yet. Keep refining.
