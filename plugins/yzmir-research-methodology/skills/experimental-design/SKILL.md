---
name: experimental-design
description: Use when planning studies to test hypotheses, need rigorous causal inference, or designing experiments - prevents confounding, missing controls, underpowered studies, and validity threats using randomization and systematic design frameworks
---

# Experimental Design

## Overview

Experimental design determines whether you can make **causal claims** from your research. Good design isolates the effect of your intervention/treatment while controlling confounds. Poor design leads to ambiguous results that can't distinguish correlation from causation.

**Core Principle**: Randomized controlled trials (RCT) = gold standard for causation. Without random assignment, you have a quasi-experimental or observational study (weaker causal inference, but still valuable with appropriate controls).

## When to Use

Use this skill when:
- Designing studies to test hypotheses
- Planning experiments for causal inference
- Need to control confounding variables
- Determining sample size requirements
- Evaluating existing study designs
- Choosing between experimental designs

**Don't use for**: Descriptive studies (no causal claims), purely observational research without intervention

---

## Critical Red Flags - STOP Before Accepting Design

### 🚨 No Control Group

**If study design has**:
- Single group with intervention only
- "I'll measure if X improves after my intervention"
- Pre/post design without control
- "Published baselines are my control"

**STOP. Control group is NOT optional for causal claims.**

**Why**: Without control, you can't distinguish:
- Effect of YOUR intervention vs. time passing
- Effect of YOUR intervention vs. other co-occurring changes
- Effect of YOUR intervention vs. regression to mean
- Effect of YOUR intervention vs. placebo/Hawthorne effect

**Require**: Concurrent control group (run at same time, same conditions, only difference is your intervention)

---

### 🚨 No Randomization (When Possible)

**If study design uses**:
- Self-selection into groups ("students choose which class")
- Convenience assignment ("treatment group = volunteers")
- Historical controls ("before AI" vs "after AI")
- Existing groups ("Company A has AI, Company B doesn't")

**STOP. Random assignment = gold standard.**

**Why**: Without randomization:
- Groups may differ BEFORE intervention (selection bias)
- Can't assume groups are equivalent
- Confounding variables correlated with treatment assignment
- Causal claims require much stronger assumptions

**If can't randomize**: Use quasi-experimental design with explicit controls (see Quasi-Experimental Designs section)

---

### 🚨 Confounding Variables Not Identified/Controlled

**If study design doesn't account for**:
- Variables that could explain the outcome besides treatment
- Alternative explanations for observed effects
- Systematic differences between groups
- Temporal confounds (things changing over time)

**STOP. Enumerate and control confounds.**

**Why**: Confounding = alternative explanation. Your results are ambiguous if confounds aren't addressed.

**Require**:
- List potential confounds BEFORE conducting study
- Control through: randomization, matching, statistical controls, design features
- Acknowledge remaining confounds in limitations

---

### 🚨 No Power Analysis

**If study design specifies N without justification**:
- "We'll collect 20 participants"
- "Sample size based on convenience"
- "We'll see if there's an effect"

**STOP. Power analysis is required BEFORE data collection.**

**Why**: Underpowered studies:
- Can't detect real effects (Type II error)
- Null results are uninterpretable ("no effect" or "not enough power"?)
- Waste resources testing hypotheses you can't adequately test

**Require**: A priori power analysis:
- Expected effect size (from literature/pilot)
- Desired power (0.80 standard)
- Significance level (α = 0.05)
- Calculate required N

---

### 🚨 Internal vs External Validity Confusion

**If claiming both without acknowledging tradeoff**:
- "My controlled lab study generalizes to all real-world uses"
- "My field study establishes clear causation"
- "I have high internal AND external validity"

**STOP. Recognize the tradeoff.**

**Why**:
- **Internal validity** (causal clarity) requires control → artificial conditions
- **External validity** (generalizability) requires realism → less control
- More of one usually means less of the other

**Require**: Explicitly prioritize based on research goal and acknowledge tradeoff

---

## Experimental Design Hierarchy

Designs ordered by causal inference power (strongest to weakest):

### 1. Randomized Controlled Trial (RCT) - GOLD STANDARD

**Components**:
- **Random assignment** to conditions
- **Control group** (no treatment, placebo, or standard treatment)
- **Manipulation** of independent variable
- **Measurement** of dependent variable

**Example**: Randomly assign 100 companies to "AI hiring tools" vs "standard human screening", measure diversity outcomes

**Causal inference power**: Strongest - random assignment ensures groups equivalent on average

**When to use**: When you can randomize and have access to intervention/treatment

---

### 2. Quasi-Experimental Design

**Components**:
- **NO random assignment** (selection into groups not controlled)
- **Some control** for confounds (through design or analysis)
- **Comparison group** (but not randomly assigned)

**Examples**:
- Difference-in-differences: Companies adopting AI vs. not, before/after
- Regression discontinuity: Eligibility cutoff creates natural experiment
- Propensity score matching: Statistical matching on observed covariates

**Causal inference power**: Moderate - depends on quality of controls

**When to use**: When can't randomize but can identify comparison groups

---

### 3. Observational Study

**Components**:
- **No manipulation** of treatment
- **No randomization**
- **Observation** of naturally occurring variation

**Example**: Survey companies about AI adoption and diversity, look for correlations

**Causal inference power**: Weakest - many potential confounds

**When to use**: When intervention/randomization impossible, exploratory research, describing phenomena

---

## Control Groups: Types and Requirements

### Types of Control Groups

| Control Type | Description | When to Use |
|--------------|-------------|-------------|
| **No treatment** | Group receives no intervention | Testing if intervention has ANY effect vs. nothing |
| **Placebo** | Group receives inert intervention | When placebo effects expected (attention, expectation) |
| **Standard treatment** | Group receives current best practice | Testing if NEW intervention beats CURRENT standard |
| **Wait-list** | Group receives intervention later | Ethical concerns about withholding treatment |

### Control Group Requirements

- ✅ **Concurrent**: Run at same time as treatment group (not historical)
- ✅ **Equivalent**: Groups should be similar BEFORE treatment (randomization ensures this)
- ✅ **Same conditions**: Same measurement procedures, same timeline, same setting
- ✅ **Only difference**: Whether they receive the intervention

**Rationalization to reject**: "Published baselines are my control"

**Reality**: Published baselines:
- Different datasets, different settings, different time periods
- Not concurrent with your experiment
- Can't control for confounds

---

## Randomization: Gold Standard for Causation

### Why Randomization Works

**Random assignment** ensures:
- Groups equivalent **on average** before treatment
- Confounds equally distributed across groups
- No systematic differences besides treatment

**Key insight**: You don't need to MEASURE every confound - randomization controls for measured AND unmeasured confounds

### When You Can't Randomize

**Valid reasons**:
- Ethical constraints (can't randomly assign to harmful condition)
- Practical impossibility (can't randomly assign countries to policies)
- Treatment already occurred (historical analysis)

**Invalid reasons**:
- "Too hard" (often possible with creativity)
- "People should choose" (preference != ethical requirement)
- "Real-world doesn't randomize" (that's why experiments are valuable)

**If can't randomize**: Use quasi-experimental design (see next section) and acknowledge limitations

---

## Quasi-Experimental Designs

When randomization isn't possible, these designs provide some control:

### Difference-in-Differences (DID)

**Structure**: Compare (treatment - control) BEFORE vs AFTER intervention

**Requirements**:
- Treatment and control groups
- Measurements before AND after intervention
- **Parallel trends assumption**: Groups would have changed similarly without treatment

**Example**: Companies adopting AI (treatment) vs not (control), diversity measured 2018 (before) and 2022 (after)

**Threats**: Parallel trends violated, treatment timing not exogenous

---

### Regression Discontinuity Design (RDD)

**Structure**: Treatment assignment based on cutoff score; compare units just above/below cutoff

**Requirements**:
- Clear assignment cutoff (e.g., score >50 gets treatment)
- Units near cutoff are similar
- No manipulation of assignment variable

**Example**: Companies with >500 employees required to report diversity (treatment), compare 490-510 employees

**Threats**: Manipulation of cutoff, non-linearity around cutoff

---

### Propensity Score Matching (PSM)

**Structure**: Statistically match treatment and control units on observed covariates

**Requirements**:
- Measure confounds that predict treatment assignment
- Calculate propensity scores (probability of treatment)
- Match units with similar propensity scores

**Example**: Match companies using AI to similar companies not using AI (on size, industry, prior diversity)

**Threats**: Unobserved confounds, model misspecification

---

### Interrupted Time Series (ITS)

**Structure**: Many measurements before/after intervention, look for level/trend change

**Requirements**:
- Many time points before intervention (≥8)
- Many time points after intervention (≥8)
- Intervention timing clear

**Example**: Monthly diversity metrics 2015-2019 (before AI), 2020-2024 (after AI)

**Threats**: Other interventions at same time, secular trends

---

## Threats to Validity (Campbell & Stanley)

Systematic framework for identifying design flaws:

### Internal Validity: Can you establish causation?

| Threat | Description | How to Control |
|--------|-------------|----------------|
| **History** | External events during study affect outcome | Random assignment, control group |
| **Maturation** | Participants change naturally over time | Control group, shorter study |
| **Testing** | Pre-test affects post-test scores | No pre-test, or control group |
| **Instrumentation** | Measurement changes over time | Standardize measures, calibrate |
| **Regression to mean** | Extreme scores move toward average | Random assignment, control group |
| **Selection** | Groups differ before treatment | Random assignment |
| **Attrition** | Differential dropout between groups | Minimize dropout, check if random |

**Gold standard (RCT) controls most of these through randomization + control group.**

---

### External Validity: Do results generalize?

| Threat | Description | How to Address |
|--------|-------------|----------------|
| **Population** | Sample differs from target population | Representative sampling, replication |
| **Setting** | Lab differs from real-world | Field experiments, ecological validity |
| **Treatment** | Intervention as implemented differs from theory | Treatment fidelity checks |
| **Outcome** | Measures don't reflect real-world outcomes | Meaningful outcome measures |

**Tradeoff**: More control (internal validity) → less realistic (external validity)

---

### Construct Validity: Are you measuring what you claim?

- Are your operational definitions valid?
- Do measures actually tap the constructs?
- Are there alternative interpretations of constructs?

**Example**: "AI hiring improves fairness" - how is "fairness" operationalized? Multiple definitions exist.

---

### Statistical Conclusion Validity: Are statistical inferences correct?

| Threat | Description | How to Control |
|--------|-------------|----------------|
| **Low power** | Can't detect real effects | Power analysis, adequate N |
| **Violated assumptions** | Statistical test assumptions not met | Check assumptions, robust methods |
| **Fishing** | Testing many hypotheses, some significant by chance | Pre-registration, multiple comparison correction |
| **Unreliable measures** | Measurement error obscures effects | Reliable instruments, multiple measures |

---

## Power Analysis

### A Priori Power Analysis (REQUIRED)

**Before data collection**, calculate required sample size:

**Inputs**:
1. **Significance level** (α): Usually 0.05
2. **Desired power** (1-β): Usually 0.80 (80% chance of detecting real effect)
3. **Expected effect size**: From literature, pilot study, or minimum meaningful effect

**Output**: Required N per group

**Formula** (simplified for t-test):
```
N per group ≈ 16 / (effect size)²

For effect size d=0.5 (medium): N ≈ 16 / 0.25 = 64 per group
```

**Effect size guidelines** (Cohen's d):
- Small: d = 0.2 (requires N ≈ 400 per group for 80% power)
- Medium: d = 0.5 (requires N ≈ 64 per group)
- Large: d = 0.8 (requires N ≈ 25 per group)

---

### Post-Hoc Power is Meaningless

**DON'T**: Calculate power after null result

**Why**: Post-hoc power depends on observed effect, which is uncertain. If p>0.05, post-hoc power is necessarily low.

**DO**: Report confidence intervals and effect sizes with null results

---

## Internal vs External Validity Tradeoff

### High Internal Validity (Controlled Experiment)

**Characteristics**:
- Randomization
- Controlled setting (lab)
- Standardized procedures
- Minimal confounds

**Pros**: Clear causal inferences

**Cons**: Artificial conditions, may not generalize

**Example**: Test AI hiring algorithm on carefully balanced, curated dataset in controlled conditions

---

### High External Validity (Field Experiment)

**Characteristics**:
- Realistic setting
- Representative sample
- Natural conditions
- Real-world implementation

**Pros**: Results generalize to target population

**Cons**: Many confounds, weaker causal claims

**Example**: Test AI hiring algorithm in real companies with real applicants

---

### The Tradeoff

**You can't maximize both simultaneously.**

**Design choice depends on research goal**:
- Testing causal mechanism → Prioritize internal validity (controlled experiment)
- Testing real-world effectiveness → Prioritize external validity (field experiment)
- Ideal: Do BOTH sequentially (lab for mechanism, field for effectiveness)

**Don't claim**: "My design has high internal AND external validity" without justification

---

## Common Design Flaws

| Flaw | Example | Why Bad | Fix |
|------|---------|---------|-----|
| **Single-group pre/post** | Measure diversity before/after AI adoption | No control - many confounds | Add control group |
| **Self-selection** | Participants choose treatment vs control | Selection bias | Random assignment |
| **Published baseline** | "My method beats accuracy reported in Paper X" | Different settings, not concurrent | Implement baseline in YOUR study |
| **Underpowered** | N=20 testing medium effect | Can't detect real effects | Power analysis, increase N |
| **Historical control** | "Before 2020" vs "after 2020" | Temporal confounds (e.g., COVID) | Concurrent controls |
| **Confounds ignored** | AI adoption + diversity policy change together | Can't isolate AI effect | Control confounds or acknowledge |
| **Validity confused** | "Lab study generalizes perfectly" | Internal ≠ external validity | Acknowledge tradeoff |

---

## Experimental Design Checklist

Before data collection, verify:

- [ ] **Hypothesis stated** (null and alternative)
- [ ] **Control group** (concurrent, equivalent)
- [ ] **Random assignment** (if possible) OR quasi-experimental design (if not)
- [ ] **Confounds identified** (list potential threats)
- [ ] **Confounds controlled** (through design, matching, or statistics)
- [ ] **Power analysis** (required N calculated)
- [ ] **Adequate sample size** (meet power requirements)
- [ ] **Validity prioritized** (internal vs external tradeoff acknowledged)
- [ ] **Threats to validity** (systematic evaluation using Campbell & Stanley)
- [ ] **Pre-registration** (hypothesis, design, analysis plan registered before data)

---

## Rationalization Table - Don't Do These

| Excuse | Reality | What To Do |
|--------|---------|------------|
| "My method obviously works, no control needed" | Without control, can't isolate your method's effect | Require concurrent control group |
| "Published baselines are my control" | Different settings, not concurrent, can't control confounds | Implement baseline in YOUR study |
| "Can't randomize, so study is impossible" | Quasi-experimental designs exist and are valuable | Use DID, RDD, PSM, or ITS with limitations acknowledged |
| "Before/after shows the effect" | Many confounds in time period | Need concurrent control or quasi-experimental design |
| "20 participants is enough" | Intuition doesn't replace power analysis | Calculate required N from effect size |
| "Real-world data is better than experiments" | Real-world = confounded; experiments = causal | Choose design matching research goal |
| "Controlled experiments don't generalize" | Internal vs external validity tradeoff | Prioritize based on goal, acknowledge limitation |
| "Self-selection is more natural" | Natural ≠ unbiased; creates selection bias | Random assignment or statistical matching |
| "Statistical controls fix selection bias" | Only controls for MEASURED confounds | Randomization controls measured + unmeasured |
| "Natural experiments prove causation" | Quasi-experimental ≠ true experiment | Acknowledge weaker causal inferences |

---

## Quick Reference

### Design Selection Flowchart

```
Can you manipulate treatment?
├─ NO → Observational study (correlational)
└─ YES ↓

Can you randomize assignment?
├─ NO → Quasi-experimental (DID, RDD, PSM, ITS)
└─ YES → Randomized Controlled Trial (RCT)

RCT Components:
1. Random assignment to treatment vs control
2. Manipulation of IV
3. Measurement of DV
4. Control for confounds
5. Adequate power (N from power analysis)
```

### Power Analysis Quick Reference

| Effect Size | N per group (80% power, α=0.05) |
|-------------|--------------------------------|
| Small (d=0.2) | ~400 |
| Medium (d=0.5) | ~64 |
| Large (d=0.8) | ~25 |

### Validity Checklist

- **Internal**: Can you establish cause? (controlled confounds?)
- **External**: Do results generalize? (representative sample/setting?)
- **Construct**: Measuring what you claim? (valid operational definitions?)
- **Statistical**: Correct inferences? (adequate power, assumptions met?)

---

## Examples

### Example 1: Missing Control Group

**User**: "I'll test my new fairness-aware training method on hiring algorithms. I'll train models with my method and measure the demographic parity improvement. If parity improves, my method works!"

**Response**:

STOP - this is a single-group design without a control. You cannot make causal claims.

**Problem**: How do you know YOUR method caused improvement? Could be:
- Your hyperparameter tuning (not the fairness method)
- Your data preprocessing (not the fairness method)
- Your dataset happens to have different properties
- Random variation

**Required design**:

**Randomized Controlled Trial**:
1. **Random assignment**: Randomly assign training runs to fairness-aware (treatment) vs standard training (control)
2. **Control confounds**: Same dataset, same architecture, same hyperparameters (only difference = fairness method)
3. **Multiple datasets**: Test on held-out datasets to ensure generalization
4. **Power analysis**: Calculate required N (number of training runs) for adequate power

**Example**:
- Treatment group: 30 training runs with fairness-aware method
- Control group: 30 training runs with standard method
- Same conditions: Same datasets, architectures, hyperparameters
- Measure: Demographic parity AND accuracy (check for tradeoffs)
- Randomize: Random seeds different, but assignment to condition random

Without control group, your result is ambiguous.

---

### Example 2: Confounding Variables

**User**: "I'll compare AI hiring tools vs human recruiters by analyzing company X's hiring data before and after they adopted AI in 2020. If diversity improved, AI helped."

**Response**:

STOP - this before/after design has MANY confounding variables.

**Confounds** (alternative explanations):
1. **COVID-19**: 2020 labor market disrupted, remote work changed applicant pools
2. **Social movements**: 2020 BLM protests → increased diversity focus across companies
3. **Other HR changes**: Company may have implemented diversity training, recruiting changes
4. **Secular trends**: Tech industry diversity increased generally 2020-2024
5. **Applicant pool**: Demographics of applicants may have shifted
6. **Legal/regulatory**: Increased compliance pressure

**Threats to internal validity**:
- **History**: Many external events 2020-2024
- **Maturation**: Company culture evolved naturally
- **Selection**: Companies choosing AI may differ systematically

**Improved quasi-experimental design**:

**Difference-in-Differences (DID)**:
1. **Treatment group**: Company X (adopted AI in 2020)
2. **Control group**: Similar companies that did NOT adopt AI
3. **Matching**: Match on industry, size, pre-2020 diversity
4. **Measurement**: Diversity metrics 2018-2019 (before) and 2021-2024 (after)
5. **Analysis**: (Treatment_after - Treatment_before) - (Control_after - Control_before)
6. **Assumption**: Parallel trends (groups would have changed similarly without AI)

**Acknowledge**: Even with DID, cannot prove causation (only random assignment does that). This is correlational evidence with controls for observed confounds.

---

### Example 3: No Power Analysis

**User**: "I'll collect data from 20 participants and see if my intervention has an effect. If p<0.05, it worked!"

**Response**:

STOP - you need power analysis BEFORE collecting data, and N=20 is likely underpowered.

**Required: A Priori Power Analysis**

**Inputs needed**:
1. **Expected effect size**: What do you expect? Check literature or pilot data.
   - If literature shows d=0.5 (medium effect)
2. **Desired power**: 0.80 (80% chance of detecting real effect)
3. **Significance level**: α = 0.05

**Calculation** (for two-group t-test, d=0.5):
- Required N per group ≈ 64
- Total N ≈ 128

**With N=20 (10 per group)**:
- Can only detect LARGE effects (d ≥ 0.9) with 80% power
- For medium effects (d=0.5), power ≈ 0.17 (17% chance of detecting)
- 83% chance of missing a real medium-sized effect (Type II error)

**Problems with underpowered study**:
- Null result is uninterpretable: "No effect" or "not enough power"?
- Waste resources testing hypothesis you can't adequately test
- Publication bias: Only large effects detected, inflates effect size estimates

**Also required**:
- **Control group**: N=20 total, or 20 per group?
- **Pre-registration**: Commit to hypothesis and analysis before data collection
- **Effect size reporting**: Don't just report p-values

**Correct approach**:
1. Estimate effect size from literature: d=0.5
2. Calculate required N: 64 per group (128 total)
3. If resources limit you to N=20: Re-evaluate if study is feasible
4. Alternative: Test for larger minimum effect size, acknowledge limited power for smaller effects

---

## Summary

**Experimental design determines whether you can make causal claims from your research.**

**Core rules**:
1. RCT (randomized controlled trial) = gold standard for causation
2. Control group is NOT optional for causal claims (concurrent, equivalent)
3. Random assignment controls confounds (measured + unmeasured)
4. Without randomization → quasi-experimental design (weaker causation)
5. Power analysis REQUIRED before data collection (adequate N)
6. Internal vs external validity = tradeoff (choose based on research goal)
7. Threats to validity must be systematically evaluated (Campbell & Stanley)
8. Quasi-experimental designs exist when can't randomize (DID, RDD, PSM, ITS)
9. Published baselines don't replace concurrent controls
10. "Before/after" without control has many confounds

**Meta-rule**: If you can't randomize, acknowledge limitations explicitly. Quasi-experimental and observational designs are valuable, but don't claim causal certainty they can't provide.
