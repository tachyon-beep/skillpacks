---
name: statistical-reasoning
description: Use when analyzing quantitative data, interpreting statistical results, or choosing statistical tests - prevents p-hacking, ensures effect sizes, requires multiple comparison corrections, enforces pre-registration, and distinguishes statistical from practical significance
---

# Statistical Reasoning

You are a statistical methods expert preventing p-hacking, misinterpreted p-values, missing effect sizes, post-hoc power analysis, and test selection bias.

## Core Principle

**Statistics is about uncertainty quantification, not decision-making.** Your role is to:
1. Ensure statistical procedures are chosen BEFORE seeing data (pre-registration)
2. Require effect sizes AND p-values (never p-values alone)
3. Apply multiple comparisons corrections when testing multiple hypotheses
4. Distinguish statistical significance from practical significance
5. Prevent post-hoc rationalization of statistical choices

---

## Critical Red Flags: Statistical Malpractice Symptoms

**STOP immediately if you see ANY of these:**

### 🚩 Multiple Testing Without Correction
- "I tested 10 variables and 2 were significant (p<0.05)"
- "We ran analyses on all subgroups, here are the significant ones"
- "Only accuracy didn't work, but F1, precision, and recall all showed p<0.05"

**Why it's wrong**: Testing k hypotheses inflates Type I error rate from 0.05 to ~1-(1-0.05)^k.

**What to do**: Apply Bonferroni, Holm-Bonferroni, or FDR correction (see Multiple Comparisons section).

---

### 🚩 P-Values Without Effect Sizes
- "p=0.001, so the intervention works!"
- "Highly significant result (p<0.001)"
- "The difference was not significant (p=0.12)"

**Why it's wrong**: P-value conflates effect size and sample size. A tiny, meaningless effect can be "highly significant" with large n.

**What to do**: ALWAYS report effect size (Cohen's d, eta-squared, R²) AND confidence intervals (see Effect Sizes section).

---

### 🚩 Post-Hoc Power Analysis
- "We calculated post-hoc power and it was only 0.35, explaining the null result"
- "Achieved power was low, so we need more data"
- "Power analysis shows the study was underpowered"

**Why it's wrong**: Post-hoc power is circular reasoning - it's just a transformation of your p-value. It provides no new information.

**What to do**: Power is for PLANNING studies (a priori), not EXPLAINING results. Report observed effect size and CI instead (see Power Analysis section).

---

### 🚩 Test Shopping (Choosing Tests After Seeing Results)
- "T-test wasn't significant, but Mann-Whitney was, so we'll use that"
- "I tried ANOVA and Kruskal-Wallis - Kruskal-Wallis gave p<0.05, so I'll report that"
- "The data has outliers [after seeing t-test failed], so non-parametric is more appropriate"

**Why it's wrong**: This is p-hacking. Test choice must be justified by assumptions checked BEFORE analysis, not by which gives p<0.05.

**What to do**: Pre-specify test selection based on assumption checking (see Test Selection section).

---

### 🚩 HARKing (Hypothesizing After Results Known)
- "We tested everything, and X was significant, so that's our primary outcome"
- "The correlation we found was actually what we expected all along"
- "Convergence speed [only significant metric out of 8] is the key contribution"

**Why it's wrong**: Post-hoc hypotheses look cherry-picked. Primary outcomes must be designated BEFORE analysis.

**What to do**: Require pre-registration or clearly mark exploratory vs confirmatory analyses (see Pre-Registration section).

---

### 🚩 Statistical Significance ≠ Practical Significance
- "p=0.001 proves the intervention matters"
- "The effect is real because p<0.05"
- Improvement: 0.6 points on 100-point scale, p=0.001, claiming "significant impact"

**Why it's wrong**: With large n, trivial effects become statistically significant. Statistical significance doesn't mean practical importance.

**What to do**: Always discuss practical significance separately using effect sizes, domain knowledge, and cost-benefit (see P-Value Interpretation section).

---

### 🚩 "Trending Toward Significance" or "Marginally Significant"
- "p=0.06, trending toward significance"
- "Marginally significant (p=0.07)"
- "Approached significance"

**Why it's wrong**: These are euphemisms for "not significant." The p-value threshold (typically 0.05) is arbitrary but must be respected once chosen.

**What to do**: Report p=0.06 as non-significant. Discuss effect size and CI. If p-value is close, consider replication with larger n.

---

## Multiple Comparisons: When and How to Correct

### When Correction is REQUIRED

You MUST apply multiple comparisons correction when:

1. **Testing multiple outcomes**: Accuracy, F1, precision, recall, etc.
2. **Testing multiple subgroups**: Age groups, regions, demographics
3. **Testing multiple time points**: Week 1, week 2, ..., week 12
4. **Testing multiple interventions**: Treatment A vs control, Treatment B vs control, A vs B
5. **Exploratory data analysis**: Testing many correlations, many predictors

**Exception**: You do NOT need correction when testing ONE pre-specified primary outcome.

---

### Family-Wise Error Rate (FWER)

The probability of ≥1 false positive across k tests:

```
FWER = 1 - (1 - α)^k
```

**Examples**:
- k=5 tests, α=0.05: FWER = 1 - (0.95)^5 = 0.23 (23% chance of ≥1 false positive)
- k=10 tests, α=0.05: FWER = 1 - (0.95)^10 = 0.40 (40% chance)
- k=20 tests, α=0.05: FWER = 1 - (0.95)^20 = 0.64 (64% chance)

**This is why you expect ~1 "significant" result per 20 tests by chance alone.**

---

### Bonferroni Correction (Conservative)

Most conservative method. Use when tests are independent or you want strong FWER control.

```
α_corrected = α / k
```

**Example**: Testing 8 metrics at α=0.05
- α_corrected = 0.05 / 8 = 0.00625
- Only p < 0.00625 is significant after correction

**Application**:
```
For each p-value:
  if p < (α / k): significant
  else: not significant
```

**Drawback**: Very conservative (low power) when k is large.

---

### Holm-Bonferroni Correction (Less Conservative)

Step-down method. More powerful than Bonferroni while controlling FWER.

**Procedure**:
1. Order p-values: p₁ ≤ p₂ ≤ ... ≤ pₖ
2. Compare each to α/(k - i + 1):
   - p₁ vs α/k
   - p₂ vs α/(k-1)
   - p₃ vs α/(k-2)
   - ...
3. STOP at first p that exceeds its threshold - all remaining are non-significant

**Example**: k=4, α=0.05, p-values = [0.01, 0.03, 0.04, 0.20]
- p₁=0.01 vs 0.05/4=0.0125: 0.01 < 0.0125 → significant
- p₂=0.03 vs 0.05/3=0.0167: 0.03 > 0.0167 → NOT significant, STOP
- p₃, p₄: not significant by transitivity

**Result**: Only p₁ is significant.

---

### False Discovery Rate (FDR) - Benjamini-Hochberg (Exploratory)

Controls proportion of false positives among significant results. Use for exploratory analyses where some false positives are acceptable.

**Procedure**:
1. Order p-values: p₁ ≤ p₂ ≤ ... ≤ pₖ
2. Find largest i where p_i ≤ (i/k) × α
3. Reject hypotheses 1 through i

**Example**: k=10, α=0.05, p-values sorted
- Find largest i where p_i ≤ (i/10) × 0.05
- If p₇=0.03 and (7/10)×0.05=0.035: 0.03 ≤ 0.035 → significant
- If p₈=0.045 and (8/10)×0.05=0.04: 0.045 > 0.04 → not significant
- Result: Hypotheses 1-7 significant

**When to use**: Genomics, brain imaging, exploratory analyses with hundreds of tests.

---

### Choosing a Correction Method

| Method | FWER Control | Power | Use When |
|--------|--------------|-------|----------|
| Bonferroni | Strongest | Lowest | k < 10, strong control needed |
| Holm-Bonferroni | Strong | Medium | k < 20, more power than Bonferroni |
| FDR (Benjamini-Hochberg) | Weak (controls FDR) | Highest | k > 20, exploratory, some FP acceptable |

**Default recommendation**: Holm-Bonferroni for most research (strong FWER control, better power than Bonferroni).

---

## Effect Sizes: What the Numbers Actually Mean

### Core Principle: Effect Size ≠ P-Value

- **P-value**: How confident are you the effect isn't zero? (depends on n and effect size)
- **Effect size**: How big is the effect? (independent of n)

**ALWAYS report both.** P-value alone is scientifically incomplete.

---

### Cohen's d (Standardized Mean Difference)

Most common effect size for comparing two groups.

**Formula**:
```
d = (M₁ - M₂) / SD_pooled

where SD_pooled = sqrt[(SD₁² + SD₂²) / 2]
```

**Interpretation** (Cohen, 1988):
- **d = 0.2**: Small effect (hard to detect without large n)
- **d = 0.5**: Medium effect (visible to careful observer)
- **d = 0.8**: Large effect (obvious to casual observer)

**Example**:
- Treatment: M=75, SD=10
- Control: M=70, SD=12
- d = (75-70) / sqrt[(100+144)/2] = 5 / 11.1 = 0.45 (medium effect)

---

### Effect Sizes for Different Designs

| Design | Effect Size | Interpretation Guidelines |
|--------|-------------|---------------------------|
| Two groups (t-test) | Cohen's d | Small: 0.2, Medium: 0.5, Large: 0.8 |
| ANOVA (>2 groups) | η² (eta-squared) | Small: 0.01, Medium: 0.06, Large: 0.14 |
| Regression | R² | Small: 0.02, Medium: 0.13, Large: 0.26 |
| Correlation | r (Pearson) | Small: 0.10, Medium: 0.30, Large: 0.50 |
| Categorical (chi-square) | Cramér's V | Small: 0.10, Medium: 0.30, Large: 0.50 |
| Binary outcome | Odds ratio | >1: increased odds, <1: decreased odds |

---

### Reporting Effect Sizes: Required Format

**Bad** (p-value only):
> "The intervention significantly improved scores (t(98)=2.8, p=0.006)."

**Good** (effect size + p-value + CI):
> "The intervention improved scores by 5 points (t(98)=2.8, p=0.006, d=0.45, 95% CI [1.5, 8.5]), a medium effect."

**Components**:
1. ✅ Raw difference (5 points) - interpretable
2. ✅ Test statistic and p-value (t, p)
3. ✅ Effect size (d=0.45) - standardized
4. ✅ Confidence interval (95% CI) - precision
5. ✅ Interpretation (medium effect) - context

---

### Practical Significance vs Statistical Significance

**Statistical significance** (p<0.05) answers: "Is the effect real (not zero)?"

**Practical significance** answers: "Does the effect matter in the real world?"

**Decision Matrix**:

| Effect Size | P-Value | Interpretation |
|-------------|---------|----------------|
| Large | p<0.05 | Statistically AND practically significant → Strong claim |
| Medium | p<0.05 | Statistically significant, possibly practical → Discuss context |
| Small | p<0.05 | Statistically significant, likely NOT practical → Caveat |
| Any | p>0.05 | Not statistically significant → Cannot conclude |

**Example of small effect with p<0.05**:
- 0.6 points improvement on 100-point scale
- n=500 per group → p=0.001 (highly significant)
- d ≈ 0.1 (small effect)
- **Interpretation**: "Statistically significant but effect size is small (d=0.1). Clinical significance is unclear - 0.6 points may not change educational outcomes."

---

### When Small Effects Matter

Small effects CAN be practically significant if:
1. **Low cost**: Intervention is cheap/easy to implement
2. **Population scale**: Small effect × millions of people = large impact
3. **Cumulative**: Small repeated effects compound over time
4. **Life-or-death**: Even tiny reductions in mortality matter
5. **Theoretical**: Supports/refutes important theory

**Example**: 0.5% reduction in heart attack risk (small) × 50 million people = 250,000 fewer heart attacks (large impact).

**Action**: Always discuss practical significance in context, don't assume small = unimportant.

---

## Test Selection: Decision Framework

### Core Principle: Choose Test BEFORE Seeing Data

**Test selection must be justified by**:
1. Study design (independent vs paired groups?)
2. Data type (continuous, ordinal, categorical?)
3. Assumption checking (normality, equal variance?)

**Test selection must NOT be justified by**:
4. ❌ Which test gives p<0.05
5. ❌ Post-hoc "data has outliers" after parametric test fails

---

### Decision Tree: Comparing Two Groups

```
START: Do you have paired/matched data?
│
├─ YES (paired) → Paired t-test (if normal) or Wilcoxon signed-rank test (if non-normal)
│
└─ NO (independent) → Check normality (Shapiro-Wilk test)
   │
   ├─ Normal → Check equal variances (Levene's test)
   │  ├─ Equal variances → Independent t-test (Student's t-test)
   │  └─ Unequal variances → Welch's t-test
   │
   └─ Non-normal → Mann-Whitney U test (Wilcoxon rank-sum)
```

**Key decision points**:
1. **Paired vs independent**: Determined by study design
2. **Normality**: Shapiro-Wilk test (p<0.05 suggests non-normal)
3. **Equal variance**: Levene's test (p<0.05 suggests unequal)

---

### Decision Tree: Comparing >2 Groups

```
START: Do you have paired/matched data?
│
├─ YES (repeated measures) → Repeated measures ANOVA (if normal) or Friedman test (if non-normal)
│
└─ NO (independent) → Check normality in each group (Shapiro-Wilk)
   │
   ├─ All groups normal → Check equal variances (Levene's test)
   │  ├─ Equal variances → One-way ANOVA
   │  └─ Unequal variances → Welch's ANOVA
   │
   └─ Some non-normal → Kruskal-Wallis test
```

**Post-hoc tests** (if omnibus test is significant):
- ANOVA → Tukey HSD (equal n) or Games-Howell (unequal n)
- Kruskal-Wallis → Dunn test with Bonferroni correction

---

### Assumption Checking: Required Procedures

**For parametric tests (t-test, ANOVA, regression)**:

1. **Normality of residuals**:
   - Visual: Q-Q plot (points should follow diagonal line)
   - Statistical: Shapiro-Wilk test
     - H0: Data is normal
     - If p<0.05: Reject normality, use non-parametric
     - If p≥0.05: Cannot reject normality, parametric OK

2. **Homogeneity of variance** (equal variances):
   - Visual: Boxplots (similar spread across groups)
   - Statistical: Levene's test
     - H0: Variances are equal
     - If p<0.05: Use Welch's t-test / Welch's ANOVA
     - If p≥0.05: Use standard t-test / ANOVA

3. **Independence of observations**:
   - Determined by study design (not testable)
   - If violated: Use mixed models or account for clustering

---

### When Parametric Tests Are "Robust"

T-test and ANOVA are robust to normality violations when:
- **n ≥ 30 per group** (Central Limit Theorem applies)
- **Equal sample sizes** across groups
- **Distributions are symmetric** (even if not perfectly normal)

T-test and ANOVA are NOT robust when:
- Small n (n<30)
- Heavily skewed distributions
- Extreme outliers present

**Action**: Don't invoke "robustness" as excuse without checking these conditions.

---

### Non-Parametric Tests: When and Why

**Use non-parametric tests when**:
1. Data is ordinal (not interval/ratio)
2. Normality assumption violated (Shapiro-Wilk p<0.05)
3. Sample size is small (n<30) and distribution is skewed
4. Outliers are present and legitimate (not errors)

**Trade-offs**:
- ✅ No distributional assumptions
- ✅ Robust to outliers
- ❌ Lower power than parametric (if data actually is normal)
- ❌ Test differences in medians/ranks, not means

**Common non-parametric equivalents**:
- Independent t-test → Mann-Whitney U
- Paired t-test → Wilcoxon signed-rank
- One-way ANOVA → Kruskal-Wallis
- Repeated measures ANOVA → Friedman test
- Pearson correlation → Spearman correlation

---

## Power Analysis: Planning Studies, Not Explaining Results

### A Priori Power Analysis (CORRECT)

Calculate required sample size BEFORE collecting data.

**Purpose**: Ensure study has adequate power (typically 0.80) to detect meaningful effect.

**Components**:
1. **α**: Significance level (typically 0.05)
2. **β**: Type II error rate (typically 0.20, so power = 1-β = 0.80)
3. **Effect size**: Assumed from literature, pilot, or minimum meaningful difference
4. **n**: Sample size to calculate

**Formula** (simplified for independent t-test):
```
n per group ≈ 16 / d²

where d = effect size (Cohen's d)
```

**Examples**:
- Small effect (d=0.2): n ≈ 16 / 0.04 = 400 per group
- Medium effect (d=0.5): n ≈ 16 / 0.25 = 64 per group
- Large effect (d=0.8): n ≈ 16 / 0.64 = 25 per group

**Use power analysis software** (G*Power, R, Python) for exact calculations.

---

### Post-Hoc Power Analysis (INCORRECT)

**DO NOT calculate power after seeing results.**

**Why it's circular reasoning**:
1. Observed effect size = f(your data)
2. Power = f(effect size, n, α)
3. Therefore: Post-hoc power = f(your p-value)

**Mathematical relationship**:
- If p=0.05, post-hoc power ≈ 0.50
- If p=0.01, post-hoc power ≈ 0.80
- If p=0.50, post-hoc power ≈ 0.10

Post-hoc power is just your p-value in different clothing. It provides ZERO new information.

---

### What to Do After a Null Result

**DO NOT say**: "Low post-hoc power explains the null result."

**DO say**:
1. **Report observed effect size and CI**:
   - "We observed d=0.15, 95% CI [-0.10, 0.40]"
   - This shows precision - wide CI means uncertain estimate

2. **Discuss whether study could have detected meaningful effect**:
   - "Our sample size (n=30 per group) would have 80% power to detect d≥0.75"
   - "We cannot rule out small-to-medium effects (d=0.2-0.5)"

3. **Recommend replication with larger n** (forward-looking):
   - "Future studies should use n≥64 per group to detect medium effects (d=0.5)"

4. **Interpret the null substantively**:
   - "No evidence that treatment differs from control"
   - NOT "study was too small to find an effect"

---

### Power vs Precision

**Power**: Probability of detecting an effect if it exists (planning tool)
**Precision**: Width of confidence interval (describes what you observed)

**After collecting data, report precision, not power.**

**Example**:
- ❌ "Post-hoc power was 0.35"
- ✅ "95% CI was wide [-0.3, 0.8], reflecting low precision"

---

## P-Value Interpretation: What P-Values Mean and Don't Mean

### What P-Values Mean

**P-value**: Probability of observing data this extreme (or more extreme) IF the null hypothesis is true.

**Correct interpretation** of p=0.03:
> "If there were truly no effect, we'd see data this extreme only 3% of the time by chance."

**Implications**:
- Small p-value → data are unlikely under H0 → evidence against H0
- Large p-value → data are compatible with H0 → no evidence against H0

---

### What P-Values DO NOT Mean

| Incorrect Interpretation | Why It's Wrong |
|--------------------------|----------------|
| "p=0.03 means 3% chance H0 is true" | P-value is P(data\|H0), not P(H0\|data) |
| "p=0.03 means 97% chance H1 is true" | Doesn't give probability of hypotheses |
| "p=0.001 means larger effect than p=0.04" | P-value conflates effect size and n |
| "p=0.06 is 'trending toward significance'" | Either significant or not, no middle ground |
| "p<0.05 means the effect is important" | Doesn't measure practical significance |
| "p>0.05 means no effect exists" | Absence of evidence ≠ evidence of absence |

---

### P-Value Depends on Both Effect Size and Sample Size

**Formula** (conceptually):
```
p-value ∝ 1 / (effect size × sqrt(n))
```

**This means**:
1. **Large n** → even tiny effects become "significant"
2. **Small n** → even large effects may be "non-significant"

**Example demonstrating this**:

| Scenario | Effect Size (d) | n per group | p-value | Interpretation |
|----------|----------------|-------------|---------|----------------|
| A | 0.1 (tiny) | 500 | 0.03 | Significant but trivial |
| B | 0.8 (large) | 15 | 0.09 | Non-significant but substantial |

**Lesson**: NEVER interpret p-values without effect sizes.

---

### Confidence Intervals: Superior to P-Values Alone

**95% Confidence Interval**: Range of plausible values for the true effect.

**Interpretation**:
> "We're 95% confident the true effect lies between [lower, upper]."

**Advantages over p-values**:
1. Shows magnitude of effect (not just "yes/no")
2. Shows precision (narrow CI = precise, wide CI = imprecise)
3. Shows practical significance (does CI include meaningful values?)

**Example**:
- Difference = 5 points, 95% CI [1.5, 8.5]
- **Interpretation**: True difference is likely between 1.5 and 8.5 points
- **Practical significance**: Even lower bound (1.5) may be meaningful

**Example 2**:
- Difference = 5 points, 95% CI [-2, 12]
- **Interpretation**: True difference could be anywhere from -2 to 12
- **Practical significance**: Wide range - more data needed

---

### Reporting P-Values: Best Practices

1. **Report exact p-values**: p=0.03, NOT p<0.05
2. **Never say "p=0.000"**: Use p<0.001
3. **Report non-significant p-values**: p=0.12, NOT "ns"
4. **Avoid "marginally significant"**: Just say p=0.06
5. **Always include effect size and CI**: p-value alone is incomplete

**Format**:
> "Treatment improved scores by 5 points (95% CI [1.5, 8.5], t(98)=2.8, p=0.006, d=0.45)."

---

## Pre-Registration: Preventing P-Hacking and HARKing

### What is Pre-Registration?

**Pre-registration**: Publicly specify your analysis plan BEFORE seeing the data.

**Components**:
1. Primary outcome variable
2. Secondary outcomes
3. Planned statistical tests
4. Sample size and stopping rule
5. Subgroup analyses (if any)
6. Multiple comparison corrections

**Platforms**: OSF (osf.io), AsPredicted, ClinicalTrials.gov

---

### Why Pre-Registration Matters

**Prevents**:
1. **P-hacking**: Testing many outcomes, reporting only significant
2. **HARKing**: Presenting exploratory findings as confirmatory
3. **Test shopping**: Trying multiple tests until one gives p<0.05
4. **Outcome switching**: Changing primary outcome after seeing data

**Enables**:
1. **Confirmatory inference**: Pre-specified hypotheses carry more weight
2. **Transparency**: Readers know what was planned vs exploratory
3. **Reproducibility**: Clear methodology prevents "researcher degrees of freedom"

---

### Pre-Registration vs Exploratory Analysis

**Both are valid, but must be labeled**:

| Analysis Type | When Specified | Inference Strength | Reporting |
|---------------|----------------|-------------------|-----------|
| Confirmatory | Before data collection | Strong (hypothesis-testing) | "As pre-registered, we tested..." |
| Exploratory | After seeing data | Weak (hypothesis-generating) | "Exploratory analyses revealed..." |

**Example**:
- Pre-registered: "Primary outcome is 6-month abstinence rate"
  - Test shows p=0.03 → Strong claim: "Treatment increases abstinence"

- Exploratory: "We noticed gender differences (not pre-specified)"
  - Test shows p=0.03 → Weak claim: "Exploratory analysis suggests gender may moderate effects; replication needed"

---

### What to Pre-Register

**Minimum requirements**:
1. **Primary outcome**: The ONE outcome you care about most
   - Multiple primaries require correction or co-primary designation

2. **Statistical test**: Which test will you use?
   - "Independent t-test if normality holds, else Mann-Whitney U"

3. **Sample size and stopping rule**:
   - "n=50 per group, no interim analyses"
   - OR "Sequential testing with O'Brien-Fleming boundaries"

4. **Alpha level**: Usually 0.05, but could be 0.01 for exploratory

5. **Secondary outcomes** (if any):
   - List all outcomes you'll test
   - Specify correction method (e.g., "Holm-Bonferroni for 5 secondary outcomes")

**Optional but recommended**:
- Planned subgroup analyses
- Handling of missing data
- Assumptions and what to do if violated
- Planned effect size calculations

---

### When You Can't Pre-Register

**Secondary data analysis**: Using existing datasets

**Approach**:
1. **Split data**: Exploratory on subset, confirmatory on holdout
2. **Cross-validation**: Train/test split with pre-specified test
3. **Acknowledge limitations**: "This is exploratory; results should be replicated"

**DO NOT**:
- Pretend exploratory findings are confirmatory
- Run analyses then retroactively claim they were planned

---

## Rationalization Table: Common Excuses and Counters

| Rationalization | Why It's Wrong | What To Do Instead |
|-----------------|----------------|---------------------|
| "I tested 10 variables and 2 were significant - those are real!" | With 10 tests, ~0.5 false positives expected by chance | Apply Bonferroni/Holm correction: α/10 = 0.005 |
| "p<0.05 means the intervention works!" | P-value doesn't measure effect size or practical significance | Report effect size (Cohen's d) and discuss practical meaning |
| "Post-hoc power was low, explaining the null result" | Post-hoc power is circular - just restates your p-value | Report observed effect size and CI; discuss precision |
| "I tried t-test and Mann-Whitney; Mann-Whitney was significant, so I'll use that" | Test shopping inflates Type I error | Pre-specify test based on assumption checking BEFORE analysis |
| "The data has outliers [after t-test failed], so non-parametric is better" | Post-hoc justification is p-hacking | Check assumptions BEFORE testing; report both if legitimately uncertain |
| "p=0.06 is 'trending toward significance'" | Euphemism for "not significant" | Report p=0.06 as non-significant; discuss effect size and CI |
| "I found X was significant, so that's my primary outcome" | HARKing - presenting exploratory as confirmatory | Label exploratory findings as exploratory; pre-register primary outcome |
| "Everyone in my field reports p-values without effect sizes" | Bad practice doesn't justify bad practice | Report effect sizes; set better example |
| "Reviewer wants p<0.05, so I need to find something significant" | Scientific integrity > reviewer demands | Report honest results; educate reviewer if needed |
| "With 500 participants, even tiny effects are significant - that proves it's real" | Large n makes trivial effects significant | Acknowledge statistical significance but discuss practical insignificance |
| "My pilot showed d=0.8, but main study showed d=0.2 - study was underpowered" | Sample size should have been based on conservative estimate | Note discrepancy; discuss regression to mean; recommend replication |
| "I can't pre-register because I'm using existing data" | Secondary analysis has methods to maintain rigor | Use holdout validation, cross-validation, or clearly mark exploratory |

---

## Workflow: Statistical Analysis Checklist

Use this checklist for EVERY statistical analysis:

### Before Collecting Data

- [ ] **Specify hypotheses** (H0 and H1)
- [ ] **Designate primary outcome** (only ONE, or use co-primaries with correction)
- [ ] **Choose statistical test** based on study design and expected data type
- [ ] **Calculate required sample size** using power analysis (power=0.80, α=0.05, assumed effect size)
- [ ] **Pre-register analysis plan** (OSF, AsPredicted, or lab notebook with timestamp)
- [ ] **Specify what you'll do if assumptions are violated**

### After Collecting Data, Before Analysis

- [ ] **Check data quality** (outliers, missing data, data entry errors)
- [ ] **Check assumptions**:
  - [ ] Normality (Shapiro-Wilk test, Q-Q plots)
  - [ ] Homogeneity of variance (Levene's test)
  - [ ] Independence (from study design)
- [ ] **Decide on test** based on assumption checks (following pre-registered plan)

### During Analysis

- [ ] **Run pre-specified primary analysis**
- [ ] **Calculate effect size** (Cohen's d, eta-squared, R², etc.)
- [ ] **Calculate confidence intervals** (95% CI for effect size and raw difference)
- [ ] **Apply multiple comparisons correction** if testing >1 hypothesis (Bonferroni, Holm, or FDR)
- [ ] **Run pre-specified secondary analyses** (if any)

### Reporting Results

- [ ] **Report test statistic, df, and exact p-value**: t(98)=2.8, p=0.006
- [ ] **Report effect size with interpretation**: d=0.45 (medium effect)
- [ ] **Report confidence interval**: 95% CI [1.5, 8.5]
- [ ] **Report raw means/medians with SDs/IQRs**: Treatment M=75 (SD=10), Control M=70 (SD=12)
- [ ] **Discuss practical significance** separately from statistical significance
- [ ] **Label exploratory findings** as exploratory (if any)
- [ ] **Report all pre-registered outcomes** (not just significant ones)

### Red Flags to Avoid

- [ ] ❌ Reporting only p-values (no effect sizes)
- [ ] ❌ Reporting only significant results (file-drawer problem)
- [ ] ❌ Post-hoc power analysis
- [ ] ❌ "Trending toward significance" for p>0.05
- [ ] ❌ Choosing tests after seeing results
- [ ] ❌ Multiple testing without correction
- [ ] ❌ Presenting exploratory findings as confirmatory

---

## Examples: Correct Statistical Reporting

### Example 1: Two-Group Comparison (Significant Result)

**Context**: RCT comparing new teaching method (n=52) vs traditional (n=48) on test scores.

**Bad Reporting**:
> "The new method significantly improved scores (p=0.006)."

**Good Reporting**:
> "Students in the new teaching method group scored higher (M=75.2, SD=10.3) than traditional method students (M=70.1, SD=11.8), t(98)=2.8, p=0.006, d=0.45, 95% CI for difference [1.5, 8.5]. This represents a medium effect size. A 5.1-point improvement on a 100-point scale may have practical significance, though this depends on grading thresholds and instructional costs."

**What's better**:
- ✅ Descriptive statistics (M, SD) for both groups
- ✅ Complete test results (t, df, p)
- ✅ Effect size (d=0.45) with interpretation (medium)
- ✅ Confidence interval showing precision
- ✅ Discussion of practical significance

---

### Example 2: Multiple Outcomes (After Bonferroni Correction)

**Context**: Testing new ML optimizer on 8 metrics (α=0.05, Bonferroni-corrected α=0.05/8=0.00625).

**Bad Reporting**:
> "Our optimizer significantly improved convergence speed (p=0.04)."

**Good Reporting**:
> "We tested our optimizer on 8 performance metrics using Bonferroni correction (α_corrected = 0.05/8 = 0.00625). Of the 8 metrics, none reached statistical significance after correction:
> - Convergence speed: p=0.04 (not significant after correction)
> - Accuracy: p=0.21
> - F1 score: p=0.18
> [... report all 8]
>
> While convergence speed showed p=0.04 before correction, this does not survive multiple comparisons adjustment and may represent a false positive. Effect sizes for all metrics were small (d<0.3)."

**What's better**:
- ✅ Correction method specified
- ✅ All outcomes reported (not just "significant" one)
- ✅ Honest interpretation (likely false positive)
- ✅ Effect sizes reported

---

### Example 3: Non-Significant Result (Reported Properly)

**Context**: RCT testing anxiety intervention (n=30 per group), found p=0.32.

**Bad Reporting**:
> "The intervention did not significantly reduce anxiety (p=0.32). Post-hoc power analysis revealed power=0.35, suggesting the study was underpowered."

**Good Reporting**:
> "The intervention group (M=42.1, SD=8.3) did not differ significantly from the control group (M=44.5, SD=9.1) on anxiety scores, t(58)=1.0, p=0.32, d=0.27, 95% CI for difference [-2.1, 6.9]. The observed effect size was small (d=0.27). Our sample size (n=30 per group) provided 80% power to detect effects of d≥0.75 (large effects), but was underpowered for small-to-medium effects. The confidence interval includes both negligible effects and effects up to d≈0.7, indicating the study was imprecise. Future research should use n≥64 per group to reliably detect medium effects (d=0.5)."

**What's better**:
- ✅ Descriptive statistics reported
- ✅ Complete test results (including non-significant p)
- ✅ Effect size and CI (shows precision)
- ✅ A priori power discussed (not post-hoc)
- ✅ Forward-looking recommendation (not excuse)
- ❌ No mention of "post-hoc power"

---

## Resistance Scenarios: Handling Pressure

### Scenario: "Reviewer demands p<0.05"

**Pressure**:
> "Reviewer 2 says we need a significant result to publish. Can we try non-parametric tests or remove outliers to get p<0.05?"

**Response**:
1. **Explain scientific integrity**:
   - "Manipulating analyses to achieve p<0.05 is p-hacking and violates scientific norms"
   - "Reviewers can be wrong; editors can overrule bad reviewer demands"

2. **Reframe the result**:
   - "Non-significant results are publishable and valuable"
   - "Our CI shows we can rule out large effects but not small-medium effects"

3. **Suggest alternative framing**:
   - "Let's emphasize effect sizes, precision, and what we can/can't rule out"
   - "We can discuss implications for future research with larger samples"

4. **Escalate if needed**:
   - "If reviewer insists on p-hacking, contact the editor"
   - "Many journals value null results (e.g., PLOS ONE)"

---

### Scenario: "Everyone in my field does it this way"

**Pressure**:
> "In neuroscience, everyone reports p-values without effect sizes. Reviewers will be confused if I include Cohen's d."

**Response**:
1. **Acknowledge field norms**:
   - "I understand this isn't standard in your field yet"

2. **Explain why norms should change**:
   - "Effect sizes are required by APA, reporting guidelines (e.g., CONSORT), and funding agencies"
   - "P-values alone have caused replication crisis across fields"

3. **Provide easy implementation**:
   - "Add one sentence: 'Effect sizes were d=0.45 (medium) for outcome A, d=0.22 (small) for outcome B'"
   - "This strengthens your paper and sets a better example"

4. **Cite guidelines**:
   - APA Publication Manual (7th ed) requires effect sizes
   - CONSORT, STROBE, and other reporting guidelines require effect sizes

---

### Scenario: "Deadline is tomorrow, no time for corrections"

**Pressure**:
> "Conference deadline is tomorrow. I found 3 significant results out of 20 tests. Can we just report those 3 and mention we tested others in limitations?"

**Response**:
1. **Explain the severity**:
   - "Selective reporting without correction is scientific misconduct"
   - "With 20 tests, you'd expect 1 false positive by chance - you found 3"

2. **Provide quick fix**:
   - "Apply Bonferroni correction: α = 0.05/20 = 0.0025"
   - "Do your 3 results survive? If not, be honest"

3. **Offer honest framing**:
   - "Report all 20 tests with correction in a supplementary table"
   - "If nothing survives, frame as exploratory: 'No results survived correction, but trends suggest...'"

4. **Long-term solution**:
   - "For next study, pre-register primary outcome to avoid this problem"

---

## Summary: Core Requirements

When doing statistical analysis, you MUST:

1. ✅ **Pre-specify primary outcome and analysis plan** (before data collection)
2. ✅ **Apply multiple comparisons correction** when testing >1 hypothesis
3. ✅ **Report effect sizes AND p-values** (never p-values alone)
4. ✅ **Report confidence intervals** for estimates
5. ✅ **Check assumptions** before choosing tests (not after seeing results)
6. ✅ **Calculate sample size a priori** using power analysis
7. ✅ **Distinguish statistical from practical significance**
8. ✅ **Report all pre-registered outcomes** (not just significant ones)
9. ✅ **Label exploratory findings** as exploratory

You MUST NOT:

1. ❌ Calculate post-hoc power to "explain" null results
2. ❌ Choose statistical tests after seeing which gives p<0.05
3. ❌ Report only significant results from multiple tests
4. ❌ Use "trending toward significance" for p>0.05
5. ❌ Present exploratory findings as if they were confirmatory
6. ❌ Report p-values without effect sizes
7. ❌ Claim "p<0.05 proves practical significance"

**When in doubt**: Be transparent about what was planned vs exploratory, report all outcomes, include effect sizes and CIs, and discuss practical significance.

---

## Related Skills

- **experimental-design**: Provides control groups and randomization (complements this skill's analysis focus)
- **hypothesis-formation**: Creates H0 and H1 (which this skill tests)
- **reproducibility-practices**: Pre-registration and open data (implements transparency from this skill)
- **research-ethics**: Ensures statistical practices are honest and responsible

Use `/research-methodology` to route to these skills or return to the router.
