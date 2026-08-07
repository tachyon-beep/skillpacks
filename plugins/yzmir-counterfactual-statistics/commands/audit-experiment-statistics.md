---
description: Adversarially review an analysis, eval harness, or paper against the 21-entry anti-pattern catalogue - severity-rated findings with evidence, each citing the sheet that fixes it
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Skill", "Task"]
argument-hint: "[path to analysis script, harness, results dir, or paper]"
---

# Audit Experiment Statistics

Adversarially review a counterfactual or paired-branch experiment for statistical defects. Dispatch the `experiment-statistics-reviewer` agent for a full pass, or work the catalogue inline for a scoped one.

## Core Principle

**Audit the code and the data, not the write-up.** The write-up describes the analysis its authors believe they ran. The defects in this catalogue are almost all invisible there — a pseudo-replicated `n`, a threshold fit on the wrong split, a horizon constant edited after the fleet finished. They are visible in the analysis script, the git history, and the results schema.

**Zero findings is a defect of the audit.** Every real pipeline has at least a Medium — an undeclared family size, a missing MDE, mean-only reporting. A clean report means the audit did not look at the code.

## Audit Order

The order matters: an early finding can invalidate later ones, so record them but do not spend effort quantifying a multiplicity problem in a result whose `n` is wrong by 24×.

```
1. Unit and pairing integrity   AP-01, AP-02, AP-09, AP-19, AP-20
2. Data-role walls              AP-04, AP-08
3. Selection and testing        AP-03, AP-05, AP-07, AP-10
4. Design adequacy              AP-12, AP-13, AP-14
5. Utility definition           AP-15, AP-16, AP-11
6. Reporting integrity          AP-06, AP-17, AP-18, AP-21
```

Full definitions, mechanisms, and detectors: [anti-pattern-catalogue.md](../skills/using-counterfactual-statistics/anti-pattern-catalogue.md).

## Phase 1: Establish Ground Truth

Before applying any check, determine from the artifacts (not the prose):

```bash
# What is the real unit count?
python -c "import pandas as pd; d=pd.read_parquet('results.parquet'); print(len(d), d.run_id.nunique())"

# Was the analysis edited after the fleet finished?
git log --format='%ad %h %s' -- analysis.py | head
git log -p -- analysis.py | grep -E '^\+.*(HORIZON|METRIC|threshold|alpha|exclude|drop|dropna)'

# Which tests are being used?
grep -rnE 'ttest_ind|ttest_rel|ttest_1samp|mannwhitneyu|wilcoxon|train_test_split|GroupShuffleSplit' .

# Additive seeding and per-branch dataloaders
grep -rnE 'manual_seed\(.*\+|seed *\+ *|default_rng\(.*\+' .
```

Record: rows, distinct units, arms present, control arm zero-anchored y/n, `K`, family size, presence of a pre-registration and its commit date relative to the fleet.

## Phase 2: Work the Catalogue

For each entry, run its detector. Findings need all six fields — a finding without evidence is an opinion, and a finding without an impact statement is not actionable:

```
AP-01  Branches counted as independent samples             CRITICAL
  Evidence     analysis.py:47 -- ttest_1samp(df.diff, 0) over 576 rows;
               df.run_id.nunique() == 24
  Mechanism    pseudo-replication; SE understated by sqrt(rows/unit)
  Impact       recomputing at the unit level moves p from 4.6e-06 to 0.091
               and the 95% CI to [-0.0021, +0.0264] -- includes zero
  Fix          statistical-units-and-clustering.md; aggregate to one
               difference per run_id before testing
  Effort       ~10 lines in analysis.py; no new data required
```

Where you can **compute** the corrected number, do — a finding that says "p becomes 0.091" is acted on; one that says "clustering was not accounted for" is filed.

## Phase 3: Check What Is Absent

Absences do not appear in a grep and are the most commonly missed findings:

- No pre-registration, or one committed after the fleet started (AP-14).
- No MDE beside a null (AP-12).
- No `K` beside a best-of-K claim (AP-03).
- No rejection/abstention/failure records in the trial store (AP-18).
- No `n_units` in the results table (AP-17).
- An audit stage with a 0% historical rejection rate (AP-08) — query it; the rate is the evidence.
- No no-op branch anywhere in the trial schema (AP-19).

## Phase 4: Rank and Report

Order by severity, then by how much of the conclusion each finding destroys. Then state plainly, in one sentence, **whether the headline claim survives**. That sentence is what the audit is for.

## Output Contract

1. **Scope** — what was read; what was not available and how that limits the audit.
2. **Ground truth** — rows vs units, arms, `K`, family size, pre-registration status.
3. **Findings** — severity-ordered, each with ID, evidence (file:line / commit / query output), mechanism, computed impact, fix sheet, effort.
4. **Verdict** — does the headline claim survive as written? If not, what is the largest claim the data does support?
5. **Recommended order of remediation** — cheapest defect-per-unit-of-conclusion first.
6. **Confidence and information gaps** — what could not be checked and what would resolve it.

## Full Adversarial Pass

For a complete review, dispatch the agent, which follows the SME Agent Protocol and reports confidence and risk per finding:

```
Task(subagent_type="experiment-statistics-reviewer",
     prompt="Audit <path> against the yzmir-counterfactual-statistics anti-pattern "
            "catalogue. Read the analysis code, the results schema, the git history "
            "of the analysis script, and any pre-registration. Report severity-rated "
            "findings with computed impact.")
```

## When Not to Use This Command

- Designing an experiment → `/design-counterfactual-experiment`.
- Running the analysis on your own trial → `/analyze-paired-trial`.
- Reviewing code quality, tests, or architecture rather than statistics → `ordis-quality-engineering` or the relevant engineering pack.
- The experiment is a general A/B or observational-causal study → `yzmir-experimentation` *(planned)*.
