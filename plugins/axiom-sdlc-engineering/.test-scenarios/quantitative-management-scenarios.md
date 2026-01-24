# Test Scenarios for quantitative-management Skill

**Skill Type**: Reference (documentation/guidance)
**Test Approach**: Retrieval, Application, Coverage
**Date**: 2026-01-25

---

## Scenario 1: Establishing DORA Metrics for a Team

**Context**: Development team wants to implement DORA metrics (Deployment Frequency, Lead Time for Changes, Change Failure Rate, MTTR) to track DevOps performance.

**Question**: "How do I set up DORA metrics for my development team? We use GitHub Actions for CI/CD and want to track our progress over time."

**Success Criteria**:
- Agent references DORA metrics reference sheet
- Provides concrete measurement definitions
- Explains how to collect each metric
- Shows visualization/tracking approach
- Addresses automation for collection
- Provides baseline establishment guidance

**Expected Information to Retrieve**:
- Definition of each DORA metric
- How to measure each in GitHub Actions
- Scripts/automation for collection
- Visualization approaches
- Baseline establishment process

---

## Scenario 2: Statistical Process Control for Defect Rates

**Context**: Team notices defect escape rate is variable. Want to implement statistical process control to detect when process is out of control.

**Question**: "Our defect escape rate varies between 5-20% per sprint. How do I use statistical process control to know when this is a real problem vs normal variation?"

**Success Criteria**:
- Agent references statistical analysis reference sheet
- Explains control charts (specifically p-chart for proportions)
- Shows how to calculate control limits
- Explains interpretation (normal variation vs special cause)
- Provides Level 3/4 guidance on sophistication
- Includes action threshold recommendations

**Expected Information to Retrieve**:
- Control chart types for defect rates
- Control limit calculation formulas
- Interpretation rules
- When to investigate vs accept variation
- Level 3→4 statistical rigor differences

---

## Scenario 3: Measurement Planning with GQM

**Context**: Organization wants to establish formal measurement program but unsure what to measure and why.

**Question**: "We want to measure our development process effectiveness. How do I determine what metrics are actually useful vs measurement theater?"

**Success Criteria**:
- Agent references measurement planning reference sheet
- Explains GQM (Goal-Question-Metric) methodology
- Provides concrete GQM examples
- Addresses measurement cost vs value
- Warns against vanity metrics
- Shows Level 2→3→4 progression

**Expected Information to Retrieve**:
- GQM methodology steps
- Example GQM decomposition
- Measurement cost considerations
- Anti-pattern: measurement theater
- Level scaling for measurement sophistication

---

## Scenario 4: Process Baselines for Estimation

**Context**: Team wants to use historical data to improve estimation accuracy for sprint planning.

**Question**: "How do I establish process baselines from our historical velocity data to improve our sprint planning estimates?"

**Success Criteria**:
- Agent references process baselines reference sheet
- Explains baseline establishment from historical data
- Shows statistical analysis approach (mean, std dev, confidence intervals)
- Addresses minimum sample size
- Explains baseline usage for estimation
- Covers baseline maintenance/updates

**Expected Information to Retrieve**:
- Baseline establishment methodology
- Statistical analysis for baselines
- Sample size requirements
- Using baselines for estimation
- Baseline update frequency

---

## Scenario 5: Level 3 to Level 4 Transition

**Context**: Organization currently at CMMI Level 3 with organizational baselines. Want to transition to Level 4 with quantitative management.

**Question**: "We have organizational baselines and trend analysis (Level 3). What's required to move to Level 4 quantitative management?"

**Success Criteria**:
- Agent references Level 2→3→4 scaling reference sheet
- Clearly distinguishes Level 3 vs Level 4 requirements
- Explains statistical process control addition
- Addresses prediction models
- Shows process performance objectives
- Provides concrete examples of Level 4 practices

**Expected Information to Retrieve**:
- Level 3 vs Level 4 differences
- Statistical process control requirements
- Prediction model approaches
- Process performance objectives
- Level 4 implementation examples

---

## Expected Baseline Results (No Skill)

**Hypothesis**: Without the skill, agents will:
1. Provide generic "track metrics" advice without concrete implementation
2. Miss GQM methodology for measurement planning
3. Provide incorrect or incomplete DORA metric definitions
4. Not distinguish between control chart types
5. Skip baseline establishment methodology
6. Not address Level 2→3→4 scaling differences
7. Provide vanity metrics without business value linkage
8. Miss automation guidance for metric collection
9. Not explain statistical significance vs noise
10. Lack production-ready scripts/examples

---

**Test Execution Plan**:
1. Run Scenario 1 in detail with subagent WITHOUT skill (baseline)
2. Document exact gaps and missing information
3. Run Scenarios 2-5 summarized to identify additional patterns
4. Write skill addressing all identified gaps
5. Re-run Scenario 1 WITH skill to verify coverage
