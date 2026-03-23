---
description: Diagnose issues with dynamic architecture growth, pruning, or integration
allowed-tools: ["Read", "Glob", "Grep", "Bash", "WebSearch"]
---

# Diagnose Dynamic Architecture Issues

Systematically diagnose problems with dynamic neural architecture - growth triggers, pruning decisions, gradient isolation, and module integration.

## Diagnostic Protocol

### Phase 1: Identify the Symptom

First, clarify what's happening:

| Symptom Category | Examples |
|------------------|----------|
| Growth issues | Not growing when expected, growing too fast, wrong capacity added |
| Pruning issues | Pruning too aggressively, keeping dead modules, thrashing |
| Integration issues | New modules destabilize training, regression after integration |
| Gradient issues | Gradients exploding/vanishing, host being affected by seed training |
| Lifecycle issues | Stuck in state, transitions not triggering, wrong gate failures |

### Phase 2: Gather Evidence

**Find relevant code:**
```
Glob: **/train*.py, **/model*.py, **/grow*.py, **/prune*.py, **/lifecycle*.py
Grep: "detach|freeze|requires_grad|alpha|blend|gate|transition"
```

**Check training logs/metrics if available:**
```
Glob: **/logs/**, **/*.log, **/metrics*
```

**Look for configuration:**
```
Glob: **/config*.py, **/config*.yaml, **/config*.json
```

### Phase 3: Check Common Failure Modes

#### Growth Failures

| Check | How to Verify | Fix |
|-------|---------------|-----|
| Plateau detector too sensitive | Loss history shows continued improvement | Increase patience |
| Plateau detector too insensitive | Loss flat for many epochs before growth | Decrease patience/threshold |
| No available slots | Check slot state management | Add slots or fix recycling |
| Budget exhausted | Count current params vs budget | Increase budget or prune first |
| Warmup too short | Instability after growth | Extend warmup period |

#### Pruning Failures

| Check | How to Verify | Fix |
|-------|---------------|-----|
| Contribution measurement broken | All modules show same contribution | Fix counterfactual measurement |
| Threshold too aggressive | Good modules being pruned | Raise contribution threshold |
| Threshold too lenient | Dead modules kept | Lower threshold or add time limit |
| No cooldown | Same slot repeatedly fails | Add embargo period |

#### Gradient Isolation Failures

| Check | How to Verify | Fix |
|-------|---------------|-----|
| Missing detach | Host params have grads from seed path | Add detach at isolation point |
| Wrong detach location | Seed not receiving gradients | Move detach to correct tensor |
| Optimizer includes frozen params | Frozen params in optimizer groups | Filter optimizer params |
| BatchNorm updating during freeze | Running stats changing | Set BN to eval mode |

#### Lifecycle Failures

| Check | How to Verify | Fix |
|-------|---------------|-----|
| Gate always failing | Log gate check results | Debug specific gate condition |
| Missing state transitions | State history shows gaps | Add missing transition rules |
| Thrashing between states | Rapid state changes in log | Add hysteresis/cooldown |
| Stuck in state | No transitions for many epochs | Check transition triggers |

### Phase 4: Report Findings

Structure your diagnosis:

```markdown
## Diagnosis Report

### Symptom
[What the user reported]

### Evidence Found
- [File:line] - [What you found]
- [Metric/log] - [What it shows]

### Root Cause
[The actual problem based on evidence]

### Recommended Fix
[Specific changes with code examples]

### Verification
[How to confirm the fix worked]
```

## Using the Dynamic Architecture Advisor

For complex issues, invoke the full advisor agent:

```
Load agent: dynamic-architecture-advisor
```

The advisor will do deeper investigation and can search for prior art in literature.
