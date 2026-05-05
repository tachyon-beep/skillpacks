---
name: when-not-to-grow
description: Use when deciding the network has reached its useful capacity, the controller is growing into pathology, or the problem doesn't benefit from morphogenesis — refusal patterns, capacity-saturation signals, and the boundary against "always grow more."
---

# When Not to Grow

## When to Use

- Considering morphogenesis for a new problem and unsure whether it is the right tool
- A morphogenetic system is in production and you suspect it is doing nothing useful — or harm
- Comparing morphogenesis to a fixed-architecture baseline and morphogenesis is losing
- A research substrate has accumulated growth events but the gains are within seed-noise
- About to scale up a morphogenetic experiment and want a sanity check before spending the compute

This sheet is the off-switch. Most other sheets in this pack assume you have decided to use morphogenesis and are designing it well. This one asks the prior question: *should you use it at all?*

---

## Core Principle

**Morphogenesis is a method, not a virtue. The default assumption for any new problem should be "a fixed architecture wins." Morphogenesis must earn its place by beating the off-switch baseline.**

The off-switch baseline is the same harness, same seed, same data, same compute — with the controller disabled and growth events disallowed. If that baseline matches your morphogenetic run, the controller did nothing. If it beats your morphogenetic run, the controller did harm.

Both outcomes are common. Both are usually undiagnosed because most morphogenetic projects do not run the off-switch baseline. Run it.

The other principle, equally important:

**"It might still help eventually" is not a reason to keep a system that is currently underperforming.**

Sunk-cost reasoning is the dominant rationalization for keeping morphogenesis around after evidence shows it is not earning its keep. The fix is not "tune more"; the fix is to either (a) demonstrate that the controller helps under specific conditions you can characterize, or (b) turn it off.

---

## Failure Modes Where Morphogenesis Hurts

These are the patterns where the controller is harmful, not merely useless. Recognize them.

### Failure Mode 1: Growth as Procrastination

The host trainer is failing on a fundamental issue — bad data, broken loss, undertrained baseline. The controller, optimizing for "loss decrease after growth," learns that growing produces a brief improvement (more parameters help any model). The system grows continuously, masking the underlying training problem.

**Signal**: Loss decreases primarily at growth events; flat or rising between them. Growth rate is high. Eventually the parameter budget caps and the loss plateaus at a worse level than a properly-trained static baseline of the same size would achieve.

**Why this is harmful, not merely useless**: The morphogenetic harness convinces you the system is "learning" because loss is going down. It is going down for the wrong reason. You ship a model that does not generalize — it has been overfit to a sequence of architectural changes rather than to the data.

### Failure Mode 2: Variance Amplification

For a problem where a small fixed architecture is near-optimal, morphogenesis adds variance with no expected return. Different seeds end at different shapes; some grow too much, some too little; the mean is unchanged but the variance widens.

**Signal**: Across seeds, mean loss matches static baseline; standard deviation is 2-5× higher.

**Why this is harmful**: A higher-variance method is strictly worse than a lower-variance method with the same mean — fewer of your seeds are usable for downstream work. Production systems built on high-variance methods need more retraining and more monitoring.

### Failure Mode 3: Controller Trains the Network, Not the Other Way Around

In some setups the controller's reward signal is the dominant gradient driving change. The host trainer's optimizer gets less effective signal than the controller's growth events. The network is being shaped by the policy, not by the data.

**Signal**: Removing data (or scrambling labels) does not significantly change the controller's growth pattern. The controller has learned a "growth schedule" independent of what the data is teaching.

**Why this is harmful**: You have a learned architecture-search procedure masquerading as a training run. Whatever the network ends up looking like is a property of the controller's prior, not the problem.

### Failure Mode 4: Coordination Failure

In multi-seed setups, the controller cannot prioritize and grows everywhere at once, blowing through the parameter budget in a few events. The resulting network has no architectural strategy — it is just "where the budget went." See `multi-seed-coordination-rl.md`.

**Signal**: Growth events cluster in time. Parameter budget is consumed quickly. Per-slot growth distribution is flat (no slot is favored).

**Why this is harmful**: A controller that does not prioritize is not making decisions, it is making mass requests. You are not getting controller value.

### Failure Mode 5: Brittleness Without Ablation

The morphogenetic system "works" — for one specific reward function, gate configuration, slot count, and seed range. Small perturbations break it. The team discovers this only when porting to a new task.

**Signal**: Internal robustness checks (different seeds, slightly different reward weights, slightly different gate thresholds) collapse the result.

**Why this is harmful**: A method that requires this much hand-tuning is not a method, it is a hyperparameter set. The next problem will require re-tuning, and you have not built understanding that transfers.

### Failure Mode 6: The Controller Has Learned to Veto Itself

After enough rollback signal, the controller learns that the safe action is no-action. Growth events become rare or zero. The system runs as a static baseline with a controller-shaped overhead.

**Signal**: Late-training growth rate near zero. Off-switch baseline matches.

**Why this is harmful**: You have proven the controller can learn to be inert. You have not proven it can learn to grow well. The result is a static-baseline run with morphogenetic complexity tax.

(See `rollback-as-rl-signal.md` for how this happens and how to design reward against it.)

---

## Domains Where a Static Architecture Wins

Not every problem benefits from morphogenesis. The following classes of problem have a known small optimal architecture; adding morphogenesis adds risk without expected return.

| Domain class | Why static wins |
|--------------|-----------------|
| Mature task with established SOTA architectures | The architecture is already known. Morphogenesis re-discovers it expensively, at best. |
| Small data regime (under ~10k examples) | Capacity is not the bottleneck. Adding parameters during training adds variance, not capability. |
| Real-time inference with strict latency budgets | Growing networks complicate deployment; the production model often has to be re-trained statically anyway. |
| Problems with cheap external NAS | If a NAS run takes 10× the cost of one training run and produces a static architecture that beats morphogenesis, do that. |
| Problems where reproducibility is more valuable than peak performance | Morphogenesis is high-variance even with full determinism discipline. |

The reverse — domains where morphogenesis can help — is narrower than enthusiasm suggests:

| Domain class | Why morphogenesis can help |
|--------------|----------------------------|
| Continual / sequential tasks where the data distribution changes | New capacity for new structure |
| Long-horizon training where the loss landscape evolves | Architecture is not constant-optimal across the run |
| Capacity-budget-constrained discovery | Where to *spend* parameters is the question, not what they look like |
| Research substrates where architecture variation is the variable being studied | Morphogenesis is the experimental method, not the goal |

If your problem is not in the second list, the burden of proof is on morphogenesis.

---

## The Required Baselines Before Concluding "Morphogenesis Helps"

Run all of these. Not running one means you do not have the result you think.

1. **Off-switch baseline**: Same harness, same seed, controller disabled. If this matches the morphogenetic run, the controller did nothing.
2. **Static-final baseline**: A static network at the morphogenetic run's final shape, same compute. If this beats morphogenesis, the controller's path-through-shapes was not better than starting at the destination.
3. **Fixed-schedule baseline**: A non-learned schedule that grows to the same shape on the same step pattern. If this matches morphogenesis, the controller's *decisions* did not contribute beyond the schedule itself.
4. **Multi-seed**: At least 10 seeds per condition. Morphogenetic variance is high; small-N comparisons are misleading.

(Full details in `evaluation-under-topology-change.md`.)

A morphogenetic project that has not run baseline 1 has not yet demonstrated the controller's existence. A project that has not run baselines 2 and 3 has not isolated the controller's contribution.

---

## Decision Procedure

### Before Starting

1. **What problem class am I in?** If it's in the "static wins" table, default to no morphogenesis.
2. **What is the off-switch baseline expected to look like?** If the answer is "I don't know," run it before designing the controller.
3. **What is the worst-case cost of being wrong?** If morphogenesis fails silently and you ship the result, what is the consequence? Use this to budget evaluation rigor.
4. **Have I tried a fixed schedule first?** A hand-designed growth schedule is cheaper to build, more reliable, and is your null hypothesis for the controller.

### After Running

1. **Does the off-switch baseline match?** If yes, your controller does nothing. Investigate before adding more complexity.
2. **Does the static-final baseline beat morphogenesis?** If yes, you would have been better off knowing the architecture upfront.
3. **Does the fixed-schedule baseline match morphogenesis?** If yes, the controller's *decisions* did not add value beyond the act of growing.
4. **Is per-seed final architecture variance low?** If high, you do not really have a single condition; you have a distribution.
5. **Across reward modes / gate configs / slot counts, are results monotonic and intuitive?** If results are erratic across nearby settings, you have a brittle method, not a robust one.

If any of these fail, the answer is **not** "tune more." It is "run the diagnostic, identify which failure mode, and decide whether to repair or remove."

---

## How to Walk It Back

If morphogenesis is in your system and the diagnostics say it should not be, the path back is:

1. **Stop adding controller complexity.** Resist "one more reward term will fix it."
2. **Capture the result.** Document the failure mode, the baselines that beat it, the conditions tested. This is a result.
3. **Replace with a fixed schedule** that ends at the morphogenetic run's typical final shape. Ship the static or fixed-schedule version.
4. **Keep the harness.** Telemetry, replay logs, governor — these are useful even for static training. Don't throw away the infrastructure; throw away the controller.
5. **Be public about the result.** A failed morphogenetic experiment is data that prevents future teams from repeating it. Not publishing is the bigger waste than the original compute.

---

## Common Mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| Skip off-switch baseline | Cannot tell whether controller helped | Run it |
| Tune the controller after a negative result | Burns compute on a method that may be wrong for the problem | Investigate failure mode first |
| Rationalize that "it would help on bigger problems" | Makes the unfalsifiable claim that the test was too small | Either run on the bigger problem or stop the claim |
| Treat morphogenesis as a default for any new problem | Adds variance and complexity for no expected return | Use the domain table; default to static |
| "We've already invested 6 weeks, we can't stop now" | Sunk cost; ignores future cost | Future cost is the relevant axis. Stop. |
| Compare morphogenetic-best-of-5 to static-mean | Selection bias dressed as a comparison | Compare distributions to distributions |
| Conflate "growth events fired" with "controller decisions mattered" | Growth events fire whether the controller is good or random | Run the fixed-schedule baseline |
| Avoid the off-switch baseline because "it's obviously going to lose" | Then it costs nothing to run and the result is published | Run it |

---

## Rationalization Resistance

These are the rationalizations that keep morphogenesis around when it should be removed. Each one is a reason to *do the test*, not to skip it.

| Rationalization | Reality | Counter |
|-----------------|---------|---------|
| "Morphogenesis is the future of ML" | A research direction is not a default. The default is static. | Run the off-switch. If it matches, you have a non-result. |
| "We just need to tune the reward function more" | After three reward iterations, you are not tuning, you are searching. | Stop tuning. Run baselines. If they match, the issue is methodological, not hyperparametric. |
| "The static baseline is unfair because it had perfect architecture knowledge" | Fixed-schedule baseline does not have that. Compare to it. | If fixed-schedule matches morphogenetic, the controller did not learn. |
| "Variance is high but the best seed is great" | Best-of-N is not a fair comparison. The fair comparison is mean and variance. | Report all seeds. If mean is no better, single great seeds are noise, not signal. |
| "We didn't run the off-switch because it's obviously worse" | Then the experiment is cheap and confirms the prior. | Run it. The cases where it isn't worse are the most interesting and most often missed. |
| "Morphogenesis works in principle, our specific implementation is just early" | "In principle" is unfalsifiable. The implementation is what is being tested. | If the implementation does not work, the implementation is the result. |
| "Disabling morphogenesis means giving up" | It means moving compute to better targets. | Stopping a failed experiment is not failure. Continuing one is. |
| "The controller is interpretable so it must be doing something" | Interpretability of a policy is independent of whether the policy is useful. | A policy can be interpretable and inert. Off-switch baseline distinguishes. |
| "Sunk cost — we've already trained the controller" | Sunk cost is not an argument for future cost. | Walk it back. Keep the harness, drop the controller. |
| "It might help eventually with more data" | If your evaluation is at the data scale you have, that is the result. | Either run at the bigger scale or stop the claim. |

---

## Red Flags Checklist

- [ ] **Off-switch baseline never run** — controller's contribution is unverified
- [ ] **Loss only decreases at growth events** — controller is masking a training problem
- [ ] **Final-architecture variance across seeds is high** — condition is not really a single condition
- [ ] **Growth rate stays high until budget caps** — controller is not prioritizing
- [ ] **Removing data does not significantly change growth pattern** — controller has learned a data-independent schedule
- [ ] **Fixed-schedule baseline matches** — controller's decisions add no value over growth-itself
- [ ] **"Best seed" is the headline metric** — selection bias, not a result
- [ ] **Late-training growth rate is zero** — controller has learned to be inert
- [ ] **Method requires re-tuning for adjacent problems** — not a method, a hyperparameter set
- [ ] **Sunk-cost language in conversation** ("we've already invested...") — pause and evaluate

---

## Diagnostic Questions

1. **What is the off-switch baseline's loss?** If you don't know, stop and find out.
2. **At your morphogenetic run's final parameter count, what does a static run achieve?** If equal or better, the controller's *path* did not help.
3. **What is the variance across 10 seeds — both in final loss and final architecture?** If either is high, you do not have a condition, you have a distribution.
4. **Is your loss decrease concentrated at growth events?** If yes, suspect Failure Mode 1.
5. **What happens if you disable the controller mid-training and continue with the current architecture?** If the run continues fine, the controller's later contribution is small.
6. **Have you tried a fixed schedule that reproduces your typical final architecture?** If not, you have not run the most important baseline.
7. **If asked to defend morphogenesis without using "could," "might," or "future" — can you?** If not, you are defending a hope, not a result.

---

## Cross-References

- **The full baseline regime and how to report results**: `evaluation-under-topology-change.md`
- **Why a controller learns to never grow (Failure Mode 6)**: `rollback-as-rl-signal.md`
- **Why a controller blows through the budget (Failure Mode 4)**: `multi-seed-coordination-rl.md`
- **The off-switch baseline and the harness it sits in**: `growth-telemetry-and-ablation.md`
- **Designing the controller before you ask whether to use one**: `rl-controller-for-morphogenesis.md`
- **General "is this method right for this problem" discipline**: `yzmir-deep-rl/rl-evaluation`
