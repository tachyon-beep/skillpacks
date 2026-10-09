# Quality Metrics for Decisions

Choose a metric because a decision needs it. Record population, denominator,
environment, time window, measurement method and limitations. Thresholds follow
product requirements and evidence, not a generic table.

| Metric | What it can help assess | What it cannot establish alone |
|---|---|---|
| Coverage | Executed branches/paths and unexamined regions. | Correct assertions, relevant scenarios, defect absence. |
| Mutation sensitivity | Whether selected changes are detected. | All meaningful bugs or complete behavioral quality. |
| Failure/retry rates | Reliability under sampled conditions. | Root cause, future absence of flakiness. |
| Duration/feedback latency | Cost and delay of checks. | Which checks are safe to remove. |
| Escaped defects/recovery | Observed production outcomes. | Fair comparison without context/denominators. |
| Task success/acceptance | Relevant observed user outcome. | Broad audience validity from synthetic or tiny samples. |

Use trends and distributions; inspect tail behavior and confounds. Do not rank
individual developer productivity by test counts, commits or defects. Avoid
optimizing a proxy at the expense of the actual task.

Report the measured result, uncertainty, proposed decision and follow-up signal.
A dashboard is not a gate until a real owner defines action and exception policy.
See [test-automation-architecture.md](test-automation-architecture.md).
