# Test Strategy and Release Evidence

Start from changed behavior and plausible failures. Use the cheapest test layer
that can observe the failure with enough realism; no universal pyramid ratio
applies to every product.

| Claim | Typical evidence |
|---|---|
| Pure behavior/invariant | Unit or property test at the owned interface. |
| Component boundary/state | Integration test using isolated real dependencies where needed. |
| Consumer/provider compatibility | Contract verification for relevant deployed versions. |
| User workflow | Focused E2E test of the actual task and failure/recovery path. |
| Visual/rendering behavior | Controlled screenshot plus interaction/accessibility checks. |
| Capacity or resilience | Defined workload/fault experiment with baseline and stop conditions. |

Inspect existing harnesses before creating another framework. Categorize checks
by purpose and environment; run quick affected checks early and broader suites at
integration/release checkpoints when required. Use parallelism only with isolated
resources, reproducible seeds and understandable failure output.

## Gate contract

Name the criterion, check, environment, required evidence and decision authority.
Do not set arbitrary coverage/latency thresholds as universal truth. A failed
material gate needs repair or an explicit defect disposition/waiver by the actual
owner; recording a failure is not the same as accepting it.

Independent review should inspect source/diff and relevant results with clear
scope. Stakeholder acceptance requires actual task/approver evidence. Separate
local verification, review, stakeholder acceptance, deployment and live checks.
Report exact checks and remaining limits; do not equate green CI with human
acceptance. Delivery implementation lives in `axiom-devops-engineering`.
