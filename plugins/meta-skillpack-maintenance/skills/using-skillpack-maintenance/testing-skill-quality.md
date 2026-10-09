# Evaluate Skill Utility

A component can be followed perfectly and still make the task worse. Evaluate
outcomes against a capable baseline; syntax/discovery checks are separate gates.

## Build a bounded comparison

Select representative tasks from actual usage: ordinary success, consequential
edge case, missing tool/information, conflicting constraints, and a plausible
shortcut that violates a real requirement. Include simple tasks to detect process
overhead. Avoid scoring refusal to perform needless ritual as a failure.

Compare, with the same model/tool environment:

1. No skill or relevant default capability.
2. Current full component.
3. Concise candidate or targeted reference/tool alternative.

Use fresh contexts for discovery and independence. Keep task inputs, available
tools, model/version, budgets and scoring criteria stable. Repeat or vary tasks
when uncertainty warrants it; do not claim a general result from one anecdote.

## Score task outcomes

| Dimension | Evidence |
|---|---|
| Correctness | Required behavior/artifact works; material constraints satisfied. |
| Verification | Appropriate checks run; limits and failures honestly reported. |
| Intent/authorization | Requested work completed within scope, unrelated state preserved. |
| Friction | Unnecessary questions, pauses, lectures, delegation or refusals. |
| Efficiency | Tokens, latency, tool calls and maintainable output. |
| Regression | Previously working cases harmed by the candidate. |

Instruction adherence matters only where the instruction protects a real
contract. A model selecting a simpler valid method is not automatically a defect.
Separate model reasoning failures from tool outages, access restrictions and
bad fixtures. Do not reward false claims of runtime success.

## Use existing validation

Validate actual schemas, links, catalogs, commands and hooks mechanically when
possible. The `.test-fixtures/flawed-plugin/` corpus supplies known structural
negatives; catching them does not establish all real workflows work. Exercise
changed workflows with concrete inputs. An inline trial can be a limited smoke
check, but disclose contamination and untested activation behavior.

## Record a result

Save task, environment/model, variant, rubric, observed outcome, cost, failures,
and conclusion. Quote a decisive action or artifact rather than a self-reported
claim that the skill helped. Report comparative runs not performed and uncertain
coverage. Keep/refocus/merge/retire based on the decision evidence, not a required
quantity of anti-patterns or pressure-resistance prose.
