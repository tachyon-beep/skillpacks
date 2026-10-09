# Architecture Evidence and Technical Debt

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

## Review from evidence

Inspect source/runtime/configuration relevant to the question. Compare intended and observed architecture; neither an attractive diagram nor a prior report establishes current behavior. Cite a concrete boundary, failure path or measured constraint for a material finding.

## Debt record

| Field | Purpose |
|---|---|
| ID, location, baseline | Stable identity and inspected revision |
| Evidence and category | Concrete code/config/data and the mechanism |
| Consequence and exposure | User/operational/security impact, affected consumers |
| Proposed options | Repair, containment, accepted risk, removal |
| Effort range and dependencies | Implementation uncertainty and prerequisite work |
| Cost of delay / urgency | What worsens and what event makes it urgent |
| Confidence and gaps | Observed facts versus inference; missing checks |
| Owner, status, revisit trigger | Accountable action and deliberate acceptance |

Prioritize demonstrated exposure, severity, recovery difficulty and dependencies. Security findings may be urgent; a domain label alone does not outrank an active availability or data-integrity incident. “Never breached” is not evidence of safety. Avoid invented precision in debt-interest/ROI estimates.

Preserve clear severity without confrontational posture. Name tradeoffs and contrary evidence; do not soften a defect into a compliment or manufacture one to satisfy a critic role.
