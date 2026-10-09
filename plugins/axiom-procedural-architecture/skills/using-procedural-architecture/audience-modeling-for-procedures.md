# Audience Constraints

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Record only audience properties that change the procedure: prerequisite knowledge/tools/credentials; memory/context support; consequence of error; cost of reversing commitment; latency/progress needs; available recovery.

Do not force six populated fields or a question sequence for a trivial workflow. Infer supported constraints from task/context and ask only when a missing property materially changes safety or usability.

A novice may need examples and finer checks; an experienced human or capable agent may use larger coherent units. Measure actual tool/context/timeout/recovery constraints rather than assuming all LLM agents have low working memory or cannot inspect runtime state.

Use explicit readiness/exit checks at consequential boundaries. Progress communication and bounded waits may be needed when a real timeout/retry risk exists, not because the consumer is labeled an agent.

Output a short audience note linked to design choices. For complex work a table of property, evidence, implication and unknown can help; no mandatory YAML declaration applies.
