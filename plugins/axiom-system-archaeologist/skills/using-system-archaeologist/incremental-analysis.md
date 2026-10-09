# Incremental Architecture Refresh

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Record a baseline revision and coverage ledger. Diff source/config/dependencies since that baseline, including dirty changes relevant to the task.

Revisit changed entities and downstream callers, contracts, tests and prior findings. Treat renames/deletions/dynamic registrations as graph changes; a textual diff alone may miss impact.

Preserve stable finding IDs and close findings only with new evidence. Mark unaffected-but-unverified assumptions explicitly. Output changed conclusions, evidence and remaining gaps rather than regenerating every document.
