# Partitioning Large Repositories

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Use domain ownership, runtime/deploy units and data/authority boundaries as candidate partitions. Cross-cutting infrastructure and generated code may need separate treatment.

Create a partition ledger: paths, owner/question, inspected scope, entry/exit contracts and cross-partition edges. Avoid duplicate deep reviews while preserving boundary overlap where useful.

Choose concurrency from tool/model budget and independent questions. Reconcile cross-boundary claims against source; document unread areas. No fixed agents-per-module or repository-size trigger applies.
