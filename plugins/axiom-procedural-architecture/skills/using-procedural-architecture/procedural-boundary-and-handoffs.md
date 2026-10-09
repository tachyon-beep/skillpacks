# Procedure Boundaries and Handoffs

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Procedure design owns readiness, decisions, state/control flow and handoff correctness. Domain implementation may need specialist facts; use those for a bounded unresolved concern, not an obligatory chain of packs.

A handoff names provider/consumer, artifact/state, authority, acceptance condition, deadline if meaningful and failure/retry behavior. Dependencies must be satisfiable; guarded rework/retry cycles are legal control flow and are distinct from prerequisite cycles.

Follow a real issue across role boundaries when needed. A critic can suggest concrete repairs and the designer can verify source. No role has a monopoly on evidence or an obligation to disagree.

Use a diagram/table when it clarifies the interface. Preserve current user authorization and state custody; additional permission is needed only for an actual missing authority decision.
