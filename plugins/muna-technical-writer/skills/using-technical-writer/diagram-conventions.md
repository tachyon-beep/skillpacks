# Diagram Evidence and Notation

Make the smallest diagram that answers the reader's question. Reuse local notation;
label system boundaries, actors, direction and the meaning of each edge.

Choose a view deliberately: context for external relationships, component/container
for ownership, sequence for ordering, state machine for transitions, data flow for
trust/data boundaries, deployment for runtime placement. Do not combine all views
into an unreadable chart or make a diagram mandatory for simple prose.

Trace nodes and edges to the actual system/design. Distinguish current behavior,
proposals and unknown relationships visually or in the caption. Include a legend
when symbols are not obvious. Keep names consistent with source and documents.

Prefer text-based, versioned diagrams when practical. Render and inspect changed
outputs; check labels, clipping, contrast and reading order. Supply a textual
explanation/alternative for consequential information conveyed only visually.
A Mermaid or other source that parses is not proof its architecture is accurate.
