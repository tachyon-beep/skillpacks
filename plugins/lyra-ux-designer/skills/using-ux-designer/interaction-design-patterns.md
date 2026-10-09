# Interaction State Contract

For each changed interaction, specify its trigger, available action, feedback,
state transition and recovery. Match the platform and existing conventions.

| State/risk | Questions |
|---|---|
| Initial/empty | Is the purpose clear and the first valid action available? |
| Input | Labels, constraints, keyboard/paste and validation make sense. |
| Waiting | Progress is honest; duplicate submissions and cancellation are handled. |
| Success/partial | User can tell what completed and what remains. |
| Failure/denial | Cause is actionable without exposing sensitive information; retry preserves work. |
| Consequence | Impact and recovery are visible; approvals follow authorization and risk. |
| Exit/return | Focus, draft state and navigation survive the transition where expected. |

Use semantic controls and predictable focus. Support keyboard and relevant touch
or controller input. Target sizes, focus behavior and announcements should be
checked against applicable accessibility/platform criteria, not assumed from a
CSS class. Do not make ordinary authorized edits pass through repeated dialogs.

Exercise the actual transition and failure paths touched. Report untested states.
Use [accessibility-and-inclusive-design.md](accessibility-and-inclusive-design.md)
and [ai-experience-patterns.md](ai-experience-patterns.md) for specialized criteria.
