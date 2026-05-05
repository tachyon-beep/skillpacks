# Skillpack Refresh Prompts

Self-contained refresh prompts for HIGH and MEDIUM priority packs from the
2026-05-05 review. Each prompt is briefing material for a fresh agent or
session — assumes no prior conversation context.

Source-of-truth review reports: `/tmp/skillpack-refresh-review/<pack>.md`
Synthesis: `/tmp/skillpack-refresh-review/_SYNTHESIS.md`

## Recommended order

1. `meta-sme-protocol.md` (MEDIUM/S, systemic — do first)
2. `meta-skillpack-maintenance.md` (MEDIUM/M, systemic — do second)
3. `yzmir-simulation-foundations.md` (HIGH/M — investigate truncation before rewriting)
4. `yzmir-ai-engineering-expert.md` (HIGH/S — small router fix unlocks AI campaign)
5. AI cluster as one campaign:
   - `yzmir-llm-specialist.md` (HIGH/L)
   - `yzmir-ml-production.md` (HIGH/L)
   - `yzmir-training-optimization.md` (HIGH/L)
   - `yzmir-pytorch-engineering.md` (HIGH/M)
6. `axiom-python-engineering.md` (HIGH/M — independent, mechanical)
7. `ordis-security-architect.md` (HIGH/M — pairs with AI campaign)
8. MEDIUM remainder in any order (10 prompts).

## Prompt convention

Every prompt has:
- **Context** — what this pack is, where it lives
- **Why refreshing** — verdict + evidence
- **Scope** — what to change, what NOT to change
- **Acceptance criteria** — concrete checks
- **Process** — read review, plan, edit, verify
- **Constraints** — preserve API contracts, don't expand scope, no fabrication

Prompts assume the agent has Read/Edit/Bash/Grep tools.
