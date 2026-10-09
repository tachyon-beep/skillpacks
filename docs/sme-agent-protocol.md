# Evidence and review contract

This is maintainer guidance, not a runtime plugin dependency. Each independently installable agent includes the small contract it needs in its own entrypoint.

- Read the relevant artifact and enough of its consumers to understand the claim. Distinguish observed evidence, inference and unknowns.
- Support material findings with a source location or reproducible observation, concrete trigger and consequence. A pattern match is a lead until traced.
- State scope, checks performed, unavailable evidence and material uncertainty. Avoid unsupported numerical confidence scores.
- Assess remedy risk when it changes the recommendation: migration, reversibility, operational impact and authority matter.
- A clean result is valid with coverage and limitations. Do not invent findings, disagreement or substantive comments to satisfy a quota.
- Use additional reviewers for an identifiable uncertainty and available capability; no fixed fan-out or elapsed-time minimum applies.
- Honor the caller's requested output format. Include evidence and gaps where they help the decision rather than appending four ceremonial sections to every answer.
- Distinguish a recommendation, local implementation/tests, CI, deployment and user acceptance. Only report the state supported by evidence.

The former standalone SME protocol pack was merged here and into inline agent contracts in marketplace 4.0.0. It is no longer required for installation. See [the consolidation record](relevance-refresh.md).
